"""
_data.py — shared data layer for the HIL dashboard
===================================================
This is the single place that knows *where the data lives* and *how to load
it*. Every page imports from here so the dashboard is decoupled from any one
producer's directory layout.

A "run" is one publisher output folder — exactly the shape written by BOTH
entry points and consumed by plot_results.load_publisher_dir():

    <run_dir>/
        topology.json
        profiles.json        (optional)
        scenarios/<id>.json  (one per scenario)
        hc.json              (optional — hosting-capacity study)
        comparison.json      (optional — one record per scenario)
        frames/<id>/<res>/   (optional — written by render_frames.py)
        live/<id>.jsonl      (optional — live streaming)

Both producers emit that folder, just under different roots:
    CLI / executor      runs/<run_id>/publisher/<network_id>/
    run_benchmark_script outputs/publisher/<net_name><RUN_NAME>/

So the dashboard never hardcodes `data/<network>/`. It discovers every run
folder under a configurable project root and lets the user pick one in the
sidebar; the choice is shared across all pages via st.session_state.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Scenario palette — mirrors plot_results.py so the dashboard matches the
# report figures exactly.
# ---------------------------------------------------------------------------
SCENARIO_ORDER = ["baseline", "oltc", "svc", "volt_var_local", "volt_var_coord", "opf"]
SCENARIO_LABELS = {
    "baseline":       "Baseline",
    "oltc":           "OLTC",
    "svc":            "SVC",
    "volt_var_local": "Volt-Var (local)",
    "volt_var_coord": "Volt-Var (+ coord)",
    "opf":            "OPF",
}
SCENARIO_COLORS = {
    "baseline":       "#555555",
    "oltc":           "#2166ac",
    "svc":            "#d6604d",
    "volt_var_local": "#4dac26",
    "volt_var_coord": "#1a9641",
    "opf":            "#7b3294",
}
SGEN_TYPE_COLORS = {"pv": "#f4a261", "wind": "#457b9d", "other": "#8d99ae"}


def scenario_label(sid: str) -> str:
    """Human label for a scenario id (falls back to the id itself)."""
    return SCENARIO_LABELS.get(sid, sid)


def scenario_color(sid: str) -> str:
    """Stable colour for a scenario id (grey fallback for plugin scenarios)."""
    return SCENARIO_COLORS.get(sid, "#8d99ae")


def sort_scenarios(sids) -> list[str]:
    """Order scenario ids by the canonical display order; unknowns last."""
    known = [s for s in SCENARIO_ORDER if s in sids]
    extra = sorted(s for s in sids if s not in SCENARIO_ORDER)
    return known + extra


# ---------------------------------------------------------------------------
# Project-root detection + run discovery
# ---------------------------------------------------------------------------
def detect_project_root() -> Path:
    """
    Best-effort guess at the project root (the folder that contains
    outputs/ and/or runs/). Resolution order:

    1. $HIL_DASHBOARD_ROOT if set.
    2. Walk up from this file; first ancestor containing `outputs` or `runs`.
    3. This file's grandparent (dashboard/ is typically one level under root).
    4. Current working directory.
    """
    env = os.environ.get("HIL_DASHBOARD_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "outputs").is_dir() or (parent / "runs").is_dir():
            return parent
    # dashboard/_data.py → dashboard/ → <root>
    if len(here.parents) >= 2:
        return here.parents[1]
    return Path.cwd()


# Glob patterns tried under the project root. A hit is any `topology.json`;
# its parent folder is the run. Ordered from most-specific to catch-all.
_RUN_PATTERNS = [
    "outputs/publisher/*/topology.json",   # run_benchmark_script.py
    "runs/*/publisher/*/topology.json",    # executor (network-nested, post-fix)
    "runs/*/publisher/topology.json",      # executor (flat — defensive)
    "data/*/topology.json",                # legacy manual copy
    "*/topology.json",                     # project_root points straight at a publisher/ dir
    "topology.json",                       # project_root IS a single run folder
]


@st.cache_data(show_spinner=False)
def _discover(root_str: str) -> dict[str, str]:
    """Return {display_label: run_dir} for every run under `root_str`."""
    root = Path(root_str)
    found: dict[str, str] = {}
    for pattern in _RUN_PATTERNS:
        for topo in root.glob(pattern):
            run_dir = topo.parent.resolve()
            try:
                label = str(run_dir.relative_to(root))
            except ValueError:
                label = str(run_dir)
            found[label or run_dir.name] = str(run_dir)
    # De-dup by resolved path while keeping the shortest label per path.
    by_path: dict[str, str] = {}
    for label, path in sorted(found.items(), key=lambda kv: len(kv[0])):
        by_path.setdefault(path, label)
    return {label: path for path, label in sorted(by_path.items(), key=lambda kv: kv[1])}


def discover_runs(project_root: Path) -> dict[str, Path]:
    """Public wrapper returning {label: Path}."""
    return {lbl: Path(p) for lbl, p in _discover(str(project_root)).items()}


# ---------------------------------------------------------------------------
# Cached JSON loaders (keyed on path + mtime so re-running a scenario refreshes)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_json(path_str: str, mtime: float) -> dict:
    with open(path_str, encoding="utf-8") as fh:
        return json.load(fh)


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    return _load_json(str(path), path.stat().st_mtime)


def load_topology(run_dir: Path) -> dict | None:
    return _read(run_dir / "topology.json")


def load_profiles(run_dir: Path) -> dict | None:
    return _read(run_dir / "profiles.json")


def load_hc(run_dir: Path) -> dict | None:
    return _read(run_dir / "hc.json")


def load_comparison(run_dir: Path) -> list | None:
    data = _read(run_dir / "comparison.json")
    return data if isinstance(data, list) else None


def load_scenario(run_dir: Path, scenario_id: str) -> dict | None:
    return _read(run_dir / "scenarios" / f"{scenario_id}.json")


def list_scenarios(run_dir: Path) -> list[str]:
    sc_dir = run_dir / "scenarios"
    if not sc_dir.exists():
        return []
    return sort_scenarios([p.stem for p in sc_dir.glob("*.json")])


def network_id_of(run_dir: Path) -> str:
    topo = load_topology(run_dir)
    if topo and topo.get("network_id"):
        return str(topo["network_id"])
    return run_dir.name


def find_benchmark_csv(project_root: Path, run_dir: Path) -> Path | None:
    """
    Locate the benchmark CSV for fig14. Searches the CLI location
    (runs/<id>/, = run_dir.parent.parent) and the script location
    (outputs/benchmarks/), excluding the HC-stressed CSV. Prefers this
    network's file, then the most recent.
    """
    net = network_id_of(run_dir)
    search = [
        run_dir.parent.parent,                      # CLI: runs/<id>/
        project_root / "outputs" / "benchmarks",    # run_benchmark_script.py
        run_dir, run_dir.parent,
    ]
    hits = []
    for d in search:
        try:
            if d.is_dir():
                hits += [p for p in d.glob("*_benchmark_*.csv") if "_hc_stressed_" not in p.name]
        except OSError:
            continue
    if not hits:
        return None
    pool = [p for p in hits if p.name.startswith(net)] or hits
    return max(pool, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Static figure catalogue (plot_results.py output) — titles + curated set
# ---------------------------------------------------------------------------
FIG_TITLES = {
    "fig01_network_generation_map.png":   "Network — DER generation map",
    "fig02_network_line_loading_map.png": "Network — line loading map",
    "fig03_installed_capacity.png":       "Installed capacity by type",
    "fig04_network_topology.png":         "Network topology",
    "fig05_voltage_heatmap.png":          "Voltage heatmap (bus × time)",
    "fig06_voltage_vs_feeder_dist.png":   "Voltage vs feeder distance",
    "fig07_annual_profiles.png":          "Annual load & DER profiles",
    "fig09_violation_heatmap.png":        "Violation heatmap (scenario × time)",
    "fig10_qv_scatter.png":               "Q(V) operating points",
    "fig11_hc_sweep.png":                 "Hosting-capacity sweep",
    "fig12_coordination_scatter.png":     "Coordination scatter",
    "fig13_curtailment_timeseries.png":   "Curtailment timeseries",
    "fig14_benchmark_summary.png":        "Benchmark summary (scenario comparison)",
}

# Impact-ordered curated subset for the "Key figures" view.
KEY_FIGS = [
    "fig14_benchmark_summary.png",
    "fig05_voltage_heatmap.png",
    "fig09_violation_heatmap.png",
    "fig10_qv_scatter.png",
    "fig02_network_line_loading_map.png",
    "fig11_hc_sweep.png",
]


def figure_title(filename: str) -> str:
    """Human title for a fig*.png (humanised fallback for unknown names)."""
    if filename in FIG_TITLES:
        return FIG_TITLES[filename]
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_", 1)
    label = parts[1] if len(parts) == 2 and parts[0].startswith("fig") else stem
    return label.replace("_", " ").capitalize()


def find_figures_dir(project_root: Path, run_dir: Path) -> tuple[Path | None, str | None]:
    """
    Locate static fig*.png (plot_results.py output). Returns (dir, kind):

      kind="run"    → <run_dir>/figures  — unambiguous, belongs to THIS run.
      kind="shared" → <root>/outputs/figures/<network_id> — a *convention*
                      guess keyed by network, NOT by run, so it may show
                      figures from a different run of the same network.
      (None, None)  → nothing found.

    plot_results.py writes wherever its --out-dir points, so this is only a
    best guess; the caller also offers a manual override.
    """
    run_local = run_dir / "figures"
    if run_local.is_dir() and any(run_local.glob("*.png")):
        return run_local, "run"
    for c in (project_root / "outputs" / "figures" / network_id_of(run_dir),
              project_root / "figures" / network_id_of(run_dir)):
        if c.is_dir() and any(c.glob("*.png")):
            return c, "shared"
    return None, None


# ---------------------------------------------------------------------------
# Live JSONL (live/<scenario_id>.jsonl) — discovery + incremental tail
# ---------------------------------------------------------------------------
def list_live_scenarios(run_dir: Path) -> list[tuple[str, float]]:
    """
    Return [(scenario_id, mtime)] for every non-empty live/<id>.jsonl, ordered
    by the canonical scenario order. mtime lets the page default to the most
    recently written file (the one currently streaming).
    """
    live = run_dir / "live"
    if not live.exists():
        return []
    out = []
    for p in sorted(live.glob("*.jsonl")):
        # Include empty files too: on_scenario_start creates live/<id>.jsonl
        # empty, and scenarios that use run_timeseries (baseline, S5) only
        # flush their frames at the end — so an empty file means "streaming,
        # first frame pending", not "no live".
        try:
            out.append((p.stem, p.stat().st_mtime))
        except OSError:
            continue
    order = {s: i for i, s in enumerate(SCENARIO_ORDER)}
    out.sort(key=lambda t: order.get(t[0], 999))
    return out


def read_jsonl_since(path: Path, offset: int) -> tuple[list[dict], int, bool]:
    """
    Read only the complete, newline-terminated JSON lines appended after
    byte `offset`. Returns (new_frames, new_offset, reset).

    - Binary mode so offsets are exact bytes.
    - Any trailing partial line (writer mid-append) is left for next time.
    - reset=True means the file is now smaller than `offset` — i.e. it was
      truncated (on_scenario_start resets the live file), so the caller should
      discard its accumulated history and start fresh.
    """
    frames: list[dict] = []
    if not path.exists():
        return frames, offset, False

    reset = False
    if path.stat().st_size < offset:
        offset, reset = 0, True

    with open(path, "rb") as fh:
        fh.seek(offset)
        chunk = fh.read()

    last_nl = chunk.rfind(b"\n")
    if last_nl == -1:
        return frames, offset, reset  # nothing complete yet

    complete = chunk[: last_nl + 1]
    for line in complete.split(b"\n"):
        if not line.strip():
            continue
        try:
            frames.append(json.loads(line.decode("utf-8")))
        except json.JSONDecodeError:
            continue  # skip a torn line; it'll be re-read intact next tick
    return frames, offset + len(complete), reset


# ---------------------------------------------------------------------------
# Log files (session logs + optional streamlit log) — discovery + tail
# ---------------------------------------------------------------------------
def discover_logs(project_root: Path, run_dir: Path) -> list[Path]:
    """
    Candidate log files, most-relevant first:
      - CLI:    runs/<id>/session.log  (= run_dir.parent.parent/session.log)
      - any *.log next to the run
      - script: outputs/logs/session_*.log (newest first)
      - streamlit*.log if the user redirected Streamlit's output to a file
    """
    cands: list[Path] = []
    run_root = run_dir.parent.parent
    for p in [run_root / "session.log", *sorted(run_root.glob("*.log"))]:
        cands.append(p)
    logs_dir = project_root / "outputs" / "logs"
    if logs_dir.is_dir():
        cands += sorted(logs_dir.glob("session_*.log"),
                        key=lambda x: x.stat().st_mtime, reverse=True)
    for d in (project_root, run_root):
        cands += sorted(d.glob("streamlit*.log"))
    seen, out = set(), []
    for p in cands:
        try:
            rp = p.resolve()
        except OSError:
            continue
        if p.exists() and rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def tail_lines(path: Path, n: int = 200, max_bytes: int = 500_000) -> tuple[list[str], int]:
    """
    Return (last n lines, file_size). Reads at most the final max_bytes so a
    huge annual log stays cheap; drops the first (possibly partial) line.
    """
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()
            data = fh.read()
        return data.decode("utf-8", errors="replace").splitlines()[-n:], size
    except OSError:
        return [], 0



def sidebar_run_selector() -> Path | None:
    """
    Render the project-root input + run selectbox in the sidebar and return
    the selected run directory (or None if nothing was found). The selection
    is persisted in st.session_state so it stays put as the user navigates
    between pages.
    """
    st.sidebar.header("Data source")

    default_root = st.session_state.get("project_root") or str(detect_project_root())
    root_str = st.sidebar.text_input(
        "Project root",
        value=default_root,
        help="Folder containing outputs/ and/or runs/. Or point straight at a "
             "publisher/ folder, or a single run folder.",
    )
    st.session_state["project_root"] = root_str
    project_root = Path(root_str).expanduser()

    if not project_root.exists():
        st.sidebar.error(f"Path does not exist:\n`{project_root}`")
        return None

    runs = discover_runs(project_root)
    if not runs:
        st.sidebar.warning(
            "No runs found. A run is any folder containing `topology.json` "
            "(e.g. `outputs/publisher/<net>/` from the script, or "
            "`runs/<run_id>/publisher/<net>/` from the CLI)."
        )
        if st.sidebar.button("↻ Rescan"):
            _discover.clear()
            st.rerun()
        return None

    labels = list(runs.keys())
    prev = st.session_state.get("run_label")
    idx = labels.index(prev) if prev in labels else len(labels) - 1
    label = st.sidebar.selectbox("Run", labels, index=idx)
    st.session_state["run_label"] = label

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("↻ Rescan"):
        _discover.clear()
        st.rerun()
    if col_b.button("↻ Reload"):
        _load_json.clear()
        st.rerun()

    run_dir = runs[label]
    st.session_state["run_dir"] = str(run_dir)
    st.sidebar.caption(f"`{run_dir}`")
    return run_dir
