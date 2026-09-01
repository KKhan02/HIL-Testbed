"""
HIL Testbed — Dashboard home
============================
Run from the dashboard/ folder with:

    streamlit run Home.py

Reads a single publisher run folder (topology.json, scenarios/*.json,
comparison.json, ...). Works unchanged whether that folder was written by
run_benchmark_script.py (outputs/publisher/<net>/) or by the CLI/executor
(runs/<run_id>/publisher/<net>/) — pick the run in the sidebar.
"""
from pathlib import Path
import subprocess
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _data

st.set_page_config(page_title="Scenario Comparison", layout="wide")

run_dir = _data.sidebar_run_selector()

st.title("⚡ HIL Grid Testbed — Scenario Comparison")

if run_dir is None:
    st.info("Pick a run in the sidebar to get started.")
    st.stop()

topo = _data.load_topology(run_dir)
net_id = (topo or {}).get("network_id", run_dir.name)
st.caption(f"Network: `{net_id}`  ·  Run: `{run_dir.name}`")

# ---------------------------------------------------------------------------
# 1. Scenario comparison — prefer the co-located comparison.json
# ---------------------------------------------------------------------------
records = _data.load_comparison(run_dir)

if not records:
    st.warning(
        "No `comparison.json` in this run folder. It's written by "
        "`publish_hc_and_comparison()` after the benchmark loop — this run may "
        "predate that step or only published topology/scenarios. The animated "
        "views and profiles still work from the sidebar pages."
    )
else:
    df = pd.DataFrame(records)
    if "scenario_id" in df.columns:
        df["_order"] = df["scenario_id"].map(
            {s: i for i, s in enumerate(_data.SCENARIO_ORDER)}
        ).fillna(len(_data.SCENARIO_ORDER))
        df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    st.subheader("Scenario comparison")

    metric_options = [
        "violation_duration_h",
        "vdi",
        "max_vm_pu",
        "min_vm_pu",
        "total_losses_mwh",
        "curtailed_energy_mwh",
        "coordination_rate",
    ]
    available = [m for m in metric_options if m in df.columns]

    col1, col2 = st.columns([1, 3])
    with col1:
        metric = st.selectbox("Metric", available) if available else None
        st.markdown(
            """
            **Reading guide**
            - `violation_duration_h`: hours/year outside the voltage band (lower = better)
            - `vdi`: voltage deviation index (lower = better)
            - `max_vm_pu` / `min_vm_pu`: voltage extremes (target 0.95–1.05)
            - `total_losses_mwh`: network energy losses
            - `curtailed_energy_mwh`: DER energy curtailed by the controller
            - `coordination_rate`: share of steps the coordinator fired (S4)
            """
        )

    with col2:
        if metric and "scenario_id" in df.columns:
            x_col = "scenario_label" if "scenario_label" in df.columns else "scenario_id"
            color_map = {
                row[x_col]: _data.scenario_color(row["scenario_id"])
                for _, row in df.iterrows()
            }
            fig = px.bar(
                df, x=x_col, y=metric, color=x_col,
                color_discrete_map=color_map,
                title=f"{metric} by scenario",
            )
            fig.update_layout(showlegend=False, height=440,
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        elif metric:
            st.bar_chart(df.set_index(df.columns[0])[metric])
        else:
            st.info("No comparable metrics present in comparison.json.")

    with st.expander("Raw comparison table"):
        st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------------------
# 2. Pre-generated static figures (plot_results.py output), if available
# ---------------------------------------------------------------------------
st.subheader("Pre-generated figures")

project_root = Path(st.session_state.get("project_root", "."))
guess_dir, guess_kind = _data.find_figures_dir(project_root, run_dir)

# --- Generate figures for THIS run (gated; runs plot_results.py as a subprocess) ---
run_figs = run_dir / "figures"
comparison_ready = (run_dir / "comparison.json").exists()
plot_script = project_root / "scenario_runners" / "plot_results.py"

with st.expander("⚙️ Generate figures for this run", expanded=guess_dir is None):
    if not plot_script.exists():
        plot_script = Path(st.text_input(
            "Path to plot_results.py", value=str(plot_script),
            help="Not found under scenario_runners/ — point me at it.",
        ))
    st.caption(f"Runs: `plot_results.py --pub-dir <run> --out-dir {run_figs}`")

    force = False
    if not comparison_ready:
        st.caption(
            "⏳ This run isn't finished (no `comparison.json`). Scenario, "
            "comparison and HC figures need completed data, so they'll be "
            "skipped or error; only topology/profile figures render."
        )
        force = st.checkbox("Generate anyway (partial set)")

    can_run = plot_script.exists() and (comparison_ready or force)
    csv_path = _data.find_benchmark_csv(project_root, run_dir)
    if csv_path:
        st.caption(f"Benchmark CSV for fig14: `{csv_path.name}`")
    else:
        st.caption("No benchmark CSV found — fig14 (summary) will be skipped.")
    if st.button("Generate / refresh figures", disabled=not can_run, type="primary"):
        cmd = [sys.executable, str(plot_script),
               "--pub-dir", str(run_dir), "--out-dir", str(run_figs)]
        if csv_path:
            cmd += ["--csv-path", str(csv_path)]
        with st.spinner("Running plot_results.py … (matplotlib batch, may take a minute)"):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            except subprocess.TimeoutExpired:
                proc = None
                st.error("plot_results.py timed out after 900 s.")
        if proc is not None:
            log = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode == 0:
                st.success(f"Figures written to {run_figs}")
                st.rerun()
            else:
                st.error(f"plot_results.py exited {proc.returncode}.")
                st.code(log or "(no output)")

# --- View figures (run-local preferred; blank override = auto-detect) ---
override = st.text_input(
    "Figures folder (blank = auto-detect)", value="", key=f"figdir_{run_dir}",
    help="Blank uses this run's figures. Paste a path to view a different folder.",
)
fig_dir = Path(override) if override.strip() else guess_dir

if fig_dir is None:
    st.info("No `fig*.png` yet — use **Generate figures for this run** above.")
elif not fig_dir.is_dir() or not any(fig_dir.glob("*.png")):
    st.info(f"No `fig*.png` in `{fig_dir}`.")
else:
    if not override.strip() and guess_kind == "shared":
        st.warning(
            "Showing the shared `outputs/figures/<network>` set — keyed by "
            "network, **not** run, so it may be from a different run. Use "
            "**Generate figures for this run** above for a run-local set."
        )
    st.caption(f"From `{fig_dir}`")
    all_pngs = {p.name: p for p in sorted(fig_dir.glob("*.png"))}

    vc1, vc2 = st.columns([2, 1])
    with vc1:
        view = st.radio("Show", ["Key figures", "All figures"], horizontal=True)
    with vc2:
        ncols = st.select_slider("Columns", options=[1, 2, 3], value=2)

    if view == "Key figures":
        names = [n for n in _data.KEY_FIGS if n in all_pngs] or list(all_pngs)
    else:
        names = list(all_pngs)

    for start in range(0, len(names), ncols):
        row = st.columns(ncols)
        for j, name in enumerate(names[start:start + ncols]):
            with row[j]:
                st.markdown(f"**{_data.figure_title(name)}**")
                st.image(str(all_pngs[name]), use_container_width=True)
