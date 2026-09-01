"""
plot_results.py
===============
Static matplotlib figure generator for HIL benchmark results.

Consumes the JSON files written by publisher.publish_result() and produces
publication-quality figures to disk.  No pandapower, simbench, or simulation
dependencies required — only the JSON files are needed.

Usage
-----
    python plot_results.py \\
        --pub-dir  outputs/publisher/1-MV-rural--2-sw \\
        --out-dir  outputs/figures/1-MV-rural--2-sw

    # With HC-stressed re-benchmark figures:
    python plot_results.py \\
        --pub-dir     outputs/publisher/1-MV-rural--2-sw \\
        --hc-pub-dir  outputs/publisher/1-MV-rural--2-sw_hc_stressed \\
        --out-dir     outputs/figures/1-MV-rural--2-sw

    # With benchmark CSV for Figs 13/14:
    python plot_results.py \\
        --pub-dir  outputs/publisher/1-MV-rural--2-sw \\
        --csv-path outputs/benchmarks/1-MV-rural--2-sw_benchmark_*.csv \\
        --out-dir  outputs/figures/1-MV-rural--2-sw

Figures produced
----------------
fig01_network_generation_map.png    — Bus map coloured by sgen type/capacity
fig02_network_line_loading_map.png  — Bus map with lines coloured by max loading
fig03_installed_capacity.png        — Bar chart of installed capacity by type
fig04_network_topology.png          — Topology with vn_kv-coloured lines + bus labels
fig05_voltage_heatmap.png           — Timestep × bus voltage magnitude heatmap
fig06_voltage_vs_feeder_dist.png    — vm_pu vs feeder distance per scenario
fig07_annual_profiles.png           — Load + DER generation profile (annual/sliced)
fig08a_timeseries_annual.png        — Multi-panel annual overview (hourly max)
fig08b_timeseries_extreme_day.png   — Multi-panel extreme DER day (10-min)
fig09_violation_heatmap.png         — Day × scenario violation flag heatmap
fig10_qv_scatter.png                — Q(V) operating point scatter (4A and 4B)
fig11_hc_sweep.png                  — Hosting capacity sweep curve
fig12_coordination_scatter.png      — Coordinated vs local-only Q scatter
fig13_curtailment_timeseries.png    — Curtailed MW time series (4A vs 4B)
fig14_benchmark_summary.png         — Grouped bar chart of all KPIs from CSV

Dependencies
------------
Required : matplotlib, numpy, pandas, networkx
Optional : contextily  (map tile basemap for SimBench; degrades gracefully)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import networkx as nx

try:
    import contextily as ctx
    _HAS_CTX = True
except ImportError:
    _HAS_CTX = False

# ===========================================================================
# Global style — IEEE single-column, serif, 300 dpi
# ===========================================================================

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.fontsize":   9,
    "legend.framealpha": 0.85,
    "lines.linewidth":   1.0,
    "figure.dpi":        300,
    "figure.figsize":    (7.16, 4.5),
    "savefig.bbox":      "tight",
    "savefig.dpi":       300,
})

# Scenario display order, labels, and colours — consistent across all figures
SCENARIO_ORDER  = ["baseline", "oltc", "svc", "volt_var_local", "volt_var_coord", "opf"]
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

V_MIN = 0.95
V_MAX = 1.05

# Voltage-level palette — highest kV first
# Red for HV slack (110 kV), blue for MV (20 kV) — immediately distinguishable
_VN_PALETTE = ["#d73027", "#2166ac", "#6baed6", "#c6dbef"]


def _vn_color_map(topology: dict) -> dict[float, str]:
    """Return {vn_kv: colour} sorted highest → lowest."""
    vn_kvs = sorted({b["vn_kv"] for b in topology["buses"]}, reverse=True)
    return {kv: _VN_PALETTE[i % len(_VN_PALETTE)] for i, kv in enumerate(vn_kvs)}


# ===========================================================================
# JSON loaders
# ===========================================================================

def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_publisher_dir(pub_dir: Path) -> dict:
    data: dict = {}
    data["topology"]   = _load(pub_dir / "topology.json")
    data["profiles"]   = _load(pub_dir / "profiles.json")

    hc_path = pub_dir / "hc.json"
    data["hc"] = _load(hc_path) if hc_path.exists() else None

    cmp_path = pub_dir / "comparison.json"
    data["comparison"] = _load(cmp_path) if cmp_path.exists() else None

    sc_dir = pub_dir / "scenarios"
    data["scenarios"] = {}
    if sc_dir.exists():
        for sc_file in sorted(sc_dir.glob("*.json")):
            data["scenarios"][sc_file.stem] = _load(sc_file)

    return data


# ===========================================================================
# Coordinate helpers
# ===========================================================================

def _is_geographic(buses: list[dict]) -> bool:
    xs = [b["x"] for b in buses if b.get("x") is not None]
    ys = [b["y"] for b in buses if b.get("y") is not None]
    if not xs:
        return False
    return (
        -180 <= min(xs) and max(xs) <= 180
        and -90 <= min(ys) and max(ys) <= 90
        and (max(xs) - min(xs)) < 5.0
    )


def _get_bus_positions(topology: dict) -> dict[int, tuple[float, float]]:
    buses = topology["buses"]
    lines = topology["lines"]
    pos = {b["index"]: (b.get("x"), b.get("y")) for b in buses}
    if not any(v[0] is not None for v in pos.values()):
        G = nx.Graph()
        G.add_nodes_from(b["index"] for b in buses)
        G.add_edges_from((l["from_bus"], l["to_bus"]) for l in lines)
        layout = nx.spring_layout(G, seed=42)
        pos = {idx: (float(xy[0]), float(xy[1])) for idx, xy in layout.items()}
    return pos


# ===========================================================================
# Shared drawing helpers
# ===========================================================================

def _draw_network_base(ax, topology: dict, pos: dict,
                       line_values: Optional[dict] = None,
                       line_cmap: str = "YlOrRd",
                       line_vmin: float = 0.0,
                       line_vmax: float = 100.0,
                       default_line_color: str = "#aaaaaa",
                       line_color_by_vn: Optional[dict] = None,
                       bus_lookup: Optional[dict] = None,
                       line_lw: float = 0.8):
    """
    Draw lines and trafos on ax.

    Priority for line colour:
      1. line_values  → continuous colourmap (e.g. loading %)
      2. line_color_by_vn + bus_lookup → discrete vn_kv colour
      3. default_line_color
    """
    cmap_fn = cm.get_cmap(line_cmap)
    norm    = mcolors.Normalize(vmin=line_vmin, vmax=line_vmax)

    for l in topology["lines"]:
        fb, tb = l["from_bus"], l["to_bus"]
        if fb not in pos or tb not in pos:
            continue
        x0, y0 = pos[fb]
        x1, y1 = pos[tb]
        if line_values is not None:
            val   = line_values.get(str(l["index"]),
                                    line_values.get(l["index"], 0.0))
            color = cmap_fn(norm(val))
        elif line_color_by_vn is not None and bus_lookup is not None:
            vn  = bus_lookup.get(fb, {}).get("vn_kv", 0.0)
            color = line_color_by_vn.get(vn, default_line_color)
        else:
            color = default_line_color
        ax.plot([x0, x1], [y0, y1], color=color, lw=line_lw, zorder=1)

    for tr in topology["trafos"]:
        hb, lb = tr["hv_bus"], tr["lv_bus"]
        if hb not in pos or lb not in pos:
            continue
        x0, y0 = pos[hb]
        x1, y1 = pos[lb]
        ax.plot([x0, x1], [y0, y1], color="#333333", lw=1.2,
                linestyle="--", zorder=1)

    return norm, cmap_fn


def _add_basemap(ax, geo: bool, alpha: float = 0.25):
    """Add OSM basemap when contextily is available and coords are geographic."""
    if geo and _HAS_CTX:
        try:
            ctx.add_basemap(ax, crs="EPSG:4326",
                            source=ctx.providers.OpenStreetMap.Mapnik,
                            alpha=alpha)
        except Exception:
            pass


def _save(fig, out_dir: Path, fname: str):
    path = out_dir / fname
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved → {path.name}")


# ===========================================================================
# Figure 1 — Network generation map
# ===========================================================================

def fig01_network_generation_map(topology: dict, out_dir: Path,
                                  geo: bool = False):
    buses  = {b["index"]: b for b in topology["buses"]}
    pos    = _get_bus_positions(topology)
    sgens  = topology["sgens"]

    fig, ax = plt.subplots(figsize=(6, 5))
    _draw_network_base(ax, topology, pos, default_line_color="#888888",
                       line_lw=0.7)

    for idx in buses:
        x, y = pos[idx]
        ax.scatter(x, y, s=8, color="#cccccc", zorder=2)

    for sg in sgens:
        if not sg.get("in_service", True):
            continue
        bidx = sg["bus"]
        if bidx not in pos:
            continue
        x, y = pos[bidx]
        t = (sg.get("type") or "").lower()
        if "pv" in t or "solar" in t or "lv_res" in t:
            color = SGEN_TYPE_COLORS["pv"]
        elif "wind" in t or "wp" in t:
            color = SGEN_TYPE_COLORS["wind"]
        else:
            color = SGEN_TYPE_COLORS["other"]
        sn   = sg.get("sn_mva") or sg.get("p_mw") or 0.1
        size = max(20, min(200, sn * 30))
        ax.scatter(x, y, s=size, color=color, edgecolors="white",
                   linewidths=0.4, zorder=3)

    _add_basemap(ax, geo, alpha=0.25)

    legend_handles = [
        Patch(color=SGEN_TYPE_COLORS["pv"],    label="PV / lv_RES"),
        Patch(color=SGEN_TYPE_COLORS["wind"],  label="Wind"),
        Patch(color=SGEN_TYPE_COLORS["other"], label="Other DER"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    ax.set_title("Network — DER Generation Distribution")
    ax.set_axis_off()
    fig.tight_layout()
    _save(fig, out_dir, "fig01_network_generation_map.png")


# ===========================================================================
# Figure 2 — Network line-loading map (worst loading per line)
# ===========================================================================

def fig02_network_line_loading_map(topology: dict, scenarios: dict,
                                    out_dir: Path, geo: bool = False,
                                    scenario_id: str = "baseline"):
    pos        = _get_bus_positions(topology)
    buses      = {b["index"]: b for b in topology["buses"]}
    vn_colors  = _vn_color_map(topology)
    sc         = scenarios.get(scenario_id)

    line_max: dict[str, float] = {}
    if sc:
        for rec in sc["timeseries"]:
            for lid, val in (rec.get("line_loading_pct") or {}).items():
                if val is not None:
                    lid_s = str(lid)
                    line_max[lid_s] = max(line_max.get(lid_s, 0.0), float(val))

    fig, ax = plt.subplots(figsize=(6, 5))
    norm, cmap_fn = _draw_network_base(ax, topology, pos,
                                        line_values=line_max,
                                        line_cmap="YlOrRd",
                                        line_vmin=0, line_vmax=120,
                                        line_lw=1.2)

    # Buses coloured by voltage level
    for idx, b in buses.items():
        x, y  = pos[idx]
        color = vn_colors.get(b["vn_kv"], "#888888")
        ax.scatter(x, y, s=12, color=color, edgecolors="white",
                   linewidths=0.3, zorder=3)

    sm = cm.ScalarMappable(cmap="YlOrRd",
                           norm=mcolors.Normalize(vmin=0, vmax=120))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Max line loading (%)", shrink=0.75)

    _add_basemap(ax, geo, alpha=0.25)

    label = SCENARIO_LABELS.get(scenario_id, scenario_id)
    ax.set_title(f"Network — Max Line Loading [{label}]")
    ax.set_axis_off()
    fig.tight_layout()
    _save(fig, out_dir, "fig02_network_line_loading_map.png")


# ===========================================================================
# Figure 3 — Installed capacity chart
# ===========================================================================

def fig03_installed_capacity(topology: dict, out_dir: Path):
    sgens = [s for s in topology["sgens"] if s.get("in_service", True)]

    type_totals: dict[str, float] = {}
    for sg in sgens:
        t   = (sg.get("type") or "other").lower()
        # lv_res is aggregated LV residential PV — classify as PV
        key = ("pv"   if ("pv" in t or "solar" in t or "lv_res" in t) else
               "wind" if ("wind" in t or "wp" in t)                    else
               "other")
        sn  = sg.get("sn_mva") or sg.get("p_mw") or 0.0
        type_totals[key] = type_totals.get(key, 0.0) + sn

    if not type_totals:
        print("  fig03 skipped — no sgens in topology")
        return

    # Fixed display order
    order  = [k for k in ["pv", "wind", "other"] if k in type_totals]
    labels = [{"pv": "PV / lv_RES", "wind": "Wind", "other": "Other"}[k]
              for k in order]
    values = [type_totals[k] for k in order]
    colors = [SGEN_TYPE_COLORS[k] for k in order]

    fig, ax = plt.subplots(figsize=(4, 3.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white",
                  linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.2f} MVA", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Installed capacity (MVA)")
    ax.set_title("Installed DER Capacity by Type")
    ax.set_ylim(0, max(values) * 1.15)
    fig.tight_layout()
    _save(fig, out_dir, "fig03_installed_capacity.png")


# ===========================================================================
# Figure 4 — Network topology (vn_kv-coloured lines + bus labels)
# ===========================================================================

def fig04_network_topology(topology: dict, out_dir: Path, geo: bool = False):
    buses      = {b["index"]: b for b in topology["buses"]}
    pos        = _get_bus_positions(topology)
    vn_colors  = _vn_color_map(topology)
    vn_kvs     = sorted(vn_colors.keys(), reverse=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Lines coloured by from_bus vn_kv
    _draw_network_base(ax, topology, pos,
                       line_color_by_vn=vn_colors,
                       bus_lookup=buses,
                       line_lw=0.9)

    # Buses coloured by voltage level
    for idx, b in buses.items():
        x, y  = pos[idx]
        color = vn_colors.get(b["vn_kv"], "#888888")
        ax.scatter(x, y, s=18, color=color, edgecolors="white",
                   linewidths=0.3, zorder=3)

    # Bus number annotations (only when network is small enough to be legible)
    if len(buses) <= 150:
        for idx in buses:
            x, y = pos[idx]
            ax.annotate(str(idx), (x, y),
                        fontsize=6.5, ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points",
                        color="#222222", zorder=4)

    legend_handles = [
        Patch(color=vn_colors[kv], label=f"{kv} kV") for kv in vn_kvs
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              title="Voltage level", fontsize=8)
    ax.set_title(f"Network Topology — {topology.get('network_id', '')}")
    ax.set_axis_off()

    _add_basemap(ax, geo, alpha=0.25)

    fig.tight_layout()
    _save(fig, out_dir, "fig04_network_topology.png")


# ===========================================================================
# Figure 5 — Voltage heatmap (timestep × bus)
# ===========================================================================

def fig05_voltage_heatmap(scenarios: dict, out_dir: Path,
                           scenario_id: str = "volt_var_coord",
                           downsample_h: int = 1):
    sc = scenarios.get(scenario_id)
    if sc is None:
        print(f"  fig05 skipped — scenario '{scenario_id}' not found")
        return

    ts = sc["timeseries"]
    bus_keys = sorted(
        {k for rec in ts for k in (rec.get("vm_pu_by_bus") or {}).keys()},
        key=lambda k: int(k),
    )
    if not bus_keys:
        print("  fig05 skipped — no vm_pu_by_bus data")
        return

    rows = []
    for rec in ts:
        vm = rec.get("vm_pu_by_bus") or {}
        rows.append([vm.get(k) for k in bus_keys])

    mat = np.array([[v if v is not None else np.nan for v in r] for r in rows])

    if downsample_h > 1:
        step   = downsample_h * 6
        n_rows = mat.shape[0] // step
        mat    = np.array([np.nanmax(mat[i*step:(i+1)*step], axis=0)
                           for i in range(n_rows)])

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(
        mat.T,
        aspect="auto",
        interpolation="nearest",
        cmap="RdYlGn_r",
        vmin=V_MIN - 0.02,
        vmax=V_MAX + 0.02,
        origin="upper",
    )
    fig.colorbar(im, ax=ax, label="Voltage (pu)", shrink=0.85)
    ax.set_xlabel("Timestep (downsampled)" if downsample_h > 1 else "Timestep")
    ax.set_ylabel("Bus index")
    label = SCENARIO_LABELS.get(scenario_id, scenario_id)
    ax.set_title(f"Voltage Heatmap — {label}")
    fig.tight_layout()
    _save(fig, out_dir, "fig05_voltage_heatmap.png")


# ===========================================================================
# Figure 6 — Voltage vs feeder distance
# ===========================================================================

def fig06_voltage_vs_feeder_dist(topology: dict, scenarios: dict,
                                  out_dir: Path,
                                  scenario_ids: Optional[list] = None):
    feeder_dist = topology.get("feeder_dist", {})
    if not feeder_dist:
        print("  fig06 skipped — no feeder_dist in topology")
        return

    dist_map = {int(k): float(v)
                for k, v in feeder_dist.items() if v is not None}

    if scenario_ids is None:
        scenario_ids = [sid for sid in SCENARIO_ORDER if sid in scenarios]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for sid in scenario_ids:
        sc = scenarios.get(sid)
        if sc is None:
            continue
        dists, vms = [], []
        for rec in sc["timeseries"]:
            if not rec.get("converged", True):
                continue
            vm_map = rec.get("vm_pu_by_bus") or {}
            for bus_str, vm in vm_map.items():
                bus_int = int(bus_str)
                if bus_int in dist_map and vm is not None:
                    dists.append(dist_map[bus_int])
                    vms.append(float(vm))

        if not dists:
            continue

        dists = np.array(dists)
        vms   = np.array(vms)
        bins     = np.linspace(0, dists.max() + 0.01, 20)
        bin_idx  = np.digitize(dists, bins)
        bin_mid, bin_med, bin_p5, bin_p95 = [], [], [], []
        for i in range(1, len(bins)):
            mask = bin_idx == i
            if mask.sum() < 2:
                continue
            v = vms[mask]
            bin_mid.append((bins[i - 1] + bins[i]) / 2)
            bin_med.append(np.median(v))
            bin_p5.append(np.percentile(v, 5))
            bin_p95.append(np.percentile(v, 95))

        color = SCENARIO_COLORS.get(sid, "#888888")
        label = SCENARIO_LABELS.get(sid, sid)
        # Reduced alpha so envelopes don't crowd the lines
        ax.fill_between(bin_mid, bin_p5, bin_p95, color=color, alpha=0.08)
        # coord gets circle markers every 3 bins to separate from local (nearly identical median)
        marker   = "o" if sid == "volt_var_coord" else None
        markevry = 3   if sid == "volt_var_coord" else None
        ax.plot(bin_mid, bin_med, color=color, label=label, linewidth=1.2,
                marker=marker, markersize=3, markevery=markevry)

    ax.axhline(V_MIN, color="red",    linestyle="--", lw=0.8)
    ax.axhline(V_MAX, color="orange", linestyle="--", lw=0.8)
    ax.set_xlabel("Feeder distance from slack (km)")
    ax.set_ylabel("Bus voltage (pu)")
    ax.set_title("Voltage vs Feeder Distance (median ± 5th–95th pct)")

    # Legend below x-axis to avoid hiding dashed limit lines
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        fontsize=8,
        framealpha=0.9,
    )
    fig.subplots_adjust(bottom=0.22)
    _save(fig, out_dir, "fig06_voltage_vs_feeder_dist.png")


# ===========================================================================
# Figure 7 — Annual / sliced load + DER profile
# ===========================================================================

def fig07_annual_profiles(profiles: dict, out_dir: Path):
    times_h = pd.to_datetime(profiles.get("times_hourly", []))
    if len(times_h) == 0:
        print("  fig07 skipped — no times_hourly in profiles")
        return

    load_h = np.array(profiles.get("load_total_mw",  []))
    pv_h   = np.array(profiles.get("pv_total_mw",    []))
    wind_h = np.array(profiles.get("wind_total_mw",  []))

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)

    ax1 = axes[0]
    ax1.fill_between(times_h, pv_h,   label="PV",  color=SGEN_TYPE_COLORS["pv"],
                     alpha=0.7, linewidth=0)
    ax1.fill_between(times_h, wind_h, label="Wind", color=SGEN_TYPE_COLORS["wind"],
                     alpha=0.7, linewidth=0)
    ax1.set_ylabel("Generation (MW)")
    ax1.set_title("Load & DER Generation Profile")
    ax1.legend(loc="upper right")

    ax2 = axes[1]
    ax2.fill_between(times_h, load_h, label="Load demand",
                     color="#6d6875", alpha=0.7, linewidth=0)
    ax2.set_ylabel("Load (MW)")
    ax2.set_xlabel("Time")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    _save(fig, out_dir, "fig07_annual_profiles.png")


# ===========================================================================
# Figure 8 — Multi-panel time series (annual + extreme day)
# ===========================================================================

def _extreme_day_slice(profiles: dict, key: str) -> Optional[pd.DatetimeIndex]:
    extreme_days = profiles.get("extreme_days", {})
    day_str = extreme_days.get(key)
    if not day_str:
        return None
    day      = pd.Timestamp(day_str).date()
    # utc=True avoids FutureWarning about mixed timezones and ensures .dt.date works
    times_10 = pd.to_datetime(profiles.get("times_10min", []), utc=True)
    mask     = pd.Series(times_10).dt.date == day
    return times_10[mask.values]


def fig08_timeseries_panels(scenarios: dict, profiles: dict,
                             out_dir: Path,
                             scenario_ids: Optional[list] = None,
                             downsample_h: int = 1):
    if scenario_ids is None:
        scenario_ids = [sid for sid in SCENARIO_ORDER if sid in scenarios]

    panel_keys   = ["max_vm_pu", "min_vm_pu", "max_line_loading_pct",
                    "tap_pos",   "losses_mw", "grid_import_mw"]
    panel_labels = ["Max voltage (pu)", "Min voltage (pu)",
                    "Max line loading (%)", "Tap position",
                    "Network losses (MW)", "Grid exchange (MW)\n(−ve = export)"]
    panel_hlims  = {
        "max_vm_pu":            (V_MAX, "red"),
        "min_vm_pu":            (V_MIN, "red"),
        "max_line_loading_pct": (100.0, "orange"),
    }
    # Tap position is a state, not a peak — use last value per window
    TAP_KEY = "tap_pos"

    n_panels = len(panel_keys)
    step = max(1, downsample_h * 6)

    # ---- Part A: downsampled annual series ----
    fig_ann, axes_ann = plt.subplots(n_panels, 1,
                                     figsize=(9, 1.6 * n_panels),
                                     sharex=True)
    for sid in scenario_ids:
        sc = scenarios.get(sid)
        if sc is None:
            continue
        ts     = sc["timeseries"]
        times  = pd.to_datetime([r["timestamp"] for r in ts], utc=True)
        color  = SCENARIO_COLORS.get(sid, "#888888")
        label  = SCENARIO_LABELS.get(sid, sid)

        for ax, key in zip(axes_ann, panel_keys):
            vals = np.array(
                [r.get(key) if r.get(key) is not None else np.nan
                 for r in ts], dtype=float)
            n_ds = len(vals) // step
            if key == TAP_KEY:
                # State: take last value in each window
                vals_ds = np.array([vals[(i + 1) * step - 1]
                                    for i in range(n_ds)])
            else:
                vals_ds = np.array([np.nanmax(vals[i*step:(i+1)*step])
                                    for i in range(n_ds)])
            # Skip all-NaN series (e.g. tap_pos for non-OLTC scenarios)
            if np.all(np.isnan(vals_ds)):
                continue
            times_ds = times[::step][:n_ds]
            ax.plot(times_ds, vals_ds, color=color, label=label,
                    linewidth=0.8)

    for ax, key, ylabel in zip(axes_ann, panel_keys, panel_labels):
        if key in panel_hlims:
            limit, lcolor = panel_hlims[key]
            ax.axhline(limit, color=lcolor, linestyle="--", lw=0.7)
        if key == TAP_KEY:
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylabel(ylabel, fontsize=8)

    axes_ann[0].legend(
        ncol=5, fontsize=7,
        loc="upper center", bbox_to_anchor=(0.5, -0.12 / n_panels),
        framealpha=0.9,
    )
    fig_ann.subplots_adjust(bottom=0.08)
    axes_ann[0].set_title("Time Series Overview (hourly max)")
    axes_ann[-1].set_xlabel("Time")
    fig_ann.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig_ann, out_dir, "fig08a_timeseries_annual.png")

    # ---- Part B: four extreme day close-ups — one PNG each ----
    extreme_day_keys = [
        ("max_der",  "fig08b_max_der_day.png",  "Highest DER Generation Day"),
        ("min_der",  "fig08c_min_der_day.png",  "Lowest DER Generation Day"),
        ("max_load", "fig08d_max_load_day.png", "Peak Load Day"),
        ("min_load", "fig08e_min_load_day.png", "Minimum Load Day"),
    ]

    for ed_key, ed_fname, ed_title in extreme_day_keys:
        day_times = _extreme_day_slice(profiles, ed_key)
        if day_times is None or len(day_times) == 0:
            print(f"  {ed_fname} skipped — no extreme_days.{ed_key} in profiles")
            continue

        day_str   = str(day_times[0].date())
        fig_day, axes_day = plt.subplots(n_panels, 1,
                                          figsize=(7, 1.6 * n_panels),
                                          sharex=True)

        for sid in scenario_ids:
            sc = scenarios.get(sid)
            if sc is None:
                continue
            ts      = sc["timeseries"]
            times_  = pd.to_datetime([r["timestamp"] for r in ts], utc=True)
            mask    = pd.Series(times_).dt.date == day_times[0].date()
            ts_day  = [r for r, m in zip(ts, mask) if m]
            t_day   = times_[mask.values]
            color   = SCENARIO_COLORS.get(sid, "#888888")
            label   = SCENARIO_LABELS.get(sid, sid)

            for ax, key in zip(axes_day, panel_keys):
                vals = np.array(
                    [r.get(key) if r.get(key) is not None else np.nan
                     for r in ts_day], dtype=float)
                if not np.all(np.isnan(vals)):
                    ax.plot(t_day, vals, color=color, label=label,
                            linewidth=1.0)

        for ax, key, ylabel in zip(axes_day, panel_keys, panel_labels):
            if key in panel_hlims:
                limit, lcolor = panel_hlims[key]
                ax.axhline(limit, color=lcolor, linestyle="--", lw=0.7)
            if key == TAP_KEY:
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_ylabel(ylabel, fontsize=8)

        axes_day[0].set_title(f"Time Series — {ed_title} ({day_str})")
        axes_day[-1].set_xlabel("Time")
        # Legend below the figure, all 5 scenarios in a single horizontal row
        handles, labels_leg = axes_day[0].get_legend_handles_labels()
        # Re-order handles to match SCENARIO_ORDER
        order_map = {SCENARIO_LABELS.get(s, s): i
                     for i, s in enumerate(SCENARIO_ORDER)}
        paired = sorted(zip(labels_leg, handles),
                        key=lambda x: order_map.get(x[0], 99))
        labels_leg, handles = zip(*paired) if paired else ([], [])
        fig_day.legend(
            handles, labels_leg,
            ncol=len(handles), fontsize=7,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            framealpha=0.9,
        )
        fig_day.tight_layout(rect=[0, 0.04, 1, 1])
        _save(fig_day, out_dir, ed_fname)


# ===========================================================================
# Figure 9 — Violation heatmap (day × scenario)
# ===========================================================================

def fig09_violation_heatmap(scenarios: dict, out_dir: Path,
                              scenario_ids: Optional[list] = None):
    if scenario_ids is None:
        scenario_ids = [sid for sid in SCENARIO_ORDER if sid in scenarios]

    cols = {}
    for sid in scenario_ids:
        sc = scenarios.get(sid)
        if sc is None:
            continue
        cols[sid] = np.array(
            [1.0 if r.get("violation_flag") else 0.0
             for r in sc["timeseries"]], dtype=float)

    if not cols:
        print("  fig09 skipped — no scenario data")
        return

    min_len = min(len(v) for v in cols.values())
    mat = np.column_stack([cols[sid][:min_len] for sid in scenario_ids
                           if sid in cols])
    step   = 144   # 144 × 10-min steps per day
    n_days = mat.shape[0] // step
    mat_ds = np.array([mat[i*step:(i+1)*step].max(axis=0)
                       for i in range(n_days)])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.imshow(mat_ds.T, aspect="auto", cmap="Reds",
              vmin=0, vmax=1, interpolation="nearest", origin="upper")
    ax.set_yticks(range(len(scenario_ids)))
    ax.set_yticklabels([SCENARIO_LABELS.get(s, s) for s in scenario_ids],
                       fontsize=8)
    ax.set_xlabel("Day of simulation period")
    ax.set_title("Violation Presence (red = ≥1 violation that day)")
    fig.tight_layout()
    _save(fig, out_dir, "fig09_violation_heatmap.png")


# ===========================================================================
# Figure 10 — Q(V) operating scatter
# ===========================================================================

def fig10_qv_scatter(topology: dict, scenarios: dict, out_dir: Path):
    sgen_to_bus = {s["index"]: s["bus"] for s in topology["sgens"]}
    target_ids  = [sid for sid in ["volt_var_local", "volt_var_coord"]
                   if sid in scenarios]
    if not target_ids:
        print("  fig10 skipped — no volt_var scenarios found")
        return

    fig, axes = plt.subplots(1, len(target_ids),
                             figsize=(4.0 * len(target_ids), 4),
                             sharey=True)
    if len(target_ids) == 1:
        axes = [axes]

    for ax, sid in zip(axes, target_ids):
        sc = scenarios[sid]
        vm_all, q_all = [], []
        for rec in sc["timeseries"]:
            if not rec.get("converged", True):
                continue
            vm_map = rec.get("vm_pu_by_bus") or {}
            q_map  = rec.get("q_mvar_by_sgen") or {}
            for sgen_str, q in q_map.items():
                if q is None:
                    continue
                bus_int = sgen_to_bus.get(int(sgen_str))
                if bus_int is None:
                    continue
                vm = vm_map.get(str(bus_int)) or vm_map.get(bus_int)
                if vm is None:
                    continue
                vm_all.append(float(vm))
                q_all.append(float(q))

        if not vm_all:
            ax.set_title(SCENARIO_LABELS.get(sid, sid))
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        ax.scatter(vm_all, q_all, s=1.5, alpha=0.3,
                   color=SCENARIO_COLORS.get(sid, "#888888"), rasterized=True)
        ax.axvline(V_MIN, color="red",     linestyle="--", lw=0.7)
        ax.axvline(V_MAX, color="orange",  linestyle="--", lw=0.7)
        ax.axhline(0,     color="#888888", linestyle="-",  lw=0.5)
        ax.set_xlabel("Bus voltage (pu)")
        ax.set_title(SCENARIO_LABELS.get(sid, sid))

    axes[0].set_ylabel("Q setpoint (MVAr)")
    fig.suptitle("Q(V) Operating Points", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, "fig10_qv_scatter.png")


# ===========================================================================
# Figure 11 — HC sweep curve
# ===========================================================================

def fig11_hc_sweep(hc: Optional[dict], out_dir: Path):
    if hc is None or "error" in hc:
        print("  fig11 skipped — no HC data")
        return

    fig, ax = plt.subplots(figsize=(5, 4))

    for case_key, color, label in [
        ("baseline", SCENARIO_COLORS["baseline"],       "Baseline"),
        ("volt_var", SCENARIO_COLORS["volt_var_local"], "Volt-Var"),
    ]:
        c = hc.get(case_key)
        if c is None:
            continue
        hc_mw = c.get("hc_mw", 0.0)
        curve  = c.get("sweep_curve", [])

        if curve:
            mws = [p["mw"]        for p in curve]
            vms = [p["max_vm_pu"] for p in curve]
            ax.plot(mws, vms, color=color, label=label, linewidth=1.2)
            hc_idx = max(
                (i for i, p in enumerate(curve) if p["mw"] <= hc_mw + 1e-6),
                default=None,
            )
            if hc_idx is not None:
                ax.scatter([curve[hc_idx]["mw"]], [curve[hc_idx]["max_vm_pu"]],
                           color=color, s=60, zorder=5,
                           label=f"{label} HC: {hc_mw:.3f} MW")
        else:
            # Fallback: vertical line when sweep_curve absent (old JSON)
            ax.axvline(hc_mw, color=color, linestyle="--", lw=1.0,
                       label=f"{label}: {hc_mw:.3f} MW")
            vbind = c.get("binding_vm_pu")
            if vbind is not None:
                ax.scatter([hc_mw], [vbind], color=color, s=60, zorder=5)

    ax.axhline(V_MAX, color="red", linestyle="--", lw=0.8,
               label=f"V_max = {V_MAX} pu")
    ax.set_xlabel("Added PV capacity (MW)")
    ax.set_ylabel("Max network voltage (pu)")
    gain = hc.get("gain_mw", 0.0)
    ax.set_title(f"Hosting Capacity Sweep  [Volt-Var gain: {gain:+.3f} MW]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "fig11_hc_sweep.png")


# ===========================================================================
# Figure 12 — Coordination scatter
# ===========================================================================

def fig12_coordination_scatter(topology: dict, scenarios: dict, out_dir: Path):
    sc_coord = scenarios.get("volt_var_coord")
    if sc_coord is None:
        print("  fig12 skipped — volt_var_coord scenario not found")
        return

    sgen_to_bus = {s["index"]: s["bus"] for s in topology["sgens"]}
    vm_coord, q_coord, vm_local, q_local = [], [], [], []

    for rec in sc_coord["timeseries"]:
        if not rec.get("converged", True):
            continue
        is_coord = rec.get("coordination_active", False)
        vm_map   = rec.get("vm_pu_by_bus") or {}
        q_map    = rec.get("q_mvar_by_sgen") or {}
        for sgen_str, q in q_map.items():
            if q is None:
                continue
            bus_int = sgen_to_bus.get(int(sgen_str))
            if bus_int is None:
                continue
            vm = vm_map.get(str(bus_int)) or vm_map.get(bus_int)
            if vm is None:
                continue
            if is_coord:
                vm_coord.append(float(vm)); q_coord.append(float(q))
            else:
                vm_local.append(float(vm)); q_local.append(float(q))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    if vm_local:
        ax.scatter(vm_local, q_local, s=1.5, alpha=0.25,
                   color=SCENARIO_COLORS["volt_var_local"],
                   label="Local Q(V) only", rasterized=True)
    if vm_coord:
        ax.scatter(vm_coord, q_coord, s=4, alpha=0.6,
                   color=SCENARIO_COLORS["volt_var_coord"],
                   label="Coordinator active", rasterized=True, zorder=4)

    ax.axvline(V_MIN, color="red",     linestyle="--", lw=0.7)
    ax.axvline(V_MAX, color="orange",  linestyle="--", lw=0.7)
    ax.axhline(0,     color="#888888", linestyle="-",  lw=0.5)

    coord_steps = sum(1 for r in sc_coord["timeseries"]
                      if r.get("coordination_active"))
    total = len(sc_coord["timeseries"])
    rate  = coord_steps / total * 100 if total else 0.0

    ax.set_xlabel("Bus voltage at DER (pu)")
    ax.set_ylabel("Q setpoint (MVAr)")
    ax.set_title(f"Coordination Scatter  [coord_rate = {rate:.1f}%]")
    ax.legend(fontsize=8, markerscale=4)
    fig.tight_layout()
    _save(fig, out_dir, "fig12_coordination_scatter.png")


# ===========================================================================
# Figure 13 — Curtailment time series (4A vs 4B)
# ===========================================================================

def fig13_curtailment_timeseries(scenarios: dict, out_dir: Path,
                                  downsample_h: int = 1):
    target_ids = [sid for sid in ["volt_var_local", "volt_var_coord"]
                  if sid in scenarios]
    if not target_ids:
        print("  fig13 skipped — no volt_var scenarios found")
        return

    step = max(1, downsample_h * 6)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax_curtail, ax_mw = axes

    for sid in target_ids:
        sc    = scenarios[sid]
        ts    = sc["timeseries"]
        times = pd.to_datetime([r["timestamp"] for r in ts], utc=True)
        color = SCENARIO_COLORS[sid]
        label = SCENARIO_LABELS[sid]

        # Panel 1: fraction of timesteps in each hour where curtailment fired
        flag = np.array([1.0 if r.get("curtailment_needed") else 0.0
                         for r in ts], dtype=float)
        n_ds     = len(flag) // step
        flag_ds  = np.array([flag[i*step:(i+1)*step].mean()
                              for i in range(n_ds)])
        times_ds = times[::step][:n_ds]
        ax_curtail.plot(times_ds, flag_ds * 100, color=color,
                        label=label, linewidth=0.8)

        # Panel 2: curtailed MW = DER profile target - actually applied per sgen
        # der_gen_mw = sum(ap.der_p.iloc[t]) = unconstrained profile target
        # p_mw_by_sgen = actual applied after curtailment loop
        # difference = curtailed MW this timestep
        curtailed_mw = []
        for rec in ts:
            der_gen  = rec.get("der_gen_mw")       # unconstrained profile total
            p_applied = rec.get("p_mw_by_sgen") or {}
            applied_sum = sum(
                float(v) for v in p_applied.values() if v is not None
            )
            if der_gen is not None:
                curtailed_mw.append(max(0.0, float(der_gen) - applied_sum))
            else:
                curtailed_mw.append(0.0)

        curt    = np.array(curtailed_mw, dtype=float)
        n_ds2   = len(curt) // step
        curt_ds = np.array([curt[i*step:(i+1)*step].sum()
                             for i in range(n_ds2)])
        ax_mw.plot(times_ds[:n_ds2], curt_ds, color=color,
                   label=label, linewidth=0.8)

    ax_curtail.set_ylabel("Timesteps curtailed (%)")
    ax_curtail.set_title("Active Power Curtailment")
    ax_curtail.legend(fontsize=8)
    ax_curtail.set_ylim(0, 105)

    ax_mw.set_ylabel("Curtailed MW per hour\n(profile target − applied)")
    ax_mw.set_xlabel("Time")
    ax_mw.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir, "fig13_curtailment_timeseries.png")


# ===========================================================================
# Figure 14 — Benchmark summary bar chart (from CSV)
# ===========================================================================

def fig14_benchmark_summary(csv_path: Optional[Path], out_dir: Path):
    if csv_path is None or not csv_path.exists():
        print("  fig14 skipped — no CSV path provided or file not found")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("  fig14 skipped — CSV is empty")
        return

    base_row = df[df["scenario_id"] == "baseline"]
    if base_row.empty:
        print("  fig14 skipped — no baseline row in CSV")
        return
    base_row = base_row.iloc[0]

    # Panel A: metrics where baseline is defined → % improvement over baseline
    pct_metrics = [
        ("n_violation_steps", "Violation\ntimesteps"),
        ("vdi",               "VDI\n(pu·steps)"),
        ("total_losses_mwh",  "Total losses\n(MWh)"),
    ]
    # Panel B: metrics where baseline is NaN → absolute values only
    abs_metrics = [
        ("curtailed_energy_mwh",  "Curtailed\nenergy (MWh)"),
        ("reactive_energy_mvarh", "Reactive energy\n(MVArh)"),
    ]

    pct_avail = [(col, lbl) for col, lbl in pct_metrics if col in df.columns]
    abs_avail = [(col, lbl) for col, lbl in abs_metrics if col in df.columns]

    if not pct_avail and not abs_avail:
        print("  fig14 skipped — no recognised metric columns in CSV")
        return

    # Non-baseline rows for both panels
    plot_df = df[df["scenario_id"] != "baseline"].reset_index(drop=True)
    n_scen  = len(plot_df)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"width_ratios": [len(pct_avail), max(1, len(abs_avail))]},
    )
    ax_pct, ax_abs = axes

    # ---- Panel A: % improvement ----
    if pct_avail:
        x_pct = np.arange(len(pct_avail))
        width  = 0.8 / n_scen
        Y_CAP  = 150.0

        for i, row in plot_df.iterrows():
            sid   = row.get("scenario_id", str(i))
            color = SCENARIO_COLORS.get(sid, f"C{i}")
            label = SCENARIO_LABELS.get(sid, row.get("scenario_label", sid))
            vals  = []
            for col, _ in pct_avail:
                v    = row.get(col)
                base = base_row.get(col)
                if pd.notna(v) and pd.notna(base) and float(base) != 0:
                    vals.append((float(base) - float(v)) / abs(float(base)) * 100)
                else:
                    vals.append(0.0)

            offset = (i - n_scen / 2 + 0.5) * width
            for xi, val in zip(x_pct + offset, vals):
                clipped = max(-Y_CAP, min(Y_CAP, val))
                ax_pct.bar(xi, clipped, width * 0.9, color=color, alpha=0.85,
                           label=label if xi == (x_pct + offset)[0] else "_nolegend_")
                if abs(val) > Y_CAP:
                    sign = "+" if val > 0 else ""
                    ax_pct.annotate(
                        f"{sign}{val:.0f}%",
                        xy=(xi, Y_CAP * np.sign(val)),
                        xytext=(0, 4 * int(np.sign(val))),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if val > 0 else "top",
                        fontsize=6, color=color,
                    )

        ax_pct.axhline(0, color="#333333", lw=0.8)
        ax_pct.set_ylim(-Y_CAP * 1.15, Y_CAP * 1.25)
        ax_pct.set_xticks(x_pct)
        ax_pct.set_xticklabels([lbl for _, lbl in pct_avail], fontsize=9)
        ax_pct.set_ylabel("Improvement vs Baseline (%)")
        ax_pct.set_title("(A) % Improvement over Baseline")
        ax_pct.legend(fontsize=8)
        ax_pct.text(0.99, 0.02, "↑ above zero = improvement",
                    transform=ax_pct.transAxes, ha="right", va="bottom",
                    fontsize=7, color="#555555")
    else:
        ax_pct.set_visible(False)

    # ---- Panel B: absolute values ----
    if abs_avail:
        x_abs = np.arange(len(abs_avail))
        width  = 0.8 / n_scen

        for i, row in plot_df.iterrows():
            sid   = row.get("scenario_id", str(i))
            color = SCENARIO_COLORS.get(sid, f"C{i}")
            label = SCENARIO_LABELS.get(sid, row.get("scenario_label", sid))
            vals  = [float(row.get(col, 0) or 0) for col, _ in abs_avail]

            offset = (i - n_scen / 2 + 0.5) * width
            ax_abs.bar(x_abs + offset, vals, width * 0.9,
                       color=color, label=label, alpha=0.85)

        ax_abs.set_xticks(x_abs)
        ax_abs.set_xticklabels([lbl for _, lbl in abs_avail], fontsize=9)
        ax_abs.set_ylabel("Absolute value (MWh / MVArh)")
        ax_abs.set_title("(B) Absolute — no baseline defined")
        ax_abs.legend(fontsize=8)
    else:
        ax_abs.set_visible(False)

    fig.suptitle("Benchmark Summary", fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, "fig14_benchmark_summary.png")


# ===========================================================================
# Main orchestrator
# ===========================================================================

def plot_results(
        pub_dir:    Path,
        out_dir:    Path,
        hc_pub_dir: Optional[Path] = None,
        csv_path:   Optional[Path] = None,
        figures:    Optional[list[int]] = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading publisher data from: {pub_dir}")
    data = load_publisher_dir(pub_dir)

    topology  = data["topology"]
    profiles  = data["profiles"]
    hc        = data["hc"]
    scenarios = data["scenarios"]

    # Merge HC-stressed scenarios if outer run had scenarios=[]
    if hc_pub_dir and hc_pub_dir.exists():
        hc_data = load_publisher_dir(hc_pub_dir)
        if not scenarios and hc_data["scenarios"]:
            print(f"  Using HC-stressed scenarios from: {hc_pub_dir}")
            scenarios = hc_data["scenarios"]

    buses = topology["buses"]
    geo   = _is_geographic(buses)
    if geo:
        print("  Coordinate type: WGS84 geographic (SimBench)")
        if not _HAS_CTX:
            print("  contextily not installed — map basemap disabled")
    else:
        has_coords = any(b.get("x") is not None for b in buses)
        print(f"  Coordinate type: "
              f"{'schematic' if has_coords else 'none (networkx layout)'}")

    print(f"  Scenarios loaded: {sorted(scenarios.keys())}")
    print(f"  Output dir: {out_dir}\n")

    all_figs = {
        1:  lambda: fig01_network_generation_map(topology, out_dir, geo=geo),
        2:  lambda: fig02_network_line_loading_map(topology, scenarios, out_dir,
                                                    geo=geo),
        3:  lambda: fig03_installed_capacity(topology, out_dir),
        4:  lambda: fig04_network_topology(topology, out_dir, geo=geo),
        5:  lambda: fig05_voltage_heatmap(scenarios, out_dir),
        6:  lambda: fig06_voltage_vs_feeder_dist(topology, scenarios, out_dir),
        7:  lambda: fig07_annual_profiles(profiles, out_dir),
        8:  lambda: fig08_timeseries_panels(scenarios, profiles, out_dir,
                                             downsample_h=1),
        9:  lambda: fig09_violation_heatmap(scenarios, out_dir),
        10: lambda: fig10_qv_scatter(topology, scenarios, out_dir),
        11: lambda: fig11_hc_sweep(hc, out_dir),
        12: lambda: fig12_coordination_scatter(topology, scenarios, out_dir),
        13: lambda: fig13_curtailment_timeseries(scenarios, out_dir,
                                                  downsample_h=1),
        14: lambda: fig14_benchmark_summary(csv_path, out_dir),
    }

    to_run = sorted(figures) if figures else sorted(all_figs.keys())
    for n in to_run:
        fn = all_figs.get(n)
        if fn is None:
            print(f"  fig{n:02d} — unknown figure number, skipped")
            continue
        print(f"  Generating fig{n:02d}...")
        try:
            fn()
        except Exception as exc:
            print(f"  fig{n:02d} ERROR: {exc}")

    print(f"\nDone. {len(to_run)} figure(s) attempted → {out_dir}")


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    _NET = "1-MV-rural--2-sw"
    _ROOT = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="Generate static matplotlib figures from publisher JSON output."
    )
    parser.add_argument(
        "--pub-dir", type=Path,
        default=_ROOT / "outputs" / "publisher" / _NET,
        help="Publisher output directory",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=_ROOT / "outputs" / "figures" / _NET,
        help="Destination directory for PNG figures",
    )
    parser.add_argument(
        "--hc-pub-dir", type=Path,
        default=_ROOT / "outputs" / "publisher" / (_NET + "_hc_stressed"),
        help="Optional HC-stressed publisher directory",
    )
    parser.add_argument(
        "--csv-path", type=Path,
        default=None,
        help="Path to benchmark CSV for Fig 14 (glob patterns not supported here — "
             "pass the exact file path)",
    )
    parser.add_argument(
        "--figs", type=int, nargs="+", default=None,
        metavar="N",
        help="Figure numbers to generate (default: all). Example: --figs 1 5 10",
    )
    args = parser.parse_args()

    # Auto-discover latest CSV if not given explicitly
    csv_path = args.csv_path
    if csv_path is None:
        candidates = sorted(
            (_ROOT / "outputs" / "benchmarks").glob(f"{_NET}_benchmark_*.csv")
        )
        if candidates:
            csv_path = candidates[-1]
            print(f"  Auto-selected CSV: {csv_path.name}")

    plot_results(
        pub_dir    = args.pub_dir,
        out_dir    = args.out_dir,
        hc_pub_dir = args.hc_pub_dir,
        csv_path   = csv_path,
        figures    = args.figs,
    )