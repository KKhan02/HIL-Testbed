"""
Pre-render network frames to PNG (run from the command line, not in Streamlit)
==============================================================================
Renders one PNG per (downsampled) timestep for a scenario, into the publisher
run folder itself, so pages 4/5 pick them up with no copying:

    <run_dir>/frames/<scenario_id>/<resolution>/0000.png, 0001.png, ..., manifest.json

A <run_dir> is any publisher output folder (containing topology.json +
scenarios/). That's what BOTH producers write:
    CLI / executor      runs/<run_id>/publisher/<network_id>/
    run_benchmark_script outputs/publisher/<net_name><RUN_NAME>/

Usage:
    python render_frames.py --run-dir outputs/publisher/1-MV-rural--2-sw --scenario baseline --resolution 6h
    python render_frames.py --run-dir runs/2026-07-29T.../publisher/1-MV-rural--2-sw --scenario oltc --resolution daily

Different resolutions live in separate subfolders, so rendering "daily" after
"6h" won't overwrite it.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI backend — just write files
import matplotlib.pyplot as plt

RESOLUTION_STEPS = {"10min": 1, "1h": 6, "3h": 18, "6h": 36, "daily": 144}


def render(run_dir: Path, scenario_id: str, resolution: str, dpi: int = 90):
    if resolution not in RESOLUTION_STEPS:
        raise ValueError(f"resolution must be one of {list(RESOLUTION_STEPS)}")
    step = RESOLUTION_STEPS[resolution]

    topo_path = run_dir / "topology.json"
    scenario_path = run_dir / "scenarios" / f"{scenario_id}.json"
    out_dir = run_dir / "frames" / scenario_id / resolution

    if not topo_path.exists():
        raise FileNotFoundError(f"topology.json not found at {topo_path}")
    if not scenario_path.exists():
        raise FileNotFoundError(f"scenario file not found at {scenario_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    topo = json.loads(topo_path.read_text(encoding="utf-8"))
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    timeseries = scenario["timeseries"][::step]

    bus_xy = {b["index"]: (b["x"], b["y"]) for b in topo["buses"] if b.get("x") is not None}
    if not bus_xy:
        raise ValueError("No bus x/y coordinates in topology.json — can't render a map.")

    lines = [
        (l["index"], l["from_bus"], l["to_bus"])
        for l in topo["lines"]
        if l["from_bus"] in bus_xy and l["to_bus"] in bus_xy
    ]

    v_min = topo.get("voltage_limits", {}).get("v_min", 0.95)
    v_max = topo.get("voltage_limits", {}).get("v_max", 1.05)

    xs = [xy[0] for xy in bus_xy.values()]
    ys = [xy[1] for xy in bus_xy.values()]
    pad_x = (max(xs) - min(xs)) * 0.08 or 1
    pad_y = (max(ys) - min(ys)) * 0.08 or 1
    xlim = (min(xs) - pad_x, max(xs) + pad_x)
    ylim = (min(ys) - pad_y, max(ys) + pad_y)

    manifest = []
    print(f"Rendering {len(timeseries)} frames to {out_dir} ...")

    for i, frame in enumerate(timeseries):
        vm_pu_by_bus = frame.get("vm_pu_by_bus") or {}
        line_loading = frame.get("line_loading_pct") or {}

        fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)

        for line_idx, fb, tb in lines:
            x0, y0 = bus_xy[fb]
            x1, y1 = bus_xy[tb]
            loading = line_loading.get(str(line_idx))
            color = "crimson" if (loading is not None and loading > 90) else "lightslategray"
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.5, zorder=1)

        bus_ids = list(bus_xy.keys())
        colors = [
            vm_pu_by_bus[str(idx)] if vm_pu_by_bus.get(str(idx)) is not None else (v_min + v_max) / 2
            for idx in bus_ids
        ]
        ax.scatter(
            [bus_xy[idx][0] for idx in bus_ids],
            [bus_xy[idx][1] for idx in bus_ids],
            c=colors, cmap="RdYlBu_r", vmin=v_min - 0.02, vmax=v_max + 0.02,
            s=90, edgecolors="black", linewidths=0.5, zorder=2,
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title((frame.get("timestamp") or "")[:16], fontsize=11)
        fig.tight_layout()

        out_path = out_dir / f"{i:04d}.png"
        fig.savefig(out_path)
        plt.close(fig)

        manifest.append({"frame": i, "timestamp": frame.get("timestamp"), "file": out_path.name})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(timeseries)}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. {len(timeseries)} frames written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True,
                        help="Path to a publisher run folder (contains topology.json + scenarios/).")
    parser.add_argument("--scenario", required=True, help="scenario id, e.g. baseline / oltc / volt_var_coord")
    parser.add_argument("--resolution", default="6h", choices=list(RESOLUTION_STEPS))
    parser.add_argument("--dpi", type=int, default=90)
    args = parser.parse_args()

    render(Path(args.run_dir).expanduser().resolve(), args.scenario, args.resolution, args.dpi)
