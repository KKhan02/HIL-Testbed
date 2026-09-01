"""
Network view
============
Two modes, chosen automatically from the topology:

  • Networks WITH bus coordinates → fast native-Plotly animation (all frames
    built once, cycled in the browser, no per-frame Python reruns), plus an
    "Inspect a timestep" panel for exact numbers at one step.
  • Networks WITHOUT coordinates (e.g. Kerber/Dickert built without
    create_generic_coordinates) → a scrubbable table of per-bus voltages plus
    the same timestep detail. The map can't be drawn, so the table is primary.

Reads topology.json + scenarios/<id>.json from the run selected in the sidebar.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _data

st.set_page_config(page_title="Network View", layout="wide")
run_dir = _data.sidebar_run_selector()
st.title("🔌 Network View")

if run_dir is None:
    st.info("Pick a run in the sidebar.")
    st.stop()

topo = _data.load_topology(run_dir)
if topo is None:
    st.error("This run has no `topology.json` yet.")
    st.stop()
scenario_ids = _data.list_scenarios(run_dir)
if not scenario_ids:
    st.info(
        "Topology loaded, but no scenario has **finished** yet. Each "
        "`scenarios/<id>.json` is written when its scenario completes and "
        "appears here then. To watch the run **in progress**, use the "
        "**Home / Live** page; hit sidebar **Rescan / Reload** once a scenario "
        "finishes."
    )
    st.stop()

buses_df = pd.DataFrame(topo["buses"])
has_coords = bool(buses_df["x"].notna().any() and buses_df["y"].notna().any())
v_min = topo.get("voltage_limits", {}).get("v_min", 0.95)
v_max = topo.get("voltage_limits", {}).get("v_max", 1.05)

RESOLUTION_STEPS = {
    "Every step (10 min)": 1, "Hourly": 6, "Every 3 hours": 18,
    "Every 6 hours": 36, "Daily": 144,
}
MAX_FRAMES_WARNING = 1000

# --- Controls (animation controls only shown when a map is drawable) --------
if has_coords:
    c1, c2, c3 = st.columns(3)
    with c1:
        scenario_id = st.selectbox("Scenario", scenario_ids, format_func=_data.scenario_label)
    with c2:
        resolution_label = st.selectbox("Resolution (downsample)", list(RESOLUTION_STEPS), index=2)
    with c3:
        animate_lines = st.toggle("Animate line loading too (heavier)", value=False)
    step = RESOLUTION_STEPS[resolution_label]
else:
    scenario_id = st.selectbox("Scenario", scenario_ids, format_func=_data.scenario_label)
    step, animate_lines = 1, False

scenario = _data.load_scenario(run_dir, scenario_id)
full_ts = scenario["timeseries"]

# ===========================================================================
# MAP (coordinate networks) — fast native-Plotly animation
# ===========================================================================
if has_coords:
    bus_xy = buses_df.set_index("index")[["x", "y"]]
    sampled = full_ts[::step]
    st.caption(
        f"Scenario **{_data.scenario_label(scenario_id)}** — {len(full_ts):,} total steps, "
        f"animating **{len(sampled):,} frames**."
    )
    if len(sampled) > MAX_FRAMES_WARNING:
        st.warning(f"{len(sampled):,} frames is a lot — try a coarser resolution if playback feels heavy.")
    speed_ms = st.slider("Frame duration (ms) — lower = faster", 20, 500, 80, step=10)

    @st.cache_data(show_spinner="Building animation frames...")
    def build_frames(sampled_frames, bus_index_order, bus_xy_records, lines, animate_lines, v_min, v_max):
        bus_xy_lookup = dict(bus_xy_records)
        line_geom, valid_lines = [], []
        for line in lines:
            fb, tb = line["from_bus"], line["to_bus"]
            if fb in bus_xy_lookup and tb in bus_xy_lookup:
                x0, y0 = bus_xy_lookup[fb]
                x1, y1 = bus_xy_lookup[tb]
                if x0 is None or x1 is None:
                    continue
                line_geom.append((x0, y0, x1, y1))
                valid_lines.append(line)
        frames = []
        for i, frame in enumerate(sampled_frames):
            vm = frame.get("vm_pu_by_bus") or {}
            colors = [vm.get(str(idx)) for idx in bus_index_order]
            hover = [
                f"Bus {idx}: {vm.get(str(idx)):.4f} pu" if vm.get(str(idx)) is not None else f"Bus {idx}: n/a"
                for idx in bus_index_order
            ]
            frame_data = [go.Scatter(marker=dict(color=colors), text=hover)]
            frame_traces = [len(valid_lines)]
            if animate_lines:
                ll = frame.get("line_loading_pct") or {}
                for line in valid_lines:
                    loading = ll.get(str(line["index"]))
                    col = "crimson" if (loading is not None and loading > 90) else "lightslategray"
                    frame_data.append(go.Scatter(line=dict(color=col)))
                frame_traces = list(range(len(valid_lines))) + [len(valid_lines)]
            frames.append(go.Frame(
                data=frame_data, traces=frame_traces, name=str(i),
                layout=go.Layout(title_text=(frame.get("timestamp") or "")[:16]),
            ))
        return frames, line_geom, valid_lines

    bus_index_order = list(buses_df["index"])
    bus_xy_records = [(idx, (row.x, row.y)) for idx, row in bus_xy.iterrows()]
    frames, line_geom, valid_lines = build_frames(
        sampled, bus_index_order, bus_xy_records, topo["lines"], animate_lines, v_min, v_max
    )

    init_data = []
    for (x0, y0, x1, y1) in line_geom:
        init_data.append(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color="lightslategray", width=2), hoverinfo="skip", showlegend=False,
        ))
    first = frames[0].data[-1]
    init_data.append(go.Scatter(
        x=buses_df["x"], y=buses_df["y"], mode="markers",
        marker=dict(
            size=14, color=first.marker.color, colorscale="RdYlBu_r",
            cmin=v_min - 0.02, cmax=v_max + 0.02, colorbar=dict(title="vm_pu"),
            line=dict(width=1, color="black"),
        ),
        text=first.text, hoverinfo="text", showlegend=False,
    ))

    fig = go.Figure(data=init_data, frames=frames)
    fig.update_layout(
        height=600, xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        margin=dict(l=10, r=10, t=40, b=10),
        title=frames[0].layout.title.text if frames[0].layout.title else "",
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.0, y=1.12, xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, dict(
                    frame=dict(duration=speed_ms, redraw=True), fromcurrent=True,
                    transition=dict(duration=0))]),
                dict(label="⏸ Pause", method="animate", args=[[None], dict(
                    frame=dict(duration=0, redraw=False), mode="immediate",
                    transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=[dict(
                method="animate",
                args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                     transition=dict(duration=0))],
                label=(sampled[i].get("timestamp") or "")[5:16],
            ) for i, f in enumerate(frames)],
            x=0, y=-0.05, len=1.0, currentvalue=dict(prefix="Time: "),
        )],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("Use the ▶ Play button on the chart itself (top-left) — playback runs in your browser.")
else:
    st.warning(
        "This network has no bus x/y coordinates (common for Kerber/Dickert "
        "networks built without `create_generic_coordinates()`), so it can't "
        "be drawn as a map — scrub the table below instead."
    )

# ===========================================================================
# TIMESTEP DETAIL — table (no-coord networks) + metrics + envelope marker
# ===========================================================================
st.divider()
st.subheader("Inspect a timestep")
if has_coords:
    st.caption("Independent of the animation above — moving this slider restarts "
               "the map; press ▶ to replay.")

t_idx = st.slider("Timestep", 0, len(full_ts) - 1, 0)
frame = full_ts[t_idx]

if not has_coords:
    vm_by = frame.get("vm_pu_by_bus") or {}
    tbl = buses_df.copy()
    tbl["vm_pu"] = tbl["index"].astype(str).map(vm_by)
    st.dataframe(tbl[["index", "name", "vn_kv", "vm_pu"]], use_container_width=True)

ts = frame.get("timestamp")
formatted = pd.to_datetime(ts).strftime("%d/%m %H:%M") if ts else "—"
m1, m2, m3, m4 = st.columns(4)
m1.metric("Timestamp", formatted)
m2.metric("Max vm_pu", f"{frame['max_vm_pu']:.4f}" if frame.get("max_vm_pu") is not None else "—")
m3.metric("Min vm_pu", f"{frame['min_vm_pu']:.4f}" if frame.get("min_vm_pu") is not None else "—")
m4.metric("Step", "⚠️ Violation" if frame.get("violation_flag")
          else ("✅ OK" if frame.get("converged") else "✗ diverged"))

# --- Voltage envelope over the full run, with a marker at the selected step -
ts_df = pd.DataFrame(full_ts)[["t", "timestamp", "max_vm_pu", "min_vm_pu"]]
ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"], utc=True, errors="coerce")
env = go.Figure()
env.add_trace(go.Scatter(x=ts_df["timestamp"], y=ts_df["max_vm_pu"], name="max vm_pu", line=dict(color="firebrick")))
env.add_trace(go.Scatter(x=ts_df["timestamp"], y=ts_df["min_vm_pu"], name="min vm_pu", line=dict(color="steelblue")))
env.add_hline(y=v_min, line_dash="dot", line_color="gray")
env.add_hline(y=v_max, line_dash="dot", line_color="gray")
cur_ts = ts_df.loc[t_idx, "timestamp"]
if pd.notna(cur_ts):
    env.add_vline(x=cur_ts, line_color="black")
env.update_xaxes(tickformat="%d %b\n%H:%M", hoverformat="%d %b %Y %H:%M")
env.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(env, use_container_width=True)
