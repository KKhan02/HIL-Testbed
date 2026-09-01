"""
Live network view
==================
Tails live/<scenario_id>.jsonl — the per-timestep stream written by
PublishHandle.on_timestep during a running benchmark (cadence =
update_every_k). Colours the network map by the latest frame's bus voltages
and updates as new frames land. Pair with Streamlit running on the RPi (it
reads its own live file) for a true live view; on a finished run it replays
the captured frames.

Incremental tailing: only new bytes are read each refresh (via a stored byte
offset), and a truncation — on_scenario_start resets the live file — cleanly
restarts the accumulated history. So this stays light even for a full run.
"""
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import _data

st.set_page_config(page_title="Live View", layout="wide")
run_dir = _data.sidebar_run_selector()
st.title("🔴 Live Network View")

if run_dir is None:
    st.info("Pick a run in the sidebar.")
    st.stop()

topo = _data.load_topology(run_dir)
if topo is None:
    st.error("This run has no `topology.json`.")
    st.stop()

live = _data.list_live_scenarios(run_dir)
if not live:
    st.info(
        "No `live/*.jsonl` in this run yet. Live frames are written during a "
        "benchmark when `publish_fn` is wired (PublishHandle). Start a run "
        "(the file appears as soon as the first scenario begins), or open a "
        "finished run to replay its captured frames."
    )
    st.stop()

# Scenario picker. When Live is ON we follow the currently-streaming file (the
# most-recently-written one) unless you deliberately pick another; when paused,
# your selection sticks.
live_ids = [sid for sid, _ in live]
active_default = max(live, key=lambda t: t[1])[0]

c1, c2, c3 = st.columns([2, 1, 1])
with c2:
    live_on = st.toggle("🔴 Live", value=False)
with c3:
    refresh_s = st.slider("Refresh (s)", 0.5, 5.0, 1.5, step=0.5)

# If Live is on and the active stream has moved on to a newer scenario, jump to
# it (only when the user hasn't manually selected a different one this session).
if live_on and st.session_state.get("live_follow", True):
    st.session_state["live_scenario"] = active_default
default_sid = st.session_state.get("live_scenario", active_default)
if default_sid not in live_ids:
    default_sid = active_default

with c1:
    prev = st.session_state.get("live_scenario")
    scenario_id = st.selectbox(
        "Scenario", live_ids, index=live_ids.index(default_sid),
        format_func=_data.scenario_label,
    )
    # A manual change turns off auto-follow; matching the active stream re-arms it.
    if scenario_id != prev:
        st.session_state["live_follow"] = (scenario_id == active_default)
    st.session_state["live_scenario"] = scenario_id

live_path = run_dir / "live" / f"{scenario_id}.jsonl"

# ---------------------------------------------------------------------------
# Accumulate frames across reruns (keyed by run + scenario; reset on switch
# or on a live-file truncation).
# ---------------------------------------------------------------------------
key = f"{run_dir}|{scenario_id}"
if st.session_state.get("live_key") != key:
    st.session_state.live_key = key
    st.session_state.live_frames = []
    st.session_state.live_offset = 0

new_frames, new_offset, reset = _data.read_jsonl_since(live_path, st.session_state.live_offset)
if reset:
    st.session_state.live_frames = []
st.session_state.live_offset = new_offset
st.session_state.live_frames.extend(new_frames)

frames = st.session_state.live_frames
if not frames:
    st.warning("Waiting for the first frame…")
    if live_on:
        time.sleep(refresh_s)
        st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Which frame to show: latest when live, scrubbable when paused.
# ---------------------------------------------------------------------------
# Frame index. The slider is ALWAYS rendered (disabled while live) so the
# element tree is identical whether Live is on or off — a slider that appears
# only when paused shifts the tree and leaves stale duplicates on toggle.
if live_on:
    idx = len(frames) - 1
    st.slider("Frame", 0, max(len(frames) - 1, 0), idx, disabled=True)
    st.caption(f"Streaming **{_data.scenario_label(scenario_id)}** — {len(frames):,} frames captured")
else:
    idx = st.slider("Frame", 0, max(len(frames) - 1, 0), len(frames) - 1)
    st.caption(f"Paused — scrubbing {len(frames):,} captured frames")

frame = frames[idx]

progress = frame.get("progress")
if progress is not None:
    st.progress(min(max(progress, 0.0), 1.0), text=f"Simulation progress: {progress * 100:.1f}%")

# ---------------------------------------------------------------------------
# NETWORK MAP (latest / selected frame)
# ---------------------------------------------------------------------------
buses_df = pd.DataFrame(topo["buses"])
vm_pu_by_bus = frame.get("vm_pu_by_bus") or {}
buses_df["vm_pu"] = buses_df["index"].astype(str).map(vm_pu_by_bus)
has_coords = buses_df["x"].notna().any() and buses_df["y"].notna().any()
v_min = topo.get("voltage_limits", {}).get("v_min", 0.95)
v_max = topo.get("voltage_limits", {}).get("v_max", 1.05)

if has_coords:
    line_loading = frame.get("line_loading_pct") or {}
    bus_xy = buses_df.set_index("index")[["x", "y"]]
    fig = go.Figure()
    for line in topo["lines"]:
        fb, tb = line["from_bus"], line["to_bus"]
        if fb not in bus_xy.index or tb not in bus_xy.index:
            continue
        x0, y0 = bus_xy.loc[fb]
        x1, y1 = bus_xy.loc[tb]
        if pd.isna(x0) or pd.isna(x1):
            continue
        loading = line_loading.get(str(line["index"]))
        color = "crimson" if (loading is not None and loading > 90) else "lightslategray"
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                 line=dict(color=color, width=2), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=buses_df["x"], y=buses_df["y"], mode="markers",
        marker=dict(size=14, color=buses_df["vm_pu"], colorscale="RdYlBu_r",
                    cmin=v_min - 0.02, cmax=v_max + 0.02, colorbar=dict(title="vm_pu"),
                    line=dict(width=1, color="black")),
        text=[f"Bus {i}: {v:.4f} pu" if v is not None and pd.notna(v) else f"Bus {i}: n/a"
              for i, v in zip(buses_df["index"], buses_df["vm_pu"])],
        hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(height=520, xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No bus coordinates — showing the table view.")
    st.dataframe(buses_df[["index", "name", "vn_kv", "vm_pu"]], use_container_width=True)

# ---------------------------------------------------------------------------
# FRAME DETAIL — voltages, convergence, and whichever control scalar applies
# ---------------------------------------------------------------------------
ts = frame.get("timestamp")
formatted = pd.to_datetime(ts).strftime("%d/%m %H:%M") if ts else "—"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Timestamp", formatted)
m2.metric("Max vm_pu", f"{frame['max_vm_pu']:.4f}" if frame.get("max_vm_pu") is not None else "—")
m3.metric("Min vm_pu", f"{frame['min_vm_pu']:.4f}" if frame.get("min_vm_pu") is not None else "—")
m4.metric("Step", "⚠️ Violation" if frame.get("violation_flag")
          else ("✅ OK" if frame.get("converged") else "✗ diverged"))

# Control scalars — ALWAYS render a fixed 4-metric row so the element tree is
# identical across scenarios; a variable st.columns(len(...)) shifts the tree
# and makes Streamlit leave a stale (duplicate) chart on scenario switch.
ctrl = [
    ("Tap position", frame.get("tap_pos")),
    ("SVC Q (Mvar)", frame.get("svc_q_mvar")),
    ("Coordinator",  frame.get("coordination_active")),
    ("Curtailment",  frame.get("curtailment_needed")),
]
ccols = st.columns(4)
for col, (label, val) in zip(ccols, ctrl):
    if val is None:
        col.metric(label, "—")
    elif isinstance(val, bool):
        col.metric(label, "yes" if val else "no")
    elif isinstance(val, float):
        col.metric(label, f"{val:.3f}")
    else:
        col.metric(label, str(val))

# ---------------------------------------------------------------------------
# ROLLING VOLTAGE ENVELOPE over captured frames
# ---------------------------------------------------------------------------
st.subheader("Voltage envelope (frames captured so far)")
env_df = pd.DataFrame([
    {"timestamp": f.get("timestamp"), "max_vm_pu": f.get("max_vm_pu"), "min_vm_pu": f.get("min_vm_pu")}
    for f in frames
])
env_df["timestamp"] = pd.to_datetime(env_df["timestamp"], utc=True, errors="coerce")

env = go.Figure()
env.add_trace(go.Scatter(x=env_df["timestamp"], y=env_df["max_vm_pu"], name="max vm_pu", line=dict(color="firebrick")))
env.add_trace(go.Scatter(x=env_df["timestamp"], y=env_df["min_vm_pu"], name="min vm_pu", line=dict(color="steelblue")))
env.add_hline(y=v_min, line_dash="dot", line_color="gray")
env.add_hline(y=v_max, line_dash="dot", line_color="gray")
cur_ts = env_df["timestamp"].iloc[idx]
if pd.notna(cur_ts):
    env.add_vline(x=cur_ts, line_color="black")
env.update_xaxes(tickformat="%d %b\n%H:%M", hoverformat="%d %b %Y %H:%M")
env.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(env, use_container_width=True)

# ---------------------------------------------------------------------------
# LOAD & GENERATION (input profile) — fixed element structure so the tree
# shape never changes between reruns (a variable/conditional block here is
# what makes Streamlit leave a stale duplicate behind on scenario switch).
# ---------------------------------------------------------------------------
st.subheader("Load & generation")
prof = _data.load_profiles(run_dir)
lg = go.Figure()
if prof and prof.get("times_10min"):
    pt = pd.to_datetime(pd.Series(prof["times_10min"]), utc=True, errors="coerce")
    lo, hi = env_df["timestamp"].min(), env_df["timestamp"].max()
    mask = ((pt >= lo) & (pt <= hi)) if (pd.notna(lo) and pd.notna(hi)) else pd.Series([True] * len(pt))
    for name, key, color in [
        ("Load",     "load_total_10min",    "#8d99ae"),
        ("PV",       "pv_total_10min",      _data.SGEN_TYPE_COLORS["pv"]),
        ("Wind",     "wind_total_10min",    _data.SGEN_TYPE_COLORS["wind"]),
        ("Net injection (gen − load)", "net_injection_10min", "#1a9641"),
    ]:
        vals = prof.get(key)
        if vals and len(vals) == len(pt):
            lg.add_trace(go.Scatter(x=pt[mask.values], y=pd.Series(vals)[mask.values],
                                    name=name, line=dict(color=color)))
    lg.add_hline(y=0, line_dash="dot", line_color="gray")
    if pd.notna(cur_ts):
        lg.add_vline(x=cur_ts, line_color="black")
    lg.update_xaxes(tickformat="%d %b\n%H:%M", hoverformat="%d %b %Y %H:%M")
else:
    lg.add_annotation(text="No profiles.json for this run", showarrow=False)
lg.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10),
                 yaxis_title="MW", legend=dict(orientation="h"))
st.plotly_chart(lg, use_container_width=True)

# ---------------------------------------------------------------------------
# LIVE REFRESH LOOP
# ---------------------------------------------------------------------------
if live_on:
    finished = progress is not None and progress >= 0.9999
    if finished:
        st.success("Scenario finished streaming — toggle Live off to scrub the captured frames.")
    else:
        time.sleep(refresh_s)
        st.rerun()
