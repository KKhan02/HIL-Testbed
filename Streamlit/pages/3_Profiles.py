"""
Load & DER profiles
====================
Consumes profiles.json (written by publish_topology_and_profiles). Shows the
annual overview at hourly resolution and a single-day view at full 10-min
resolution, with quick jumps to the four extreme days the builder found.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _data

st.set_page_config(page_title="Profiles", layout="wide")
run_dir = _data.sidebar_run_selector()
st.title("📈 Load & DER Profiles")

if run_dir is None:
    st.info("Pick a run in the sidebar.")
    st.stop()

prof = _data.load_profiles(run_dir)
if prof is None:
    st.error("This run has no `profiles.json`.")
    st.stop()

# ---------------------------------------------------------------------------
# Annual overview (hourly)
# ---------------------------------------------------------------------------
st.subheader("Annual overview (hourly means)")
times_h = pd.to_datetime(pd.Series(prof.get("times_hourly", [])), utc=True)

overview = go.Figure()
series_h = {
    "Load": ("load_total_mw", "#8d99ae"),
    "PV": ("pv_total_mw", _data.SGEN_TYPE_COLORS["pv"]),
    "Wind": ("wind_total_mw", _data.SGEN_TYPE_COLORS["wind"]),
    "Net injection": ("net_injection_mw", "#1a9641"),
}
for name, (key, color) in series_h.items():
    vals = prof.get(key)
    if vals:
        overview.add_trace(go.Scatter(x=times_h, y=vals, name=name, line=dict(color=color)))
overview.add_hline(y=0, line_dash="dot", line_color="gray")
overview.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                       yaxis_title="MW", legend=dict(orientation="h"))
overview.update_xaxes(tickformat="%d %b", hoverformat="%d %b %Y %H:%M")
st.plotly_chart(overview, use_container_width=True)

# ---------------------------------------------------------------------------
# Single-day 10-min detail
# ---------------------------------------------------------------------------
st.subheader("Day detail (10-min resolution)")
times_10 = pd.to_datetime(pd.Series(prof.get("times_10min", [])), utc=True)
if times_10.empty:
    st.info("No 10-min series in this profiles.json.")
    st.stop()

day_min = times_10.min().date()
day_max = times_10.max().date()
extreme = prof.get("extreme_days", {}) or {}

st.caption(
    "Extreme days — "
    + " · ".join(f"{k.replace('_', ' ')}: `{v}`" for k, v in extreme.items() if v)
)

quick = st.radio(
    "Jump to", ["Custom"] + [f"{k.replace('_', ' ')} ({v})" for k, v in extreme.items() if v],
    horizontal=True,
)
if quick != "Custom":
    default_day = pd.to_datetime(quick.split("(")[-1].rstrip(")")).date()
else:
    default_day = extreme.get("max_der")
    default_day = pd.to_datetime(default_day).date() if default_day else day_min

default_day = min(max(default_day, day_min), day_max)
day = st.date_input("Day", value=default_day, min_value=day_min, max_value=day_max)

mask = times_10.dt.date == day
if not mask.any():
    st.warning("No samples on that day.")
    st.stop()

detail = go.Figure()
series_10 = {
    "Load": ("load_total_10min", "#8d99ae"),
    "PV": ("pv_total_10min", _data.SGEN_TYPE_COLORS["pv"]),
    "Wind": ("wind_total_10min", _data.SGEN_TYPE_COLORS["wind"]),
    "Net injection": ("net_injection_10min", "#1a9641"),
}
x = times_10[mask]
for name, (key, color) in series_10.items():
    vals = prof.get(key)
    if vals:
        y = pd.Series(vals)[mask.values]
        detail.add_trace(go.Scatter(x=x, y=y, name=name, line=dict(color=color)))
detail.add_hline(y=0, line_dash="dot", line_color="gray")
detail.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                     yaxis_title="MW", legend=dict(orientation="h"))
detail.update_xaxes(tickformat="%H:%M", hoverformat="%d %b %Y %H:%M")
st.plotly_chart(detail, use_container_width=True)
