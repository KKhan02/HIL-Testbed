"""
Hosting capacity
================
Consumes hc.json (written by publish_hc_and_comparison for hosting-capacity
studies). Shows baseline vs Volt-Var hosting capacity, the added-MW gain, and
the sweep curve (added MW vs worst-case bus voltage) for each case.
"""
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _data

st.set_page_config(page_title="Hosting Capacity", layout="wide")
run_dir = _data.sidebar_run_selector()
st.title("🔋 Hosting Capacity")

if run_dir is None:
    st.info("Pick a run in the sidebar.")
    st.stop()

hc = _data.load_hc(run_dir)
if hc is None:
    st.error("This run has no `hc.json`. It's only written for hosting-capacity studies (run_hc=True).")
    st.stop()
if "error" in hc:
    st.warning(f"HC results unavailable for this run: {hc['error']}")
    st.stop()

baseline = hc.get("baseline", {})
volt_var = hc.get("volt_var", {})
gain = hc.get("gain_mw")

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
m1, m2, m3 = st.columns(3)
m1.metric("Baseline HC", f"{baseline.get('hc_mw', float('nan')):.3f} MW")
m2.metric("Volt-Var HC", f"{volt_var.get('hc_mw', float('nan')):.3f} MW")
if gain is not None:
    m3.metric("HC gain", f"{gain:+.3f} MW")

# ---------------------------------------------------------------------------
# Sweep curve: added MW vs worst-case bus voltage
# ---------------------------------------------------------------------------
st.subheader("Sweep curve — worst-case bus voltage vs added DER")

topo = _data.load_topology(run_dir) or {}
v_max = topo.get("voltage_limits", {}).get("v_max", 1.05)

fig = go.Figure()
for case, color, label in (
    (baseline, _data.SCENARIO_COLORS["baseline"], "Baseline"),
    (volt_var, _data.SCENARIO_COLORS["volt_var_local"], "Volt-Var"),
):
    curve = case.get("sweep_curve") or []
    if curve:
        fig.add_trace(go.Scatter(
            x=[p["mw"] for p in curve],
            y=[p["max_vm_pu"] for p in curve],
            name=label, mode="lines+markers", line=dict(color=color),
        ))
    hc_mw = case.get("hc_mw")
    if hc_mw is not None:
        fig.add_vline(x=hc_mw, line_dash="dot", line_color=color)

fig.add_hline(y=v_max, line_dash="dash", line_color="crimson",
              annotation_text=f"v_max = {v_max}", annotation_position="top left")
fig.update_layout(
    height=420, margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Added DER capacity [MW]", yaxis_title="max vm_pu",
    legend=dict(orientation="h"),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Dotted verticals mark each case's hosting capacity (last non-violating "
    "step). The curve has terminal points only — per-step voltages aren't "
    "stored in HCResult."
)

# ---------------------------------------------------------------------------
# Detail table
# ---------------------------------------------------------------------------
with st.expander("HC detail"):
    fields = [
        "hc_mw", "violated_at_mw", "binding_bus", "binding_vm_pu", "n_steps",
        "hc_limit_reached", "endoffeeder_bus", "dist_voltage_kv",
        "qv_converged", "qv_iters_max",
        "sweep_start_mw", "sweep_step_mw", "sweep_max_mw",
    ]
    rows = {f: {"baseline": baseline.get(f), "volt_var": volt_var.get(f)} for f in fields}
    st.dataframe(rows, use_container_width=True)
