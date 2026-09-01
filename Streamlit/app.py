
"""
HIL Testbed — Live/Results Dashboard (starter example)
========================================================
Run with:
    streamlit run app.py

This is intentionally decoupled from the simulation code (der_dynamics.py,
pandapower, simbench, etc. are NOT imported here). It only reads the files
the simulation already produced:

    outputs/benchmarks/<network>_benchmark_*.csv   <- one row per scenario
    outputs/figures/<network>/*.png                <- pre-made plots

Point DATA_DIR below at wherever you've copied/kept the `outputs/` folder
from the main repo (or your own `data/` copy of it — doesn't matter, this
script doesn't care where it came from).
"""

from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------------------------------------------------------------------
# 1. CONFIG — the only place you need to edit paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"     # <- change if your data lives elsewhere
NETWORK  = "1-MV-rural--2-sw"                  # the network folder/prefix used in outputs

BENCHMARK_DIR = DATA_DIR / "benchmarks"
FIGURES_DIR   = DATA_DIR / "figures" / NETWORK

st.set_page_config(page_title="HIL Grid Testbed Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# 2. DATA LOADING — cached so it only reads from disk once per session
# ---------------------------------------------------------------------------
@st.cache_data
def load_benchmark_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def find_benchmark_csv(network: str) -> Path | None:
    """Grab the most recent non-'hc_stressed' benchmark CSV for this network."""
    if not BENCHMARK_DIR.exists():
        return None
    candidates = sorted(
        p for p in BENCHMARK_DIR.glob(f"{network}_benchmark_*.csv")
    )
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# 3. LAYOUT
# ---------------------------------------------------------------------------
st.title("⚡ HIL Grid Testbed — Scenario Comparison")
st.caption(f"Network: `{NETWORK}`")

csv_path = find_benchmark_csv(NETWORK)

if csv_path is None:
    st.error(
        f"No benchmark CSV found in `{BENCHMARK_DIR}`. "
        "Copy the outputs/benchmarks folder from the main repo into your data/ folder."
    )
    st.stop()

df = load_benchmark_csv(csv_path)
st.success(f"Loaded `{csv_path.name}` — {len(df)} scenarios")

# --- Raw table (always useful, cheap to show) ---
with st.expander("Raw comparison table"):
    st.dataframe(df, use_container_width=True)

# --- Pick metrics to compare across scenarios ---
st.subheader("Scenario comparison")

metric_options = [
    "violation_duration_h",
    "vdi",
    "max_vm_pu",
    "min_vm_pu",
    "total_losses_mwh",
    "curtailed_energy_mwh",
]
available_metrics = [m for m in metric_options if m in df.columns]

col1, col2 = st.columns([1, 3])
with col1:
    metric = st.selectbox("Metric", available_metrics)
    st.markdown(
        """
        **Quick reading guide**
        - `violation_duration_h`: hours/year outside voltage limits (lower = better)
        - `vdi`: voltage deviation index (lower = better)
        - `max_vm_pu` / `min_vm_pu`: voltage extremes (should stay within 0.95–1.05)
        - `total_losses_mwh`: network energy losses
        - `curtailed_energy_mwh`: solar/wind energy thrown away by the controller
        """
    )

with col2:
    fig = px.bar(
        df,
        x="scenario_label",
        y=metric,
        color="scenario_label",
        title=f"{metric} by scenario",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# --- Existing static figures the simulation already generated ---
st.subheader("Pre-generated figures")

if FIGURES_DIR.exists():
    figs = sorted(FIGURES_DIR.glob("*.png"))
    if figs:
        pick = st.selectbox("Choose a figure", [f.name for f in figs])
        st.image(str(FIGURES_DIR / pick), use_container_width=True)
    else:
        st.info("No PNG figures found in the figures folder.")
else:
    st.info(f"Figures folder not found at `{FIGURES_DIR}` — copy it from the repo's outputs/figures/{NETWORK}.")
