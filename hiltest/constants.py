"""
hiltest/constants.py
====================
Scalar configuration constants only.
No pandapower, simbench, or project module imports.

Network family catalogues have moved to hiltest/catalogues.py.
Representative network specs are in hiltest/networks.py.
"""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DWD_DATA_DIR  = "data/dwd"
ERA5_DATA_DIR = "data/era5"

ERA5_FILE_MAP = {
    "RAD-G": "era5_solar.csv",
    "F":     "era5_wind.csv",
    "T2M":   "era5_temp.csv",
}

ERA5_COL_MAP = {
    "timestamp": "timestamp",
    "solar":     "GHI_Wm2",
    "wind":      "WS_ms",
    "temp":      "AT_degC",
    "sep":       ",",
}

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
SHOW_PLOTS: bool = False   # True = interactive; False = headless / CI / RPi

# ---------------------------------------------------------------------------
# Hardware timing
# ---------------------------------------------------------------------------
# Maximum acceptable wall-clock time for the full coordinated HIL timestep
# (two pp.runpp() solves + Schur complement + serial exchange).
# Derivation: control-loop target is 1-second timesteps; budget is 80 % of
# that for the full cycle, leaving 200 ms for overhead.
# Adjust if your control-loop target changes.
HIL_MAX_CYCLE_MS: float = 800.0

# Maximum acceptable serial exchange time only (Arduino roundtrip).
# Must be well under HIL_MAX_CYCLE_MS to leave room for PF solves.
# Derivation: 500 ms ≤ 50 % of a 1-second loop.
HIL_MAX_EXCHANGE_MS: float = 500.0
