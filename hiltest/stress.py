"""
hiltest/stress.py
==================
Shared network stress helpers for all test sections that exercise
voltage-control algorithms.

Overvoltage stress rationale
-----------------------------
High generation + low demand = maximum reverse power flow = overvoltage.
This is the summer midday worst case: full solar irradiance with collapsed
demand (school holidays, industry shut down).  Without explicit stress most
networks show no violations at the nominal pandapower operating point, so
control tests would trivially pass without exercising anything meaningful.

Stress levels
-------------
All families use the same dry-run stress:
    sgen.p_mw   = sn_mva × 0.90   (near-rated PV output)
    load.p_mw  *= 0.20             (demand collapsed to 20 % of nominal)
    load.q_mvar = scaled to preserve original power factor (see below)

The hardware section applies gentler stress for Synthetic LV only, because
the Synthetic LV nominal values are low and the 90/20 condition produces
degenerate load-flow solutions on some variants.

Changes from original test_suite.py
-------------------------------------
- sn_mva clipping: after NaN/Inf sanitisation, sn_mva is clipped to a
  minimum of 1e-6 MVA element-wise.  Previously fillna(p_mw) could produce
  sn_mva=0.0 when p_mw was also 0.0, causing ZeroDivisionError or NaN
  injection into the sensitivity coordinator's Schur complement.
- q_mvar handling: load q_mvar is now scaled proportionally to p_mw rather
  than set to zero.  Setting q_mvar=0 forces unity power factor, which
  understates reactive demand stress and changes voltage behaviour.
- Timing: time.perf_counter() throughout.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Stress parameters (single source of truth — also printed in summaries)
# ---------------------------------------------------------------------------
STRESS_SGEN_FRACTION: float = 0.90   # sgen.p_mw = sn_mva × this
STRESS_LOAD_FRACTION: float = 0.20   # load.p_mw *= this; q_mvar scaled same


def _sanitise_sgen(net) -> None:
    """
    In-place: replace NaN / ±Inf in sgen columns with safe defaults.

    Guard: if net.sgen is empty (pure load network, no static generators)
    this function is a no-op — callers must check net.sgen.empty separately
    before applying generation stress.

    sn_mva is clipped to a minimum of 1e-6 MVA after fillna to prevent
    ZeroDivisionError downstream (coordinator q_max = Q_RATIO × p_installed,
    and p_installed = sn_mva × STRESS_SGEN_FRACTION).
    """
    if net.sgen.empty:
        return

    for col in ("p_mw", "q_mvar"):
        if col in net.sgen.columns:
            net.sgen[col] = (
                net.sgen[col]
                .replace([float("inf"), float("-inf")], np.nan)
                .fillna(0.0)
            )

    if "sn_mva" in net.sgen.columns:
        net.sgen["sn_mva"] = (
            net.sgen["sn_mva"]
            .replace([float("inf"), float("-inf")], np.nan)
            .fillna(net.sgen["p_mw"])   # best guess from p_mw
            .clip(lower=1e-6)           # never zero — prevents /0
        )


def _sanitise_load(net) -> None:
    """In-place: replace NaN / ±Inf in load columns with 0.0.

    Guard: no-op on empty load table.
    """
    if net.load.empty:
        return

    for col in ("p_mw", "q_mvar"):
        if col in net.load.columns:
            net.load[col] = (
                net.load[col]
                .replace([float("inf"), float("-inf")], np.nan)
                .fillna(0.0)
            )


def apply_overvoltage_stress(net, fraction: float = STRESS_LOAD_FRACTION) -> None:
    """
    Apply the standard overvoltage stress condition to a pandapower network.

    Steps
    -----
    1. Sanitise sgen and load tables (NaN / Inf / zero sn_mva).
    2. Set sgen.p_mw = sn_mva × STRESS_SGEN_FRACTION  (skipped if no sgens).
    3. Scale load.p_mw and load.q_mvar by `fraction` (default 0.20).
       q_mvar is scaled proportionally — NOT set to zero — so original power
       factor is preserved.
    4. Reset sgen.q_mvar = 0.0 (invariant required by coordinate()).

    Networks with no sgen table (pure load networks) have only their load
    scaled; generation steps are skipped. The calling test will then find
    n_ders == 0 and mark the case as SKIP via the no-DER gate.

    Parameters
    ----------
    net      : pandapowerNet — modified in-place.
    fraction : Load scaling factor (default STRESS_LOAD_FRACTION = 0.20).
    """
    _sanitise_sgen(net)
    _sanitise_load(net)

    if not net.sgen.empty and "sn_mva" in net.sgen.columns:
        net.sgen["p_mw"]   = net.sgen["sn_mva"] * STRESS_SGEN_FRACTION
        net.sgen["q_mvar"] = 0.0         # q=0 invariant for coordinate()

    if not net.load.empty:
        net.load["p_mw"]  *= fraction
        net.load["q_mvar"] *= fraction   # preserve power factor, not force to zero


def apply_hw_synthetic_stress(net) -> None:
    """
    Gentler stress for hardware Synthetic LV tests.

    Synthetic LV networks have very low nominal p_mw/sn_mva values; the
    standard 90/20 condition produces degenerate load-flow solutions on some
    variants.  Hardware tests use 20 % sgen / 40 % load instead.
    """
    _sanitise_sgen(net)
    _sanitise_load(net)

    if not net.sgen.empty and "sn_mva" in net.sgen.columns:
        net.sgen["p_mw"]   = net.sgen["sn_mva"] * 0.20
        net.sgen["q_mvar"] = 0.0

    if not net.load.empty:
        net.load["p_mw"]  *= 0.40
        net.load["q_mvar"] *= 0.40


'''def lv_runpp_kwargs(label: str) -> dict:
    """
    DEPRECIATED. DONOT NEED BSFW HARDCODED
    Return solver kwargs for LV networks (bfsw, max 30 iterations, flat init).
    Returns empty dict for MV/MVLV networks.

    Triggers on: 'synthetic', 'lv' (case-insensitive), 'dickert', 'kerber'.
    Matched against the network label string.
    """
    lv_markers = ("synthetic", "lv", "dickert", "kerber")
    if any(m in label.lower() for m in lv_markers):
        return {"algorithm": "bfsw", "max_iteration": 30, "init": "flat"}
    return {}'''

def stress_description() -> str:
    """
    Return a human-readable stress description for use in summary headers.
    Single source of truth — summaries import this instead of hardcoding.
    """
    return (
        f"  Stress: sgen {int(STRESS_SGEN_FRACTION * 100)}% of sn_mva, "
        f"load {int(STRESS_LOAD_FRACTION * 100)}% of nominal "
        f"(all families — dry-run)"
    )
