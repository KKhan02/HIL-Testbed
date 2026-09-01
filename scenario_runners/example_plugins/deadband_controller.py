"""
example_plugins/deadband_controller.py
=======================================
Example plugin — deadband Q(V) controller with configurable thresholds.

Fully self-contained: imports only numpy.

Implements a four-breakpoint piecewise linear Q(V) characteristic, the same
family as VDE-AR-N 4110 Bild 8 (mirrored by QVCharacteristic in
volt_var_controller.py and by the Arduino firmware), but with every
breakpoint exposed as a keyword argument so students can sweep curve shapes
from a YAML file without touching framework code:

    vm <= v_sat_low                  ->  q = +q_ratio * p_mw   (full injection)
    v_sat_low  < vm < v_db_low       ->  linear  +q_max -> 0
    v_db_low  <= vm <= v_db_high     ->  q = 0                 (deadband)
    v_db_high  < vm < v_sat_high     ->  linear  0 -> -q_max
    vm >= v_sat_high                 ->  q = -q_ratio * p_mw   (full absorption)

Defaults reproduce the VDE-AR-N 4110 breakpoints used by the built-in
scenarios (0.96 / 0.99 / 1.01 / 1.04) with q_ratio = 0.25, so with default
kwargs and gate_clean_timesteps: true this plugin should closely track
Scenario 4A (local Q(V), dry-run) — a useful end-to-end plumbing check
before experimenting with custom curves.  (Not bit-identical: 4A adds PT1
dynamics and the curtailment backstop, which this path omits by design.)
"""

from __future__ import annotations

import numpy as np


def compute_setpoints(
        vm_pu:      np.ndarray,
        p_mw:       np.ndarray,
        v_sat_low:  float = 0.96,
        v_db_low:   float = 0.99,
        v_db_high:  float = 1.01,
        v_sat_high: float = 1.04,
        q_ratio:    float = 0.25,
) -> np.ndarray:
    """
    Piecewise linear deadband Q(V) characteristic.

    Parameters
    ----------
    vm_pu : np.ndarray, shape (n_ders,)
        Voltage magnitude (pu) at each DER bus, in sgen_indices order.
    p_mw : np.ndarray, shape (n_ders,)
        Installed capacity (MW) per DER, same order.
    v_sat_low, v_db_low, v_db_high, v_sat_high : float
        Curve breakpoints (pu).  Must satisfy
        v_sat_low < v_db_low <= v_db_high < v_sat_high.
    q_ratio : float
        Q clamp as a fraction of installed capacity: |q| <= q_ratio * p_mw.

    Returns
    -------
    np.ndarray, shape (n_ders,)
        q_mvar setpoints, same order as the inputs.

    Raises
    ------
    ValueError : if the breakpoints are not strictly ordered.
    """
    if not (v_sat_low < v_db_low <= v_db_high < v_sat_high):
        raise ValueError(
            "deadband_controller: breakpoints must satisfy "
            "v_sat_low < v_db_low <= v_db_high < v_sat_high, got "
            f"{v_sat_low} / {v_db_low} / {v_db_high} / {v_sat_high}."
        )

    vm_pu = np.asarray(vm_pu, dtype=float)
    p_mw  = np.asarray(p_mw,  dtype=float)
    q_max = q_ratio * np.abs(p_mw)

    # np.interp is monotone piecewise-linear with flat extrapolation at both
    # ends — exactly the clamped characteristic.  Interpolate the NORMALISED
    # curve (per-unit of q_max), then scale per DER.
    q_norm = np.interp(
        vm_pu,
        [v_sat_low, v_db_low, v_db_high, v_sat_high],
        [1.0,       0.0,      0.0,       -1.0],
    )
    return q_norm * q_max
