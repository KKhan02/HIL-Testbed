"""
example_plugins/droop_controller.py
====================================
Example plugin — proportional Q-P droop controller.

Fully self-contained: imports only numpy.  No project modules, no
pandapower.  This is the reference example for the controller_fn contract
documented in custom_controller.py.

Control law
-----------
    q = -droop_slope * (vm_pu - 1.0) * p_mw      (outside the deadband)
    q = 0                                        (inside  the deadband)
    clipped to +/- q_ratio * p_mw

Sign convention (pandapower): overvoltage (vm_pu > 1) produces negative Q
(absorption, lowers voltage); undervoltage produces positive Q (injection,
raises voltage).

Units note on droop_slope
-------------------------
droop_slope has units of 1/pu: q [MVAr] = -slope * dV [pu] * p_mw [MW].
To reach the full +/- q_ratio * p_mw clamp at a voltage deviation dV_sat,
choose  slope = q_ratio / dV_sat.  Example: q_ratio=0.25 saturating at
dV=0.05 pu  ->  slope = 5.0.  The YAML format example in the project spec
uses droop_slope: 0.05, which produces very small Q (0.25 % of p_mw at a
5 % deviation) — fine for demonstrating the plumbing, too weak for
meaningful voltage control.  droop_controller.yaml ships with 5.0.

The deadband is applied with an offset (the droop acts on the voltage
deviation BEYOND the deadband edge) so Q is continuous at the deadband
boundary — no step in the control law.

q_ratio should mirror Q_RATIO in volt_var_controller.py (currently 0.25)
for a fair comparison against Scenarios 4A/4B, but the plugin is
deliberately independent of that module, so it is a plain kwarg here.
"""

from __future__ import annotations

import numpy as np


def compute_setpoints(
        vm_pu:       np.ndarray,
        p_mw:        np.ndarray,
        droop_slope: float = 5.0,
        deadband:    float = 0.01,
        q_ratio:     float = 0.25,
) -> np.ndarray:
    """
    Proportional Q-P droop with symmetric deadband.

    Parameters
    ----------
    vm_pu : np.ndarray, shape (n_ders,)
        Voltage magnitude (pu) at each DER bus, in sgen_indices order.
    p_mw : np.ndarray, shape (n_ders,)
        Installed capacity (MW) per DER, same order.
    droop_slope : float
        Droop gain in 1/pu (see units note in the module docstring).
    deadband : float
        Half-width of the no-action band around 1.0 pu (pu).
    q_ratio : float
        Q clamp as a fraction of installed capacity: |q| <= q_ratio * p_mw.

    Returns
    -------
    np.ndarray, shape (n_ders,)
        q_mvar setpoints, same order as the inputs.
    """
    vm_pu = np.asarray(vm_pu, dtype=float)
    p_mw  = np.asarray(p_mw,  dtype=float)

    dv = vm_pu - 1.0
    # Deviation beyond the deadband edge (0 inside the band, continuous at
    # the boundary).
    dv_eff = np.sign(dv) * np.maximum(np.abs(dv) - deadband, 0.0)

    q     = -droop_slope * dv_eff * p_mw
    q_max = q_ratio * np.abs(p_mw)

    return np.clip(q, -q_max, q_max)
