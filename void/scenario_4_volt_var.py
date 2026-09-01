"""
scenario_4_volt_var.py
======================
Scenario 4 — Rule-based Volt-Var HIL (main contribution).

Implements the full Tier 1 manual timestep loop:

    [A] Select p_target in controller.sgen_indices order.
        Do not write net.sgen.p_mw here — run_coordinated_timestep owns that write.
    [B] Write profile load P and Q to net.load (index-explicit)
    [C] call run_coordinated_timestep() → CoordinatorResult
        This single call encapsulates:
            [C1] Pre-PF (q=0, p=p_target)          → report_pre
            [C2] Arduino Q(V) exchange  (Item 2)    → q_initial
            [C3] Sensitivity coordination (Item 3)  → q_adjusted
            [C4] DER dynamics PT1 + ramp  (Item 4)  → q_applied, p_applied
            [C5] Write q_applied, p_applied to net.sgen
            [C6] Post-PF                            → report_post
    [D] Build TimestepRecord from CoordinatorResult
    [E] Log curtailment_needed flag (placeholder — not yet acted upon)
    [F] Advance timestep

Architecture notes
------------------
- VoltVarController, SensitivityCoordinator, and DERDynamics are constructed
  once before the loop and reused across all timesteps.  Their state (q_prev,
  p_prev in DERDynamics) is what makes the loop stateful.

- DERDynamics.reset() is called once before the loop with the first profile
  row so that q_prev = 0 and p_prev = p_target[0].  Without this, the first
  timestep's ramp check is against zeros which may produce a spurious large
  ramp delta.

- The sgen_indices owned by VoltVarController are used for all index-explicit
  net.sgen assignments.  Do NOT use positional indices; the controller's
  sgen_idx is sorted ascending and may not be contiguous.

- run_coordinated_timestep() writes p_applied and q_applied to net.sgen
  internally (step [C5]).  The runner must NOT write to net.sgen after this
  call in the same timestep.

- curtailment_needed is logged but not acted upon.  When active_power_curtailment
  is implemented, it will be called here when this flag is True, and the
  loop will need a second runpp() to assess the post-curtailment state.

Usage
-----
    from profile_builder import build_profiles
    from scenario_result import adapt_profiles
    from scenario_4_volt_var import run_scenario_4

    net      = sb.get_simbench_net("1-MV-rural--2-sw")
    profiles = build_profiles(net, ...)

    # Hardware mode:
    result = run_scenario_4(net, profiles, port="/dev/ttyACM0")

    # Dry-run (no Arduino):
    result = run_scenario_4(net, profiles, dry_run=True)

    print(result.n_violation_steps)
    print(result.curtailment_steps)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandapower as pp
import pandas as pd

from der_dynamics import (
    DERDynamics,
    MV_T95_Q_S,
    MV_RAMP_RATE_P_BASE,
)
from sensitivity_coordinator import (
    SensitivityCoordinator,
    run_coordinated_timestep,
)
from volt_var_controller import (
    ArduinoSerialInterface,
    VoltVarController,
    Q_RATIO,
)
from scenario_result import (
    AdaptedProfiles,
    ScenarioResult,
    TimestepRecord,
    adapt_profiles,
)
from violation_detector import (
    V_MIN,
    V_MAX,
    LINE_MAX_LOADING,
    TRAFO_MAX_LOADING,
    VOLTAGE_EPSILON,
    LOADING_EPSILON,
)

logger = logging.getLogger(__name__)


def _build_dynamics(ctrl: VoltVarController, dt_s: float) -> DERDynamics:
    """
    Construct a DERDynamics instance from the VoltVarController's DER array.

    q_max_mvar is derived from the controller's p_installed_mw using the
    same Q_RATIO constant used by the VDE-AR-N 4110 Q(V) curve, so the
    dynamics layer and the Q(V) curve share the same Q ceiling.

    p_rated_mw uses p_installed_mw directly (sn_mva where available, p_mw
    fallback — same resolution logic as VoltVarController._resolve_p_installed).

    Standard preset: MV_T95_Q_S=10 s, MV_RAMP_RATE_P_BASE=0.5%/s.
    Both are the central modelling values from VDE-AR-N 4110.  Pass
    alternative presets (MV_T95_Q_SLOW_S, MV_RAMP_RATE_P_FAST etc.) for
    sensitivity studies.
    """
    p_rated  = ctrl.p_installed_mw                         # shape (n_ders,) [MW]
    q_max    = Q_RATIO * p_rated                           # 0.48 × P_b_inst [MVAr]

    # Guard against zero/negative p_rated that would fail DERDynamics
    # validation — replace with a small positive sentinel (1 W = 1e-6 MW).
    p_rated  = np.where(p_rated > 0.0, p_rated, 1e-6)
    q_max    = np.where(q_max   > 0.0, q_max,   1e-6)

    return DERDynamics(
        dt_s               = dt_s,
        t95_q_s            = MV_T95_Q_S,
        ramp_rate_p_frac_s = MV_RAMP_RATE_P_BASE,
        q_max_mvar         = q_max,
        p_rated_mw         = p_rated,
    )


def run_scenario_4(
        net,
        profiles:   dict,
        network_id: str   = "unknown",
        port:       str   = "/dev/ttyACM0",
        dry_run:    bool  = False,
        v_min:      float = V_MIN,
        v_max:      float = V_MAX,
) -> ScenarioResult:
    """
    Run Scenario 4 — Rule-based Volt-Var HIL (main contribution).

    Parameters
    ----------
    net        : pandapower network.  Modified in place every timestep.
                 Caller should deep-copy if the original net is needed later.
    profiles   : dict from profile_builder.build_profiles().
    network_id : human-readable identifier stored in ScenarioResult.
    port       : serial port for Arduino (e.g. "/dev/ttyACM0" on RPi,
                 "COM3" on Windows).  Ignored when dry_run=True.
    dry_run    : if True, skips all serial communication and computes Q
                 locally using the pure-Python QVCharacteristic.  Use for
                 offline testing and CI.
    v_min      : lower voltage planning limit (pu).  Default 0.95.
    v_max      : upper voltage planning limit (pu).  Default 1.05.

    Returns
    -------
    ScenarioResult  with scenario_id="volt_var".
    """
    t_start = time.perf_counter()

    ap: AdaptedProfiles = adapt_profiles(net, profiles)
    time_steps = range(len(ap.times))
    der_idx    = ap.der_p.columns     # sgen indices for all DERs (PV + wind)

    # Guard: Scenario 4 requires at least one profiled DER.  ap.der_p is the
    # source of truth for DER ordering; an empty der_p means VoltVarController
    # would fall back to all in-service sgens, breaking index alignment.
    if ap.der_p.empty:
        raise ValueError(
            "Scenario 4 requires at least one profiled DER in ap.der_p. "
            "Check that profiles['pv'] or profiles['wind'] are non-empty."
        )

    logger.info(
        "[Scenario 4 | %s] Starting Volt-Var HIL: %d timesteps, "
        "%d DERs (%d PV + %d wind), %d loads | dry_run=%s",
        network_id, len(time_steps),
        len(der_idx), len(ap.pv_idx), len(ap.wind_idx),
        len(ap.load_idx), dry_run,
    )

    # ------------------------------------------------------------------
    # Reset any existing controllers (ConstControl etc.) left from a
    # previous scenario run on the same net object.
    # ------------------------------------------------------------------
    net.controller.drop(net.controller.index, inplace=True)
    pp.reset_results(net)

    records: list[TimestepRecord] = []

    # ------------------------------------------------------------------
    # Open Arduino (or skip in dry-run) and run the main loop.
    # ArduinoSerialInterface is used as a context manager so the port is
    # always closed even if an exception occurs mid-run.
    # ------------------------------------------------------------------
    iface: Optional[ArduinoSerialInterface] = None

    def _run_loop(arduino_iface: Optional[ArduinoSerialInterface]) -> None:
        nonlocal records

        # Construct controller with sgen_indices constrained to ap.der_p.columns.
        # This makes ap.der_p the single source of truth for DER ordering.
        # Without this, VoltVarController would discover all in-service sgens,
        # which may include elements absent from der_p — causing a KeyError
        # when indexing ap.der_p.iloc[t][ctrl.sgen_indices].
        ctrl = VoltVarController(
            net,
            interface    = arduino_iface,
            sgen_indices = ap.der_p.columns,
            dry_run      = dry_run,
        )
        ctrl.configure()

        coordinator = SensitivityCoordinator(net, ctrl)
        dynamics    = _build_dynamics(ctrl, ap.dt_s)

        # Initialise dynamics state.
        # q_prev = 0 (no prior Q injection at t=-1).
        # p_prev = first profile row so t=0 ramp is checked against the
        #          true starting P, not zero.
        p_init = ap.der_p.iloc[0].reindex(ctrl.sgen_indices).fillna(0.0).values.astype(float)
        dynamics.reset(q_init=0.0, p_init=p_init)

        # algorithm="nr" is mandatory: run_coordinated_timestep requires the
        # Newton-Raphson Jacobian stored in net._ppc["internal"]["J"].
        # Other algorithms (e.g. "bfsw") do not populate the Jacobian and
        # will cause a KeyError inside SensitivityCoordinator.
        runpp_kwargs = {
            "voltage_depend_loads": False,
            "algorithm":            "nr",
        }

        for t in time_steps:
            timestamp = ap.times[t]

            # [A] Select p_target in ctrl.sgen_indices order.
            #     Do NOT write net.sgen.p_mw here — run_coordinated_timestep
            #     owns the first write at step [0] to keep pre-PF consistent.
            p_target_row = ap.der_p.iloc[t].reindex(ctrl.sgen_indices).fillna(0.0).values.astype(float)

            # [B] Write load profiles (index-explicit).
            if not ap.load_p.empty:
                net.load.loc[ap.load_idx, "p_mw"]   = ap.load_p.iloc[t].values
                net.load.loc[ap.load_idx, "q_mvar"] = ap.load_q.iloc[t].values

            # [C] Full Tier 1 timestep.
            #     run_coordinated_timestep:
            #       [0] writes p_target to net.sgen.p_mw
            #       [1] resets q_mvar to 0
            #       [2] pre-PF → report_pre
            #       [3] Arduino Q(V) exchange → q_initial
            #       [4] sensitivity coordinate → q_adjusted
            #       [5] DERDynamics.step → q_applied, p_applied
            #       [6] writes q_applied, p_applied to net.sgen
            #       [7] post-PF → report_post
            result = run_coordinated_timestep(
                net          = net,
                controller   = ctrl,
                coordinator  = coordinator,
                dynamics     = dynamics,
                p_target     = p_target_row,
                runpp_kwargs = runpp_kwargs,
            )

            # [D] Extract results into TimestepRecord.
            #     Use post-PF results (report_post) as the definitive state.
            converged = result.post_pf_ok
            if converged and result.report_post is not None:
                rp = result.report_post
                vm = net.res_bus["vm_pu"].copy()
                ll = net.res_line["loading_percent"].copy()
                tl = net.res_trafo["loading_percent"].copy()

                ov_buses  = vm.index[vm > v_max + VOLTAGE_EPSILON].tolist()
                uv_buses  = vm.index[vm < v_min - VOLTAGE_EPSILON].tolist()
                ov_lines  = ll.index[ll > LINE_MAX_LOADING  + LOADING_EPSILON].tolist()
                ov_trafos = tl.index[tl > TRAFO_MAX_LOADING + LOADING_EPSILON].tolist()

                q_ser = pd.Series(result.q_applied,
                                  index=ctrl.sgen_indices, dtype=float)
                p_ser = pd.Series(result.p_applied,
                                  index=ctrl.sgen_indices, dtype=float)
            else:
                vm = pd.Series(dtype=float)
                ll = pd.Series(dtype=float)
                tl = pd.Series(dtype=float)
                ov_buses = uv_buses = ov_lines = ov_trafos = []
                q_ser = None
                p_ser = None

            rec = TimestepRecord(
                t=t, timestamp=timestamp,
                vm_pu=vm, line_loading=ll, trafo_loading=tl,
                over_voltage_buses=ov_buses, under_voltage_buses=uv_buses,
                overloaded_lines=ov_lines, overloaded_trafos=ov_trafos,
                q_applied_mvar=q_ser,
                p_applied_mw=p_ser,
                curtailment_needed=result.curtailment_needed,
                converged=converged,
                p_target_mw=pd.Series(p_target_row,
                                      index=ctrl.sgen_indices, dtype=float),
            )

            # [E] Curtailment placeholder — log but do not act.
            if result.curtailment_needed:
                logger.debug(
                    "T=%d (%s): curtailment_needed=True — "
                    "active_power_curtailment not yet implemented.",
                    t, timestamp,
                )

            records.append(rec)

            if t % 96 == 0:    # log progress every ~1 day (96 × 15 min)
                logger.info(
                    "[Scenario 4 | %s] t=%d/%d (%.1f %%) | "
                    "violations=%d | curtail_steps=%d",
                    network_id, t, len(time_steps),
                    100.0 * t / max(len(time_steps), 1),
                    sum(1 for r in records if r.over_voltage_buses
                        or r.under_voltage_buses),
                    sum(1 for r in records if r.curtailment_needed),
                )

    # ------------------------------------------------------------------
    # Hardware vs dry-run dispatch
    # ------------------------------------------------------------------
    if dry_run:
        logger.info("[Scenario 4 | %s] dry_run=True — no serial port opened.",
                    network_id)
        _run_loop(arduino_iface=None)
    else:
        with ArduinoSerialInterface(port=port) as iface:
            logger.info(
                "[Scenario 4 | %s] Arduino opened on %s.", network_id, port
            )
            _run_loop(arduino_iface=iface)

    elapsed = time.perf_counter() - t_start

    result = ScenarioResult.from_records(
        scenario_id = "volt_var",
        network_id  = network_id,
        records     = records,
        elapsed_s   = elapsed,
        dt_s        = ap.dt_s,
    )

    logger.info(
        "[Scenario 4 | %s] Done. %.1f s | %d/%d converged | "
        "%d violation steps | %d curtailment steps",
        network_id, elapsed, result.n_converged, result.n_timesteps,
        result.n_violation_steps, result.curtailment_steps,
    )
    return result
