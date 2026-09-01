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
    [E] Active power curtailment loop (linear decay, up to MAX_CURTAIL_ITERS)
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

- curtailment_needed triggers a bounded linear-decay P curtailment loop at
  step [E]. Each iteration reduces P by CURTAIL_STEP_FRAC × p_target (10%)
  and re-runs runpp(). The loop exits when violations clear, P=0, or
  MAX_CURTAIL_ITERS (10) is reached. curtail_exhausted is set True when the
  loop terminates without clearing violations.

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
)
# Q_RATIO is read as a module attribute at USE time (not from-imported) so
# that a runtime override via volt_var_controller.set_qv_parameters() — used
# by the CLI executor for per-run Q(V) characteristics — is seen here too.
# A from-import would freeze the import-time value and desynchronise the
# dynamics/saturation ceilings from the curve the Arduino actually runs.
import volt_var_controller as _vvc
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
    detect_violations,
)

CURTAIL_STEP_FRAC: float = 0.10   # linear P reduction per iteration (fraction of p_target)
MAX_CURTAIL_ITERS: int   = 10     # 10 × 10% → floor at P=0 on final iteration

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
    q_max    = _vvc.Q_RATIO * p_rated                      # Q_RATIO × P_b_inst [MVAr] (call-time read)

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
        coordination: bool = True,   # False = 4A, True = 4B
        v_min:      float = V_MIN,
        v_max:      float = V_MAX,
        publish_fn         = None,
        enable_checkpointing: bool = True,
        live_csv_rewrite_fn = None,
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
    _T = len(ap.times)
    if publish_fn is not None:
        publish_fn.on_scenario_start(
            "volt_var_coord" if coordination else "volt_var_local",
            "Volt-Var HIL (+ coord)" if coordination else "Volt-Var HIL (local Q(V))",
            _T,
        )
    scenario_id_for_resume = "volt_var_coord" if coordination else "volt_var_local"
    resumed_records: list[TimestepRecord] = []
    if publish_fn is not None and enable_checkpointing:
        resumed_records = publish_fn.get_resume_records(scenario_id_for_resume)

    start_t = (resumed_records[-1].t + 1) if resumed_records else 0
    _T_full = len(ap.times)

    if start_t >= _T_full and resumed_records:
        logger.info(
            "[Scenario 4%s | %s] Checkpoint already covers all %d steps — skipping simulation.",
            "-coord" if coordination else "-local", network_id, _T_full,
        )
        elapsed = publish_fn.cumulative_elapsed_s() if publish_fn is not None else time.perf_counter() - t_start
        result = ScenarioResult.from_records(
            scenario_id=scenario_id_for_resume, network_id=network_id,
            records=resumed_records, elapsed_s=elapsed, dt_s=ap.dt_s,
        )
        if publish_fn is not None:
            publish_fn.on_scenario_end(result)
        return result

    if start_t > 0:
        logger.info(
            "[Scenario 4%s | %s] Resuming from t=%d/%d (%d records recovered).",
            "-coord" if coordination else "-local", network_id,
            start_t, _T_full, len(resumed_records),
        )

    time_steps = range(start_t, _T_full)

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
        "[Scenario %s | %s] Starting Volt-Var HIL: %d timesteps, "
        "%d DERs (%d PV + %d wind), %d loads | dry_run=%s | coordination=%s",
        "4B" if coordination else "4A",
        network_id, len(time_steps),
        len(der_idx), len(ap.pv_idx), len(ap.wind_idx),
        len(ap.load_idx), dry_run, coordination,
    )

    # ------------------------------------------------------------------
    # Reset any existing controllers (ConstControl etc.) left from a
    # previous scenario run on the same net object.
    # ------------------------------------------------------------------
    net.controller.drop(net.controller.index, inplace=True)
    pp.reset_results(net)

    records: list[TimestepRecord] = resumed_records.copy()

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

        # Q clamp array for q_saturated_count — same formula as _build_dynamics()
        # so the saturation check uses the identical ceiling that DERDynamics clips to.
        # Computed once here to avoid repeating the guarded arithmetic inside the loop.
        q_max_arr = np.where(
            _vvc.Q_RATIO * ctrl.p_installed_mw > 0.0,
            _vvc.Q_RATIO * ctrl.p_installed_mw,
            1e-6,
        )

        # Initialise dynamics state.
        # q_prev = 0 (no prior Q injection at t=-1).
        # p_prev = first profile row so t=0 ramp is checked against the
        #          true starting P, not zero.
        if resumed_records:
            last = resumed_records[-1]
            q_init = (
                last.q_applied_mvar.reindex(ctrl.sgen_indices).fillna(0.0).values.astype(float)
                if last.q_applied_mvar is not None
                else np.zeros(len(ctrl.sgen_indices))
            )
            p_init = (
                last.p_applied_mw.reindex(ctrl.sgen_indices).fillna(0.0).values.astype(float)
                if last.p_applied_mw is not None
                else ap.der_p.iloc[start_t].reindex(ctrl.sgen_indices).fillna(0.0).values.astype(float)
            )
            dynamics.reset(q_init=q_init, p_init=p_init)
            logger.info(
                "[Scenario 4%s | %s] DERDynamics state seeded from last checkpoint record.",
                "-coord" if coordination else "-local", network_id,
            )
        else:
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
        pre_violation_steps = 0
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
                coordination = coordination,
            )

            if result.report_pre is not None and result.report_pre.any_violations:
                pre_violation_steps += 1

            # [D] Extract converged-state variables needed for TimestepRecord.
            converged = result.post_pf_ok

            der_gen_mw_t = float(ap.der_p.iloc[t].sum())  if not ap.der_p.empty  else 0.0
            load_mw_t    = float(ap.load_p.iloc[t].sum()) if not ap.load_p.empty else 0.0

            if converged and result.report_post is not None:
                q_ser = pd.Series(result.q_applied,
                                  index=ctrl.sgen_indices, dtype=float)
                p_ser = pd.Series(result.p_applied,
                                  index=ctrl.sgen_indices, dtype=float)
                losses_mw_t      = float(net.res_line["pl_mw"].sum()
                                         + net.res_trafo["pl_mw"].sum())
                grid_import_mw_t = float(net.res_ext_grid["p_mw"].sum())
                coordination_active_t = bool(coordination) and bool(
                    np.any(np.abs(result.q_adjusted - result.q_initial) > 1e-6)
                )
                q_saturated_count_t = int(
                    np.sum(np.abs(result.q_applied) >= q_max_arr - 1e-6)
                )
            else:
                q_ser = None
                p_ser = None
                losses_mw_t = grid_import_mw_t = None
                coordination_active_t = None
                q_saturated_count_t   = None

            # [E] Active power curtailment — fires when Q coordination was
            #     insufficient to clear violations (curtailment_needed=True).
            #     Linear decay: p_curtailed[k] = p_target - k * step_mw.
            #     Iteration 10 reaches P=0 exactly. No dynamics.step() call
            #     inside the loop — p_prev updated once at exit to avoid
            #     corrupting ramp state with multiple intra-timestep steps.
            curtail_exhausted_t: Optional[bool] = None
            active_report = result.report_post   # will be replaced if curtailment runs
            active_p_ser  = p_ser                # will be replaced if curtailment runs

            if result.curtailment_needed and converged:
                step_mw             = CURTAIL_STEP_FRAC * p_target_row
                curtail_report      = result.report_post
                curtail_exhausted_t = False
                curtail_ok          = True
                p_curtailed         = p_target_row.copy()

                for curtail_iter in range(1, MAX_CURTAIL_ITERS + 1):
                    p_curtailed = np.maximum(
                        p_target_row - curtail_iter * step_mw, 0.0
                    )
                    net.sgen.loc[ctrl.sgen_indices, "p_mw"] = p_curtailed
                    # Q unchanged — already written by run_coordinated_timestep

                    try:
                        pp.runpp(net, **runpp_kwargs)
                        curtail_report = detect_violations(net)
                    except Exception as exc:
                        logger.warning(
                            "T=%d (%s): curtailment runpp() failed at iter %d: %s",
                            t, timestamp, curtail_iter, exc,
                        )
                        curtail_ok = False
                        curtail_exhausted_t = True
                        break

                    if not curtail_report.any_violations:
                        break   # violations cleared

                    if np.all(p_curtailed <= 0.0):
                        curtail_exhausted_t = True
                        break   # P=0 reached, violation is structural
                else:
                    curtail_exhausted_t = True   # loop ran MAX_CURTAIL_ITERS, no break

                logger.log(
                    logging.DEBUG if not curtail_exhausted_t else logging.WARNING,
                    "T=%d (%s): curtailment %s after %d iter(s) | "
                    "p_final=[%.3f..%.3f] MW | violations_remain=%s",
                    t, timestamp,
                    "exhausted" if curtail_exhausted_t else "cleared",
                    curtail_iter,
                    float(p_curtailed.min()), float(p_curtailed.max()),
                    curtail_report.any_violations if curtail_ok else "unknown",
                )

                if curtail_ok:
                    # Update ramp state to final curtailed P — no step() call
                    dynamics.p_prev = p_curtailed.copy()
                    active_report   = curtail_report
                    active_p_ser    = pd.Series(
                        p_curtailed, index=ctrl.sgen_indices, dtype=float
                    )
                    # Refresh energy metrics from the post-curtailment PF state
                    losses_mw_t      = float(net.res_line["pl_mw"].sum()
                                             + net.res_trafo["pl_mw"].sum())
                    grid_import_mw_t = float(net.res_ext_grid["p_mw"].sum())
            else:
                curtail_exhausted_t = False

            # [F] Build TimestepRecord from the final settled state.
            #     active_report and active_p_ser reflect post-curtailment if
            #     curtailment ran, or post-Q-control if it did not.
            if converged and active_report is not None:
                vm = net.res_bus["vm_pu"].copy()
                ll = net.res_line["loading_percent"].copy()
                tl = net.res_trafo["loading_percent"].copy()

                ov_buses  = vm.index[vm > v_max + VOLTAGE_EPSILON].tolist()
                uv_buses  = vm.index[vm < v_min - VOLTAGE_EPSILON].tolist()
                ov_lines  = ll.index[ll > LINE_MAX_LOADING  + LOADING_EPSILON].tolist()
                ov_trafos = tl.index[tl > TRAFO_MAX_LOADING + LOADING_EPSILON].tolist()
            else:
                vm = pd.Series(dtype=float)
                ll = pd.Series(dtype=float)
                tl = pd.Series(dtype=float)
                ov_buses = uv_buses = ov_lines = ov_trafos = []

            rec = TimestepRecord(
                t=t, timestamp=timestamp,
                vm_pu=vm, line_loading=ll, trafo_loading=tl,
                over_voltage_buses=ov_buses, under_voltage_buses=uv_buses,
                overloaded_lines=ov_lines, overloaded_trafos=ov_trafos,
                q_applied_mvar=q_ser,
                p_applied_mw=active_p_ser,
                curtailment_needed=result.curtailment_needed,
                curtail_exhausted=curtail_exhausted_t,
                converged=converged,
                p_target_mw=pd.Series(p_target_row,
                                      index=ctrl.sgen_indices, dtype=float),
                losses_mw=losses_mw_t,
                grid_import_mw=grid_import_mw_t,
                der_gen_mw=der_gen_mw_t,
                load_mw=load_mw_t,
                coordination_active=coordination_active_t,
                q_saturated_count=q_saturated_count_t,
                hil_latency_ms=result.t_exchange_ms,
                t_total_ms=result.t_total_ms,
            )
            if publish_fn is not None:
                publish_fn.on_timestep(rec)

            records.append(rec)

            if t % 96 == 0:
                if live_csv_rewrite_fn is not None:
                    partial = ScenarioResult.from_records(
                        scenario_id=scenario_id_for_resume, network_id=network_id,
                        records=records, elapsed_s=(publish_fn.cumulative_elapsed_s() 
                                                    if publish_fn is not None else time.perf_counter() - t_start),
                        dt_s=ap.dt_s,
                    )
                    live_csv_rewrite_fn(partial)
                q_so_far = sum(
                    float(np.abs(r.q_applied_mvar).sum())
                    for r in records
                    if r.q_applied_mvar is not None
                )

                post_ov_steps = sum(1 for r in records if r.over_voltage_buses)
                post_uv_steps = sum(1 for r in records if r.under_voltage_buses)
                post_ol_steps = sum(1 for r in records if r.overloaded_lines)
                post_ot_steps = sum(1 for r in records if r.overloaded_trafos)
                post_violation_steps = sum(
                    1 for r in records
                    if r.over_voltage_buses or r.under_voltage_buses
                    or r.overloaded_lines or r.overloaded_trafos
                )

                logger.info(
                    "[Scenario 4%s | %s] t=%d/%d (%.1f %%) | "
                    "pre_steps=%d | post_steps=%d "
                    "(ov=%d uv=%d ol=%d ot=%d) | "
                    "curtail_steps=%d | Q_cum=%.2f MVAr",
                    "-coord" if coordination else "-local",
                    network_id,
                    t,
                    _T_full,
                    100.0 * t / max(_T_full, 1),
                    pre_violation_steps,
                    post_violation_steps,
                    post_ov_steps,
                    post_uv_steps,
                    post_ol_steps,
                    post_ot_steps,
                    sum(1 for r in records if r.curtailment_needed),
                    q_so_far,
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

    elapsed = publish_fn.cumulative_elapsed_s() if publish_fn is not None else time.perf_counter() - t_start

    result = ScenarioResult.from_records(
        scenario_id = "volt_var_coord" if coordination else "volt_var_local",
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
    if publish_fn is not None:
        publish_fn.on_scenario_end(result)
    return result
