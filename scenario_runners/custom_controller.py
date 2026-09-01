"""
custom_controller.py
====================
Generic scenario runner for user-supplied Q(V) controller plugins.

Executes the standard two-runpp per-timestep HIL loop (structurally identical
to the Scenario 4 loop in scenario_4_volt_var.py) but delegates the reactive
power computation to an arbitrary caller-supplied callable instead of
run_coordinated_timestep().  This is the execution backbone of the plugin
system in plugin_runner.py — but it is also directly usable from Python
without any YAML involved.

Per-timestep sequence
---------------------
    [A] p_target selected from ap.der_p in ctrl.sgen_indices order
    [B] Load profiles written to net.load (index-explicit)
    [0] p_target written to net.sgen.p_mw
    [1] net.sgen.q_mvar reset to 0.0            (mandatory pre-PF invariant)
    [2] Pre-PF: runpp(net, voltage_depend_loads=False)  → report_pre
        (gate: optionally skip [3]–[5] on clean timesteps — see below)
    [3] controller_fn(vm_pu_at_ders, p_installed_mw)    → q_mvar
        clamped via VoltVarController._clamp_to_net_limits (optional backstop)
    [4] q_mvar written to net.sgen.q_mvar
    [5] Post-PF: runpp() → detect_violations()          → report_post
    [F] TimestepRecord built via make_record_from_report(), then the
        instrumentation and energy-balance fields are populated post-
        construction (t_total_ms, p_target_mw, losses_mw, grid_import_mw,
        der_gen_mw, load_mw).

Differences from Scenario 4 — read before comparing results
------------------------------------------------------------
- No SensitivityCoordinator (Item 3), no DERDynamics (Item 4 — PT1/ramp),
  and no active power curtailment loop (Item 5).  The plugin's Q output is
  applied directly (after the optional net-limit clamp).  Consequently
  curtailment_steps is 0 by construction and curtailment_needed is always
  recorded False — residual violations appear in n_violation_steps instead.
- coordination_active, q_saturated_count, and hil_latency_ms are None.
- By default the controller is called on EVERY timestep (two runpp calls
  per step).  Scenario 4's run_coordinated_timestep() gates on the pre-PF:
  clean timesteps hold Q=0 and skip the post-PF entirely.  Pass
  gate_clean_timesteps=True to reproduce that behaviour for a fair
  elapsed_s / reactive-energy comparison against 4A/4B.

controller_fn contract
----------------------
    def controller_fn(vm_pu: np.ndarray, p_mw: np.ndarray) -> np.ndarray

    Parameters
    ----------
    vm_pu : np.ndarray, shape (n_ders,), float
        Bus voltage magnitude in per-unit at each DER's connection bus,
        read from the PRE-control power flow (q_mvar = 0 state).
        Order follows ctrl.sgen_indices — the sorted-ascending sgen element
        indices of the profiled DERs (ap.der_p.columns).
    p_mw : np.ndarray, shape (n_ders,), float
        Installed capacity per DER in MW, in the SAME order as vm_pu.
        Resolved identically to the built-in scenarios: net.sgen.sn_mva
        where finite and positive, else net.sgen.p_mw
        (VoltVarController._resolve_p_installed).
        NOTE: this is the rated/installed capacity, not the instantaneous
        profile output at the current timestep.

    Returns
    -------
    np.ndarray, shape (n_ders,), float
        q_mvar setpoint per DER in the SAME order.  pandapower sign
        convention: q > 0 injects reactive power (capacitive, raises
        voltage); q < 0 absorbs (inductive, lowers voltage).
        Every element must be finite.  Shape or finiteness violations
        raise immediately with a message naming the plugin — they are
        programming errors in the plugin, not runtime conditions.

    Extra configuration parameters (droop slopes, deadbands, ...) are bound
    with functools.partial by plugin_runner.py before the callable reaches
    this module, so from here the signature is always exactly two arrays in,
    one array out.

Non-convergence handling
------------------------
Any runpp() exception (pre- or post-PF) is caught: the timestep is recorded
with converged=False (empty result Series, no violation lists) and the loop
continues to the next timestep.  A pre-PF failure additionally skips the
controller call — there is no valid voltage to feed it.

Usage
-----
    from custom_controller import run_custom_controller_scenario

    def my_droop(vm_pu, p_mw):
        return np.clip(-5.0 * (vm_pu - 1.0) * p_mw,
                       -0.25 * p_mw, 0.25 * p_mw)

    result = run_custom_controller_scenario(
        net, profiles,
        controller_fn = my_droop,
        scenario_id   = "my_droop",
        label         = "My droop controller",
        network_id    = "1-MV-rural--2-sw",
    )
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import numpy as np
import pandapower as pp
import pandas as pd

from scenario_result import (
    AdaptedProfiles,
    ScenarioResult,
    TimestepRecord,
    adapt_profiles,
    make_record_from_report,
)
from violation_detector import (
    V_MIN,
    V_MAX,
    detect_violations,
)
from volt_var_controller import VoltVarController

logger = logging.getLogger(__name__)


def _validate_controller_output(
        q_raw,
        n_ders:      int,
        scenario_id: str,
) -> np.ndarray:
    """
    Validate and coerce the plugin's return value.

    Raises TypeError / ValueError naming the plugin so a shape or NaN bug
    is diagnosed at the first offending timestep rather than surfacing as
    a silent index misalignment or an unexplained runpp() divergence.
    """
    try:
        q_arr = np.asarray(q_raw, dtype=float)
    except Exception as exc:
        raise TypeError(
            f"Custom controller '{scenario_id}': return value could not be "
            f"converted to a float ndarray ({type(q_raw).__name__}): {exc}"
        ) from exc

    if q_arr.ndim != 1:
        raise ValueError(
            f"Custom controller '{scenario_id}': expected a 1-D array of "
            f"q_mvar setpoints, got ndim={q_arr.ndim} (shape {q_arr.shape})."
        )
    if len(q_arr) != n_ders:
        raise ValueError(
            f"Custom controller '{scenario_id}': expected {n_ders} q_mvar "
            f"values (one per DER in sgen_indices order), got {len(q_arr)}."
        )
    if not np.all(np.isfinite(q_arr)):
        bad = np.flatnonzero(~np.isfinite(q_arr)).tolist()
        raise ValueError(
            f"Custom controller '{scenario_id}': non-finite q_mvar at "
            f"positions {bad[:10]}{' ...' if len(bad) > 10 else ''}. "
            "All setpoints must be finite floats."
        )
    return q_arr


def run_custom_controller_scenario(
        net,
        profiles:      dict,
        controller_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        scenario_id:   str,
        label:         str,
        network_id:    str   = "unknown",
        v_min:         float = V_MIN,
        v_max:         float = V_MAX,
        publish_fn            = None,
        gate_clean_timesteps: bool = False,
        clamp_to_net_limits:  bool = True,
        enable_checkpointing: bool = True,
        live_csv_rewrite_fn=None,
) -> ScenarioResult:
    """
    Run the generic two-runpp HIL loop with a user-supplied Q controller.

    Parameters
    ----------
    net           : pandapower network.  Modified in place every timestep.
                    Caller should deep-copy if the original is needed later
                    (benchmark_runner does this per scenario).
    profiles      : dict from profile_builder.build_profiles() /
                    build_annual_profiles().
    controller_fn : callable(vm_pu, p_mw) -> q_mvar.  See the module
                    docstring for the full contract.  Extra kwargs must
                    already be bound (functools.partial) before this call.
    scenario_id   : machine-readable identifier stored in ScenarioResult
                    and used by the publisher for scenarios/<id>.json.
                    Must not collide with the built-in scenario_ids.
    label         : human-readable display name (comparison table, publisher).
    network_id    : human-readable network identifier.
    v_min, v_max  : voltage planning limits (pu) for violation detection.
    publish_fn    : optional PublishHandle.  on_scenario_start(),
                    on_timestep(), and on_scenario_end() are called,
                    mirroring the built-in runners.
    gate_clean_timesteps : if True, mirror run_coordinated_timestep()'s
                    gate — when the pre-PF shows no violations, hold Q=0,
                    reuse the pre-PF state as the settled state, and skip
                    both the controller call and the post-PF.  Default
                    False: the controller is evaluated every timestep.
    clamp_to_net_limits : if True (default), pass the controller's output
                    through VoltVarController._clamp_to_net_limits() —
                    explicit min/max_q_mvar columns plus the apparent-power
                    cap |Q| <= sqrt(sn_mva² − p_mw²) — before writing to
                    net.sgen.  Safety backstop against physically
                    impossible plugin output.

    Returns
    -------
    ScenarioResult with the caller-supplied scenario_id.
    """
    t_start = time.perf_counter()

    ap: AdaptedProfiles = adapt_profiles(net, profiles)
    _T = len(ap.times)

    if ap.der_p.empty:
        raise ValueError(
            f"Custom controller scenario '{scenario_id}' requires at least "
            "one profiled DER in ap.der_p. Check that profiles['pv'] or "
            "profiles['wind'] are non-empty."
        )

    if publish_fn is not None:
        publish_fn.on_scenario_start(scenario_id, label, _T)

    # ------------------------------------------------------------------
    # DER metadata — reuse VoltVarController in dry-run mode so that
    # sgen ordering, installed-capacity resolution (sn_mva with p_mw
    # fallback), bus mapping, and the net-limit Q clamp are all EXACTLY
    # the semantics of the built-in scenarios.  No serial port is opened
    # and configure() is a no-op in dry-run mode.
    # ------------------------------------------------------------------
    ctrl = VoltVarController(
        net,
        interface    = None,
        sgen_indices = ap.der_p.columns,
        dry_run      = True,
    )
    ctrl.configure()

    sgen_idx   = ctrl.sgen_indices
    sgen_buses = ctrl._sgen_buses
    p_inst     = ctrl.p_installed_mw
    n_ders     = ctrl.n_ders

    logger.info(
        "[Custom '%s' | %s] Starting: %d timesteps, %d DERs "
        "(%d PV + %d wind), %d loads | gate_clean=%s | clamp=%s",
        scenario_id, network_id, _T, n_ders,
        len(ap.pv_idx), len(ap.wind_idx), len(ap.load_idx),
        gate_clean_timesteps, clamp_to_net_limits,
    )

    # Reset any controllers left from a previous scenario on the same net
    # object, mirroring scenario_4_volt_var.py.
    net.controller.drop(net.controller.index, inplace=True)
    pp.reset_results(net)

    runpp_kwargs = {"voltage_depend_loads": False}

    resumed_records: list[TimestepRecord] = []
    if publish_fn is not None and enable_checkpointing and hasattr(publish_fn, "get_resume_records"):
        resumed_records = publish_fn.get_resume_records(scenario_id)
    start_t = (resumed_records[-1].t + 1) if resumed_records else 0

    if start_t >= _T and resumed_records:
        logger.info(
            "[Custom '%s' | %s] Checkpoint already covers all %d steps — skipping simulation.",
            scenario_id, network_id, _T,
        )
        elapsed = publish_fn.cumulative_elapsed_s() if publish_fn is not None else time.perf_counter() - t_start
        result = ScenarioResult.from_records(
            scenario_id=scenario_id, network_id=network_id,
            records=resumed_records, elapsed_s=elapsed, dt_s=ap.dt_s,
        )
        if publish_fn is not None:
            publish_fn.on_scenario_end(result)
        return result

    if start_t > 0:
        logger.info(
            "[Custom '%s' | %s] Resuming from t=%d/%d.",
            scenario_id, network_id, start_t, _T,
        )

    records: list[TimestepRecord] = resumed_records.copy()

    for t in range(start_t, _T):

        t0 = time.perf_counter()
        timestamp = ap.times[t]

        # [A] p_target in sgen_indices order.
        p_target_row = (
            ap.der_p.iloc[t].reindex(sgen_idx).fillna(0.0)
            .values.astype(float)
        )

        # [B] Load profiles (index-explicit).
        if not ap.load_p.empty:
            net.load.loc[ap.load_idx, "p_mw"]   = ap.load_p.iloc[t].values
            net.load.loc[ap.load_idx, "q_mvar"] = ap.load_q.iloc[t].values

        # Profile-derived energy fields — populated regardless of convergence.
        der_gen_mw_t = float(ap.der_p.iloc[t].sum())  if not ap.der_p.empty  else 0.0
        load_mw_t    = float(ap.load_p.iloc[t].sum()) if not ap.load_p.empty else 0.0

        # [0] Write p_target, [1] reset Q — mandatory pre-PF invariant.
        net.sgen.loc[sgen_idx, "p_mw"]   = p_target_row
        net.sgen.loc[sgen_idx, "q_mvar"] = 0.0

        # [2] Pre-PF.
        pre_pf_ok = True
        try:
            pp.runpp(net, **runpp_kwargs)
        except Exception as exc:
            pre_pf_ok = False
            logger.warning(
                "[Custom '%s'] T=%d (%s): pre-PF runpp() raised: %s. "
                "Recording non-converged timestep.",
                scenario_id, t, timestamp, exc,
            )

        report_pre = detect_violations(net, v_min=v_min, v_max=v_max)

        if not pre_pf_ok or not report_pre.converged:
            rec = make_record_from_report(
                t, timestamp, net, converged=False,
                v_min=v_min, v_max=v_max,
            )
            rec.t_total_ms  = (time.perf_counter() - t0) * 1e3
            rec.p_target_mw = pd.Series(p_target_row, index=sgen_idx, dtype=float)
            rec.der_gen_mw  = der_gen_mw_t
            rec.load_mw     = load_mw_t
            if publish_fn is not None:
                publish_fn.on_timestep(rec)
            records.append(rec)
            continue

        # ── Optional clean-timestep gate (mirrors run_coordinated_timestep) ──
        if gate_clean_timesteps and not report_pre.any_violations:
            q_arr = np.zeros(n_ders, dtype=float)
            # Pre-PF state IS the settled state: q=0 already written in [1],
            # p_target already written in [0].  No post-PF needed.
            rec = make_record_from_report(
                t, timestamp, net, converged=True,
                v_min=v_min, v_max=v_max,
                q_applied=pd.Series(q_arr,        index=sgen_idx, dtype=float),
                p_applied=pd.Series(p_target_row, index=sgen_idx, dtype=float),
                curtailment_needed=False,
            )
            rec.losses_mw      = float(net.res_line["pl_mw"].sum()
                                       + net.res_trafo["pl_mw"].sum())
            rec.grid_import_mw = float(net.res_ext_grid["p_mw"].sum())
            rec.t_total_ms     = (time.perf_counter() - t0) * 1e3
            rec.p_target_mw    = pd.Series(p_target_row, index=sgen_idx, dtype=float)
            rec.der_gen_mw     = der_gen_mw_t
            rec.load_mw        = load_mw_t
            if publish_fn is not None:
                publish_fn.on_timestep(rec)
            records.append(rec)
            if live_csv_rewrite_fn is not None and t % 96 == 0:
                partial = ScenarioResult.from_records(
                    scenario_id=scenario_id, network_id=network_id,
                    records=records, elapsed_s=(publish_fn.cumulative_elapsed_s() if publish_fn is not None else time.perf_counter() - t_start),
                    dt_s=ap.dt_s,
                )
                live_csv_rewrite_fn(partial)
            _periodic_log(scenario_id, network_id, t, _T, records)
            continue
        # ── END GATE ─────────────────────────────────────────────────────────

        # [3] Controller call on pre-PF voltages at the DER buses.
        vm_pu_at_ders = net.res_bus.loc[sgen_buses, "vm_pu"].values.astype(float)
        q_arr = _validate_controller_output(
            controller_fn(vm_pu_at_ders, p_inst),
            n_ders, scenario_id,
        )

        if clamp_to_net_limits:
            q_arr = ctrl._clamp_to_net_limits(q_arr)

        # [4] Apply Q.  p_mw unchanged from [0] — no dynamics/ramp layer.
        net.sgen.loc[sgen_idx, "q_mvar"] = q_arr

        # [5] Post-PF.
        post_pf_ok = True
        try:
            pp.runpp(net, **runpp_kwargs)
        except Exception as exc:
            post_pf_ok = False
            logger.warning(
                "[Custom '%s'] T=%d (%s): post-PF runpp() raised: %s. "
                "Recording non-converged timestep.",
                scenario_id, t, timestamp, exc,
            )

        report_post = detect_violations(net, v_min=v_min, v_max=v_max)
        converged   = post_pf_ok and report_post.converged

        # [F] Build the record.  curtailment_needed is recorded False by
        #     design — there is no curtailment stage in the custom path, so
        #     residual violations are counted in n_violation_steps only.
        if converged:
            rec = make_record_from_report(
                t, timestamp, net, converged=True,
                v_min=v_min, v_max=v_max,
                q_applied=pd.Series(q_arr,        index=sgen_idx, dtype=float),
                p_applied=pd.Series(p_target_row, index=sgen_idx, dtype=float),
                curtailment_needed=False,
            )
            rec.losses_mw      = float(net.res_line["pl_mw"].sum()
                                       + net.res_trafo["pl_mw"].sum())
            rec.grid_import_mw = float(net.res_ext_grid["p_mw"].sum())
        else:
            rec = make_record_from_report(
                t, timestamp, net, converged=False,
                v_min=v_min, v_max=v_max,
            )

        rec.t_total_ms  = (time.perf_counter() - t0) * 1e3
        rec.p_target_mw = pd.Series(p_target_row, index=sgen_idx, dtype=float)
        rec.der_gen_mw  = der_gen_mw_t
        rec.load_mw     = load_mw_t

        if publish_fn is not None:
            publish_fn.on_timestep(rec)
        records.append(rec)
        if live_csv_rewrite_fn is not None and t % 96 == 0:
            partial = ScenarioResult.from_records(
                scenario_id=scenario_id, network_id=network_id,
                records=records, elapsed_s=(publish_fn.cumulative_elapsed_s() if publish_fn is not None else time.perf_counter() - t_start),
                dt_s=ap.dt_s,
            )
            live_csv_rewrite_fn(partial)

        _periodic_log(scenario_id, network_id, t, _T, records)

    elapsed = publish_fn.cumulative_elapsed_s() if publish_fn is not None else time.perf_counter() - t_start

    result = ScenarioResult.from_records(
        scenario_id = scenario_id,
        network_id  = network_id,
        records     = records,
        elapsed_s   = elapsed,
        dt_s        = ap.dt_s,
    )

    logger.info(
        "[Custom '%s' | %s] Done. %.1f s | %d/%d converged | "
        "%d violation steps | Q_total=%.2f MVAr",
        scenario_id, network_id, elapsed,
        result.n_converged, result.n_timesteps, result.n_violation_steps,
        result.q_total_mvar_abs if result.q_total_mvar_abs is not None else 0.0,
    )

    if publish_fn is not None:
        publish_fn.on_scenario_end(result)

    return result


def _periodic_log(
        scenario_id: str,
        network_id:  str,
        t:           int,
        T:           int,
        records:     list,
) -> None:
    """Daily progress line at 15-min resolution (every 96 timesteps)."""
    if t % 96 != 0:
        return
    q_so_far = sum(
        float(np.abs(r.q_applied_mvar).sum())
        for r in records if r.q_applied_mvar is not None
    )
    violation_steps = sum(
        1 for r in records
        if r.over_voltage_buses or r.under_voltage_buses
        or r.overloaded_lines or r.overloaded_trafos
    )
    logger.info(
        "[Custom '%s' | %s] t=%d/%d (%.1f %%) | violation_steps=%d | "
        "Q_cum=%.2f MVAr",
        scenario_id, network_id, t, T,
        100.0 * t / max(T, 1), violation_steps, q_so_far,
    )
