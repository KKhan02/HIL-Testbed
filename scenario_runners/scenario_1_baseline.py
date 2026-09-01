"""
scenario_1_baseline.py
======================
Scenario 1 — Baseline (no control).

Runs a full annual time series using pandapower's built-in run_timeseries()
engine with ConstControl data sources.  No reactive power control, no Arduino,
no manual timestep loop.  This is the reference against which all controlled
scenarios are compared.

Violation detection runs in post-processing from the OutputWriter logs, not
inside the loop.  This is correct for Scenario 1 — the no-control reference
never needs per-timestep hooks.

Usage
-----
    from profile_builder import build_profiles
    from scenario_result import adapt_profiles
    from scenario_1_baseline import run_scenario_1

    net      = sb.get_simbench_net("1-MV-rural--2-sw")
    profiles = build_profiles(net, ...)
    result   = run_scenario_1(net, profiles)

    print(result.n_violation_steps)
    df = pd.DataFrame([result.summary_dict()])

Notes
-----
voltage_depend_loads=False is passed as a direct **kwarg to run_timeseries(),
which forwards it to the internal runpp call.  This is mandatory for SimBench
networks on pandapower 3.2.0+ (singular matrix without it).

ConstControl columns must be integer sgen/load indices, not positional
integers.  This is enforced by the DataFrame column construction below.

Violation thresholds are imported from violation_detector.py (V_MIN, V_MAX,
LINE_MAX_LOADING, TRAFO_MAX_LOADING, VOLTAGE_EPSILON, LOADING_EPSILON) so
post-processing is always consistent with the live detection layer.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandapower as pp
import pandapower.control as ctrl
import pandapower.timeseries as ts
from pandapower.timeseries import DFData, OutputWriter
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
    LINE_MAX_LOADING,
    TRAFO_MAX_LOADING,
    VOLTAGE_EPSILON,
    LOADING_EPSILON,
)

logger = logging.getLogger(__name__)

# Per-timestep wall-clock accumulator for the run= hook in run_timeseries().
# Cleared at the start of each call; populated by _timed_runpp; read back
# during the post-processing loop below.
_RUNPP_TIMING: dict = {}

def _timed_runpp(net, **kwargs):
    """Wraps pp.runpp() to record per-timestep elapsed time, keyed by the
    OutputWriter's current time_step. Accumulates if called more than once
    within the same timestep."""
    t0 = time.perf_counter()
    pp.runpp(net, **kwargs)
    dt_ms = (time.perf_counter() - t0) * 1e3
    ts_step = net["output_writer"].iat[0, 0].time_step
    _RUNPP_TIMING[ts_step] = _RUNPP_TIMING.get(ts_step, 0.0) + dt_ms

def run_scenario_1(
        net,
        profiles:   dict,
        network_id: str  = "unknown",
        v_min:      float = V_MIN,
        v_max:      float = V_MAX,
        publish_fn          = None,
        enable_checkpointing: bool = True,   # accepted, not implemented — see note below
        live_csv_rewrite_fn         = None,  # accepted, not implemented — see note below

) -> ScenarioResult:
    """
    Run Scenario 1 — Baseline (no voltage control).

    Note: enable_checkpointing and live_csv_rewrite_fn are accepted for
    signature compatibility with benchmark_runner's unconditional kwargs
    injection, but are NOT implemented here. run_timeseries() is a single
    opaque call covering the full year with no mid-run hook available, so
    neither crash-resume nor a live progress CSV is possible for this
    scenario without restructuring it away from run_timeseries() entirely.

    Parameters
    ----------
    net        : pandapower network.  Modified in place by run_timeseries;
                 caller should deep-copy if the original net is needed later.
    profiles   : dict from profile_builder.build_profiles().

    Returns
    -------
    ScenarioResult  with scenario_id="baseline".
    """
    t_start = time.perf_counter()

    ap: AdaptedProfiles = adapt_profiles(net, profiles)
    _T = len(ap.times)
    if publish_fn is not None:
        publish_fn.on_scenario_start("baseline", "Baseline (no control)", _T)
    time_steps = list(range(len(ap.times)))

    def _integer_indexed(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.index = time_steps
        return out

    der_p_ts = _integer_indexed(ap.der_p)
    load_p_ts = _integer_indexed(ap.load_p)
    load_q_ts = _integer_indexed(ap.load_q)

    for name, df in {
        "load_p": load_p_ts,
        "load_q": load_q_ts,
        "der_p": der_p_ts,
    }.items():
        missing = pd.Index(time_steps).difference(df.index)
        if len(missing):
            raise ValueError(
                f"{name} is missing {len(missing)} requested integer time steps. "
                f"First missing: {missing[0]}"
            )

    logger.info(
        "[Scenario 1 | %s] Starting baseline run: %d timesteps, "
        "%d DERs (%d PV + %d wind), %d loads",
        network_id, len(time_steps),
        len(ap.der_p.columns), len(ap.pv_idx), len(ap.wind_idx),
        len(ap.load_idx),
    )

    # ------------------------------------------------------------------
    # [1] Reset any existing controllers and result tables
    # ------------------------------------------------------------------
    net.controller.drop(net.controller.index, inplace=True)
    pp.reset_results(net)

    # ------------------------------------------------------------------
    # [2] Attach ConstControl for DER active power
    #     One ConstControl per element type (sgen, load) is sufficient;
    #     pandapower maps columns to element indices via element_index.
    # ------------------------------------------------------------------
    if not ap.der_p.empty:
        der_ds = DFData(der_p_ts)
        ctrl.ConstControl(
            net,
            element         = "sgen",
            element_index   = ap.der_p.columns.tolist(),
            variable        = "p_mw",
            data_source     = der_ds,
            profile_name    = ap.der_p.columns.tolist(),
        )

    if not ap.load_p.empty:
        load_p_ds = DFData(load_p_ts)
        ctrl.ConstControl(
            net,
            element         = "load",
            element_index   = ap.load_p.columns.tolist(),
            variable        = "p_mw",
            data_source     = load_p_ds,
            profile_name    = ap.load_p.columns.tolist(),
        )

        load_q_ds = DFData(load_q_ts)
        ctrl.ConstControl(
            net,
            element         = "load",
            element_index   = ap.load_q.columns.tolist(),
            variable        = "q_mvar",
            data_source     = load_q_ds,
            profile_name    = ap.load_q.columns.tolist(),
        )

    # ------------------------------------------------------------------
    # [3] OutputWriter — log vm_pu, line loading, trafo loading
    #     These are the only tables needed for post-processing.
    # ------------------------------------------------------------------
    ow = OutputWriter(net, time_steps, output_path=None, output_file_type=None, log_variables=list())
    ow.log_variable("res_bus",      "vm_pu")
    ow.log_variable("res_line",     "loading_percent")
    ow.log_variable("res_trafo",    "loading_percent")
    ow.log_variable("res_line",     "pl_mw")
    ow.log_variable("res_trafo",    "pl_mw")
    ow.log_variable("res_ext_grid", "p_mw")

    # ------------------------------------------------------------------
    # [4] Run time series
    #     voltage_depend_loads=False is mandatory for SimBench networks.
    #     Passed as a direct **kwarg — run_timeseries forwards **kwargs to
    #     the internal runpp call (confirmed from pandapower 3.4.0 source).
    #     continue_on_divergence=True prevents a single non-converging
    #     timestep from aborting the full annual run.
    # ------------------------------------------------------------------
    logger.info("[Scenario 1 | %s] Calling run_timeseries ...", network_id)
    ts.run_timeseries(
        net,
        time_steps             = time_steps,
        continue_on_divergence = True,
        verbose                = False,
        voltage_depend_loads   = False,
        run                    = _timed_runpp,
    )
    logger.info("[Scenario 1 | %s] run_timeseries complete.", network_id)

    # ------------------------------------------------------------------
    # [5] Post-process OutputWriter results into TimestepRecord list
    #     ow.output["res_bus.vm_pu"] is a DataFrame (T × N_buses).
    #     ow.output["res_line.loading_percent"] is (T × N_lines).
    #     Row index = integer timestep (matching time_steps range).
    # ------------------------------------------------------------------
    vm_log   = ow.output.get("res_bus.vm_pu",             pd.DataFrame())
    ll_log   = ow.output.get("res_line.loading_percent",  pd.DataFrame())
    tl_log   = ow.output.get("res_trafo.loading_percent", pd.DataFrame())
    pl_line_log  = ow.output.get("res_line.pl_mw",    pd.DataFrame())
    pl_trafo_log = ow.output.get("res_trafo.pl_mw",   pd.DataFrame())
    p_grid_log   = ow.output.get("res_ext_grid.p_mw", pd.DataFrame())

    records = []
    for i, t in enumerate(time_steps):
        timestamp = ap.times[i]
        converged = (
            not vm_log.empty
            and t in vm_log.index
            and not vm_log.loc[t].isna().all()
        )

        if not converged:
            der_gen_mw_t = float(ap.der_p.iloc[i].sum()) if not ap.der_p.empty else 0.0
            load_mw_t    = float(ap.load_p.iloc[i].sum()) if not ap.load_p.empty else 0.0
            records.append(TimestepRecord(
                t=i,
                timestamp=timestamp,
                vm_pu=pd.Series(dtype=float),
                line_loading=pd.Series(dtype=float),
                trafo_loading=pd.Series(dtype=float),
                over_voltage_buses=[],
                under_voltage_buses=[],
                overloaded_lines=[],
                overloaded_trafos=[],
                q_applied_mvar=None,
                p_applied_mw=None,
                p_target_mw=None,
                curtailment_needed=False,
                converged=False,
                losses_mw=None,
                grid_import_mw=None,
                der_gen_mw=der_gen_mw_t,
                load_mw=load_mw_t,
                t_total_ms=_RUNPP_TIMING.get(t, None),
            ))
            continue

        vm = vm_log.loc[t].dropna()
        ll = (
            ll_log.loc[t].dropna()
            if not ll_log.empty and t in ll_log.index
            else pd.Series(dtype=float)
        )
        tl = (
            tl_log.loc[t].dropna()
            if not tl_log.empty and t in tl_log.index
            else pd.Series(dtype=float)
        )

        ov_buses = vm.index[vm > v_max + VOLTAGE_EPSILON].tolist()
        uv_buses = vm.index[vm < v_min - VOLTAGE_EPSILON].tolist()
        ov_lines = ll.index[ll > LINE_MAX_LOADING + LOADING_EPSILON].tolist()
        ov_trafos = tl.index[tl > TRAFO_MAX_LOADING + LOADING_EPSILON].tolist()

        # Energy balance for this converged timestep
        losses_mw_t = (
            float(pl_line_log.loc[t].sum() + pl_trafo_log.loc[t].sum())
            if (not pl_line_log.empty and t in pl_line_log.index
                and not pl_trafo_log.empty and t in pl_trafo_log.index)
            else None
        )
        grid_import_mw_t = (
            float(p_grid_log.loc[t].sum())
            if not p_grid_log.empty and t in p_grid_log.index
            else None
        )
        der_gen_mw_t = float(ap.der_p.iloc[i].sum())  if not ap.der_p.empty  else 0.0
        load_mw_t    = float(ap.load_p.iloc[i].sum()) if not ap.load_p.empty else 0.0
        
        rec = TimestepRecord(
            t=i,
            timestamp=timestamp,
            vm_pu=vm,
            line_loading=ll,
            trafo_loading=tl,
            over_voltage_buses=ov_buses,
            under_voltage_buses=uv_buses,
            overloaded_lines=ov_lines,
            overloaded_trafos=ov_trafos,
            q_applied_mvar=None,
            p_applied_mw=None,
            p_target_mw=None,
            curtailment_needed=False,
            converged=True,
            losses_mw=losses_mw_t,
            grid_import_mw=grid_import_mw_t,
            der_gen_mw=der_gen_mw_t,
            load_mw=load_mw_t,
            t_total_ms=_RUNPP_TIMING.get(t, None),
        )
        
        if publish_fn is not None:
            publish_fn.on_timestep(rec)
        records.append(rec)

    elapsed = time.perf_counter() - t_start

    result = ScenarioResult.from_records(
        scenario_id = "baseline",
        network_id  = network_id,
        records     = records,
        elapsed_s   = elapsed,
        dt_s        = ap.dt_s,
    )

    logger.info(
        "[Scenario 1 | %s] Done. %.1f s | %d/%d converged | %d violation steps",
        network_id, elapsed, result.n_converged, result.n_timesteps,
        result.n_violation_steps,
    )
    if publish_fn is not None:
        publish_fn.on_scenario_end(result)
    return result
