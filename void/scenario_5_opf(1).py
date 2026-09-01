"""
scenario_5_opf.py
=================
Scenario 5 — AC OPF benchmark attempt.

This runner attempts one pandapower AC OPF per timestep. It is intended as an
"ideal controller" reference, not as part of the hardware loop.

Important interpretation
------------------------
pandapower.runopp() uses the PYPOWER OPF backend. This backend is useful, but
it is known to have weak convergence behaviour on some networks. Therefore:

    OPF converged      -> use the returned dispatch as the Scenario 5 benchmark
    OPF not converged  -> record the timestep as non-converged

A non-converged timestep is not automatically physical infeasibility. It means
that this PYPOWER OPF formulation did not return a valid solution.

Optimisation target
-------------------
For every timestep, only DERs with available active power are marked
controllable and assigned a negative linear cost:

    cp1_eur_per_mw = -1.0  -> objective contains -sum(P_DER)

So, if the OPF converges, it maximises accepted DER active power subject to the
network constraints and DER capability limits. Curtailment is then:

    p_target_mw - p_applied_mw

where p_target_mw is the available DER profile at that timestep.

Main safeguards
---------------
- Clears stale controllers and cost rows.
- Makes ext_grid.controllable an explicit boolean.
- Uses feeder-scale ext_grid bounds, not placeholder +/-1e6 values.
- Rebuilds DER costs per timestep, only for active flexible DERs.
- Resets inactive DERs to p=q=0 before every OPF call.
- Keeps initial sgen p/q values inside their OPF bounds.
- Computes Q limits from both VDE Q_RATIO and the inverter apparent-power circle.
- Allows thermal constraints to be enabled, relaxed, or disabled for debugging.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.optimal_powerflow
import cyipopt

from pandapower.toolbox import create_continuous_bus_index
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
from volt_var_controller import Q_RATIO

logger = logging.getLogger(__name__)

_EXT_GRID_COST = 0.001
_DER_COST = -1.0
_ACTIVE_POWER_EPS = 1e-9
_DISABLED_THERMAL_LIMIT = 1e6


def _clear_poly_cost(net) -> None:
    """Remove stale OPF costs from previous runs."""
    if hasattr(net, "poly_cost") and not net.poly_cost.empty:
        n_old = len(net.poly_cost)
        net.poly_cost.drop(net.poly_cost.index, inplace=True)
        logger.debug("_clear_poly_cost: removed %d stale poly_cost rows.", n_old)


def _clear_controllers(net) -> None:
    """Remove time-series controllers that may have been attached by Scenario 1."""
    if hasattr(net, "controller") and not net.controller.empty:
        n_old = len(net.controller)
        net.controller.drop(net.controller.index, inplace=True)
        logger.debug("_clear_controllers: removed %d stale controller rows.", n_old)


def _series_peak_sum(df: pd.DataFrame) -> float:
    """Return max timestep sum of absolute values, or 0 for an empty frame."""
    if df is None or df.empty:
        return 0.0
    return float(df.abs().sum(axis=1).max())


def _prepare_ext_grid_for_opf(net, ap: AdaptedProfiles) -> None:
    """
    Prepare ext_grid as a controllable slack source with feeder-scale bounds.

    SimBench networks may leave ext_grid.controllable as NaN. PYPOWER OPF uses
    boolean operations on this column, so it must be a real bool column.
    """
    if net.ext_grid.empty:
        raise ValueError("Scenario 5 requires at least one ext_grid/slack source.")

    net.ext_grid["controllable"] = True
    net.ext_grid["controllable"] = net.ext_grid["controllable"].astype(bool)

    for col in ["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]:
        if col not in net.ext_grid.columns:
            net.ext_grid[col] = np.nan

    p_load_peak = max(
        _series_peak_sum(ap.load_p),
        float(net.load["p_mw"].abs().sum()) if not net.load.empty else 0.0,
    )
    q_load_peak = max(
        _series_peak_sum(ap.load_q),
        float(net.load["q_mvar"].abs().sum()) if not net.load.empty else 0.0,
    )
    p_der_peak = max(
        _series_peak_sum(ap.der_p),
        float(net.sgen["p_mw"].abs().sum()) if not net.sgen.empty else 0.0,
    )

    if "sn_mva" in net.sgen.columns and not net.sgen.empty:
        q_der_proxy = float(net.sgen["sn_mva"].fillna(0.0).abs().sum())
    else:
        q_der_proxy = p_der_peak

    p_margin = max(50.0, 3.0 * (p_load_peak + p_der_peak))
    q_margin = max(50.0, 3.0 * (q_load_peak + q_der_proxy))

    net.ext_grid.loc[:, "min_p_mw"] = -p_margin
    net.ext_grid.loc[:, "max_p_mw"] = p_margin
    net.ext_grid.loc[:, "min_q_mvar"] = -q_margin
    net.ext_grid.loc[:, "max_q_mvar"] = q_margin


def _apply_network_constraints(
    net,
    v_min: float,
    v_max: float,
    line_limit_percent: Optional[float],
    trafo_limit_percent: Optional[float],
) -> None:
    """Write OPF voltage and thermal constraints to the pandapower tables."""
    net.bus["min_vm_pu"] = float(v_min)
    net.bus["max_vm_pu"] = float(v_max)

    line_limit = (
        _DISABLED_THERMAL_LIMIT
        if line_limit_percent is None
        else float(line_limit_percent)
    )
    trafo_limit = (
        _DISABLED_THERMAL_LIMIT
        if trafo_limit_percent is None
        else float(trafo_limit_percent)
    )

    net.line["max_loading_percent"] = line_limit
    net.trafo["max_loading_percent"] = trafo_limit


def _setup_opf(
    net,
    ap: AdaptedProfiles,
    v_min: float,
    v_max: float,
    line_limit_percent: Optional[float],
    trafo_limit_percent: Optional[float],
) -> None:
    """
    Prepare static OPF table columns.

    DER active/reactive bounds and DER costs are rebuilt inside the timestep
    loop, because available DER power changes with the profile.
    """
    der_idx = ap.der_p.columns

    _clear_poly_cost(net)

    net.sgen["controllable"] = False
    net.sgen.loc[der_idx, "controllable"] = False

    for col in ["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]:
        if col not in net.sgen.columns:
            net.sgen[col] = np.nan

    net.sgen.loc[der_idx, "min_p_mw"] = 0.0
    net.sgen.loc[der_idx, "max_p_mw"] = 0.0
    net.sgen.loc[der_idx, "min_q_mvar"] = 0.0
    net.sgen.loc[der_idx, "max_q_mvar"] = 0.0
    net.sgen.loc[der_idx, "p_mw"] = 0.0
    net.sgen.loc[der_idx, "q_mvar"] = 0.0

    _prepare_ext_grid_for_opf(net, ap)
    _apply_network_constraints(
        net,
        v_min=v_min,
        v_max=v_max,
        line_limit_percent=line_limit_percent,
        trafo_limit_percent=trafo_limit_percent,
    )
    pp.diagnostic(net, warnings_only=False)
    logger.info(
        "_setup_opf: prepared %d profiled DERs. DER flexibility and costs "
        "will be rebuilt per timestep. line_limit=%s trafo_limit=%s",
        len(der_idx),
        "disabled" if line_limit_percent is None else f"{line_limit_percent:.1f}%",
        "disabled" if trafo_limit_percent is None else f"{trafo_limit_percent:.1f}%",
    )


def _compute_sn_rated(net, ap: AdaptedProfiles, der_idx: pd.Index) -> pd.Series:
    """
    Determine apparent-power rating for each profiled DER.

    Priority:
    1. net.sgen.sn_mva
    2. absolute initial net.sgen.p_mw
    3. peak available DER profile
    """
    sn_vals = net.sgen.loc[der_idx, "sn_mva"].astype(float)
    p_fb = net.sgen.loc[der_idx, "p_mw"].abs().astype(float)
    p_peak = ap.der_p.reindex(columns=der_idx).max(axis=0).fillna(0.0).astype(float)

    sn_rated = sn_vals.where(sn_vals.notna() & (sn_vals > 0.0), p_fb)
    sn_rated = sn_rated.where(sn_rated.notna() & (sn_rated > 0.0), p_peak)

    bad = sn_rated[sn_rated <= 0.0]
    if not bad.empty:
        raise ValueError(
            "Scenario 5 cannot infer positive sn_rated for sgen indices "
            f"{bad.index.tolist()}. Set net.sgen.sn_mva or provide nonzero profiles."
        )

    return sn_rated


def _write_timestep_opf_state(
    net,
    ap: AdaptedProfiles,
    der_idx: pd.Index,
    sn_rated: pd.Series,
    t: int,
) -> tuple[pd.Series, pd.Index]:
    """
    Write timestep-specific DER P/Q bounds, operating point, loads, and costs.

    Returns
    -------
    p_bound_ser:
        Available DER active power at this timestep.
    active_der_idx:
        DERs with p_bound > eps and therefore actual flexibility/cost.
    """
    profile_row = ap.der_p.iloc[t].reindex(der_idx).fillna(0.0).astype(float)
    p_bound_ser = profile_row.clip(lower=0.0)

    active_der_idx = p_bound_ser.index[p_bound_ser > _ACTIVE_POWER_EPS]

    q_vde = Q_RATIO * sn_rated
    q_circle = np.sqrt(np.maximum(0.0, sn_rated.pow(2) - p_bound_ser.pow(2)))
    q_lim_ser = pd.Series(
        np.minimum(q_vde.values, q_circle),
        index=der_idx,
        dtype=float,
    )
    q_lim_ser.loc[p_bound_ser <= _ACTIVE_POWER_EPS] = 0.0

    # Reset all profiled DERs first. This prevents inactive DERs from carrying
    # stale p_mw/q_mvar values into the OPF.
    net.sgen.loc[der_idx, "controllable"] = False
    net.sgen.loc[der_idx, "min_p_mw"] = 0.0
    net.sgen.loc[der_idx, "max_p_mw"] = 0.0
    net.sgen.loc[der_idx, "p_mw"] = 0.0
    net.sgen.loc[der_idx, "min_q_mvar"] = 0.0
    net.sgen.loc[der_idx, "max_q_mvar"] = 0.0
    net.sgen.loc[der_idx, "q_mvar"] = 0.0

    if len(active_der_idx) > 0:
        net.sgen.loc[active_der_idx, "controllable"] = True
        net.sgen.loc[active_der_idx, "max_p_mw"] = p_bound_ser.loc[active_der_idx].values
        net.sgen.loc[active_der_idx, "p_mw"] = p_bound_ser.loc[active_der_idx].values
        net.sgen.loc[active_der_idx, "min_q_mvar"] = -q_lim_ser.loc[active_der_idx].values
        net.sgen.loc[active_der_idx, "max_q_mvar"] = q_lim_ser.loc[active_der_idx].values

    if not ap.load_p.empty:
        net.load.loc[ap.load_idx, "p_mw"] = ap.load_p.iloc[t].values
        net.load.loc[ap.load_idx, "q_mvar"] = ap.load_q.iloc[t].values

    # Rebuild costs so that inactive zero-profile DERs do not receive stale
    # curtailment-minimising costs.
    _clear_poly_cost(net)
    for eg_idx in net.ext_grid.index:
        pp.create_poly_cost(net, eg_idx, "ext_grid", cp1_eur_per_mw=_EXT_GRID_COST)
    for idx in active_der_idx:
        pp.create_poly_cost(net, idx, "sgen", cp1_eur_per_mw=_DER_COST)

    return p_bound_ser, active_der_idx


def _print_opf_debug(net, der_idx: pd.Index) -> None:
    """Print one-step OPF diagnostics. Intended only for short debug runs."""
    print("\n=== OPF task ===")
    print(pp.opf_task(net))

    print("\n=== ext_grid OPF fields ===")
    print(
        net.ext_grid[[
            "controllable",
            "min_p_mw", "max_p_mw",
            "min_q_mvar", "max_q_mvar",
        ]].to_string()
    )
    print(net.ext_grid[["controllable"]].dtypes)

    debug_sgen = net.sgen.loc[der_idx, [
        "p_mw", "q_mvar",
        "min_p_mw", "max_p_mw",
        "min_q_mvar", "max_q_mvar",
        "controllable",
    ]]
    print("\n=== sgen OPF bounds ===")
    print(debug_sgen.to_string())

    bad_bounds_p = debug_sgen[
        (debug_sgen["max_p_mw"] < debug_sgen["min_p_mw"])
        | debug_sgen["max_p_mw"].isna()
        | debug_sgen["min_p_mw"].isna()
    ]
    bad_bounds_q = debug_sgen[
        (debug_sgen["max_q_mvar"] < debug_sgen["min_q_mvar"])
        | debug_sgen["max_q_mvar"].isna()
        | debug_sgen["min_q_mvar"].isna()
    ]
    bad_initial_p = debug_sgen[
        (debug_sgen["p_mw"] < debug_sgen["min_p_mw"] - 1e-9)
        | (debug_sgen["p_mw"] > debug_sgen["max_p_mw"] + 1e-9)
    ]
    bad_initial_q = debug_sgen[
        (debug_sgen["q_mvar"] < debug_sgen["min_q_mvar"] - 1e-9)
        | (debug_sgen["q_mvar"] > debug_sgen["max_q_mvar"] + 1e-9)
    ]

    print("\nBad P bounds:")
    print(bad_bounds_p.to_string())
    print("\nBad Q bounds:")
    print(bad_bounds_q.to_string())
    print("\nBad initial P values:")
    print(bad_initial_p.to_string())
    print("\nBad initial Q values:")
    print(bad_initial_q.to_string())


def run_scenario_5(
    net,
    profiles: dict,
    network_id: str = "unknown",
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    verbose_opf: bool = False,
    opf_init: str = "flat",
    line_limit_percent: Optional[float] = LINE_MAX_LOADING,
    trafo_limit_percent: Optional[float] = TRAFO_MAX_LOADING,
    debug_opf_task: bool = False,
    debug_first_only: bool = True,
    max_warning_timesteps: int = 3,
) -> ScenarioResult:
    """
    Run Scenario 5 — AC OPF benchmark attempt.

    Parameters
    ----------
    net:
        pandapower network. Modified in place.
    profiles:
        Profile dictionary accepted by scenario_result.adapt_profiles().
    network_id:
        Human-readable network identifier.
    v_min, v_max:
        OPF voltage constraints.
    verbose_opf:
        Forwarded to pp.runopp().
    opf_init:
        "flat" or "pf".
    line_limit_percent, trafo_limit_percent:
        Thermal constraints. Use None to disable that class of thermal
        constraint during diagnostics.
    debug_opf_task:
        If True, prints pp.opf_task() and bound checks before runopp().
    debug_first_only:
        If True, prints diagnostics only for the first timestep.
    max_warning_timesteps:
        Number of individual non-convergence warnings before suppressing
        repeated messages.
    """
    if opf_init not in {"flat", "pf"}:
        raise ValueError("opf_init must be either 'flat' or 'pf'.")

    t_start = time.perf_counter()

    create_continuous_bus_index(net, start=0)

    ap: AdaptedProfiles = adapt_profiles(net, profiles)
    if ap.der_p.empty:
        raise ValueError(
            "Scenario 5 requires at least one profiled DER in ap.der_p. "
            "Check profiles['pv'] and profiles['wind']."
        )

    time_steps = range(len(ap.times))
    der_idx = pd.Index(ap.der_p.columns)

    logger.info(
        "[Scenario 5 | %s] Starting OPF benchmark attempt: %d timesteps, "
        "%d DERs (%d PV + %d wind), %d loads",
        network_id,
        len(time_steps),
        len(der_idx),
        len(ap.pv_idx),
        len(ap.wind_idx),
        len(ap.load_idx),
    )

    _clear_controllers(net)
    pp.reset_results(net)

    _setup_opf(
        net,
        ap,
        v_min=v_min,
        v_max=v_max,
        line_limit_percent=line_limit_percent,
        trafo_limit_percent=trafo_limit_percent,
    )

    pp.set_user_pf_options(net, voltage_depend_loads=False)

    sn_rated = _compute_sn_rated(net, ap, der_idx)
    logger.debug(
        "[Scenario 5 | %s] sn_rated: min=%.4f max=%.4f MVA",
        network_id,
        float(sn_rated.min()),
        float(sn_rated.max()),
    )

    records: list[TimestepRecord] = []
    suppressed_warning_logged = False

    for t in time_steps:
        timestamp = ap.times[t]

        p_bound_ser, active_der_idx = _write_timestep_opf_state(
            net=net,
            ap=ap,
            der_idx=der_idx,
            sn_rated=sn_rated,
            t=t,
        )

        if debug_opf_task and (not debug_first_only or t == 0):
            _print_opf_debug(net, der_idx)

        try:
            pp.runopp(
                net,
                verbose=verbose_opf,
                init=opf_init,
                OPF_SOLVER="cyipopt",
            )
            converged = True
        except pandapower.optimal_powerflow.OPFNotConverged as exc:
            converged = False
            n_failed_so_far = sum(1 for r in records if not r.converged)
            if n_failed_so_far < max_warning_timesteps:
                logger.warning(
                    "[Scenario 5 | %s] T=%d (%s): OPF did not converge — %s",
                    network_id,
                    t,
                    timestamp,
                    exc,
                )
            elif not suppressed_warning_logged:
                logger.warning(
                    "[Scenario 5 | %s] Further OPF non-convergence messages suppressed.",
                    network_id,
                )
                suppressed_warning_logged = True

        if converged:
            vm = net.res_bus["vm_pu"].copy()
            ll = (
                net.res_line["loading_percent"].copy()
                if not net.res_line.empty
                else pd.Series(dtype=float)
            )
            tl = (
                net.res_trafo["loading_percent"].copy()
                if not net.res_trafo.empty
                else pd.Series(dtype=float)
            )

            ov_buses = vm.index[vm > v_max + VOLTAGE_EPSILON].tolist()
            uv_buses = vm.index[vm < v_min - VOLTAGE_EPSILON].tolist()
            ov_lines = ll.index[ll > LINE_MAX_LOADING + LOADING_EPSILON].tolist()
            ov_trafos = tl.index[tl > TRAFO_MAX_LOADING + LOADING_EPSILON].tolist()

            p_result = net.res_sgen["p_mw"].reindex(der_idx).fillna(0.0).copy()
            q_result = net.res_sgen["q_mvar"].reindex(der_idx).fillna(0.0).copy()
            p_target = p_bound_ser.copy()
        else:
            vm = pd.Series(dtype=float)
            ll = pd.Series(dtype=float)
            tl = pd.Series(dtype=float)
            ov_buses = []
            uv_buses = []
            ov_lines = []
            ov_trafos = []
            p_result = None
            q_result = None
            p_target = None

        rec = TimestepRecord(
            t=t,
            timestamp=timestamp,
            vm_pu=vm,
            line_loading=ll,
            trafo_loading=tl,
            over_voltage_buses=ov_buses,
            under_voltage_buses=uv_buses,
            overloaded_lines=ov_lines,
            overloaded_trafos=ov_trafos,
            q_applied_mvar=q_result,
            p_applied_mw=p_result,
            p_target_mw=p_target,
            curtailment_needed=bool(ov_buses or uv_buses or ov_lines or ov_trafos),
            converged=converged,
        )
        records.append(rec)

        if t % 96 == 0:
            logger.info(
                "[Scenario 5 | %s] t=%d/%d (%.1f %%) | active_DER=%d | "
                "violations=%d | non-converged=%d",
                network_id,
                t,
                len(time_steps),
                100.0 * t / max(len(time_steps), 1),
                len(active_der_idx),
                sum(
                    1
                    for r in records
                    if r.over_voltage_buses
                    or r.under_voltage_buses
                    or r.overloaded_lines
                    or r.overloaded_trafos
                ),
                sum(1 for r in records if not r.converged),
            )

    elapsed = time.perf_counter() - t_start

    result = ScenarioResult.from_records(
        scenario_id="opf",
        network_id=network_id,
        records=records,
        elapsed_s=elapsed,
        dt_s=ap.dt_s,
    )

    logger.info(
        "[Scenario 5 | %s] Done. %.1f s | %d/%d converged | "
        "%d violation steps | curtailed=%.3f MWh",
        network_id,
        elapsed,
        result.n_converged,
        result.n_timesteps,
        result.n_violation_steps,
        result.curtailed_energy_mwh or 0.0,
    )

    return result
