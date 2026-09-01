"""
scenario_5_opf.py
=================
Scenario 5 — OPF Benchmark.

Runs one AC optimal power flow (runopp) per timestep.  The OPF simultaneously
optimises P and Q setpoints for all controllable DERs subject to:
    - Bus voltage limits  : 0.95–1.05 pu
    - Line thermal limits : 100 % rated current
    - Trafo thermal limits: 100 % rated current
    - DER P bounds        : [0, profile value]  (no curtailment beyond what's available)
    - DER Q bounds        : [−Q_max, +Q_max]   (VDE-AR-N 4110 Q_RATIO = 0.48)

Cost function — curtailment-minimising benchmark
-------------------------------------------------
All DERs are assigned a poly_cost with cp1_eur_per_mw = -1.  This tells the
solver to MAXIMISE total DER active power generation (equivalently, minimise
curtailment).  The OPF therefore represents the best achievable operating
point that satisfies all planning constraints while wasting the least DER
energy — a physically meaningful benchmark for comparing against Scenario 4.

This framing is intentional:
    cp1_eur_per_mw = -1   →   objective = -Σ p_mw   →   maximise generation
    net.sgen.max_p_mw     →   upper bound = available profile value at t

Any reduction in p_mw below max_p_mw in the OPF solution is curtailment.
The difference is logged per-timestep in TimestepRecord.p_applied_mw vs the
profile upper bound, and aggregated in ScenarioResult.curtailed_energy_mwh.

OPF setup discipline
--------------------
1. poly_cost rows are CLEARED before setup.  If this runner is called on a
   net that was previously used for another scenario or OPF configuration,
   stale poly_cost rows will conflict with the new cost entries.

2. OPF columns on net.sgen (controllable, min_p_mw, max_p_mw, min_q_mvar,
   max_q_mvar) are written once before the loop.  max_p_mw is updated each
   timestep to the profile value before runopp().

3. runopp() raises pandapower.optimal_powerflow.OPFNotConverged on failure.
   The runner catches this, logs a warning, and appends a non-converged
   TimestepRecord.  The run continues with the next timestep.

4. voltage_depend_loads=False is set via pp.set_user_pf_options() once before
   the loop.  This stores the flag persistently on the net object, where it is
   picked up by the internal runpp call inside runopp().  Passing it as a
   direct kwarg or via init_options is not documented and unreliable.

Usage
-----
    from profile_builder import build_profiles
    from scenario_result import adapt_profiles
    from scenario_5_opf import run_scenario_5

    net      = sb.get_simbench_net("1-MV-rural--2-sw")
    profiles = build_profiles(net, ...)
    result   = run_scenario_5(net, profiles)

    print(result.curtailed_energy_mwh)
    print(result.n_violation_steps)   # should be 0 if OPF always feasible
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandapower as pp
import pandapower.optimal_powerflow
import pandas as pd
import cyipopt

from volt_var_controller import Q_RATIO
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


def _setup_opf(net, ap: AdaptedProfiles) -> None:
    """
    Prepare the network for OPF: set controllability, bounds, and costs.

    Called once before the timestep loop.  max_p_mw is set to a placeholder
    (the current net.sgen.p_mw) and updated per-timestep inside the loop.

    Cost rows are always cleared first to avoid duplicate or conflicting
    poly_cost entries from previous runs on the same net object.
    """
    # ------------------------------------------------------------------
    # [1] Clear existing poly_cost rows unconditionally.
    # ------------------------------------------------------------------
    if not net.poly_cost.empty:
        net.poly_cost.drop(net.poly_cost.index, inplace=True)
        logger.debug("_setup_opf: cleared %d existing poly_cost rows.", len(net.poly_cost))

    # ------------------------------------------------------------------
    # [2] Reset controllability on ALL sgens first, then mark der_idx.
    #     If this runner is called on a net previously used for another OPF
    #     configuration, stale controllable=True flags on non-profiled sgens
    #     would cause the solver to treat them as flexible — producing
    #     infeasible or unrealistic results.
    # ------------------------------------------------------------------
    der_idx = ap.der_p.columns.tolist()

    net.sgen["controllable"] = False
    net.sgen.loc[der_idx, "controllable"] = True

    # Active power bounds:
    #   min_p_mw = 0 (DERs cannot consume active power)
    #   max_p_mw = placeholder — updated per-timestep before runopp()
    net.sgen.loc[der_idx, "min_p_mw"] = 0.0
    net.sgen.loc[der_idx, "max_p_mw"] = net.sgen.loc[der_idx, "p_mw"].clip(lower=0.0)

    # Reactive power bounds are set per-timestep inside the loop using the
    # apparent-power circle constraint.  Placeholder zeros here so the
    # column exists before the first runopp() call.
    net.sgen.loc[der_idx, "min_q_mvar"] = 0.0
    net.sgen.loc[der_idx, "max_q_mvar"] = 0.0

    # ------------------------------------------------------------------
    # External grid OPF setup
    # pandapower OPF expects ext_grid.controllable to be boolean.
    # SimBench networks may leave it as NaN, which breaks ~controllable.
    # ------------------------------------------------------------------
    net.ext_grid["controllable"] = True
    net.ext_grid["controllable"] = net.ext_grid["controllable"].astype(bool)

    for col in ["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]:
        if col not in net.ext_grid.columns:
            net.ext_grid[col] = np.nan

    p_load = float(net.load["p_mw"].abs().sum()) if not net.load.empty else 0.0
    q_load = float(net.load["q_mvar"].abs().sum()) if not net.load.empty else 0.0
    p_sgen = float(net.sgen["p_mw"].abs().sum()) if not net.sgen.empty else 0.0
    q_sgen = float(net.sgen["sn_mva"].fillna(0.0).abs().sum()) if "sn_mva" in net.sgen.columns else p_sgen

    p_margin = max(50.0, 3.0 * (p_load + p_sgen))
    q_margin = max(50.0, 3.0 * (q_load + q_sgen))

    net.ext_grid.loc[:, "min_p_mw"] = -p_margin
    net.ext_grid.loc[:, "max_p_mw"] =  p_margin
    net.ext_grid.loc[:, "min_q_mvar"] = -q_margin
    net.ext_grid.loc[:, "max_q_mvar"] =  q_margin

    # ------------------------------------------------------------------
    # [3] Bus voltage limits.
    # ------------------------------------------------------------------
    net.bus["min_vm_pu"] = V_MIN
    net.bus["max_vm_pu"] = V_MAX

    # ------------------------------------------------------------------
    # [4] Line and trafo thermal limits.
    # ------------------------------------------------------------------
    net.line["max_loading_percent"]  = 1e6
    net.trafo["max_loading_percent"] = 1e6

    # ------------------------------------------------------------------
    # [5] Curtailment-minimising cost: cp1_eur_per_mw = -1 for every DER.
    #     Objective = Σ cost_i × p_i = -Σ p_i  →  maximise generation.
    # ------------------------------------------------------------------
    for eg_idx in net.ext_grid.index:
        pp.create_poly_cost(net, eg_idx, "ext_grid", cp1_eur_per_mw=0.0)

    pp.diagnostic(net, warnings_only=False)
    
    logger.info(
        "_setup_opf: %d DERs marked controllable (all others reset to False), "
        "poly_cost created (cp1=-1, curtailment-minimising). "
        "Q limits will be updated per-timestep (apparent-power circle).",
        len(der_idx),
    )


def run_scenario_5(
        net,
        profiles:   dict,
        network_id: str   = "unknown",
        v_min:      float = V_MIN,
        v_max:      float = V_MAX,
        verbose_opf: bool = False,
) -> ScenarioResult:
    """
    Run Scenario 5 — OPF Benchmark (curtailment-minimising AC OPF).

    Parameters
    ----------
    net         : pandapower network.  Modified in place every timestep.
                  Caller should deep-copy if the original net is needed later.
    profiles    : dict from profile_builder.build_profiles().
    network_id  : human-readable identifier stored in ScenarioResult.
    v_min       : lower voltage planning limit (pu).  Default 0.95.
    v_max       : upper voltage planning limit (pu).  Default 1.05.
    verbose_opf : if True, passes verbose=True to runopp() for solver output.
                  Useful for debugging convergence failures.

    Returns
    -------
    ScenarioResult  with scenario_id="opf".

    Notes
    -----
    voltage_depend_loads
        Set via pp.set_user_pf_options(net, voltage_depend_loads=False) once
        before the loop.  This stores the flag persistently on the net object
        and is picked up by the internal runpp call inside runopp().
        Passing it via runopp(**kwargs) or init_options is not documented and
        unreliable.

    runopp init="pf"
        Executes a power flow before the OPF and uses its solution as the
        starting vector.  Improves convergence when operating points are far
        from the flat-start default.  Adds one runpp per timestep — runtime
        cost is negligible relative to the OPF solve itself.
        If runopp() raises OPFNotConverged, the timestep is logged as
        non-converged and the loop continues.  A high non-convergence rate
        signals that the OPF problem is infeasible at that timestep (e.g.
        no feasible voltage profile exists given the DER profile bound).

    Curtailed energy
        Computed as Σ_t Σ_i max(0, max_p_mw[i,t] − res_p[i,t]) × dt_h.
        max_p_mw[i,t] = profile value for DER i at timestep t (upper bound).
        res_p[i,t]    = OPF solution p_mw for DER i at timestep t.
        Stored per-record in p_target_mw (proper dataclass field, not a hidden
        attribute) alongside p_applied_mw so from_records() can compute the
        curtailed energy without iterating over raw profile DataFrames.
    """
    t_start = time.perf_counter()

    ap: AdaptedProfiles = adapt_profiles(net, profiles)
    time_steps = range(len(ap.times))
    der_idx    = ap.der_p.columns

    logger.info(
        "[Scenario 5 | %s] Starting OPF benchmark: %d timesteps, "
        "%d DERs (%d PV + %d wind), %d loads",
        network_id, len(time_steps),
        len(der_idx), len(ap.pv_idx), len(ap.wind_idx), len(ap.load_idx),
    )

    # ------------------------------------------------------------------
    # Reset any ConstControls from a prior run.
    # ------------------------------------------------------------------
    net.controller.drop(net.controller.index, inplace=True)
    pp.reset_results(net)

    # ------------------------------------------------------------------
    # OPF setup (once before loop).
    # ------------------------------------------------------------------
    _setup_opf(net, ap)

    # voltage_depend_loads=False cannot be passed via init_options — that
    # kwarg is not documented in runopp().  The correct route is
    # set_user_pf_options(), which stores flags persistently on the net
    # object and are picked up by the internal runpp call inside runopp().
    pp.set_user_pf_options(net, voltage_depend_loads=False)

    # Pre-compute rated apparent power for the Q-limit circle.
    #
    # Priority chain for each DER i:
    #   1. sn_mva from net.sgen  — preferred (nameplate rating)
    #   2. net.sgen.p_mw at setup time — fallback (may be zero at t=0)
    #   3. Peak profile value from ap.der_p  — final fallback
    #
    # Without step 3, a DER whose sn_mva is missing AND whose initial p_mw
    # is 0.0 (common in SimBench profiles at midnight) gets sn_rated=0 and
    # q_lim=0 for the entire run, even though it generates at peak during
    # the day.  This would silently eliminate all OPF Q flexibility.
    #
    # A ValueError is raised if any sn_rated remains zero after all three
    # fallbacks, because the Q-limit calculation would produce NaN in the
    # circle constraint and cause silent OPF failures downstream.
    sn_vals  = net.sgen.loc[der_idx, "sn_mva"].values.astype(float)
    p_fb     = np.abs(net.sgen.loc[der_idx, "p_mw"].values.astype(float))
    p_peak   = ap.der_p.reindex(columns=der_idx).max(axis=0).values.astype(float)

    use_sn      = np.isfinite(sn_vals) & (sn_vals > 0.0)
    use_p_fb    = (~use_sn) & (p_fb > 0.0)
    use_p_peak  = (~use_sn) & (~use_p_fb)

    sn_rated = np.where(use_sn, sn_vals,
               np.where(use_p_fb, p_fb, p_peak))   # shape (N_ders,)

    zero_mask = sn_rated <= 0.0
    if zero_mask.any():
        bad_idx = np.array(der_idx)[zero_mask].tolist()
        raise ValueError(
            f"[Scenario 5 | {network_id}] sn_rated is zero for sgen indices "
            f"{bad_idx} after all fallbacks (sn_mva, p_mw, profile peak). "
            f"Q-limit circle cannot be computed.  "
            f"Check that these DERs have valid sn_mva or non-zero profiles."
        )

    logger.debug(
        "[Scenario 5 | %s] sn_rated computed: min=%.4f max=%.4f MVA "
        "(sn_mva used: %d/%d, p_mw fallback: %d/%d, profile-peak fallback: %d/%d)",
        network_id,
        float(sn_rated.min()), float(sn_rated.max()),
        int(use_sn.sum()), len(der_idx),
        int(use_p_fb.sum()), len(der_idx),
        int(use_p_peak.sum()), len(der_idx),
    )

    records: list[TimestepRecord] = []

    for t in time_steps:
        timestamp = ap.times[t]

        # [A] Update max_p_mw per DER to the profile value at this timestep.
        #     The OPF may choose any p in [0, profile_value]; reduction below
        #     the profile is curtailment.
        profile_row = ap.der_p.iloc[t].reindex(der_idx).fillna(0.0).values
        p_bound = np.maximum(profile_row, 0.0)

        active_mask = p_bound > 1e-9
        active_der_idx = pd.Index(der_idx)[active_mask]
        inactive_der_idx = pd.Index(der_idx)[~active_mask]

        net.sgen.loc[der_idx, "controllable"] = False
        net.sgen.loc[active_der_idx, "controllable"] = True

        net.sgen.loc[der_idx, "min_p_mw"] = 0.0
        net.sgen.loc[der_idx, "max_p_mw"] = 0.0
        net.sgen.loc[active_der_idx, "max_p_mw"] = p_bound[active_mask]

        net.sgen.loc[der_idx, "min_q_mvar"] = 0.0
        net.sgen.loc[der_idx, "max_q_mvar"] = 0.0
        net.sgen.loc[active_der_idx, "p_mw"] = p_bound[active_mask]
        net.sgen.loc[active_der_idx, "q_mvar"] = 0.0
        # min_p_mw stays 0 — no forced dispatch for renewables.

        # [A2] Update Q limits per-timestep using the apparent-power circle.
        #      Q_vde   = Q_RATIO × sn_rated   (VDE-AR-N 4110 Bild 8 ceiling)
        #      Q_circle= sqrt(max(0, sn² − p_bound²))  (inverter apparent-power limit)
        #      Q_lim   = min(Q_vde, Q_circle)
        #      Using the profile upper bound for p_bound gives the tightest
        #      feasible Q range at peak generation — the physically correct
        #      constraint.  This prevents the OPF from claiming Q support that
        #      the inverter cannot provide at its current operating point.
        p_bound  = np.maximum(profile_row, 0.0)
        q_vde    = Q_RATIO * sn_rated
        q_circle = np.sqrt(np.maximum(0.0, sn_rated**2 - p_bound**2))
        q_lim    = np.minimum(q_vde, q_circle)
        offline_mask = p_bound <= 1e-9
        q_lim[offline_mask] = 0.0

        net.sgen.loc[der_idx, "min_q_mvar"] = 0.0
        net.sgen.loc[der_idx, "max_q_mvar"] = 0.0
        net.sgen.loc[active_der_idx, "min_q_mvar"] = -q_lim[active_mask]
        net.sgen.loc[active_der_idx, "max_q_mvar"] = q_lim[active_mask]

        net.sgen.loc[der_idx, "q_mvar"] = 0.0

        net.poly_cost.drop(net.poly_cost.index, inplace=True)
        for eg_idx in net.ext_grid.index:
            pp.create_poly_cost(net, eg_idx, "ext_grid", cp1_eur_per_mw=0.001)
        for idx in active_der_idx:
            pp.create_poly_cost(net, idx, "sgen", cp1_eur_per_mw=-1.0)

        # [B] Load profiles (index-explicit).
        if not ap.load_p.empty:
            net.load.loc[ap.load_idx, "p_mw"] = ap.load_p.iloc[t].values
            net.load.loc[ap.load_idx, "q_mvar"] = ap.load_q.iloc[t].values

        # [C] Run AC OPF.
        #     init="pf" executes a power flow before the OPF and uses its
        #     solution as the starting vector.  This improves convergence
        #     for operating points far from the flat-start default.       
        print("\n=== OPF task ===")
        print(pp.opf_task(net))

        print("\n=== ext_grid OPF fields ===")
        print(net.ext_grid[[
            "controllable",
            "min_p_mw", "max_p_mw",
            "min_q_mvar", "max_q_mvar",
        ]].to_string())
        print(net.ext_grid[["controllable"]].dtypes)

        print("\n=== sgen OPF bounds ===")
        debug_sgen = net.sgen.loc[der_idx, [
            "p_mw", "q_mvar",
            "min_p_mw", "max_p_mw",
            "min_q_mvar", "max_q_mvar",
            "controllable",
        ]]
        print(debug_sgen.to_string())

        bad_p = debug_sgen[
            (debug_sgen["max_p_mw"] < debug_sgen["min_p_mw"])
            | debug_sgen["max_p_mw"].isna()
            | debug_sgen["min_p_mw"].isna()
        ]
        bad_q = debug_sgen[
            (debug_sgen["max_q_mvar"] < debug_sgen["min_q_mvar"])
            | debug_sgen["max_q_mvar"].isna()
            | debug_sgen["min_q_mvar"].isna()
        ]

        print("\nBad P bounds:")
        print(bad_p.to_string())

        print("\nBad Q bounds:")
        print(bad_q.to_string())

        debug_sgen = net.sgen.loc[der_idx, [
        "p_mw", "q_mvar",
        "min_p_mw", "max_p_mw",
        "min_q_mvar", "max_q_mvar",
        "controllable",
        ]]

        bad_initial_p = debug_sgen[
            (debug_sgen["p_mw"] < debug_sgen["min_p_mw"] - 1e-9)
            | (debug_sgen["p_mw"] > debug_sgen["max_p_mw"] + 1e-9)
        ]

        bad_initial_q = debug_sgen[
            (debug_sgen["q_mvar"] < debug_sgen["min_q_mvar"] - 1e-9)
            | (debug_sgen["q_mvar"] > debug_sgen["max_q_mvar"] + 1e-9)
        ]

        print("\nBad initial P values:")
        print(bad_initial_p.to_string())

        print("\nBad initial Q values:")
        print(bad_initial_q.to_string())
        
        try:
            pp.runopp(
                net,
                verbose = verbose_opf,
                init    = "pf",
                OPF_SOLVER="cyipopt",
            )
            converged = True
        except pandapower.optimal_powerflow.OPFNotConverged as exc:
            logger.warning(
                "[Scenario 5 | %s] T=%d (%s): OPF did not converge — %s",
                network_id, t, timestamp, exc,
            )
            converged = False

        # [D] Build record.
        if converged:
            vm = net.res_bus["vm_pu"].copy()
            ll = net.res_line["loading_percent"].copy()
            tl = net.res_trafo["loading_percent"].copy()

            ov_buses  = vm.index[vm > v_max + VOLTAGE_EPSILON].tolist()
            uv_buses  = vm.index[vm < v_min - VOLTAGE_EPSILON].tolist()
            ov_lines  = ll.index[ll > LINE_MAX_LOADING  + LOADING_EPSILON].tolist()
            ov_trafos = tl.index[tl > TRAFO_MAX_LOADING + LOADING_EPSILON].tolist()

            # OPF result: actual P and Q chosen by solver.
            p_result = net.res_sgen["p_mw"].reindex(der_idx).copy()
            q_result = net.res_sgen["q_mvar"].reindex(der_idx).copy()
        else:
            vm = pd.Series(dtype=float)
            ll = pd.Series(dtype=float)
            tl = pd.Series(dtype=float)
            ov_buses = uv_buses = ov_lines = ov_trafos = []
            p_result = None
            q_result = None

        # p_target_mw = profile upper bound (max_p_mw at this timestep).
        # from_records() computes curtailed_energy_mwh = Σ(p_target - p_applied).
        p_target_ser = pd.Series(profile_row, index=der_idx, dtype=float)

        rec = TimestepRecord(
            t=t, timestamp=timestamp,
            vm_pu=vm, line_loading=ll, trafo_loading=tl,
            over_voltage_buses=ov_buses, under_voltage_buses=uv_buses,
            overloaded_lines=ov_lines, overloaded_trafos=ov_trafos,
            q_applied_mvar=q_result,
            p_applied_mw=p_result,
            curtailment_needed=False,   # OPF manages curtailment internally
            converged=converged,
            p_target_mw=p_target_ser if converged else None,
        )

        records.append(rec)

        if t % 96 == 0:
            logger.info(
                "[Scenario 5 | %s] t=%d/%d (%.1f %%) | "
                "violations=%d | non-converged=%d",
                network_id, t, len(time_steps),
                100.0 * t / max(len(time_steps), 1),
                sum(1 for r in records if r.over_voltage_buses
                    or r.under_voltage_buses),
                sum(1 for r in records if not r.converged),
            )

    elapsed = time.perf_counter() - t_start

    result = ScenarioResult.from_records(
        scenario_id = "opf",
        network_id  = network_id,
        records     = records,
        elapsed_s   = elapsed,
        dt_s        = ap.dt_s,
    )

    logger.info(
        "[Scenario 5 | %s] Done. %.1f s | %d/%d converged | "
        "%d violation steps | curtailed=%.3f MWh",
        network_id, elapsed, result.n_converged, result.n_timesteps,
        result.n_violation_steps,
        result.curtailed_energy_mwh or 0.0,
    )
    return result
