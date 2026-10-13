"""
scenario_3_svc.py
=================
Scenario 3 — Centralised SVC voltage regulation benchmark.

Models a single Static VAr Compensator (SVC) as a pandapower sgen element
with p_mw=0 and controllable q_mvar.  The SVC is placed at one fixed MV bus
selected by a pre-run stress analysis, and held there for the full annual run.
No Arduino, no DER reactive power control.

Control law — deadbanded droop
-------------------------------
At every timestep the SVC reads the voltage at its host bus and computes:

    error     = SVC_V_TARGET - vm_pu
    deadband  = SVC_DEADBAND                   # 0.01 pu → inactive ±0.01

    if abs(error) <= deadband:
        q_cmd = 0.0
    elif error > deadband:                      # undervoltage → inject Q
        q_cmd = k_q * (error - deadband)
    else:                                       # overvoltage  → absorb Q
        q_cmd = k_q * (error + deadband)

    q_cmd = clip(q_cmd, -Q_MAX, +Q_MAX)

where:
    Q_MAX = SVC_Q_MAX_RATIO  × net.trafo.sn_mva.sum()   (= 0.20 × Σ sn_mva)
    k_q   = Q_MAX / (SVC_FULL_ACTION_DV − SVC_DEADBAND)  (= Q_MAX / 0.03)

Sign convention: q_mvar > 0 injects reactive power (raises voltage).

SVC bus selection
-----------------
The device is centralised at the substation: one compensator at the point of
common coupling regulating the whole MV group, as opposed to the distributed
inverter control of Scenarios 4 and 5.  Placement is therefore read from the
topology, not inferred from a voltage ranking.

Primary: the MV-side busbar of the in-service step-down substation
         transformer (net.trafo.lv_bus).
Fallback: the eligible MV bus with the fewest branches to the slack.
Last resort: the lowest-index eligible MV bus.
Guard: only MV buses (vn_kv matching the dominant MV voltage level) are
       eligible.  HV slack buses are excluded.

Parallel transformer handling
------------------------------
SVC Q_MAX is scaled from net.trafo.sn_mva.sum(), which includes all
transformers.  This is fair across all network sizes.

Usage
-----
    from profile_builder import build_profiles
    from scenario_result import adapt_profiles
    from scenario_3_svc import run_scenario_3

    net      = sb.get_simbench_net("1-MV-rural--2-sw")
    profiles = build_profiles(net, ...)
    result   = run_scenario_3(net, profiles)

    print(result.svc_bus)
    print(result.svc_q_max)
    print(result.n_violation_steps)

Notes
-----
voltage_depend_loads=False is mandatory for all runpp() calls on SimBench
networks (pandapower 3.2.0+ singular matrix without it).

net.sgen.q_mvar for DERs is forced to 0.0 every timestep.  Scenario 3 is a
pure SVC benchmark — DERs inject no reactive power.

The SVC sgen element is created once at the selected bus and removed at the
end of the run so the net object is left in a clean state for the caller.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandapower as pp
import pandas as pd

from scenario_result import (
    AdaptedProfiles,
    ScenarioResult,
    TimestepRecord,
    adapt_profiles,
)
from violation_detector import (
    V_MIN,
    V_MAX,
    VOLTAGE_EPSILON,
    LOADING_EPSILON,
)
# LINE_MAX_LOADING / TRAFO_MAX_LOADING are read through the module at CALL
# time, not from-imported: executor.apply_violation_limits() and
# run_benchmark_script.py rebind them per run, and a from-import would freeze
# the import-time value, so the per-record violation lists would disagree with
# detect_violations() (which resolves them late).
import violation_detector as _vd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SVC_V_TARGET:       float = 1.00   # voltage setpoint (pu)
SVC_DEADBAND:       float = 0.01   # deadband half-width (pu) → active ±0.01
SVC_FULL_ACTION_DV: float = 0.04   # voltage error at which Q_MAX is reached
SVC_Q_MAX_RATIO:    float = 0.20   # Q_MAX = 0.20 × net.trafo.sn_mva.sum()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _steps_per_day(dt_s: float) -> int:
    """
    Timesteps in one simulated day at this resolution.

    Used for the once-a-day progress line and the live-CSV rewrite cadence.
    A hardcoded 96 assumed SimBench's 15-min step, so at a finer resolution
    those fired many times per simulated day — at 10 s, every 16 minutes of
    simulated time, which for the CSV rewrite is 90x the file I/O.
    """
    try:
        spd = int(round(86400.0 / float(dt_s)))
    except (TypeError, ValueError, ZeroDivisionError):
        return 96
    return max(1, spd)


def _mv_kv(net) -> float:
    """
    Return the dominant MV voltage level in kV.

    Uses the modal vn_kv of buses that are neither the HV slack (vn_kv >=
    66 kV) nor LV (vn_kv <= 1 kV).  Falls back to all bus vn_kv values if
    the MV window returns nothing.
    """
    vn = net.bus["vn_kv"]
    mv_mask = (vn > 1.0) & (vn < 66.0)
    mv_levels = vn[mv_mask]
    if mv_levels.empty:
        # Fallback: use all buses except obvious HV slack
        mv_levels = vn[vn < 66.0]
    if mv_levels.empty:
        return float(vn.iloc[0])
    return float(mv_levels.mode().iloc[0])


def _eligible_mv_buses(net) -> pd.Index:
    """
    Return the index of in-service MV buses eligible for SVC placement.

    Excludes:
    - HV buses (vn_kv >= 66 kV)
    - LV buses (vn_kv <= 1 kV)
    - Out-of-service buses
    - The ext_grid slack bus (infinite busbar — injecting Q there is
      physically meaningless)
    """
    slack_buses = set(net.ext_grid["bus"].values)
    target_kv   = _mv_kv(net)

    mask = (
        (net.bus["vn_kv"] == target_kv)
        & (net.bus["in_service"] == True)
        & (~net.bus.index.isin(slack_buses))
    )
    return net.bus.index[mask]


def _substation_lv_buses(net) -> list[int]:
    """
    Return the MV-side busbars of the step-down substation transformers, in
    preference order.

    A bus qualifies if it is the lv_bus of an in-service two-winding
    transformer whose hv_bus sits at a strictly higher nominal voltage. That
    is the substation busbar — the point of common coupling between the HV
    system and the MV feeders.

    Ordering
    --------
    1. Transformers fed directly from an in-service ext_grid (the infeed
       actually carrying the substation) before any others.
    2. Descending sn_mva (the larger of two parallel transformers).
    3. Ascending bus index, so the choice is deterministic for a network
       whose busbars are otherwise identical (SimBench MV networks pair
       busbar1.1 / busbar1.2 at the same voltage level).

    Returns an empty list for a network with no step-down transformer, e.g.
    an LV feeder model supplied directly by an ext_grid.
    """
    if net.trafo.empty:
        return []

    vn = net.bus["vn_kv"]
    slack: set = set()
    if not net.ext_grid.empty:
        in_svc = net.ext_grid["in_service"] == True        # noqa: E712
        slack = {int(b) for b in net.ext_grid.loc[in_svc, "bus"].values}

    ranked: list[tuple[int, float, int]] = []
    for idx in net.trafo.index[net.trafo["in_service"] == True]:   # noqa: E712
        hv = int(net.trafo.at[idx, "hv_bus"])
        lv = int(net.trafo.at[idx, "lv_bus"])
        if hv not in vn.index or lv not in vn.index:
            continue
        if not bool(net.bus.at[lv, "in_service"]):
            continue
        if float(vn.at[hv]) <= float(vn.at[lv]):
            continue                      # not a step-down transformer
        sn = net.trafo.at[idx, "sn_mva"]
        sn = float(sn) if pd.notna(sn) else 0.0
        ranked.append((0 if hv in slack else 1, -sn, lv))

    ranked.sort()
    ordered: list[int] = []
    for _, _, lv in ranked:
        if lv not in ordered:
            ordered.append(lv)
    return ordered


def _select_svc_bus_substation(net, mv_buses: pd.Index) -> Optional[int]:
    """
    Primary selection: place the SVC at the substation MV busbar.

    This is the centralised-compensation configuration the scenario is
    designed to benchmark — one device at the point of common coupling,
    regulating the whole MV group, as distinct from the distributed
    inverter-based control of Scenarios 4 and 5. Placement is therefore a
    topological fact about the network, not something to be inferred from a
    voltage ranking: it is read straight from net.trafo and does not depend
    on a power flow, on the profile data, or on which stress case happens
    to be applied.

    Returns None only if the network has no in-service step-down transformer
    whose MV busbar is in the eligible set.
    """
    for bus in _substation_lv_buses(net):
        if bus in mv_buses:
            logger.info(
                "_select_svc_bus_substation: SVC bus selected = %d "
                "(substation MV busbar, centralised placement).", bus,
            )
            return bus

    logger.warning(
        "_select_svc_bus_substation: no step-down transformer with an "
        "eligible MV busbar found (net.trafo has %d rows) — falling back to "
        "the eligible MV bus nearest the slack.", len(net.trafo),
    )
    return None


def _select_svc_bus_nearest_slack(net, mv_buses: pd.Index) -> Optional[int]:
    """
    Fallback selection: the eligible MV bus electrically nearest the infeed.

    Counts branches from the slack bus over the connectivity graph —
    transformers included, switches respected — and returns the eligible MV
    bus with the fewest. For a radial distribution network that is the
    substation busbar or its immediate neighbour, so the result stays
    consistent with _select_svc_bus_substation().

    This is used only when the transformer table does not expose a
    step-down busbar. A voltage ranking is deliberately NOT used here: under
    the overvoltage stress case the minimum-voltage bus is the stiffest bus
    only while every feeder is generation-dominated. On a network with one
    load-dominated feeder (CIGRE MV: buses 12-14 carry almost no DER) the
    minimum lands at the far end of that feeder instead, which is remote
    placement wearing the label of the centralised case.

    Returns None if the graph cannot be built or no eligible bus is reachable
    from the slack.
    """
    if net.ext_grid.empty:
        logger.warning(
            "_select_svc_bus_nearest_slack: network has no ext_grid, so "
            "there is no infeed to measure distance from."
        )
        return None

    try:
        import networkx as nx
        import pandapower.topology as top
        graph = top.create_nxgraph(net, respect_switches=True)
    except Exception as exc:                              # noqa: BLE001
        logger.warning(
            "_select_svc_bus_nearest_slack: could not build the "
            "connectivity graph (%s).", exc,
        )
        return None

    in_svc = net.ext_grid["in_service"] == True           # noqa: E712
    slack_buses = [int(b) for b in net.ext_grid.loc[in_svc, "bus"].values]

    best: Optional[tuple[int, int]] = None                # (hops, bus)
    for slack in slack_buses:
        if slack not in graph:
            continue
        hops = nx.single_source_shortest_path_length(graph, slack)
        for bus in mv_buses:
            d = hops.get(int(bus))
            if d is None:
                continue
            cand = (int(d), int(bus))
            if best is None or cand < best:
                best = cand

    if best is None:
        logger.warning(
            "_select_svc_bus_nearest_slack: no eligible MV bus is reachable "
            "from any in-service ext_grid."
        )
        return None

    hops, selected = best
    logger.info(
        "_select_svc_bus_nearest_slack: SVC bus selected = %d "
        "(%d branch(es) from the slack — nearest eligible MV bus).",
        selected, hops,
    )
    return selected


def _select_svc_bus(net) -> int:
    """
    Select the fixed SVC bus. Centralised placement: the substation MV
    busbar.

    Priority
    --------
    1. net.trafo: the MV-side busbar of the step-down substation
       transformer.
    2. The eligible MV bus with the fewest branches to the slack, for a
       network that exposes no step-down transformer.
    3. Last resort: the lowest-index eligible MV bus. In pandapower and
       SimBench networks the substation busbars carry the lowest MV bus
       indices, so this is the closest guess available from the bus table
       alone.

    All three are topological and need no power flow, no profile data and no
    stress case, so the placement is reproducible and independent of the run
    configuration. Every path targets the same location, so a failure of
    path 1 lowers the confidence in the placement but does not change the
    configuration being benchmarked.

    The previous chain mixed objectives: it ranked buses by voltage under an
    overvoltage stress case and fell back to the largest deviation at the
    first violating timestep, which placed the device at the most remote
    feeder bus and silently reported a single-point distributed compensator
    as the centralised case.

    Raises
    ------
    ValueError if no eligible MV bus exists.
    """
    mv_buses = _eligible_mv_buses(net)
    if mv_buses.empty:
        raise ValueError(
            "_select_svc_bus: no eligible MV bus found. "
            "Check net.bus.vn_kv and net.bus.in_service."
        )

    # Primary: the substation MV busbar, read from the transformer table
    bus = _select_svc_bus_substation(net, mv_buses)
    if bus is not None:
        return bus

    # Fallback: the eligible MV bus nearest the infeed
    bus = _select_svc_bus_nearest_slack(net, mv_buses)
    if bus is not None:
        return bus

    # Last resort: lowest-index eligible MV bus (nearest the infeed)
    bus = int(mv_buses[0])
    logger.warning(
        "_select_svc_bus: using last-resort bus %d — verify placement.", bus,
    )
    return bus


def _compute_svc_params(net) -> tuple[float, float]:
    """
    Compute Q_MAX and k_q from network transformer data.

    Q_MAX = SVC_Q_MAX_RATIO × Σ(net.trafo.sn_mva)    [MVAr]
    k_q   = Q_MAX / (SVC_FULL_ACTION_DV - SVC_DEADBAND)

    Returns
    -------
    (q_max, k_q)  both in MVAr / pu_error units.

    Raises
    ------
    ValueError if net.trafo is empty (no transformer → no Q_MAX reference).
    """
    if net.trafo.empty:
        raise ValueError(
            "_compute_svc_params: net.trafo is empty — cannot derive Q_MAX. "
            "Scenario 3 requires at least one transformer."
        )
    sn_total = float(net.trafo["sn_mva"].sum())
    q_max    = SVC_Q_MAX_RATIO * sn_total
    k_q      = q_max / (SVC_FULL_ACTION_DV - SVC_DEADBAND)
    logger.info(
        "_compute_svc_params: sn_mva_sum=%.2f MVA | "
        "Q_MAX=%.3f MVAr | k_q=%.3f MVAr/pu",
        sn_total, q_max, k_q,
    )
    return q_max, k_q


def _droop_q(vm_pu: float, q_max: float, k_q: float) -> tuple[float, bool]:
    """
    Compute the SVC reactive power command from the deadbanded droop law.

    Parameters
    ----------
    vm_pu : measured voltage at the SVC bus (pu).
    q_max : maximum reactive power magnitude (MVAr).
    k_q   : droop gain (MVAr / pu error beyond deadband).

    Returns
    -------
    (q_cmd, saturated)
        q_cmd     : reactive power command (MVAr).  Positive = inject.
        saturated : True if |q_cmd| reached q_max before clipping.
    """
    error = SVC_V_TARGET - vm_pu

    if abs(error) <= SVC_DEADBAND:
        return 0.0, False

    if error > SVC_DEADBAND:
        q_raw = k_q * (error - SVC_DEADBAND)
    else:
        q_raw = k_q * (error + SVC_DEADBAND)

    saturated = abs(q_raw) >= q_max
    q_cmd     = float(np.clip(q_raw, -q_max, q_max))
    return q_cmd, saturated


def _empty_series() -> pd.Series:
    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_scenario_3(
        net,
        profiles:   dict,
        network_id: str   = "unknown",
        v_min:      float = V_MIN,
        v_max:      float = V_MAX,
        publish_fn        = None,
        enable_checkpointing: bool = True,
        live_csv_rewrite_fn = None,
) -> ScenarioResult:
    """
    Run Scenario 3 — centralised SVC voltage regulation benchmark.

    Parameters
    ----------
    net        : pandapower network.  Modified in place every timestep.
                 Caller should deep-copy if the original net is needed later.
                 The SVC sgen element is removed from net at the end of the
                 run so the caller receives a clean network.
    profiles   : dict from profile_builder.build_profiles().
    network_id : human-readable identifier stored in ScenarioResult.
    v_min      : lower voltage planning limit (pu).  Default 0.95.
    v_max      : upper voltage planning limit (pu).  Default 1.05.

    Returns
    -------
    ScenarioResult  with scenario_id="svc", svc_bus and svc_q_max populated.
    """
    t_start = time.perf_counter()
    ap: AdaptedProfiles = adapt_profiles(net, profiles)
    _log_every = _steps_per_day(ap.dt_s)   # one progress line per sim day
    _T = len(ap.times)
    if publish_fn is not None:
        publish_fn.on_scenario_start("svc", "SVC", _T)
    time_steps = range(len(ap.times))

    resumed_records: list[TimestepRecord] = []
    if publish_fn is not None and enable_checkpointing:
        resumed_records = publish_fn.get_resume_records("svc")
    start_t = (resumed_records[-1].t + 1) if resumed_records else 0
    _T_full = len(ap.times)

    if start_t >= len(ap.times) and resumed_records:
        q_max, k_q = _compute_svc_params(net)
        svc_bus    = _select_svc_bus(net)
        logger.info(
            "[Scenario 3 | %s] Checkpoint already covers all %d steps — skipping simulation.",
            network_id, len(ap.times),
        )
        elapsed = publish_fn.cumulative_elapsed_s() if publish_fn is not None else time.perf_counter() - t_start
        result = ScenarioResult.from_records(
            scenario_id="svc", network_id=network_id,
            records=resumed_records, elapsed_s=elapsed, dt_s=ap.dt_s,
            svc_bus=svc_bus, svc_q_max=q_max,
        )
        if publish_fn is not None:
            publish_fn.on_scenario_end(result)
        return result
    if start_t > 0:
        logger.info("[Scenario 3 | %s] Resuming from t=%d/%d.", network_id, start_t, len(ap.times))
    time_steps = range(start_t, len(ap.times))

    # ------------------------------------------------------------------
    # Reset controllers and results from any prior scenario on this net
    # ------------------------------------------------------------------
    net.controller.drop(net.controller.index, inplace=True)
    pp.reset_results(net)

    # ------------------------------------------------------------------
    # SVC parameters
    # ------------------------------------------------------------------
    q_max, k_q = _compute_svc_params(net)
    svc_bus    = _select_svc_bus(net)

    # Create SVC as a static generator with p_mw=0
    svc_idx = pp.create_sgen(
        net,
        bus       = svc_bus,
        p_mw      = 0.0,
        q_mvar    = 0.0,
        name      = "SVC_Scenario3",
        type      = "SVC",
        in_service= True,
    )

    logger.info(
        "[Scenario 3 | %s] SVC setup: bus=%d, Q_MAX=%.3f MVAr, "
        "k_q=%.3f MVAr/pu, deadband=±%.3f pu | "
        "%d timesteps, %d DERs, %d loads",
        network_id, svc_bus, q_max, k_q, SVC_DEADBAND,
        len(time_steps),
        len(ap.der_p.columns), len(ap.load_idx),
    )

    records: list[TimestepRecord] = resumed_records.copy()

    try:
        # --------------------------------------------------------------
        # Timestep loop
        # --------------------------------------------------------------
        for t in time_steps:
            t0 = time.perf_counter()
            timestamp = ap.times[t]

            # [1] Write profiles (index-explicit)
            if not ap.load_p.empty:
                net.load.loc[ap.load_idx, "p_mw"]   = ap.load_p.iloc[t].values
                net.load.loc[ap.load_idx, "q_mvar"] = ap.load_q.iloc[t].values

            if not ap.der_p.empty:
                net.sgen.loc[ap.der_p.columns, "p_mw"]   = ap.der_p.iloc[t].values
                net.sgen.loc[ap.der_p.columns, "q_mvar"] = 0.0

            # SVC starts at q_mvar=0 before each pre-control solve
            net.sgen.at[svc_idx, "q_mvar"] = 0.0

            # [2] Pre-control power flow (SVC q=0)
            try:
                pp.runpp(net, voltage_depend_loads=False)
                converged_pre = True
            except Exception:
                converged_pre = False

            if not converged_pre:
                logger.warning(
                    "[Scenario 3 | %s] t=%d: pre-PF diverged — SVC holds q=0.",
                    network_id, t,
                )
                records.append(TimestepRecord(
                    t=t, timestamp=timestamp,
                    vm_pu=_empty_series(),
                    line_loading=_empty_series(),
                    trafo_loading=_empty_series(),
                    over_voltage_buses=[], under_voltage_buses=[],
                    overloaded_lines=[], overloaded_trafos=[],
                    q_applied_mvar=pd.Series({svc_idx: 0.0}, dtype=float),
                    p_applied_mw=None,
                    p_target_mw=None, curtailment_needed=False,
                    converged=False,
                    svc_q_mvar=0.0,
                    svc_saturated=False,
                    losses_mw=None,
                    grid_import_mw=None,
                    der_gen_mw=float(ap.der_p.iloc[t].sum()) if not ap.der_p.empty  else 0.0,
                    load_mw=float(ap.load_p.iloc[t].sum())   if not ap.load_p.empty else 0.0,
                    t_total_ms=(time.perf_counter() - t0) * 1e3,
                ))
                continue

            # [3] Droop controller — read SVC bus voltage
            vm_svc = float(net.res_bus.at[svc_bus, "vm_pu"])
            q_cmd, saturated = _droop_q(vm_svc, q_max, k_q)

            # [4] Apply SVC setpoint and post-control power flow
            net.sgen.at[svc_idx, "q_mvar"] = q_cmd

            if abs(q_cmd) > 0.0:
                try:
                    pp.runpp(net, voltage_depend_loads=False)
                    converged_post = True
                except Exception:
                    converged_post = False
                    logger.warning(
                        "[Scenario 3 | %s] t=%d: post-SVC PF diverged "
                        "(q_cmd=%.4f MVAr).",
                        network_id, t, q_cmd,
                    )
            else:
                # SVC in deadband — reuse pre-control results
                converged_post = True

            # [5] Build TimestepRecord
            converged = converged_post

            der_gen_mw_t = float(ap.der_p.iloc[t].sum())  if not ap.der_p.empty  else 0.0
            load_mw_t    = float(ap.load_p.iloc[t].sum()) if not ap.load_p.empty else 0.0

            if converged:
                vm  = net.res_bus["vm_pu"].copy()
                ll  = net.res_line["loading_percent"].copy()
                tl  = net.res_trafo["loading_percent"].copy()

                ov_buses  = vm.index[vm > v_max + VOLTAGE_EPSILON].tolist()
                uv_buses  = vm.index[vm < v_min - VOLTAGE_EPSILON].tolist()
                ov_lines  = ll.index[ll > _vd.LINE_MAX_LOADING  + LOADING_EPSILON].tolist()
                ov_trafos = tl.index[tl > _vd.TRAFO_MAX_LOADING + LOADING_EPSILON].tolist()

                losses_mw_t      = float(net.res_line["pl_mw"].sum()
                                         + net.res_trafo["pl_mw"].sum())
                grid_import_mw_t = float(net.res_ext_grid["p_mw"].sum())
            else:
                vm = ll = tl = _empty_series()
                ov_buses = uv_buses = ov_lines = ov_trafos = []
                losses_mw_t = grid_import_mw_t = None

            rec = TimestepRecord(
                t=t, timestamp=timestamp,
                vm_pu=vm, line_loading=ll, trafo_loading=tl,
                over_voltage_buses=ov_buses, under_voltage_buses=uv_buses,
                overloaded_lines=ov_lines, overloaded_trafos=ov_trafos,
                q_applied_mvar=pd.Series({svc_idx: q_cmd}, dtype=float),
                p_applied_mw=None,
                p_target_mw=None,
                curtailment_needed=False,
                converged=converged,
                svc_q_mvar=q_cmd,
                svc_saturated=saturated,
                losses_mw=losses_mw_t,
                grid_import_mw=grid_import_mw_t,
                der_gen_mw=der_gen_mw_t,
                load_mw=load_mw_t,
                t_total_ms=(time.perf_counter() - t0) * 1e3,
            )
            if publish_fn is not None:
                publish_fn.on_timestep(rec)
            records.append(rec)

            if t % _log_every == 0:
                if live_csv_rewrite_fn is not None:
                    partial = ScenarioResult.from_records(
                        scenario_id="svc", network_id=network_id,
                        records=records, elapsed_s=(publish_fn.cumulative_elapsed_s() 
                                                    if publish_fn is not None else time.perf_counter() - t_start),
                        dt_s=ap.dt_s,   # correct value, already in scope, no placeholder
                    )
                    live_csv_rewrite_fn(partial)
                logger.info(
                    "[Scenario 3 | %s] t=%d/%d (%.1f%%) | "
                    "vm_svc=%.4f | q_cmd=%.3f MVAr | violations=%d",
                    network_id, t, _T_full,
                    100.0 * t / max(_T_full, 1),
                    vm_svc, q_cmd,
                    sum(1 for r in records
                        if r.over_voltage_buses or r.under_voltage_buses),
                )

    finally:
        # ------------------------------------------------------------------
        # Always remove the SVC sgen from net, even on exception.
        # This leaves the net object clean for the caller.
        # ------------------------------------------------------------------
        net.sgen.drop(index=svc_idx, inplace=True)
        pp.reset_results(net)
        logger.debug(
            "[Scenario 3 | %s] SVC sgen (idx=%d) removed from net.",
            network_id, svc_idx,
        )

    # ------------------------------------------------------------------
    # Build ScenarioResult
    # ------------------------------------------------------------------
    elapsed = publish_fn.cumulative_elapsed_s() if publish_fn is not None else time.perf_counter() - t_start

    result = ScenarioResult.from_records(
        scenario_id = "svc",
        network_id  = network_id,
        records     = records,
        elapsed_s   = elapsed,
        dt_s        = ap.dt_s,
        svc_bus     = svc_bus,
        svc_q_max   = q_max,
    )

    # SVC activity summary
    svc_active    = sum(1 for r in records if r.svc_q_mvar and abs(r.svc_q_mvar) > 0.0)
    svc_saturated = sum(1 for r in records if r.svc_saturated)
    q_total_abs   = sum(
        abs(r.svc_q_mvar) for r in records
        if r.svc_q_mvar is not None
    )

    logger.info(
        "[Scenario 3 | %s] Done. %.1f s | %d/%d converged | "
        "%d violation steps | SVC active %d steps | "
        "saturated %d steps | |Q| total=%.2f MVAr·steps",
        network_id, elapsed,
        result.n_converged, result.n_timesteps,
        result.n_violation_steps,
        svc_active, svc_saturated, q_total_abs,
    )
    if publish_fn is not None:
        publish_fn.on_scenario_end(result)
    return result
