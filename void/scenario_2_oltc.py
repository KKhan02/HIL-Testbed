"""
scenario_2_oltc.py
==================
Scenario 2 — OLTC-only voltage regulation benchmark.

Implements a manual per-timestep OLTC controller that reads the MV busbar
voltage and adjusts the HV/MV transformer tap position to keep the controlled
bus within a deadband.  No Arduino, no reactive power control from DERs.

Control law
-----------
The OLTC monitors the LV-side busbar of the HV/MV transformer group.
One tap step is issued per timestep when the monitored voltage leaves the
control band [OLTC_CONTROL_LOWER, OLTC_CONTROL_UPPER].  The sign of the tap
command is calibrated once at startup by probing the network response — no
reliance on tap_side string parsing.

Parallel transformer handling
------------------------------
SimBench MV rural uses two parallel HV/MV transformers.  This scenario
applies synchronised (ganged) tap control: both transformers always share the
same tap position.  This represents the settled state of a master-follower
OLTC scheme.  Transient follower lag and ±1 tap mismatch protection logic are
outside the quasi-static benchmark scope.

Rollback on post-tap non-convergence
--------------------------------------
If the post-tap runpp diverges, the tap command is rejected and the group is
restored to the previous validated position.  The timestep is logged with
tap_blocked_reason = "post_pf_non_convergence" and tap_changed = False.

Usage
-----
    from profile_builder import build_profiles
    from scenario_result import adapt_profiles
    from scenario_2_oltc import run_scenario_2

    net      = sb.get_simbench_net("1-MV-rural--2-sw")
    profiles = build_profiles(net, ...)
    result   = run_scenario_2(net, profiles)

    print(result.n_violation_steps)
    df = pd.DataFrame([result.summary_dict()])

Notes
-----
voltage_depend_loads=False is mandatory for all runpp() calls on SimBench
networks (pandapower 3.2.0+ singular matrix without it).

net.sgen.q_mvar is forced to 0.0 every timestep.  Scenario 2 is a pure
OLTC benchmark — DERs inject no reactive power.
"""

from __future__ import annotations

import copy
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
    LINE_MAX_LOADING,
    TRAFO_MAX_LOADING,
    VOLTAGE_EPSILON,
    LOADING_EPSILON,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLTC_CONTROL_LOWER:       float = 0.98   # OLTC deadband lower bound (pu)
OLTC_CONTROL_UPPER:       float = 1.02   # OLTC deadband upper bound (pu)
MAX_TAP_STEPS_PER_ACTION: int   = 1      # tap steps per control action


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_oltc_trafos(net) -> pd.Index:
    """
    Select the transformer(s) to be controlled by the OLTC.

    Uses a three-tier fallback so the runner works across MV and LV networks:

    Tier 1 — HV/MV trafos (vn_hv_kv >= 66 kV).
        Catches SimBench MV (110 kV) and CIGRE MV (110 kV).

    Tier 2 — Trafo whose hv_bus is the ext_grid slack bus.
        Catches MV/LV head trafos in LV-only networks (Kerber, Dickert,
        Synthetic LV, CIGRE LV) where vn_hv_kv is typically 10–20 kV.

    Tier 3 — Highest HV voltage level (last resort).
        Catches any remaining network topology where neither tier applies.

    Only in-service trafos with non-NaN tap metadata are considered at every
    tier.  Ganging validation is performed by _validate_tap_metadata() after
    this function returns.

    Raises
    ------
    ValueError
        If no eligible trafo is found after all three tiers.
    """
    required_cols = ["tap_min", "tap_max", "tap_neutral", "tap_step_percent"]
    candidates = net.trafo[net.trafo["in_service"] == True].copy()
    candidates = candidates.dropna(subset=required_cols)

    if candidates.empty:
        raise ValueError(
            "_select_oltc_trafos: no in-service transformer with valid tap "
            "metadata found in net.trafo."
        )

    # Tier 1: HV/MV
    hv_mv = candidates[candidates["vn_hv_kv"] >= 66]
    if not hv_mv.empty:
        logger.debug(
            "_select_oltc_trafos: Tier 1 — %d HV/MV trafo(s) selected.",
            len(hv_mv),
        )
        return hv_mv.index

    # Tier 2: slack-connected
    slack_buses = set(net.ext_grid["bus"].values)
    slack_conn  = candidates[candidates["hv_bus"].isin(slack_buses)]
    if not slack_conn.empty:
        logger.debug(
            "_select_oltc_trafos: Tier 2 — %d slack-connected trafo(s) selected.",
            len(slack_conn),
        )
        return slack_conn.index

    # Tier 3: highest HV voltage
    max_hv   = candidates["vn_hv_kv"].max()
    fallback = candidates[candidates["vn_hv_kv"] == max_hv]
    logger.warning(
        "_select_oltc_trafos: Tier 3 fallback — selecting %d trafo(s) at "
        "vn_hv_kv=%.1f kV.  Verify this is the intended OLTC group.",
        len(fallback), max_hv,
    )
    return fallback.index


def _validate_tap_metadata(net, trafo_idx: pd.Index) -> None:
    """
    Validate that the selected transformer group can be ganged.

    Checks
    ------
    - All required tap columns are non-NaN (redundant after _select_oltc_trafos
      but kept as an explicit guard).
    - All trafos share the same tap_neutral (ganging requires one reference).
    - All trafos share the same tap_side (ganging requires consistent direction).
    - Ganged tap range [max(tap_min), min(tap_max)] is non-empty.

    Raises
    ------
    ValueError on any failed check.
    """
    required_cols = ["tap_min", "tap_max", "tap_neutral",
                     "tap_step_percent", "tap_side"]
    for col in required_cols:
        if net.trafo.loc[trafo_idx, col].isna().any():
            raise ValueError(
                f"_validate_tap_metadata: NaN in '{col}' for trafo(s) "
                f"{trafo_idx.tolist()}."
            )

    neutrals = net.trafo.loc[trafo_idx, "tap_neutral"].unique()
    if len(neutrals) > 1:
        raise ValueError(
            f"_validate_tap_metadata: ganged trafos must share tap_neutral. "
            f"Found: {neutrals.tolist()}."
        )

    sides = net.trafo.loc[trafo_idx, "tap_side"].unique()
    if len(sides) > 1:
        raise ValueError(
            f"_validate_tap_metadata: ganged trafos must share tap_side. "
            f"Found: {sides.tolist()}."
        )

    tap_min_gang = int(net.trafo.loc[trafo_idx, "tap_min"].max())
    tap_max_gang = int(net.trafo.loc[trafo_idx, "tap_max"].min())
    if tap_min_gang > tap_max_gang:
        raise ValueError(
            f"_validate_tap_metadata: ganged tap range is infeasible. "
            f"max(tap_min)={tap_min_gang} > min(tap_max)={tap_max_gang}."
        )


def _select_controlled_bus(net, trafo_idx: pd.Index) -> int:
    """
    Return the bus index monitored by the OLTC controller.

    The OLTC regulates the LV-side (secondary) busbar of the HV/MV trafo
    group.  All parallel trafos in the gang must share the same lv_bus —
    validated here.

    Raises
    ------
    ValueError if the trafos share different lv_buses.
    """
    lv_buses = net.trafo.loc[trafo_idx, "lv_bus"].unique()
    if len(lv_buses) != 1:
        raise ValueError(
            f"_select_controlled_bus: ganged trafos must share the same "
            f"lv_bus.  Found: {lv_buses.tolist()}."
        )
    return int(lv_buses[0])


def _calibrate_tap_sign(
        net,
        trafo_idx:    pd.Index,
        ctrl_bus:     int,
        tap_neutral:  int,
        tap_min_gang: int,
        tap_max_gang: int,
) -> int:
    """
    Determine the sign convention for the tap controller by network probing.

    Runs two power flows on a deep copy of the network — one at tap_neutral,
    one at tap_neutral ± 1 — and observes the voltage change at ctrl_bus.
    Returns the sign such that:

        new_tap = current_tap + sign * MAX_TAP_STEPS_PER_ACTION

    always moves the controlled bus voltage *down* (used when overvoltage is
    detected).

    Using a deep copy ensures the working net's res_bus is never touched by
    calibration and no stale results are left behind.

    Raises
    ------
    ValueError  if the tap range is too narrow to probe (tap_neutral is the
                only available position).
    RuntimeError if the calibration power flows do not converge.
    """
    probe_net = copy.deepcopy(net)

    # Choose probe direction within the ganged range
    if tap_neutral + 1 <= tap_max_gang:
        probe_pos = tap_neutral + 1
        direction = +1
    elif tap_neutral - 1 >= tap_min_gang:
        probe_pos = tap_neutral - 1
        direction = -1
    else:
        raise ValueError(
            "_calibrate_tap_sign: tap range too narrow to probe sign "
            f"(tap_min_gang={tap_min_gang}, tap_max_gang={tap_max_gang}, "
            f"tap_neutral={tap_neutral})."
        )

    # Baseline at neutral
    probe_net.trafo.loc[trafo_idx, "tap_pos"] = tap_neutral
    try:
        pp.runpp(probe_net, voltage_depend_loads=False)
    except Exception as exc:
        raise RuntimeError(
            "_calibrate_tap_sign: baseline runpp at tap_neutral failed."
        ) from exc
    v_neutral = float(probe_net.res_bus.at[ctrl_bus, "vm_pu"])

    # Probe at neutral + direction
    probe_net.trafo.loc[trafo_idx, "tap_pos"] = probe_pos
    try:
        pp.runpp(probe_net, voltage_depend_loads=False)
    except Exception as exc:
        raise RuntimeError(
            "_calibrate_tap_sign: probe runpp at tap_pos="
            f"{probe_pos} failed."
        ) from exc
    v_probe = float(probe_net.res_bus.at[ctrl_bus, "vm_pu"])

    logger.debug(
        "_calibrate_tap_sign: v_neutral=%.5f  v_probe=%.5f  "
        "probe_pos=%d  direction=%+d",
        v_neutral, v_probe, probe_pos, direction,
    )

    # Sign = direction that lowers ctrl_bus voltage
    if direction == +1:
        sign = +1 if v_probe < v_neutral else -1
    else:
        sign = -1 if v_probe < v_neutral else +1

    logger.info(
        "_calibrate_tap_sign: tap sign = %+d  "
        "(tap+1 %s LV voltage by %.5f pu)",
        sign,
        "lowers" if v_probe < v_neutral else "raises",
        abs(v_probe - v_neutral),
    )
    return sign


def _empty_series() -> pd.Series:
    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_scenario_2(
        net,
        profiles:   dict,
        network_id: str   = "unknown",
        v_min:      float = V_MIN,
        v_max:      float = V_MAX,
) -> ScenarioResult:
    """
    Run Scenario 2 — OLTC-only voltage regulation benchmark.

    Parameters
    ----------
    net        : pandapower network.  Modified in place every timestep.
                 Caller should deep-copy if the original net is needed later.
    profiles   : dict from profile_builder.build_profiles().
    network_id : human-readable identifier stored in ScenarioResult.
    v_min      : lower voltage planning limit (pu).  Default 0.95.
    v_max      : upper voltage planning limit (pu).  Default 1.05.

    Returns
    -------
    ScenarioResult  with scenario_id="oltc".
    """
    t_start = time.perf_counter()

    ap: AdaptedProfiles = adapt_profiles(net, profiles)
    time_steps = range(len(ap.times))

    # ------------------------------------------------------------------
    # Reset controllers and results from any prior scenario on this net
    # ------------------------------------------------------------------
    net.controller.drop(net.controller.index, inplace=True)
    pp.reset_results(net)

    # ------------------------------------------------------------------
    # Trafo setup — select, validate, calibrate
    # ------------------------------------------------------------------
    trafo_idx    = _select_oltc_trafos(net)
    _validate_tap_metadata(net, trafo_idx)

    tap_min_gang = int(net.trafo.loc[trafo_idx, "tap_min"].max())
    tap_max_gang = int(net.trafo.loc[trafo_idx, "tap_max"].min())
    tap_neutral  = int(net.trafo.loc[trafo_idx, "tap_neutral"].iloc[0])
    ctrl_bus     = _select_controlled_bus(net, trafo_idx)
    sign         = _calibrate_tap_sign(
        net, trafo_idx, ctrl_bus,
        tap_neutral, tap_min_gang, tap_max_gang,
    )

    # Initialise at neutral tap
    current_tap = tap_neutral
    net.trafo.loc[trafo_idx, "tap_pos"] = current_tap

    logger.info(
        "[Scenario 2 | %s] OLTC setup: %d trafo(s), ctrl_bus=%d, "
        "tap_range=[%d, %d], tap_neutral=%d, sign=%+d | "
        "%d timesteps, %d DERs, %d loads",
        network_id, len(trafo_idx), ctrl_bus,
        tap_min_gang, tap_max_gang, tap_neutral, sign,
        len(time_steps),
        len(ap.der_p.columns), len(ap.load_idx),
    )

    records: list[TimestepRecord] = []

    # ------------------------------------------------------------------
    # Timestep loop
    # ------------------------------------------------------------------
    for t in time_steps:
        timestamp = ap.times[t]

        # [1] Write profiles (index-explicit)
        if not ap.load_p.empty:
            net.load.loc[ap.load_idx, "p_mw"]   = ap.load_p.iloc[t].values
            net.load.loc[ap.load_idx, "q_mvar"] = ap.load_q.iloc[t].values

        if not ap.der_p.empty:
            net.sgen.loc[ap.der_p.columns, "p_mw"]   = ap.der_p.iloc[t].values
            net.sgen.loc[ap.der_p.columns, "q_mvar"] = 0.0

        # [2] Pre-action power flow at current tap
        try:
            pp.runpp(net, voltage_depend_loads=False)
            converged_pre = True
        except Exception:
            converged_pre = False

        # [3] Non-convergence: hold tap, log blocked reason, advance
        if not converged_pre:
            logger.warning(
                "[Scenario 2 | %s] t=%d: pre-PF diverged — tap held at %d.",
                network_id, t, current_tap,
            )
            records.append(TimestepRecord(
                t=t, timestamp=timestamp,
                vm_pu=_empty_series(),
                line_loading=_empty_series(),
                trafo_loading=_empty_series(),
                over_voltage_buses=[], under_voltage_buses=[],
                overloaded_lines=[], overloaded_trafos=[],
                q_applied_mvar=None, p_applied_mw=None,
                p_target_mw=None, curtailment_needed=False,
                converged=False,
                tap_pos=current_tap,
                tap_changed=False,
                tap_attempted=False,
                tap_candidate=None,
                post_pf_reused=False,
                tap_blocked_reason="pre_pf_non_convergence",
            ))
            continue

        # [4] OLTC decision — read controlled bus voltage
        vm_ctrl = float(net.res_bus.at[ctrl_bus, "vm_pu"])

        if vm_ctrl > OLTC_CONTROL_UPPER:
            tap_candidate = int(np.clip(
                current_tap + sign * MAX_TAP_STEPS_PER_ACTION,
                tap_min_gang, tap_max_gang,
            ))
            tap_attempted = True
        elif vm_ctrl < OLTC_CONTROL_LOWER:
            tap_candidate = int(np.clip(
                current_tap - sign * MAX_TAP_STEPS_PER_ACTION,
                tap_min_gang, tap_max_gang,
            ))
            tap_attempted = True
        else:
            tap_candidate = None
            tap_attempted = False

        # [5] Apply tap if attempted and different from current
        if tap_attempted and tap_candidate != current_tap:
            prev_tap = current_tap
            net.trafo.loc[trafo_idx, "tap_pos"] = tap_candidate

            # [5a] Post-tap power flow
            try:
                pp.runpp(net, voltage_depend_loads=False)
                converged_post = True
            except Exception:
                converged_post = False

            if converged_post:
                # Accept tap move
                current_tap    = tap_candidate
                tap_changed    = True
                post_pf_reused = False
                blocked_reason = None
                logger.debug(
                    "[Scenario 2 | %s] t=%d: tap %d → %d | vm_ctrl=%.4f",
                    network_id, t, prev_tap, current_tap, vm_ctrl,
                )
            else:
                # Rollback: restore previous tap and re-run
                logger.warning(
                    "[Scenario 2 | %s] t=%d: post-tap PF diverged at "
                    "candidate=%d — rolling back to %d.",
                    network_id, t, tap_candidate, prev_tap,
                )
                net.trafo.loc[trafo_idx, "tap_pos"] = prev_tap
                current_tap    = prev_tap
                tap_changed    = False
                post_pf_reused = False
                blocked_reason = "post_pf_non_convergence"

                # Attempt rollback PF to restore valid res_bus
                try:
                    pp.runpp(net, voltage_depend_loads=False)
                    converged_post = True
                except Exception:
                    converged_post = False
                    logger.error(
                        "[Scenario 2 | %s] t=%d: rollback PF also diverged.",
                        network_id, t,
                    )

        elif tap_attempted and tap_candidate == current_tap:
            # Tap rail reached — controller wanted to move but tap is already
            # at the ganged range limit.  Reuse pre-action results.
            converged_post = True
            tap_changed    = False
            post_pf_reused = True
            blocked_reason = "tap_limit_reached"
            logger.debug(
                "[Scenario 2 | %s] t=%d: tap rail at %d — "
                "tap_limit_reached (vm_ctrl=%.4f).",
                network_id, t, current_tap, vm_ctrl,
            )

        else:
            # No tap movement — deadband inactive, reuse pre-action results
            converged_post = True
            tap_changed    = False
            blocked_reason = None
            post_pf_reused = True

        # [6] Build TimestepRecord from settled state
        converged = converged_post

        if converged:
            vm  = net.res_bus["vm_pu"].copy()
            ll  = net.res_line["loading_percent"].copy()
            tl  = net.res_trafo["loading_percent"].copy()

            ov_buses  = vm.index[vm > v_max + VOLTAGE_EPSILON].tolist()
            uv_buses  = vm.index[vm < v_min - VOLTAGE_EPSILON].tolist()
            ov_lines  = ll.index[ll > LINE_MAX_LOADING  + LOADING_EPSILON].tolist()
            ov_trafos = tl.index[tl > TRAFO_MAX_LOADING + LOADING_EPSILON].tolist()
        else:
            vm = ll = tl = _empty_series()
            ov_buses = uv_buses = ov_lines = ov_trafos = []

        records.append(TimestepRecord(
            t=t, timestamp=timestamp,
            vm_pu=vm, line_loading=ll, trafo_loading=tl,
            over_voltage_buses=ov_buses, under_voltage_buses=uv_buses,
            overloaded_lines=ov_lines, overloaded_trafos=ov_trafos,
            q_applied_mvar=None,
            p_applied_mw=None,
            p_target_mw=None,
            curtailment_needed=False,
            converged=converged,
            tap_pos=current_tap,
            tap_changed=tap_changed,
            tap_attempted=tap_attempted,
            tap_candidate=tap_candidate,
            post_pf_reused=post_pf_reused,
            tap_blocked_reason=blocked_reason,
        ))

        if t % 96 == 0:
            logger.info(
                "[Scenario 2 | %s] t=%d/%d (%.1f%%) | tap=%d | "
                "violations=%d",
                network_id, t, len(time_steps),
                100.0 * t / max(len(time_steps), 1),
                current_tap,
                sum(1 for r in records
                    if r.over_voltage_buses or r.under_voltage_buses),
            )

    # ------------------------------------------------------------------
    # Build ScenarioResult
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_start

    result = ScenarioResult.from_records(
        scenario_id = "oltc",
        network_id  = network_id,
        records     = records,
        elapsed_s   = elapsed,
        dt_s        = ap.dt_s,
    )

    # Tap movement summary
    tap_moves    = sum(1 for r in records if r.tap_changed)
    tap_blocks   = sum(1 for r in records if r.tap_blocked_reason)
    tap_min_seen = min(
        (r.tap_pos for r in records if r.tap_pos is not None),
        default=tap_neutral,
    )
    tap_max_seen = max(
        (r.tap_pos for r in records if r.tap_pos is not None),
        default=tap_neutral,
    )

    logger.info(
        "[Scenario 2 | %s] Done. %.1f s | %d/%d converged | "
        "%d violation steps | %d tap moves | %d blocked | "
        "tap range seen [%d, %d]",
        network_id, elapsed,
        result.n_converged, result.n_timesteps,
        result.n_violation_steps,
        tap_moves, tap_blocks,
        tap_min_seen, tap_max_seen,
    )
    return result
