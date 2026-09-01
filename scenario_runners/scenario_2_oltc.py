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
TAP_RESPONSE_EPSILON:     float = 1e-5   # minimum vm_pu delta to confirm tap is electrically active

# Conservative benchmark defaults used only when a selected transformer
# lacks tap metadata.  Existing valid network values are never overwritten.
# A future LV OLTC extension should define its own fill dict.
DEFAULT_TAP_METADATA_FILL: dict = {
    "tap_changer_type":     "Ratio",
    "tap_dependency_table": False,
    "tap_side":             "lv",
    "tap_neutral":          0,
    "tap_min":             -9,
    "tap_max":              9,
    "tap_pos":              0,
    "tap_step_percent":     1.5,
    "tap_step_degree":      0.0,
}

# Columns printed by _print_tap_metadata() for audit logging.
_TAP_PRINT_COLS: list[str] = [
    "name", "vn_hv_kv", "vn_lv_kv", "hv_bus", "lv_bus",
    "tap_side", "tap_changer_type", "tap_dependency_table",
    "tap_neutral", "tap_min", "tap_max", "tap_pos",
    "tap_step_percent", "tap_step_degree",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_missing_tap_value(value) -> bool:
    """
    Return True if a tap metadata field is absent or placeholder.

    pd.isna() alone misses cases where pandapower stores the string
    "None" or "nan" in a mixed-type column.  The string fallback
    covers those cases.
    """
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "none", "nan", "<na>"}


def _print_tap_metadata(net, trafo_idx: pd.Index, title: str) -> None:
    """
    Log tap metadata for the selected trafos at INFO level.

    Prints only columns that exist in net.trafo from _TAP_PRINT_COLS
    so the function does not crash on networks missing optional columns.
    Called twice in run_scenario_2: once before and once after
    completion/override so the audit trail shows exactly what changed.
    """
    available = [c for c in _TAP_PRINT_COLS if c in net.trafo.columns]
    subset    = net.trafo.loc[trafo_idx, available]
    logger.info(
        "[OLTC] %s:\n%s",
        title,
        subset.to_string(),
    )


def _complete_missing_tap_metadata(
        net,
        trafo_idx:  pd.Index,
        network_id: str,
        defaults:   dict,
) -> None:
    """
    Fill only missing or inactive placeholder tap metadata fields from defaults.

    Existing valid network values are preserved.  A field is filled only
    when _is_missing_tap_value() returns True for its current value.
    Every fill is logged at WARNING level with column, trafo index, and
    the value that was applied so the audit trail is unambiguous.
    """
    filled: dict = {}

    for col, default_value in defaults.items():
        if col not in net.trafo.columns:
            net.trafo[col] = pd.NA

        for idx in trafo_idx:
            current = net.trafo.at[idx, col]
            if _is_missing_tap_value(current):
                net.trafo.at[idx, col] = default_value
                filled.setdefault(col, {})[idx] = default_value

    if filled:
        logger.warning(
            "[Scenario 2 | %s] Completed missing OLTC tap metadata "
            "from defaults: %s",
            network_id, filled,
        )
    else:
        logger.info(
            "[Scenario 2 | %s] No tap metadata fields needed completion "
            "(all fields already valid in the network).",
            network_id,
        )


def _apply_user_tap_metadata_override(
        net,
        trafo_idx:     pd.Index,
        network_id:    str,
        user_override: dict | None,
) -> None:
    """
    Apply explicit user-provided tap metadata overrides.

    Only fields present in user_override are written.  This overwrites
    both the original network value and any default-completion value.
    Called only when the caller explicitly provides tap_metadata_override;
    a None or empty dict is a no-op.
    """
    if not user_override:
        return

    logger.warning(
        "[Scenario 2 | %s] Applying user-provided OLTC tap metadata "
        "override: %s",
        network_id, user_override,
    )

    for col, value in user_override.items():
        if col not in net.trafo.columns:
            net.trafo[col] = pd.NA
        net.trafo.loc[trafo_idx, col] = value


def _select_oltc_trafos(net) -> pd.Index:
    """
    Select the transformer(s) to be controlled by the OLTC.

    Selection is topology-based only — tap metadata is NOT required at
    this stage.  Missing fields are completed and validated later in the
    7-step setup sequence.  This separation allows the completion step
    to fill metadata for networks (CIGRE, Kerber, Dickert, custom) that
    may have valid transformers with absent tap_changer_type etc.

    Three-tier fallback (same topology logic as before):

    Tier 1 — HV/MV trafos (vn_hv_kv >= 66 kV).
        Catches SimBench MV (110 kV) and CIGRE MV (110 kV).

    Tier 2 — Trafo whose hv_bus is the ext_grid slack bus.
        Catches MV/LV head trafos in LV-only networks (Kerber, Dickert,
        Synthetic LV, CIGRE LV) where vn_hv_kv is typically 10–20 kV.

    Tier 3 — Highest HV voltage level (last resort).
        Catches any remaining topology where neither tier applies.

    Raises
    ------
    ValueError
        If no eligible trafo is found after all three tiers, or if
        required topology columns are absent from net.trafo.
    """
    if "in_service" in net.trafo.columns:
        candidates = net.trafo[net.trafo["in_service"] == True].copy()
    else:
        candidates = net.trafo.copy()

    if candidates.empty:
        raise ValueError(
            "_select_oltc_trafos: no in-service transformer found in net.trafo."
        )

    # Tier 1: HV/MV
    if "vn_hv_kv" in candidates.columns:
        hv_mv = candidates[candidates["vn_hv_kv"] >= 66]
        if not hv_mv.empty:
            logger.debug(
                "_select_oltc_trafos: Tier 1 — %d HV/MV trafo(s) selected.",
                len(hv_mv),
            )
            return hv_mv.index

    # Tier 2: slack-connected
    if not net.ext_grid.empty and "hv_bus" in candidates.columns:
        slack_buses = set(net.ext_grid["bus"].values)
        slack_conn  = candidates[candidates["hv_bus"].isin(slack_buses)]
        if not slack_conn.empty:
            logger.debug(
                "_select_oltc_trafos: Tier 2 — %d slack-connected trafo(s) selected.",
                len(slack_conn),
            )
            return slack_conn.index

    # Tier 3: highest HV voltage
    if "vn_hv_kv" in candidates.columns:
        max_hv   = candidates["vn_hv_kv"].max()
        fallback = candidates[candidates["vn_hv_kv"] == max_hv]
        logger.warning(
            "_select_oltc_trafos: Tier 3 fallback — selecting %d trafo(s) at "
            "vn_hv_kv=%.1f kV.  Verify this is the intended OLTC group.",
            len(fallback), max_hv,
        )
        return fallback.index

    raise ValueError(
        "_select_oltc_trafos: could not select OLTC transformer group. "
        "net.trafo lacks required topology columns such as hv_bus or vn_hv_kv."
    )


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
    # tap_changer_type is now a required field (completion fills it;
    # auto_complete_tap_metadata=False users must supply it themselves).
    required_cols = [
        "tap_min", "tap_max", "tap_neutral",
        "tap_step_percent", "tap_side", "tap_changer_type",
    ]
    for col in required_cols:
        if col not in net.trafo.columns or net.trafo.loc[trafo_idx, col].isna().any():
            raise ValueError(
                f"_validate_tap_metadata: missing or NaN in required "
                f"field '{col}' for trafo(s) {trafo_idx.tolist()}. "
                f"Enable auto_complete_tap_metadata or supply via override."
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

    # tap_neutral must lie inside each trafo's own tap range
    for idx in trafo_idx:
        tn = int(net.trafo.at[idx, "tap_neutral"])
        tmin = int(net.trafo.at[idx, "tap_min"])
        tmax = int(net.trafo.at[idx, "tap_max"])
        if not (tmin <= tn <= tmax):
            raise ValueError(
                f"_validate_tap_metadata: tap_neutral={tn} is outside "
                f"[tap_min={tmin}, tap_max={tmax}] for trafo {idx}."
            )

    # tap_step_percent: numeric, finite, nonzero, shared across gang
    step_pcts = pd.to_numeric(
        net.trafo.loc[trafo_idx, "tap_step_percent"],
        errors="coerce",
    )
    if step_pcts.isna().any():
        raise ValueError(
            "_validate_tap_metadata: tap_step_percent contains non-numeric "
            f"values: {net.trafo.loc[trafo_idx, 'tap_step_percent'].to_dict()}."
        )
    if (step_pcts.abs() < 1e-12).any():
        raise ValueError(
            "_validate_tap_metadata: tap_step_percent must be nonzero. "
            f"Found: {step_pcts.to_dict()}."
        )
    if step_pcts.round(9).nunique() > 1:
        raise ValueError(
            "_validate_tap_metadata: ganged trafos must share tap_step_percent. "
            f"Found: {step_pcts.to_dict()}."
        )

    # tap_changer_type: must be "Ratio" (case-insensitive), shared across gang
    types = (
        net.trafo.loc[trafo_idx, "tap_changer_type"]
        .astype(str).str.strip().str.lower()
        .unique()
    )
    if len(types) > 1:
        raise ValueError(
            "_validate_tap_metadata: ganged trafos must share tap_changer_type. "
            f"Found: {types.tolist()}."
        )
    if types[0] != "ratio":
        raise ValueError(
            "_validate_tap_metadata: tap_changer_type must be 'Ratio' for "
            f"an amplitude-only OLTC. Found: {types[0]!r}."
        )

    # tap_dependency_table must be False (no LTC lookup table used)
    if "tap_dependency_table" in net.trafo.columns:
        dep = net.trafo.loc[trafo_idx, "tap_dependency_table"]
        bad = dep[dep.astype(bool) == True]
        if not bad.empty:
            raise ValueError(
                f"_validate_tap_metadata: tap_dependency_table must be "
                f"False for all trafos in the OLTC group. "
                f"Found True for trafo(s): {bad.index.tolist()}."
            )


def _select_controlled_buses(net, trafo_idx: pd.Index) -> list[int]:
    """
    Return the sorted list of secondary (MV) bus indices for OLTC measurement.

    Scope
    -----
    The current OLTC implementation assumes one upstream regulated substation
    per selected distribution network.  Multiple HV or MV busbar sections
    inside that substation are allowed and handled through a mean
    controlled-bus voltage across all secondary busbars of the OLTC group.

    A future multi-substation implementation would need to group trafos by
    network topology rather than by raw bus index.  That is outside the
    current benchmark scope.

    Validation
    ----------
    - Raises ValueError if no LV-side buses are found (degenerate selection).
    - Raises ValueError if the secondary buses span different voltage levels,
      which would indicate structurally incompatible trafos in the group.
    - Issues a WARNING (not an error) when the group spans multiple HV busbar
      sections; this is valid for split-busbar substation models such as
      SimBench 1-MV-rural--2-sw and treated as one synchronised OLTC group.

    Returns
    -------
    list[int]
        Sorted unique LV bus indices.  Length 1 for single-transformer or
        common-secondary-bus substations.  Length > 1 for split-busbar.
    """
    lv_buses = sorted(
        int(b) for b in net.trafo.loc[trafo_idx, "lv_bus"].unique()
    )

    if not lv_buses:
        raise ValueError(
            "_select_controlled_buses: no LV-side buses found for the "
            "selected OLTC trafos."
        )

    vn_levels = net.bus.loc[lv_buses, "vn_kv"].round(6).unique()
    if len(vn_levels) != 1:
        raise ValueError(
            "_select_controlled_buses: selected OLTC secondary buses operate "
            f"at different voltage levels: "
            f"{dict(net.bus.loc[lv_buses, 'vn_kv'])}. "
            f"Cannot form a single mean-voltage measurement signal."
        )

    hv_buses = sorted(
        int(b) for b in net.trafo.loc[trafo_idx, "hv_bus"].unique()
    )
    if len(hv_buses) > 1:
        logger.warning(
            "_select_controlled_buses: selected trafos connect to multiple "
            "HV busbar sections %s. This is allowed as a split-busbar "
            "substation model; the OLTC group remains synchronised.",
            hv_buses,
        )

    logger.info(
        "_select_controlled_buses: controlled secondary busbar group = %s.",
        lv_buses,
    )
    return lv_buses


def _calibrate_tap_sign(
        net,
        trafo_idx:    pd.Index,
        ctrl_buses:   list[int],
        tap_neutral:  int,
        tap_min_gang: int,
        tap_max_gang: int,
) -> int:
    """
    Determine the sign convention for the tap controller by network probing.

    Runs two power flows on a deep copy of the network — one at tap_neutral,
    one at tap_neutral ± 1 — and observes the mean voltage change across
    ctrl_buses (all secondary busbars of the OLTC group).
    Returns the sign such that:

        new_tap = current_tap + sign * MAX_TAP_STEPS_PER_ACTION

    always moves the mean controlled voltage *down* (used when overvoltage
    is detected).

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
    v_neutral = float(probe_net.res_bus.loc[ctrl_buses, "vm_pu"].mean())

    # Probe at neutral + direction
    probe_net.trafo.loc[trafo_idx, "tap_pos"] = probe_pos
    try:
        pp.runpp(probe_net, voltage_depend_loads=False)
    except Exception as exc:
        raise RuntimeError(
            "_calibrate_tap_sign: probe runpp at tap_pos="
            f"{probe_pos} failed."
        ) from exc
    v_probe = float(probe_net.res_bus.loc[ctrl_buses, "vm_pu"].mean())

    delta_v = v_probe - v_neutral

    logger.debug(
        "_calibrate_tap_sign: v_neutral=%.8f  v_probe=%.8f  "
        "delta_v=%+.8f  probe_pos=%d  direction=%+d",
        v_neutral, v_probe, delta_v, probe_pos, direction,
    )

    # Guard: negligible response means the tap is electrically inactive.
    # Metadata can pass validation and still produce no voltage change —
    # e.g. tap_side mismatch or a disabled regulator.  Fail explicitly
    # with the full tap metadata so the user can diagnose the root cause.
    if abs(delta_v) < TAP_RESPONSE_EPSILON:
        tap_meta_cols = [
            "hv_bus", "lv_bus", "tap_pos", "tap_neutral",
            "tap_min", "tap_max", "tap_step_percent",
            "tap_step_degree", "tap_side", "tap_changer_type",
            "tap_dependency_table",
        ]
        available = [c for c in tap_meta_cols if c in net.trafo.columns]
        tap_meta  = net.trafo.loc[trafo_idx, available]
        raise RuntimeError(
            "_calibrate_tap_sign: tap probe produced negligible voltage "
            f"response abs(delta_v)={abs(delta_v):.2e} pu < "
            f"{TAP_RESPONSE_EPSILON:.1e} pu. "
            "Check tap_changer_type, tap_side, selected transformers, "
            "and controlled buses.\n"
            f"{tap_meta.to_string()}"
        )

    # Sign = direction that lowers mean controlled-bus voltage
    if direction == +1:
        sign = +1 if delta_v < 0.0 else -1
    else:
        sign = -1 if delta_v < 0.0 else +1

    logger.info(
        "_calibrate_tap_sign: tap sign = %+d  "
        "(probe direction %+d changes controlled voltage by %+.8f pu)",
        sign, direction, delta_v,
    )
    return sign


def _empty_series() -> pd.Series:
    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_scenario_2(
        net,
        profiles:                   dict,
        network_id:                 str        = "unknown",
        v_min:                      float      = V_MIN,
        v_max:                      float      = V_MAX,
        auto_complete_tap_metadata: bool       = True,
        tap_metadata_override:      dict | None = None,
        publish_fn                              = None,
        enable_checkpointing:       bool       = True,
        live_csv_rewrite_fn = None,
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
    auto_complete_tap_metadata : bool, default True.
        If True, fill any missing or invalid tap metadata fields from
        DEFAULT_TAP_METADATA_FILL before validation.  Existing valid
        network values are never overwritten.  Set False to run on the
        raw network data only (will fail validation if fields are absent).
    tap_metadata_override : dict or None, default None.
        Explicit field overrides applied after auto-completion.  Overwrites
        both the original network value and any auto-completed value.
        Intended for CLI or BenchmarkConfig to inject user-specified tap
        parameters without modifying the network object before calling.
        Example: {"tap_side": "hv", "tap_step_percent": 1.25}.

    Returns
    -------
    ScenarioResult  with scenario_id="oltc".
    """
    t_start = time.perf_counter()
    ap: AdaptedProfiles = adapt_profiles(net, profiles)
    _T = len(ap.times)
    if publish_fn is not None:
        publish_fn.on_scenario_start("oltc", "OLTC-only", _T)
    time_steps = range(len(ap.times))

    resumed_records: list[TimestepRecord] = []
    if publish_fn is not None and enable_checkpointing:
        resumed_records = publish_fn.get_resume_records("oltc")
    start_t = (resumed_records[-1].t + 1) if resumed_records else 0
    _T_full = len(ap.times)

    if start_t >= len(ap.times) and resumed_records:
        logger.info(
            "[Scenario 2 | %s] Checkpoint already covers all %d steps — skipping simulation.",
            network_id, len(ap.times),
        )
        elapsed = publish_fn.cumulative_elapsed_s() if publish_fn is not None else time.perf_counter() - t_start
        result = ScenarioResult.from_records(
            scenario_id="oltc", network_id=network_id,
            records=resumed_records, elapsed_s=elapsed, dt_s=ap.dt_s,
        )
        if publish_fn is not None:
            publish_fn.on_scenario_end(result)
        return result
    if start_t > 0:
        logger.info("[Scenario 2 | %s] Resuming from t=%d/%d.", network_id, start_t, len(ap.times))
    time_steps = range(start_t, len(ap.times))

    # ------------------------------------------------------------------
    # Reset controllers and results from any prior scenario on this net
    # ------------------------------------------------------------------
    net.controller.drop(net.controller.index, inplace=True)
    pp.reset_results(net)

    # ------------------------------------------------------------------
    # Trafo setup — 7-step sequence
    # Step 1: select; Step 2: print original; Step 3: complete missing;
    # Step 4: apply user override; Step 5: print final; Step 6: validate;
    # Step 7: calibrate tap sign by network probing.
    # ------------------------------------------------------------------

    # [1] Select OLTC transformer group
    trafo_idx = _select_oltc_trafos(net)

    # [2] Print original metadata — audit trail before any modification
    _print_tap_metadata(
        net, trafo_idx,
        title="Original selected transformer tap metadata",
    )

    # [3] Complete missing or invalid fields from safe defaults
    if auto_complete_tap_metadata:
        _complete_missing_tap_metadata(
            net, trafo_idx, network_id, DEFAULT_TAP_METADATA_FILL
        )

    # [4] Apply explicit user / CLI overrides
    _apply_user_tap_metadata_override(
        net, trafo_idx, network_id, tap_metadata_override
    )

    # [5] Print final metadata — shows what completion/override changed
    _print_tap_metadata(
        net, trafo_idx,
        title="Final selected transformer tap metadata after completion/override",
    )

    # [6] Validate — raises ValueError on any structural problem
    _validate_tap_metadata(net, trafo_idx)

    # [7] Derive gang parameters and calibrate tap sign
    tap_min_gang = int(net.trafo.loc[trafo_idx, "tap_min"].max())
    tap_max_gang = int(net.trafo.loc[trafo_idx, "tap_max"].min())
    tap_neutral  = int(net.trafo.loc[trafo_idx, "tap_neutral"].iloc[0])
    ctrl_buses   = _select_controlled_buses(net, trafo_idx)
    sign         = _calibrate_tap_sign(
        net, trafo_idx, ctrl_buses,
        tap_neutral, tap_min_gang, tap_max_gang,
    )

    # Initialise tap position — from the last checkpoint record if resuming,
    # otherwise at neutral as before.
    if resumed_records and resumed_records[-1].tap_pos is not None:
        current_tap = resumed_records[-1].tap_pos
        logger.info(
            "[Scenario 2 | %s] Resuming with tap_pos=%d from checkpoint.",
            network_id, current_tap,
        )
    else:
        current_tap = tap_neutral
    net.trafo.loc[trafo_idx, "tap_pos"] = current_tap

    logger.info(
        "[Scenario 2 | %s] OLTC setup: %d trafo(s), ctrl_buses=%s, "
        "tap_range=[%d, %d], tap_neutral=%d, sign=%+d | "
        "%d timesteps, %d DERs, %d loads",
        network_id, len(trafo_idx), ctrl_buses,
        tap_min_gang, tap_max_gang, tap_neutral, sign,
        len(time_steps),
        len(ap.der_p.columns), len(ap.load_idx),
    )

    records: list[TimestepRecord] = resumed_records.copy()

    # ------------------------------------------------------------------
    # Timestep loop
    # ------------------------------------------------------------------
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
                t_total_ms=(time.perf_counter() - t0) * 1e3,
            ))
            continue

        # [4] OLTC decision — mean voltage across all OLTC secondary busbars.
        #     vm_ctrl     : mean used for the tap decision (deadband comparison).
        #     vm_ctrl_min : diagnostic — not used in the control law.
        #     vm_ctrl_max : diagnostic — not used in the control law.
        vm_ctrl     = float(net.res_bus.loc[ctrl_buses, "vm_pu"].mean())
        vm_ctrl_min = float(net.res_bus.loc[ctrl_buses, "vm_pu"].min())
        vm_ctrl_max = float(net.res_bus.loc[ctrl_buses, "vm_pu"].max())

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
                    "[Scenario 2 | %s] t=%d: tap %d -> %d | "
                    "vm_ctrl=%.4f (min=%.4f max=%.4f)",
                    network_id, t, prev_tap, current_tap,
                    vm_ctrl, vm_ctrl_min, vm_ctrl_max,
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
            losses_mw_t      = float(net.res_line["pl_mw"].sum() + net.res_trafo["pl_mw"].sum())
            grid_import_mw_t = float(net.res_ext_grid["p_mw"].sum())
            der_gen_mw_t     = float(ap.der_p.iloc[t].sum()) if not ap.der_p.empty else 0.0
            load_mw_t        = float(ap.load_p.iloc[t].sum()) if not ap.load_p.empty else 0.0
        else:
            vm = ll = tl = _empty_series()
            ov_buses = uv_buses = ov_lines = ov_trafos = []

        rec = TimestepRecord(
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
            losses_mw      = losses_mw_t if converged else None,
            grid_import_mw = grid_import_mw_t if converged else None,
            der_gen_mw     = der_gen_mw_t,
            load_mw        = load_mw_t,
            t_total_ms     = (time.perf_counter() - t0) * 1e3,
        )
        if publish_fn is not None:
            publish_fn.on_timestep(rec)
        records.append(rec)

        if t % 96 == 0:
            if live_csv_rewrite_fn is not None:
                    partial = ScenarioResult.from_records(
                        scenario_id="oltc", network_id=network_id,
                        records=records, elapsed_s=(publish_fn.cumulative_elapsed_s() 
                                                    if publish_fn is not None else time.perf_counter() - t_start),
                        dt_s=ap.dt_s,   # correct value, already in scope, no placeholder
                    )
                    live_csv_rewrite_fn(partial)
            voltage_violation_steps = sum(
                1 for r in records
                if r.over_voltage_buses or r.under_voltage_buses
            )
            any_violation_steps = sum(
                1 for r in records
                if (
                    r.over_voltage_buses or r.under_voltage_buses
                    or r.overloaded_lines or r.overloaded_trafos
                )
            )
            logger.info(
                "[Scenario 2 | %s] t=%d/%d (%.1f%%) | tap=%d | "
                "vm_ctrl=%.4f | voltage_steps=%d | any_steps=%d",
                network_id, t, _T_full,
                100.0 * t / max(_T_full, 1),
                current_tap,
                vm_ctrl,
                voltage_violation_steps,
                any_violation_steps,
            )

    # ------------------------------------------------------------------
    # Build ScenarioResult
    # ------------------------------------------------------------------
    elapsed = publish_fn.cumulative_elapsed_s() if publish_fn is not None else time.perf_counter() - t_start

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
    if publish_fn is not None:
        publish_fn.on_scenario_end(result)
    return result
