"""
scenario_result.py
==================
Shared output types for all five HIL scenario runners.

Every runner returns a ScenarioResult built from a list of TimestepRecord
objects.  The comparison layer (metrics, plots, hosting capacity) consumes
ScenarioResult objects and can therefore treat all five scenarios uniformly
without knowing which runner produced them.

Schema
------
TimestepRecord  — one entry per simulated timestep.  Contains raw per-bus
                  and per-element results plus pre-derived violation lists so
                  the comparison layer never needs to re-threshold.

ScenarioResult  — aggregate over the full time series.  Derived metrics are
                  computed once at construction time from the record list.
                  All None fields indicate that the scenario does not produce
                  that output (e.g. Scenario 1 produces no q_applied).

Profile adapter
---------------
adapt_profiles() converts the dict returned by profile_builder.build_profiles()
into an AdaptedProfiles object with consistent field names used by all runners:

    profiles["load"]  →  load_p (p_mw), load_q (q_mvar derived from net ratios)
    profiles["pv"]    →  combined into der_p alongside wind
    profiles["wind"]  →  combined into der_p alongside pv

All runners import AdaptedProfiles and call adapt_profiles() before the loop.

Usage
-----
    from scenario_result import adapt_profiles, ScenarioResult, TimestepRecord

    ap  = adapt_profiles(net, profiles)
    rec = TimestepRecord(t=0, timestamp=ap.times[0], ...)
    result = ScenarioResult.from_records("baseline", "1-MV-rural--2-sw",
                                         records, elapsed_s=12.3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from violation_detector import (
    V_MIN,
    V_MAX,
    LINE_MAX_LOADING,
    TRAFO_MAX_LOADING,
    VOLTAGE_EPSILON,
    LOADING_EPSILON,
)
from collections import Counter

logger = logging.getLogger(__name__)


# ===========================================================================
# Profile adapter
# ===========================================================================

@dataclass
class AdaptedProfiles:
    """
    Normalised view of the dict returned by profile_builder.build_profiles().

    Attributes
    ----------
    load_p : pd.DataFrame
        Shape (T × N_loads).  Columns = load element indices.  Values = MW.
        Clipped to ≥ 0 (inherited from profile_builder).

    load_q : pd.DataFrame
        Shape (T × N_loads).  Columns = load element indices.  Values = MVAr.
        Derived by scaling load_p by the per-load Q/P ratio from net.load at
        the time adapt_profiles() is called.  Loads with p_mw ≤ 0 in net.load
        get ratio = 0 (no reactive injection assumed).

    der_p : pd.DataFrame
        Shape (T × N_ders).  Columns = sgen element indices (PV and wind
        combined, sorted ascending).  Values = MW.  Used by all manual-loop
        scenarios to write net.sgen.p_mw index-explicitly.

    pv_idx : pd.Index
        Sgen indices of PV units (subset of der_p.columns).

    wind_idx : pd.Index
        Sgen indices of wind units (subset of der_p.columns).

    load_idx : pd.Index
        Load element indices (= load_p.columns = load_q.columns).

    times : pd.DatetimeIndex
        Full time axis aligned to load_p / der_p index.

    dt_s : float
        Timestep duration in seconds, inferred from times.freq or by
        differencing the first two timestamps.  Used to construct DERDynamics.
    """
    load_p:   pd.DataFrame
    load_q:   pd.DataFrame
    der_p:    pd.DataFrame
    pv_idx:   pd.Index
    wind_idx: pd.Index
    load_idx: pd.Index
    times:    pd.DatetimeIndex
    dt_s:     float


def adapt_profiles(net, profiles: dict) -> AdaptedProfiles:
    """
    Convert a profile_builder output dict into an AdaptedProfiles object.

    Parameters
    ----------
    net      : pandapower network.  Used only to derive per-load Q/P ratios
               from net.load.p_mw and net.load.q_mvar at call time.
    profiles : dict returned by profile_builder.build_profiles().
               Must contain keys "load", "pv", "wind", "times".

    Returns
    -------
    AdaptedProfiles

    Raises
    ------
    KeyError  : If a required key is absent from profiles.
    ValueError: If pv and wind DataFrames have no overlapping index with times.

    Notes
    -----
    Q/P ratio derivation
        For each load i:  ratio_i = q_mvar_i / p_mw_i   (from net.load).
        If p_mw_i ≤ 0 the ratio is set to 0.0 (division avoided).
        load_q = load_p × ratio (broadcast: each column gets its own scalar).
        This preserves the network's power-factor assumptions across all T.

    DER index alignment
        pv_df and wind_df may have disjoint column sets when the network has
        only PV or only wind.  pd.concat handles this safely; missing columns
        from either side fill with 0.0 and are then sorted.

    dt_s inference
        Preferred: pd.tseries.frequencies.to_offset(times.freq).nanos / 1e9.
        Fallback : (times[1] - times[0]).total_seconds() when freq is None.
        A minimum of 1.0 s is enforced to guard against degenerate indices.
    """
    required = ("load", "pv", "wind", "times")
    for k in required:
        if k not in profiles:
            raise KeyError(
                f"adapt_profiles: profiles dict missing key '{k}'. "
                f"Keys present: {list(profiles.keys())}"
            )

    times:    pd.DatetimeIndex = profiles["times"]
    load_p:   pd.DataFrame     = profiles["load"].copy()
    pv_df:    pd.DataFrame     = profiles["pv"].copy()
    wind_df:  pd.DataFrame     = profiles["wind"].copy()

    # Align load_p to times.  profile_builder usually pre-aligns, but an
    # explicit reindex is cheap insurance against mismatched indices when
    # ERA5 or custom profiles are used.  Missing rows fill with 0.0 (no
    # load) rather than NaN, which would silently corrupt load_q later.
    load_p = load_p.reindex(times).fillna(0.0)

    # ------------------------------------------------------------------
    # Load Q: derive from net.load Q/P ratio, broadcast across time
    # ------------------------------------------------------------------
    load_idx = load_p.columns  # integer load element indices

    net_p   = net.load.loc[load_idx, "p_mw"].values.astype(float)
    net_q   = net.load.loc[load_idx, "q_mvar"].values.astype(float)

    # Avoid division by zero; loads with p_mw ≤ 0 get ratio = 0.
    # np.divide with `out` and `where` avoids evaluating net_q / net_p on
    # zero rows entirely, preventing divide warnings that np.where cannot
    # suppress (np.where evaluates both branches before selecting).
    ratio = np.divide(
        net_q,
        net_p,
        out=np.zeros_like(net_q, dtype=float),
        where=net_p > 0.0,
    )  # shape (N_loads,)

    # Broadcast: multiply each column of load_p by its scalar ratio.
    load_q = load_p.multiply(ratio, axis="columns")

    # ------------------------------------------------------------------
    # DER P: merge PV and wind into one DataFrame sorted by sgen index
    # ------------------------------------------------------------------
    pv_idx   = pv_df.columns
    wind_idx = wind_df.columns

    if pv_df.empty and wind_df.empty:
        logger.warning(
            "adapt_profiles: both pv and wind DataFrames are empty. "
            "der_p will be an empty DataFrame."
        )
        der_p = pd.DataFrame(index=times, dtype=float)
    else:
        der_p = (
            pd.concat([pv_df, wind_df], axis=1)
            .sort_index(axis=1)         # ascending sgen index order
            .reindex(times)             # align to times (fills NaN → 0)
            .fillna(0.0)
        )

    # ------------------------------------------------------------------
    # dt_s inference
    # ------------------------------------------------------------------
    if len(times) >= 2:
        try:
            if times.freq is not None:
                import pandas.tseries.frequencies as ptf
                dt_s = ptf.to_offset(times.freq).nanos / 1e9
            else:
                dt_s = (times[1] - times[0]).total_seconds()
        except Exception:
            dt_s = (times[1] - times[0]).total_seconds()
    else:
        dt_s = 900.0   # safe default: 15-min SimBench resolution
        logger.warning(
            "adapt_profiles: times has fewer than 2 entries; "
            "dt_s defaulting to %.1f s.", dt_s
        )

    dt_s = max(dt_s, 1.0)   # guard against degenerate indices

    logger.info(
        "adapt_profiles: %d timesteps | %d loads | %d PV | %d wind | dt=%.0f s",
        len(times), len(load_idx), len(pv_idx), len(wind_idx), dt_s,
    )

    return AdaptedProfiles(
        load_p   = load_p,
        load_q   = load_q,
        der_p    = der_p,
        pv_idx   = pd.Index(pv_idx),
        wind_idx = pd.Index(wind_idx),
        load_idx = pd.Index(load_idx),
        times    = times,
        dt_s     = dt_s,
    )


# ===========================================================================
# Utility 1 — Oversized inverter helper
# ===========================================================================

def oversize_inverters(net, factor: float = 1.1) -> None:
    """
    Scale net.sgen["sn_mva"] to simulate oversized inverters.

    Models the scenario where inverter apparent power capacity exceeds the
    generator's rated active power by ``factor``.  This raises Q_max for
    all scenarios that derive it from sn_mva (VoltVarController,
    SensitivityCoordinator, DERDynamics all use Q_RATIO × sn_mva).

    The standard case is ``factor = 1.1`` (10 % oversizing), which gives:
        Q_max = Q_RATIO × (1.1 × P_rated) = 0.48 × 1.1 × P_rated = 0.528 × P_rated

    versus the baseline:
        Q_max = Q_RATIO × P_rated = 0.48 × P_rated

    Parameters
    ----------
    net : pandapower network — modified in place.
    factor : float
        sn_mva multiplier.  Must be >= 1.0.  Typical values:
            1.1  → 10 % oversize (standard comparison case)
            1.2  → 20 % oversize
            1.5  → 50 % oversize (theoretical upper bound study)

    Raises
    ------
    ValueError : if factor < 1.0 (undersizing is not supported here;
                 use profile scaling for reduced-output studies).

    Notes
    -----
    - Call this on the net object BEFORE run_benchmark() or any individual
      scenario runner.  benchmark_runner.py deep-copies net once per
      scenario, so a single pre-call here propagates to all five runners.
    - net.sgen["p_rated"] is NOT a standard pandapower column.  The correct
      source is net.sgen["sn_mva"] before any prior oversize call.  If you
      want to re-run with a different factor on the same net, reload the
      network fresh rather than calling oversize_inverters() twice.
    - Scenario 3 (SVC) derives Q_MAX from net.trafo.sn_mva.sum(), not from
      sgen.sn_mva, so this function has no effect on the SVC scenario.
    - Scenario 5 (OPF) recomputes q_lim from sn_mva inside
      _write_timestep_opf_state(), so it will also benefit from the
      increased sn_mva.

    Example
    -------
    ::

        import simbench as sb
        from profile_builder import build_annual_profiles
        from scenario_result import oversize_inverters
        from benchmark_runner import BenchmarkConfig, run_benchmark

        net      = sb.get_simbench_net("1-MV-rural--2-sw")
        profiles = build_annual_profiles(net, ...)

        # Standard run (baseline Q capacity)
        result_std = run_benchmark(net, profiles, network_id="std")

        # Oversized run (10 % larger inverters)
        import copy
        net_os = copy.deepcopy(net)
        oversize_inverters(net_os, factor=1.1)
        result_os = run_benchmark(net_os, profiles, network_id="oversized_1.1")
    """
    if factor < 1.0:
        raise ValueError(
            f"oversize_inverters: factor must be >= 1.0, got {factor:.4f}. "
            "Use profile scaling (der_scaling in ParameterConfig) for "
            "reduced-output studies."
        )
    if net.sgen.empty:
        logger.warning("oversize_inverters: net.sgen is empty — no sgens to resize.")
        return

    if "sn_mva" not in net.sgen.columns:
        logger.warning(
            "oversize_inverters: net.sgen has no 'sn_mva' column — nothing to scale."
        )
        return

    # Filter to actual DER inverters only (PV and wind).
    # SimBench networks store load elements as sgens with negative p_mw;
    # those must not be treated as inverters.
    # Mirror the same type-string logic used in profile_builder.build_annual_profiles().
    type_col = net.sgen.get("type", pd.Series("", index=net.sgen.index)).fillna("")
    is_pv   = type_col.str.lower().str.contains("pv|solar", na=False)
    is_wind = type_col.str.lower().str.contains("wind|wp",  na=False)
    der_mask = (is_pv | is_wind) & (net.sgen["in_service"] == True)

    der_idx = net.sgen.index[der_mask]

    if der_idx.empty:
        logger.warning(
            "oversize_inverters: no PV or wind sgens found "
            "(type column checked for 'pv', 'solar', 'wind', 'wp'). "
            "Nothing scaled."
        )
        return

    # Sanitise first: NaN / inf sn_mva would silently propagate.
    # Replace with p_mw as best estimate (same logic as stress._sanitise_sgen).
    sn = (
        net.sgen["sn_mva"]
        .replace([float("inf"), float("-inf")], np.nan)
        .fillna(net.sgen["p_mw"])
        .clip(lower=1e-6)
    )
    net.sgen["sn_mva"] = sn * factor
    logger.info(
        "oversize_inverters: scaled sn_mva by %.4f for %d DER sgens "
        "(PV=%d, wind=%d) | new mean sn_mva = %.4f MVA.",
        factor,
        len(der_idx),
        int(is_pv[der_idx].sum()),
        int(is_wind[der_idx].sum()),
        float(net.sgen.loc[der_idx, "sn_mva"].mean()),
    )


# ===========================================================================
# Utility 2 — Profile time-slice helper
# ===========================================================================

def slice_profiles(
        profiles: dict,
        period:   str,
        index:    int = 1,
) -> dict:
    """
    Return a new profiles dict sliced to a sub-period of the annual series.

    All runners (via adapt_profiles) use the "times" key to determine the
    timestep axis.  This function slices every DataFrame in the profiles dict
    to match the requested calendar window, so the runners see only that
    window without any code changes inside them.

    The original profiles dict is not modified.

    Parameters
    ----------
    profiles : dict
        Output of profile_builder.build_annual_profiles().  Must contain at
        minimum "load", "pv", "wind", "times".  Any extra keys whose values
        are DataFrames or Series with a DatetimeIndex are sliced as well.
    period : {"month", "week", "day"}
        Granularity of the slice.
        - "month" : calendar month (Jan=1 … Dec=12).
        - "week"  : ISO week number (1–52/53).
        - "day"   : day-of-year (1–365/366).
    index : int
        Which period to select.
        - month : 1–12  (default 1 = January)
        - week  : 1–53  (default 1 = first ISO week)
        - day   : 1–366 (default 1 = 1 Jan)

    Returns
    -------
    dict
        Shallow copy of profiles with DataFrames/Series replaced by their
        sliced counterparts.  The "times" key is updated to the sliced
        DatetimeIndex.  All other scalar/non-indexed values are passed
        through unchanged.

    Raises
    ------
    ValueError
        - ``period`` is not one of {"month", "week", "day"}.
        - ``index`` is out of range for the given period.
        - The slice produces an empty index (requested period not present in
          the data, e.g. week 53 in a non-leap year).

    Notes
    -----
    Slicing does NOT reset integer positions.  The index of the returned
    DataFrames is still a DatetimeIndex.  adapt_profiles() uses
    ``iloc[t]`` (integer position within the sliced frame), so this is
    correct — the runners iterate over ``range(len(ap.times))`` which aligns
    with the sliced frame's positional axis, not its calendar labels.

    For Scenario 1 (run_timeseries / ConstControl), the integer reindex
    step inside run_scenario_1 rebuilds a 0-based integer index from
    ``range(len(ap.times))``, so sliced profiles work there too.

    Example — run benchmark on January only
    ----------------------------------------
    ::

        from scenario_result import slice_profiles
        from benchmark_runner import BenchmarkConfig, run_benchmark

        jan_profiles = slice_profiles(profiles, period="month", index=1)
        result = run_benchmark(net, jan_profiles, network_id="1-MV-rural--2-sw_Jan")

    Example — run on a single representative day (e.g. summer solstice ≈ day 172)
    -------------------------------------------------------------------------------
    ::

        summer_profiles = slice_profiles(profiles, period="day", index=172)
        result = run_benchmark(net, summer_profiles, network_id="1-MV-rural--2-sw_day172")

    Example — run on ISO week 28 (peak-summer)
    --------------------------------------------
    ::

        week28_profiles = slice_profiles(profiles, period="week", index=28)
        result = run_benchmark(net, week28_profiles, network_id="1-MV-rural--2-sw_wk28")
    """
    _VALID_PERIODS = {"month", "week", "day"}
    if period not in _VALID_PERIODS:
        raise ValueError(
            f"slice_profiles: period must be one of {sorted(_VALID_PERIODS)}, "
            f"got '{period}'."
        )

    # --- Validate index range -----------------------------------------------
    _PERIOD_RANGES = {
        "month": (1, 12),
        "week":  (1, 53),
        "day":   (1, 366),
    }
    lo, hi = _PERIOD_RANGES[period]
    if not (lo <= index <= hi):
        raise ValueError(
            f"slice_profiles: index={index} out of range for period='{period}'. "
            f"Valid range: {lo}–{hi}."
        )

    # --- Build boolean mask on the times index ------------------------------
    times: pd.DatetimeIndex = profiles["times"]

    if period == "month":
        mask = times.month == index
    elif period == "week":
        mask = times.isocalendar().week.values == index
    else:  # "day"
        mask = times.day_of_year == index

    sliced_times = times[mask]

    if len(sliced_times) == 0:
        raise ValueError(
            f"slice_profiles: {period}={index} produced an empty slice. "
            f"The profiles span {times[0].date()} to {times[-1].date()}. "
            f"Check that {period} {index} falls within the data range."
        )

    out = profiles.copy()
    out["times"] = times[mask]
    for key in ("load", "pv", "wind"):
        if key in profiles and profiles[key] is not None:
            out[key] = profiles[key].loc[out["times"]].copy()

    try:
        from profile_builder import find_extreme_days
        out["extreme_days"] = find_extreme_days(out)
    except Exception:
        out["extreme_days"] = profiles.get("extreme_days", {})

    logger.info(
        "slice_profiles: %s=%d → %d timesteps (%.2f h)",
        period, index, len(sliced_times),
        len(sliced_times) * (
            (sliced_times[1] - sliced_times[0]).total_seconds() / 3600
            if len(sliced_times) >= 2 else 0.0
        ),
    )
    return out

# ===========================================================================
# TimestepRecord
# ===========================================================================

@dataclass
class TimestepRecord:
    """
    Raw result snapshot for a single simulated timestep.

    All list fields contain pandapower element indices (not positional
    integers) so they can be joined back to net.bus / net.line / net.trafo
    without ambiguity.

    Attributes
    ----------
    t : int
        Zero-based timestep index into the time series.

    timestamp : pd.Timestamp
        Wall-clock label for this timestep (from AdaptedProfiles.times[t]).

    vm_pu : pd.Series
        Bus voltage magnitudes in per-unit, indexed by bus index.
        Empty Series (dtype=float) if the power flow did not converge.

    line_loading : pd.Series
        Line thermal loading in %, indexed by line index.
        Empty if not converged.

    trafo_loading : pd.Series
        Transformer thermal loading in %, indexed by trafo index.
        Empty if not converged.

    over_voltage_buses : List[int]
        Bus indices where vm_pu > V_MAX (1.05 pu).

    under_voltage_buses : List[int]
        Bus indices where vm_pu < V_MIN (0.95 pu).

    overloaded_lines : List[int]
        Line indices where loading_percent > 100 %.

    overloaded_trafos : List[int]
        Trafo indices where loading_percent > 100 %.

    q_applied_mvar : pd.Series or None
        Reactive power applied by the scenario actuator this timestep,
        indexed by sgen index.  For Scenario 3 this is the SVC sgen q_mvar.
        For Scenario 4 this is DER q_mvar across all controlled sgens.
        None for Scenario 1 (Baseline) and Scenario 2 (OLTC — no reactive
        control on sgens).

    p_applied_mw : pd.Series or None
        Active power actually applied to net.sgen after ramp limiting (may
        differ from profile value when DERDynamics clips a large ramp).
        None for Scenario 1.

    curtailment_needed : bool
        True if the Tier 1 chain exhausted Q capacity and violations remain.
        Always False for Scenarios 1 and 5.

    converged : bool
        True if the power flow (runpp / runopp) converged this timestep.

    p_target_mw : pd.Series or None
        Raw profile P value before ramp limiting or OPF adjustment, indexed
        by sgen index.  Set by Scenarios 4 and 5; None for Scenario 1.
        Used by ScenarioResult.from_records() to compute curtailed_energy_mwh.
        Stored as a proper dataclass field so it survives export, copy, and
        reconstruction — not as a hidden attribute.

    tap_pos : int or None
        Tap position in effect on the OLTC transformer group after this
        timestep.  None for all scenarios except Scenario 2.

    tap_changed : bool or None
        True if the tap position actually moved this timestep.  False when
        the voltage was inside the control deadband or a tap command was
        blocked.  None outside Scenario 2.

    tap_attempted : bool or None
        True if a tap command was issued this timestep (regardless of whether
        it was accepted).  Distinguishes "no action taken" (False) from
        "action attempted but blocked" (True, tap_changed=False).
        None outside Scenario 2.

    tap_candidate : int or None
        The tap position that was attempted.  Set whenever tap_attempted is
        True; None otherwise.  When tap_blocked_reason is set, tap_candidate
        records the rejected position and tap_pos retains the prior value.
        None outside Scenario 2.

    post_pf_reused : bool or None
        True when the pre-action power flow result is reused as the settled
        state because the tap did not move (deadband or rail hit).  False
        when a post-tap runpp was executed.  None outside Scenario 2.

    tap_blocked_reason : str or None
        Reason a tap command was not applied.  One of:
            "pre_pf_non_convergence"  — pre-action runpp diverged; no voltage
                                        reading available to decide tap action.
            "post_pf_non_convergence" — post-tap runpp diverged; tap rolled
                                        back to previous validated position.
            "tap_limit_reached"       — voltage was outside the control band
                                        but the tap was already at the ganged
                                        range limit (tap_min_gang or
                                        tap_max_gang); no further movement
                                        possible.
        None when no blocking occurred.  None outside Scenario 2.

    svc_q_mvar : float or None
        Reactive power command issued to the SVC sgen this timestep (MVAr).
        Positive = injection (raises voltage), negative = absorption (lowers
        voltage).  None for all scenarios except Scenario 3.

    svc_saturated : bool or None
        True if the deadbanded droop reached ±Q_MAX this timestep.  Indicates
        the SVC was operating at its reactive power limit.  None outside
        Scenario 3.
    """
    t:                    int
    timestamp:            pd.Timestamp
    vm_pu:                pd.Series
    line_loading:         pd.Series
    trafo_loading:        pd.Series
    over_voltage_buses:   List[int]
    under_voltage_buses:  List[int]
    overloaded_lines:     List[int]
    overloaded_trafos:    List[int]
    q_applied_mvar:       Optional[pd.Series]
    p_applied_mw:         Optional[pd.Series]
    curtailment_needed:   bool
    converged:            bool
    p_target_mw:          Optional[pd.Series] = None

    # Scenario 2 — OLTC only
    tap_pos:            int  | None = None   # tap position in effect after this timestep
    tap_changed:        bool | None = None   # True if tap actually moved
    tap_attempted:      bool | None = None   # True if a tap command was issued
    tap_candidate:      int  | None = None   # the tap position that was attempted
    post_pf_reused:     bool | None = None   # True when pre-PF reused (tap unchanged)
    tap_blocked_reason: str  | None = None   # "pre_pf_non_convergence"  |
                                            # "post_pf_non_convergence" |
                                            # "tap_limit_reached"       | None

    # Scenario 3 — SVC only
    svc_q_mvar:    float | None = None  # Q command issued (MVAr)
    svc_saturated: bool | None  = None  # True if droop reached ±Q_MAX

    # Energy balance — populated by all runners for converged timesteps.
    # None for non-converged timesteps.
    losses_mw:      Optional[float] = None
    # Joule losses in lines + transformers (MW).  Summed from
    # net.res_line["pl_mw"] + net.res_trafo["pl_mw"] after the settled PF.

    grid_import_mw: Optional[float] = None
    # Net active power drawn from the external grid at the slack bus (MW).
    # Signed: positive = network imports from the grid (net load > generation),
    # negative = network exports to the grid (net generation > load).
    # Source: net.res_ext_grid["p_mw"].sum() after the settled PF.

    der_gen_mw:     Optional[float] = None
    # Total DER (PV + wind) active power from the profile at this timestep (MW).
    # Derived from ap.der_p.iloc[t].sum() — profile value, not PF result.
    # Always populated regardless of convergence.

    load_mw:        Optional[float] = None
    # Total load active power from the profile at this timestep (MW).
    # Derived from ap.load_p.iloc[t].sum() — profile value, not PF result.
    # Always populated regardless of convergence.

    # Control effort — Scenario 4 only.  None for all other scenarios.
    coordination_active: Optional[bool] = None
    # True if the sensitivity coordinator (Item 3) modified q_initial from
    # Item 2 by more than 1e-6 MVAr on any DER.  False if the coordinator
    # ran but left q unchanged (e.g. J matrix was ill-conditioned).
    # None for scenarios without sensitivity coordination.

    q_saturated_count: Optional[int] = None
    # Number of DERs whose |q_applied| reached the ±(Q_RATIO × p_installed)
    # clamp at this timestep.  Indicates inverter reactive capability
    # exhaustion.  None for scenarios without DER reactive control.

    curtail_exhausted: Optional[bool] = None
    # True if the curtailment loop reached MAX_CURTAIL_ITERS without clearing
    # violations (or hit P=0 and violations persisted). Distinguishes
    # "curtailment triggered and resolved" from "curtailment exhausted, residual
    # violation is structural/background". None for all scenarios except 4.
    hil_latency_ms: Optional[float] = None
    t_total_ms: Optional[float] = None

    def to_checkpoint_dict(self) -> dict:
        """
        Full, lossless serialization for resume — distinct from the compact
        dashboard live frame (build_live_frame in publisher.py), which drops
        fields not needed for the voltage-heatmap animation.
        """
        def _s(x):
            return None if x is None else x.to_dict()
        return {
            "t": self.t,
            "timestamp": self.timestamp.isoformat(),
            "vm_pu": _s(self.vm_pu),
            "line_loading": _s(self.line_loading),
            "trafo_loading": _s(self.trafo_loading),
            "over_voltage_buses": self.over_voltage_buses,
            "under_voltage_buses": self.under_voltage_buses,
            "overloaded_lines": self.overloaded_lines,
            "overloaded_trafos": self.overloaded_trafos,
            "q_applied_mvar": _s(self.q_applied_mvar),
            "p_applied_mw": _s(self.p_applied_mw),
            "curtailment_needed": self.curtailment_needed,
            "converged": self.converged,
            "p_target_mw": _s(self.p_target_mw),
            # Scenario 2 — OLTC only
            "tap_pos": self.tap_pos,
            "tap_changed": self.tap_changed,
            "tap_attempted": self.tap_attempted,
            "tap_candidate": self.tap_candidate,
            "post_pf_reused": self.post_pf_reused,
            "tap_blocked_reason": self.tap_blocked_reason,
            # Scenario 3 — SVC only
            "svc_q_mvar": self.svc_q_mvar,
            "svc_saturated": self.svc_saturated,
            # Energy balance — all runners
            "losses_mw": self.losses_mw,
            "grid_import_mw": self.grid_import_mw,
            "der_gen_mw": self.der_gen_mw,
            "load_mw": self.load_mw,
            # Scenario 4 — control effort
            "coordination_active": self.coordination_active,
            "q_saturated_count": self.q_saturated_count,
            "curtail_exhausted": self.curtail_exhausted,
            # Timing
            "hil_latency_ms": self.hil_latency_ms,
            "t_total_ms": self.t_total_ms,
            "_ckpt_version": 1,
        }

    @classmethod
    def from_checkpoint_dict(cls, d: dict) -> "TimestepRecord":
        """Reconstruct a TimestepRecord from to_checkpoint_dict() output."""
        def _d(x):
            if x is None:
                return None
            return pd.Series({int(k): v for k, v in x.items()}, dtype=float)

        def _series_or_empty(x):
            s = _d(x)
            return s if s is not None else pd.Series(dtype=float)

        if d.get("_ckpt_version", 0) != 1:
            logger.warning("checkpoint schema v%s != 1; some fields may default to None.",
                           d.get("_ckpt_version", 0))

        return cls(
            t=d["t"],
            timestamp=pd.Timestamp(d["timestamp"]),
            vm_pu=_series_or_empty(d["vm_pu"]),
            line_loading=_series_or_empty(d["line_loading"]),
            trafo_loading=_series_or_empty(d["trafo_loading"]),
            over_voltage_buses=d["over_voltage_buses"],
            under_voltage_buses=d["under_voltage_buses"],
            overloaded_lines=d["overloaded_lines"],
            overloaded_trafos=d["overloaded_trafos"],
            q_applied_mvar=_d(d["q_applied_mvar"]),
            p_applied_mw=_d(d["p_applied_mw"]),
            curtailment_needed=d["curtailment_needed"],
            converged=d["converged"],
            p_target_mw=_d(d.get("p_target_mw")),
            tap_pos=d.get("tap_pos"),
            tap_changed=d.get("tap_changed"),
            tap_attempted=d.get("tap_attempted"),
            tap_candidate=d.get("tap_candidate"),
            post_pf_reused=d.get("post_pf_reused"),
            tap_blocked_reason=d.get("tap_blocked_reason"),
            svc_q_mvar=d.get("svc_q_mvar"),
            svc_saturated=d.get("svc_saturated"),
            losses_mw=d.get("losses_mw"),
            grid_import_mw=d.get("grid_import_mw"),
            der_gen_mw=d.get("der_gen_mw"),
            load_mw=d.get("load_mw"),
            coordination_active=d.get("coordination_active"),
            q_saturated_count=d.get("q_saturated_count"),
            curtail_exhausted=d.get("curtail_exhausted"),
            hil_latency_ms=d.get("hil_latency_ms"),
            t_total_ms=d.get("t_total_ms"),
        )


def _empty_series(dtype=float) -> pd.Series:
    """Return a typed empty Series — used when PF did not converge."""
    return pd.Series(dtype=dtype)


def make_record_from_report(
        t:          int,
        timestamp:  pd.Timestamp,
        net,
        converged:  bool,
        v_min:      float = V_MIN,
        v_max:      float = V_MAX,
        q_applied:  Optional[pd.Series] = None,
        p_applied:  Optional[pd.Series] = None,
        curtailment_needed: bool = False,
) -> TimestepRecord:
    """
    Build a TimestepRecord from the current net result tables.

    Call this after runpp() / runopp() has populated net.res_bus,
    net.res_line, net.res_trafo.  If converged=False, all result Series
    are empty and all violation lists are empty.

    Parameters
    ----------
    t, timestamp : timestep index and calendar label.
    net          : pandapower network with populated result tables.
    converged    : whether the power flow converged this timestep.
    v_min, v_max : voltage planning limits (default V_MIN/V_MAX from
                   violation_detector).
    q_applied    : Q setpoints written to net.sgen (None for Scenario 1).
    p_applied    : P after ramp limiting (None for Scenario 1).
    curtailment_needed : flag from CoordinatorResult (Scenario 4 only).

    Returns
    -------
    TimestepRecord
    """
    if not converged:
        return TimestepRecord(
            t=t, timestamp=timestamp,
            vm_pu=_empty_series(), line_loading=_empty_series(),
            trafo_loading=_empty_series(),
            over_voltage_buses=[], under_voltage_buses=[],
            overloaded_lines=[], overloaded_trafos=[],
            q_applied_mvar=None, p_applied_mw=None,
            curtailment_needed=False, converged=False,
        )

    vm   = net.res_bus["vm_pu"].copy()
    ll   = net.res_line["loading_percent"].copy()
    tl   = net.res_trafo["loading_percent"].copy()

    ov_buses  = vm.index[vm > v_max + VOLTAGE_EPSILON].tolist()
    uv_buses  = vm.index[vm < v_min - VOLTAGE_EPSILON].tolist()
    ov_lines  = ll.index[ll > LINE_MAX_LOADING  + LOADING_EPSILON].tolist()
    ov_trafos = tl.index[tl > TRAFO_MAX_LOADING + LOADING_EPSILON].tolist()

    return TimestepRecord(
        t=t, timestamp=timestamp,
        vm_pu=vm, line_loading=ll, trafo_loading=tl,
        over_voltage_buses=ov_buses, under_voltage_buses=uv_buses,
        overloaded_lines=ov_lines, overloaded_trafos=ov_trafos,
        q_applied_mvar=q_applied, p_applied_mw=p_applied,
        curtailment_needed=curtailment_needed,
        converged=True,
    )


# ===========================================================================
# ScenarioResult
# ===========================================================================

@dataclass
class ScenarioResult:
    """
    Aggregate output for one scenario run over the full time series.

    Derived metrics are computed once from the record list via from_records()
    and stored as plain numeric fields so the comparison layer never iterates
    over records itself.

    Attributes
    ----------
    scenario_id : str
        One of "baseline" | "oltc" | "svc" | "volt_var" | "opf".

    network_id : str
        Human-readable network identifier, e.g. "1-MV-rural--2-sw".

    records : List[TimestepRecord]
        Full per-timestep record list.  Not used by derived metrics; retained
        for post-hoc inspection, plotting, and export.

    elapsed_s : float
        Total wall-clock time for the scenario run in seconds.

    n_timesteps : int
        Total number of timesteps attempted (converged or not).

    n_converged : int
        Timesteps where the power flow converged.

    n_violation_steps : int
        Converged timesteps with at least one voltage or thermal violation.

    violation_duration_h : float
        n_violation_steps converted to hours: n_violation_steps × dt_s / 3600.
        More meaningful than raw timestep counts for published results —
        directly comparable across networks with different dt_s values.
        Example: 480 violation steps at 15-min resolution = 120 h/yr.

    total_overvoltage_bus_steps : int
        Sum of len(record.over_voltage_buses) across all converged timesteps.
        Measures how many bus-timestep pairs experienced overvoltage.

    total_undervoltage_bus_steps : int
        Same for undervoltage.

    total_overloaded_line_steps : int
        Sum of len(record.overloaded_lines) across all converged timesteps.

    total_overloaded_trafo_steps : int
        Sum of len(record.overloaded_trafos) across all converged timesteps.

    max_vm_pu : float
        Maximum bus voltage across all converged timesteps (pu).

    min_vm_pu : float
        Minimum bus voltage across all converged timesteps (pu).

    max_line_loading_pct : float
        Maximum line thermal loading across all converged timesteps (%).

    max_trafo_loading_pct : float
        Maximum trafo thermal loading across all converged timesteps (%).

    vdi : float
        Voltage Deviation Index — sum of |vm_pu − 1.0| across all buses and
        all converged timesteps.  Dimensionless.  Captures the aggregate
        voltage quality degradation, not just binary violation counts.
        Directly comparable across scenarios on the same network.
        Not normalised by N_buses or T so that it preserves additivity (two
        identical subnetworks produce twice the VDI of one).
        NaN if no converged timestep has a non-empty vm_pu Series.

    q_total_mvar_abs : float or None
        Sum of |q_applied_mvar| across all sgens and all timesteps (MVAr per
        timestep accumulated).  None for Scenario 1 (no reactive control).

    reactive_energy_mvarh : float or None
        q_total_mvar_abs converted to proper energy units (MVArh/yr):
            q_total_mvar_abs × dt_s / 3600.
        Quantifies the total inverter reactive burden imposed by the control
        strategy.  None for Scenario 1.  Zero for Scenarios 2 (OLTC only
        has no sgen reactive control) — depends on whether q_applied was set.

    curtailment_steps : int
        Number of timesteps where curtailment_needed was True.
        Always 0 for Scenarios 1 and 5.

    curtailed_energy_mwh : float or None
        Total curtailed energy estimated as:
            Σ_t Σ_i  max(0, p_target_i[t] − p_applied_i[t])  × (dt_s / 3600)
        None for Scenario 1 (no ramp limiting) and Scenario 5 (OPF directly
        determines p_applied; curtailment flag is not used).
        For Scenario 4, p_target is the raw profile value; p_applied is after
        ramp limiting.  This is a lower bound on true curtailment since DER
        dynamics may absorb part of the P reduction.

    total_losses_mwh : float or None
        Annual network Joule losses in MWh: Σ_t losses_mw[t] × dt_s / 3600.
        Sum of line and transformer active losses across all converged
        timesteps.  None if no converged record has losses_mw populated.

    grid_import_mwh : float or None
        Total energy imported from the external grid (MWh/yr):
            Σ_t  max(0, grid_import_mw[t])  × dt_s / 3600.
        None if grid_import_mw is absent from records.

    grid_export_mwh : float or None
        Total energy exported to the external grid (MWh/yr):
            Σ_t  |min(0, grid_import_mw[t])|  × dt_s / 3600.
        None if grid_import_mw is absent from records.

    der_gen_mwh : float or None
        Total DER (PV + wind) generation from the profile (MWh/yr):
            Σ_t der_gen_mw[t] × dt_s / 3600.
        Derived from profiles, not from PF results.  Always available
        when der_gen_mw is populated by the runner.

    load_demand_mwh : float or None
        Total load demand from the profile (MWh/yr):
            Σ_t load_mw[t] × dt_s / 3600.
        Derived from profiles, not from PF results.

    coordination_steps : int or None
        Number of converged timesteps where the sensitivity coordinator
        (Item 3) actively modified the Arduino's raw Q response by more
        than 1e-6 MVAr on at least one DER.  None for Scenarios 1–3 and 5.

    coordination_rate : float or None
        coordination_steps / n_converged.  Fraction of converged timesteps
        where the RPi's coordination layer was needed beyond the Arduino's
        Q(V) curve alone.  None outside Scenario 4.

    q_saturation_rate : float or None
        Fraction of converged timesteps where at least one DER reached its
        ±(Q_RATIO × p_installed) reactive power clamp.  Indicates how
        close the system is to exhausting inverter reactive capability.
        None outside Scenario 4.
    """
    scenario_id:    str
    network_id:     str
    records:        List[TimestepRecord]
    elapsed_s:      float

    # Derived fields — populated by from_records(), not set by caller directly.
    n_timesteps:                  int   = field(default=0)
    n_converged:                  int   = field(default=0)
    n_violation_steps:            int   = field(default=0)
    violation_duration_h:         float = field(default=0.0)
    total_overvoltage_bus_steps:  int   = field(default=0)
    total_undervoltage_bus_steps: int   = field(default=0)
    total_overloaded_line_steps:  int   = field(default=0)
    total_overloaded_trafo_steps: int   = field(default=0)
    max_vm_pu:                    float = field(default=float("nan"))
    min_vm_pu:                    float = field(default=float("nan"))
    max_line_loading_pct:         float = field(default=float("nan"))
    max_trafo_loading_pct:        float = field(default=float("nan"))
    vdi:                          float = field(default=float("nan"))
    q_total_mvar_abs:             Optional[float] = field(default=None)
    reactive_energy_mvarh:        Optional[float] = field(default=None)
    curtailment_steps:            int              = field(default=0)
    curtailed_energy_mwh:         Optional[float]  = field(default=None)
    curtail_exhausted_steps: int = field(default=0)
    # Timesteps where curtailment was triggered but could not clear violations
    # even after exhausting all iterations (P reduced to floor).
    # Always 0 for Scenarios 1–3 and 5.
    svc_bus:       int | None   = None  # fixed SVC bus index (Scenario 3 only)
    svc_q_max:     float | None = None  # Q_MAX used (0.20 × trafo sn_mva sum)

    # Energy balance metrics — all scenarios
    total_losses_mwh: Optional[float] = field(default=None)
    grid_import_mwh:  Optional[float] = field(default=None)
    grid_export_mwh:  Optional[float] = field(default=None)
    der_gen_mwh:      Optional[float] = field(default=None)
    load_demand_mwh:  Optional[float] = field(default=None)

    # Control effort metrics — Scenario 4 only
    coordination_steps: Optional[int]   = field(default=None)
    coordination_rate:  Optional[float] = field(default=None)
    q_saturation_rate:  Optional[float] = field(default=None)

    @classmethod
    def from_records(
            cls,
            scenario_id:  str,
            network_id:   str,
            records:      List[TimestepRecord],
            elapsed_s:    float,
            dt_s:         float = 900.0,
            svc_bus:      Optional[int]   = None,
            svc_q_max:    Optional[float] = None,
    ) -> "ScenarioResult":
        """
        Construct a ScenarioResult and compute all derived metrics.

        Parameters
        ----------
        scenario_id, network_id : identifiers.
        records     : list of TimestepRecord, one per timestep.
        elapsed_s   : total wall-clock time for the run.
        dt_s        : timestep duration in seconds.  Used only for
                      curtailed_energy_mwh calculation.  Defaults to 900 s
                      (15-min SimBench resolution).
        svc_bus     : fixed SVC bus index (Scenario 3 only).  None for all
                      other scenarios.
        svc_q_max   : Q_MAX used by the SVC droop (MVAr).  None for all
                      other scenarios.
        """
        if records:
            observed_ts = sorted(r.t for r in records)
            expected_ts = list(range(observed_ts[0], observed_ts[-1] + 1))
            if observed_ts != expected_ts:
                missing = sorted(set(expected_ts) - set(observed_ts))
                raise ValueError(
                    f"ScenarioResult.from_records({scenario_id!r}, {network_id!r}): "
                    f"records has {len(missing)} gap(s) in timestep coverage between "
                    f"t={observed_ts[0]} and t={observed_ts[-1]} — first few missing: "
                    f"{missing[:10]}{'...' if len(missing) > 10 else ''}. "
                    f"This usually means a checkpoint file was written with a sparse "
                    f"cadence (e.g. update_every_k) instead of every timestep, and a "
                    f"resumed run silently lost coverage for the pre-resume segment. "
                    f"Do not trust this result — re-run the scenario from scratch."
                )
            dup_ts = [t for t, c in Counter(observed_ts).items() if c > 1]
            if dup_ts:
                raise ValueError(
                    f"ScenarioResult.from_records({scenario_id!r}, {network_id!r}): "
                    f"records has duplicate timestep(s): {sorted(set(dup_ts))[:10]}. "
                    f"This usually means resumed_records and newly-simulated records "
                    f"overlapped — check the resume start_t computation."
                )

        converged = [r for r in records if r.converged]

        n_timesteps  = len(records)
        n_converged  = len(converged)

        n_violation_steps = sum(
            1 for r in converged
            if r.over_voltage_buses or r.under_voltage_buses
               or r.overloaded_lines or r.overloaded_trafos
        )

        # Violation duration in hours.  More meaningful than raw timestep counts
        # for papers — directly comparable across networks with different dt_s.
        violation_duration_h = n_violation_steps * dt_s / 3600.0
        total_ov  = sum(len(r.over_voltage_buses)  for r in converged)
        total_uv  = sum(len(r.under_voltage_buses) for r in converged)
        total_ol  = sum(len(r.overloaded_lines)    for r in converged)
        total_ot  = sum(len(r.overloaded_trafos)   for r in converged)

        # Voltage extremes
        if converged:
            max_vm = max(
                (r.vm_pu.max() for r in converged if not r.vm_pu.empty),
                default=float("nan"),
            )
            min_vm = min(
                (r.vm_pu.min() for r in converged if not r.vm_pu.empty),
                default=float("nan"),
            )
            max_ll = max(
                (r.line_loading.max() for r in converged if not r.line_loading.empty),
                default=float("nan"),
            )
            max_tl = max(
                (r.trafo_loading.max() for r in converged if not r.trafo_loading.empty),
                default=float("nan"),
            )
        else:
            max_vm = min_vm = max_ll = max_tl = float("nan")

        # Voltage Deviation Index — sum of |vm_pu - 1.0| across all buses and
        # all converged timesteps.  Not normalised by N_buses or T so it
        # preserves additivity.  float() cast ensures a plain Python float
        # (not numpy scalar) is stored, consistent with other derived fields.
        # NaN guard: if no converged record has a non-empty vm_pu, vdi = NaN.
        vm_records = [r for r in converged if not r.vm_pu.empty]
        if vm_records:
            vdi = float(sum((r.vm_pu - 1.0).abs().sum() for r in vm_records))
        else:
            vdi = float("nan")

        # Reactive energy
        q_records = [r for r in converged if r.q_applied_mvar is not None]
        if q_records:
            q_total = sum(r.q_applied_mvar.abs().sum() for r in q_records)
        else:
            q_total = None

        # Reactive energy in proper SI units (MVArh/yr).
        # q_total_mvar_abs accumulates MVAr per timestep; × dt_s/3600 converts.
        reactive_energy_mvarh = (
            float(q_total) * (dt_s / 3600.0) if q_total is not None else None
        )

        # Curtailment
        curtailment_steps = sum(1 for r in records if r.curtailment_needed)

        curtailed_energy: Optional[float] = None

        p_records = [
            r for r in converged
            if r.p_applied_mw is not None and r.p_target_mw is not None
        ]
        if p_records:
            mwh = 0.0
            for r in p_records:
                curtailed_mw = (r.p_target_mw - r.p_applied_mw).clip(lower=0.0)
                mwh += curtailed_mw.sum() * (dt_s / 3600.0)
            curtailed_energy = mwh
        
        curtail_exhausted_steps = sum(
            1 for r in records if r.curtail_exhausted is True
            )

        # ------------------------------------------------------------------
        # Energy balance metrics
        # ------------------------------------------------------------------
        # Losses (line + trafo Joule losses) — converged timesteps only.
        loss_recs = [r for r in records if r.losses_mw is not None]
        total_losses_mwh: Optional[float] = (
            float(sum(r.losses_mw for r in loss_recs)) * (dt_s / 3600.0)
            if loss_recs else None
        )

        # Grid import / export — signed scalar split into two non-negative totals.
        grid_recs = [r for r in records if r.grid_import_mw is not None]
        if grid_recs:
            grid_import_mwh: Optional[float] = float(
                sum(max(0.0, r.grid_import_mw) for r in grid_recs)
            ) * (dt_s / 3600.0)
            grid_export_mwh: Optional[float] = float(
                sum(abs(min(0.0, r.grid_import_mw)) for r in grid_recs)
            ) * (dt_s / 3600.0)
        else:
            grid_import_mwh = grid_export_mwh = None

        # DER generation and load demand — from profiles, not PF results.
        # Populated for all timesteps regardless of convergence.
        gen_recs = [r for r in records if r.der_gen_mw is not None]
        der_gen_mwh: Optional[float] = (
            float(sum(r.der_gen_mw for r in gen_recs)) * (dt_s / 3600.0)
            if gen_recs else None
        )

        ld_recs = [r for r in records if r.load_mw is not None]
        load_demand_mwh: Optional[float] = (
            float(sum(r.load_mw for r in ld_recs)) * (dt_s / 3600.0)
            if ld_recs else None
        )

        # ------------------------------------------------------------------
        # Control effort metrics — Scenario 4 only
        # ------------------------------------------------------------------
        # coordination_active is None for all scenarios except 4.
        coord_recs = [r for r in converged if r.coordination_active is not None]
        if coord_recs:
            coordination_steps_val: Optional[int] = sum(
                1 for r in coord_recs if r.coordination_active
            )
            coordination_rate_val: Optional[float] = (
                coordination_steps_val / len(converged) if converged else 0.0
            )
        else:
            coordination_steps_val = coordination_rate_val = None

        # q_saturated_count is None for all scenarios except 4.
        sat_recs = [r for r in converged if r.q_saturated_count is not None]
        q_saturation_rate_val: Optional[float] = (
            sum(1 for r in sat_recs if r.q_saturated_count > 0) / len(converged)
            if sat_recs and converged else None
        )

        result = cls(
            scenario_id   = scenario_id,
            network_id    = network_id,
            records       = records,
            elapsed_s     = elapsed_s,
            n_timesteps                  = n_timesteps,
            n_converged                  = n_converged,
            n_violation_steps            = n_violation_steps,
            violation_duration_h         = violation_duration_h,
            total_overvoltage_bus_steps  = total_ov,
            total_undervoltage_bus_steps = total_uv,
            total_overloaded_line_steps  = total_ol,
            total_overloaded_trafo_steps = total_ot,
            max_vm_pu                    = max_vm,
            min_vm_pu                    = min_vm,
            max_line_loading_pct         = max_ll,
            max_trafo_loading_pct        = max_tl,
            vdi                          = vdi,
            q_total_mvar_abs             = q_total,
            reactive_energy_mvarh        = reactive_energy_mvarh,
            curtailment_steps            = curtailment_steps,
            curtailed_energy_mwh         = curtailed_energy,
            curtail_exhausted_steps      = curtail_exhausted_steps,
            svc_bus                      = svc_bus,
            svc_q_max                    = svc_q_max,
            total_losses_mwh             = total_losses_mwh,
            grid_import_mwh              = grid_import_mwh,
            grid_export_mwh              = grid_export_mwh,
            der_gen_mwh                  = der_gen_mwh,
            load_demand_mwh              = load_demand_mwh,
            coordination_steps           = coordination_steps_val,
            coordination_rate            = coordination_rate_val,
            q_saturation_rate            = q_saturation_rate_val,
        )

        logger.info(
            "[%s | %s] T=%d converged=%d violations=%d "
            "OV=%d UV=%d OL=%d OT=%d | max_V=%.4f min_V=%.4f | %.1f s",
            scenario_id, network_id,
            n_timesteps, n_converged, n_violation_steps,
            total_ov, total_uv, total_ol, total_ot,
            max_vm, min_vm, elapsed_s,
        )
        return result

    def summary_dict(self) -> dict:
        """
        Return a flat dict of all scalar derived metrics.

        Suitable for pd.DataFrame construction when comparing multiple
        ScenarioResult objects side-by-side.
        """
        return {
            "scenario_id":                   self.scenario_id,
            "network_id":                    self.network_id,
            "n_timesteps":                   self.n_timesteps,
            "n_converged":                   self.n_converged,
            "n_violation_steps":             self.n_violation_steps,
            "violation_duration_h":          self.violation_duration_h,
            "total_overvoltage_bus_steps":   self.total_overvoltage_bus_steps,
            "total_undervoltage_bus_steps":  self.total_undervoltage_bus_steps,
            "total_overloaded_line_steps":   self.total_overloaded_line_steps,
            "total_overloaded_trafo_steps":  self.total_overloaded_trafo_steps,
            "max_vm_pu":                     self.max_vm_pu,
            "min_vm_pu":                     self.min_vm_pu,
            "max_line_loading_pct":          self.max_line_loading_pct,
            "max_trafo_loading_pct":         self.max_trafo_loading_pct,
            "vdi":                           self.vdi,
            "q_total_mvar_abs":              self.q_total_mvar_abs,
            "reactive_energy_mvarh":         self.reactive_energy_mvarh,
            "curtailment_steps":             self.curtailment_steps,
            "curtailed_energy_mwh":          self.curtailed_energy_mwh,
            "curtail_exhausted_steps":       self.curtail_exhausted_steps,
            "svc_bus":                       self.svc_bus,
            "svc_q_max":                     self.svc_q_max,
            "total_losses_mwh":              self.total_losses_mwh,
            "grid_import_mwh":               self.grid_import_mwh,
            "grid_export_mwh":               self.grid_export_mwh,
            "der_gen_mwh":                   self.der_gen_mwh,
            "load_demand_mwh":               self.load_demand_mwh,
            "coordination_steps":            self.coordination_steps,
            "coordination_rate":             self.coordination_rate,
            "q_saturation_rate":             self.q_saturation_rate,
            "elapsed_s":                     self.elapsed_s,
        }
