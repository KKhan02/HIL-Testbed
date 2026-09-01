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

    q_total_mvar_abs : float or None
        Sum of |q_applied_mvar| across all sgens and all timesteps.
        None for Scenario 1 (no reactive control).

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
    """
    scenario_id:    str
    network_id:     str
    records:        List[TimestepRecord]
    elapsed_s:      float

    # Derived fields — populated by from_records(), not set by caller directly.
    n_timesteps:                  int   = field(default=0)
    n_converged:                  int   = field(default=0)
    n_violation_steps:            int   = field(default=0)
    total_overvoltage_bus_steps:  int   = field(default=0)
    total_undervoltage_bus_steps: int   = field(default=0)
    total_overloaded_line_steps:  int   = field(default=0)
    total_overloaded_trafo_steps: int   = field(default=0)
    max_vm_pu:                    float = field(default=float("nan"))
    min_vm_pu:                    float = field(default=float("nan"))
    max_line_loading_pct:         float = field(default=float("nan"))
    max_trafo_loading_pct:        float = field(default=float("nan"))
    q_total_mvar_abs:             Optional[float] = field(default=None)
    curtailment_steps:            int              = field(default=0)
    curtailed_energy_mwh:         Optional[float]  = field(default=None)
    svc_bus:       int | None   = None  # fixed SVC bus index (Scenario 3 only)
    svc_q_max:     float | None = None  # Q_MAX used (0.20 × trafo sn_mva sum)

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
        converged = [r for r in records if r.converged]

        n_timesteps  = len(records)
        n_converged  = len(converged)

        n_violation_steps = sum(
            1 for r in converged
            if r.over_voltage_buses or r.under_voltage_buses
               or r.overloaded_lines or r.overloaded_trafos
        )
        total_ov  = sum(len(r.over_voltage_buses)  for r in converged)
        total_uv  = sum(len(r.under_voltage_buses) for r in converged)
        total_ol  = sum(len(r.overloaded_lines)    for r in converged)
        total_ot  = sum(len(r.overloaded_trafos)   for r in converged)

        # Voltage extremes
        if converged:
            max_vm = max(r.vm_pu.max() for r in converged if not r.vm_pu.empty)
            min_vm = min(r.vm_pu.min() for r in converged if not r.vm_pu.empty)
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

        # Reactive energy
        q_records = [r for r in converged if r.q_applied_mvar is not None]
        if q_records:
            q_total = sum(r.q_applied_mvar.abs().sum() for r in q_records)
        else:
            q_total = None

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

        result = cls(
            scenario_id   = scenario_id,
            network_id    = network_id,
            records       = records,
            elapsed_s     = elapsed_s,
            n_timesteps                  = n_timesteps,
            n_converged                  = n_converged,
            n_violation_steps            = n_violation_steps,
            total_overvoltage_bus_steps  = total_ov,
            total_undervoltage_bus_steps = total_uv,
            total_overloaded_line_steps  = total_ol,
            total_overloaded_trafo_steps = total_ot,
            max_vm_pu                    = max_vm,
            min_vm_pu                    = min_vm,
            max_line_loading_pct         = max_ll,
            max_trafo_loading_pct        = max_tl,
            q_total_mvar_abs             = q_total,
            curtailment_steps            = curtailment_steps,
            curtailed_energy_mwh         = curtailed_energy,
            svc_bus                      = svc_bus,
            svc_q_max                    = svc_q_max,
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
            "total_overvoltage_bus_steps":   self.total_overvoltage_bus_steps,
            "total_undervoltage_bus_steps":  self.total_undervoltage_bus_steps,
            "total_overloaded_line_steps":   self.total_overloaded_line_steps,
            "total_overloaded_trafo_steps":  self.total_overloaded_trafo_steps,
            "max_vm_pu":                     self.max_vm_pu,
            "min_vm_pu":                     self.min_vm_pu,
            "max_line_loading_pct":          self.max_line_loading_pct,
            "max_trafo_loading_pct":         self.max_trafo_loading_pct,
            "q_total_mvar_abs":              self.q_total_mvar_abs,
            "curtailment_steps":             self.curtailment_steps,
            "curtailed_energy_mwh":          self.curtailed_energy_mwh,
            "svc_bus":                       self.svc_bus,
            "svc_q_max":                     self.svc_q_max,
            "elapsed_s":                     self.elapsed_s,
        }
