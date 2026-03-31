"""
violation_detector.py
=====================
Detects voltage, thermal, and angle violations in a pandapower network
after runpp() or runpp_3ph() has converged. Returns a structured
ViolationReport (balanced) or ViolationReport3ph (unbalanced) consumed
by all downstream Tier 1 control algorithms.

This module is purely RPi-side. No Arduino communication here.
Arduino enters the loop at Tier 1 item 2 (Q(V) Volt-VAr control),
which reads from the violation report to decide whether to act.

Supported power flows
---------------------
    runpp()      → detect_violations()     → ViolationReport
    runpp_3ph()  → detect_violations_3ph() → ViolationReport3ph

Violation categories — balanced (runpp)
----------------------------------------
    Voltage  : buses outside V_MIN–V_MAX (EN 50160 / VDE-AR-N 4100/4105)
    Thermal  : lines and trafos above rated loading
    Angle    : voltage angle difference across lines above VA_DIFF_MAX_DEGREE

Violation categories — three-phase (runpp_3ph)
-----------------------------------------------
    Per-phase V  : any phase outside V_MIN–V_MAX per bus
    Unbalance    : voltage unbalance above UNBALANCE_MAX_PERCENT (IEC 62749)
    Thermal      : max-phase loading on lines and trafos above rated

Thresholds
----------
    V_MIN                = 0.95 pu   lower planning limit
    V_MAX                = 1.05 pu   upper planning limit
    LINE_MAX_LOADING     = 100  %    rated current limit
    TRAFO_MAX_LOADING    = 100  %    rated current limit
    VA_DIFF_MAX_DEGREE   = 30   °    max angle difference across a line
    UNBALANCE_MAX_PERCENT= 2.0  %    IEC 62749 voltage unbalance limit

    V_MIN/V_MAX sit inside the VDE-AR-N 4110 generator ride-through
    envelope (0.90–1.10 pu), giving the Q(V) controller room to act
    before generators are at risk of disconnection.

    All thresholds can be overridden per detect_violations() call.

Notes on runpp_3ph
------------------
runpp_3ph() requires zero-sequence parameters on all network elements:
    net.line     : r0_ohm_per_km, x0_ohm_per_km, c0_nf_per_km
    net.trafo    : vk0_percent, vkr0_percent, mag0_percent, mag0_rx,
                   vector_group, si0_hv_partial
    net.ext_grid : s_sc_max_mva, rx_max, x0x_max, r0x0_max

Standard pandapower test networks do not include this by default.
Use pp.add_zero_impedance_parameters(net) for networks built with typed
standard cable types that include zero-sequence data. For networks built
from parameters, supply r0_ohm_per_km etc. in create_line_from_parameters().

Usage
-----
    from violation_detector import detect_violations, detect_violations_3ph

    # Balanced (all five HIL scenarios)
    pp.runpp(net, voltage_depend_loads=False)
    report = detect_violations(net)
    if report.any_violations:
        ...  # pass to Q(V) controller, Q coordination, curtailment

    # Three-phase (LV networks with zero-sequence data)
    pp.runpp_3ph(net)
    report = detect_violations_3ph(net)
    if report.any_violations:
        ...
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"simbench\.converter\.csv_pp_converter"
)

# ===========================================================================
# Thresholds — edit here only, all scenarios inherit automatically
# ===========================================================================

V_MIN                 = 0.95   # pu   lower voltage planning limit
V_MAX                 = 1.05   # pu   upper voltage planning limit
LINE_MAX_LOADING      = 100.0  # %    line thermal limit
TRAFO_MAX_LOADING     = 100.0  # %    transformer thermal limit
VA_DIFF_MAX_DEGREE    = 30.0   # deg  max angle difference across a line
UNBALANCE_MAX_PERCENT = 2.0    # %    IEC 62749 voltage unbalance limit

# Newton-Raphson noise tolerances — prevents spurious violations from
# floating-point residuals in the HIL closed loop.
# 1e-6 pu ≈ 0.02 V on 20 kV — well below any real violation magnitude.
VOLTAGE_EPSILON = 1e-6   # pu
LOADING_EPSILON = 1e-6   # %
ANGLE_EPSILON   = 1e-4   # degrees


# ===========================================================================
# ViolationReport — balanced (runpp)
# ===========================================================================

@dataclass
class ViolationReport:
    """
    Structured snapshot of planning constraint violations from runpp().
    Produced by detect_violations(). Consumed by all Tier 1 algorithms.

    Attributes
    ----------
    over_voltage : pd.DataFrame
        Buses above V_MAX. Columns: vm_pu, deviation_pu.
        deviation_pu = vm_pu - V_MAX  (always positive).

    under_voltage : pd.DataFrame
        Buses below V_MIN. Columns: vm_pu, deviation_pu.
        deviation_pu = V_MIN - vm_pu  (always positive).

    overloaded_lines : pd.DataFrame
        Lines above LINE_MAX_LOADING. Columns: loading_percent, excess_percent.

    overloaded_trafos : pd.DataFrame
        Trafos above TRAFO_MAX_LOADING. Columns: loading_percent, excess_percent.

    angle_violations : pd.DataFrame
        Lines with |va_from - va_to| > VA_DIFF_MAX_DEGREE.
        Columns: va_from_degree, va_to_degree, va_diff_degree.

    any_violations : bool
        True if any violation of any kind exists. Gate condition for all
        downstream control algorithms — check this before computing anything.

    converged : bool
        Whether runpp() produced valid results. If False, all frames are
        empty with correct column schemas. Callers must handle this.

    v_min_used, v_max_used : float
        Thresholds used, stored for traceability in result logs.

    Notes
    -----
    All DataFrames use pandapower element indices as their index, enabling
    direct joins against net.bus, net.line, net.trafo, net.sgen without
    index translation.
    """

    over_voltage     : pd.DataFrame = field(default_factory=pd.DataFrame)
    under_voltage    : pd.DataFrame = field(default_factory=pd.DataFrame)
    overloaded_lines : pd.DataFrame = field(default_factory=pd.DataFrame)
    overloaded_trafos: pd.DataFrame = field(default_factory=pd.DataFrame)
    angle_violations : pd.DataFrame = field(default_factory=pd.DataFrame)

    any_violations   : bool  = False
    converged        : bool  = True

    v_min_used       : float = V_MIN
    v_max_used       : float = V_MAX

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def any_voltage_violations(self) -> bool:
        """True if any bus is outside the voltage band."""
        return not self.over_voltage.empty or not self.under_voltage.empty

    @property
    def any_thermal_violations(self) -> bool:
        """True if any line or trafo is overloaded."""
        return not self.overloaded_lines.empty or not self.overloaded_trafos.empty

    @property
    def any_angle_violations(self) -> bool:
        """True if any line exceeds the angle difference limit."""
        return not self.angle_violations.empty

    @property
    def n_over_voltage(self) -> int:
        return len(self.over_voltage)

    @property
    def n_under_voltage(self) -> int:
        return len(self.under_voltage)

    @property
    def n_overloaded_lines(self) -> int:
        return len(self.overloaded_lines)

    @property
    def n_overloaded_trafos(self) -> int:
        return len(self.overloaded_trafos)

    @property
    def n_angle_violations(self) -> int:
        return len(self.angle_violations)

    @property
    def worst_over_voltage(self) -> Optional[float]:
        """Highest vm_pu among overvoltage buses. None if no violation."""
        if self.over_voltage.empty:
            return None
        return float(self.over_voltage["vm_pu"].max())

    @property
    def worst_under_voltage(self) -> Optional[float]:
        """Lowest vm_pu among undervoltage buses. None if no violation."""
        if self.under_voltage.empty:
            return None
        return float(self.under_voltage["vm_pu"].min())

    @property
    def worst_line_loading(self) -> Optional[float]:
        """Highest loading_percent among overloaded lines. None if no violation."""
        if self.overloaded_lines.empty:
            return None
        return float(self.overloaded_lines["loading_percent"].max())

    @property
    def worst_trafo_loading(self) -> Optional[float]:
        """Highest loading_percent among overloaded trafos. None if no violation."""
        if self.overloaded_trafos.empty:
            return None
        return float(self.overloaded_trafos["loading_percent"].max())

    @property
    def worst_angle_diff(self) -> Optional[float]:
        """Largest angle difference across a single line. None if no violation."""
        if self.angle_violations.empty:
            return None
        return float(self.angle_violations["va_diff_degree"].max())

    @property
    def most_severe_over_voltage_buses(self) -> pd.Index:
        """Bus indices sorted by deviation descending — for Q coordination."""
        if self.over_voltage.empty:
            return pd.Index([])
        return self.over_voltage.sort_values("deviation_pu", ascending=False).index

    @property
    def most_severe_under_voltage_buses(self) -> pd.Index:
        """Bus indices sorted by deviation descending — for Q coordination."""
        if self.under_voltage.empty:
            return pd.Index([])
        return self.under_voltage.sort_values("deviation_pu", ascending=False).index

    # -----------------------------------------------------------------------
    # Summary and detail output
    # -----------------------------------------------------------------------

    def summary(self) -> str:
        """One-line human-readable summary of all violations at this timestep."""
        if not self.converged:
            return "ViolationReport: runpp() did not converge — no results."
        if not self.any_violations:
            return (f"ViolationReport: no violations "
                    f"(V {self.v_min_used}–{self.v_max_used} pu | "
                    f"thermal {LINE_MAX_LOADING:.0f}% | "
                    f"angle {VA_DIFF_MAX_DEGREE:.0f}°)")
        parts = []
        if self.n_over_voltage:
            parts.append(
                f"{self.n_over_voltage} overvoltage bus(es) "
                f"[worst: {self.worst_over_voltage:.4f} pu]"
            )
        if self.n_under_voltage:
            parts.append(
                f"{self.n_under_voltage} undervoltage bus(es) "
                f"[worst: {self.worst_under_voltage:.4f} pu]"
            )
        if self.n_overloaded_lines:
            parts.append(
                f"{self.n_overloaded_lines} overloaded line(s) "
                f"[worst: {self.worst_line_loading:.1f}%]"
            )
        if self.n_overloaded_trafos:
            parts.append(
                f"{self.n_overloaded_trafos} overloaded trafo(s) "
                f"[worst: {self.worst_trafo_loading:.1f}%]"
            )
        if self.n_angle_violations:
            parts.append(
                f"{self.n_angle_violations} angle violation(s) "
                f"[worst: {self.worst_angle_diff:.1f}°]"
            )
        return "ViolationReport: " + " | ".join(parts)

    def detail(self) -> str:
        """Multi-line breakdown of all violations — for debug and reporting."""
        lines = [self.summary()]
        if not self.over_voltage.empty:
            lines.append("\nOvervoltage buses (vm_pu, deviation_pu):")
            lines.append(self.over_voltage.to_string())
        if not self.under_voltage.empty:
            lines.append("\nUndervoltage buses (vm_pu, deviation_pu):")
            lines.append(self.under_voltage.to_string())
        if not self.overloaded_lines.empty:
            lines.append("\nOverloaded lines (loading_percent, excess_percent):")
            lines.append(self.overloaded_lines.to_string())
        if not self.overloaded_trafos.empty:
            lines.append("\nOverloaded trafos (loading_percent, excess_percent):")
            lines.append(self.overloaded_trafos.to_string())
        if not self.angle_violations.empty:
            lines.append(
                "\nAngle violations (va_from_degree, va_to_degree, va_diff_degree):"
            )
            lines.append(self.angle_violations.to_string())
        return "\n".join(lines)


# ===========================================================================
# ViolationReport3ph — three-phase (runpp_3ph)
# ===========================================================================

@dataclass
class ViolationReport3ph:
    """
    Structured snapshot of planning constraint violations from runpp_3ph().
    Produced by detect_violations_3ph().

    Attributes
    ----------
    over_voltage : pd.DataFrame
        Buses where any phase exceeds V_MAX.
        Columns: vm_a_pu, vm_b_pu, vm_c_pu, worst_phase, worst_vm_pu,
                 deviation_pu.

    under_voltage : pd.DataFrame
        Buses where any phase is below V_MIN.
        Columns: vm_a_pu, vm_b_pu, vm_c_pu, worst_phase, worst_vm_pu,
                 deviation_pu.

    unbalance_violations : pd.DataFrame
        Buses exceeding UNBALANCE_MAX_PERCENT (IEC 62749).
        Columns: unbalance_percent, deviation_percent.

    overloaded_lines : pd.DataFrame
        Lines where max-phase loading exceeds LINE_MAX_LOADING.
        Columns: loading_percent, excess_percent.
        loading_percent = maximum across all three phases.

    overloaded_trafos : pd.DataFrame
        Trafos where max-phase loading exceeds TRAFO_MAX_LOADING.
        Columns: loading_percent, excess_percent.

    any_violations : bool
        True if any violation of any kind exists.

    converged : bool
        Whether runpp_3ph() produced valid results.

    Notes
    -----
    runpp_3ph() requires fully parameterised zero-sequence network data.
    Standard pandapower test networks require manual parameterisation.
    See module docstring for required parameters per element type.
    """

    over_voltage         : pd.DataFrame = field(default_factory=pd.DataFrame)
    under_voltage        : pd.DataFrame = field(default_factory=pd.DataFrame)
    unbalance_violations : pd.DataFrame = field(default_factory=pd.DataFrame)
    overloaded_lines     : pd.DataFrame = field(default_factory=pd.DataFrame)
    overloaded_trafos    : pd.DataFrame = field(default_factory=pd.DataFrame)

    any_violations      : bool  = False
    converged           : bool  = True

    v_min_used          : float = V_MIN
    v_max_used          : float = V_MAX
    unbalance_max_used  : float = UNBALANCE_MAX_PERCENT

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def any_voltage_violations(self) -> bool:
        return not self.over_voltage.empty or not self.under_voltage.empty

    @property
    def any_unbalance_violations(self) -> bool:
        return not self.unbalance_violations.empty

    @property
    def any_thermal_violations(self) -> bool:
        return not self.overloaded_lines.empty or not self.overloaded_trafos.empty

    @property
    def n_over_voltage(self) -> int:
        return len(self.over_voltage)

    @property
    def n_under_voltage(self) -> int:
        return len(self.under_voltage)

    @property
    def n_unbalance_violations(self) -> int:
        return len(self.unbalance_violations)

    @property
    def n_overloaded_lines(self) -> int:
        return len(self.overloaded_lines)

    @property
    def n_overloaded_trafos(self) -> int:
        return len(self.overloaded_trafos)

    @property
    def worst_over_voltage(self) -> Optional[float]:
        if self.over_voltage.empty:
            return None
        return float(self.over_voltage["worst_vm_pu"].max())

    @property
    def worst_under_voltage(self) -> Optional[float]:
        if self.under_voltage.empty:
            return None
        return float(self.under_voltage["worst_vm_pu"].min())

    @property
    def worst_unbalance(self) -> Optional[float]:
        if self.unbalance_violations.empty:
            return None
        return float(self.unbalance_violations["unbalance_percent"].max())

    @property
    def worst_line_loading(self) -> Optional[float]:
        if self.overloaded_lines.empty:
            return None
        return float(self.overloaded_lines["loading_percent"].max())

    @property
    def worst_trafo_loading(self) -> Optional[float]:
        if self.overloaded_trafos.empty:
            return None
        return float(self.overloaded_trafos["loading_percent"].max())

    def summary(self) -> str:
        """One-line summary of all three-phase violations."""
        if not self.converged:
            return "ViolationReport3ph: runpp_3ph() did not converge — no results."
        if not self.any_violations:
            return (f"ViolationReport3ph: no violations "
                    f"(V {self.v_min_used}–{self.v_max_used} pu | "
                    f"thermal {LINE_MAX_LOADING:.0f}% | "
                    f"unbalance {self.unbalance_max_used:.0f}%)")
        parts = []
        if self.n_over_voltage:
            parts.append(
                f"{self.n_over_voltage} overvoltage bus(es) "
                f"[worst: {self.worst_over_voltage:.4f} pu]"
            )
        if self.n_under_voltage:
            parts.append(
                f"{self.n_under_voltage} undervoltage bus(es) "
                f"[worst: {self.worst_under_voltage:.4f} pu]"
            )
        if self.n_unbalance_violations:
            parts.append(
                f"{self.n_unbalance_violations} unbalance bus(es) "
                f"[worst: {self.worst_unbalance:.2f}%]"
            )
        if self.n_overloaded_lines:
            parts.append(
                f"{self.n_overloaded_lines} overloaded line(s) "
                f"[worst: {self.worst_line_loading:.1f}%]"
            )
        if self.n_overloaded_trafos:
            parts.append(
                f"{self.n_overloaded_trafos} overloaded trafo(s) "
                f"[worst: {self.worst_trafo_loading:.1f}%]"
            )
        return "ViolationReport3ph: " + " | ".join(parts)

    def detail(self) -> str:
        """Multi-line breakdown — for debug and reporting."""
        lines = [self.summary()]
        if not self.over_voltage.empty:
            lines.append(
                "\nOvervoltage buses (per-phase vm_pu, worst_phase, deviation_pu):"
            )
            lines.append(self.over_voltage.to_string())
        if not self.under_voltage.empty:
            lines.append(
                "\nUndervoltage buses (per-phase vm_pu, worst_phase, deviation_pu):"
            )
            lines.append(self.under_voltage.to_string())
        if not self.unbalance_violations.empty:
            lines.append(
                "\nUnbalance violations (unbalance_percent, deviation_percent):"
            )
            lines.append(self.unbalance_violations.to_string())
        if not self.overloaded_lines.empty:
            lines.append("\nOverloaded lines (loading_percent, excess_percent):")
            lines.append(self.overloaded_lines.to_string())
        if not self.overloaded_trafos.empty:
            lines.append("\nOverloaded trafos (loading_percent, excess_percent):")
            lines.append(self.overloaded_trafos.to_string())
        return "\n".join(lines)


# ===========================================================================
# Internal helpers — balanced (runpp)
# ===========================================================================

def _runpp_converged(net) -> bool:
    """
    Returns True only if runpp() both converged and produced populated results.

    pandapower can set net.converged=True in edge cases where result tables
    are empty. Checking res_bus is the simplest secondary guard — if voltage
    results exist, line/trafo results will too.
    """
    flag_ok    = bool(getattr(net, "converged", False))
    results_ok = (
        hasattr(net, "res_bus")
        and not net.res_bus.empty
        and "vm_pu" in net.res_bus.columns
    )
    return flag_ok and results_ok


def _check_voltage(net,
                   v_min: float,
                   v_max: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits res_bus.vm_pu into overvoltage and undervoltage frames.
    Restricts to in-service buses. Applies epsilon tolerance to suppress
    Newton-Raphson floating-point noise.

    Returns
    -------
    over_v  : DataFrame  index=bus_idx  columns=[vm_pu, deviation_pu]
    under_v : DataFrame  index=bus_idx  columns=[vm_pu, deviation_pu]
    """
    _empty = pd.DataFrame(columns=["vm_pu", "deviation_pu"])

    if "vm_pu" not in net.res_bus.columns or net.res_bus.empty:
        return _empty.copy(), _empty.copy()

    vm = net.res_bus["vm_pu"].copy()
    in_service = net.bus.index[net.bus["in_service"] == True]
    vm = vm.loc[vm.index.intersection(in_service)].dropna()

    over_mask  = vm > (v_max + VOLTAGE_EPSILON)
    under_mask = vm < (v_min - VOLTAGE_EPSILON)

    over_v = pd.DataFrame({
        "vm_pu":        vm[over_mask],
        "deviation_pu": vm[over_mask] - v_max,
    })
    under_v = pd.DataFrame({
        "vm_pu":        vm[under_mask],
        "deviation_pu": v_min - vm[under_mask],
    })
    return over_v, under_v


def _check_lines(net, max_loading: float) -> pd.DataFrame:
    """
    Returns lines exceeding max_loading percent.

    Returns
    -------
    DataFrame  index=line_idx  columns=[loading_percent, excess_percent]
    """
    _empty = pd.DataFrame(columns=["loading_percent", "excess_percent"])

    if net.res_line.empty:
        warnings.warn(
            "[violation_detector] res_line is empty after runpp(). "
            "Line thermal results unavailable — check runpp() completed normally.",
            RuntimeWarning, stacklevel=3,
        )
        return _empty.copy()

    if "loading_percent" not in net.res_line.columns:
        warnings.warn(
            "[violation_detector] 'loading_percent' missing from res_line. "
            "Line thermal violations cannot be detected.",
            RuntimeWarning, stacklevel=3,
        )
        return _empty.copy()

    loading = net.res_line["loading_percent"].copy()
    in_service = net.line.index[net.line["in_service"] == True]
    loading = loading.loc[loading.index.intersection(in_service)].dropna()

    mask = loading > (max_loading + LOADING_EPSILON)
    return pd.DataFrame({
        "loading_percent": loading[mask],
        "excess_percent":  loading[mask] - max_loading,
    })


def _check_trafos(net, max_loading: float) -> pd.DataFrame:
    """
    Returns trafos exceeding max_loading percent.

    Returns
    -------
    DataFrame  index=trafo_idx  columns=[loading_percent, excess_percent]
    """
    _empty = pd.DataFrame(columns=["loading_percent", "excess_percent"])

    if net.res_trafo.empty:
        warnings.warn(
            "[violation_detector] res_trafo is empty after runpp(). "
            "Trafo thermal results unavailable.",
            RuntimeWarning, stacklevel=3,
        )
        return _empty.copy()

    if "loading_percent" not in net.res_trafo.columns:
        warnings.warn(
            "[violation_detector] 'loading_percent' missing from res_trafo. "
            "Trafo thermal violations cannot be detected.",
            RuntimeWarning, stacklevel=3,
        )
        return _empty.copy()

    loading = net.res_trafo["loading_percent"].copy()
    in_service = net.trafo.index[net.trafo["in_service"] == True]
    loading = loading.loc[loading.index.intersection(in_service)].dropna()

    mask = loading > (max_loading + LOADING_EPSILON)
    return pd.DataFrame({
        "loading_percent": loading[mask],
        "excess_percent":  loading[mask] - max_loading,
    })


def _check_angle_diff(net, va_diff_max: float) -> pd.DataFrame:
    """
    Returns lines where |va_from_degree - va_to_degree| exceeds va_diff_max.

    Large angle differences across lines indicate heavily stressed feeders
    and potential stability concerns, particularly on long rural MV cables.
    Requires calculate_voltage_angles=True in runpp() (pandapower default
    for networks with buses above 70 kV; auto for MV).

    Returns
    -------
    DataFrame  index=line_idx
               columns=[va_from_degree, va_to_degree, va_diff_degree]
    """
    _empty = pd.DataFrame(
        columns=["va_from_degree", "va_to_degree", "va_diff_degree"]
    )

    if net.res_line.empty:
        return _empty.copy()

    needed = {"va_from_degree", "va_to_degree"}
    if not needed.issubset(net.res_line.columns):
        return _empty.copy()

    va_from = net.res_line["va_from_degree"].copy()
    va_to   = net.res_line["va_to_degree"].copy()
    in_service = net.line.index[net.line["in_service"] == True]
    va_from = va_from.loc[va_from.index.intersection(in_service)]
    va_to   = va_to.loc[va_to.index.intersection(in_service)]

    va_diff  = (va_from - va_to).abs().dropna()
    viol_idx = va_diff.index[va_diff > (va_diff_max + ANGLE_EPSILON)]

    return pd.DataFrame({
        "va_from_degree": va_from.loc[viol_idx],
        "va_to_degree":   va_to.loc[viol_idx],
        "va_diff_degree": va_diff.loc[viol_idx],
    })


# ===========================================================================
# Internal helpers — three-phase (runpp_3ph)
# ===========================================================================

def _runpp_3ph_converged(net) -> bool:
    """
    Returns True only if runpp_3ph() both converged and produced results.
    Uses res_bus_3ph as the primary guard (analogous to res_bus for runpp).
    """
    flag_ok    = bool(getattr(net, "converged", False))
    results_ok = (
        hasattr(net, "res_bus_3ph")
        and not net.res_bus_3ph.empty
        and "vm_a_pu" in net.res_bus_3ph.columns
    )
    return flag_ok and results_ok


def _check_voltage_3ph(net,
                       v_min: float,
                       v_max: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Checks per-phase voltages from res_bus_3ph against the voltage band.
    A bus is flagged if any single phase violates. The worst phase is
    identified for each flagged bus.

    Returns
    -------
    over_v  : DataFrame  index=bus_idx
              columns=[vm_a_pu, vm_b_pu, vm_c_pu,
                       worst_phase, worst_vm_pu, deviation_pu]
    under_v : DataFrame  same columns
    """
    _cols = [
        "vm_a_pu", "vm_b_pu", "vm_c_pu",
        "worst_phase", "worst_vm_pu", "deviation_pu",
    ]
    _empty = pd.DataFrame(columns=_cols)

    needed = {"vm_a_pu", "vm_b_pu", "vm_c_pu"}
    if not needed.issubset(net.res_bus_3ph.columns) or net.res_bus_3ph.empty:
        return _empty.copy(), _empty.copy()

    phase_cols = ["vm_a_pu", "vm_b_pu", "vm_c_pu"]
    df = net.res_bus_3ph[phase_cols].copy()
    in_service = net.bus.index[net.bus["in_service"] == True]
    df = df.loc[df.index.intersection(in_service)].dropna()

    # Overvoltage: any phase above V_MAX
    over_mask = (df > (v_max + VOLTAGE_EPSILON)).any(axis=1)
    if over_mask.any():
        over_df = df[over_mask].copy()
        over_df["worst_phase"]  = over_df[phase_cols].idxmax(axis=1)
        over_df["worst_vm_pu"]  = over_df[phase_cols].max(axis=1)
        over_df["deviation_pu"] = over_df["worst_vm_pu"] - v_max
    else:
        over_df = _empty.copy()

    # Undervoltage: any phase below V_MIN
    under_mask = (df < (v_min - VOLTAGE_EPSILON)).any(axis=1)
    if under_mask.any():
        under_df = df[under_mask].copy()
        under_df["worst_phase"]  = under_df[phase_cols].idxmin(axis=1)
        under_df["worst_vm_pu"]  = under_df[phase_cols].min(axis=1)
        under_df["deviation_pu"] = v_min - under_df["worst_vm_pu"]
    else:
        under_df = _empty.copy()

    return over_df, under_df


def _check_unbalance(net, unbalance_max: float) -> pd.DataFrame:
    """
    Returns buses where voltage unbalance exceeds unbalance_max percent.
    unbalance_percent is computed by pandapower per IEC 62749.

    Returns
    -------
    DataFrame  index=bus_idx  columns=[unbalance_percent, deviation_percent]
    """
    _empty = pd.DataFrame(columns=["unbalance_percent", "deviation_percent"])

    if net.res_bus_3ph.empty or "unbalance_percent" not in net.res_bus_3ph.columns:
        return _empty.copy()

    ub = net.res_bus_3ph["unbalance_percent"].copy()
    in_service = net.bus.index[net.bus["in_service"] == True]
    ub = ub.loc[ub.index.intersection(in_service)].dropna()

    mask = ub > (unbalance_max + LOADING_EPSILON)
    return pd.DataFrame({
        "unbalance_percent": ub[mask],
        "deviation_percent": ub[mask] - unbalance_max,
    })


def _check_lines_3ph(net, max_loading: float) -> pd.DataFrame:
    """
    Returns lines where max-phase loading exceeds max_loading percent.
    Uses res_line_3ph["loading_percent"] = maximum across all three phases.

    Returns
    -------
    DataFrame  index=line_idx  columns=[loading_percent, excess_percent]
    """
    _empty = pd.DataFrame(columns=["loading_percent", "excess_percent"])

    if not hasattr(net, "res_line_3ph") or net.res_line_3ph.empty:
        warnings.warn(
            "[violation_detector] res_line_3ph is empty after runpp_3ph(). "
            "Line thermal results unavailable.",
            RuntimeWarning, stacklevel=3,
        )
        return _empty.copy()

    if "loading_percent" not in net.res_line_3ph.columns:
        warnings.warn(
            "[violation_detector] 'loading_percent' missing from res_line_3ph.",
            RuntimeWarning, stacklevel=3,
        )
        return _empty.copy()

    loading = net.res_line_3ph["loading_percent"].copy()
    in_service = net.line.index[net.line["in_service"] == True]
    loading = loading.loc[loading.index.intersection(in_service)].dropna()

    mask = loading > (max_loading + LOADING_EPSILON)
    return pd.DataFrame({
        "loading_percent": loading[mask],
        "excess_percent":  loading[mask] - max_loading,
    })


def _check_trafos_3ph(net, max_loading: float) -> pd.DataFrame:
    """
    Returns trafos where max-phase loading exceeds max_loading percent.
    Uses res_trafo_3ph["loading_percent"] (max across phases).

    Returns
    -------
    DataFrame  index=trafo_idx  columns=[loading_percent, excess_percent]
    """
    _empty = pd.DataFrame(columns=["loading_percent", "excess_percent"])

    if not hasattr(net, "res_trafo_3ph") or net.res_trafo_3ph.empty:
        warnings.warn(
            "[violation_detector] res_trafo_3ph is empty after runpp_3ph(). "
            "Trafo thermal results unavailable.",
            RuntimeWarning, stacklevel=3,
        )
        return _empty.copy()

    if "loading_percent" not in net.res_trafo_3ph.columns:
        warnings.warn(
            "[violation_detector] 'loading_percent' missing from res_trafo_3ph.",
            RuntimeWarning, stacklevel=3,
        )
        return _empty.copy()

    loading = net.res_trafo_3ph["loading_percent"].copy()
    in_service = net.trafo.index[net.trafo["in_service"] == True]
    loading = loading.loc[loading.index.intersection(in_service)].dropna()

    mask = loading > (max_loading + LOADING_EPSILON)
    return pd.DataFrame({
        "loading_percent": loading[mask],
        "excess_percent":  loading[mask] - max_loading,
    })


# ===========================================================================
# Public entry points
# ===========================================================================

def detect_violations(
        net,
        v_min:       float = V_MIN,
        v_max:       float = V_MAX,
        line_max:    float = LINE_MAX_LOADING,
        trafo_max:   float = TRAFO_MAX_LOADING,
        va_diff_max: float = VA_DIFF_MAX_DEGREE,
) -> ViolationReport:
    """
    Detect all planning constraint violations after runpp() has been called.

    Sole entry point for all five comparison scenarios and hosting capacity
    analysis. Downstream algorithms consume the returned ViolationReport —
    they never re-query pandapower results directly.

    Parameters
    ----------
    net         : pandapower network (runpp() must have been called first)
    v_min       : lower voltage limit pu       (default 0.95)
    v_max       : upper voltage limit pu       (default 1.05)
    line_max    : line thermal limit %         (default 100)
    trafo_max   : trafo thermal limit %        (default 100)
    va_diff_max : max line angle difference °  (default 30)

    Returns
    -------
    ViolationReport
        If runpp() did not converge, report.converged=False and all frames
        are empty with correct column schemas.

    Examples
    --------
    Scenario 4 HIL manual loop::

        net.sgen.p_mw = p_array[t]
        pp.runpp(net, voltage_depend_loads=False)
        report = detect_violations(net)

        if report.any_violations:
            q_setpoints = arduino_exchange(net.res_bus.vm_pu)
            net.sgen.q_mvar = q_setpoints
            pp.runpp(net, voltage_depend_loads=False)
            report_post = detect_violations(net)

    Hosting capacity analysis::

        while True:
            scale_pv(net, pv_scale)
            pp.runpp(net, voltage_depend_loads=False)
            report = detect_violations(net)
            if report.any_violations:
                break  # first violation found — record hosting capacity
            pv_scale += step
    """
    if not _runpp_converged(net):
        return ViolationReport(converged=False)

    over_v, under_v   = _check_voltage(net, v_min, v_max)
    overloaded_lines  = _check_lines(net, line_max)
    overloaded_trafos = _check_trafos(net, trafo_max)
    angle_viols       = _check_angle_diff(net, va_diff_max)

    any_violations = (
        not over_v.empty
        or not under_v.empty
        or not overloaded_lines.empty
        or not overloaded_trafos.empty
        or not angle_viols.empty
    )

    return ViolationReport(
        over_voltage      = over_v,
        under_voltage     = under_v,
        overloaded_lines  = overloaded_lines,
        overloaded_trafos = overloaded_trafos,
        angle_violations  = angle_viols,
        any_violations    = any_violations,
        converged         = True,
        v_min_used        = v_min,
        v_max_used        = v_max,
    )


def detect_violations_3ph(
        net,
        v_min:         float = V_MIN,
        v_max:         float = V_MAX,
        line_max:      float = LINE_MAX_LOADING,
        trafo_max:     float = TRAFO_MAX_LOADING,
        unbalance_max: float = UNBALANCE_MAX_PERCENT,
) -> ViolationReport3ph:
    """
    Detect all planning constraint violations after runpp_3ph() has been called.

    Parameters
    ----------
    net           : pandapower network (runpp_3ph() must have been called first)
    v_min         : lower voltage limit pu        (default 0.95)
    v_max         : upper voltage limit pu        (default 1.05)
    line_max      : line thermal limit %          (default 100)
    trafo_max     : trafo thermal limit %         (default 100)
    unbalance_max : voltage unbalance limit %     (default 2.0, IEC 62749)

    Returns
    -------
    ViolationReport3ph
        If runpp_3ph() did not converge, report.converged=False and all
        frames are empty with correct column schemas.

    Notes
    -----
    runpp_3ph() requires zero-sequence parameters on all network elements.
    Standard pandapower test networks require manual parameterisation first.
    See module docstring for required parameters per element type.

    Example — custom parameterised LV network::

        pp.add_zero_impedance_parameters(net)  # for typed standard cables
        net.ext_grid["s_sc_max_mva"] = 1000
        net.ext_grid["x0x_max"]      = 1.0
        net.ext_grid["r0x0_max"]     = 0.1
        pp.runpp_3ph(net)
        report = detect_violations_3ph(net)
    """
    if not _runpp_3ph_converged(net):
        return ViolationReport3ph(converged=False)

    over_v, under_v   = _check_voltage_3ph(net, v_min, v_max)
    unbalance_viols   = _check_unbalance(net, unbalance_max)
    overloaded_lines  = _check_lines_3ph(net, line_max)
    overloaded_trafos = _check_trafos_3ph(net, trafo_max)

    any_violations = (
        not over_v.empty
        or not under_v.empty
        or not unbalance_viols.empty
        or not overloaded_lines.empty
        or not overloaded_trafos.empty
    )

    return ViolationReport3ph(
        over_voltage         = over_v,
        under_voltage        = under_v,
        unbalance_violations = unbalance_viols,
        overloaded_lines     = overloaded_lines,
        overloaded_trafos    = overloaded_trafos,
        any_violations       = any_violations,
        converged            = True,
        v_min_used           = v_min,
        v_max_used           = v_max,
        unbalance_max_used   = unbalance_max,
    )
