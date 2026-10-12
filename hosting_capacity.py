"""
hosting_capacity.py
===================
Hosting Capacity (HC) analysis — deterministic worst-case snapshot method.

Not a standalone scenario.  Runs as a post-simulation analysis called by
benchmark_runner.py after the five comparison scenarios complete.

Two public functions
--------------------
run_baseline_hc(net, network_id)
    Case A — no voltage control.  Increments PV at the end-of-feeder bus
    until the first voltage violation.  HC is the last violation-free
    total PV injection.

run_hc_with_volt_var(net, network_id)
    Case B — VDE-AR-N 4110 Q(V) Volt-Var active.  Same sweep as Case A
    but wraps each snapshot runpp() in a Q(V) fixed-point iteration loop
    using QVCharacteristic.compute_setpoints() via VoltVarController
    (dry_run=True).  HC gain = hc_mw(B) − hc_mw(A).

Methodology
-----------
Traditional deterministic HC is a static worst-case snapshot, not a
quasi-static time-series (QSTS) sweep.  For overvoltage — the binding
constraint for PV integration — the worst case is:

    minimum load  ×  maximum irradiance (PV output = 1.0 p.u.)

This is applied once before the sweep begins and held fixed across all
steps.  A QSTS sweep per step would multiply runtime by ~52,560 (annual
10-min resolution) and is not justified for a baseline benchmark.

"PV output = 1.0 p.u." is 1.0 p.u. of rated ACTIVE power, not of sn_mva.
sn_mva is the inverter's apparent rating, so setting p_mw = sn_mva forces
cos φ = 1 and leaves |Q| ≤ sqrt(sn² − p²) = 0; Case B would then have no
reactive capability to deploy and would be identical to Case A on every
network.  The snapshot therefore sets p_mw = sn_mva / sqrt(1 + Q_RATIO²),
which leaves exactly the Q_RATIO × P capability the Q(V) characteristic
asks for — see _p_at_max_irradiance().

Validity of the starting point
------------------------------
HC is ADDITIONAL capacity, so the snapshot must be compliant before the
sweep begins.  Where a network's pre-placed DER already breaches a limit
under min load × max irradiance, HC is undefined for that network, and
hc_mw is reported as NaN with binding_element naming the pre-existing
violation (see _infeasible_snapshot()).  It is not reported as 0.00 MW,
which would read as a measurement.  End-of-feeder incremental HC was
specified for networks with no pre-placed DER (Kerber, Dickert, CIGRE LV,
Synthetic Voltage Control LV); on those the snapshot is compliant by
construction.

Conversely, on a network with NO pre-placed DER no element is Q(V)-
controlled — the swept HC sgens are deliberately excluded, see
run_hc_with_volt_var() — so Case B reduces to Case A and the HC gain is
zero by construction rather than by measurement.  A non-zero HC gain
therefore requires a network that has pre-placed DER AND a compliant
snapshot; CIGRE MV (with_der="pv_wind") satisfies both.

DER placement
-------------
New PV sgens are added at the end-of-feeder bus — the bus with the
greatest topological distance from the slack bus on the distribution
voltage level.  This is the conservative lower-bound placement:
- Reproducible (deterministic, no Monte Carlo).
- Physically correct: maximum feeder impedance → maximum voltage rise
  per MW injected.
- Consistent across all networks, including those with pre-placed DERs
  (SimBench MV, CIGRE MV) where existing sgens are scaled to max output
  and new capacity is added on top at the weakest electrical point.

Network voltage level is inferred as the statistical mode of net.bus.vn_kv.
The HV slack bus (minority vn_kv) is excluded from the end-of-feeder search.

HC parameters
-------------
Controlled by HC_PARAMS dict keyed by "MV" or "LV":

    MV: start=0.0 MW, step=0.5 MW, max=20.0 MW  (40 steps)
    LV: start=0.0 MW, step=0.01 MW, max=0.5 MW  (50 steps)

Network type is inferred from the modal vn_kv:
    vn_kv > 1.0 kV  →  "MV"
    vn_kv ≤ 1.0 kV  →  "LV"

Q(V) iteration (Case B)
-----------------------
Fixed-point iteration converges Q setpoints for each snapshot:

    for iter in range(MAX_QV_ITERS):
        runpp()
        q_new = QVCharacteristic.compute_setpoints(vm_pu, p_installed)
        q_new = ctrl._clamp_to_net_limits(q_new)
        write q_new to net.sgen.q_mvar
        if max|q_new - q_prev| < Q_CONV_TOL: break
        q_prev = q_new
    runpp()  # final PF with converged Q

MAX_QV_ITERS = 10, Q_CONV_TOL = 1e-4 MVAr.  Q(V) is a contractive
mapping under normal grid conditions; convergence is typically achieved
in 3–5 iterations.

VoltVarController is instantiated with interface=None, dry_run=True so
_clamp_to_net_limits() is available without Arduino hardware.  This
ensures HC inherits the identical physical Q constraints as Scenario 4.

Net mutation contract
---------------------
Both public functions operate on a deep copy of the caller's net object.
The original net is never modified.  Callers do not need to deepcopy.

Scope
-----
- ALL constraint classes terminate the sweep: voltage (0.95-1.05 pu per
  VDE-AR-N 4110/4105), line and transformer thermal loading, and line
  angle difference — whatever detect_violations() reports. Voltage is the
  usual binding constraint under high PV, but a thermally-bound result is
  reported as such rather than silently ignored; HCResult.binding_bus and
  binding_vm_pu are NaN/-1 in that case and binding_element names it.
- Monte Carlo placement: SP7 (post-semester).
- QSTS sweep: not in scope for semester benchmark.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from statistics import mode
from typing import Optional

import numpy as np
import pandapower as pp
import pandapower.topology as pptop
import pandas as pd

from violation_detector import detect_violations
from volt_var_controller import QVCharacteristic, VoltVarController

# Q_RATIO through the module, not from-imported: set_qv_parameters() rebinds it
# per run, and a from-import would freeze the import-time value, so the
# snapshot's active-power derating would stop matching the Q(V) capability the
# controller actually applies.
import volt_var_controller as _vvc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HC sweep parameters — keyed by network voltage class
# ---------------------------------------------------------------------------
HC_PARAMS: dict[str, dict] = {
    "MV": {"start": 0.0, "step": 0.5,  "max": 20.0},   # MW, 40 steps
    "LV": {"start": 0.0, "step": 0.01, "max": 0.5},     # MW, 50 steps
}

# Q(V) fixed-point iteration parameters (Case B only)
MAX_QV_ITERS: int   = 10
Q_CONV_TOL:   float = 1e-4   # MVAr — convergence tolerance per DER

# Standard type for newly added end-of-feeder PV sgens.
# "PV" is a recognised pandapower sgen type that sets cos_phi=1.0 (P only);
# Q is controlled externally by the Volt-Var loop in Case B.
_HC_SGEN_TYPE: str = "PV"
_HC_SGEN_NAME: str = "hc_pv_endoffeeder"

# runpp kwargs applied to every power flow call in this module
_RUNPP_BASE: dict = {"voltage_depend_loads": False, "algorithm": "nr"}


# ---------------------------------------------------------------------------
# HCResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class HCResult:
    """
    Result of one hosting capacity sweep (baseline or volt_var).

    Fields
    ------
    network_id      : identifier string passed by the caller.
    case            : "baseline" or "volt_var".
    hc_mw           : total PV injection at the last violation-free step (MW).
                      This is the hosting capacity.  If no violation is found
                      within the sweep range, equals the max sweep value and
                      hc_limit_reached is True.
    violated_at_mw  : total PV injection at the first violating step (MW).
                      NaN if no violation was found.
    binding_bus     : bus index where the first violation appeared.
                      -1 if no violation was found.
    binding_vm_pu   : vm_pu at binding_bus at the violating step.
                      NaN if no violation was found, or if the binding
                      constraint was thermal or angular rather than voltage.
    binding_element : human-readable description of what actually stopped
                      the sweep ("overvoltage at bus 42", "line 7 at 104.2 %
                      loading", "none"). The sweep terminates on ANY
                      violation class, so this is the only field that
                      distinguishes a voltage-bound from a thermally-bound
                      hosting capacity.
    n_steps         : number of sweep steps executed.
    hc_limit_reached: True if the sweep reached max without any violation.
    qv_converged    : True if Q(V) iteration converged in all steps (Case B).
                      None for Case A (baseline).
    qv_iters_max    : maximum Q(V) iterations used across all steps (Case B).
                      None for Case A (baseline).
    endoffeeder_bus : bus index selected as end-of-feeder placement target.
    dist_voltage_kv : inferred distribution voltage level (modal vn_kv).
    params          : HC_PARAMS snapshot used for this run.
    """
    network_id:        str
    case:              str
    hc_mw:             float
    violated_at_mw:    float
    binding_bus:       int
    binding_vm_pu:     float
    binding_element:   str
    n_steps:           int
    hc_limit_reached:  bool
    endoffeeder_bus:   int
    dist_voltage_kv:   float
    params:            dict
    qv_converged:      Optional[bool]  = field(default=None)
    qv_iters_max:      Optional[int]   = field(default=None)
    sweep_curve:       list           = field(default_factory=list)

    def summary_dict(self) -> dict:
        """Flat dict for DataFrame construction / CSV export."""
        return {
            "network_id":       self.network_id,
            "case":             self.case,
            "hc_mw":            self.hc_mw,
            "violated_at_mw":   self.violated_at_mw,
            "binding_bus":      self.binding_bus,
            "binding_vm_pu":    self.binding_vm_pu,
            "binding_element":  self.binding_element,
            "n_steps":          self.n_steps,
            "hc_limit_reached": self.hc_limit_reached,
            "endoffeeder_bus":  self.endoffeeder_bus,
            "dist_voltage_kv":  self.dist_voltage_kv,
            "qv_converged":     self.qv_converged,
            "qv_iters_max":     self.qv_iters_max,
            "hc_step_mw":       self.params["step"],
            "hc_max_mw":        self.params["max"],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_dist_voltage(net: pp.pandapowerNet) -> float:
    """
    Return the distribution voltage level (kV) as the statistical mode of
    net.bus.vn_kv.  In any mixed HV/MV or MV/LV network the slack/HV bus
    is a minority; the dominant voltage level is the distribution level.
    """
    return float(mode(net.bus["vn_kv"].tolist()))


def _hc_params_for(dist_voltage_kv: float) -> dict:
    """Return HC_PARAMS entry appropriate for the given voltage level."""
    return HC_PARAMS["LV"] if dist_voltage_kv <= 1.0 else HC_PARAMS["MV"]


def _find_endoffeeder_bus(net: pp.pandapowerNet, dist_voltage_kv: float) -> int:
    """
    Find the end-of-feeder bus on the distribution voltage level.

    Strategy
    --------
    1. Identify the slack bus from net.ext_grid.bus.
    2. Compute topological distances from the slack bus to all buses using
       calc_distance_to_bus(weight='weight') — distances in km.
    3. Restrict candidates to buses where vn_kv == dist_voltage_kv.
    4. Return the candidate bus with the maximum distance (argmax).

    Parameters
    ----------
    net             : pandapower network (not mutated).
    dist_voltage_kv : modal distribution voltage level (kV).

    Returns
    -------
    int — bus index of the end-of-feeder bus.

    Raises
    ------
    ValueError if no distribution-level bus is reachable from the slack.
    """
    slack_bus = int(net.ext_grid["bus"].iloc[0])

    # calc_distance_to_bus returns a Series indexed by bus index,
    # values = shortest path distance in km (float).
    # Unreachable buses are absent from the result.
    dist = pptop.calc_distance_to_bus(
        net,
        slack_bus,
        respect_switches=True,   # open switches = open circuit
        weight="weight",         # km distances
    )

    # Restrict to distribution-level buses only
    dist_buses = net.bus.index[net.bus["vn_kv"] == dist_voltage_kv]
    dist_in_reach = dist.loc[dist.index.intersection(dist_buses)]

    if dist_in_reach.empty:
        raise ValueError(
            f"_find_endoffeeder_bus: no distribution-level buses (vn_kv="
            f"{dist_voltage_kv} kV) reachable from slack bus {slack_bus}."
        )

    endoffeeder = int(dist_in_reach.idxmax())
    logger.debug(
        "_find_endoffeeder_bus: slack=%d, dist_voltage=%.3f kV, "
        "end-of-feeder=%d (%.3f km from slack)",
        slack_bus, dist_voltage_kv, endoffeeder,
        float(dist_in_reach.max()),
    )
    return endoffeeder


def _p_at_max_irradiance(sn_mva: float) -> float:
    """
    Rated ACTIVE power of an inverter whose rated APPARENT power is sn_mva,
    given the Q(V) capability the controller applies.

        P = sn_mva / sqrt(1 + Q_RATIO²)      ⇒  sqrt(sn² − P²) = Q_RATIO × P

    "PV output = 1.0 p.u." in the worst-case snapshot means 1.0 p.u. of rated
    active power, which is what an irradiance of 1000 W/m² delivers. It does
    NOT mean P = sn_mva: sn_mva is the apparent rating, so P = sn_mva forces
    cos φ = 1 and leaves |Q| ≤ sqrt(sn² − P²) = 0. Under that reading the
    Case B sweep has exactly zero reactive capability to deploy and is
    identical to Case A by construction — no Q(V) action, no HC gain,
    whatever the network.

    The factor is derived from volt_var_controller.Q_RATIO rather than being
    a constant of its own, so the derating and the Q(V) characteristic cannot
    drift apart: at Q_RATIO = 0.25 (VDE-AR-N 4110 Bild 8) it is 0.9701, and
    the resulting headroom is exactly Q_RATIO × P — the full capability the
    characteristic asks for, available at rated active power as
    VDE-AR-N 4110 §10.2.2 requires over 0.05 < P_mom/P_b,inst < 1.
    """
    q_ratio = float(_vvc.Q_RATIO)
    return float(sn_mva) / np.sqrt(1.0 + q_ratio ** 2)


def _set_worst_case_snapshot(net: pp.pandapowerNet) -> None:
    """
    Set the network to the worst-case overvoltage snapshot in place:
        - All loads at minimum (scaling = 0.1 × rated P/Q).
        - All existing in-service sgens at maximum output.

    This represents peak irradiance + minimum demand — the condition that
    maximises voltage rise for a given PV injection.

    The 0.1 scaling factor (not zero) avoids degenerate load-flow cases
    where islands or isolated buses cause non-convergence.

    Rated power inference
    ---------------------
    max_p_mw is an OPF constraint column written only by Scenario 5 and is
    NOT a standard pandapower sgen column.  Rated capacity is inferred using
    the same priority chain as _compute_sn_rated() in scenario_5_opf.py and
    _resolve_p_installed() in volt_var_controller.py:

        1. sn_mva — inverter rated apparent power.  Maximum output is the
                    corresponding rated ACTIVE power, sn_mva / sqrt(1 +
                    Q_RATIO²), leaving the Q(V) headroom the controller
                    needs — see _p_at_max_irradiance().
        2. p_mw   — current active power output (fallback when sn_mva absent
                    or non-positive).  Used as-is: with no apparent rating
                    the controller's apparent-power clamp does not bind, so
                    no derating is required.
        3. leave p_mw unchanged if both are degenerate (zero or NaN)
    """
    # Minimum load: 10% of rated
    net.load["p_mw"]   = net.load["p_mw"]   * 0.1
    net.load["q_mvar"] = net.load["q_mvar"] * 0.1

    # Maximum existing PV/wind output — infer rated P from sn_mva → p_mw
    has_sn_mva = "sn_mva" in net.sgen.columns
    for idx in net.sgen.index[net.sgen["in_service"]]:
        rated_p = float("nan")

        if has_sn_mva:
            sn = net.sgen.at[idx, "sn_mva"]
            if pd.notna(sn) and sn > 0.0:
                rated_p = _p_at_max_irradiance(sn)

        if not (pd.notna(rated_p) and rated_p > 0.0):
            # Fallback: current p_mw as proxy for rated output
            p_now = net.sgen.at[idx, "p_mw"]
            if pd.notna(p_now) and p_now > 0.0:
                rated_p = p_now

        if pd.notna(rated_p) and rated_p > 0.0:
            net.sgen.at[idx, "p_mw"] = rated_p
        # else: leave p_mw unchanged (degenerate sgen — do not force to zero)

        # q_mvar reset to 0 — Q is controlled by Volt-Var in Case B
        net.sgen.at[idx, "q_mvar"] = 0.0


def _add_pv_at_bus(
        net:        pp.pandapowerNet,
        bus:        int,
        step_mw:    float,
        step_index: int,
) -> int:
    """
    Add one PV sgen of size step_mw at the given bus.

    Returns the new sgen index.  A unique name encodes the step so that
    added sgens are distinguishable in logs and net.sgen inspection.
    """
    idx = pp.create_sgen(
        net,
        bus=bus,
        p_mw=step_mw,
        q_mvar=0.0,
        name=f"{_HC_SGEN_NAME}_step{step_index:03d}",
        type=_HC_SGEN_TYPE,
        in_service=True,
    )
    return idx


def _build_runpp_kwargs(extra: Optional[dict]) -> dict:
    """Merge caller overrides with mandatory base kwargs."""
    kwargs = dict(_RUNPP_BASE)
    if extra:
        kwargs.update(extra)
        kwargs["voltage_depend_loads"] = False   # never allow override
    return kwargs


def _infeasible_snapshot(
        net:        pp.pandapowerNet,
        kwargs:     dict,
        network_id: str,
        case:       str,
        eof_bus:    int,
        dist_kv:    float,
        params:     dict,
) -> Optional[HCResult]:
    """
    Guard the sweep's premise: the worst-case snapshot must itself be
    compliant before any capacity is added.

    Hosting capacity is defined as ADDITIONAL PV the feeder can absorb, so it
    is only meaningful from a violation-free starting point. On a network
    whose pre-placed DER already breaches a limit under min-load × max-
    irradiance, the first sweep step violates and hc_mw is left at
    params["start"] — reported as a flat 0.00 MW, which reads like a measured
    result rather than an undefined one. The HC gain then shows +0.000 MW for
    both cases and looks like Volt-Var achieving nothing.

    Returns an HCResult carrying hc_mw = NaN and a binding_element naming the
    pre-existing violation when the snapshot is already non-compliant, or
    None when the sweep may proceed.

    Note this is a property of the network under this snapshot definition,
    not an error: end-of-feeder incremental HC was specified for networks
    with no pre-placed DER, and the result for a network whose installed DER
    already exceeds its compliant limit is correctly "undefined", not zero.
    """
    try:
        pp.runpp(net, **kwargs)
    except Exception as exc:                                  # noqa: BLE001
        reason = f"snapshot power flow did not converge ({exc})"
    else:
        report = detect_violations(net)
        if not report.any_violations:
            return None
        _, _, what = _extract_binding(report)
        reason = f"snapshot already violating at 0 MW added: {what}"

    logger.warning(
        "[HC-%s | %s] %s — hosting capacity is undefined for this network "
        "under the worst-case snapshot (min load x max irradiance). "
        "Reporting NaN rather than %.2f MW.",
        case, network_id, reason, params["start"],
    )
    return HCResult(
        network_id       = network_id,
        case             = case,
        hc_mw            = float("nan"),
        violated_at_mw   = float(params["start"]),
        binding_bus      = -1,
        binding_vm_pu    = float("nan"),
        binding_element  = reason,
        n_steps          = 0,
        hc_limit_reached = False,
        endoffeeder_bus  = eof_bus,
        dist_voltage_kv  = dist_kv,
        params           = dict(params),
        qv_converged     = None,
        qv_iters_max     = None,
        sweep_curve      = [],
    )


def _extract_binding(report) -> tuple[int, float, str]:
    """
    Return (binding_bus, binding_vm_pu, binding_element) from a
    ViolationReport.

    The sweep stops on ANY violation class, so a result can be bound by a
    thermal or angle constraint rather than a voltage one. Previously those
    cases returned (-1, nan) with no indication of why, which read as a
    lookup failure. binding_element now names the class and the element so a
    thermally-bound hosting capacity is not mistaken for a voltage one.
    """
    if not report.over_voltage.empty:
        binding_bus = int(report.most_severe_over_voltage_buses[0])
        binding_vm  = float(report.over_voltage.at[binding_bus, "vm_pu"])
        return binding_bus, binding_vm, f"overvoltage at bus {binding_bus}"
    if not report.under_voltage.empty:
        binding_bus = int(
            report.under_voltage.sort_values("deviation_pu", ascending=False).index[0]
        )
        binding_vm = float(report.under_voltage.at[binding_bus, "vm_pu"])
        return binding_bus, binding_vm, f"undervoltage at bus {binding_bus}"
    if not report.overloaded_lines.empty:
        line = report.overloaded_lines["loading_percent"].idxmax()
        pct  = float(report.overloaded_lines.at[line, "loading_percent"])
        return -1, float("nan"), f"line {int(line)} at {pct:.1f} % loading"
    if not report.overloaded_trafos.empty:
        tr  = report.overloaded_trafos["loading_percent"].idxmax()
        pct = float(report.overloaded_trafos.at[tr, "loading_percent"])
        return -1, float("nan"), f"trafo {int(tr)} at {pct:.1f} % loading"
    if not report.angle_violations.empty:
        ln  = report.angle_violations["va_diff_degree"].idxmax()
        deg = float(report.angle_violations.at[ln, "va_diff_degree"])
        return -1, float("nan"), f"line {int(ln)} angle diff {deg:.1f} deg"
    return -1, float("nan"), "none"


def _qv_converge(
        net:         pp.pandapowerNet,
        ctrl:        VoltVarController,
        runpp_kwargs: dict,
) -> tuple[bool, int]:
    """
    Run Q(V) fixed-point iteration to convergence for the current snapshot.

    Iterates:
        runpp() → compute_setpoints(vm_pu, p_installed) → clamp → apply
    until max|Δq| < Q_CONV_TOL or MAX_QV_ITERS is reached.

    A final runpp() is NOT called here — the caller runs it after this
    function returns to obtain the post-Q power flow result and violation
    report.  This avoids a redundant PF call at the last iteration.

    Parameters
    ----------
    net          : network with snapshot already applied.
    ctrl         : VoltVarController(dry_run=True) built for this net.
    runpp_kwargs : merged kwargs for pp.runpp().

    Returns
    -------
    (converged, n_iters_used)
        converged    : True if |Δq| < Q_CONV_TOL before MAX_QV_ITERS.
        n_iters_used : number of iterations executed (1 … MAX_QV_ITERS).
    """
    sgen_idx   = ctrl.sgen_indices
    p_installed = ctrl.p_installed_mw   # shape (n_ders,), MW

    q_prev  = np.zeros(len(sgen_idx), dtype=float)
    converged = False

    for n_iter in range(1, MAX_QV_ITERS + 1):
        try:
            pp.runpp(net, **runpp_kwargs)
        except Exception as exc:
            logger.warning("_qv_converge: runpp() failed at iter %d: %s", n_iter, exc)
            return False, n_iter

        vm_pu = net.res_bus.loc[
            net.sgen.loc[sgen_idx, "bus"].values, "vm_pu"
        ].values

        vm_s = pd.Series(vm_pu,      index=sgen_idx)
        p_s  = pd.Series(p_installed, index=sgen_idx)

        q_new = QVCharacteristic.compute_setpoints(vm_s, p_s).values
        q_new = ctrl._clamp_to_net_limits(q_new)

        net.sgen.loc[sgen_idx, "q_mvar"] = q_new

        delta = float(np.max(np.abs(q_new - q_prev)))
        if delta < Q_CONV_TOL:
            converged = True
            break

        q_prev = q_new

    return converged, n_iter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_baseline_hc(
        net:          pp.pandapowerNet,
        network_id:   str,
        runpp_kwargs: Optional[dict] = None,
) -> tuple[HCResult, pp.pandapowerNet]:
    """
    Hosting capacity sweep — no voltage control (Case A).

    Sets a deep copy of net to the worst-case snapshot (min load, max
    existing PV output), then adds PV incrementally at the end-of-feeder
    bus until the first voltage violation (outside 0.95–1.05 pu).

    HC is the total PV injection at the last violation-free step.

    Parameters
    ----------
    net          : pandapower network.  Not modified — deep copy is taken.
    network_id   : identifier string for logging and HCResult.
    runpp_kwargs : optional extra kwargs forwarded to pp.runpp().
                   voltage_depend_loads=False is always enforced.

    Returns
    -------
    tuple[HCResult, pp.pandapowerNet]
        HCResult    : hosting capacity metrics.
        net         : deep copy of the network at the last violation-free
                      step (hc_mw).  The violating sgen has been removed so
                      this net is ready for a downstream HC-stressed benchmark
                      without requiring another deepcopy or correction.
    """
    net = copy.deepcopy(net)
    kwargs = _build_runpp_kwargs(runpp_kwargs)

    dist_kv      = _infer_dist_voltage(net)
    params       = _hc_params_for(dist_kv)
    eof_bus      = _find_endoffeeder_bus(net, dist_kv)

    _set_worst_case_snapshot(net)

    logger.info(
        "[HC-baseline | %s] dist_voltage=%.3f kV, end-of-feeder bus=%d, "
        "sweep: %.2f–%.2f MW in %.3f MW steps",
        network_id, dist_kv, eof_bus,
        params["start"], params["max"], params["step"],
    )

    infeasible = _infeasible_snapshot(
        net, kwargs, network_id, "baseline", eof_bus, dist_kv, params,
    )
    if infeasible is not None:
        return infeasible, net

    hc_mw            = params["start"]
    violated_at_mw   = float("nan")
    binding_bus      = -1
    binding_vm_pu    = float("nan")
    binding_element  = "none"
    hc_limit_reached = False
    n_steps          = 0
    sweep_curve: list = []

    step_mw  = params["step"]
    total_mw = params["start"]

    while total_mw <= params["max"]:
        n_steps += 1

        # Add one PV sgen of step_mw at end-of-feeder bus
        last_added_idx = _add_pv_at_bus(net, eof_bus, step_mw, n_steps)
        total_mw += step_mw
        total_mw  = round(total_mw, 6)   # avoid float accumulation drift

        try:
            pp.runpp(net, **kwargs)
        except Exception as exc:
            logger.warning(
                "[HC-baseline | %s] runpp() failed at %.3f MW: %s — "
                "treating as violation.",
                network_id, total_mw, exc,
            )
            violated_at_mw = total_mw
            binding_bus    = -1
            binding_vm_pu  = float("nan")
            binding_element = "power flow did not converge"
            net.sgen.drop(index=last_added_idx, inplace=True)
            break

        report = detect_violations(net)

        logger.debug(
            "[HC-baseline | %s] %.3f MW total: violations=%s "
            "max_vm=%.4f pu",
            network_id, total_mw, report.any_violations,
            float(net.res_bus["vm_pu"].max()),
        )

        if report.any_violations:
            violated_at_mw            = total_mw
            binding_bus, binding_vm_pu, binding_element = _extract_binding(report)
            net.sgen.drop(index=last_added_idx, inplace=True)
            sweep_curve.append({
                "mw":        total_mw,
                "max_vm_pu": float(net.res_bus["vm_pu"].max()),
            })
            break

        # No violation — advance HC
        hc_mw = total_mw
        sweep_curve.append({
            "mw":        total_mw,
            "max_vm_pu": float(net.res_bus["vm_pu"].max()),
        })

    else:
        # Loop completed without violation
        hc_limit_reached = True
        logger.info(
            "[HC-baseline | %s] No violation found up to %.2f MW — "
            "HC is >= %.2f MW (limit reached).",
            network_id, params["max"], hc_mw,
        )

    logger.info(
        "[HC-baseline | %s] HC=%.3f MW | violated_at=%.3f MW | "
        "bound by: %s | binding_vm=%.4f pu | steps=%d",
        network_id, hc_mw, violated_at_mw, binding_element, binding_vm_pu, n_steps,
    )

    return (
        HCResult(
            network_id       = network_id,
            case             = "baseline",
            hc_mw            = hc_mw,
            violated_at_mw   = violated_at_mw,
            binding_bus      = binding_bus,
            binding_vm_pu    = binding_vm_pu,
            binding_element  = binding_element,
            n_steps          = n_steps,
            hc_limit_reached = hc_limit_reached,
            endoffeeder_bus  = eof_bus,
            dist_voltage_kv  = dist_kv,
            params           = dict(params),
            qv_converged     = None,
            qv_iters_max     = None,
            sweep_curve      = sweep_curve,
        ),
        net,
    )


def run_hc_with_volt_var(
        net:          pp.pandapowerNet,
        network_id:   str,
        runpp_kwargs: Optional[dict] = None,
) -> HCResult:
    """
    Hosting capacity sweep — VDE-AR-N 4110 Q(V) Volt-Var active (Case B).

    Same sweep as run_baseline_hc() but wraps each snapshot power flow in
    a Q(V) fixed-point iteration loop.  Q setpoints are computed by
    QVCharacteristic.compute_setpoints() and clamped via
    VoltVarController._clamp_to_net_limits() — the identical physical
    constraints as Scenario 4.

    HC gain = hc_mw(Case B) − hc_mw(Case A).

    Parameters
    ----------
    net          : pandapower network.  Not modified — deep copy is taken.
    network_id   : identifier string for logging and HCResult.
    runpp_kwargs : optional extra kwargs forwarded to pp.runpp().
                   voltage_depend_loads=False is always enforced.

    Returns
    -------
    HCResult
    """
    net = copy.deepcopy(net)
    kwargs = _build_runpp_kwargs(runpp_kwargs)

    dist_kv = _infer_dist_voltage(net)
    params  = _hc_params_for(dist_kv)
    eof_bus = _find_endoffeeder_bus(net, dist_kv)

    _set_worst_case_snapshot(net)

    # Build VoltVarController once — dry_run=True, no Arduino.
    # configure() discovers sgen_indices from net.sgen at this moment.
    # New sgens added during the sweep will NOT be in ctrl.sgen_indices
    # because they are added after configure() is called — intentional:
    # HC sgens represent incremental capacity being tested, not controlled
    # DERs.  Only pre-existing DERs participate in Q(V) control, which
    # is consistent with Scenario 4 (controller owns ap.der_p.columns).
    ctrl = VoltVarController(net, interface=None, dry_run=True)
    ctrl.configure()

    logger.info(
        "[HC-voltvar | %s] dist_voltage=%.3f kV, end-of-feeder bus=%d, "
        "%d controlled DERs, sweep: %.2f–%.2f MW in %.3f MW steps",
        network_id, dist_kv, eof_bus, ctrl.n_ders,
        params["start"], params["max"], params["step"],
    )

    if ctrl.n_ders == 0:
        # Not an error, but the gain it feeds is structurally zero: the swept
        # HC sgens are deliberately excluded from control (see below), so with
        # no pre-placed DER there is nothing for Q(V) to act on and Case B
        # reduces to Case A. Say so rather than letting a +0.000 MW gain read
        # as "Volt-Var achieved nothing".
        logger.warning(
            "[HC-voltvar | %s] network has no pre-placed DER, so no element "
            "is Q(V)-controlled and this sweep is identical to the baseline. "
            "The HC gain for this network is zero by construction, not by "
            "measurement — do not quote it as a Volt-Var result.",
            network_id,
        )

    infeasible = _infeasible_snapshot(
        net, kwargs, network_id, "volt_var", eof_bus, dist_kv, params,
    )
    if infeasible is not None:
        return infeasible

    hc_mw            = params["start"]
    violated_at_mw   = float("nan")
    binding_bus      = -1
    binding_vm_pu    = float("nan")
    binding_element  = "none"
    hc_limit_reached = False
    n_steps          = 0
    sweep_curve: list = []
    all_qv_converged = True
    qv_iters_max     = 0

    step_mw  = params["step"]
    total_mw = params["start"]

    while total_mw <= params["max"]:
        n_steps += 1

        # Add one PV sgen of step_mw at end-of-feeder bus.
        # q_mvar=0 — this sgen is not in ctrl.sgen_indices and receives
        # no Q setpoint from the Volt-Var loop.
        _add_pv_at_bus(net, eof_bus, step_mw, n_steps)
        total_mw += step_mw
        total_mw  = round(total_mw, 6)

        if ctrl.n_ders > 0:
            # Reset q_mvar of controlled DERs to 0 before iteration
            net.sgen.loc[ctrl.sgen_indices, "q_mvar"] = 0.0

            converged, n_iters = _qv_converge(net, ctrl, kwargs)
            all_qv_converged = all_qv_converged and converged
            qv_iters_max     = max(qv_iters_max, n_iters)

            if not converged:
                logger.warning(
                    "[HC-voltvar | %s] Q(V) did not converge at %.3f MW "
                    "after %d iterations — proceeding with last iterate.",
                    network_id, total_mw, n_iters,
                )

        # Final power flow with converged Q applied
        try:
            pp.runpp(net, **kwargs)
        except Exception as exc:
            logger.warning(
                "[HC-voltvar | %s] final runpp() failed at %.3f MW: %s — "
                "treating as violation.",
                network_id, total_mw, exc,
            )
            violated_at_mw = total_mw
            binding_bus    = -1
            binding_vm_pu  = float("nan")
            binding_element = "power flow did not converge"
            break

        report = detect_violations(net)

        logger.debug(
            "[HC-voltvar | %s] %.3f MW total: violations=%s "
            "max_vm=%.4f pu | Q(V) iters=%d converged=%s",
            network_id, total_mw, report.any_violations,
            float(net.res_bus["vm_pu"].max()),
            n_iters if ctrl.n_ders > 0 else 0,
            converged if ctrl.n_ders > 0 else "n/a",
        )

        if report.any_violations:
            violated_at_mw             = total_mw
            binding_bus, binding_vm_pu, binding_element = _extract_binding(report)
            sweep_curve.append({
                "mw":        total_mw,
                "max_vm_pu": float(net.res_bus["vm_pu"].max()),
            })
            break

        hc_mw = total_mw
        sweep_curve.append({
            "mw":        total_mw,
            "max_vm_pu": float(net.res_bus["vm_pu"].max()),
        })

    else:
        hc_limit_reached = True
        logger.info(
            "[HC-voltvar | %s] No violation found up to %.2f MW — "
            "HC is >= %.2f MW (limit reached).",
            network_id, params["max"], hc_mw,
        )

    logger.info(
        "[HC-voltvar | %s] HC=%.3f MW | violated_at=%.3f MW | "
        "bound by: %s | binding_vm=%.4f pu | steps=%d | "
        "qv_converged=%s | qv_iters_max=%d",
        network_id, hc_mw, violated_at_mw, binding_element, binding_vm_pu,
        n_steps, all_qv_converged, qv_iters_max,
    )

    return HCResult(
        network_id       = network_id,
        case             = "volt_var",
        hc_mw            = hc_mw,
        violated_at_mw   = violated_at_mw,
        binding_bus      = binding_bus,
        binding_vm_pu    = binding_vm_pu,
        binding_element  = binding_element,
        n_steps          = n_steps,
        hc_limit_reached = hc_limit_reached,
        endoffeeder_bus  = eof_bus,
        dist_voltage_kv  = dist_kv,
        params           = dict(params),
        qv_converged     = all_qv_converged,
        qv_iters_max     = qv_iters_max,
        sweep_curve      = sweep_curve,
    )
