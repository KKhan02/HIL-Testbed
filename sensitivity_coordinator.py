"""
sensitivity_coordinator.py
===========================
Phase 1 Item 3 — Sensitivity-based Q coordination.

Computes the dV/dQ sensitivity matrix analytically from pandapower's
Newton-Raphson Jacobian and uses it to coordinate reactive power setpoints
across all DERs, correcting residual voltage violations left after Item 2's
per-DER Q(V) control.

Architecture
------------
Sits between Item 2 (Arduino Q(V)) and net.sgen.q_mvar application.
VoltVarController.run_timestep() cannot be used directly — the HIL loop is
managed manually so that Item 3 can intercept between Q computation and
application.  Use run_coordinated_timestep() for the full single-step sequence.

    [Pre-PF]  net.sgen.q_mvar = 0; runpp()
    [Item 2]  Arduino / dry-run  → q_initial[n_ders]   (local VDE-AR-N 4110)
    [Item 3]  coordinate()       → q_adjusted[n_ders]  (global residual correction)
    [Apply]   net.sgen.q_mvar = q_adjusted
    [Post-PF] runpp()

Mathematical pipeline
---------------------
1.  Extract four NR Jacobian subblocks from net._ppc["internal"]["J"] (sparse).
2.  Sparse LU factorisation of J_PP (genuinely sparse, no fill-in).
3.  Dense Schur complement: J_red = J_QQ - J_QP J_PP^{-1} J_PQ.
    J_red is fully dense due to fill-in; scipy.linalg.lu_factor is used.
4.  Targeted column solve: X = J_red^{-1}[:, der_cols]   shape (n_pq, n_ders).
    One factorisation reused for all DER right-hand sides.
5.  Double-count correction: predict vm after q_initial, then solve only the
    residual gap with np.linalg.lstsq.
6.  Convert pu correction back to MVAr; clip to ±q_max.

Units note
----------
The pandapower NR Jacobian relates ΔP_pu/ΔQ_pu to Δθ/Δ(|V|/|V|), so
X is dimensionless (pu voltage / pu power).  The |V| ≈ 1.0 approximation is
applied (valid in the 0.95–1.05 pu operating band; max error < 5 %).

Usage
-----
Scenario 4 HIL loop::

    import pandapower as pp
    import simbench as sb
    from volt_var_controller import VoltVarController, ArduinoSerialInterface
    from sensitivity_coordinator import SensitivityCoordinator, run_coordinated_timestep

    net = sb.get_simbench_net("1-MV-rural--2-sw")

    with ArduinoSerialInterface(port="/dev/ttyACM0") as arduino:
        ctrl        = VoltVarController(net, arduino)
        ctrl.configure()
        coordinator = SensitivityCoordinator(net, ctrl)

        for t in time_steps:
            net.sgen.loc[ctrl.sgen_indices, "p_mw"] = p_profiles.iloc[t]
            net.load.loc[:, "p_mw"]                 = load_p.iloc[t]
            net.load.loc[:, "q_mvar"]               = load_q.iloc[t]

            result = run_coordinated_timestep(net, ctrl, coordinator)

Dry-run (no Arduino)::

    ctrl        = VoltVarController(net, interface=None, dry_run=True)
    ctrl.configure()
    coordinator = SensitivityCoordinator(net, ctrl)
    result      = run_coordinated_timestep(net, ctrl, coordinator)

Dependencies
------------
    pandapower >= 3.4.0
    scipy  (sparse + linalg)
    numpy
    volt_var_controller.py  (same project, Item 2)
    violation_detector.py   (same project)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandapower as pp
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import splu

from violation_detector import ViolationReport, detect_violations
import violation_detector as _vd
from volt_var_controller import (
    QVCharacteristic,
    VoltVarController,
    ArduinoSerialInterface,
    ArduinoProtocolError,
    SerialTimeoutError,
)
# Q_RATIO is read at USE time via the module attribute so runtime overrides
# (volt_var_controller.set_qv_parameters(), used by the CLI executor) size
# the coordinator's q_max with the SAME value the Arduino receives over CFG.
import volt_var_controller as _vvc
from der_dynamics import DERDynamics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PYPOWER bus type constants  (ppc["bus"][:, BUS_TYPE_COL])
# ---------------------------------------------------------------------------
_BUS_TYPE_COL: int = 1    # column index of BUS_TYPE in ppc["bus"]
_BUS_PQ:       int = 1    # load bus
_BUS_PV:       int = 2    # generator bus
_BUS_REF:      int = 3    # slack / reference bus
_BUS_VM_COL:   int = 7    # VM voltage magnitude in p.u. in PYPOWER bus matrix
# ---------------------------------------------------------------------------
# Control constants
# ---------------------------------------------------------------------------
V_TARGET_PU:      float = 1.00    # desired voltage at violated buses
MIN_VIOLATION_DV: float = 1e-3    # ignore violations < 1 milli-pu (numerical noise)
RCOND_THRESHOLD:  float = 1e-12   # reciprocal condition number; below → ill-conditioned
SATURATION_TOL:   float = 1e-6    # tolerance for float saturation check


# ===========================================================================
# CoordinatorResult — structured output from run_coordinated_timestep()
# ===========================================================================

@dataclass
class CoordinatorResult:
    """
    Structured output from run_coordinated_timestep().

    Attributes
    ----------
    report_pre          : ViolationReport before any Q action (q_mvar=0 state).
                          Pre-PF uses p_target (raw profile), q=0.
    report_post         : ViolationReport after dynamics and post-PF.
                          Post-PF uses p_applied and q_applied — physical reality.
                          None if post-PF failed.
    q_initial           : Q setpoints from Item 2 (Arduino / dry-run) [MVAr].
    q_adjusted          : Coordinated Q target from Item 3. NOT the applied value.
                          q_adjusted is the setpoint before PT1 dynamics. [MVAr]
    q_applied           : Q after PT1 filter and ±q_max clip. Written to
                          net.sgen.q_mvar (after _clamp_to_net_limits). [MVAr]
    p_target            : Raw profile P (or curtailed P from Item 5). [MW]
    p_applied           : P after ramp limiting. Written to net.sgen.p_mw
                          before post-PF. [MW]
    curtailment_needed  : True if violations persist after post-PF.
                          False if post-PF failed (uncertain state).
    post_pf_ok          : True if post-PF runpp() converged.
    n_retries           : Serial retry count. 0 in dry-run mode.
    t_total_ms          : Wall-clock time for the full timestep [ms].
    """
    report_pre:         ViolationReport
    report_post:        Optional[ViolationReport]
    q_initial:          np.ndarray   # Item 2 output (Arduino / dry-run) [MVAr]
    q_adjusted:         np.ndarray   # Item 3 output — coordinated Q target, NOT applied [MVAr]
    q_applied:          np.ndarray   # after PT1 dynamics — written to net.sgen [MVAr]
    p_target:           np.ndarray   # raw profile value (or curtailed) [MW]
    p_applied:          np.ndarray   # after ramp limiting — written to net.sgen [MW]
    curtailment_needed: bool
    post_pf_ok:         bool
    n_retries:          int   = 0
    t_total_ms:         float = 0.0
    mode:               str = "coordinated"   # "local" for 4A, "coordinated" for 4B
    t_exchange_ms:      float = 0.0

    @property
    def dq_correction(self) -> np.ndarray:
        """Coordination correction from Item 3: q_adjusted − q_initial [MVAr]."""
        return self.q_adjusted - self.q_initial

    @property
    def dq_dynamics(self) -> np.ndarray:
        """PT1 smoothing gap: q_applied − q_adjusted [MVAr].
        Near zero at 15-min resolution (α → 1); visible at 1-s HIL loop."""
        return self.q_applied - self.q_adjusted

    @property
    def dp_ramp(self) -> np.ndarray:
        """Ramp-limiting gap: p_applied − p_target [MW].
        Near zero at 15-min resolution; binding only during steep ramps."""
        return self.p_applied - self.p_target

    @property
    def violations_resolved(self) -> bool:
        """True if violations present pre-control and absent post-control."""
        if not self.post_pf_ok or self.report_post is None:
            return False
        return self.report_pre.any_violations and not self.report_post.any_violations

    def summary(self) -> str:
        """One-line summary for logging."""
        pre  = self.report_pre.summary()
        post = self.report_post.summary() if self.report_post else "no post-PF"
        dq_local = np.abs(self.q_initial).sum()          # total Q from Q(V) curve
        dq_coord = np.abs(self.dq_correction).max()      # coordination delta
        q_applied_total = float(np.abs(self.q_applied).sum())   # total Q actually injected
        dq_dyn   = np.abs(self.dq_dynamics).max()
        dp_ramp  = np.abs(self.dp_ramp).max()
        if self.mode == "local":
            return (
                f"LocalQV | pre: {pre} | post: {post} | "
                f"Q_local_sum: {dq_local:.4f} MVAr | "
                f"Q_applied_sum: {q_applied_total:.4f} MVAr | "
                f"curtail: {self.curtailment_needed} | {self.t_total_ms:.1f}ms"
            )
        else:
            return (
                f"CoordQV | pre: {pre} | post: {post} | "
                f"Q_local_sum: {dq_local:.4f} MVAr | "
                f"max|dQ_coord|: {dq_coord:.4f} MVAr | "
                f"max|dQ_dyn|: {dq_dyn:.4f} MVAr | "
                f"max|dP_ramp|: {dp_ramp:.4f} MW | "
                f"Q_applied_sum: {q_applied_total:.4f} MVAr | "
                f"curtail: {self.curtailment_needed} | "
                f"retries: {self.n_retries} | {self.t_total_ms:.1f}ms"
            )


# ===========================================================================
# SensitivityCoordinator — Item 3 main class
# ===========================================================================

class SensitivityCoordinator:
    """
    Sensitivity-based Q coordination layer for Scenario 4 HIL.

    Computes the dV/dQ Jacobian sensitivity matrix from the Newton-Raphson
    Jacobian stored in net._ppc after runpp(), and uses it to correct residual
    voltage violations left after Item 2's per-DER Q(V) control.

    Parameters
    ----------
    net                  : pandapower network (same mutable object as HIL loop).
    volt_var_controller  : VoltVarController instance (Item 2).
                           Provides sgen_indices, p_installed_mw, n_ders.

    Attributes
    ----------
    curtailment_needed : bool
        Provisional flag after coordinate().  Overridden by run_coordinated_timestep()
        with the authoritative post-PF violation check.
    """

    def __init__(
            self,
            net,
            volt_var_controller: VoltVarController,
    ) -> None:
        self._net         = net
        self._ctrl        = volt_var_controller
        self._sgen_idx    = volt_var_controller.sgen_indices       # pd.Index, ascending
        self._n_ders      = volt_var_controller.n_ders
        self._p_installed = volt_var_controller.p_installed_mw     # ndarray (n_ders,)
        self._rank_warning_count = 0
        self._rank_warning_examples = []

        self.curtailment_needed: bool = False

        # q_max mirrors Item 2: Q_RATIO × p_installed, same source (_resolve_p_installed)
        self._q_max: np.ndarray = _vvc.Q_RATIO * self._p_installed  # (n_ders,) [MVAr] (call-time read)

        zero_mask = self._q_max <= 0.0
        if zero_mask.any():
            logger.warning(
                "q_max=0 for sgen indices %s. "
                "These DERs will be excluded from coordination.",
                self._sgen_idx[zero_mask].tolist(),
            )

        # DER bus indices — pandapower bus indices, same order as sgen_idx
        self._sgen_buses: np.ndarray = (
            self._net.sgen.loc[self._sgen_idx, "bus"].values.copy()
        )

    # ------------------------------------------------------------------
    # Private: Jacobian block extraction
    # ------------------------------------------------------------------

    def _build_jacobian_blocks(self) -> dict:
        """
        Extract the four NR Jacobian subblocks from net._ppc.

        The PYPOWER NR Jacobian J has shape (n_pv_pq + n_pq) × (n_pv_pq + n_pq):

            [ J_PP  J_PQ ]    rows 0:n_pv_pq  → ΔP equations (all non-slack buses)
            [ J_QP  J_QQ ]    rows n_pv_pq:   → ΔQ equations (PQ buses only)

            cols 0:n_pv_pq  → Δθ
            cols n_pv_pq:   → Δ(|V|/|V|)    (PQ buses only)

        All slicing operates on the sparse J matrix.  .toarray() is never called
        on the full J.

        Returns
        -------
        dict with keys:
            J_PP, J_PQ, J_QP, J_QQ  : scipy.sparse subblocks
            pq_mask                  : bool ndarray (n_ppc_buses,), ppc ordering
            n_pv_pq, n_pq            : int
            n_ppc_buses              : int
            pq_bus_indices_ppc       : ndarray (n_pq,) — sorted ppc indices of PQ buses

        Raises
        ------
        RuntimeError
            If net._ppc is absent (runpp() not called yet).
        """
        if not hasattr(self._net, "_ppc") or self._net._ppc is None:
            raise RuntimeError(
                "net._ppc not found. Call runpp() before SensitivityCoordinator."
            )

        internal = self._net._ppc.get("internal",{})
        if "J" not in internal:
            raise RuntimeError(
                "net._ppc['internal']['J'] not found. "
                "SensitivityCoordinator requires a Newton-Raphson runpp result. "
                "Use algorithm='nr' or 'iwamoto_nr'; do not use 'bfsw'."
            )
        J = internal["J"]             

        bus_types   = self._net._ppc["bus"][:, _BUS_TYPE_COL]
        n_ppc_buses = len(bus_types)

        pq_mask    = (bus_types == _BUS_PQ)
        pv_pq_mask = (bus_types == _BUS_PQ) | (bus_types == _BUS_PV)

        n_pv_pq = int(pv_pq_mask.sum())
        n_pq    = int(pq_mask.sum())

        pq_bus_indices_ppc: np.ndarray = np.where(pq_mask)[0]   # sorted ascending

        # Slice sparse subblocks — no .toarray() on full J
        J_PP = J[:n_pv_pq,  :n_pv_pq]
        J_PQ = J[:n_pv_pq,  n_pv_pq:]
        J_QP = J[n_pv_pq:,  :n_pv_pq]
        J_QQ = J[n_pv_pq:,  n_pv_pq:]

        return {
            "J_PP":               J_PP,
            "J_PQ":               J_PQ,
            "J_QP":               J_QP,
            "J_QQ":               J_QQ,
            "pq_mask":            pq_mask,
            "n_pv_pq":            n_pv_pq,
            "n_pq":               n_pq,
            "n_ppc_buses":        n_ppc_buses,
            "pq_bus_indices_ppc": pq_bus_indices_ppc,
        }

    def _bus_lookup_maps(self) -> tuple[np.ndarray, dict]:
        """
        Return:
            pd2ppc : ndarray mapping pandapower bus label -> ppc bus index
            ppc2pd : dict mapping ppc bus index -> actual pandapower bus index

        Important:
        pandapower bus indices are DataFrame labels. In SimBench, these labels
        can be sparse and much larger than len(net.bus). Therefore, do not
        enumerate pd2ppc by compact position. Iterate over net.bus.index and
        index pd2ppc by the actual bus label.
        """
        if not hasattr(self._net, "_pd2ppc_lookups"):
            raise RuntimeError(
                "net._pd2ppc_lookups not found. Call runpp() before coordinate()."
            )

        pd2ppc: np.ndarray = self._net._pd2ppc_lookups["bus"]
        ppc2pd: dict = {}

        for pd_bus_idx in self._net.bus.index:
            pd_bus_int = int(pd_bus_idx)

            if pd_bus_int < 0 or pd_bus_int >= len(pd2ppc):
                continue

            ppc_bus = int(pd2ppc[pd_bus_int])

            if ppc_bus < 0:
                continue

            ppc2pd[ppc_bus] = pd_bus_idx

        return pd2ppc, ppc2pd

    # ------------------------------------------------------------------
    # Private: vm_pu in ppc PQ-bus ordering
    # ------------------------------------------------------------------
    '''DEPRECIATED FUNCTION FOR TESTING PPC BUS NOT FOUND ERROR
    def _get_vm_pu_ppc(
            self,
            pq_mask:            np.ndarray,
            pq_bus_indices_ppc: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        Return vm_pu at PQ buses in ppc bus ordering.

        All S-matrix row indexing throughout coordinate() uses this ordering,
        ensuring viol_mask and S rows are always aligned.

        Parameters
        ----------
        pq_mask             : bool ndarray (n_ppc_buses,) from _build_jacobian_blocks()
        pq_bus_indices_ppc  : ndarray (n_pq,) from _build_jacobian_blocks()

        Returns
        -------
        vm_pu_ppc : ndarray (n_pq,)  voltages in ppc PQ-bus order
        ppc2pd    : dict  ppc bus index → pandapower bus index

        Raises
        ------
        RuntimeError
            If NaN detected in net.res_bus.vm_pu (runpp() divergence).
            If a PQ bus in ppc has no matching pandapower bus (internal inconsistency).
        """
        vm_pu_pd: pd.Series = self._net.res_bus["vm_pu"]

        if vm_pu_pd.isna().any():
            bad_buses = vm_pu_pd.index[vm_pu_pd.isna()].tolist()
            raise RuntimeError(
                f"NaN in net.res_bus.vm_pu at buses {bad_buses}. "
                "Likely runpp() divergence — check pre-PF convergence before "
                "calling SensitivityCoordinator.coordinate()."
            )

        _pd2ppc, ppc2pd = self._bus_lookup_maps()

        # Extract vm_pu in ppc PQ-bus order with KeyError guard
        vm_pu_ppc_list: list = []
        for ppc_idx in pq_bus_indices_ppc:
            pd_idx = ppc2pd.get(int(ppc_idx), None)
            if pd_idx is None:
                raise RuntimeError(
                    f"ppc bus {ppc_idx} (PQ type) not found in ppc2pd mapping. "
                    "Inconsistency between ppc bus array and pd2ppc lookup. "
                    "Re-run runpp() on a consistent network state."
                )
            vm_pu_ppc_list.append(float(vm_pu_pd.at[pd_idx]))

        return np.array(vm_pu_ppc_list, dtype=float), ppc2pd
'''

    def _get_vm_pu_ppc(
            self,
            pq_mask:            np.ndarray,
            pq_bus_indices_ppc: np.ndarray,
    ) -> Tuple[np.ndarray, dict, np.ndarray]:
        """
        Return vm_pu at PQ buses in PPC row ordering.

        Important:
        Do not map PPC PQ buses back through net.res_bus here. Some internal
        PPC buses can appear in net._ppc["bus"] and in the Jacobian but do not
        have a clean pandapower res_bus counterpart. The PPC bus matrix already
        contains solved voltage magnitude in column VM, so use it directly.

        Update#1: vm_pu_ppc may contain NaN for internal/auxiliary PPC rows.
        valid_target_mask marks rows that can be used as physical control targets 

        Returns
        -------
        vm_pu_ppc : ndarray (n_pq,)
            Voltage magnitudes in PPC PQ-row order.
        ppc2pd : dict
            Best-effort ppc bus number -> pandapower bus index map, retained
            for diagnostics / downstream compatibility.
        """
        ppc_bus = self._net._ppc["bus"]

        if ppc_bus is None or len(ppc_bus) == 0:
            raise RuntimeError("net._ppc['bus'] is empty. Run runpp() first.")

        vm_all_ppc = ppc_bus[:, _BUS_VM_COL].astype(float)

        #if np.isnan(vm_all_ppc).any():
        #   bad_rows = np.where(np.isnan(vm_all_ppc))[0].tolist()
        #  raise RuntimeError(
        #     f"NaN in net._ppc['bus'] VM column at PPC rows {bad_rows}. "
        #    "Likely runpp() divergence."
            #)

        vm_pu_ppc = vm_all_ppc[pq_bus_indices_ppc]

        # Best-effort reverse lookup. Do not require full coverage.
        _pd2ppc, ppc2pd = self._bus_lookup_maps()

        valid_target_mask = np.array(
            [
                np.isfinite(vm_pu_ppc[i]) and int(ppc_idx) in ppc2pd
                for i,ppc_idx in enumerate(pq_bus_indices_ppc)
            ],
            dtype=bool,
        )

        n_invalid = int((~valid_target_mask).sum())
        if n_invalid:
            logger.info(
                "Ignoring %d internal/non-observable PQ PPC rows as voltage targets.",
                n_invalid,
            )

        return np.array(vm_pu_ppc, dtype=float), ppc2pd, valid_target_mask
    # ------------------------------------------------------------------
    # Private: sensitivity matrix (targeted DER columns only)
    # ------------------------------------------------------------------

    def _compute_der_sensitivity(
            self,
            blocks: dict,
            ppc2pd: dict,
    ) -> Tuple[Optional[np.ndarray], Optional[List]]:
        """
        Compute X: the dV/dQ sensitivity matrix for DER columns only.

        Mathematical pipeline
        --------------------
        1.  Map each DER pandapower bus → column index in J_QQ (O(1) per DER).
        2.  Sparse LU of J_PP (genuinely sparse, appropriate for splu).
        3.  Fill-in block (dense): J_PP_inv_J_PQ = J_PP^{-1} J_PQ.
        4.  Dense Schur complement: J_red = J_QQ - J_QP J_PP_inv_J_PQ.
        5.  Dense LU of J_red (lu_factor/lu_solve — appropriate for dense matrix).
        6.  Reciprocal condition estimate from U diagonal at zero extra cost.
        7.  Targeted column solve: X_valid = J_red^{-1} E_der.
        8.  Assemble full X (n_pq × n_ders); zero columns for excluded DERs.

        Parameters
        ----------
        blocks : dict from _build_jacobian_blocks()
        ppc2pd : dict from _get_vm_pu_ppc()

        Returns
        -------
        (X, der_col_positions) on success.
        (None, None)           on any factorisation failure.
            X                 : ndarray (n_pq, n_ders); col j=0 if DER excluded.
            der_col_positions : list (n_ders,) of int or None.
        """
        J_PP               = blocks["J_PP"]
        J_PQ               = blocks["J_PQ"]
        J_QP               = blocks["J_QP"]
        J_QQ               = blocks["J_QQ"]
        n_pq               = blocks["n_pq"]
        n_ppc_buses        = blocks["n_ppc_buses"]
        pq_bus_indices_ppc = blocks["pq_bus_indices_ppc"]

        pd2ppc, _ppc2pd = self._bus_lookup_maps()

        # O(1) lookup: ppc bus index → position in J_QQ columns
        pq_pos_lookup: dict = {
            int(ppc_idx): pos
            for pos, ppc_idx in enumerate(pq_bus_indices_ppc)
        }

        # Map each DER pandapower bus to its J_QQ column position
        der_col_positions: list = []

        for j, pd_bus in enumerate(self._sgen_buses):
            pd_bus_int = int(pd_bus)
            sgen_label = self._sgen_idx[j]

            if pd_bus_int < 0 or pd_bus_int >= len(pd2ppc):
                logger.warning(
                    "sgen %s: pd_bus %d out of pd2ppc range [0, %d). Excluded.",
                    sgen_label, pd_bus_int, len(pd2ppc),
                )
                der_col_positions.append(None)
                continue

            ppc_bus = int(pd2ppc[pd_bus_int])

            if ppc_bus == -1:
                logger.warning(
                    "sgen %s: pd_bus %d is out-of-service (ppc_bus=-1). Excluded.",
                    sgen_label, pd_bus_int,
                )
                der_col_positions.append(None)
                continue

            if ppc_bus < 0 or ppc_bus >= n_ppc_buses:
                logger.warning(
                    "sgen %s: ppc_bus %d out of ppc bus range [0, %d). Excluded.",
                    sgen_label, ppc_bus, n_ppc_buses,
                )
                der_col_positions.append(None)
                continue

            col = pq_pos_lookup.get(ppc_bus, None)

            if col is None:
                logger.warning(
                    "sgen %s: bus %d (ppc %d) is not a PQ bus in this operating point. "
                    "Excluded from coordination; q_initial passes through.",
                    sgen_label, pd_bus_int, ppc_bus,
                )

            der_col_positions.append(col)

        # ------------------------------------------------------------------
        # Sparse LU of J_PP
        # Wrapped in try/except: splu raises RuntimeError on structural singularity;
        # near-singular J_PP produces garbage J_PP_inv_J_PQ without raising, which
        # is caught downstream by the J_red rcond check.
        # ------------------------------------------------------------------

        
        try:
            lu_PP = splu(J_PP.tocsc())
        except Exception as exc:
            logger.warning(
                "splu(J_PP) failed: %s. J_PP may be singular. "
                "Returning q_initial unchanged this timestep.",
                exc,
            )
            return None, None

        # Fill-in block: lu_PP.solve expects dense RHS → always dense result
        J_PP_inv_J_PQ: np.ndarray = lu_PP.solve(J_PQ.toarray())    # (n_pv_pq, n_pq)

        # Reduced Jacobian: dense due to Schur complement fill-in
        J_red: np.ndarray = (
            J_QQ.toarray() - J_QP.toarray() @ J_PP_inv_J_PQ       # (n_pq, n_pq)
        )

        # ------------------------------------------------------------------
        # Dense LU of J_red (lu_factor/lu_solve — efficient for dense matrix)
        # ------------------------------------------------------------------
        try:
            lu_red, piv_red = lu_factor(J_red)
        except Exception as exc:
            logger.warning(
                "lu_factor(J_red) failed: %s. Returning q_initial unchanged.",
                exc,
            )
            return None, None

        # Reciprocal condition estimate from U diagonal — zero extra cost.
        # lu_factor packs L and U into lu_red; np.diag(lu_red) == diagonal of U.
        U_diag    = np.abs(np.diag(lu_red))
        U_max     = float(U_diag.max()) if len(U_diag) > 0 else 0.0
        U_min     = float(U_diag.min()) if len(U_diag) > 0 else 0.0
        rcond_est = (U_min / U_max) if U_max > 0.0 else 0.0

        if rcond_est < RCOND_THRESHOLD:
            logger.warning(
                "J_red ill-conditioned (rcond_est=%.2e < %.2e). "
                "Network may be near voltage collapse. "
                "Returning q_initial unchanged this timestep.",
                rcond_est, RCOND_THRESHOLD,
            )
            return None, None

        # ------------------------------------------------------------------
        # Build E_der: unit-column RHS for valid DER positions only
        # ------------------------------------------------------------------
        valid_pairs: list = [
            (j, col)
            for j, col in enumerate(der_col_positions)
            if col is not None
        ]
        n_valid = len(valid_pairs)

        if n_valid == 0:
            logger.warning(
                "No DERs mapped to PQ buses. Coordination skipped entirely."
            )
            return None, None

        E_der = np.zeros((n_pq, n_valid), dtype=float)
        for k, (j, col) in enumerate(valid_pairs):
            E_der[col, k] = 1.0

        # Targeted column solve: one factorisation, n_valid right-hand sides
        X_valid: np.ndarray = lu_solve((lu_red, piv_red), E_der)   # (n_pq, n_valid)

        # Assemble full X; columns for excluded DERs remain zero
        X = np.zeros((n_pq, self._n_ders), dtype=float)
        for k, (j, col) in enumerate(valid_pairs):
            X[:, j] = X_valid[:, k]

        return X, der_col_positions

    # ------------------------------------------------------------------
    # Public: main coordination entry point
    # ------------------------------------------------------------------

    def coordinate(self, q_initial: np.ndarray) -> np.ndarray:
        """
        Adjust Item 2 Q setpoints using network sensitivity.

        The pre-PF must have been called with net.sgen.q_mvar reset to 0.0.
        This is enforced by the assertion in Step 2 below.  The HIL loop
        (run_coordinated_timestep) guarantees this invariant.

        Because q_current = 0 at entry, the double-count correction simplifies to:
            dq_pu = q_initial / net.sn_mva

        Parameters
        ----------
        q_initial  : ndarray (n_ders,) [MVAr]
                     Q setpoints from Arduino/dry-run, same order as sgen_indices.

        Returns
        -------
        q_adjusted : ndarray (n_ders,) [MVAr], clipped to ±q_max.
        """

        # Step 1 — Input validation
        if len(q_initial) != self._n_ders:
            raise ValueError(
                f"q_initial length {len(q_initial)} != n_ders {self._n_ders}. "
                f"Must match sgen_indices order: {self._sgen_idx.tolist()}"
            )

        # Step 2 — Enforce q_current = 0 invariant
        # Pre-PF must run with q_mvar = 0 so that X @ dq_pu predicts the full
        # voltage shift from q_initial, not just the delta from a prior state.
        q_current = (
            self._net.sgen.loc[self._sgen_idx, "q_mvar"]
            .values.astype(float)
        )
        max_q_current = float(np.abs(q_current).max()) if len(q_current) > 0 else 0.0
        if max_q_current > 1e-9:
            raise RuntimeError(
                f"net.sgen.q_mvar is non-zero (max |q| = {max_q_current:.4f} MVAr) "
                "before coordinate() was called. "
                "The HIL loop must reset net.sgen.q_mvar to 0.0 and call runpp() "
                "before calling SensitivityCoordinator.coordinate()."
            )

        # Step 3 — Build Jacobian blocks once; pass to sub-functions
        blocks = self._build_jacobian_blocks()

        # Step 4 — vm_pu in ppc PQ-bus ordering (NaN guard inside)
        vm_pu_ppc, ppc2pd, valid_target_mask = self._get_vm_pu_ppc(
            blocks["pq_mask"],
            blocks["pq_bus_indices_ppc"],
        )
        
        # Step 5 — Sensitivity matrix (DER columns only)
        X, der_col_positions = self._compute_der_sensitivity(blocks, ppc2pd)

        if X is None:
            logger.warning(
                "Sensitivity computation failed. Returning q_initial clipped to ±q_max."
            )
            self.curtailment_needed = False
            return np.clip(q_initial, -self._q_max, self._q_max)

        # Step 6 — Predict vm_pu after q_initial is applied (double-count fix)
        # q_current = 0 (enforced above), so dq = q_initial
        dq_pu:           np.ndarray = q_initial / self._net.sn_mva
        dV_from_initial: np.ndarray = X @ dq_pu
        vm_pu_predicted: np.ndarray = vm_pu_ppc + dV_from_initial

        # Step 7 — Identify residual violations in predicted state
        resid_mask: np.ndarray = (
            valid_target_mask
            & np.isfinite(vm_pu_predicted)
            & (
                (vm_pu_predicted > _vd.V_MAX + MIN_VIOLATION_DV)
                | (vm_pu_predicted < _vd.V_MIN - MIN_VIOLATION_DV)
            )
        )

        if not resid_mask.any():
            self.curtailment_needed = False
            logger.debug(
                "coordinate(): resid_mask empty — q_initial resolves violations. "
                "max_vm_predicted=%.4f pu  max|vm_predicted-1.0|=%.4f pu  "
                "max|q_initial|=%.4f MVAr  sum|q_initial|=%.4f MVAr",
                float(np.max(vm_pu_predicted[valid_target_mask]))
                if valid_target_mask.any() else float("nan"),
                float(np.max(np.abs(vm_pu_predicted[valid_target_mask] - 1.0)))
                if valid_target_mask.any() else float("nan"),
                float(np.max(np.abs(q_initial))),
                float(np.sum(np.abs(q_initial))),
            )
            return np.clip(q_initial, -self._q_max, self._q_max)

        logger.debug(
            "coordinate(): %d residual violated bus(es) after q_initial. "
            "max|vm_predicted-1.0|=%.4f pu  max|q_initial|=%.4f MVAr  "
            "max|dV_from_initial|=%.4f pu",
            int(resid_mask.sum()),
            float(np.max(np.abs(vm_pu_predicted[resid_mask] - 1.0))),
            float(np.max(np.abs(q_initial))),
            float(np.max(np.abs(dV_from_initial))),
        )

        # Step 8 — Residual-violation submatrix + rank diagnostic
        S_viol:  np.ndarray = X[resid_mask, :]
        n_resid: int        = int(resid_mask.sum())

        rank = np.linalg.matrix_rank(S_viol)

        rank_limit = min(n_resid, self._n_ders)

        if rank < rank_limit:
            self._rank_warning_count += 1

            if len(self._rank_warning_examples) < 5:
                self._rank_warning_examples.append(
                    {
                        "rank": int(rank),
                        "rank_limit": int(rank_limit),
                        "n_resid": int(n_resid),
                        "n_ders": int(self._n_ders),
                    }
                )

            if self._rank_warning_count == 1 or self._rank_warning_count % 96 == 0:
                logger.warning(
                    "S_viol rank-deficient in %d timestep(s) so far. "
                    "Latest: rank=%d < min(n_resid=%d, n_ders=%d)=%d. "
                    "Some violated buses may be weakly controllable from available DERs.",
                    self._rank_warning_count,
                    rank,
                    n_resid,
                    self._n_ders,
                    rank_limit,
                )
            else:
                logger.debug(
                    "S_viol rank %d < min(%d, %d).",
                    rank,
                    n_resid,
                    self._n_ders,
                )
            # Rank-deficient: lstsq null-space components worsen voltages.
            # Pass q_initial through unchanged rather than corrupt it.
            self.curtailment_needed = True
        # return np.clip(q_initial, -self._q_max, self._q_max)   # ← THIS LINE

        # Step 9 — Residual voltage gap (pu)
        dV_residual: np.ndarray = V_TARGET_PU - vm_pu_predicted[resid_mask]
        dQ_corr_pu: np.ndarray
       
        # Least-squares solve for correction (pu)
        dQ_corr_pu, _, _, _ = np.linalg.lstsq(S_viol, dV_residual, rcond=None)

        # Pre-clip: guard against extreme values from ill-conditioned S_viol
        max_dQ_pu = 2.0 * self._q_max / self._net.sn_mva
        dQ_corr_pu = np.clip(dQ_corr_pu, -max_dQ_pu, max_dQ_pu)

        # Step 10 — Convert to MVAr, combine with q_initial, clip to capacity
        q_adjusted: np.ndarray = np.clip(
            q_initial + dQ_corr_pu * self._net.sn_mva,
            -self._q_max,
            self._q_max,
        )

        # Step 11 — Provisional saturation flag (overridden by post-PF in HIL loop)
        non_skipped = np.array(
            [col is not None for col in der_col_positions], dtype=bool
        )
        if non_skipped.any():
            saturated = (
                np.abs(q_adjusted[non_skipped]) >=
                self._q_max[non_skipped] - SATURATION_TOL
            )
            self.curtailment_needed = bool(saturated.all())
        else:
            self.curtailment_needed = False

        return q_adjusted


# ===========================================================================
# run_coordinated_timestep — full Item 2 + Item 3 single-step sequence
# ===========================================================================

def run_coordinated_timestep(
        net,
        controller:   VoltVarController,
        coordinator:  SensitivityCoordinator,
        dynamics:     DERDynamics,
        p_target:     np.ndarray,
        runpp_kwargs: Optional[dict] = None,
        coordination: bool = True,       # NEW: False = 4A, True = 4B
) -> CoordinatorResult:
    """
    Execute one HIL timestep: pre-PF → Item 2 → Item 3 → Item 4 → post-PF.

    Sequence
    --------
    [0] Write p_target to net.sgen.p_mw (raw profile — used by pre-PF).
    [1] Reset q_mvar to 0.
    [2] Pre-PF  →  report_pre  (uncontrolled snapshot: p_target, q=0).
    [3] Item 2  →  q_initial  (Arduino or dry-run Q(V) from pre-PF vm_pu).
    [4] Item 3  →  q_adjusted  (coordinated Q target; NOT yet applied).
    [5] Item 4  →  dynamics.step(q_adjusted, p_target)
                   →  (q_applied, p_applied).
    [6] Write p_applied to net.sgen.p_mw  (overwrites step [0]).
        _clamp_to_net_limits(q_applied) uses p_applied — correct apparent-
        power cap (reads net.sgen.p_mw directly; must be p_applied here).
        Write q_clamped to net.sgen.q_mvar.
    [7] Post-PF  →  report_post  (physical reality: p_applied, q_applied).
    [8] Authoritative curtailment_needed from post-PF.
    [9] Build and return CoordinatorResult.

    The caller must set net.load.p_mw and net.load.q_mvar before calling.
    Do NOT write net.sgen.p_mw before calling — this function owns that write.

    On pre-PF failure:
        dynamics.step() is NOT called (state is not advanced for a failed
        timestep). q_applied = zeros; p_applied = p_target (no ramp applied).

    Parameters
    ----------
    net           : pandapower network (modified in place).
    controller    : VoltVarController instance (Item 2).
    coordinator   : SensitivityCoordinator instance (Item 3).
    dynamics      : DERDynamics instance (Item 4). reset() must have been
                    called before the first call to this function.
    p_target      : np.ndarray, shape (n_ders,) [MW].
                    Raw profile P for this timestep, aligned with
                    controller.sgen_indices ordering.
    runpp_kwargs  : Extra kwargs forwarded to pp.runpp().
                    voltage_depend_loads=False is always enforced.

    Returns
    -------
    CoordinatorResult
    """
    import time
    t0 = time.perf_counter()
    t_exchange_ms: float = 0.0

    kwargs: dict = {"voltage_depend_loads": False}
    if runpp_kwargs:
        kwargs.update(runpp_kwargs)
        kwargs["voltage_depend_loads"] = False

    sgen_idx:   pd.Index    = controller.sgen_indices
    p_inst:     np.ndarray  = controller.p_installed_mw
    sgen_buses: np.ndarray  = controller._sgen_buses
    empty_q:    np.ndarray  = np.zeros(controller.n_ders, dtype=float)

    # ------------------------------------------------------------------
    # [0] Write raw profile P  (used by pre-PF for uncontrolled snapshot)
    # ------------------------------------------------------------------
    net.sgen.loc[sgen_idx, "p_mw"] = p_target

    # ------------------------------------------------------------------
    # [1] Reset q_mvar to 0 (mirrors VoltVarController.run_timestep line 923)
    #     Required for: (a) clean report_pre, (b) coordinate() q_current=0 invariant
    # ------------------------------------------------------------------
    net.sgen.loc[sgen_idx, "q_mvar"] = 0.0

    # ------------------------------------------------------------------
    # [2] Pre-PF
    # ------------------------------------------------------------------
    pf_pre_ok = True
    try:
        pp.runpp(net, **kwargs)
    except Exception as exc:
        pf_pre_ok = False
        logger.error("Pre-PF runpp() raised: %s. Skipping timestep.", exc)

    report_pre = detect_violations(net)

    if not pf_pre_ok or not report_pre.converged:
        # dynamics.step() intentionally NOT called — state not advanced
        # for a failed timestep. p_prev / q_prev remain unchanged.
        return CoordinatorResult(
            report_pre         = report_pre,
            report_post        = None,
            q_initial          = empty_q,
            q_adjusted         = empty_q,
            q_applied          = empty_q,
            p_target           = p_target.copy(),
            p_applied          = p_target.copy(),  # no ramp applied
            curtailment_needed = False,
            post_pf_ok         = False,
            t_total_ms         = (time.perf_counter() - t0) * 1e3,
            t_exchange_ms = t_exchange_ms,
        )

    # ── GATE ──────────────────────────────────────────────────────────────
    # If no violations, hold Q=0, reuse pre-PF result, skip post-PF.
    # This prevents the Q(V) ramp zones from disturbing clean timesteps.
    if not report_pre.any_violations:
        empty_q = np.zeros(controller.n_ders, dtype=float)
        # Still advance dynamics with q_target=0 so p_prev/q_prev stay valid
        _, p_applied = dynamics.step(q_target=empty_q, p_target=p_target)
        net.sgen.loc[sgen_idx, "p_mw"] = p_applied
        # q stays 0 — already written in [1]
        return CoordinatorResult(
            report_pre         = report_pre,
            report_post        = report_pre,   # reuse — nothing changed
            q_initial          = empty_q,
            q_adjusted         = empty_q,
            q_applied          = empty_q,
            p_target           = p_target.copy(),
            p_applied          = p_applied,
            curtailment_needed = False,
            post_pf_ok         = True,        # pre-PF converged → valid
        )
    # ── END GATE ───────────────────────────────────────────────────────────

    # ------------------------------------------------------------------
    # [3] Item 2 — get q_initial WITHOUT writing to net.sgen
    # ------------------------------------------------------------------
    vm_pu_at_ders = net.res_bus.loc[sgen_buses, "vm_pu"].values.astype(float)
    n_retries = 0

    if controller._dry or controller._iface is None:
        vm_s = pd.Series(vm_pu_at_ders, index=sgen_idx)
        p_s  = pd.Series(p_inst,        index=sgen_idx)
        q_initial: np.ndarray = QVCharacteristic.compute_setpoints(vm_s, p_s).values
    else:
        try:
            t_exch0 = time.perf_counter()
            q_initial, n_attempts = controller._iface.exchange_batched(vm_pu_at_ders, p_inst)
            t_exchange_ms = (time.perf_counter() - t_exch0) * 1e3
            n_retries = n_attempts - 1
            
        except (ArduinoProtocolError, SerialTimeoutError) as exc:
            logger.warning(
                "Serial failure: %s. Falling back to local Q(V) for this timestep.",
                exc,
            )
            vm_s = pd.Series(vm_pu_at_ders, index=sgen_idx)
            p_s  = pd.Series(p_inst,        index=sgen_idx)
            q_initial = QVCharacteristic.compute_setpoints(vm_s, p_s).values
            n_retries = controller._iface.max_retries

    # ------------------------------------------------------------------
    # [4] Item 3 — coordinate (skipped when coordination=False, i.e. mode 4A)
    #     net.sgen.q_mvar is still 0 (set in [1], not touched since).
    #     The coordinate() assertion verifies this invariant.
    # ------------------------------------------------------------------
    if coordination:
        q_adjusted: np.ndarray = coordinator.coordinate(q_initial)
    else:
        q_adjusted = np.clip(q_initial, -coordinator._q_max, coordinator._q_max)
    # q_adjusted = coordinated Q target. NOT applied yet. NOT physically achieved.

    # ------------------------------------------------------------------
    # [5] Item 4 — DER dynamics
    #     PT1 on Q; ramp limiting on P.
    # ------------------------------------------------------------------
    q_applied: np.ndarray
    p_applied: np.ndarray
    q_applied, p_applied = dynamics.step(
        q_target=q_adjusted,
        p_target=p_target,
    )

    # ------------------------------------------------------------------
    # [6] Apply setpoints
    #     p_applied written FIRST — _clamp_to_net_limits reads p_mw from
    #     net.sgen directly, so the apparent-power cap uses p_applied.
    # ------------------------------------------------------------------
    net.sgen.loc[sgen_idx, "p_mw"]  = p_applied
    q_clamped: np.ndarray = controller._clamp_to_net_limits(q_applied)
    net.sgen.loc[sgen_idx, "q_mvar"] = q_clamped

    # ------------------------------------------------------------------
    # [7] Post-PF  (reflects physical reality: p_applied, q_applied)
    # ------------------------------------------------------------------
    post_pf_ok = True
    try:
        pp.runpp(net, **kwargs)
    except Exception as exc:
        post_pf_ok = False
        logger.warning("Post-PF runpp() raised: %s.", exc)

    report_post = detect_violations(net)

    # ------------------------------------------------------------------
    # [8] Authoritative curtailment flag
    #     Only updated from post-PF if it converged.
    # ------------------------------------------------------------------
    if post_pf_ok:
        coordinator.curtailment_needed = report_post.any_violations
    else:
        logger.warning(
            "Post-PF failed. curtailment_needed flag not updated from post-PF; "
            "provisional value %s retained.", coordinator.curtailment_needed,
        )

    result = CoordinatorResult(
        report_pre         = report_pre,
        report_post        = report_post,
        q_initial          = q_initial,
        q_adjusted         = q_adjusted,         # coordinated target — NOT applied
        q_applied          = q_clamped,          # after PT1 + apparent-power clamp
        p_target           = p_target.copy(),
        p_applied          = p_applied,          # after ramp limiting
        curtailment_needed = coordinator.curtailment_needed,
        post_pf_ok         = post_pf_ok,
        n_retries          = n_retries,
        t_total_ms         = (time.perf_counter() - t0) * 1e3,
        mode               = "coordinated" if coordination else "local",
        t_exchange_ms      = t_exchange_ms,   

    )

    logger.info("%s", result.summary())
    return result