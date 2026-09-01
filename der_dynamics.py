"""
der_dynamics.py
===============
Phase 1 Item 4 — DER physical response layer.

Applies VDE-AR-N 4110 / VDE-AR-N 4105 dynamic constraints between the
computed setpoint target and the value written to net.sgen:

    Reactive power  : Exact discrete PT1 exponential filter.
    Active power    : Symmetric linear ramp rate limiting.

This layer is network-agnostic.  The caller (CLI wizard or scenario runner)
selects the appropriate standard preset and passes numeric parameters.
der_dynamics.py applies those parameters; it does not know whether it is
operating on an MV or LV network.

Standards basis
---------------
VDE-AR-N 4110 (MV networks)
    Q response : PT1, 95% of setpoint in 6–60 s, default 10 s
                 (3τ = 10 s  →  τ ≈ 3.33 s)
    P ramp     : 0.33% – 0.66% PN/s; central modelling value 0.5% PN/s
    Qmax       : 0.48 · Pb,inst  (cos φ ≈ 0.90, Bild 8 Q(U) curve)

VDE-AR-N 4105 (LV networks)
    Q response : PT1, 95% of setpoint in 6–60 s, default 10 s (same τ)
    P ramp     : 0.33% – 0.66% PAmax/s; central value 0.5% PAmax/s
    Qmax       : 0.33 · PAmax  (cos φ = 0.95)

Discrete PT1 — exact exponential form
--------------------------------------
The update equation is:

    q_applied[k] = q_prev + α · (q_target − q_prev)

where the coefficient uses the exact exponential form:

    α = 1 − exp(−Δt / τ)    τ = T_95 / 3

This is resolution-safe without clipping:
    large Δt (e.g., 900 s for 15-min profiles):  α → 1.0
        (unit step; dynamics invisible — correct, VDE ramp is non-binding)
    small Δt (e.g., 1 s HIL loop):  α → Δt/τ
        (Euler approximation holds, Q response smoothly visible)

The linear approximation α ≈ Δt/τ is NOT used.  It requires clipping to
avoid α > 1 and is only valid when Δt << τ.

Sensitivity study presets (pass to DERDynamics constructor)
-----------------------------------------------------------
Case            t95_q_s     ramp_rate_p_frac_s
MV base         10 s        MV_RAMP_RATE_P_BASE  (0.005  = 0.50% PN/s)
MV slow         60 s        MV_RAMP_RATE_P_SLOW  (0.0033 = 0.33% PN/s)
MV fast          6 s        MV_RAMP_RATE_P_FAST  (0.0066 = 0.66% PN/s)
LV base         10 s        LV_RAMP_RATE_P_BASE  (0.005)
LV slow         60 s        LV_RAMP_RATE_P_SLOW  (0.0033)
LV fast          6 s        LV_RAMP_RATE_P_FAST  (0.0066)

Lifecycle (mandatory)
---------------------
1. Construct DERDynamics with dt_s, t95_q_s, ramp_rate_p_frac_s,
   q_max_mvar, p_rated_mw.
2. Read the first profile row  →  p_target_t0  (np.ndarray, shape (n_ders,)).
3. Call reset(q_init=0.0, p_init=p_target_t0).
4. Enter the timestep loop.
5. Each step: call step(q_target, p_target) → (q_applied, p_applied).

reset() MUST be called before the first step().  step() raises RuntimeError
if called on an uninitialised instance (_initialized = False).

q_init = 0.0 is correct: Q control has not yet acted at study-window start.
p_init = p_target_t0 is correct: the DER was already generating at that
    profile level before the study window started.

Integration with run_coordinated_timestep()
--------------------------------------------
DERDynamics.step() is called inside run_coordinated_timestep() (Option Y),
after the SensitivityCoordinator produces q_adjusted and before the post-PF.
Sequence within run_coordinated_timestep():

    [pre-PF]    net.sgen.p_mw  = p_target   (raw profile)
                net.sgen.q_mvar = 0
                runpp()  →  report_pre

    [Item 2]    q_initial = Arduino / dry-run Q(V)
    [Item 3]    q_adjusted = coordinator.coordinate(q_initial)

    [Item 4]    q_applied, p_applied = dynamics.step(q_adjusted, p_target)
                net.sgen.p_mw   = p_applied   (overwrites raw profile)
                q_clamped       = controller._clamp_to_net_limits(q_applied)
                    # _clamp_to_net_limits reads p_mw from net.sgen →
                    # apparent-power cap uses p_applied, not p_target
                net.sgen.q_mvar = q_clamped
                runpp()  →  report_post

report_pre  uses p_target / q=0    (uncontrolled snapshot)
report_post uses p_applied / q_applied  (physically achieved state)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standard preset constants — pass to DERDynamics constructor
# ---------------------------------------------------------------------------

# VDE-AR-N 4110 (MV)
MV_T95_Q_S:          float = 10.0    # s — default 3τ time, no grid-operator value
MV_T95_Q_SLOW_S:     float = 60.0   # s — slow sensitivity bound
MV_T95_Q_FAST_S:     float = 6.0    # s — fast sensitivity bound
MV_RAMP_RATE_P_BASE: float = 0.005  # 0.50% PN/s — central modelling value
MV_RAMP_RATE_P_SLOW: float = 0.0033 # 0.33% PN/s — slow bound
MV_RAMP_RATE_P_FAST: float = 0.0066 # 0.66% PN/s — fast bound

# VDE-AR-N 4105 (LV) — same time constants; Qmax differs (set via q_max_mvar)
LV_T95_Q_S:          float = 10.0
LV_T95_Q_SLOW_S:     float = 60.0
LV_T95_Q_FAST_S:     float = 6.0
LV_RAMP_RATE_P_BASE: float = 0.005
LV_RAMP_RATE_P_SLOW: float = 0.0033
LV_RAMP_RATE_P_FAST: float = 0.0066


# ===========================================================================
# DERDynamics
# ===========================================================================

class DERDynamics:
    """
    Physical DER response layer: exact discrete PT1 Q filter + P ramp limiter.

    Network-agnostic.  All physical limits are constructor arguments; the
    caller selects the appropriate standard preset (MV or LV).

    Parameters
    ----------
    dt_s : float
        Control loop timestep [s].
        Use 1.0 for HIL inner loop (1 s dynamics visible).
        Use 900.0 for 15-min annual study (dynamics invisible — correct).
        Must be > 0.

    t95_q_s : float
        Time for Q output to reach 95% of a step setpoint [s].
        VDE-AR-N 4110/4105: adjustable 6–60 s, default 10 s if the grid
        operator specifies no value.
        Internally: τ = t95_q_s / 3;  α = 1 − exp(−dt_s / τ).
        Must be > 0.

    ramp_rate_p_frac_s : float
        Maximum P change per second as a fraction of p_rated_mw [dimensionless/s].
        VDE-AR-N 4110/4105: 0.33%–0.66%/s; use MV_RAMP_RATE_P_BASE (0.005)
        as the central modelling value.
        Per-timestep limit: δP_max = ramp_rate_p_frac_s · p_rated_mw · dt_s [MW].
        Must be > 0.

    q_max_mvar : array-like, shape (n_ders,)
        Per-DER Q capacity [MVAr].  All values must be > 0.
        MV: Q_RATIO · p_installed_mw  (Q_RATIO sourced from
        volt_var_controller at call time; do not assume a literal here —
        it is an overridable run parameter)        
        LV: 0.33 · p_installed_mw

    p_rated_mw : array-like, shape (n_ders,)
        Per-DER rated (installed) active power [MW].  All values must be > 0.
        Used to: (a) scale δP_max per DER, (b) enforce the physical P ceiling.

    Attributes (read-only after construction)
    ------------------------------------------
    alpha : float
        Exact discrete PT1 coefficient: 1 − exp(−dt_s / τ).
    n_ders : int
        Number of controlled DERs.
    initialized : bool
        True after reset() has been called.

    State (set by reset(), advanced by step())
    -------------------------------------------
    q_prev : np.ndarray, shape (n_ders,)  [MVAr]
    p_prev : np.ndarray, shape (n_ders,)  [MW]

    Raises
    ------
    ValueError   : invalid constructor arguments (non-positive, shape mismatch,
                   empty arrays).
    RuntimeError : step() called before reset().
    ValueError   : step() input shape mismatch.
    """

    def __init__(
        self,
        dt_s:               float,
        t95_q_s:            float,
        ramp_rate_p_frac_s: float,
        q_max_mvar:         np.ndarray,
        p_rated_mw:         np.ndarray,
    ) -> None:
        # --- scalar validation ---
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be > 0, got {dt_s!r}")
        if t95_q_s <= 0.0:
            raise ValueError(f"t95_q_s must be > 0, got {t95_q_s!r}")
        if ramp_rate_p_frac_s <= 0.0:
            raise ValueError(
                f"ramp_rate_p_frac_s must be > 0, got {ramp_rate_p_frac_s!r}"
            )

        # --- array conversion and validation ---
        q_max   = np.asarray(q_max_mvar, dtype=float)
        p_rated = np.asarray(p_rated_mw, dtype=float)

        if q_max.ndim != 1:
            raise ValueError(
                f"q_max_mvar must be 1-D, got shape {q_max.shape}"
            )
        if p_rated.ndim != 1:
            raise ValueError(
                f"p_rated_mw must be 1-D, got shape {p_rated.shape}"
            )
        if q_max.shape != p_rated.shape:
            raise ValueError(
                f"Shape mismatch: q_max_mvar {q_max.shape} "
                f"!= p_rated_mw {p_rated.shape}"
            )
        if q_max.size == 0:
            raise ValueError("q_max_mvar and p_rated_mw must not be empty")
        if np.any(q_max <= 0.0):
            bad = np.where(q_max <= 0.0)[0].tolist()
            raise ValueError(
                f"All q_max_mvar values must be > 0. "
                f"Non-positive at DER indices: {bad}"
            )
        if np.any(p_rated <= 0.0):
            bad = np.where(p_rated <= 0.0)[0].tolist()
            raise ValueError(
                f"All p_rated_mw values must be > 0. "
                f"Non-positive at DER indices: {bad}"
            )

        # --- store parameters ---
        self._dt_s:               float        = float(dt_s)
        self._t95_q_s:            float        = float(t95_q_s)
        self._ramp_rate_p_frac_s: float        = float(ramp_rate_p_frac_s)
        self._q_max:              np.ndarray   = q_max
        self._p_rated:            np.ndarray   = p_rated
        self._n_ders:             int          = q_max.size

        # --- derived constants ---
        tau:          float        = t95_q_s / 3.0
        self._alpha:  float        = float(1.0 - np.exp(-dt_s / tau))
        self._dp_max: np.ndarray   = ramp_rate_p_frac_s * p_rated * dt_s
        # _dp_max[i] = max MW change per timestep for DER i

        # --- state (uninitialised until reset()) ---
        self._initialized: bool       = False
        self.q_prev:       np.ndarray = np.zeros(self._n_ders, dtype=float)
        self.p_prev:       np.ndarray = np.zeros(self._n_ders, dtype=float)

        logger.debug(
            "DERDynamics created: n_ders=%d, dt=%.1fs, t95=%.1fs, "
            "τ=%.3fs, α=%.6f, ramp=%.4f%%/s, "
            "dp_max=[%.4f..%.4f] MW/step",
            self._n_ders, self._dt_s, self._t95_q_s,
            tau, self._alpha, self._ramp_rate_p_frac_s * 100.0,
            float(self._dp_max.min()), float(self._dp_max.max()),
        )

    # -----------------------------------------------------------------------
    # Read-only properties
    # -----------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """Exact discrete PT1 coefficient: α = 1 − exp(−dt_s / τ)."""
        return self._alpha

    @property
    def n_ders(self) -> int:
        """Number of controlled DERs."""
        return self._n_ders

    @property
    def initialized(self) -> bool:
        """True after reset() has been called at least once."""
        return self._initialized

    # -----------------------------------------------------------------------
    # reset()
    # -----------------------------------------------------------------------

    def reset(
        self,
        q_init: float | np.ndarray,
        p_init: float | np.ndarray,
    ) -> None:
        """
        Initialise DER state before the timestep loop.

        Must be called once after construction and before the first step().
        Re-calling mid-loop is allowed (e.g., after a scenario restart) but
        discards all accumulated state.

        Parameters
        ----------
        q_init : float or np.ndarray
            Initial Q state [MVAr].  Scalar is broadcast to all DERs.
            Correct value: 0.0 — Q control has not yet acted at the
            study-window start.  Clipped to ±q_max before storage.

        p_init : float or np.ndarray
            Initial P state [MW].  Scalar is broadcast to all DERs.
            Correct value: first profile row for the relevant DER indices —
            the DER was already generating at that level before the study
            window began.  Clipped to [0, p_rated] before storage.

        Raises
        ------
        ValueError : q_init or p_init cannot be broadcast to (n_ders,).
        """
        try:
            q = np.broadcast_to(
                np.asarray(q_init, dtype=float), (self._n_ders,)
            ).copy()
        except ValueError:
            raise ValueError(
                f"q_init shape {np.asarray(q_init).shape} cannot be "
                f"broadcast to (n_ders={self._n_ders},)"
            )
        try:
            p = np.broadcast_to(
                np.asarray(p_init, dtype=float), (self._n_ders,)
            ).copy()
        except ValueError:
            raise ValueError(
                f"p_init shape {np.asarray(p_init).shape} cannot be "
                f"broadcast to (n_ders={self._n_ders},)"
            )

        self.q_prev        = np.clip(q, -self._q_max,  self._q_max)
        self.p_prev        = np.clip(p,  0.0,          self._p_rated)
        self._initialized  = True

        logger.debug(
            "DERDynamics.reset(): "
            "q_prev=[%.4f..%.4f] MVAr, p_prev=[%.4f..%.4f] MW",
            float(self.q_prev.min()), float(self.q_prev.max()),
            float(self.p_prev.min()), float(self.p_prev.max()),
        )

    # -----------------------------------------------------------------------
    # step()
    # -----------------------------------------------------------------------

    def step(
        self,
        q_target: np.ndarray,
        p_target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Advance one control timestep: apply PT1 to Q, ramp limit to P.

        Updates q_prev and p_prev at the end of the step.

        Parameters
        ----------
        q_target : np.ndarray, shape (n_ders,)
            Coordinated Q target from SensitivityCoordinator.coordinate()
            [MVAr].  Called q_adjusted in CoordinatorResult — this is the
            desired setpoint, not yet physically achieved.

        p_target : np.ndarray, shape (n_ders,)
            Profile P value for this timestep (or curtailed P from Item 5)
            [MW].  Ramp limiting constrains the achievable step from p_prev.

        Returns
        -------
        q_applied : np.ndarray, shape (n_ders,)
            Q after PT1 filter and ±q_max clip [MVAr].
            Written to net.sgen.q_mvar (after _clamp_to_net_limits).

        p_applied : np.ndarray, shape (n_ders,)
            P after ramp limiting and [0, p_rated] clip [MW].
            Written to net.sgen.p_mw before the post-PF.

        Raises
        ------
        RuntimeError : reset() has not been called.
        ValueError   : q_target or p_target shape != (n_ders,).

        Notes
        -----
        Q — exact discrete PT1:

            q_applied = clip(q_prev + α·(q_target − q_prev), −q_max, +q_max)

        P — symmetric ramp then physical clip:

            p_ramped  = clip(p_target, p_prev − δP_max, p_prev + δP_max)
            p_applied = clip(p_ramped, 0, p_rated)

        Resolution behaviour at 15-min profile step (dt_s = 900 s):
            Q: α = 1 − exp(−900/3.33) → 1.0  (unit step, non-binding)
            P: δP_max = 0.005 · p_rated · 900 = 4.5 · p_rated  (non-binding)
        Both are physically correct for long timestep annual studies.
        """
        if not self._initialized:
            raise RuntimeError(
                "DERDynamics.step() called before reset(). "
                "Call reset(q_init=0.0, p_init=p_target_t0) with the first "
                "profile row before entering the timestep loop."
            )

        q_t = np.asarray(q_target, dtype=float)
        p_t = np.asarray(p_target, dtype=float)

        if q_t.shape != (self._n_ders,):
            raise ValueError(
                f"q_target shape {q_t.shape} != (n_ders={self._n_ders},). "
                "Must match sgen_indices ordering from VoltVarController."
            )
        if p_t.shape != (self._n_ders,):
            raise ValueError(
                f"p_target shape {p_t.shape} != (n_ders={self._n_ders},). "
                "Must match sgen_indices ordering from VoltVarController."
            )

        # Q — exact discrete PT1
        q_applied: np.ndarray = np.clip(
            self.q_prev + self._alpha * (q_t - self.q_prev),
            -self._q_max,
            self._q_max,
        )

        # P — symmetric ramp limiting then physical floor / ceiling
        p_applied: np.ndarray = np.clip(
            np.clip(p_t, self.p_prev - self._dp_max, self.p_prev + self._dp_max),
            0.0,
            self._p_rated,
        )

        # Advance state
        self.q_prev = q_applied.copy()
        self.p_prev = p_applied.copy()

        logger.debug(
            "DERDynamics.step(): "
            "max|q_target−q_applied|=%.6f MVAr, "
            "max|p_target−p_applied|=%.6f MW",
            float(np.abs(q_t - q_applied).max()),
            float(np.abs(p_t - p_applied).max()),
        )

        return q_applied, p_applied

    # -----------------------------------------------------------------------
    # repr
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DERDynamics("
            f"dt={self._dt_s}s, "
            f"t95={self._t95_q_s}s, "
            f"α={self._alpha:.6f}, "
            f"ramp={self._ramp_rate_p_frac_s * 100:.2f}%/s, "
            f"n_ders={self._n_ders}, "
            f"initialized={self._initialized}"
            f")"
        )
