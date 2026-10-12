# TASK 4 — DOCUMENT 4
## DER Dynamics — `der_dynamics.py`
### The PT1 reactive-power filter and active-power ramp limiter
### Complete line-by-line walkthrough

> **Source verified:** `der_dynamics.py` (474 lines) was read in full before
> writing. If the file is edited, re-verify line numbers.
>
> **Read alongside:** Master Index §5–§8; **Document 3B** step [5] (the call site —
> `dynamics.step(q_adjusted, p_target)` inside `run_coordinated_timestep`);
> **Document 1A/1C** (the `Q_RATIO`/`q_max` and `p_installed` this layer's ceilings
> derive from).
>
> **Flowchart:** `flow_der_dynamics.*` (Task 1a).

---

## TABLE OF CONTENTS

1. [Overview](#1-overview)
2. [File logic flow — pseudocode](#2-file-logic-flow--pseudocode)
3. [Call-site map — where every entity is used](#3-call-site-map--where-every-entity-is-used)
4. [The physics: PT1 filter and ramp limiting](#4-the-physics-pt1-filter-and-ramp-limiting)
5. [Section A — Preset constants (L112–126)](#5-section-a--preset-constants-l112126)
6. [Section B — Class & __init__ (L133–271)](#6-section-b--class--__init__-l133271)
7. [Section C — Properties (L277–290)](#7-section-c--properties-l277290)
8. [Section D — reset (L296–353)](#8-section-d--reset-l296353)
9. [Section E — step (L359–458)](#9-section-e--step-l359458)
10. [Section F — __repr__ (L464–474)](#10-section-f-__repr__-l464474)
11. [Coverage checklist](#11-coverage-checklist)

---

## 1. Overview

`DERDynamics` is the **physical response layer** — Item 4 in the Scenario 4
pipeline. It sits between the *target* setpoints (what the control law wants) and
the *applied* setpoints (what a real DER can physically achieve in one timestep),
enforcing two VDE-AR-N 4110/4105 dynamic constraints:

- **Reactive power** — an *exact discrete PT1* (first-order lag) filter: a DER
  cannot jump its reactive output instantaneously; it settles toward the target
  with a time constant.
- **Active power** — *symmetric linear ramp limiting*: a DER's active output
  cannot change faster than a fixed rate (a fraction of rated power per second).

It is **network-agnostic**: all physical limits are constructor arguments, and
the caller passes the right MV or LV preset. The class does not know or care
whether it is on an MV or LV feeder.

**Where it sits in the loop (Document 3B step [5]):**

```
q_adjusted ──(PT1 filter)──▶ q_applied ──▶ net.sgen.q_mvar
p_target   ──(ramp limit)──▶ p_applied ──▶ net.sgen.p_mw
```

`step()` takes the coordinator's `q_adjusted` (Doc 3B) and the profile `p_target`,
returns the physically achievable `(q_applied, p_applied)`, and **advances its own
state** (`q_prev`, `p_prev`) — this per-DER memory is what makes the whole HIL
loop *stateful*: each timestep's achievable output depends on where the previous
one ended.

**The resolution story (why this layer is often invisible).** At the 15-minute
annual-study resolution (`dt = 900 s`) both constraints are non-binding — the PT1
fully settles (`α → 1`, `q_applied = q_target`) and the ramp allowance dwarfs any
one profile step. At the 1-second HIL inner loop (`dt = 1 s`) both become visible.
This is exactly the `CoordinatorResult.dq_dynamics`/`dp_ramp` deltas (Doc 3A §B):
≈0 at 15-min, measurable at 1-s.

---

## 2. File logic flow — pseudocode

```
CONSTANTS: MV and LV presets — Q settling times (95% in 6/10/60 s) and
           P ramp rates (0.33 / 0.50 / 0.66 % of rated per second)

CLASS DERDynamics — one instance per network, holds per-DER state:

    __init__(dt, t95_q, ramp_rate, q_max[], p_rated[]):
        validate every argument (all strictly positive, matching 1-D shapes)
        derive the PT1 constants:  τ = t95/3 ;  α = 1 − exp(−dt/τ)
        derive the per-DER ramp allowance:  δP_max[i] = ramp_rate · p_rated[i] · dt
        create empty state (q_prev, p_prev = 0); mark "not initialised"

    reset(q_init, p_init):
        broadcast the initial Q and P to all DERs
        clip Q to ±q_max, P to [0, p_rated]; store as state; mark "initialised"
        (correct values: q_init=0 — control hasn't acted yet;
                         p_init=first profile row — the DER was already generating)

    step(q_target, p_target)  → (q_applied, p_applied):
        require reset() was called; require matching shapes
        Q:  q_applied = clip( q_prev + α·(q_target − q_prev),  ±q_max )   # PT1
        P:  p_ramped  = clip( p_target, p_prev ± δP_max )                 # ramp
            p_applied = clip( p_ramped, 0, p_rated )                       # physical
        advance state: q_prev ← q_applied ; p_prev ← p_applied
        return (q_applied, p_applied)

    alpha / n_ders / initialized : read-only accessors
    __repr__ : one-line debug string
```

---

## 3. Call-site map — where every entity is used

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `MV_T95_Q_S` (L113) | constant | `scenario_4_volt_var.py` `_build_dynamics` L143 (default MV preset). |
| `MV_T95_Q_SLOW_S` / `MV_T95_Q_FAST_S` (L114–115) | constants | Sensitivity-study presets; passed by callers doing τ sweeps. **No default use in tree.** Verdict: **intentional** (sensitivity presets). |
| `MV_RAMP_RATE_P_BASE` (L116) | constant | `scenario_4` `_build_dynamics` L144 (default). |
| `MV_RAMP_RATE_P_SLOW/FAST` (L117–118) | constants | Sensitivity presets. **No default use.** Verdict: **intentional**. |
| `LV_*` (L121–126) | constants | LV-network callers (the class is network-agnostic; LV runners pass these). |
| `DERDynamics` | class | Instantiated in `scenario_4_volt_var.py` `_build_dynamics` L142; type-hinted in `run_coordinated_timestep` (Doc 3B). |
| `__init__` | method | Via `_build_dynamics`. |
| `alpha` | property | **No caller in tree.** Verdict: **intentional** (inspection/test accessor; the class uses `self._alpha` internally). |
| `n_ders` | property | **No caller in tree.** Verdict: **intentional** (inspection accessor). |
| `initialized` | property | **No caller in tree.** Verdict: **intentional** (inspection accessor; `step()` uses `self._initialized`). |
| `reset` | method | `scenario_4_volt_var.py` L263 (once, before the timestep loop). |
| `step` | method | `run_coordinated_timestep` L991 (clean-timestep GATE, `q_target=0`) and L1051 (main path). |
| `__repr__` | method | **No explicit caller.** Verdict: **intentional** (debug/logging convenience). |

---

## 4. The physics: PT1 filter and ramp limiting

**PT1 (first-order lag) — the continuous model.** A real inverter cannot change
its reactive output instantaneously; VDE-AR-N 4110/4105 model the response as a
first-order lag:

```
τ · dq/dt + q = q_target
```

Its step response (constant `q_target`, from `q(0)`) is exponential:

```
q(t) = q_target + (q(0) − q_target)·e^(−t/τ)
```

**Why τ = t95 / 3.** The standard specifies the *settling time* — time to reach
95% of a step — not τ directly. Setting `q(t95) − q(0) = 0.95·(q_target − q(0))`:

```
1 − e^(−t95/τ) = 0.95  ⇒  e^(−t95/τ) = 0.05  ⇒  t95/τ = ln(20) = 2.996 ≈ 3
⇒  τ = t95 / 3
```

So the default `t95 = 10 s` gives `τ ≈ 3.33 s`.

**Why the discrete form is *exact*, not Euler.** Over one timestep `[t_{k−1}, t_k]`
of length `dt` with `q_target` held constant (which it is — it is the coordinator's
per-timestep output), the ODE solution evaluated at `t_k` is:

```
q[k] = q_target + (q[k−1] − q_target)·e^(−dt/τ)
     = q[k−1] + (1 − e^(−dt/τ))·(q_target − q[k−1])
     = q[k−1] + α·(q_target − q[k−1]) ,   α = 1 − e^(−dt/τ)
```

This is the **analytic** solution for a piecewise-constant input — no
discretisation error, unlike a forward-Euler `q[k] = q[k−1] + (dt/τ)(q_target −
q[k−1])` which would be only first-order accurate and could overshoot for large
`dt/τ`. The exact form is unconditionally stable and correct for any `dt`.

**Ramp limiting — the P constraint.** Active power may change by at most a fixed
fraction of rated power per second, so per timestep:

```
δP_max = ramp_rate · p_rated · dt        [MW/step]
p_ramped = clip(p_target, p_prev − δP_max, p_prev + δP_max)
```

then a physical clip to `[0, p_rated]`.

**Resolution behaviour (both constraints).**

| | `dt = 1 s` (HIL) | `dt = 900 s` (15-min annual) |
|--|------------------|------------------------------|
| `α = 1 − e^(−dt/τ)` | `1 − e^(−1/3.33) = 0.259` (partial) | `1 − e^(−270) ≈ 1.0` (full settle) |
| `δP_max` (ramp 0.5%, `p_rated=0.5`) | `0.0025 MW/step` (binding) | `2.25 MW/step` ≫ 0.5 (non-binding) |

At 15-min both are inert — `q_applied = q_target`, `p_applied = p_target` — which
is *physically correct*: 15 minutes is far longer than any DER's few-second
response, so the dynamics have fully played out within one profile step.

---

## 5. Section A — Preset constants (L112–126)

```python
# VDE-AR-N 4110 (MV)
MV_T95_Q_S:          float = 10.0    # s — default 3τ time
MV_T95_Q_SLOW_S:     float = 60.0    # s — slow sensitivity bound
MV_T95_Q_FAST_S:     float = 6.0     # s — fast sensitivity bound
MV_RAMP_RATE_P_BASE: float = 0.005   # 0.50% PN/s — central modelling value
MV_RAMP_RATE_P_SLOW: float = 0.0033  # 0.33% PN/s
MV_RAMP_RATE_P_FAST: float = 0.0066  # 0.66% PN/s
# VDE-AR-N 4105 (LV) — same time constants; Qmax differs
LV_T95_Q_S = 10.0 ; LV_T95_Q_SLOW_S = 60.0 ; LV_T95_Q_FAST_S = 6.0
LV_RAMP_RATE_P_BASE = 0.005 ; LV_RAMP_RATE_P_SLOW = 0.0033 ; LV_RAMP_RATE_P_FAST = 0.0066
```

#### Line 112–126: The MV and LV preset menus

Two sets of presets, one per standard, each with a **base** (central modelling
value), a **slow** bound, and a **fast** bound — for the settling time and the
ramp rate. **Design rationale — a menu, not a hardcode.** The class takes these as
*arguments*, so a run can select the central values (default) or a slow/fast
bound for a **sensitivity study** (does the conclusion hold if DERs respond at 6 s
vs 60 s?). The MV and LV settling times are identical (6/10/60 s); only the Q
*ceiling* differs between standards, and that enters via the `q_max_mvar`
argument, not here.

> **Note on the module docstring (L23, L31).** The docstring cites the VDE
> nominal `Qmax = 0.48 · Pb,inst` (MV, cos φ ≈ 0.90) and `0.33 · PAmax` (LV). The
> *actual* `q_max_mvar` passed in by `scenario_4._build_dynamics` is
> `_vvc.Q_RATIO · p_rated` with the call-time `Q_RATIO = 0.25`, **not** 0.48 —
> the 0.48 is the standard's illustrative figure, not the value in force. Same
> family as the `Q_RATIO` stale-value flags in Doc 1A; noted so a reader does not
> take 0.48 as the operative ceiling.

**Where used:** `MV_T95_Q_S` and `MV_RAMP_RATE_P_BASE` are the defaults in
`_build_dynamics`; the SLOW/FAST variants are sensitivity presets with no default
call site (§3).

---

## 6. Section B — Class & __init__ (L133–271)

#### Line 204–243: Exhaustive constructor validation

```python
        if dt_s <= 0.0:                 raise ValueError(...)
        if t95_q_s <= 0.0:              raise ValueError(...)
        if ramp_rate_p_frac_s <= 0.0:   raise ValueError(...)
        q_max   = np.asarray(q_max_mvar, dtype=float)
        p_rated = np.asarray(p_rated_mw, dtype=float)
        if q_max.ndim != 1: raise ValueError(...)          # 1-D only
        if p_rated.ndim != 1: raise ValueError(...)
        if q_max.shape != p_rated.shape: raise ValueError(...)   # matching lengths
        if q_max.size == 0: raise ValueError(...)          # non-empty
        if np.any(q_max <= 0.0):   raise ValueError(f"... Non-positive at DER indices: {bad}")
        if np.any(p_rated <= 0.0): raise ValueError(f"... Non-positive at DER indices: {bad}")
```

Every argument is validated before any state is built: the three scalars strictly
positive, the two arrays 1-D, same shape, non-empty, and strictly positive
element-wise (naming the offending DER indices). **Design rationale — fail at
construction, not mid-loop.** A zero or negative `q_max`/`p_rated` would poison the
PT1 clip or the ramp scaling *thousands of timesteps later* with no obvious cause;
validating here turns it into an immediate, indexed error. This is why
`_build_dynamics` (scenario_4 L137–138) replaces any non-positive `p_rated`/`q_max`
with a `1e-6` sentinel *before* construction — to satisfy this validation for an
inert DER rather than crash.

#### Line 245–257: Store parameters and derive the dynamics constants

```python
        self._dt_s = float(dt_s); self._t95_q_s = float(t95_q_s)
        self._ramp_rate_p_frac_s = float(ramp_rate_p_frac_s)
        self._q_max = q_max; self._p_rated = p_rated; self._n_ders = q_max.size
        tau = t95_q_s / 3.0
        self._alpha  = float(1.0 - np.exp(-dt_s / tau))
        self._dp_max = ramp_rate_p_frac_s * p_rated * dt_s      # per-DER MW/step
```

**The two derived constants (§4).** `τ = t95/3` (the 95%-settling relationship);
`α = 1 − e^(−dt/τ)` (the exact discrete PT1 coefficient, a scalar — all DERs share
the same time constant); `δP_max = ramp_rate · p_rated · dt` (a **per-DER** array,
since each DER's ramp allowance scales with its own rated power). **Units:** `α`
dimensionless; `δP_max` = (1/s)·MW·s = MW ✓. **Worked example** (dt=900, t95=10,
τ=3.33): `α = 1 − e^(−270) ≈ 1.0`; (dt=1): `α = 1 − e^(−0.3) = 0.259`.
`δP_max = 0.005·0.5·900 = 2.25 MW` (dt=900) vs `0.0025 MW` (dt=1) for a 0.5 MW DER.

#### Line 259–271: Uninitialised state + debug log

```python
        self._initialized = False
        self.q_prev = np.zeros(self._n_ders, dtype=float)
        self.p_prev = np.zeros(self._n_ders, dtype=float)
        logger.debug("DERDynamics created: ... α=%.6f, dp_max=[%.4f..%.4f] MW/step", ...)
```

State starts zeroed and `_initialized = False` — `step()` refuses to run until
`reset()` sets real initial values. **Design rationale:** construction derives the
*constants*; the *state* is a separate concern set at loop start, so the same
`DERDynamics` object can be reset and reused across scenario restarts. **Where
used:** instantiated in `_build_dynamics` (scenario_4 L142).

---

## 7. Section C — Properties (L277–290)

```python
    @property
    def alpha(self):       return self._alpha
    @property
    def n_ders(self):      return self._n_ders
    @property
    def initialized(self): return self._initialized
```

#### Line 277–290: Read-only inspection accessors

Three read-only views of internal state: the PT1 coefficient, the DER count, and
whether `reset()` has run. **Design rationale:** expose the computed `α` and the
init flag for tests, logging, and external inspection without allowing mutation
(no setters). The class's own code reads the private `self._alpha`/`self._initialized`
directly, so these public properties are purely for outside consumers. **Where
used:** none in the current tree — inspection/test API (verdicts: intentional,
§3).

---

## 8. Section D — reset (L296–353)

```python
        try:
            q = np.broadcast_to(np.asarray(q_init, dtype=float), (self._n_ders,)).copy()
        except ValueError:
            raise ValueError(f"q_init shape ... cannot be broadcast to (n_ders={self._n_ders},)")
        try:
            p = np.broadcast_to(np.asarray(p_init, dtype=float), (self._n_ders,)).copy()
        except ValueError:
            raise ValueError(f"p_init shape ... cannot be broadcast ...")
        self.q_prev = np.clip(q, -self._q_max, self._q_max)
        self.p_prev = np.clip(p,  0.0,         self._p_rated)
        self._initialized = True
```

#### Line 296–353: Initialise per-DER state before the loop

Broadcasts `q_init`/`p_init` (scalar or array) to `(n_ders,)`, clips to physical
bounds (Q to `±q_max`, P to `[0, p_rated]`), stores as `q_prev`/`p_prev`, and marks
initialised. **Design rationale — the `.copy()` after `broadcast_to`.**
`np.broadcast_to` returns a *read-only view*; `.copy()` makes an independent
writable array so `step()` can advance it. **The correct initial values** (docstring
L312–319): `q_init = 0.0` (reactive control has not acted at the study-window
start) and `p_init =` the first profile row (the DER was already generating at that
level). **Design rationale — re-callable:** resetting mid-run is allowed (scenario
restart) and discards accumulated state cleanly. **Worked example:**
`reset(q_init=0.0, p_init=[0.3, 0.4])` → `q_prev = [0, 0]`, `p_prev = [0.3, 0.4]`
(both within bounds). **Where used:** scenario_4 L263, once before the loop.

---

## 9. Section E — step (L359–458)

The core. Applies the PT1 to Q and ramp-limiting to P, advances state, returns the
achievable setpoints.

#### Line 411–430: Preconditions

```python
        if not self._initialized:
            raise RuntimeError("DERDynamics.step() called before reset(). Call reset(q_init=0.0, p_init=p_target_t0) ...")
        q_t = np.asarray(q_target, dtype=float); p_t = np.asarray(p_target, dtype=float)
        if q_t.shape != (self._n_ders,): raise ValueError("... Must match sgen_indices ordering ...")
        if p_t.shape != (self._n_ders,): raise ValueError("...")
```

Refuses to run before `reset()` (uninitialised `q_prev`/`p_prev` would silently
produce wrong results), and requires both inputs to be exactly `(n_ders,)` in
`sgen_indices` order — a shape mismatch means Q/P aligned to the wrong DER set.
**Design rationale:** the same "fail on misalignment, don't guess" discipline as
the controller and coordinator; the error text names the required ordering.

#### Line 432–437: The PT1 filter on Q

```python
        q_applied = np.clip(
            self.q_prev + self._alpha * (q_t - self.q_prev),
            -self._q_max, self._q_max,
        )
```

**§4's exact discrete PT1 in code:** `q_applied = q_prev + α·(q_target − q_prev)`,
then clipped to `±q_max`. **Code-to-physics mapping:** `q_prev` = last step's
achieved Q; `α·(q_target − q_prev)` = the fraction of the remaining gap closed this
step; the clip enforces the reactive ceiling. **Units:** MVAr throughout (α
dimensionless). **Worked example** (dt=1, α=0.259, `q_prev=0`, `q_target=−0.125`):
`q_applied = 0 + 0.259·(−0.125) = −0.0324 MVAr` — 26% of the way in one 1 s step;
successive steps converge exponentially, reaching 95% (`−0.119`) by t≈10 s (t95).
At dt=900 (α≈1): `q_applied = q_target = −0.125` in one step — fully settled.

#### Line 439–444: Ramp limiting then physical clip on P

```python
        p_applied = np.clip(
            np.clip(p_t, self.p_prev - self._dp_max, self.p_prev + self._dp_max),
            0.0, self._p_rated,
        )
```

**§4's ramp constraint in code — a nested clip.** The *inner* clip limits
`p_target` to within `±δP_max` of `p_prev` (the ramp constraint); the *outer* clip
enforces the physical `[0, p_rated]` envelope. **Design rationale — order matters:**
ramp-limit first (relative to where we were), then floor/ceiling (absolute
physical bound), so a large profile jump is first slowed to the ramp rate and only
then, if still out of range, clamped to the hardware limit. **Units:** MW.
**Worked example** (dt=1, ramp=0.005, `p_rated=0.5` → `δP_max=0.0025`, `p_prev=0.30`,
`p_target=0.45`): inner clip → `clip(0.45, 0.2975, 0.3025) = 0.3025` (the 0.15 MW
jump is throttled to 2.5 kW); outer clip → `0.3025` (in range). At dt=900
(`δP_max=2.25`): inner clip → `0.45` (unthrottled); the DER reaches the profile in
one step.

#### Line 446–458: Advance state and return

```python
        self.q_prev = q_applied.copy()
        self.p_prev = p_applied.copy()
        logger.debug("DERDynamics.step(): max|q_target−q_applied|=%.6f MVAr, max|p_target−p_applied|=%.6f MW", ...)
        return q_applied, p_applied
```

**The statefulness.** `q_prev`/`p_prev` are updated to *this* step's achieved
values (`.copy()` so the returned arrays and the stored state are independent),
which become the starting point for the next step's PT1 and ramp. This per-DER
memory is what the scenario-4 docstring means by "the loop is stateful." The debug
log reports the two gaps (`dq_dynamics`, `dp_ramp` — Doc 3A §B). **Where used:**
`run_coordinated_timestep` L991 and L1051 (Doc 3B step [5]).

---

## 10. Section F — __repr__ (L464–474)

```python
    def __repr__(self):
        return (f"DERDynamics(dt={self._dt_s}s, t95={self._t95_q_s}s, α={self._alpha:.6f}, "
                f"ramp={self._ramp_rate_p_frac_s*100:.2f}%/s, n_ders={self._n_ders}, "
                f"initialized={self._initialized})")
```

#### Line 464–474: One-line debug representation

Renders the key parameters (dt, t95, the derived α, ramp % , DER count, init flag)
for logging and interactive inspection. **Design rationale:** surfaces the
*derived* `α` — the single most useful number for sanity-checking whether the
dynamics are in the "visible" (α<1) or "settled" (α≈1) regime for the chosen `dt`.
**Where used:** no explicit caller — debug/logging convenience (verdict:
intentional, §3).

---

## 11. Coverage checklist

Every named entity in `der_dynamics.py`, each confirmed documented.

| Entity | Lines | Kind | Documented | Where-used stated |
|--------|-------|------|:----------:|:-----------------:|
| `MV_T95_Q_S` | 113 | constant | ✅ | ✅ |
| `MV_T95_Q_SLOW_S` / `MV_T95_Q_FAST_S` | 114–115 | constants | ✅ | ✅ (sensitivity presets) |
| `MV_RAMP_RATE_P_BASE` | 116 | constant | ✅ | ✅ |
| `MV_RAMP_RATE_P_SLOW` / `_FAST` | 117–118 | constants | ✅ | ✅ (sensitivity presets) |
| `LV_T95_Q_S` … `LV_RAMP_RATE_P_FAST` | 121–126 | constants | ✅ | ✅ |
| `DERDynamics` | 133–474 | class | ✅ | ✅ |
| `__init__` | 195–271 | method | ✅ | ✅ |
| `alpha` | 277–280 | property | ✅ | ✅ (no caller — intentional) |
| `n_ders` | 282–285 | property | ✅ | ✅ (no caller — intentional) |
| `initialized` | 287–290 | property | ✅ | ✅ (no caller — intentional) |
| `reset` | 296–353 | method | ✅ | ✅ |
| `step` | 359–458 | method | ✅ | ✅ |
| `__repr__` | 464–474 | method | ✅ | ✅ (no caller — intentional) |

**Also covered (not standalone entities):** the `τ = t95/3` derivation, the exact-
vs-Euler PT1 justification, the `broadcast_to().copy()` read-only-view fix
(L326–337), the ramp-then-physical nested-clip order (L440–444), and the state-
advance `.copy()` (L447–448).

**Flags raised (not edited):** the module docstring's `Qmax = 0.48 · Pb,inst`
(L23) is the VDE nominal, not the operative ceiling (`Q_RATIO = 0.25` is passed in
by `_build_dynamics`).

**Uncalled public entities (verdicts, §3):** `alpha`, `n_ders`, `initialized`,
`__repr__` (all intentional — inspection/debug API); `MV_T95_Q_SLOW/FAST_S`,
`MV_RAMP_RATE_P_SLOW/FAST` (intentional — sensitivity-study presets).

---

*End of Document 4. Next per the queue: Document 5 — Arduino Firmware
(`volt_var_arduino.ino`), the device-side mirror of the Q(V) law (1A) and the
serial protocol (1B), and the source of the Q-formatting leg of the quantisation
finding.*
