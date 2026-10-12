# TASK 4 — DOCUMENT 1A  (v2)
## Q(U) Characteristic Core — `volt_var_controller.py`, Lines 126–408
### Complete line-by-line walkthrough

> **Source verified:** lines 126–408 of the current `volt_var_controller.py`
> were read directly before writing (v2 extends v1's 126–394 range after
> `slope_inject`/`slope_absorb` were found to have been truncated past). If the
> file is edited, re-verify the line numbers cited here.
>
> **Read alongside:** Master Index §5 (mandatory sections), §6 (block format),
> §7 (conventions), §8 (canonical worked-example inputs). Read **Document 5
> (Arduino firmware)** immediately after this — the Arduino `compute_q()` is
> the exact mirror of the `QVCharacteristic` documented here.
>
> **Flowchart:** the Q(V) exchange flow diagram from Task 1a
> (`flow_volt_var_exchange.*`).

---

## TABLE OF CONTENTS

1. [Overview — what this slice of the file is](#1-overview)
2. [File logic flow — pseudocode (this slice)](#2-file-logic-flow--pseudocode-this-slice)
3. [Call-site map — where every entity is used](#3-call-site-map--where-every-entity-is-used)
4. [Section A — Characteristic constants (L126–143)](#4-section-a--characteristic-constants-l126143)
5. [Section B — set_qv_parameters() (L146–235)](#5-section-b--set_qv_parameters-l146235)
6. [Section C — reset_qv_parameters() (L238–240)](#6-section-c--reset_qv_parameters-l238240)
7. [Section D — QVCharacteristic class (L282–408)](#7-section-d--qvcharacteristic-class-l282408)
   - [compute_setpoint — the five-segment law (L297–347)](#compute_setpoint--the-five-segment-law-l297347)
   - [compute_setpoints — vectorised (L349–389)](#compute_setpoints--vectorised-l349389)
   - [q_max_mvar (L391–394)](#q_max_mvar-l391394)
   - [slope_inject (L396–402)](#slope_inject-l396402)
   - [slope_absorb (L404–408)](#slope_absorb-l404408)
8. [The characteristic as one picture](#8-the-characteristic-as-one-picture)
9. [Failure modes and customisation](#9-failure-modes-and-customisation)
10. [Coverage checklist](#10-coverage-checklist)

---

## 1. Overview

This slice of `volt_var_controller.py` is the **pure control law**: given a bus
voltage, what reactive power should a DER produce? It has **no hardware and no
network dependency** — it is arithmetic on floats. Everything downstream (the
serial interface in 1B, the HIL loop in 1C, the Arduino firmware in 5) exists
to *feed* this law voltages and *carry* its answers to and from silicon.

Four things live here:

- **The constants** (L126–143) that define the shape of the curve — the *single
  source of truth*, transmitted to the Arduino at session start via `CFG:`.
- **`set_qv_parameters()` / `reset_qv_parameters()`** (L146–240): the runtime
  override path, with validation that mirrors the firmware's rejection matrix.
- **`QVCharacteristic`** (L282–408): the stateless implementation of the
  VDE-AR-N 4110 Bild 8 piecewise-linear Q(U) characteristic, with two
  diagnostic slope helpers.

**How 1A connects to 1B and 1C** (the same file, split three ways): 1A is the
*law*; **1B** (`ArduinoSerialInterface`) is the *wire* that carries voltages to
the Arduino and Q values back, and that transmits these constants via `CFG:`;
**1C** (`VoltVarController`) is the *loop* that, each timestep, runs a power
flow, reads bus voltages, calls either 1A's `compute_setpoints` (dry-run) or
1B's serial exchange (hardware), clamps the result, applies it, and re-runs the
power flow. The constants defined in 1A are read at call time by all three.

---

## 2. File logic flow — pseudocode (this slice)

Plain-language flow of the 1A slice — *what happens in what order*, no Python.

```
ON MODULE LOAD:
    define the four voltage breakpoints U1<U2<U3<U4  (the curve's corners)
    define Q_RATIO                                    (reactive capability fraction)
    define PLAUSIBLE_PU_RANGE                          (typo-catcher band for breakpoints)
    take a frozen snapshot of {Q_RATIO, U1..U4} into _QV_DEFAULTS
        → so a later reset can restore the originals

FUNCTION set_qv_parameters(optionally: q_ratio, u1, u2, u3, u4):
    FOR each of the five parameters:
        if the caller supplied it   → use it (coerced to a real number)
        if the caller left it out    → keep the current module value
    VALIDATE the resulting five values — if ANY check fails, raise and change NOTHING:
        (1) all five are finite (no NaN, no infinity)
        (2) 0 < q_ratio ≤ 1
        (3) breakpoints strictly increasing: u1 < u2 < u3 < u4
        (4) every breakpoint lies inside PLAUSIBLE_PU_RANGE
    COMMIT: write all five values to the module globals in one atomic step
    RETURN the committed values (so the caller can log what took effect)

FUNCTION reset_qv_parameters():
    feed the frozen _QV_DEFAULTS back through set_qv_parameters()
        → restores canonical curve, re-using the same validation + atomic commit

THE CHARACTERISTIC (QVCharacteristic) — pure, stateless:

    compute_setpoint(voltage V, installed capacity P) → one DER's Q:
        if P is not finite      → return 0 (and warn)
        if V is not finite      → return 0 (and warn)          [e.g. a diverged PF]
        q_max = Q_RATIO × |P|
        SELECT by where V falls on the curve:
            V ≤ U1                → +q_max          (saturate: inject hard)
            U1 < V < U2           → +q_max, ramped down toward 0
            U2 ≤ V ≤ U3           → 0               (deadband: do nothing)
            U3 < V < U4           → −q_max, ramped up from 0
            V ≥ U4                → −q_max          (saturate: absorb hard)
        return the signed Q

    compute_setpoints(voltages[], capacities[]) → all DERs' Q:
        require the two series to share an identical index (same DERs, same order)
            → else raise, because a positional mismatch silently mis-assigns Q
        apply compute_setpoint to every (V, P) pair
        return the results as a series labelled by DER index

    q_max_mvar(P)      → Q_RATIO × |P|            (the saturation ceiling alone)
    slope_inject()     → Q_RATIO / (U2 − U1)      (steepness of the inject ramp)
    slope_absorb()     → −Q_RATIO / (U4 − U3)     (steepness of the absorb ramp)
```

---

## 3. Call-site map — where every entity is used

Every entity in this slice and where it is called or read from in the whole
tree (verified by grep). Entities with **no caller** are marked — that is a
finding, not an oversight.

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `U1_PU`..`U4_PU` | constants | Read at call time by `compute_setpoint` (L338–345), `slope_inject`/`slope_absorb` (L402/408), and `set_qv_parameters` validation (L206–222). Transmitted to the Arduino by `ArduinoSerialInterface.configure()`'s `CFG:` message (Doc 1B). |
| `Q_RATIO` | constant | Read by `compute_setpoint` (L332), `q_max_mvar` (L394), `slope_inject`/`slope_absorb`. **Imported by** `scenario_5_opf.py` L73 (`from volt_var_controller import Q_RATIO`). Rebindable via `set_qv_parameters`. |
| `PLAUSIBLE_PU_RANGE` | constant | Used in `set_qv_parameters` validation (L222). **Imported by** `wizard.py` L6 — the wizard reuses it to sanity-check breakpoint entry at prompt time. |
| `_QV_DEFAULTS` | constant (dict) | Read only by `reset_qv_parameters` (L240). |
| `set_qv_parameters` | function | **`executor.py` L488** (CLI wizard path, per-run override); **`run_benchmark_script.py` L183** (direct-script path, `--q-ratio`/`--u1..--u4`). Referenced in docstrings of `wizard.py`, `scenario_4_volt_var.py`, `sensitivity_coordinator.py`, `run_plan.py`. |
| `reset_qv_parameters` | function | **No current call site in the tree.** Provided as a between-runs reset for interactive/notebook/test use in a shared interpreter (curve-side analogue of `violation_detector.reset_limits()`). |
| `QVCharacteristic` | class | **Imported by** `hosting_capacity.py` L110, `scenario_4_volt_var.py` L86, `sensitivity_coordinator.py` L95. |
| `compute_setpoint` | staticmethod | Called **only** by `compute_setpoints` (L386) via `np.vectorize`. Its logic is independently *mirrored* (re-implemented, not called) in `volt_var_arduino.ino`'s `compute_q()` and in `compare_runs_updated.ipynb` L1409 for offline analysis. |
| `compute_setpoints` | staticmethod | **The workhorse.** `VoltVarController.run_timestep` L1179 (dry-run path); `hosting_capacity.py` L66 and L435 (HC Q(V) fixed point); `sensitivity_coordinator.py` L1016 and L1031 (computes the local-Q(V) baseline `q_initial`, and the serial-failure fallback). Named as the schema default `function:` in `plugin_runner.py` L19 — custom controller plugins mirror this signature. |
| `q_max_mvar` | staticmethod | **No current call site in the tree.** Public helper. *Note:* the identically-named `q_max_mvar` *parameter* in `der_dynamics.py` and `scenario_4_volt_var.py` (L146) is a different thing — it is derived from the controller's `p_installed_mw` and passed into `DERDynamics`; it does **not** call this method. |
| `slope_inject` | staticmethod | **No current call site in the tree.** Diagnostic helper: reports the injection-ramp gain dQ/dV per MW. |
| `slope_absorb` | staticmethod | **No current call site in the tree.** Diagnostic helper: reports the absorption-ramp gain. |

**Consolidated finding + verdicts** (per the standing uncalled-API convention,
Master Index §5). Four public entities have no caller in the current tree. All
are pure and cheap — legitimate API, not correctness-risking dead code. Verdict
for each:

- **`reset_qv_parameters` → intentionally uncalled.** Every run explicitly calls
  `set_qv_parameters` (executor L488, run_benchmark_script L183), so a reset in
  the executor path would be redundant. Its value is interactive/notebook/test
  use in a shared interpreter. Recommendation: leave an "intentionally uncalled —
  public reset API" marker; optionally call in test-teardown fixtures. Do **not**
  wire into the executor.
- **`q_max_mvar` → wire into diagnostics.** It is the canonical per-DER Q
  ceiling. Any per-DER headroom reporting (a `diagnose_*` script,
  `compare_scenarios`, or coordinator headroom logging) should call it rather
  than re-derive `Q_RATIO·|P|`. Intentional marker until wired.
- **`slope_inject` / `slope_absorb` → wire into diagnostics.** Natural home: a
  Q(V)-tuning summary (the `read_qv_params` cell in `compare_runs_updated.ipynb`,
  or a diagnose script that echoes the active characteristic). When wiring, fix
  `slope_inject`'s stale "16.0" docstring and reconcile the sign-convention
  asymmetry (`slope_inject` returns a magnitude, `slope_absorb` a signed slope).
  Intentional marker until then.

**Flagged, not changed** — these are recorded in the Master Index §5 register.

---

## 4. Section A — Characteristic constants (L126–143)

```python
# Lines 126-138
# ===========================================================================
# Characteristic constants -- VDE-AR-N 4110 Bild 8
# SINGLE SOURCE OF TRUTH: these values are pushed to the Arduino at session
# startup via the CFG: message (ArduinoSerialInterface.configure()). The
# constants in volt_var_arduino.ino are boot defaults only. Edit here only.
# ===========================================================================

U1_PU   = 0.96    # lower saturation  -- full Q injection at or below this
U2_PU   = 0.99    # deadband lower edge
U3_PU   = 1.01    # deadband upper edge
U4_PU   = 1.04    # upper saturation  -- full Q absorption at or above this
Q_RATIO = 0.25    # Q_max / P_b_inst  (VDE-AR-N 4110 Bild 8). Modifying to 0.15 from 0.48 to check control response
PLAUSIBLE_PU_RANGE = (0.5, 2.5)
```

#### Line 133–136: The characteristic's four corners

**Derivation from first principles.** VDE-AR-N 4110 is the German MV
grid-connection standard; its Bild 8 specifies a Q(U) characteristic as a
piecewise-linear curve with two saturation plateaus, two linear ramps, and a
central deadband — fully determined by four voltage breakpoints and one
magnitude (Q_max).

```
          Q
       +Qmax ┤■■■■■■■─╲
             │        ╲                     ■ = saturated (constant Q)
           0 ┤         ╲■■■■■■■■■╲          ╲ = linear ramp
             │          U2      U3 ╲          (flat middle = deadband)
       -Qmax ┤                     ╲─■■■■■■■
             └──┬────┬────┬────┬────┬──── U (pu)
               U1   U2  1.00  U3   U4
              0.96 0.99      1.01 1.04
```

- **`U1_PU = 0.96`** — at/below, DER fully injects Q (max capacitive support).
- **`U2_PU = 0.99`** — lower deadband edge; U1→U2 injection ramps +Q_max → 0.
- **`U3_PU = 1.01`** — upper deadband edge; U2..U3 the DER does nothing.
- **`U4_PU = 1.04`** — at/above, DER fully absorbs Q; U3→U4 absorption ramps
  0 → −Q_max.

**Code-to-physics mapping.** All four are **per-unit** voltages (dimensionless
ratios of actual to nominal). Deadband half-width `U3 − 1.00 = 0.01 pu (1%)`;
full control band `U4 − U1 = 0.08 pu (8%)`.

**Units check.** Pure per-unit (V/V), dimensionless. Ramp widths
`U2 − U1 = U4 − U3 = 0.03 pu` — symmetric, which is why equal off-deadband
excursions give equal |Q| (§8).

**Worked numerical example.** A DER whose bus sits at 1.005 pu is inside
`[0.99, 1.01]` → does nothing (no wasted reactive current, no OLTC hunting).
Only outside that ±1% window does it act.

**Design rationale.**
- *Why a deadband?* Without it the DER chases every trivial wobble, producing
  continuous reactive current/losses and hunting against the OLTC. ±1% dead zone
  buys stability for free.
- *Why strictly increasing?* The ramp denominators are `(U2−U1)` and `(U4−U3)`;
  a collision would divide by zero. Enforced in `set_qv_parameters` (L206–211).
- *Why 0.96/0.99/1.01/1.04?* Canonical VDE-AR-N 4110 Bild 8 values; the
  demonstration baseline (at `Q_RATIO=0.25` on `--1-sw`, S4B cut violations 46%
  vs S4A, 9.1% coordinator firing).

**Where used.** Read at call time by `compute_setpoint`, the slope helpers, and
`set_qv_parameters` validation; transmitted to the Arduino via `CFG:` (Doc 1B).

---

#### Line 137: `Q_RATIO` — the reactive-to-active capacity ratio

**Derivation.** `Q_max = Q_RATIO · P_installed`. VDE-AR-N 4110 caps reactive
capability as a ratio of rated active power; `0.25` = 25% of rated MW available
as MVAr. Ties each DER's control authority to its size automatically.

**Units check.** `Q_RATIO` is `Q_max / P_b_inst` → **MVAr·MW⁻¹**:

```
Q_max [MVAr] = Q_RATIO [MVAr·MW⁻¹] · P_installed [MW]     (MW cancels → MVAr) ✓
```

**Worked example.** DER-A: `0.25 · 0.5 = 0.125 MVAr`. DER-B: `0.25 · 1.5 =
0.375 MVAr`.

**Design rationale — and a flagged stale comment.** 0.25 keeps the coordinator
firing at a measurable rate (0.30 or 0.48 make the local Q(V) resolve
everything, so the coordinator never fires — memory of the confirmed tuning).
**⚠ The inline comment "Modifying to 0.15 from 0.48" does not match the assigned
value 0.25** (neither 0.15 nor 0.48). Leftover from earlier tuning. **Flagged,
not edited.**

**Where used.** `compute_setpoint` L332, `q_max_mvar` L394, both slope helpers;
**imported by `scenario_5_opf.py` L73**; rebindable via `set_qv_parameters`.

---

#### Line 138: `PLAUSIBLE_PU_RANGE` — the breakpoint sanity band

A `(0.5, 2.5)` pu typo-catcher for `set_qv_parameters`. Monotonicity alone does
not catch a strictly-increasing-but-absurd value (`u4 = 6.0` typed for `1.06`
still satisfies `u3 < u4`). Real networks stay within ~±20% of nominal, so
anything outside [0.5, 2.5] pu is almost certainly a data-entry error. This
check is **deliberately not in the firmware** — catch the typo on the host,
where the error message can be long and specific.

**Where used.** `set_qv_parameters` L222; **imported by `wizard.py` L6** for
prompt-time breakpoint validation.

---

#### Line 142–143: `_QV_DEFAULTS` — frozen defaults for reset and provenance

```python
_QV_DEFAULTS = {"q_ratio": Q_RATIO, "u1": U1_PU, "u2": U2_PU,
                "u3": U3_PU, "u4": U4_PU}
```

A snapshot of the five constants **as defined at import time**. Because
`set_qv_parameters` rebinds the globals, the originals would be lost unless
saved; this dict is that save, consumed by `reset_qv_parameters`. Leading
underscore = module-private. Capture timing is intentional: the compile-time
literals *are* the defaults by definition.

**Where used.** Read only by `reset_qv_parameters` (L240).

---

## 5. Section B — set_qv_parameters() (L146–235)

The runtime override — changes the curve for one run without editing the file,
while guaranteeing the dry-run curve, the coordinator's Q_max sizing, and the
value sent to the Arduino all move *together* (single source of truth).

```python
# Lines 146-152
def set_qv_parameters(
        q_ratio: float = None,
        u1:      float = None,
        u2:      float = None,
        u3:      float = None,
        u4:      float = None,
) -> dict:
```

#### Line 146–152: Signature — every parameter optional, defaulting to None

**Parameters:** `q_ratio` (float, MVAr·MW⁻¹, e.g. 0.30, range `(0,1]`); `u1..u4`
(float, pu, e.g. 0.96, range within `PLAUSIBLE_PU_RANGE`, strictly increasing).
`None` on any → keep current. **Returns** the committed `{q_ratio,u1..u4}` dict.

**Design rationale — why `None`, not current-value defaults.** `None` gives
*partial override*: `set_qv_parameters(q_ratio=0.30)` leaves the breakpoints
untouched. Current-value defaults would be captured at *definition* time and
would not track later overrides — a mutable-default-style trap.

**Where used.** Called by `executor.py` L488 (wizard path) and
`run_benchmark_script.py` L183 (direct-script path) — the two entry points.

---

#### Line 187: The `global` declaration

```python
    global Q_RATIO, U1_PU, U2_PU, U3_PU, U4_PU
```

Declares that assignments rebind the *module-level* constants, not create
locals. Without it, the L233 commit would silently create function-locals and
the module constants would never change — the override would appear to work
while every consumer read the old value. **Load-bearing for the entire
single-source-of-truth guarantee.**

---

#### Line 189–193: None-keeps-current resolution

```python
    new_q  = float(q_ratio) if q_ratio is not None else Q_RATIO
    new_u1 = float(u1)      if u1      is not None else U1_PU
    new_u2 = float(u2)      if u2      is not None else U2_PU
    new_u3 = float(u3)      if u3      is not None else U3_PU
    new_u4 = float(u4)      if u4      is not None else U4_PU
```

Compute five concrete floats into `new_*` without writing back yet. `float()`
coercion guards against NumPy scalars / CLI/YAML strings (`"0.30"` → `0.30`; a
non-numeric string raises here, early). First half of a **parse-then-commit**
pattern (finished L231–233): nothing observable changes until every check
passes.

---

#### Line 195–200: Finiteness check

```python
    vals = (new_q, new_u1, new_u2, new_u3, new_u4)
    if not all(np.isfinite(v) for v in vals):
        raise ValueError(... "(firmware ERR:CFG_INVALID).")
```

Rejects `NaN`/`±inf`. A `NaN` breakpoint would poison every comparison
(`V <= NaN` is `False`), sending the DER into a wrong branch. The error text
ends `(firmware ERR:CFG_INVALID)` — a signpost that this validation **mirrors
the Arduino's rejection matrix**: a value rejected here is one the firmware
would also reject. Recurring theme: fail on the host where the wire would fail,
but with a better message.

---

#### Line 201–205: q_ratio bound

```python
    if not (0.0 < new_q <= 1.0):
        raise ValueError(... "(firmware ERR:CFG_INVALID).")
```

**Derivation.** `Q_max = Q_RATIO·P`. Ratio 0 disables the DER (no authority);
ratio > 1 asks for more MVAr than rated MW (beyond typical inverter apparent
power). Bound `(0, 1]`. **Worked example:** `0.30` passes; `1.5` fails; `0`
fails.

---

#### Line 206–211: Strict monotonicity

```python
    if not (new_u1 < new_u2 < new_u3 < new_u4):
        raise ValueError(... "(firmware ERR:CFG_INVALID).")
```

**Derivation — why *strict*.** The ramps divide by `(U2−U1)` and `(U4−U3)`.
`U1==U2` or `U3==U4` → division by zero. Strict `<` keeps both denominators
strictly positive; the same guard the AVR enforces before accepting `CFG:`.
**Worked example:** `set_qv_parameters(u2=0.96)` gives `0.96 < 0.96` → fails,
catching the collision before it can zero a denominator.

---

#### Line 213–229: Plausibility-range check

```python
    breakpoint_names = ("u1", "u2", "u3", "u4")
    for name, v in zip(breakpoint_names, (new_u1, new_u2, new_u3, new_u4)):
        if not (PLAUSIBLE_PU_RANGE[0] <= v <= PLAUSIBLE_PU_RANGE[1]):
            raise ValueError(f"{name}={v} is outside the plausible ... typo ...")
```

Rejects any breakpoint outside `[0.5, 2.5]`, naming the offender. Separate from
monotonicity (which cannot catch `6.0` typed for `1.06`). The message is long
and *actionable* — names the breakpoint, gives the canonical typo, says what to
change — the payoff of validating on the host. **Worked example:**
`set_qv_parameters(u1=0.90,u2=0.95,u3=1.05,u4=6.0)` passes monotonicity but
fails on `u4=6.0`; the three valid breakpoints are never committed.

---

#### Line 231–235: Parse-then-commit

```python
    # nothing is written until every check has passed
    Q_RATIO, U1_PU, U2_PU, U3_PU, U4_PU = vals
    return {"q_ratio": Q_RATIO, "u1": U1_PU, "u2": U2_PU,
            "u3": U3_PU, "u4": U4_PU}
```

*Only now*, after all four checks pass, are the five globals rebound in one
atomic assignment; the dict is returned so a CLI/wizard layer can echo what took
effect. **Atomicity guarantee:** the globals were untouched until this line, so
a failed check leaves the live curve exactly as before — never three new
breakpoints and one old one. Mirrors the firmware's parse-whole-line-then-commit
`CFG:` handling. A half-applied curve (new `u1`, old `u4`) could violate
monotonicity *between* old and new values and silently corrupt control; this
pattern makes that unreachable.

---

## 6. Section C — reset_qv_parameters() (L238–240)

```python
def reset_qv_parameters() -> dict:
    """Restore the compile-time default Q(V) parameters (see _QV_DEFAULTS)."""
    return set_qv_parameters(**_QV_DEFAULTS)
```

Splats the frozen `_QV_DEFAULTS` back through `set_qv_parameters()`, restoring
`q_ratio=0.25` and the canonical breakpoints, returning them. **Routing through
`set_qv_parameters` (not direct assignment)** means the restore uses the same
validation and the same atomic commit — a reset can never reach a state a normal
override could not. Matters in a long-lived interpreter (notebook, multi-run
script) after a parametric sweep with overrides; the curve-side analogue of
`violation_detector.reset_limits()` (Task 2 B.1).

**Where used.** No current call site in the tree — provided for
interactive/notebook/test reset between runs. (Flagged in §3.)

---

## 7. Section D — QVCharacteristic class (L282–408)

```python
# Lines 282-295
class QVCharacteristic:
    """
    VDE-AR-N 4110 Bild 8 Q(U) piecewise linear characteristic.
    Stateless static methods only. No hardware or network dependency.
    Mirrors compute_q() in volt_var_arduino.ino exactly ...
    """
```

**Design rationale — all-static class.** The characteristic is pure (same V in,
same Q out, no state). `@staticmethod` on a named class groups the methods under
one documenting namespace (`QVCharacteristic.compute_setpoint`) and signals "no
`self`, nothing to mutate." The key docstring clause is the **mirror
invariant**: this class and the Arduino `compute_q()` implement the same curve
*shape* independently; `CFG:` keeps the *values* in sync, but structural changes
must be made in both. Document 5 verifies the firmware side against this class.

**Where used.** Imported by `hosting_capacity.py` L110, `scenario_4_volt_var.py`
L86, `sensitivity_coordinator.py` L95.

---

### compute_setpoint — the five-segment law (L297–347)

The single most important function in the module.

```python
# Lines 297-298
    @staticmethod
    def compute_setpoint(vm_pu: float, p_installed_mw: float) -> float:
```

#### Line 297–314: Signature and parameter contract

**Parameters:** `vm_pu` (float, pu, e.g. 1.025, stressed-feeder range
`[0.90,1.10]`, non-finite → 0); `p_installed_mw` (float, MW, e.g. 0.5, `abs()`
taken, `0` → non-controllable → Q=0, non-finite → 0 + warn). **Returns** float
q_mvar (>0 inject, <0 absorb).

**Design rationale — scalar, vectorised separately.** Scalar per-DER logic is a
line-for-line match to the Arduino's per-DER `compute_q()` (inherently scalar in
C); `compute_setpoints` wraps *this*, so the curve logic lives in exactly one
place on the Python side.

**Where used.** Called only by `compute_setpoints` (L386); logic mirrored in the
firmware and in `compare_runs_updated.ipynb`.

---

#### Line 315–330: The two non-finite guards

```python
        if not np.isfinite(p_installed_mw):
            warnings.warn(...); return 0.0
        if not np.isfinite(vm_pu):
            warnings.warn(...); return 0.0
```

If either input is `NaN`/`inf`, return safe `Q=0` and warn. **Why Q=0 is safe:**
a DER that cannot compute a trustworthy setpoint should do *nothing* — inject and
absorb are both destabilising on garbage; zero is neutral. **Where a `NaN` V
comes from:** a non-converged `pp.runpp()` leaves `res_bus.vm_pu` as `NaN`;
Guard 2 stops that becoming a `NaN` Q written to `net.sgen.q_mvar` that would
poison the *next* power flow. `stacklevel=2` points the warning at the caller;
`!r` prints an unambiguous repr (`nan`, `inf`) in logs.

---

#### Line 331–332: Q_max — the magnitude of the whole curve

```python
        # abs incase somehow a value of p_installed_mw is -ve (due to faulty dataset)
        q_max = Q_RATIO * abs(p_installed_mw)
```

`Q_max = Q_RATIO · P_installed` — the height of the plateaus and the peak of
both ramps. `abs()` because capacity is a magnitude (a negative dataset value is
an error, not "negative capacity").

**Units check.** `MVAr·MW⁻¹ × MW = MVAr` ✓. **Worked example (DER-A):**
`0.25 · |0.5| = 0.125 MVAr`. **Design rationale:** `Q_RATIO` is read fresh (the
module global) on every call — so a `set_qv_parameters` override changes this
line's result without editing here.

---

#### Line 334–347: The piecewise-linear characteristic

```python
        # Inclusive boundary at saturation points avoids ... division-by-zero
        if vm_pu <= U1_PU:
            return q_max                                            # saturate inject
        elif vm_pu < U2_PU:
            return q_max * (U2_PU - vm_pu) / (U2_PU - U1_PU)     # ramp -> 0
        elif vm_pu <= U3_PU:
            return 0.0                                              # deadband
        elif vm_pu < U4_PU:
            return -q_max * (vm_pu - U3_PU) / (U4_PU - U3_PU)    # ramp -> absorb
        else:
            return -q_max                                           # saturate absorb
```

VDE-AR-N 4110 Bild 8 in code — five branches, one per segment.

**Derivation — the two ramps as two-point line forms.**

*Inject ramp (`U1<U<U2`)* through `(U1,+q_max)` and `(U2,0)`:

```
slope = (0 − q_max)/(U2 − U1) = −q_max/(U2 − U1)
q(U)  = q_max + slope·(U − U1) = q_max·(U2 − U)/(U2 − U1)     ← exactly L341
```

*Absorb ramp (`U3<U<U4`)* through `(U3,0)` and `(U4,−q_max)`:

```
slope = (−q_max − 0)/(U4 − U3) = −q_max/(U4 − U3)
q(U)  = 0 + slope·(U − U3) = −q_max·(U − U3)/(U4 − U3)        ← exactly L345
```

Both code lines are the simplified two-point forms — exact, not approximate.

**Continuity proof (why the `<=` vs `<` choice is safe).**

```
U=U1: q_max        vs  q_max·(U2−U1)/(U2−U1)=q_max        → equal ✓
U=U2: q_max·0/(…)=0 vs deadband 0                          → equal ✓
U=U3: deadband 0    vs −q_max·0/(…)=0                       → equal ✓
U=U4: −q_max·(U4−U3)/(U4−U3)=−q_max vs −q_max              → equal ✓
```

Continuous at all four breakpoints, so it does not matter which branch owns a
boundary. The code puts inclusive `<=` on the *saturation* side so an
exact-boundary voltage takes the constant branch and never touches a ramp's
division — purely defensive (denominators are already non-zero by validation),
costs nothing.

**Code-to-physics mapping.**

```
 V ≤ 0.96          → +q_max                DER injects hard, props a sagging bus
 0.96 < V < 0.99   → +q_max·(0.99−V)/0.03  injection tapering off
 0.99 ≤ V ≤ 1.01   → 0                     deadband — idle, voltage is fine
 1.01 < V < 1.04   → −q_max·(V−1.01)/0.03  absorption ramping up
 V ≥ 1.04          → −q_max                DER absorbs hard, pulls a high bus down
```

**Units check (ramp).** `(U2−V)/(U2−U1)` = pu/pu → dimensionless in `[0,1]`; so
`q_max [MVAr] × dimensionless = MVAr` ✓. Sign carried by the leading `+`/`−`.

**Worked numerical examples (DER-A, `q_max=0.125`, canonical breakpoints):**

```
V=0.950 (≤U1):    q = +0.125 MVAr                     full injection
V=0.975 (U1..U2): q = 0.125·(0.99−0.975)/0.03
                    = 0.125·0.5 = +0.0625 MVAr         half injection
V=1.000 (dead):   q = 0.000 MVAr                       idle
V=1.025 (U3..U4): q = −0.125·(1.025−1.01)/0.03
                    = −0.125·0.5 = −0.0625 MVAr         half absorption
V=1.050 (≥U4):    q = −0.125 MVAr                     full absorption
```

The ±0.015-pu symmetry (both |q|=0.0625) follows from equal ramp widths.

**Design rationale.** `elif` chaining → exactly one branch fires, short-circuits,
mirrors the firmware `if/else if` ladder. Early `return` per branch → each
segment is trivially checkable against Bild 8.

**Where used.** Only `compute_setpoints` (L386).

---

### compute_setpoints — vectorised (L349–389)

```python
# Lines 349-353
    @staticmethod
    def compute_setpoints(vm_pu: pd.Series, p_installed_mw: pd.Series) -> pd.Series:
```

#### Line 349–372: Signature and index-equality contract

**Parameters:** `vm_pu` (`pd.Series`, pu, indexed by sgen index);
`p_installed_mw` (`pd.Series`, MW, same index). **Returns** `pd.Series` q_mvar,
indexed by sgen index, dtype float, named `"q_mvar"`.

**Design rationale — index alignment is a hard error, not a silent reindex.**
The worst positional-vs-label hazard in the module: combined *positionally* with
different orders, DER *i*'s voltage pairs with DER *j*'s capacity — wrong Q to
wrong DER, uncatchable downstream (shapes match, numbers finite, PF converges).
So the function refuses to guess.

---

#### Line 373–383: The index-equality guard

```python
        if not vm_pu.index.equals(p_installed_mw.index):
            raise ValueError("... same index ... reindex(vm_pu.index) if ordering differs.")
```

`Index.equals` checks labels *and* order — stricter than "same set" on purpose;
pandas would otherwise misalign. Forcing an explicit `.reindex` makes the
alignment intentional and visible.

---

#### Line 384–389: Vectorisation and re-wrapping

```python
        q = np.vectorize(QVCharacteristic.compute_setpoint, otypes=[float])(
            vm_pu.values, p_installed_mw.values
        )
        return pd.Series(q, index=vm_pu.index, dtype=float, name="q_mvar")
```

**`np.vectorize`** is a loop wrapper (one place the scalar fn is called → the
scalar and vectorised paths cannot diverge). It is *not* a speed vectorisation —
it still calls Python per element; for ≤~100 DERs that is fine, and correctness
parity matters more. **`otypes=[float]`** pins the output dtype: without it,
NumPy infers from the first element, so a first DER landing exactly in the
deadband (`0.0`) could yield an integer-ish array. **Re-attaching `vm_pu.index`
and the name** keeps the result aligned by sgen label so downstream
(`net.sgen.loc[idx,"q_mvar"] = …`, Doc 1C) assigns each Q by label, not
position.

**Worked example.** `vm_pu={3:0.975,7:1.025}`, `P={3:0.5,7:0.5}` →
`{3:+0.0625, 7:−0.0625}` — DER 3 (low) injects, DER 7 (high) absorbs, each by
its own label.

**Where used.** `run_timestep` L1179 (dry-run); `hosting_capacity.py` L66/L435;
`sensitivity_coordinator.py` L1016/L1031 (`q_initial`); schema default in
`plugin_runner.py` L19.

---

### q_max_mvar (L391–394)

```python
    @staticmethod
    def q_max_mvar(p_installed_mw: float) -> float:
        """Maximum reactive power (MVAr) for the given installed capacity."""
        return Q_RATIO * abs(p_installed_mw)
```

Returns `Q_RATIO·|P|` — the same Q_max computed inside `compute_setpoint`
(L332), exposed standalone for callers wanting the ceiling without evaluating
the whole curve. Factoring it out means any consumer reads Q_max from the *same*
expression the characteristic uses, reading `Q_RATIO` at call time (single
source of truth). **Units/worked example:** `q_max_mvar(0.5)=0.125`,
`q_max_mvar(1.5)=0.375 MVAr`.

**Where used.** **No current call site.** Public helper. Not to be confused with
the `q_max_mvar` *parameter* in `der_dynamics.py`/`scenario_4_volt_var.py`,
which is derived from `p_installed_mw` and passed into `DERDynamics` — it does
not call this method. (Flagged in §3.)

---

### slope_inject (L396–402)

```python
    @staticmethod
    def slope_inject() -> float:
        """dQ/dV slope in the injection ramp [U1_PU, U2_PU]. Positive.
        Units: pu_Q per (pu_V * MW_installed).
        slope_inject() = 16.0 means each 1 MW of installed capacity can
        change Q by 16 MVAr per pu of voltage change in the ramp zone"""
        return Q_RATIO / (U2_PU - U1_PU)
```

#### Line 396–402: The injection-ramp gain

**Derivation.** In the inject ramp, `q = q_max·(U2−V)/(U2−U1)`, so
`dq/dV = −q_max/(U2−U1) = −Q_RATIO·P/(U2−U1)`. Normalised per MW of installed
capacity, the gain *magnitude* is `Q_RATIO/(U2−U1)` — what this method returns
(positive; the sign of the actual slope is negative, but the helper reports the
steepness).

**Code-to-physics mapping / units.** `Q_RATIO [MVAr·MW⁻¹] / (U2−U1) [pu]` →
**MVAr per pu-voltage per MW-installed**. The docstring's "pu_Q per (pu_V ·
MW_installed)" is the same thing expressed in the project's pu-Q convention.

**Units check + worked example.**

```
slope_inject() = Q_RATIO / (U2 − U1) = 0.25 / (0.99 − 0.96) = 0.25 / 0.03 = 8.33
```

For DER-A (0.5 MW) the *actual* ramp slope magnitude is `8.33 · 0.5 = 4.167
MVAr/pu`; across the 0.03-pu ramp the total Q swing is `4.167 · 0.03 = 0.125 =
q_max` ✓ — internally consistent with the plateau height.

**⚠ Flagged stale docstring.** The docstring says `slope_inject() = 16.0`. That
was true at the *old* `Q_RATIO = 0.48` (`0.48/0.03 = 16.0`); at the current
`Q_RATIO = 0.25` the value is **8.33**. The prose example is stale by the same
factor (0.48/0.25) as the value. **Flagged, not edited** — recommended fix: make
the example `Q_RATIO`-agnostic ("with Q_RATIO=0.25 this returns 8.33").

**Design rationale.** A single scalar summarising how aggressively the DER
reacts per volt in the ramp — useful for reporting/comparing tunings without
evaluating the curve. Reads `Q_RATIO`, `U1_PU`, `U2_PU` at call time.

**Where used.** **No current call site.** Diagnostic helper. (Flagged in §3.)

---

### slope_absorb (L404–408)

```python
    @staticmethod
    def slope_absorb() -> float:
        """dQ/dV slope in the absorption ramp [U3_PU, U4_PU]. Negative.
        Units: pu_Q per (pu_V * MW_installed)."""
        return -Q_RATIO / (U4_PU - U3_PU)
```

#### Line 404–408: The absorption-ramp gain

**Derivation.** In the absorb ramp, `q = −q_max·(V−U3)/(U4−U3)`, so
`dq/dV = −q_max/(U4−U3) = −Q_RATIO·P/(U4−U3)`. Per MW, the (signed) gain is
`−Q_RATIO/(U4−U3)` — returned here. **Negative** because Q *decreases* (moves
more negative) as voltage rises through the absorb ramp: rising voltage → more
absorption.

**Units check + worked example.**

```
slope_absorb() = −Q_RATIO / (U4 − U3) = −0.25 / (1.04 − 1.01) = −0.25 / 0.03 = −8.33
```

Equal in magnitude to `slope_inject` (8.33), opposite sign — a direct
consequence of the equal ramp widths (`U2−U1 = U4−U3 = 0.03`). The symmetry is
why the §8 worked examples give |q|=0.0625 at ±0.015 pu off either deadband
edge.

**Design rationale.** The absorb-side counterpart to `slope_inject`; the sign
convention (negative) makes it the true `dQ/dV` slope, whereas `slope_inject`
reports a magnitude — a minor asymmetry in the two helpers worth noting if they
are ever wired into reporting. Reads `Q_RATIO`, `U3_PU`, `U4_PU` at call time.

**Where used.** **No current call site.** Diagnostic helper. (Flagged in §3.)

---

## 8. The characteristic as one picture

For DER-A (q_max = 0.125 MVAr) with the canonical breakpoints:

```
  q_mvar
 +0.125 ┤●━━━━━━━●
        │        ╲                          ● = the five §8 worked-example points
 +0.062 ┤         ●
        │          ╲
  0.000 ┤           ●━━━━━━━━━●
        │           0.99    1.01 ╲
 −0.062 ┤                          ●
        │                           ╲
 −0.125 ┤                            ●━━━━━━━●
        └──┬─────┬─────┬─────┬─────┬─────┬──── vm_pu
          0.95  0.96  0.99  1.01  1.04  1.05
        inject  │◄ramp►│dead │◄ramp►│  absorb
        (sat)   slope_inject  slope_absorb (sat)
                 = 8.33/MW    = −8.33/MW
```

Left half raises voltage (inject, +Q); right half lowers it (absorb, −Q); the
middle does nothing. The DER only ever *reacts* to its local voltage — no
communication in this layer, which is exactly why it is the cheap, distributed
tier the coordinator (Docs 2–3) sits on top of. The two slope helpers report the
steepness of the two sloped segments.

---

## 9. Failure modes and customisation

| Condition | Where caught | Result |
|-----------|--------------|--------|
| `NaN`/`inf` `p_installed_mw` | L317 guard | Q = 0, warn |
| `NaN`/`inf` `vm_pu` (non-converged PF) | L325 guard | Q = 0, warn |
| Negative `p_installed_mw` in a dataset | L332 `abs()` | Magnitude used |
| Mismatched Series indices | L375 guard | `ValueError` before any Q computed |
| First-DER-in-deadband dtype inference | L386 `otypes=[float]` | Float array guaranteed |
| Non-finite / out-of-range / non-monotonic override | L195–229 | `ValueError`, curve unchanged |
| Half-applied override | L231–233 parse-then-commit | Unreachable state |

**Customising the curve — the supported path.** Change *shape* by editing
L133–137 (and mirroring the shape in the firmware); change *values* for a single
run via `set_qv_parameters(...)`. Never edit `QVCharacteristic` to hardcode a
different curve for one run — that breaks the firmware mirror and the
single-source-of-truth guarantee.

**What breaks it (for the next document).** This slice is pure and robust; the
fragility is at the *boundaries* — a `NaN` `vm_pu` from a silently non-converged
power flow (root cause in the loop, Doc 1C), or a positional index mismatch set
up upstream (alignment configured in the controller, Doc 1C). The serial path
(Doc 1B) introduces the *quantisation* the paper studies: the Q values here are
exact float64, but crossing the wire as 4-decimal ASCII rounds them — ~208 ×
`SATURATION_TOL` for a 0.5 MW DER. This function is where the exact value is
born; Doc 1B is where it is quantised.

---

## 10. Coverage checklist

Every named entity in L126–408, each confirmed documented. This is the
mechanical guard against skipping a small item.

| Entity | Lines | Kind | Documented | Where-used stated |
|--------|-------|------|:----------:|:-----------------:|
| `U1_PU` | 133 | constant | ✅ | ✅ |
| `U2_PU` | 134 | constant | ✅ | ✅ |
| `U3_PU` | 135 | constant | ✅ | ✅ |
| `U4_PU` | 136 | constant | ✅ | ✅ |
| `Q_RATIO` | 137 | constant | ✅ (stale comment flagged) | ✅ |
| `PLAUSIBLE_PU_RANGE` | 138 | constant | ✅ | ✅ |
| `_QV_DEFAULTS` | 142–143 | constant (dict) | ✅ | ✅ |
| `set_qv_parameters` | 146–235 | function | ✅ | ✅ |
| `reset_qv_parameters` | 238–240 | function | ✅ | ✅ (no caller) |
| `QVCharacteristic` | 282–408 | class | ✅ | ✅ |
| `QVCharacteristic.compute_setpoint` | 297–347 | staticmethod | ✅ | ✅ |
| `QVCharacteristic.compute_setpoints` | 349–389 | staticmethod | ✅ | ✅ |
| `QVCharacteristic.q_max_mvar` | 391–394 | staticmethod | ✅ | ✅ (no caller) |
| `QVCharacteristic.slope_inject` | 396–402 | staticmethod | ✅ (stale docstring flagged) | ✅ (no caller) |
| `QVCharacteristic.slope_absorb` | 404–408 | staticmethod | ✅ | ✅ (no caller) |

**Also covered (not standalone entities):** the two non-finite guards (L315–330),
the `global` declaration (L187), the parse-then-commit commit line (L231–233),
and the module docstring's protocol/architecture summary (referenced;
walked line-by-line in Doc 1B).

**Deliberately deferred to Document 1B (not a coverage gap).** Lines 243–275 sit
numerically inside this document's 126–408 span but belong to the *serial
transport* layer, not the characteristic: the serial-protocol constants
`BAUD_RATE`, `ARDUINO_MAX_DERS`, `DEFAULT_TIMEOUT_S`, `ACK_TIMEOUT_S`,
`DEFAULT_MAX_RETRIES`, `ARDUINO_RESET_DELAY_S` (L247–252) and the three custom
exceptions `SerialTimeoutError`, `ArduinoProtocolError`, `SerialConfigError`
(L259–275). They are documented in full in Document 1B, where their call sites
live. Noted here so the numeric gap is explicit and intentional.

---

*End of Document 1A (v2). Next: Document 1B — Serial Protocol & Arduino
Interface (`ArduinoSerialInterface`), where the exact Q values computed here are
framed, transmitted, and — critically for the paper — quantised to 4-decimal
ASCII.*
