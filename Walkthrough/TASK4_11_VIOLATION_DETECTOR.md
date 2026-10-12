# TASK 4 — DOCUMENT 11
## Violation Detector
### `violation_detector.py` — the single source of violation truth (1085 lines)
### Complete line-by-line walkthrough

> **Source verified:** read in full from `/mnt/project/violation_detector.py` (1085
> lines). **This is the foundational module** every scenario, the coordinator, the
> custom-controller path, and hosting-capacity analysis depend on — the `±ε`
> thresholding and the `V_MIN`/`V_MAX` constants referenced throughout Docs 1–12 and
> 6A–6E are defined here.
>
> **Read alongside:** every scenario runner (6A-iii–6E) and `run_coordinated_timestep`
> (3B) — they consume the `ViolationReport`; **Document 9** (`executor` —
> `apply_violation_limits` uses `set_limit`).

---

## TABLE OF CONTENTS

1. [Overview — the detector's role](#1-overview)
2. [File logic flow — pseudocode](#2-file-logic-flow--pseudocode)
3. [Call-site map](#3-call-site-map)
4. [Section A — Thresholds & epsilons (L96–129)](#4-section-a--thresholds--epsilons-l96129)
5. [Section B — set_limit & reset_limits (L109–163)](#5-section-b--set_limit--reset_limits-l109163)
6. [Section C — ViolationReport (L170–375)](#6-section-c--violationreport-l170375)
7. [Section D — ViolationReport3ph (L382–566)](#7-section-d--violationreport3ph-l382566)
8. [Section E — Balanced check helpers (L573–738)](#8-section-e--balanced-check-helpers-l573738)
9. [Section F — Three-phase check helpers (L745–908)](#9-section-f--three-phase-check-helpers-l745908)
10. [Section G — detect_violations & detect_violations_3ph (L915–1085)](#10-section-g--detect_violations--detect_violations_3ph-l9151085)
11. [Findings and flags](#11-findings-and-flags)
12. [Coverage checklist](#12-coverage-checklist)

---

## 1. Overview

`violation_detector.py` turns a solved pandapower network into a **structured
`ViolationReport`** — the object every downstream algorithm reads instead of re-querying
`net.res_*` directly. It is the **single source of violation truth**: every scenario's
`over_voltage_buses`/`overloaded_lines`/… lists, the coordinator's target-bus ordering, and
the hosting-capacity stopping condition all come from here, so they are guaranteed
consistent.

Four ideas define the module:

1. **The `±ε` tolerance.** Every threshold test adds a tiny epsilon (`VOLTAGE_EPSILON =
   1e-6` pu ≈ 0.02 V on 20 kV) so Newton–Raphson floating-point residuals in the HIL closed
   loop don't register as spurious violations. This is the `±ε` cited in every scenario doc.
2. **Late-bound defaults.** `detect_violations`' `None` arguments resolve to the *current*
   module constants **at call time** — so the CLI executor can rebind those constants
   process-wide (via `set_limit`) and have user limits apply everywhere, without threading
   five kwargs through every runner.
3. **Structured, not boolean.** The report carries per-element DataFrames (indexed by
   pandapower element index for direct joins), ~20 convenience properties (counts, worst
   severities, deviation-sorted bus indices), and `summary()`/`detail()` renderers.
4. **Balanced and three-phase.** `detect_violations` (runpp) and `detect_violations_3ph`
   (runpp_3ph — per-phase voltage, IEC 62749 unbalance, max-phase loading) are parallel
   paths. The five main scenarios use the balanced path; the 3ph path supports unbalanced LV
   analysis.

The thresholds encode standards: `V_MIN/V_MAX = 0.95/1.05` (EN 50160 / VDE-AR-N 4100/4105),
sitting **inside** the VDE-AR-N 4110 generator ride-through envelope (0.90–1.10) so the Q(V)
controller has room to act before generators risk disconnection.

---

## 2. File logic flow — pseudocode

```
CONSTANTS: V_MIN/V_MAX (0.95/1.05), LINE/TRAFO_MAX_LOADING (100%), VA_DIFF_MAX (30°),
           UNBALANCE_MAX (2%); epsilons (V 1e-6, loading 1e-6, angle 1e-4)
_LIMIT_DEFAULTS (import-time snapshot); reset_limits() restores them
set_limit(attr, value, lo, hi, unit, warn, confirm): validated process-wide setter

@dataclass ViolationReport:   5 DataFrames + any_violations/converged + limits-used
    properties: any_*_violations, n_*, worst_*, most_severe_*_buses; summary(); detail()
@dataclass ViolationReport3ph: per-phase + unbalance variant

balanced helpers:  _runpp_converged, _check_voltage, _check_lines, _check_trafos, _check_angle_diff
3ph helpers:       _runpp_3ph_converged, _check_voltage_3ph, _check_unbalance, _check_lines/trafos_3ph

detect_violations(net, v_min=None, …):        # late-bound defaults
    resolve None → current module constant; guard convergence; run the four checks; assemble report
detect_violations_3ph(net, …):                 # parallel, + unbalance
```

---

## 3. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `detect_violations` | function | **Every scenario runner** (6A-iii–6E via `make_record_from_report`/inline), `run_coordinated_timestep` (3B), `custom_controller` (12A), `hosting_capacity`. The sole balanced entry. |
| `detect_violations_3ph` | function | Unbalanced LV analysis paths. |
| `V_MIN`, `V_MAX`, `VOLTAGE_EPSILON`, `LOADING_EPSILON` | constants | Imported by nearly every module (the shared thresholds/tolerances). |
| `set_limit` | function | `executor.apply_violation_limits` (CLI wizard path) + `run_benchmark_script` (direct-script path) — shared validated setter. |
| `reset_limits` | function | Before a new run's overrides in a shared interpreter. |
| `ViolationReport`(`.any_violations`, `.most_severe_over_voltage_buses`, …) | dataclass | Consumed by all Tier-1 algorithms + the coordinator's target ordering (Doc 2/3). |

---

## 4. Section A — Thresholds & epsilons (L96–129)

```python
V_MIN = 0.95; V_MAX = 1.05; LINE_MAX_LOADING = 100.0; TRAFO_MAX_LOADING = 100.0
VA_DIFF_MAX_DEGREE = 30.0; UNBALANCE_MAX_PERCENT = 2.0; THERMAL_LOADING_PLAUSIBLE_MAX = 300.0
VOLTAGE_EPSILON = 1e-6; LOADING_EPSILON = 1e-6; ANGLE_EPSILON = 1e-4
```

#### Line 96–129: The planning limits and noise tolerances

The **standards-derived thresholds** (§1) — edited *here only*, so every scenario inherits
them. The **epsilons** are the crux of the module's robustness: `VOLTAGE_EPSILON = 1e-6`
(≈ 0.02 V on 20 kV, far below any real violation) is added to every voltage test so a bus at
exactly `V_MAX` plus NR round-off dust is **not** flagged — this is the `±ε` that keeps a
scenario's pre-derived violation lists identical to what the detector would compute, and
prevents spurious HIL-loop oscillation. `ANGLE_EPSILON = 1e-4` (degrees) and `LOADING_EPSILON
= 1e-6` (%) do the same for the angle and thermal checks. **`THERMAL_LOADING_PLAUSIBLE_MAX =
300.0`** is defined here but has no consumer in this module (likely read by a plotting/
diagnostic layer) — noted in §11.

---

## 5. Section B — set_limit & reset_limits (L109–163)

```python
_LIMIT_DEFAULTS = {"V_MIN": V_MIN, "V_MAX": V_MAX, "LINE_MAX_LOADING": ...}   # import-time snapshot

def reset_limits() -> None:
    for attr, val in _LIMIT_DEFAULTS.items(): globals()[attr] = val

def set_limit(attr, value, lo, hi, unit, warn=print, confirm=None) -> None:
    if value is None: return                              # no-op
    try: v = float(value); ok = np.isfinite(v) and lo < v <= hi
    except (TypeError, ValueError): ok = False
    if not ok: warn(f"Ignoring invalid {attr}={value!r} ...; keeping {current}."); return
    if v != current: globals()[attr] = v; confirm and confirm(f"{attr}: {current} -> {v} ...")
```

#### Line 109–163: The process-wide limit machinery

**The design problem:** `detect_violations`' late-bound defaults (§10) read the *module
constants*, so applying user limits means **rebinding those globals** — but a long-lived
interpreter (notebook, the CLI server) would then leak run A's overrides into run B.
`_LIMIT_DEFAULTS` snapshots the import-time values and **`reset_limits`** restores them, so
each run starts from a pristine baseline. **`set_limit`** is the *shared validated setter*
used by **both** entry points — `executor.apply_violation_limits` (CLI wizard) and
`run_benchmark_script` (direct-script) — so both apply identical range checks and
fallback-on-invalid behaviour instead of duplicating logic. Its contract: `None` is a no-op
(leave unchanged); an invalid value **warns and keeps** the current value (never raises); a
valid change optionally calls `confirm` (so the wizard prints a confirmation line while the
silent direct-script path passes `confirm=None`). **Design rationale — mutate-globals with
hygiene:** rebinding module constants is how limits reach `detect_violations` without kwarg
threading; `reset_limits` + validation make that safe. *(Minor: the `if v != current:` body
is over-indented (L160–163) — valid, cosmetic, §11.)*

---

## 6. Section C — ViolationReport (L170–375)

The balanced report — five DataFrames plus flags plus a rich property surface.

#### Line 214–224: Fields

`over_voltage`, `under_voltage`, `overloaded_lines`, `overloaded_trafos`, `angle_violations`
(all default-empty DataFrames), plus `any_violations`, `converged`, and `v_min_used`/
`v_max_used`. **Design rationale (docstring):** each DataFrame is **indexed by pandapower
element index**, so a caller joins it directly against `net.bus`/`net.line`/`net.trafo`
without index translation; and on non-convergence every frame is empty *with the correct
column schema*, so downstream `.empty`/`.max()` never special-case. The deviation columns are
pre-computed and **always positive** (`vm_pu − V_MAX` for over, `V_MIN − vm_pu` for under).

#### Line 230–375: Convenience properties, summary & detail

~20 properties: the `any_*_violations` category gates, `n_*` counts, `worst_*` severities
(returning `None` when empty, not `NaN`), and the two **`most_severe_*_voltage_buses`**
properties that return bus indices **sorted by deviation descending** — this is the ordering
the `SensitivityCoordinator` uses to target the worst buses first (Docs 2/3). `summary()`
renders a one-line human digest (or the non-convergence notice); `detail()` adds the full
per-category DataFrame dumps for debugging. **Design rationale:** the report is a *rich
object*, not a boolean — downstream code expresses intent (`report.any_violations`,
`report.most_severe_over_voltage_buses`) rather than re-deriving from raw results, which is
what keeps violation semantics consistent across the whole codebase.

---

## 7. Section D — ViolationReport3ph (L382–566)

The three-phase report — the analogue of §6 for `runpp_3ph()`, with per-phase voltage and an
unbalance category. Every entity is walked below.

#### Line 426–437: Fields

```python
    over_voltage/under_voltage : DataFrame  # per-phase, index=bus_idx
    unbalance_violations       : DataFrame  # IEC 62749
    overloaded_lines/trafos    : DataFrame  # max-across-phases loading
    any_violations : bool = False; converged : bool = True
    v_min_used/v_max_used/unbalance_max_used : float   # limits stamped for traceability
```

Five DataFrames plus the two flags plus **three** `*_used` limit stamps (the balanced report
has two; the 3ph adds `unbalance_max_used`). **Code-to-physics:** the voltage frames carry
`vm_a_pu`/`vm_b_pu`/`vm_c_pu` (the three phase-to-neutral magnitudes), `worst_phase` (the
label `"vm_a_pu"`/`"vm_b_pu"`/`"vm_c_pu"` of the offending phase), `worst_vm_pu`, and
`deviation_pu`; the thermal frames use the **maximum-across-phases** `loading_percent` (a line
is overloaded if *any* phase overloads). **Units:** voltages pu, unbalance %, loading %,
deviations in the same unit as their quantity. **Design rationale (docstring `Notes`):**
`runpp_3ph()` needs **fully-parameterised zero-sequence data** absent from the standard type
libraries — so this report exists for genuinely-unbalanced LV studies that supply those
parameters, not the five balanced benchmark scenarios (a fact the module docstring and the KB
both record).

#### Line 443–503: Convenience properties

```python
    any_voltage_violations   = not over_voltage.empty or not under_voltage.empty
    any_unbalance_violations = not unbalance_violations.empty
    any_thermal_violations   = not overloaded_lines.empty or not overloaded_trafos.empty
    n_over_voltage/n_under_voltage/n_unbalance_violations/n_overloaded_lines/n_overloaded_trafos = len(...)
    worst_over_voltage  = over_voltage["worst_vm_pu"].max()  or None
    worst_under_voltage = under_voltage["worst_vm_pu"].min() or None
    worst_unbalance     = unbalance_violations["unbalance_percent"].max() or None
    worst_line_loading/worst_trafo_loading = overloaded_*["loading_percent"].max() or None
```

The same property surface as the balanced report **plus the unbalance category**
(`any_unbalance_violations`, `n_unbalance_violations`, `worst_unbalance`). Each `worst_*`
returns `None` (not `NaN`) on an empty frame, so a caller can test `is None` cleanly. Note
`worst_over_voltage` reads the **`worst_vm_pu`** column (already the max-across-phases per
bus), so the report-level worst is a max of per-bus maxima — the single most-over-voltage
phase anywhere. **Worked example:** a bus with `vm = [1.02, 1.06, 1.03]` and `V_MAX = 1.05`
appears in `over_voltage` with `worst_phase = "vm_b_pu"`, `worst_vm_pu = 1.06`,
`deviation_pu = 0.01`; `worst_over_voltage` across the report returns 1.06 if that is the
worst bus.

#### Line 505–566: summary() and detail()

`summary()` returns the non-convergence notice, else a "no violations" line quoting the three
limit bands (`V`, thermal, **unbalance**), else a `|`-joined digest with the worst value per
non-empty category (voltages to 4 dp, unbalance to 2 dp, loading to 1 dp). `detail()` prepends
`summary()` then dumps each non-empty frame with `to_string()` (including the per-phase columns
and `worst_phase`). **Design rationale:** identical consumption ergonomics to the balanced
report so 3-phase analysis code reads the same way; the per-phase detail is purely additive.

---

## 8. Section E — Balanced check helpers (L573–738)

Five internal helpers, each returning a **schema-correct** DataFrame (correct columns even when
empty). Each is walked with a worked example.

#### Line 573–587: _runpp_converged

```python
    flag_ok    = bool(getattr(net, "converged", False))
    results_ok = hasattr(net, "res_bus") and not net.res_bus.empty and "vm_pu" in net.res_bus.columns
    return flag_ok and results_ok
```

The **two-guard** convergence test. **Derivation of the second guard (docstring):** pandapower
can, in edge cases, leave `net.converged = True` while `res_bus` is empty or lacks `vm_pu`
(e.g. a solve that technically "converged" on a degenerate net) — reading voltages then
`KeyError`s or silently returns nothing. Requiring a **populated `res_bus` with a `vm_pu`
column** as well makes "converged" mean "converged *and* usable". **Worked example:** after a
diverged solve, `net.converged = False` → returns `False` immediately → `detect_violations`
short-circuits to a `converged=False` report; after a good solve, both guards pass.

#### Line 590–623: _check_voltage

```python
    vm = net.res_bus["vm_pu"] restricted to in-service buses, dropna()
    over  = vm[vm > v_max + VOLTAGE_EPSILON];  over["deviation_pu"]  = over  - v_max
    under = vm[vm < v_min - VOLTAGE_EPSILON];   under["deviation_pu"] = v_min - under
    return over_df, under_df   # index=bus_idx, columns=[vm_pu, deviation_pu]
```

Splits in-service bus voltages into over- and under-voltage frames using the **`±VOLTAGE_EPSILON`
tolerance** and a **positive** `deviation_pu` (over: `vm − v_max`; under: `v_min − vm`).
**Units:** pu. **Worked example** (`V_MAX = 1.05`, `ε = 1e-6`): a bus at `1.0500005 pu` is
**not** flagged (`1.0500005 < 1.05 + 1e-6 = 1.050001`) — NR round-off suppressed; a bus at
`1.06 pu` is flagged with `deviation_pu = 0.01`. This is the `±ε` every scenario's
over/under-voltage lists inherit.

#### Line 626–697: _check_lines & _check_trafos

```python
    if res_line missing / empty / no "loading_percent": warnings.warn(RuntimeWarning); return empty-schema
    loading = res_line["loading_percent"] restricted to in-service, dropna()
    mask = loading > (max_loading + LOADING_EPSILON)
    return DataFrame({loading_percent, excess_percent = loading - max_loading})
```

In-service line/trafo loading above `max_loading + LOADING_EPSILON`, with `excess_percent`.
**The distinguishing choice — warn, don't hide:** if `res_line`/`res_trafo` is missing or
lacks `loading_percent`, the helper emits a **`RuntimeWarning`** (not a silent empty frame),
because a missing thermal result is a *broken solve* the user should see, not "no violations".
**Worked example** (`LINE_MAX_LOADING = 100`): a line at `156.2 %` (the SVC case, 6C) →
`loading_percent = 156.2`, `excess_percent = 56.2`; a line at `100.0000005 %` → not flagged
(ε). `stacklevel=3` points the warning at the caller of `detect_violations`, not this
internal helper.

#### Line 700–738: _check_angle_diff

```python
    if "va_degree" columns absent on res_line: return empty-schema   # SILENT (no warning)
    d = |va_from_degree − va_to_degree|
    mask = d > (VA_DIFF_MAX_DEGREE + ANGLE_EPSILON)
    return DataFrame({angle_diff_degree, excess_degree})
```

Flags lines whose end-to-end voltage-angle difference exceeds `VA_DIFF_MAX_DEGREE` (30°) — a
stability/loadability proxy. **The deliberate asymmetry vs the thermal checks:** absent angle
columns return an **empty frame silently, with no warning**, because voltage angles are
*optional* — `calculate_voltage_angles` is automatic for MV but not guaranteed on every network,
so a missing angle result is a normal condition, not a broken solve. **Worked example**
(`VA_DIFF_MAX = 30°`, `ANGLE_EPSILON = 1e-4`): a line with `va_from = 2.0°`, `va_to = −29.5°`
→ `d = 31.5° > 30.0001°` → flagged, `excess_degree = 1.5`.

**Common pattern (all five):** filter to **in-service** elements and `.dropna()` before
thresholding, so out-of-service or unsolved elements never generate phantom violations, and
return the correct empty schema so downstream `.empty`/`.max()` never special-case.

---

## 9. Section F — Three-phase check helpers (L745–908)

The `runpp_3ph()` analogues. Each gets its own block; the logic parallels §8 but on per-phase
results, with the worst-phase identification being the substantive addition.

#### Line 745–756: _runpp_3ph_converged

```python
    flag_ok    = bool(getattr(net, "converged", False))
    results_ok = hasattr(net, "res_bus_3ph") and not net.res_bus_3ph.empty and "vm_a_pu" in net.res_bus_3ph.columns
    return flag_ok and results_ok
```

The two-guard test with **`res_bus_3ph`** and the **`vm_a_pu`** column as the result guard
(the 3ph analogue of `res_bus`/`vm_pu`). Same rationale as §8's `_runpp_converged`: a
`converged` flag alone is insufficient; usable per-phase results must exist.

#### Line 759–809: _check_voltage_3ph

```python
    df = res_bus_3ph[["vm_a_pu","vm_b_pu","vm_c_pu"]] restricted to in-service, dropna()
    over_mask  = (df > v_max + VOLTAGE_EPSILON).any(axis=1)     # ANY phase over
    over["worst_phase"]  = df.idxmax(axis=1); over["worst_vm_pu"] = df.max(axis=1); dev = worst − v_max
    under_mask = (df < v_min - VOLTAGE_EPSILON).any(axis=1)     # ANY phase under
    under["worst_phase"] = df.idxmin(axis=1); under["worst_vm_pu"] = df.min(axis=1); dev = v_min − worst
```

**The substantive 3ph logic:** a bus is flagged if **any single phase** violates
(`.any(axis=1)` across the three phase columns), and the **worst phase is identified** via
`idxmax`/`idxmin` (over-voltage → the highest phase; under-voltage → the lowest), with
`worst_vm_pu` and a positive `deviation_pu`. The `±VOLTAGE_EPSILON` tolerance is the same as
balanced. **Worked example** (`V_MAX = 1.05`): a bus with `vm = [1.04, 1.062, 1.03]` →
`over_mask = True` (phase B), `worst_phase = "vm_b_pu"`, `worst_vm_pu = 1.062`,
`deviation_pu = 0.012`; a balanced bus at `[1.02, 1.02, 1.02]` is not flagged. **Design
rationale:** flagging on the worst phase is the correct planning-limit semantics — a limit
breach on *one* phase is a breach.

#### Line 812–834: _check_unbalance

```python
    ub = res_bus_3ph["unbalance_percent"] restricted to in-service, dropna()
    mask = ub > (unbalance_max + LOADING_EPSILON)
    return DataFrame({unbalance_percent, deviation_percent = ub − unbalance_max})
```

Flags buses whose **voltage unbalance factor** (pandapower's `unbalance_percent`, per **IEC
62749** — the ratio of negative- to positive-sequence voltage) exceeds `UNBALANCE_MAX_PERCENT`
(2 %). **Units:** %. **Worked example** (`UNBALANCE_MAX = 2.0`): a bus at `2.7 %` →
`unbalance_percent = 2.7`, `deviation_percent = 0.7`. This is the category with **no balanced
analogue** — it exists only in the 3ph path (and, per Doc 9B, currently has no caller in the
benchmark stack, since no runner performs a 3-phase flow).

#### Line 837–871: _check_lines_3ph

```python
    if res_line_3ph missing/empty: warnings.warn(RuntimeWarning); return empty-schema
    if "loading_percent" missing:  warnings.warn(RuntimeWarning); return empty-schema
    loading = res_line_3ph["loading_percent"] restricted to in-service, dropna()   # max across phases
    mask = loading > (max_loading + LOADING_EPSILON); return {loading_percent, excess_percent}
```

Lines whose **maximum-across-phases** loading exceeds `max_loading` — `res_line_3ph.loading_percent`
is already the per-line max over the three phases, so an unbalanced overload on a single phase
is caught. Same **warn-don't-hide** behaviour as the balanced `_check_lines` (two `RuntimeWarning`
guards: empty table, missing column). **Worked example** (`LINE_MAX_LOADING = 100`): a line
loaded `[70, 108, 65] %` per phase → `loading_percent = 108` → flagged, `excess_percent = 8`.

#### Line 874–908: _check_trafos_3ph

The transformer analogue of `_check_lines_3ph` — identical structure on `res_trafo_3ph.loading_percent`
(max across phases), the same two `RuntimeWarning` guards, in-service filtering, and
`excess_percent`. **Worked example** (`TRAFO_MAX_LOADING = 100`): a trafo at `[95, 101, 90] %`
→ `loading_percent = 101` → flagged, `excess_percent = 1`.

**Common pattern (all five, as §8):** in-service filtering, `.dropna()`, `±ε` tolerance,
schema-correct empties, and warn-on-missing for the thermal tables (the quantities that
indicate a broken solve if absent).

---

## 10. Section G — detect_violations & detect_violations_3ph (L915–1085)

```python
def detect_violations(net, v_min=None, v_max=None, line_max=None, trafo_max=None, va_diff_max=None):
    if v_min is None: v_min = V_MIN                    # LATE-BOUND: current module constant, at call time
    ... (same for the others) ...
    if not _runpp_converged(net): return ViolationReport(converged=False)
    over_v, under_v = _check_voltage(net, v_min, v_max); overloaded_lines = _check_lines(...); ...
    any_violations = not over_v.empty or not under_v.empty or ...
    return ViolationReport(over_voltage=over_v, ..., any_violations=any_violations, converged=True, v_min_used=v_min, ...)
```

#### Line 915–1085: The late-bound public entries

The two public functions run the checks and assemble the report. **The late-bound-defaults
mechanism (L976–982) is the key design point:** a `None` argument resolves to the *current*
module constant **at call time**, so `executor.apply_violation_limits` can `set_limit` the
globals **before** a run and every internal `detect_violations()` call (in every runner)
picks up the user's limits — *without* threading `v_min`/`v_max`/… through the whole call
tree. **Explicit arguments always win** over the constants (so a caller can still override
per-call). Non-convergence short-circuits to a `converged=False` report with empty
schema-correct frames. `v_min_used`/`v_max_used` are stored on the report for traceability in
result logs. `detect_violations_3ph` is the parallel path with `unbalance_max` in place of
`va_diff_max`. **Design rationale:** one process-wide knob (the module constants) + late
binding = user limits everywhere, cleanly.

*Scope note (from the project's parametric-limits finding):* the **CLI wizard** path fully
uses this — `executor.apply_violation_limits` rebinds all limit constants pre-run. The
**direct-script + YAML** path only wires `voltage_limits` and even those reach only
`BenchmarkConfig` cosmetic highlighting, **not** `detect_violations` — so on the direct-script
path thermal/loading overrides silently do nothing. That asymmetry is a property of the
*callers* (Doc 9), not of this module, which is correct.

---

## 11. Findings and flags

**No functional bugs.** The module is clean, foundational, and defensively written. Minor
cosmetic/cleanliness notes only:

1. **Over-indented `set_limit` body (L160–163).** The `if v != current:` block is indented
   12 spaces where 8 is conventional — valid Python, cosmetic (the same over-indent style
   seen in scenarios 2/3/5).
2. **Duplicate `import warnings`** (L77 and L84) and a **redundant local `import numpy as np`**
   inside `set_limit` (L147, already imported at L81) — harmless, tidy-up candidates.
3. **`THERMAL_LOADING_PLAUSIBLE_MAX` (L102) has no consumer in this module** — likely read by
   a plotting/diagnostic layer (Doc 10). If genuinely unused anywhere, it is dead; if used
   externally, a comment pointing there would help.

**Not bugs (documented for clarity):** the process-wide global mutation is intentional (the
late-binding mechanism) and made safe by `reset_limits`; the angle check's silent-empty on
missing columns (vs the thermal checks' warnings) is a deliberate softer failure for an
optional quantity; the two-guard convergence check is necessary given pandapower's edge-case
`converged=True`-with-empty-results.

---

## 12. Coverage checklist

| Entity | Lines | Kind | Documented | Where-used stated |
|--------|-------|------|:----------:|:-----------------:|
| thresholds + epsilons | 96–129 | constants | ✅ | ✅ |
| `_LIMIT_DEFAULTS` / `reset_limits` | 109–121 | state/fn | ✅ | ✅ |
| `set_limit` | 131–163 | function | ✅ | ✅ |
| `ViolationReport` (fields + ~20 properties + summary/detail) | 170–375 | dataclass | ✅ | ✅ |
| `ViolationReport3ph` (fields + properties + `summary`/`detail`) | 382–566 | dataclass | ✅ (full block, §7) | ✅ |
| `_runpp_converged` | 573–587 | helper | ✅ | ✅ |
| `_check_voltage` | 590–623 | helper | ✅ | ✅ |
| `_check_lines` / `_check_trafos` | 626–697 | helpers | ✅ | ✅ |
| `_check_angle_diff` | 700–738 | helper | ✅ | ✅ |
| `_runpp_3ph_converged` | 745–756 | helper | ✅ | ✅ |
| `_check_voltage_3ph` | 759–809 | helper | ✅ (full block, §9) | worst-phase logic |
| `_check_unbalance` | 812–834 | helper | ✅ (full block, §9) | IEC 62749 |
| `_check_lines_3ph` | 837–871 | helper | ✅ (full block, §9) | max-phase loading |
| `_check_trafos_3ph` | 874–908 | helper | ✅ (full block, §9) | max-phase loading |
| `detect_violations` | 915–1009 | function | ✅ | ✅ |
| `detect_violations_3ph` | 1012–1085 | function | ✅ | ✅ |

**Also covered (not standalone entities):** the `±ε` single-source-of-truth, the standards
mapping (EN 50160 / VDE-AR-N inside the ride-through envelope), the late-bound-defaults
mechanism and its process-wide-limit rationale, the two-guard convergence check, the
element-index-joinable schema-correct frames, and the deviation-sorted bus ordering feeding
the coordinator.

**Flags raised (not edited):** duplicate `import warnings`; redundant local `np` import;
over-indented `set_limit` body; possibly-unused `THERMAL_LOADING_PLAUSIBLE_MAX`. No functional
bugs.

---

*End of Document 11. Remaining Task 4 queue: Document 9 (orchestration —
`benchmark_runner`/`executor`/`wizard`/`run_plan`/`run_benchmark_script`/`helpers`/`__main__`)
and Document 10 (output — `publisher`/`plot_results`/`network_plotter`).*
