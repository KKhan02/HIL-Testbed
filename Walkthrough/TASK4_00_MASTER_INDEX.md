# TASK 4 — HIL Testbed Line-by-Line Code Walkthroughs
## MASTER INDEX

> **This is the navigation hub for the Task 4 walkthrough set.** It is *not*
> itself a walkthrough. It defines the document split, the reading order, the
> single canonical set of worked-example inputs, the six-part block format that
> every walkthrough obeys, the two mandatory per-document sections (file
> pseudocode and call-site map), and the cross-reference map back to the
> flowcharts (Task 1a), the module documentation (Task 1b), and the audit
> (Task 2).
>
> Read this once, then read the walkthroughs. Every walkthrough assumes you
> have the *Conventions* and *Canonical worked-example inputs* sections below
> in mind, so it does not repeat them.

---

## TABLE OF CONTENTS

1. [What this set is](#1-what-this-set-is)
2. [How this relates to Tasks 1a, 1b and 2](#2-how-this-relates-to-tasks-1a-1b-and-2)
3. [Document set and reading order](#3-document-set-and-reading-order)
4. [Flagged deviations from the original brief](#4-flagged-deviations-from-the-original-brief)
5. [Mandatory per-document sections](#5-mandatory-per-document-sections)
6. [How to read a walkthrough block — the six-part format](#6-how-to-read-a-walkthrough-block--the-six-part-format)
7. [Conventions](#7-conventions)
8. [Canonical worked-example inputs](#8-canonical-worked-example-inputs)
9. [Cross-reference map](#9-cross-reference-map)
10. [Non-negotiable working rules](#10-non-negotiable-working-rules)
11. [Status tracker](#11-status-tracker)
12. [Change log](#12-change-log)

---

## 1. What this set is

Exhaustive, line-by-line code-walkthrough documentation for the HIL testbed,
matching the depth of the reference documents supplied from the previous
(solar-thermal) project. The reference standard is
`20_LEVEL2_PART2_CODE_WALKTHROUGH.md`: **43 line-level explanation blocks
across 3,349 lines**, roughly **75 lines of explanation per code line-group**.
A walkthrough that summarises a function in a paragraph has failed this task.

**Audience.** The reader who must *defend* the code — you at a supervisor viva
(Andreas, Adrian), a co-author reading before extending it, or a future SuRE
student inheriting the testbed. This is the deepest layer of the documentation
stack; it assumes the reader already has the overview from Task 1b and the
structural picture from the Task 1a flowcharts.

**Scope.** The full system, module by module. This is broader than the paper
(`paper.tex`), which deliberately reports only the Scenario 4 serial-protocol
ASCII-quantisation finding. That breadth belongs here and in the thesis, not in
the paper.

**Completeness rule (elevated after the 1A review).** *Every* function, method,
class, property, constant and helper is documented — no matter how small. The
omission of `slope_inject`/`slope_absorb` from the first 1A draft is the
cautionary precedent: "too small to bother" is not a permitted judgement. A
one-line helper gets a short honest block; it does not get skipped. Each
document ends with a **coverage checklist** enumerating every named entity in
its source range and confirming each is documented.

**Post-fix state.** All 18 Task 2 audit findings are fixed in the live source.
Every walkthrough documents the **post-fix** code. Where a stale docstring or
comment still survives, the walkthrough flags it explicitly rather than
silently documenting around it.

---

## 2. How this relates to Tasks 1a, 1b and 2

| Task | Deliverable | Layer | This set's relationship |
|------|-------------|-------|-------------------------|
| **1a** | 26 `flow_*.svg/pdf/py` swimlane diagrams | *Structure* — how modules connect | Walkthroughs cite the relevant flow diagram at the top of each document |
| **1b** | `TASK1B_MODULE_DOCUMENTATION.md` (2,361 lines, Parts I–VI) | *Overview* — what each module does, with headline equations | Walkthroughs go **one level deeper**: every line, every guard, every constant. Task 1b is a summary and must be re-verified against source, not trusted |
| **2** | `TASK2_AUDIT_REPORT.md` + `TASK2_PROPOSED_FIXES.md` | *Correctness* — 18 findings, all fixed | Walkthroughs document the post-fix behaviour and reference the finding ID (e.g. B.1, F.1) where a line exists *because* of an audit fix |

The layers stack: **1a tells you where you are, 1b tells you what it does, Task 4
tells you exactly how and why every line does it.**

---

## 3. Document set and reading order

Produce in this order. Each document is completed and signed off before the
next begins (per the brief's "produce one document completely before moving
on" rule and the user's "do not make the entire doc set in one pass"
instruction).

| # | Document | Primary source | Est. size | Status |
|---|----------|----------------|-----------|--------|
| **00** | **Master Index** (this file) | — | — | ✅ done |
| **1A** | Q(U) Characteristic Core | `volt_var_controller.py` **L126–408** | large | ✅ done (v2) |
| **1B** | Serial Protocol & Arduino Interface | `volt_var_controller.py` L243–275 + `ArduinoSerialInterface` L494–895 | very large | ✅ done |
| **1C** | VoltVarController HIL Loop | `volt_var_controller.py` `VoltVarResult` L411–487 + `VoltVarController` L897–1229 | large | ✅ done |
| **2** | Sensitivity Coordinator — Theory | Jacobian, Schur complement, `lstsq`, rank deficiency | large | ✅ done |
| **3A** | Coordinator — Sensitivity Matrix | `sensitivity_coordinator.py` L84–686 (`CoordinatorResult`, `__init__`, `_build_jacobian_blocks`, `_bus_lookup_maps`, `_get_vm_pu_ppc`, `_compute_der_sensitivity`) | very large | ✅ done |
| **3B** | Coordinator — Algorithm & Orchestration | `sensitivity_coordinator.py` `coordinate()` L692–871 + `run_coordinated_timestep()` L878–1106 | very large | ✅ done |
| **4** | DER Dynamics | `der_dynamics.py` (PT1, ramp) | medium | ✅ done |
| **5** | Arduino Firmware | `volt_var_arduino.ino` (C — same standard) | medium | ✅ done |
| **6A-i** | Scenario Result — Profile Adapter | `scenario_result.py` `AdaptedProfiles`+`adapt_profiles`+`oversize_inverters`+`slice_profiles` (L64–524) | large | ✅ done |
| **6A-ii** | Scenario Result — Record & Result | `scenario_result.py` `TimestepRecord` L530–790 + `ScenarioResult` L791–1226 | very large | ✅ done |
| **6A-iii** | Scenario Runner — Baseline | `scenario_1_baseline.py` | medium | ✅ done |
| **6B-i** | Scenario Runner — OLTC Setup | `scenario_2_oltc.py` L86–590 (metadata, selection, validation, tap-sign calibration) | large | ✅ done |
| **6B-ii** | Scenario Runner — OLTC Loop | `scenario_2_oltc.py` `run_scenario_2` L597–995 (checkpointing) | large | ✅ done |
| **6C** | Scenario Runners — SVC | `scenario_3_svc.py` (checkpointing) | medium | ✅ done |
| **6D** | Scenario Runners — Volt-Var (HIL) | `scenario_4_volt_var.py` (checkpointing + DER-state seeding) | large | ✅ done |
| **6E** | Scenario Runners — OPF | `scenario_5_opf.py` (checkpointing) | medium | ✅ done |
| **7A** | Profile Builder — PV physics | `profile_builder.py` (Erbs→POA→NOCT→DC/AC) | large | ⬜ |
| **7B** | Profile Builder — Wind | `profile_builder.py` (piecewise-cubic power curve) | medium | ⬜ |
| **7C** | Profile Builder — Load | `profile_builder.py` (BDEW 2025 SLPs) | medium | ⬜ |
| **8** | Hosting Capacity | `hosting_capacity.py` | medium | ⬜ |
| **9A** | Orchestration — Benchmark Runner | `benchmark_runner.py` | large | ✅ done |
| **9B** | Orchestration — Executor | `executor.py` | large | ✅ done |
| **9C** | Orchestration — Wizard & Run Plan | `wizard.py`, `run_plan.py` | large | ✅ done |
| **9D** | Orchestration — Script Entry & Helpers | `run_benchmark_script.py`, `helpers.py`, `__main__.py` | large | ✅ done |
| **9E** | Orchestration — CLI Support | `network_catalogue.py`, `_data.py`, `_console.py`, `test_helpers.py` | medium | ✅ done |
| **10A** | Output — Publisher | `publisher.py` | large | ✅ done |
| **10B** | Output — Report Figures | `plot_results.py` | large | ✅ done |
| **10C** | Output — Network Plots | `network_plotter.py` | large | ✅ done |
| **11** | Violation Detector | `violation_detector.py` | medium | ✅ done |
| **12A** | Custom Controller & Plugin — Controller Loop | `custom_controller.py` | medium | ✅ done |
| **12B** | Custom Controller & Plugin — Plugin Runner | `plugin_runner.py` | large | ✅ done |
| **12C** | Custom Controller & Plugin — Network Plugin | `network_plugin.py` | large | ✅ done |
| **12D** | Custom Controller & Plugin — Loaders & Examples | `my_network_loader.py`, `network_export.py`, `droop_controller.py`, `deadband_controller.py`, example networks + YAMLs | medium | ✅ done |

**Recommended reading order for a new reader** differs from the production
order: read **5 (firmware)** immediately after **1A**, because the Arduino
`compute_q()` is the exact mirror of the `QVCharacteristic` and the two only
make sense side by side. Then **1B → 1C** for the loop that drives them, then
**2 → 3** for the coordinator, then the rest in numeric order.

---

## 4. Flagged deviations from the original brief

The brief asked that any deviation from its proposed 10-document split be
flagged rather than made silently.

1. **Document 1 splits into 1A / 1B / 1C.** `volt_var_controller.py` is 1,229
   lines; at the required density a single walkthrough would exceed 5,000
   lines. Split by responsibility: pure Q(U) characteristic (1A), serial
   transport (1B), HIL orchestration loop (1C). **[confirmed]**
2. **`violation_detector.py` added as Document 11.** Not in the brief's
   10-document list, but `VoltVarController.run_timestep` and every scenario
   runner depend on `detect_violations`; it also carries the Task 2 finding B.1
   fix (`reset_limits()`). **[confirmed in scope]**
3. **Document 6 splits into 6A–6E** (result container + baseline, OLTC, SVC,
   Volt-Var HIL, OPF) and **Document 7 splits into 7A–7C** (PV, wind, load).
   **[confirmed]** 6D (Volt-Var) and 7A (PV) are the largest sub-parts and may
   themselves need further splitting; that will be flagged when reached.
4. **Document 9 may sub-split.** It now names all eight orchestration/CLI
   modules explicitly; if the combined walkthrough runs too long it will split
   along the wizard / executor / runner boundaries. Flagged as a possibility.
5. **Document 3 splits into 3A / 3B.** `sensitivity_coordinator.py` is 1,106
   lines; at the required density a single walkthrough would far exceed one
   document. Split by responsibility: **3A** builds the sensitivity matrix `X`
   (constants, `CoordinatorResult`, `__init__`, the Jacobian/bus-lookup/
   sensitivity machinery — Doc 2 §4–§6); **3B** uses it (`coordinate()`'s
   11-step algorithm and `run_coordinated_timestep()` — Doc 2 §7–§12).
   **[confirmed]**
6. **Document 6A splits into 6A-i / 6A-ii / 6A-iii.** `scenario_result.py` alone
   is **1,226 lines** of shared infrastructure (not the "medium" originally
   sized). Split: **6A-i** the profile adapter (`AdaptedProfiles`,
   `adapt_profiles`, `oversize_inverters`, `slice_profiles`); **6A-ii** the
   result types (`TimestepRecord`, `ScenarioResult`); **6A-iii**
   `scenario_1_baseline.py` (the baseline runner). **[confirmed]**
7. **Document 6B splits into 6B-i / 6B-ii.** `scenario_2_oltc.py` is **956
   lines**. Split: **6B-i** the OLTC setup infrastructure (tap-metadata
   completion/override, topology selection, ganging validation, tap-sign
   calibration); **6B-ii** `run_scenario_2` (the 7-step setup + per-timestep tap
   control loop). **[confirmed]**
8. **Document 12 added — Custom Controller & Plugin Subsystem.** `custom_controller.py`
   (the 499-line generic user-controller HIL loop) had no walkthrough home.
   **Confirmed as Document 12**, scoped (per user) to the whole extension subsystem:
   `custom_controller.py`, `plugin_runner.py` (moved out of Doc 9),
   `network_plugin.py`, `my_network_loader.py`, `network_export.py`, the example
   controllers (`droop_controller.py`, `deadband_controller.py`), and the example
   networks + their YAMLs. **Blocked pending the `custom_controller.py` SyntaxError
   fix + re-upload** (line numbers will shift on fix). **[confirmed]**

---

## 5. Mandatory per-document sections

Every walkthrough document contains, in addition to the line-by-line blocks:

**(a) File logic flow — human-readable pseudocode.** Near the top of each
document, a plain-language / structured-pseudocode description of the file's
logic flow and sequence of steps. It conveys *what happens in what order and
why*, **without restating the Python** — no Python syntax, no literal code
lines. For a split file (e.g. `volt_var_controller.py` across 1A/1B/1C) each
part carries the pseudocode for its own slice, plus a one-paragraph note on how
the parts connect. The goal: a reader can follow the algorithm from the
pseudocode alone, then drop into the line-by-line blocks for the detail.

**(b) Call-site map — "where used".** Every function, method, class, property
and constant documented in the file gets an explicit statement of **where it is
called or read from in the whole system** — file and line. This is provided
both as a consolidated table near the top and inline in each entity's block. If
an entity has **no caller in the tree**, that is stated plainly ("no current
call site; provided as a public/diagnostic helper") rather than left implied —
an uncalled function is a finding, not an omission. **For every uncalled public
entity, the document gives a concrete verdict** — either a *wire-in target*
(the diagnostics/reporting site where it should be called) or an *intentional
marker* (why it is legitimately uncalled) — rather than merely noting the
absence. These verdicts accumulate in the register below.

**Uncalled public API register** (running list across all documents):

| Entity | Doc | Verdict | Target / reason |
|--------|-----|---------|-----------------|
| `reset_qv_parameters` | 1A | **Intentional** | Every run already calls `set_qv_parameters`, so an executor-path reset is redundant. Value is interactive/notebook/test only; optionally call in test-teardown fixtures. Do **not** wire into `executor`. |
| `QVCharacteristic.q_max_mvar` | 1A | **Wire-in** | Canonical per-DER Q ceiling. Natural home: `diagnose_*`/`compare_scenarios` per-DER headroom reporting, or coordinator headroom logging — call it there instead of re-deriving `Q_RATIO·\|P\|`. Intentional until wired. |
| `QVCharacteristic.slope_inject` | 1A | **Wire-in** | Q(V)-tuning summary (e.g. `compare_runs_updated.ipynb` `read_qv_params` cell, or a diagnose script). Fix stale "16.0" docstring when wiring. Intentional until then. |
| `QVCharacteristic.slope_absorb` | 1A | **Wire-in** | Same target as `slope_inject`; fix the sign-convention asymmetry (it returns a signed slope, `slope_inject` a magnitude) when wiring. Intentional until then. |
| `VoltVarController.run_timestep` | 1C | **Intentional** | Reference single-step loop + module usage examples + test entry point. Production Scenario 4 uses `run_coordinated_timestep` (Doc 3). Keep the two in sync. |
| `VoltVarResult.converged_pre` | 1C | **Intentional** | Public convenience accessor. |
| `VoltVarResult.violations_resolved` | 1C | **Wire-in** | Per-run benchmark summary (historically the test suite's `_print_volt_var_summary`). Intentional until wired. |
| `VoltVarResult.voltage_violations_reduced` | 1C | **Wire-in** | Same target as `violations_resolved`. Intentional until then. |
| `VoltVarResult.summary` | 1C | **Intentional** | Logging convenience (the coordinator logs its own `CoordinatorResult.summary`). |
| `CoordinatorResult.violations_resolved` | 3A | **Wire-in** | Per-run benchmark summary (mirrors `VoltVarResult.violations_resolved`). Intentional until wired. |

**(c) Coverage checklist.** The document ends with a checklist enumerating
every named entity in its source range, each ticked as documented. This is the
mechanical guard against the "skipped a small function" failure mode.

---

## 6. How to read a walkthrough block — the six-part format

Every physics-bearing or non-obvious code line (or tightly-coupled line group)
gets a `####` heading with its **real, current line number(s)** and the
following six parts. A trivially simple line gets a short honest note instead —
density comes from substance, not padding, but *nothing is skipped entirely*.

```
#### Line NN: <what this line achieves>

<the literal code, fenced, with its line number(s)>

**Derivation from first principles** — where an equation comes from (a
conservation law, a standard, a linearisation, a two-point line form), not just
the final formula.

**Code-to-physics mapping** — which symbol is which physical quantity, units.

**Units check** — prove the units resolve.

**Worked numerical example** — canonical inputs from §8, arithmetic shown,
result interpreted physically.

**Design rationale** — why the line is written this way (this guard, this
`.copy()`, this inclusive `<=`, this constant value).

**Where used** — call sites / read sites for the enclosing entity (see §5b).
```

Not every part applies to every line — a pure bookkeeping line has no
"derivation." Include every part that carries information; omit only the ones
that would be filler; never omit a part a supervisor would probe.

---

## 7. Conventions

**Line-number policy.** Every cited line number is the line in the *current*
source at the time the document was written, verified by reading the file
immediately before citing. If a source file is edited later, the affected
walkthrough's line numbers must be re-verified.

**Post-fix state.** All Task 2 fixes are applied. Documents describe the code
as it is now. Surviving stale comments/docstrings are flagged inline.

**British English** throughout the prose: modelling, behaviour, normalise,
initialise, minimise, artefact. (Source code is quoted verbatim, mixed spelling
and all.)

**Sign convention (pandapower, used everywhere):**

```
q_mvar > 0  →  inject reactive power   (capacitive)  →  raises local voltage
q_mvar < 0  →  absorb reactive power   (inductive)   →  lowers local voltage
```

**Canonical constants** (single source of truth,
`volt_var_controller.py` L133–138, verified):

| Symbol | Value | Meaning |
|--------|-------|---------|
| `U1_PU` | 0.96 | Lower saturation — full Q injection at or below |
| `U2_PU` | 0.99 | Deadband lower edge |
| `U3_PU` | 1.01 | Deadband upper edge |
| `U4_PU` | 1.04 | Upper saturation — full Q absorption at or above |
| `Q_RATIO` | 0.25 | Q_max / P_installed (MVAr per MW) |
| `PLAUSIBLE_PU_RANGE` | (0.5, 2.5) | Sanity band for breakpoint entry |
| `SATURATION_TOL` | 1e-6 | Coordinator saturation tolerance (MVAr) — see Docs 2/3 |

> **Stale in-source annotations flagged so far** (documented, not edited):
> - `Q_RATIO` (L137) inline comment reads "Modifying to 0.15 from 0.48" — value
>   in force is **0.25**, matching neither.
> - `slope_inject()` (L400) docstring reads "= 16.0" — true at the old
>   `Q_RATIO=0.48` (0.48/0.03); at the current `0.25` it is **8.33**.

---

## 8. Canonical worked-example inputs

Every worked example across the whole set uses **these** numbers, so the
arithmetic is comparable document to document. A walkthrough that needs a
different value states so and says why.

**Network:** SimBench `1-MV-rural--2-sw` (the primary MV network; the paper's
subject). Radial rural MV feeder, HV/MV transformer at the slack.

**Characteristic:** the canonical constants above — `Q_RATIO = 0.25`,
breakpoints `0.96 / 0.99 / 1.01 / 1.04`.

**Representative DERs:**

| Label | `p_installed_mw` | `q_max = Q_RATIO · p` |
|-------|------------------|------------------------|
| DER-A (small) | 0.5 MW | 0.125 MVAr |
| DER-B (mid) | 1.5 MW | 0.375 MVAr |

DER-A's 0.5 MW is also the DER used in the serial-protocol quantisation
finding, so the same number carries through to Documents 1B and 5.

**Representative bus voltages** exercising the five segments:

| `vm_pu` | Segment | Expected DER-A `q_mvar` |
|---------|---------|--------------------------|
| 0.950 | ≤ U1 — saturate inject | +0.125 |
| 0.975 | U1–U2 ramp | +0.0625 |
| 1.000 | deadband | 0.000 |
| 1.025 | U3–U4 ramp | −0.0625 |
| 1.050 | ≥ U4 — saturate absorb | −0.125 |

**Ramp gains** (Document 1A `slope_inject`/`slope_absorb`): at `Q_RATIO=0.25`,
`slope_inject = 0.25/0.03 = 8.33` and `slope_absorb = −8.33` (MVAr per pu-V per
MW-installed).

**Quantisation reference figure** (Documents 1B and 5): worst-case 4-decimal
ASCII rounding of a 0.5 MW DER's Q value ≈ 2.08 × 10⁻⁴ MVAr ≈ **208 ×
`SATURATION_TOL`** — the paper's headline result.

---

## 9. Cross-reference map

| Module | Task 4 doc | Task 1b part | Task 2 findings on it |
|--------|-----------|--------------|------------------------|
| `volt_var_controller.py` | 1A / 1B / 1C | Part I | — |
| `sensitivity_coordinator.py` | 2 / 3 | Part I | (V_BAND now dynamic — confirmed fixed) |
| `der_dynamics.py` | 4 | Part I | — |
| `volt_var_arduino.ino` | 5 | Part I | — |
| `scenario_result.py` | 6A | Part II | F.1 (empty-`vm_pu` crash) |
| `scenario_1_baseline.py` | 6A | Part II | — |
| `scenario_2_oltc.py` | 6B | Part II | J.1 (switch semantics) |
| `scenario_3_svc.py` | 6C | Part II | — |
| `scenario_4_volt_var.py` | 6D | Part II | G.1 (S2 Q(V) curve reproducibility) |
| `scenario_5_opf.py` | 6E | Part II | — |
| `profile_builder.py` | 7A/7B/7C | Part IV | D.3 (RAD-G contract — re-graded, not a bug) |
| `violation_detector.py` | 11 | Part IV | B.1 (`reset_limits()` cross-run leak) |
| `hosting_capacity.py` | 8 | Part IV | — |
| `executor.py` | 9 | Part III | B.1 (calls `reset_limits()`), H.1/H.3 |
| `benchmark_runner.py` | 9 | Part III | — |
| `wizard.py` | 9 | Part III | K.1, H.1/H.3 |
| `run_benchmark_script.py` | 9 | Part III | G.1 (canonical defaults + `--u1..--u4`) |
| `publisher.py` | 10 | Part V | (research-integrity: `_Encoder` NaN→null) |
| `plot_results.py` | 10 | Part V | (research-integrity: figNN verified) |
| diagnostic scripts | (folded into relevant docs) | Part VI | I.1/I.2 |

---

## 10. Non-negotiable working rules

1. **Verify every claim against the actual current source** before writing it.
   Docstrings were confirmed stale during Task 2; the Task 1b overview is a
   summary. Neither is trusted — the walkthrough goes back to the source.
2. **Line numbers must be real.** Re-read before citing.
3. **Document the post-fix state.** All Task 2 fixes are applied.
4. **Never apply code changes.** Propose → await confirmation → apply. Where a
   walkthrough notices a residual issue (a stale comment, an uncalled method),
   it *flags* it; it does not edit it.
5. **No filler, but no skipping.** Length comes from density. A trivial line
   gets a short honest note — but every named entity is covered (§1 completeness
   rule, §5c coverage checklist).
6. **Every entity gets a "where used".** Uncalled entities are stated as such,
   each with a concrete verdict (wire-in target or intentional marker) recorded
   in the §5 register.
7. **Every file gets a human-readable pseudocode** of its logic flow (§5a).
8. **Ask before starting a new document** which one and in what order.

---

## 11. Status tracker

| Document | State | Notes |
|----------|-------|-------|
| 00 Master Index | ✅ complete (v2) | Doc 11 confirmed; Docs 6/7 split; pseudocode + call-site conventions added |
| 1A Q(U) Characteristic Core | ✅ complete (v2) | **Range corrected to L126–408**; `slope_inject`/`slope_absorb` added; call-site map + pseudocode + coverage checklist added |
| 1B Serial Protocol | ✅ complete | Constants L243–252, exceptions L259–275, `ArduinoSerialInterface` L494–895; quantisation finding (§7) verified against `QUANTISATION_MECHANISM_FINDINGS.md`; stale "2 s" reset docstring flagged |
| 1C VoltVarController Loop | ✅ complete | `VoltVarResult` L411–487, `VoltVarController` L897–1229; `run_timestep` has no production caller (reference loop; production uses `run_coordinated_timestep`); stale "line 923" mirror comment and mislabelled "Layer 2" comment flagged |
| 2 Coordinator Theory | ✅ complete (v2) | Jacobian blocks, Schur complement `J_red = J_QQ − J_QP·J_PP⁻¹·J_PQ`, sensitivity `X = J_red⁻¹`, q=0 double-count invariant, minimum-norm `lstsq`, radial rank deficiency. **§10 corrected v2: rank check is diagnostic-only; lstsq proceeds (per verified source), not an early return** |
| 3A Coordinator — Sensitivity Matrix | ✅ complete | `CoordinatorResult`, `__init__`, Jacobian block slicing, bus-lookup maps, ppc-VM read, Schur-complement pipeline; deprecated `_get_vm_pu_ppc` dead string, commented-out NaN-raise, and unused `_BUS_REF` flagged |
| 3B Coordinator — Algorithm & Orchestration | ✅ complete | `coordinate()` 11 steps + `run_coordinated_timestep()`. **Found: rank-deficiency early return is commented out (L838) — lstsq proceeds; inline comment stale; `curtailment_needed=True` dead assignment. Doc 2 §10 corrected.** Write-order (p-before-q clamp) and clean-timestep GATE documented |
| 4 DER Dynamics | ✅ complete | Exact discrete PT1 (`τ=t95/3`, `α=1−e^(−dt/τ)`) + symmetric ramp limiting; resolution behaviour (α→1 at 15-min, visible at 1-s); `alpha`/`n_ders`/`initialized`/`__repr__` uncalled (inspection API); docstring `Qmax=0.48` flagged as VDE nominal not operative value |
| 5 Arduino Firmware | ⬜ not started | Next in queue |
| 4 DER Dynamics | ⬜ not started | — |
| 5 Arduino Firmware | ⬜ not started | Read immediately after 1A |
| 5 Arduino Firmware | ✅ complete | `compute_q` verified branch-for-branch mirror of 1A; `handle_message` = device end of 1B (INIT/CFG/P/V/END + ERR taxonomy); `FLOAT_DEC=4` Q-quantisation leg; ATmega328P SRAM-budget engineering (buffer reuse, `strtod`/`strtol`, flash strings) |
| 6A-i Profile Adapter | ✅ complete | `AdaptedProfiles`, `adapt_profiles` (Q/P-ratio load_q, PV+wind merge, dt_s inference), `oversize_inverters`, `slice_profiles`; `oversize_inverters` docstring `Q_RATIO=0.48` flagged |
| 6A-ii Record & Result | ✅ complete (v2) | `TimestepRecord` (+ `to_checkpoint_dict`/`from_checkpoint_dict`), `ScenarioResult`+`from_records`, `summary_dict`. **`min_vm` double-compute now FIXED.** Two checkpoint robustness fixes proposed (NaN-encoder in `publisher._append_jsonl`; `d.get` cross-version) — not applied |
| 6A-iii Baseline runner | ✅ complete | Declarative `run_timeseries`+`ConstControl`+`OutputWriter` (vs manual loops); integer reindex; NaN-based convergence; ±ε thresholding. **Found: `_RUNPP_TIMING` never cleared (docstring claims it is) → cross-run timing inflation; `make_record_from_report` imported-but-unused** |
| 6B-i OLTC Setup | ✅ complete | Tap-metadata completion/override, 3-tier topology trafo selection, ganging validation (intersection range), and empirical tap-sign calibration by network probing; OLTC deadband (±2%) tighter than violation band (±5%) explained. No bugs found |
| 6B-ii OLTC Loop | ✅ complete | `run_scenario_2` — 7-step setup orchestration + the tap control state machine (deadband → move/rollback/rail/deadband, mapped to the six `tap_*` fields), checkpoint resume with **tap-position re-seeding**, `post_pf_reused` PF-count optimisation. Two cosmetic flags: resume progress-% overshoot (L950–951), over-indented live-CSV block |
| 6C SVC | ✅ complete | `scenario_3_svc.py` — centralised SVC: 3-tier bus selection (stress → first-violation scan → remote fallback), `Q_MAX=0.20·Σsn_mva`, deadbanded proportional V-Q droop (continuity at deadband edge), temporary sgen created + `finally`-removed, **stateless resume** (no seeding needed). Cosmetic progress-% overshoot flagged |
| 6D Volt-Var (HIL) | ✅ complete | The core scenario: three-tier hierarchy (local Q(V) → coordination → **curtailment**), delegates Tiers 1+2 to `run_coordinated_timestep` (Doc 3B), adds Tier 3 curtailment (linear decay, single `p_prev` update — no intra-step `step()`), 4A/4B split with separate checkpoints, dry-run/hardware dispatch (quantisation origin), **DER-state resume seeding** (PT1/ramp continuity). Cosmetic %-overshoot + redundant `for…else` flagged |
| 6E OPF | ✅ complete | `scenario_5_opf.py` — AC OPF theoretical bound: objective (`_DER_COST=-1` maximises DER/minimises curtailment), fair `min(q_vde, q_circle)` reactive envelope, controllable slack, mandatory `create_continuous_bus_index`, per-step flexibility/cost rebuild, cyipopt + throttled non-convergence (SimBench MV fails, CIGRE MV only), **stateless resume**. NOTE: OPF progress-% is CORRECT (reference pattern) 
| 6A-iii Baseline runner | ⬜ not started | `scenario_1_baseline.py` |
| 6B OLTC | ⬜ not started | — |
| 6C SVC | ⬜ not started | — |
| 6D Volt-Var (HIL) | ⬜ not started | Largest 6-part; may sub-split |
| 6E OPF | ✅ complete | scenario-runner series complete (6A–6E) 
| 7A Profile Builder — PV | ⬜ not started | Largest 7-part; may sub-split |
| 7B Profile Builder — Wind | ⬜ not started | — |
| 7C Profile Builder — Load | ⬜ not started | — |
| 8 Hosting Capacity | ⬜ not started | — |
| 9A Benchmark Runner | ✅ complete | `benchmark_runner.py` — the orchestration hub: `SCENARIO_REGISTRY` (6 scenarios, capability flags), `BenchmarkConfig`/`BenchmarkResult` (tri-state `results[n]`), `_build_kwargs` (capability-gated; publish_fn+enable_checkpointing to all, dry_run/port/coordination to S4, verbose_opf to S5/6), `run_benchmark` (two-layer resume: L1 whole-scenario JSON-skip + L2 per-timestep; deepcopy-in-try isolation; checkpoint archive→.completed; per-scenario live_csv_rewrite_fn injection; HC + guarded recursive HC-stressed re-benchmark via profile_factory). No bugs |
| 9B Executor | ✅ complete | `executor.py` — RunPlan→executed benchmark: typed ExitCode/ExecutorError failure identity, `build_benchmark_config`, the runtime channels (`apply_qv_overrides` = one-call Q(V) to all consumers; `apply_violation_limits` = the late-binding rebind partner to Doc 11, reset_limits+set_limit, honest scope note incl unbalance no-op), `build_net_and_profiles` + dataset-faithful `profile_factory`, `validate_plan` cross-field checks, and the phased `execute` pipeline (per-phase typed exit codes, failure-class-by-where-it-surfaces). No bugs |
| 9C Wizard & Run Plan | ✅ complete | `run_plan.py` (4 config dataclasses — pure data container, `to_dict`/`from_dict` reproducibility incl null→None scaling-key coercion + `switches_to_open` alias + `or {}` guard) + `wizard.py` (exception-based back-navigation, per-field prompt builders, `run_wizard` step machine with conditional steps). Full depth. No bugs — enforces the wizard(per-field)/executor(cross-field) validation split |
| 9D Script & Helpers | ✅ complete | `__main__.py` (interactive entry: preset/wizard→execute, typed-exit-code propagation, semantic `_validate_loaded_plan`) + `run_benchmark_script.py` (direct fast-iteration path: argparse, network-selector "menu as code", PF-preserving scaling, DER/switch templates = the reference for `NetworkConfig` mods, config + run dispatch) + `helpers.py` (Rich console output). Flags: duplicate `print_run_plan` (cosmetic); direct-script never calls `apply_violation_limits` → thermal-limit overrides no-op on that path |
| 9E CLI Support | ✅ complete | `network_catalogue.py` (preset menu data — data-only, executor resolves preset_name) + `_console.py` (shared Rich theme + single console) + `_data.py` (the dashboard READ layer mirroring publisher: 6-pattern run discovery, mtime-keyed JSON loaders, the robust byte-offset `read_jsonl_since` live follower with truncation detection, log tail) + `test_helpers.py` (console-helpers smoke test). Full depth. No bugs. **Doc 9 + Task 4 set COMPLETE** |
| 10A Publisher | ✅ complete | `publisher.py` — JSON/JSONL output + checkpointing: NaN-safe writers (both suggested fixes applied), payload builders (topology/profiles/scenario/hc/live-frame), the two-stream `PublishHandle` (live truncated vs checkpoint never-truncated held-open handle = RPi mmc_rescan fix; every-step checkpoint vs every-k live; crash-surviving elapsed; .completed fallback), three-way publish split. No bugs |
| 10B Report Figures | ✅ complete | `plot_results.py` — 14 report figures (fig01–14) consuming published JSON (decoupled from net); IEEE style + centralised scenario colours; shared network-base drawing; orchestrator with per-figure exception isolation + HC-only merge. Flag: `cm.get_cmap` deprecated/removed in matplotlib ≥3.9 (network-map figs) |
| 10C Network Plots | ✅ complete | `network_plotter.py` — net-direct network-type-aware topology (SimBench geo+structured 2-panel; CIGRE/Kerber schematic PNG+simple_plot), 3-tier `get_bus_xy` (pandapower 3.x geodata migration), no-mutation deep-copy structured layout, annual/day/extreme-day profile figures. No functional bugs (fragile sgen-restyle heuristic + CRLF noted). Doc 10 complete |
| 11 Violation Detector | ✅ complete | `violation_detector.py` — the single source of violation truth: the ±ε tolerances, late-bound-defaults mechanism (`None`→current module constant at call time, so `executor.apply_violation_limits`+`set_limit` apply user limits process-wide), `set_limit`/`reset_limits` hygiene, `ViolationReport`(+3ph) rich object (deviation-sorted buses feed the coordinator), balanced+3ph check helpers. No functional bugs; minor cleanliness flags |

---

## 12. Change log

| Version | Change |
|---------|--------|
| v1 | Master index created. Document set defined with flagged deviations. Canonical worked-example inputs fixed. Document 1A produced (L126–394). |
| **v2 (this session)** | **Doc 11 (`violation_detector.py`) confirmed in scope.** **Doc 6 split into 6A–6E; Doc 7 split into 7A–7C; Doc 9 flagged as possibly sub-splitting.** Added §5 mandatory per-document sections (**file pseudocode**, **call-site "where used" map**, **coverage checklist**) and §1 completeness rule, prompted by the omission of `slope_inject`/`slope_absorb` from the 1A v1 draft. **Document 1A re-verified and reissued (v2):** range corrected to **L126–408**; the two slope helpers added with full treatment; a call-site map, file pseudocode, and coverage checklist added; `slope_inject` stale docstring "16.0" flagged. |
| **v3 (this session)** | Adopted the **uncalled-public-API verdict convention** (§5 register): `reset_qv_parameters` marked intentional; `q_max_mvar`/`slope_inject`/`slope_absorb` marked wire-into-diagnostics. Added the L243–275 deferral note to 1A. **Produced Document 1B** (serial transport layer) with the quantisation finding derived from verified constants and cross-checked against `QUANTISATION_MECHANISM_FINDINGS.md`; flagged the stale "2 s" reset-delay docstring. |
| **v4 (this session)** | **Produced Document 1C** (`VoltVarResult` + `VoltVarController`), completing the `volt_var_controller.py` set (1A+1B+1C). Findings: `run_timestep` has no production caller (production Scenario 4 uses `run_coordinated_timestep`); several `VoltVarResult` convenience properties uncalled — added to the §5 register with verdicts. Flagged the coordinator's stale "line 923" mirror comment and the mislabelled "Layer 2" comment in `_clamp_to_net_limits`. Apparent-power cap derived from `S²=P²+Q²`. |
| **v5 (this session)** | **Produced Document 2** (Sensitivity Coordinator — Theory): full derivation of the NR Jacobian block structure, the Schur-complement reduction to `J_red`, the dV/dQ sensitivity matrix `X`, the `q_current=0` double-count invariant, the minimum-norm `lstsq` correction, and radial-feeder rank deficiency. Two mandatory-section adaptations for a theory doc noted (concept→implementation map replaces the call-site map; coverage checklist deferred to Doc 3). All constants (`V_TARGET_PU`, `MIN_VIOLATION_DV`, `RCOND_THRESHOLD`, `SATURATION_TOL`) verified against source; confirmed the V-band is read dynamically from `violation_detector`. |
| **v6 (this session)** | **Split Document 3 into 3A/3B** (flagged deviation #5; `sensitivity_coordinator.py` is 1,106 lines). **Produced Document 3A** (sensitivity-matrix construction): `CoordinatorResult`, `__init__`, `_build_jacobian_blocks`, `_bus_lookup_maps`, `_get_vm_pu_ppc`, `_compute_der_sensitivity` — the Schur-complement pipeline implementing Doc 2 §4–§6, with the three-indexing-scheme bookkeeping explained. Flagged: deprecated `_get_vm_pu_ppc` preserved as a dead string literal (L394–447), commented-out NaN-raise (L481–486), unused `_BUS_REF` (L116). Added `CoordinatorResult.violations_resolved` to the §5 register. |
| **v7 (this session)** | **Produced Document 3B** (`coordinate()` 11-step algorithm + `run_coordinated_timestep()`), completing the coordinator set (3A+3B). **Material finding:** the rank-deficiency early return is commented out (L838) so `lstsq` proceeds regardless of rank (numerically safe via minimum-norm SVD), but the inline comment (L835–836) is stale and `curtailment_needed=True` (L837) is a dead assignment. **Document 2 §10 corrected** to match verified source (rank check is diagnostic, not a gate). Documented the clean-timestep GATE and the p-before-q write-order dependency. |
| **v8 (this session)** | **Produced Document 4** (DER Dynamics): the exact discrete PT1 reactive filter (derived `τ=t95/3` from the 95%-settling relationship and `α=1−e^(−dt/τ)` as the analytic, not Euler, discretisation) and symmetric active-power ramp limiting, with the α→1 (15-min) vs α<1 (1-s HIL) resolution behaviour tied to `CoordinatorResult.dq_dynamics`. Flagged the docstring's `Qmax=0.48` as the VDE nominal (operative value is `Q_RATIO=0.25`). Four inspection-API entities marked uncalled-intentional. |
| **v9 (this session)** | **Produced Document 5** (Arduino firmware): verified `compute_q` is the branch-for-branch mirror of 1A's `compute_setpoint`; mapped `handle_message` to 1B's message/error taxonomy (closing the `ERR:UNKNOWN`-transient / deterministic-`ERR` loop); documented the `FLOAT_DEC=4` `dtostrf` **Q-quantisation leg** (complement to 1B's V leg) and the AVR-float32 term; walked the ATmega328P 2 KB-SRAM engineering (buffer reuse, `strtod`/`strtol` vs `atoi`, `F()` flash strings, `skip_to_newline` overflow recovery). Confirmed the mirror invariant. |
| **v10 (this session)** | **Split Document 6A into 6A-i/ii/iii** (flagged deviation #6; `scenario_result.py` is 1,226 lines of shared infrastructure). **Produced Document 6A-i** (the profile adapter): `AdaptedProfiles` and `adapt_profiles` (network-Q/P-ratio `load_q` derivation, PV+wind merge to `der_p`, three-tier `dt_s` inference), plus the `oversize_inverters` and `slice_profiles` study utilities. Flagged the stale `Q_RATIO=0.48` in `oversize_inverters`'s docstring; documented the `np.divide(out=,where=)` vs `np.where` idiom and the positional-not-label slice subtlety. |
| **v11 (this session)** | **Produced Document 6A-ii** (the result types): `TimestepRecord` (the wide `Optional`-per-scenario record + `make_record_from_report` factory with ±ε thresholding matching `violation_detector`) and `ScenarioResult`/`from_records` (deriving VDI, `violation_duration_h`, `coordination_rate`, reactive/curtailed energy, and the signed grid import/export split) + `summary_dict`. **Material finding:** `from_records` computes `min_vm` twice (L1024–1028), the second dropping the `default=nan` empty-sequence guard → latent `ValueError`; recommend deleting L1028. |
| **v14 (source update — checkpointing)** | User uploaded updated `scenario_result.py`, `scenario_2..5`, `benchmark_runner.py`, `publisher.py` adding **crash-resume checkpointing + live-CSV rewriting**. Ran Task 2 (correctness) + Task 1b (doc) checks: `min_vm` bug **not** fixed (now L1117–1121); checkpoint round-trip sound (`int(k)` JSON key coercion) but flagged NaN-encoder dependency and `from_checkpoint_dict` cross-version `d[...]` KeyError risk; OPF (`scenario_5`) now genuinely streams `publish_fn` (stale "ignored" comment correctly removed); `PublishHandle` docstring updated well. **Reissued Document 6A-ii (v2)** with new line numbers + the two checkpoint methods. **6A-i and 6B-i confirmed unaffected** (changes fall outside their ranges). Pending docs **6B-ii/6C/6D/6E/9/10** will incorporate the checkpointing/resume logic (esp. Scenario-4 `DERDynamics` state seeding and Scenario-2 tap-position seeding on resume) when produced. |
| **v15 (fixes)** | `min_vm` double-compute **fixed** by user (redundant unguarded line removed) — 6A-ii updated to reflect resolution. Proposed two robustness fixes (as text, not applied): (a) **checkpoint NaN → invalid JSON** — root cause is `_Encoder.default()`'s NaN branch never firing for native floats; fix = `_nan_to_none` sanitiser + `allow_nan=False` in `publisher._append_jsonl`/`_dump` (covers live + checkpoint), optional source-level `to_checkpoint_dict._s` reinforcement; (b) **`from_checkpoint_dict` cross-version** — switch the five `None`-defaulted fields (`p_target_mw`, `losses_mw`, `grid_import_mw`, `der_gen_mw`, `load_mw`) from `d[...]` to `d.get(...)`, keep required fields as `d[...]`, optional `_ckpt_version` stamp. |
| **v16 (source update — publisher/custom_controller/plugin_runner)** | Ran Task 2/1b checks. **🔴 Critical: `custom_controller.py` L182–183 declare `live_csv_rewrite_fn` twice → `SyntaxError`, file won't import (breaks `plugin_runner` + the `--controller` path).** Fix suggested (delete one line). **Fix (a) confirmed applied** to `publisher._append_jsonl` (`_nan_to_none` + `allow_nan=False`) — but **not to `_dump`** (summary payloads with NaN scalars still emit invalid JSON); Fix 2 suggested. `PublishHandle` docstring updated well; stale bare-callable `publish_fn` usage removed (`publish_fn` is now the handle object). **No written walkthrough edited:** `publisher`→Doc 10 (unwritten), `custom_controller`/`plugin_runner`→no/Doc-9 home; secondary call-site pointers to `custom_controller` deliberately **not** re-pinned (line numbers will shift again after the SyntaxError fix). Flagged `custom_controller.py` doc-gap (proposed Doc 12). |
| **v17 (this session)** | Answered the mid-run file-replacement question (Python caches modules at first import; `plugin_runner` imports `custom_controller` at top-level L80, fired at plugin registration/setup — so a broken file crashes at setup, and a mid-run edit is never re-read; fix + restart is the correct workflow). **Confirmed Document 12** (Custom Controller & Plugin Subsystem — deviation #8, expanded scope, `plugin_runner` moved out of Doc 9). **Produced Document 6B-ii** (`run_scenario_2`): the 7-step setup orchestration + the tap control **state machine** (deadband decision → accept/rollback/rail/deadband, mapped one-to-one onto the six `tap_*` fields), the checkpoint resume block with **tap-position re-seeding** for state continuity, and the `post_pf_reused` power-flow-count optimisation. Two cosmetic flags: resume progress-% overshoot (L950–951); over-indented live-CSV block (L929–935). |
| **v18 (this session)** | **Produced Document 6C** (SVC): the centralised static-var-compensator scenario — documented the 3-tier bus selection (stress-based weakest bus → 7-day first-violation pre-scan → remote-bus fallback), the `Q_MAX = 0.20·Σsn_mva` sizing and `k_q` droop gain, the deadbanded proportional V-Q droop law (derived, with the continuity-at-the-deadband-edge point mirroring Doc 1A §7), the temporary-sgen model created at setup and removed in a `finally` block (clean-net guarantee), and the **stateless resume** (no state seeding needed, unlike OLTC/Volt-Var — a genuine simplification). One cosmetic flag: resume progress-% overshoot (L590), same as OLTC. |
| **v19 (this session)** | **Produced Document 6D** (Volt-Var HIL — the core scenario): documented the three-tier hierarchy (local Q(V) → coordination → active-power curtailment) and its reactive-before-active justification; how Tiers 1+2 are delegated to `run_coordinated_timestep` (Doc 3B) while this file adds **Tier 3 curtailment** (linear 10%/iter decay, terminating at clear-or-P=0, with the subtle single-`p_prev`-update-no-intra-step-`step()` ramp handling); the 4A/4B split threaded through the `scenario_id`/checkpoint key; the dry-run-vs-hardware dispatch (context-managed Arduino port — the origin of the quantisation finding); and the **DER-state resume seeding** (PT1/ramp continuity, the Scenario-4 analogue of OLTC's tap re-seed). Two flags: cosmetic %-overshoot (L533); defensively-redundant `for…else` on the curtailment loop (L428–429). |
| **v20 (re-verification pass — walkthroughs vs project tree)** | Re-checked every completed walkthrough against the current `/mnt/project` source (now synced ahead of the chat uploads). **Verified current, unchanged:** Docs 2, 3A, 3B (`sensitivity_coordinator.py` 1106), 4 (`der_dynamics.py` 474), 5 (`volt_var_arduino.ino` 578); and 1A/1B/1C (`volt_var_controller.py` 1228, −1 immaterial — key anchors `compute_setpoint` L298 / `ArduinoSerialInterface` L494 / `_clamp_to_net_limits` L1006 / `run_timestep` L1089 all still match). **Updated:** **6A-ii→v3** (re-pinned to 1,348-line `scenario_result.py`; documented the new **timestep-coverage integrity check** in `from_records`; closed all three v2 flags — `min_vm` fixed, publisher NaN encoder applied, cross-version `d.get`+`_ckpt_version` fixes applied); **6A-iii** (new accepted-but-not-implemented `enable_checkpointing`/`live_csv_rewrite_fn` params + rationale; `_RUNPP_TIMING`-never-cleared flag still stands); **6A-i** (anchor re-pin ≤+2, content unaffected); **6B-ii/6C/6D** (header re-verification notes; +2/+2/+1 drift, content current). |
| **v21 (this session)** | **Produced Document 6E** (OPF — `scenario_5_opf.py`, from the current project tree), completing the scenario-runner series (6A–6E). Documented the AC OPF as the **theoretical bound**: the objective (`_DER_COST = −1` maximises DER output / minimises curtailment; `_EXT_GRID_COST = +0.001`), the **fair `min(q_vde, q_circle)` reactive envelope** (OPF gets the same VDE `Q_RATIO` + apparent-power-circle limit as Volt-Var), the controllable-slack setup, the mandatory `create_continuous_bus_index` before `adapt_profiles` (PYPOWER contiguous-bus requirement), the per-timestep reset-then-activate flexibility + cost rebuild, `cyipopt` `runopp` with throttled non-convergence (SimBench MV ill-conditioned → CIGRE MV only), and the **stateless resume** (no seeding, like SVC). **Positive finding:** OPF's progress-% (`100·t/_T_full`) is **correct** — the reference pattern the scenarios 2/3/4 overshoot bug should adopt. One cosmetic flag (over-indented live-CSV block). |
| **v22 (this session)** | **Split Document 12 into 12A–12D** (the subsystem is ~2,200 lines: `custom_controller.py` 498, `plugin_runner.py` 636, `network_plugin.py` 775, plus small loaders + examples). **Produced Document 12A** (`custom_controller.py`, now compiling): the generic user-controller HIL loop — the plugin contract (`controller_fn(vm_pu, p_installed)→q_mvar`), the dry-run `VoltVarController` reuse for identical DER semantics, the static two-runpp loop (no dynamics/ramp — a clean sibling of Scenario 4), the `_validate_controller_output` + `_clamp_to_net_limits` safety layers, the opt-in clean-timestep gate, and the stateless resume. **Positives found:** SyntaxError fixed; correct live-CSV indentation; correct progress-% (no resume overshoot); defensive `hasattr` resume. No bugs. |
| **v23 (this session)** | **Produced Document 12B** (`plugin_runner.py`): the plugin lifecycle — `load_plugin` (YAML schema validation, built-in-id collision guard, YAML-relative path resolution), `_import_controller_fn` (dynamic file-location import under a mangled unique `sys.modules` name), `_allocate_plugin_num`, `HardwareControllerFn` (the Arduino-backed `controller_fn` — real HIL for custom controllers), and `register_and_run` (software/hardware failsafe cascade "Python only ⇒ dry run", temporary registration with `finally` cleanup, config-copy immutability, public-boundary `ScenarioResult` reconstruction). **Confirmed stale docstring flagged:** `load_plugin` claims `hardware: true` raises `NotImplementedError`, but the code fully supports hardware plugins (firmware validation + `HardwareControllerFn`) — code correct, docstring stale. |
| **v24 (this session)** | **Produced Document 12C** (`network_plugin.py`): the custom-network subsystem — three network sources (JSON/pickle via `pp.from_pickle`/zero-arg function loader), four profile strategies (`simbench_native` portable via `sb.get_absolute_values` on the loaded net, `dwd_pvlib`, `flat`, `custom`) with the `simbench_native→dwd_pvlib` fallback, the `_dwd_safe_name` name-collision gotcha handling, `load_network_from_yaml` (+ additive `plugin_meta`), the 6-check warnings-not-exceptions `validate_network_plugin`, and `make_profile_factory` (HC re-benchmark, strategy-faithful rebuild). No bugs — clean, defensively designed; strengths recorded. |
| **v25 (this session)** | **Produced Document 12D**, **completing Document 12** (12A–12D): the loaders + worked examples — `my_network_loader.py` (function-source contract), `network_export.py` (JSON export snippet), `droop_controller.py` (proportional Q–P droop — derived, deadband-continuous), `deadband_controller.py` (4-breakpoint Q(V) via a neat `np.interp`, tracks 4A with defaults), and the six example YAMLs (coverage matrix over source × strategy). **Flags:** the stale `hardware: true` claim's **third location** found (`droop_controller.yaml` comment; the other two are `plugin_runner.load_plugin` docstring L119/L130); `Scenario_3.yaml` has a mis-pasted header and a Windows-backslash `data_dir` that won't resolve on the Linux RPi. No `.py` bugs. **Doc 12 subsystem complete.** |
| **v26 (this session)** | **Produced Document 11** (`violation_detector.py`): the foundational single-source-of-violation-truth module — the `±ε` tolerances (`VOLTAGE_EPSILON=1e-6` etc.) referenced throughout every scenario doc; the **late-bound-defaults mechanism** (`detect_violations` `None` args → current module constant at call time, so the CLI executor rebinds the globals via `set_limit` and user limits apply process-wide without kwarg threading); `set_limit`/`reset_limits` process-global hygiene; the rich `ViolationReport`(+`ViolationReport3ph`) object (deviation-sorted `most_severe_*_buses` feed the coordinator's target ordering); and the balanced + 3-phase check helpers (in-service filtering, schema-correct empty frames, two-guard convergence). No functional bugs; minor cleanliness flags (duplicate `import warnings`, redundant local `np`, over-indented `set_limit` body, possibly-unused `THERMAL_LOADING_PLAUSIBLE_MAX`). |
| **v27 (this session)** | **Split Document 10 into 10A/B/C** (output subsystem ~3,435 lines) and **produced Document 10A** (`publisher.py`): the JSON/JSONL output + checkpointing layer. Documented the NaN-safe writers (**both suggested fixes confirmed applied** — `_nan_to_none` + `allow_nan=False` in `_dump`/`_jsonl_line`/`_append_jsonl`), the payload builders (topology/profiles/scenario/hc/live-frame), the two-stream `PublishHandle` (live truncated vs checkpoint never-truncated **held-open handle = the RPi `mmc_rescan`-hang fix**, confirmed in-code; every-step gap-free checkpoint vs every-`k` coarse live; crash-surviving cumulative elapsed; `.completed` archive fallback; partial-tail-line tolerance), and the three-way publish split (crash-resilience + no end-of-run write burst). No bugs. |
| **v28 (this session)** | **Produced Document 10B** (`plot_results.py`): the 14 matplotlib report figures (fig01–14) as a figure catalogue — the JSON-consuming decoupling (reads published payloads via `load_publisher_dir`, never a live net), the IEEE style + centralised `SCENARIO_ORDER/LABELS/COLORS`, the shared network-base drawing (3-priority line colouring, networkx layout fallback, geo/schematic detection), and the `plot_results` orchestrator (number→lambda registry, per-figure exception isolation, HC-only-run merge). **Flag:** `cm.get_cmap` (L208) is deprecated in matplotlib ≥3.7 and removed in ≥3.9 → network-map figures would fail on a modern matplotlib (recommend `plt.get_cmap`); version-dependent. |
| **v29 (this session)** | **Produced Document 10C**, **completing Document 10** (10A–10C): `network_plotter.py` — the net-direct, network-type-aware documentation figures (distinct from 10B's JSON-consuming results figures). Documented the SimBench geographic+structured two-panel vs CIGRE/Kerber schematic-PNG+`simple_plot` topology logic, the robust 3-tier `get_bus_xy` handling the pandapower 3.x geodata migration (`bus_geodata`→`bus["x"/"y"]`→`bus["geo"]` GeoJSON), the no-mutation deep-copy + `create_generic_coordinates` structured layout, and the annual/day/extreme-day profile figures (fill-plus-daily-mean idiom, shared extreme-day markers). No functional bugs (fragile sgen-patch restyle heuristic + CRLF line endings noted). **Output subsystem (Doc 10) complete.** |
| **v30 (this session)** | **Split Document 9 into 9A–9E** (orchestration/CLI ~6,000 lines) and **produced Document 9A** (`benchmark_runner.py`): the orchestration hub — `SCENARIO_REGISTRY` (6 scenarios via capability flags; 4A/4B both `run_scenario_4`, 6=OPF), `BenchmarkConfig`/`BenchmarkResult` (tri-state `results[n]` = ScenarioResult|dict|None), `_build_kwargs` (capability-gated injection), and `run_benchmark` — the **two-layer resume** (Layer 1 whole-scenario skip via `scenarios/<id>.json`, Layer 2 per-timestep in runners), `deepcopy`-in-`try` fail-isolation, checkpoint archive-not-delete (`.completed`), per-scenario `live_csv_rewrite_fn` injection, and HC + the **guarded recursive HC-stressed re-benchmark** (`profile_factory` = 12C `make_profile_factory`; `run_hc=False`/`run_hc_scenarios=False` recursion guard). No bugs (the `publish_fn`-unconditional concern is resolved — all six runners accept it). |
| **v31 (this session)** | **Produced Document 9B** (`executor.py`, the largest file at 1564 lines): the CLI run executor — the typed `ExitCode`/`ExecutorError` failure-identity design, `build_benchmark_config` (RunPlan→BenchmarkConfig), the two **runtime channels** (`apply_qv_overrides` pushing Q(V) to dry-run curve + coordinator + Arduino CFG in one call; **`apply_violation_limits`** — the late-binding rebind partner to Doc 11: `reset_limits()` + `set_limit`, given fuller treatment with a worked example + the honest scope note including the self-documented `unbalance_max_percent` no-op), `build_net_and_profiles` + the dataset-faithful `profile_factory` closure, `validate_plan`'s cross-field checks, and the phased `execute` pipeline (per-phase typed exit codes; failure-class-by-where-it-surfaces — plugin-load vs controller-crash vs framework-crash). No bugs. |
| **v32 (depth-uniformity pass)** | Deepened three docs to full control-stack density per request (no catalogue shortcuts): **Doc 11** — expanded the 3-phase report and every 3ph check helper (`_runpp_3ph_converged`, `_check_voltage_3ph` worst-phase logic, `_check_unbalance` IEC 62749, `_check_lines_3ph`, `_check_trafos_3ph`) into full per-entity blocks with worked examples, plus worked examples on the balanced helpers (checklist updated, no more "by parallel"); **Doc 9A** — worked examples for `_is_lv_network` (modal vn_kv), `_build_kwargs` (4B vs OPF traces), comparison-row builders, and a full two-layer-resume crash/re-launch trace; **Doc 10A** — deepened `build_profiles_payload` (dual-resolution sizes), `build_hc_payload` (gain_mw), `build_live_frame` (compact-vs-lossless frame). |
| **v33 (this session)** | **Produced Document 9C** at full depth (`run_plan.py` + `wizard.py`): the declarative-config layer + its interactive builder. `run_plan.py` — the four config dataclasses (`ParameterConfig`/`NetworkConfig`/`DatasetConfig`/`RunPlan`), each walked with the reproducibility contract (`to_dict`=asdict, `from_dict` with `.get()` forward-compat, the `"null"→None` scaling-key coercion, the `switches_to_open` rename alias, the `or {}` nested guard) and worked examples; `RunPlan` as the pure data container (separation-of-concerns → why `executor.validate_plan` exists). `wizard.py` — the exception-based back-navigation (`BackRequested`/`BACK_TOKEN "<"`, `_ask`/`_ask_value`), the eight per-field prompt builders (per-field validation only), and `run_wizard`'s ordered step machine (back-nav via `index-1`, two conditional steps, worked back-nav trace). No bugs; the wizard(per-field)/executor(cross-field) validation split is enforced and documented. |
| **v34 (this session)** | **Produced Document 9D** at full depth (`__main__.py` + `run_benchmark_script.py` + `helpers.py`): the two entry paths reaching one core. `__main__.py` — the interactive entry (preset-or-wizard → confirm → `execute` with **typed-exit-code propagation**), and the **semantic** `_validate_loaded_plan` (a months-old preset re-checked against the current codebase — the third validation layer, warnings-not-raises). `run_benchmark_script.py` — the direct fast-iteration path (argparse `--network`/`--controller`, the network-selector "menu as code" with 8 commented alternatives, PF-preserving oversize/scaling mirroring `executor._apply_scaling`, the DER-injection/switch templates that are the *reference* for `NetworkConfig.der_placements`/`switches_to_flip`, and the run dispatch identical to executor Phase 3). `helpers.py` — the shared Rich console output. **Flags:** duplicate `print_run_plan` in `main` (L172/L174, cosmetic); the direct-script path never calls `apply_violation_limits` → thermal/angle limit overrides are a **no-op** there (use the wizard path). |
| **v35 (this session)** | **Produced Document 9E**, **completing Document 9 AND the entire Task 4 walkthrough set**. The CLI support modules at full depth: `network_catalogue.py` (preset menu data — data-only discipline keeps heavy imports out of the wizard), `_console.py` (shared Rich theme + single `console` with semantic style names), `_data.py` (the dashboard **read layer** mirroring the publisher — `detect_project_root`, 6-pattern `discover_runs` reconciling the two entry paths' dir shapes, **mtime-keyed** JSON loaders that auto-refresh re-run scenarios, the robust byte-offset **`read_jsonl_since`** live follower with whole-lines-only parsing + torn-line skipping + size-shrink truncation detection, log discovery/tail, `sidebar_run_selector`), and `test_helpers.py` (the console-helpers smoke test). No bugs. **Task 4 now spans the entire codebase** — control stack (1–5), scenario runners (6A–6E), orchestration (9A–9E), output (10A–10C), violation detector (11), plugin subsystem (12A–12D); every entity enumerated, all against current source. Remaining open items are the previously-flagged stale docstrings/cosmetics awaiting go-ahead, not undocumented code. |
| **v12 (this session)** | **Produced Document 6A-iii** (baseline runner), completing Document 6A. Drew out the declarative (`run_timeseries`/`ConstControl`/`OutputWriter`) vs manual-loop architectural contrast; documented the integer-reindex requirement, the element-index profile mapping, the three mandatory `run_timeseries` kwargs, and NaN-based convergence detection. **Two findings:** `_RUNPP_TIMING` is never cleared despite its docstring claiming so (cross-run `t_total_ms` inflation — recommend `.clear()` at function start); `make_record_from_report` imported but unused (records built inline from logs). |
| **v13 (this session)** | **Split Document 6B into 6B-i/ii** (flagged deviation #7; `scenario_2_oltc.py` is 956 lines). **Produced Document 6B-i** (OLTC setup infrastructure): tap-metadata missing-value detection (string-placeholder aware), non-destructive completion + override precedence, 3-tier topology transformer selection, ganging-feasibility validation (intersection range), and the empirical `_calibrate_tap_sign` probing method. Highlighted the OLTC ±2% regulation deadband being tighter than the ±5% violation band. No bugs found in this slice. |
