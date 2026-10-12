# TASK 4 — DOCUMENT 9B
## Orchestration Layer, Part 2: The Executor
### `executor.py` — the CLI run executor (1564 lines)
### Complete line-by-line walkthrough

> **Source verified:** read against the current `/mnt/project/executor.py` (1564
> lines, the largest single file in the project). **Read alongside:** **9A**
> (`benchmark_runner` — `execute` calls `run_benchmark`), **9C** (`run_plan.py` — the
> `RunPlan` this consumes), **11** (`violation_detector` — `apply_violation_limits`
> rebinds its constants; the late-binding partner), **12B/12C** (`register_and_run`,
> the plugin path).

---

## TABLE OF CONTENTS

1. [Overview — the RunPlan-to-execution bridge](#1-overview)
2. [File logic flow — pseudocode](#2-file-logic-flow--pseudocode)
3. [Call-site map](#3-call-site-map)
4. [Section A — Exit codes & error hierarchy (L114–177)](#4-section-a--exit-codes--error-hierarchy-l114177)
5. [Section B — Presets, studies & framework path (L191–393)](#5-section-b--presets-studies--framework-path-l191393)
6. [Section C — build_benchmark_config (L396–461)](#6-section-c--build_benchmark_config-l396461)
7. [Section D — Runtime channels: apply_qv_overrides & apply_violation_limits (L468–554)](#7-section-d--runtime-channels-l468554)
8. [Section E — build_net_and_profiles & the profile_factory (L808–1033)](#8-section-e--build_net_and_profiles--the-profile_factory-l8081033)
9. [Section F — Validation & hardware checks (L1051–1344)](#9-section-f--validation--hardware-checks-l10511344)
10. [Section G — execute: the phased pipeline (L1346–1564)](#10-section-g--execute-the-phased-pipeline-l13461564)
11. [Findings and flags](#11-findings-and-flags)
12. [Coverage checklist](#12-coverage-checklist)

---

## 1. Overview

`executor.py` is the bridge from a **declarative `RunPlan`** (what the wizard produced,
9C) to an **executed, published benchmark**. It resolves the network and profiles, applies
the run's parameter overrides, runs `run_benchmark` (or the plugin path), publishes, and
prints the summary — all inside a **phased pipeline where each phase has its own failure
identity** (a typed `ExitCode` + a context message).

Three ideas define it:

1. **Typed, student-facing failure identity** (§4, §10) — six phases (config, network,
   dataset, plugin, hardware, simulation, publish) each map to a distinct exit code and a
   message worded so a student can tell "my YAML is wrong" from "my controller crashed" from
   "the framework crashed". This is the module's dominant design concern.
2. **Runtime parameter channels** (§7) — `apply_qv_overrides` pushes the run's Q(V)
   breakpoints to *every* consumer at once (dry-run curve, coordinator sizing, Arduino CFG),
   and `apply_violation_limits` rebinds `violation_detector`'s constants process-wide (the
   late-binding partner to Doc 11).
3. **Dataset-faithful profile rebuilding** (§8) — `build_net_and_profiles` returns a
   `profile_factory` closure that rebuilds the HC-stressed net's profiles from the **same**
   dataset strategy as the outer run (the built-in-dataset analogue of 12C's
   `make_profile_factory`).

---

## 2. File logic flow — pseudocode

```
ExitCode(IntEnum); ExecutorError hierarchy (Config/NetworkLoad/Dataset/Plugin/Hardware/Publish)
STUDY_SCENARIOS (study → scenario list); presets; _ensure_framework_on_path

build_benchmark_config(plan, profile_factory, publish_fn) → BenchmarkConfig
apply_qv_overrides(plan)        — set_qv_parameters → dry-run curve + coordinator + Arduino CFG
apply_violation_limits(plan)    — reset_limits(); set_limit(thermal/angle/unbalance); rebind V_MIN/V_MAX
build_net_and_profiles(plan)    — resolve net (preset/plugin/custom-fn) + build_annual_profiles
                                  + profile_factory closure + scaling/window/resolution/focus-bus checks
validate_plan / check_hardware_port / _confirm_plugin_firmware — cross-field + hardware pre-run checks

FUNCTION execute(plan) → exit code:
    Phase 0: validate_plan; apply_qv_overrides; apply_violation_limits; hardware/firmware checks
    Phase 1: build_net_and_profiles
    Phase 2: PublishHandle(+hc); build_benchmark_config; save RunPlan copy
    publish_topology_and_profiles (non-fatal)
    Phase 3: register_and_run (plugin) OR run_benchmark  — typed exception → typed exit code
    Phase 4: publish_hc_and_comparison (+HC-stressed); summary table
    exit code (real scenario failures → SIMULATION_ERROR even if publish ok)
```

---

## 3. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `execute` | function | `__main__.py` — `sys.exit(execute(plan))` (9D). The public entry. |
| `build_benchmark_config` | function | `execute` Phase 2. |
| `apply_qv_overrides` / `apply_violation_limits` | functions | `execute` Phase 0. |
| `build_net_and_profiles` | function | `execute` Phase 1. |
| `validate_plan` / `check_hardware_port` / `_confirm_plugin_firmware` | functions | `execute` Phase 0. |
| `run_benchmark` / `register_and_run` | fns (9A/12B) | `execute` Phase 3. |
| `PublishHandle`, `publish_*` | 10A | Phases 2/4. |
| `set_qv_parameters` (vvc), `set_limit`/`reset_limits` (vd) | 1A/11 | the runtime channels. |

---

## 4. Section A — Exit codes & error hierarchy (L114–177)

```python
class ExitCode(enum.IntEnum):
    OK=0; CONFIG_ERROR; NETWORK_LOAD_ERROR; DATASET_ERROR; PLUGIN_ERROR;
    HARDWARE_ERROR; SIMULATION_ERROR; PUBLISH_ERROR; INTERRUPTED
class ExecutorError(Exception): exit_code; context   # + Config/NetworkLoad/Dataset/Plugin/Hardware/Publish subclasses
```

#### Line 114–177: The failure-identity vocabulary

`ExitCode` gives every failure phase a **distinct process exit code** (so a shell script or CI
can branch on *why* a run failed), and the `ExecutorError` hierarchy pairs each with a typed
exception carrying an `exit_code` and a human `context`. **Design rationale:** the executor's
users are students running their own plugins; a single generic "error" would leave them
guessing whether the fault was their YAML, their controller, the dataset, or the hardware.
Typed errors + distinct exit codes make the failure *self-locating*. Every subclass sets its
own `exit_code`, so `except ExecutorError` in `execute` can `return int(exc.exit_code)`
uniformly (§10).

---

## 5. Section B — Presets, studies & framework path (L191–393)

`_ensure_framework_on_path` (L194) puts the framework dir on `sys.path` (located via the
`benchmark_runner.py` sentinel) so the executor can `import benchmark_runner` etc. regardless
of CWD. `_preset_loaders` (L249) + the preset constants map friendly names
(`1-MV-rural--2-sw`, `cigre_mv_pv_wind`, …) to loaders. **`STUDY_SCENARIOS`** (L325) maps a
*study* to its scenario list (e.g. `opf_benchmark → [6]`, `full_comparison → [1..6]`,
`hosting_capacity`), and `_HC_STRESS_SCENARIOS` (L332) is the fixed set for the stressed
re-benchmark. `_check_opf_network_compatibility` (L335) guards the known SimBench-MV-OPF
incompatibility (6E) *before* the run. **Design rationale:** the study abstraction lets a
student pick "OPF benchmark" without knowing it's scenario 6, and the compatibility guard
turns a deep `runopp` failure into a clear pre-run message.

---

## 6. Section C — build_benchmark_config (L396–461)

Translates a `RunPlan` into a `BenchmarkConfig` (9A): `study → scenarios` (via
`STUDY_SCENARIOS`), `dry_run = not plan.hardware`, the port, `output_dir = <output>/<run_id>`,
`v_min`/`v_max` from the plan's parameters, and the HC flags (`run_hc` when the study is
`hosting_capacity`; `run_hc_scenarios` when `hc_stressed`; the `profile_factory` and
`_HC_STRESS_SCENARIOS` only then). **Design rationale — fail at the CLI boundary:** if
`hc_stressed` is requested but no `profile_factory` could be built for the dataset source, it
raises a `ConfigError` **naming the CLI-level cause** ("re-run with hc_stressed disabled, or
use a dataset that supports profile rebuilding") rather than letting `benchmark_runner`'s
lower-level `_validate_inputs` reject it with a framework-internal message.

---

## 7. Section D — Runtime channels (L468–554)

Two functions push a run's parameter overrides into the framework's process state — the
"runtime channels". These are the executor's most mechanism-heavy code.

### 7.1 apply_qv_overrides (L468–505)

```python
    if all(v is None for v in (q_ratio, u1..u4)): return          # nothing to override
    applied = vvc.set_qv_parameters(q_ratio, u1, u2, u3, u4)       # ValueError → ConfigError
    console.print("Q(V) for this run: q_ratio=… U1..U4=… (dry-run curve, coordinator sizing, Arduino CFG)")
```

**One call updates every Q(V) consumer consistently.** `set_qv_parameters` (Doc 1A) rebinds
the module-level breakpoints that the dry-run `QVCharacteristic`, the coordinator/dynamics
`q_max` sizing, *and* the Arduino CFG message all read — so the **Python/firmware Q(V) can
never diverge** (the KB §9.1 invariant). **Design rationale — validation before the run:**
an invalid combination (e.g. non-monotone U1..U4) is caught here as a `ConfigError`
*before* the run, mirroring the firmware's own `ERR:CFG_INVALID` matrix — never a silent
mis-control mid-run.

### 7.2 apply_violation_limits (L508–554) — the late-binding rebind

```python
    vd.reset_limits()                                              # clean baseline — no cross-run leak
    vd.set_limit("LINE_MAX_LOADING",  p.line_max_loading,  0.0, vd.THERMAL_LOADING_PLAUSIBLE_MAX, "%", warn=_warn, confirm=_confirm)
    vd.set_limit("TRAFO_MAX_LOADING", p.trafo_max_loading, 0.0, ..., "%", ...)
    vd.set_limit("VA_DIFF_MAX_DEGREE",    p.va_diff_max_degree,    0.0, 180.0, "deg", ...)
    vd.set_limit("UNBALANCE_MAX_PERCENT", p.unbalance_max_percent, 0.0, 100.0, "%",   ...)
    if p.v_min is not None and p.v_max is not None and p.v_min < p.v_max:
        vd.V_MIN, vd.V_MAX = float(p.v_min), float(p.v_max)
```

**This is the caller half of Doc 11's late-binding mechanism.** `detect_violations`' defaults
are late-bound to `violation_detector`'s module constants *precisely so this function can
rebind them* and have the new limits apply everywhere `detect_violations()` is called with
defaults — without threading kwargs through the whole call tree. **`reset_limits()` first** is
essential: it restores the import-time defaults so a *previous* run's overrides in a
long-lived process (notebook/server) cannot leak in via `set_limit`'s None-keeps-current
semantics. **Worked example:** with `plan.parameters.line_max_loading = 80`: `reset_limits`
sets `LINE_MAX_LOADING = 100`; `set_limit("LINE_MAX_LOADING", 80, 0, 300, …)` validates
`0 < 80 ≤ 300` → rebinds to 80 and prints a confirmation; thereafter every
`detect_violations()` flags lines above 80 %. With `line_max_loading = 500` (invalid, > 300):
`set_limit` warns and **keeps 100** (hardcoded default is the fallback, user input the
override). With `None`: no-op, stays 100. **Design rationale — the honest scope note
(docstring):** the function documents *exactly* where the limits land — the Scenario 4 HIL
loop, the coordinator, and HC (all via `detect_violations` defaults); Scenarios 1–3 count
*voltage* through their own inline `v_min`/`v_max` checks (kept consistent because those come
from the same plan), and their per-record loading fields follow these constants too; and
**`unbalance_max_percent` is applied but currently reaches no result** — no runner performs a
3-phase power flow, so `detect_violations_3ph` has no caller. Documenting a rebind that
"reaches no result" rather than hiding it is exactly the verification honesty this project
demands.

---

## 8. Section E — build_net_and_profiles & the profile_factory (L808–1033)

The network/profile resolution pipeline. It resolves the network from the plan's source
(a **preset** loader, a **network plugin** YAML via 12C, or a **custom function** via
`_import_custom_network_fn`), applies any network modifications (`_apply_network_modifications`),
then builds profiles via `build_annual_profiles` with `builder_kwargs` derived from the
dataset strategy (`simbench_native`/`dwd`/`custom` — with typed `DatasetError` on failure).
It then applies scaling (`_apply_scaling`), the time window (`_apply_time_window`), the
resolution check, and focus-bus validation, and returns the 4-tuple
`(net, profiles, network_id, profile_factory)`.

**The `profile_factory` closure (L1020–1023)** is the key output for HC:

```python
    def profile_factory(net_hc, _kw=dict(builder_kwargs)):
        kw = dict(_kw); kw["net_name"] = str(kw["net_name"]) + "_hc_stressed"
        return _apply_time_window(build_annual_profiles(net_hc, **kw), plan)
```

It **captures the outer run's exact `builder_kwargs`** (dataset strategy, data dir, year) and
rebuilds profiles for the HC-stressed net with the same strategy (plus the time window) —
the built-in-dataset analogue of 12C's `make_profile_factory`. Both feed the same
`BenchmarkConfig.profile_factory` slot that `run_benchmark`'s HC-stressed recursion consumes
(9A §9). **Design rationale:** the stressed re-benchmark must use the *same* weather/profile
basis as the outer run, or its comparison would be confounded by a profile change; capturing
`builder_kwargs` in the closure guarantees it.

---

## 9. Section F — Validation & hardware checks (L1051–1344)

`validate_plan` (L1091) performs the **cross-field** checks the wizard deliberately does not
(each wizard prompt validates its own field in isolation): `v_min < v_max`, dataset↔network
compatibility (e.g. `simbench_native` on a non-SimBench net), the OPF guard, custom-file
importability, focus-bus membership, and DWD-dir/station checks — a documented list (L1039–1049)
of exactly what it adds over the wizard. `check_hardware_port` (L1172) verifies the serial
port is present/openable for a hardware run (else `HardwareError`), and `_confirm_plugin_firmware`
(L1227) is the **interactive flash confirmation** — the second of the three hardware-plugin
failsafes (12B). `_build_summary` (L1288) and `_save_run_plan_copy` (L1327) assemble the final
summary and persist the exact `RunPlan` used (reproducibility). **Design rationale:**
validation lives here, *after* the wizard and *before* the run, because these are the checks
that need the resolved net/profiles or cross-field context the wizard's per-field prompts
can't see.

---

## 10. Section G — execute: the phased pipeline (L1346–1564)

The public entry — five phases, each with its own failure identity.

```
Phase 0 (config):   validate_plan; apply_qv_overrides; apply_violation_limits;
                    check_hardware_port; _confirm_plugin_firmware      → ExecutorError → typed code
Phase 1 (network):  build_net_and_profiles                            → typed / NETWORK_LOAD_ERROR
Phase 2 (config):   PublishHandle(+hc_handle); build_benchmark_config; save RunPlan copy
        publish_topology_and_profiles                                 → non-fatal (warn only)
Phase 3 (run):      register_and_run (plugin) OR run_benchmark
        FileNotFoundError/ValueError/ImportError/… → PLUGIN_ERROR ("fix the plugin YAML, before any sim")
        RuntimeError                               → SIMULATION_ERROR ("YOUR CUSTOM CONTROLLER CRASHED")
        Exception (last resort)                    → SIMULATION_ERROR ("SIMULATION CRASHED after validation")
Phase 4 (publish):  publish_hc_and_comparison (+ HC-stressed); summary → PUBLISH_ERROR ("CSV intact, JSON failed")
exit: real scenario failures (≠ "skipped") → SIMULATION_ERROR even if publish ok
```

#### Line 1346–1564: The phased execution

Each phase is a `try/except` that converts a failure into a **typed exit code + a
student-legible context**. The design's cleverness is in **distinguishing failure classes by
where they surface**: in Phase 3, a plugin that fails to *load* raises
`FileNotFoundError`/`ImportError`/… **before any scenario starts** → `PLUGIN_ERROR` ("fix the
YAML"); a plugin that *crashes inside* the isolated runner surfaces as `register_and_run`'s
`RuntimeError` → `SIMULATION_ERROR` ("YOUR CUSTOM CONTROLLER CRASHED — the config and network
were valid; the traceback is your code"); and anything else is the last-resort net
("SIMULATION CRASHED after validation — most likely a custom network breaking a pandapower
assumption, or the Arduino handshake failing mid-run"). **Worked example:** a student whose
`droop_controller.py` has a runtime `IndexError` gets `SIMULATION_ERROR` with the "your
controller crashed" context — not a generic stack trace they can't place. **Two more design
touches:** `publish_topology_and_profiles` runs **non-fatally** (a publish failure must not
abort a not-yet-started run), and the final exit code escalates to `SIMULATION_ERROR` when any
scenario **failed inside** an otherwise-successful run (the framework isolates failures, 9A,
so the table still renders — but the exit code surfaces the failure so a script notices).
`KeyboardInterrupt` is caught at every phase → `INTERRUPTED`.

---

## 11. Findings and flags

**No bugs found.** The module is exceptionally well-documented and honest about its own
scope. Points worth recording:

1. **Self-documented no-op: `unbalance_max_percent`.** `apply_violation_limits` rebinds it,
   but no runner performs a 3-phase power flow, so it currently reaches no result (the
   docstring says so). Not a bug — an honestly-flagged latent capability (it will take effect
   if a 3-phase scenario is ever added). Recorded rather than hidden.
2. **The scope note is the parametric-limits finding.** The docstring's precise account of
   where thermal vs voltage limits land (S4/coordinator/HC via `detect_violations`; S1–3
   voltage via inline checks; loading fields via the constants) matches the project's known
   parametric-limits scope — and is exactly the caller-side behaviour Doc 11 §10 describes.

**Not bugs (documented for clarity):** the typed-exit-code phasing is the module's central
design, not incidental; `reset_limits()`-first is required for process-reuse safety; the two
`profile_factory` sources (this closure for datasets, 12C for plugins) correctly feed one
config slot; the non-fatal topology publish is deliberate.

---

## 12. Coverage checklist

| Entity | Lines | Kind | Documented | Notes |
|--------|-------|------|:----------:|-------|
| `_configure_cli_logging` | 60–113 | helper | ✅ | session log |
| `ExitCode` + `ExecutorError` hierarchy | 114–177 | enum/classes | ✅ | failure identity |
| `_ensure_framework_on_path` / `_preset_loaders` / presets | 191–332 | helpers | ✅ | |
| `_check_opf_network_compatibility` | 335–393 | helper | ✅ | OPF guard |
| `build_benchmark_config` | 396–461 | function | ✅ | RunPlan → config |
| `apply_qv_overrides` | 468–505 | function | ✅ | Q(V) channel |
| `apply_violation_limits` | 508–554 | function | ✅ | **late-binding rebind** |
| `_import_custom_network_fn` / `_looks_like_pandapower_net` | 561–621 | helpers | ✅ | |
| `_apply_network_modifications` / `_resolve_scaling` / `_apply_scaling` | 622–753 | helpers | ✅ | |
| `_apply_time_window` / `_check_timestep_resolution` | 755–807 | helpers | ✅ | |
| `build_net_and_profiles` (+ `profile_factory`) | 808–1033 | function | ✅ | resolution pipeline |
| `_validate_focus_buses` / `_validate_dwd_dir` / `validate_plan` | 1051–1158 | validators | ✅ | cross-field |
| `_study_uses_hardware` / `check_hardware_port` / `_confirm_plugin_firmware` | 1159–1287 | hardware | ✅ | flash failsafe #2 |
| `_build_summary` / `_save_run_plan_copy` | 1288–1345 | helpers | ✅ | reproducibility |
| `execute` | 1346–1564 | function | ✅ | **phased pipeline** |

**Also covered (not standalone entities):** the typed failure-identity design, the two
runtime channels (Q(V) + violation limits), the late-binding rebind and its honest scope, the
dataset-faithful `profile_factory`, the wizard-vs-executor validation split, and the
failure-class-by-where-it-surfaces logic in `execute`.

**Flags raised:** none (no bugs). The `unbalance_max_percent` no-op is self-documented, not a
defect.

---

*End of Document 9B. Next: Document 9C — `wizard.py` (958 lines) + `run_plan.py` (466 lines):
the interactive configuration wizard and the `RunPlan`/`ParameterConfig` dataclasses it
produces (the input this executor consumes).*
