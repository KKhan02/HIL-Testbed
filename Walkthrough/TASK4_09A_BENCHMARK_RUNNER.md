# TASK 4 — DOCUMENT 9A
## Orchestration Layer, Part 1: The Benchmark Runner
### `benchmark_runner.py` — the scenario orchestrator (1204 lines)
### Complete line-by-line walkthrough

> **Source verified:** read against the current `/mnt/project/benchmark_runner.py`
> (1204 lines, the project tree). **This is the orchestration hub** — every scenario
> doc's call-site map points here; it dispatches the six runners, wires the publisher
> and checkpointing, isolates failures, and drives hosting-capacity analysis.
>
> **Document 9 covers the orchestration/CLI layer** (~6,000 lines) and splits into:
> - **9A (this document): `benchmark_runner.py`** — the orchestrator.
> - **9B: `executor.py`** — the CLI run executor (1564 lines).
> - **9C: `wizard.py` + `run_plan.py`** — the interactive wizard + the run plan.
> - **9D: `run_benchmark_script.py` + `helpers.py` + `__main__.py`** — the direct-script
>   entry + shared helpers + CLI entry.
> - **9E: `network_catalogue.py` + `_data.py` + `_console.py` + `test_helpers.py`** —
>   the CLI support modules.
>
> **Read alongside:** the scenario runners **6A-iii–6E** (dispatched here), **10A**
> (`publisher` — `publish_scenario_result`, `PublishHandle`), **11**
> (`violation_detector` — the shared limits), **12B/12C** (`register_and_run` recurses
> into `run_benchmark`; `profile_factory` = `make_profile_factory`).

---

## TABLE OF CONTENTS

1. [Overview — the orchestration hub](#1-overview)
2. [File logic flow — pseudocode](#2-file-logic-flow--pseudocode)
3. [Call-site map](#3-call-site-map)
4. [Section A — ScenarioSpec & SCENARIO_REGISTRY (L117–162)](#4-section-a--scenariospec--scenario_registry-l117162)
5. [Section B — BenchmarkConfig & BenchmarkResult (L169–309)](#5-section-b--benchmarkconfig--benchmarkresult-l169309)
6. [Section C — Input helpers & _build_kwargs (L399–530)](#6-section-c--input-helpers--_build_kwargs-l399530)
7. [Section D — Comparison table & summary (L538–895)](#7-section-d--comparison-table--summary-l538895)
8. [Section E — run_benchmark: the loop & two-layer resume (L900–1097)](#8-section-e--run_benchmark-the-loop--two-layer-resume-l9001097)
9. [Section F — HC analysis & the recursive re-benchmark (L1098–1204)](#9-section-f--hc-analysis--the-recursive-re-benchmark-l10981204)
10. [Findings and flags](#10-findings-and-flags)
11. [Coverage checklist](#11-coverage-checklist)

---

## 1. Overview

`benchmark_runner.py` runs all (or a subset of) the scenarios on one network and
assembles the comparison. It is the **hub** everything else plugs into: the six runners
(via the registry), the publisher (per-scenario JSON + checkpoint archival), the
checkpointing/resume machinery, and hosting-capacity analysis.

Five ideas define it:

1. **The registry** (§4) — `SCENARIO_REGISTRY` maps a number → `ScenarioSpec` (id, label,
   runner, capability flags). It is the single dispatch table; `_build_kwargs` reads the
   capability flags to inject the right kwargs per scenario.
2. **Fail-isolated execution** (§8) — each scenario runs in a `try/except` (with the
   `deepcopy` *inside* the try), so one scenario's failure becomes a "failed" row, never an
   aborted benchmark.
3. **Two-layer resume** (§8) — **Layer 1** skips a whole scenario if its
   `scenarios/<id>.json` already exists (a prior attempt finished it); **Layer 2** (inside
   each runner) resumes a scenario that crashed partway via the per-timestep checkpoint.
4. **Fair comparison** — the same `v_min`/`v_max` go to every runner, scenarios always run
   in ascending order, and LV networks skip DER-requiring scenarios (`supports_lv=False`)
   unless DERs are actually present.
5. **Hosting capacity + recursion** (§9) — after the scenarios, it runs baseline/volt-var HC
   and can **recursively re-benchmark the HC-stressed network** (with a guard against
   infinite recursion), using the `profile_factory` (12C) to rebuild profiles for the
   stressed net.

---

## 2. File logic flow — pseudocode

```
ScenarioSpec (num, id, label, runner, supports_hardware/opf_verbose/coordination/lv)
SCENARIO_REGISTRY = {1:baseline, 2:oltc, 3:svc, 4:volt_var_local(4A), 5:volt_var_coord(4B), 6:opf}
BenchmarkConfig (scenarios, dry_run, port, output_dir, checkpointing, live_csv, HC knobs, publish_fn)
BenchmarkResult (results{n: ScenarioResult|dict|None}, errors, comparison_df, hc_*, net_hc)

_build_kwargs(spec, config): v_min/v_max/publish_fn/enable_checkpointing to all;
    + dry_run/port/coordination (S4); + verbose_opf (S5/6)

FUNCTION run_benchmark(net, profiles, network_id, config):
    validate; is_lv; has_ders
    define _rewrite_live_csv (merge completed + partial → CSV)
    FOR n in sorted(scenarios):
        LV-skip if !supports_lv and no DERs
        LAYER 1: scenarios/<id>.json exists → load summary dict into results[n]; skip
        TRY: net_copy = deepcopy; kwargs = _build_kwargs (+ live_csv_rewrite_fn); result = spec.runner(...)
             publish_scenario_result; archive checkpoint → .completed
        EXCEPT: errors[n]=traceback; results[n]=None; continue (no re-raise)
    if run_hc: [baseline_hc, volt_var_hc], net_hc   (isolated)
    if run_hc_scenarios and net_hc: profiles_hc = profile_factory(net_hc);
        recursive run_benchmark(net_hc, ..., run_hc=False, run_hc_scenarios=False)   # recursion guard
    comparison_df; write CSV; print summary; return BenchmarkResult
```

---

## 3. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `run_benchmark` | function | `executor` (9B), `run_benchmark_script` (9D), `plugin_runner.register_and_run` (12B — recursively), and `run_benchmark` itself (the HC-stressed recursion). |
| `SCENARIO_REGISTRY` / `ScenarioSpec` | table/dataclass | `plugin_runner` (allocates a plugin num, appends a spec); `_build_kwargs`/the loop. |
| `BenchmarkConfig` / `BenchmarkResult` | dataclasses | The public config/return; built by the CLI (9B/9C) and callers. |
| `_build_kwargs` | helper | The loop — per-scenario kwargs. |
| the six `run_scenario_*` runners | functions | Referenced by the registry (Docs 6A-iii–6E). |
| `publish_scenario_result`, `publish_topology_and_profiles` | fns (10A) | Per-scenario + HC-stressed publishing. |
| `run_baseline_hc`, `run_hc_with_volt_var` | fns (`hosting_capacity`) | The HC block. |

---

## 4. Section A — ScenarioSpec & SCENARIO_REGISTRY (L117–162)

```python
@dataclass(frozen=True)
class ScenarioSpec:
    num; scenario_id; label; runner
    supports_hardware=False; supports_opf_verbose=False; supports_coordination=False; supports_lv=True

SCENARIO_REGISTRY = {1:baseline, 2:oltc, 3:svc(lv=F),
    4:volt_var_local(hw,coord=F,lv=F), 5:volt_var_coord(hw,coord=T,lv=F), 6:opf(opf_verbose,lv=F)}
```

#### Line 117–162: The dispatch table

`ScenarioSpec` is a frozen descriptor: the scenario number, its `scenario_id` (**which must
match the string the runner passes to `from_records`**, so the registry and the published
JSON agree), its display label, the runner callable, and four **capability flags**. The
registry's six entries are the canonical scenario set: **4A** (`volt_var_local`,
`coordination=False`) and **4B** (`volt_var_coord`, `coordination=True`) are *both*
`run_scenario_4` distinguished only by the flag; **6** is `run_scenario_5` (OPF). **Design
rationale — flags, not signature inspection:** `_build_kwargs` gates kwargs on these flags
rather than introspecting runner signatures at runtime, so passing `dry_run`/`port` to a
scenario that doesn't accept them (which would `TypeError`) is structurally prevented.
`supports_lv=False` on 3/4/5/6 marks the DER/OPF scenarios skipped on DER-free LV networks.

---

## 5. Section B — BenchmarkConfig & BenchmarkResult (L169–309)

**`BenchmarkConfig`** (L169–253) is the keyword-only run config. Beyond the obvious
(`scenarios`, `dry_run`, `port`, `output_dir`, `write_csv`, `v_min`/`v_max`, `verbose_opf`),
its documented fields encode the whole feature surface: **`enable_checkpointing`** (with the
detailed force-clean semantics), **`live_csv_path`** (the in-place partial CSV), the **HC
knobs** (`run_hc`, `run_hc_scenarios`, **`profile_factory`** = `make_profile_factory` from
12C, `hc_stress_scenarios`, `hc_publish_fn`), and **`publish_fn`** (the `PublishHandle`).
**Design rationale — `kw_only=True`:** every field is keyword-only so a long config is
constructed unambiguously and a new field never shifts positional meaning.

**`BenchmarkResult`** (L260–309) is the return. Its key subtlety is documented on the
`results` field: **`results[n]` is a `ScenarioResult` for a scenario that ran, a plain
summary-dict for one skipped by Layer-1 "already complete" (reloaded from JSON), or `None`
for a failure** — the tri-state the downstream `_ok_summary_row` handles with an `isinstance`
check (and the same shape `plugin_runner` reconstructs, 12B). It also carries `errors`
(tracebacks), `comparison_df`, `hc_results`/`hc_error`, the recursive `hc_benchmark`, and
`net_hc` (the HC-stressed net for export).

---

## 6. Section C — Input helpers & _build_kwargs (L399–530)

#### Line 410–485: _validate_inputs & _is_lv_network

`_validate_inputs` checks the config and profile keys up front — every requested scenario
number exists in `SCENARIO_REGISTRY`, and `profiles` carries the keys the runners need
(`load`/`pv`/`wind`/`times`) — so a bad request fails fast with a named message instead of a
`KeyError` mid-run. `_is_lv_network` infers the distribution level from the **modal `vn_kv`**
of `net.bus`. **Derivation — why the mode, not the min/max:** an LV feeder still has one HV
slack bus (e.g. 20 kV) and its MV/LV transformer's HV winding; taking the *most common*
voltage level ignores those few high-voltage buses and reports the level the *bulk* of the
network sits at. **Worked example:** a CIGRE LV net with 40 buses at 0.4 kV and 1 slack at
20 kV → mode = 0.4 kV → `is_lv = True`; a SimBench MV net (buses at 20 kV, one 110 kV slack)
→ mode = 20 kV → `is_lv = False`. Same inference as `hosting_capacity`, so the two agree on
what "LV" means.

#### Line 499–530: _build_kwargs — the capability-gated kwargs builder

```python
    kwargs = {"network_id", "v_min", "v_max"}
    if spec.supports_hardware: kwargs += {"dry_run", "port", "coordination"}   # S4 (4A/4B)
    if spec.supports_opf_verbose: kwargs += {"verbose_opf"}                     # S5/6
    kwargs["publish_fn"] = config.publish_fn          # ALL runners (None = no-op)
    kwargs["enable_checkpointing"] = config.enable_checkpointing               # ALL runners
```

`v_min`/`v_max`/`publish_fn`/`enable_checkpointing` go to **every** runner (the fair-limits
and streaming/checkpointing contract); the capability flags gate the rest. **Worked example —
building kwargs for scenario 5 (4B, `volt_var_coord`):** its spec has
`supports_hardware=True`, `supports_coordination=True`, `supports_opf_verbose=False`, so
`_build_kwargs` produces `{network_id, v_min, v_max, dry_run, port, coordination=True,
publish_fn, enable_checkpointing}` — note `coordination` is read from **`spec.supports_coordination`**,
which is what makes 4B coordinated and 4A (same runner, `supports_coordination=False`) local.
**Contrast — scenario 6 (OPF):** `supports_hardware=False`, `supports_opf_verbose=True` →
`{network_id, v_min, v_max, verbose_opf, publish_fn, enable_checkpointing}` — no
`dry_run`/`port`/`coordination` (OPF has no hardware path), which is exactly why gating on
flags (not signature introspection) prevents a `TypeError` from passing `port` to
`run_scenario_5`. **Note (resolved concern):** `publish_fn` is injected **unconditionally** —
now correct because **all six runners accept it** (baseline included, 6A-iii; OPF's old
"deliberately ignored" workaround is gone, 6E). `live_csv_rewrite_fn` is **not** built here —
it's injected per-scenario in the loop (§8) because it must close over the current scenario
number.

---

## 7. Section D — Comparison table & summary (L538–895)

The reporting machinery that turns the per-scenario results into the comparison table.

#### Line 538–679: The summary-row builders & _build_comparison_df

```python
    _ok_summary_row(n, result):        # result is ScenarioResult OR reloaded dict
        d = result.summary_dict() if isinstance(result, ScenarioResult) else result
        return {"scenario", "status":"ok", "violations", "vdi_pct", "curtailed_mwh", ...}
    _failed_summary_row(n):  {"scenario", "status":"failed", <numeric cols>: nan}
    _skipped_summary_row(n): {"scenario", "status":"skipped", <numeric cols>: nan}
    _build_comparison_df(rows): DataFrame(rows)[_COMPARISON_COLS]   # fixed column order
```

Each scenario contributes **exactly one row**. The OK builder **`isinstance`-gates** the
tri-state result (§5): a `ScenarioResult` yields its `summary_dict()`; a reloaded dict (Layer-1
"already complete") is used directly — the same summary shape either way. Failed and skipped
scenarios get rows too, with `status` set and the numeric columns `NaN`. **Worked example — a
run where SVC failed on an LV net:** the rows are baseline `status=ok`, oltc `ok`, svc
`skipped` (LV, `supports_lv=False`), volt_var_local `ok`, volt_var_coord `ok`, opf `skipped`
(LV) — six rows, stable shape, the two skips clearly marked. **Design rationale — one row per
scenario, always:** the comparison table and CSV have a **fixed shape** regardless of which
scenarios ran, so downstream tooling (the dashboard, `plot_results` fig14) can rely on the
columns.

#### Line 685–895: _write_csv & the console renderers

`_write_csv` (L685) writes the timestamped comparison CSV (the durable artefact — the run's
headline result). `_fmt_cell`/`_print_summary_rich`/`_print_summary_plain` render the console
table: a **Rich** table when the library is available, a plain-text fallback otherwise
(`_print_summary_plain`), so the summary always prints even in a minimal environment.
**Design rationale:** the CSV is the machine artefact (stable columns), the console table is
the human artefact (formatted, colourised) — separated so neither compromises the other.

---

## 8. Section E — run_benchmark: the loop & two-layer resume (L900–1097)

The orchestration loop. Each scenario, in ascending order:

**LV skip (L997).** `is_lv and not supports_lv and not _has_ders` → skip (an LV network with
no DERs has nothing for a DER/OPF scenario to act on) — but an HC-stressed LV net *with* added
DERs (`_has_ders`) is **not** skipped.

**Layer 1 — whole-scenario resume (L1019–1038).** If `scenarios/<id>.json` already exists in
the publish output dir, the scenario finished in a prior attempt: load its **summary dict**
into `results[n]` and skip the runner entirely (with a re-run fallback if the JSON is
corrupt). **Design rationale:** this sits *above* the runners' per-timestep checkpoint (Layer
2) — Layer 1 avoids re-running a *completed* scenario; Layer 2 resumes a *crashed* one.

**Worked example — a crash on the RPi partway through scenario 4, then a re-launch:** first
attempt completes baseline and oltc (`scenarios/baseline.json`, `scenarios/oltc.json` written;
their checkpoints archived to `.completed`), then dies at timestep 21,000 of `volt_var_local`
(leaving `checkpoint/volt_var_local.jsonl` with 21,000 records, no `scenarios/volt_var_local.json`).
On re-launch: baseline and oltc hit **Layer 1** (their JSON exists → loaded as summary dicts,
runners skipped); `volt_var_local` has **no** `scenarios/*.json` so it runs — but its runner's
**Layer 2** `get_resume_records` finds the 21,000-record checkpoint, re-seeds the DER dynamics
from record 20,999, and continues from timestep 21,000 (not 0). The two layers compose: whole
completed scenarios are skipped, the one crashed scenario resumes mid-stream — no work is
repeated.

**Isolated execution (L1045–1097).** The `deepcopy(net)` is **inside** the `try` (a copy
failure → failed row, not abort); `_build_kwargs` builds the kwargs, and **`live_csv_rewrite_fn`
is injected here** as a per-scenario lambda binding `_n=n` (so the callback rewrites the CSV
with *this* scenario's partial state, §7's `_rewrite_live_csv` merging completed + partial).
On success, `publish_scenario_result` writes the final JSON and the **checkpoint is archived**
(`checkpoint/<id>.jsonl` → `.jsonl.completed`, L1080–1087) — renamed not deleted, so
`get_resume_records` can't mistake a finished scenario's checkpoint for a live one while the
raw stream survives for audit (the `.completed` fallback, 10A). On exception, the traceback is
stored, `results[n]=None`, and the loop **continues without re-raising** (the isolation
guarantee).

---

## 9. Section F — HC analysis & the recursive re-benchmark (L1098–1204)

**HC analysis (L1107–1129).** When `run_hc`, it runs `run_baseline_hc(net)` and
`run_hc_with_volt_var(net)` → `[baseline, volt_var]` `HCResult`s + `net_hc` (the net loaded to
baseline HC capacity), logging the headline **gain** (volt_var − baseline MW). Wrapped in
`try/except` so an HC failure is recorded (`hc_error`) but does not lose the scenario results.

**The recursive HC-stressed re-benchmark (L1131–1178).** When `run_hc_scenarios` and `net_hc`
exists: `profile_factory(net_hc)` (12C `make_profile_factory`) rebuilds profiles for the
stressed net, topology/profiles are published to `hc_publish_fn`'s dir, and **`run_benchmark`
calls itself** on the stressed net with a config whose **`run_hc=False` and
`run_hc_scenarios=False`** — the **recursion guard** that stops it re-entering HC (one level of
recursion only). `hc_stress_scenarios` (or the outer list) selects which scenarios re-run.
**Design rationale:** this measures how the control schemes perform on a network deliberately
loaded to its hosting-capacity limit — a stress test that reuses the entire benchmark machinery
via one guarded recursive call. Finally, `_build_comparison_df` + `_write_csv` +
`_print_summary_rich` finalise, and a fully-populated `BenchmarkResult` is returned.

---

## 10. Findings and flags

**No bugs found.** The module is meticulously documented and the one historical concern is
resolved:

1. **`publish_fn` unconditional injection is now correct.** Earlier, OPF didn't accept
   `publish_fn` (the "deliberately ignored" workaround); all six runners now accept it (6A-iii,
   6E), so `_build_kwargs` forwarding it to every runner is correct, not a latent `TypeError`.

**Not bugs (documented for clarity):** the two-layer resume (Layer 1 whole-scenario /
Layer 2 per-timestep) is a deliberate, clean architecture; the checkpoint archive-not-delete
is intentional (audit + resume-safety); the `deepcopy`-inside-`try` is the isolation
mechanism; the recursion guard (`run_hc=False`/`run_hc_scenarios=False` in the stressed
config) correctly bounds recursion at one level; `results[n]` being tri-state
(`ScenarioResult`/dict/None) is handled by the `isinstance`-gated summary rows.

---

## 11. Coverage checklist

| Entity | Lines | Kind | Documented | Notes |
|--------|-------|------|:----------:|-------|
| `ScenarioSpec` | 117–149 | dataclass | ✅ | capability flags |
| `SCENARIO_REGISTRY` | 151–162 | table | ✅ | 6 scenarios (4A/4B/OPF) |
| `BenchmarkConfig` | 169–253 | dataclass | ✅ | full feature surface |
| `BenchmarkResult` | 260–309 | dataclass | ✅ | tri-state `results[n]` |
| `_COMPARISON_COLS` / `_NUMERIC_COLS` | 318–394 | constants | ✅ | table schema |
| `_safe_filename` / `_validate_inputs` / `_is_lv_network` | 399–496 | helpers | ✅ | |
| `_build_kwargs` | 499–530 | helper | ✅ | capability-gated |
| `_ok/_failed/_skipped_summary_row` / `_build_comparison_df` | 538–679 | helpers | ✅ | one row per scenario |
| `_write_csv` / `_fmt_cell` / `_print_summary_rich` / `_print_summary_plain` | 685–895 | helpers | ✅ | |
| `run_benchmark` | 900–1204 | function | ✅ | orchestration + resume + HC |
| — `_rewrite_live_csv` (nested) | 985–992 | fn | ✅ | merge completed + partial |
| — loop + Layer-1 resume + isolation | 994–1097 | — | ✅ | |
| — HC + recursive re-benchmark | 1098–1178 | — | ✅ | recursion guard |
| — finalise + return | 1179–1204 | — | ✅ | |

**Also covered (not standalone entities):** the registry-as-dispatch design, the two-layer
resume architecture, the fail-isolation guarantee, the per-scenario `live_csv_rewrite_fn`
injection, the checkpoint archive-not-delete, the fair-limits contract, the LV/DER skip logic,
and the guarded HC-stressed recursion.

**Flags raised:** none (no bugs; the `publish_fn` concern is resolved).

---

*End of Document 9A. Next: Document 9B — `executor.py` (1564 lines): the CLI run executor —
`apply_violation_limits`, the run assembly, and the execution of a `RunPlan`.*
