# TASK 4 — DOCUMENT 9D
## Orchestration Layer, Part 4: Entry Points & Helpers
### `__main__.py` (205) + `run_benchmark_script.py` (563) + `helpers.py` (210)
### Complete line-by-line walkthrough

> **Source verified:** read against the current `/mnt/project/__main__.py`,
> `run_benchmark_script.py`, and `helpers.py` (the project tree). **Read alongside:**
> **9B** (`executor.execute` — the wizard path's target), **9C** (`run_wizard`,
> `RunPlan` — consumed here), **9A** (`run_benchmark` — the direct-script path's
> target), **11**/**9B** (the `apply_violation_limits` asymmetry surfaced here).

---

## TABLE OF CONTENTS

1. [Overview — the two entry paths](#1-overview)
2. [File logic flow — pseudocode](#2-file-logic-flow--pseudocode)
3. [Call-site map](#3-call-site-map)
4. [PART 1 — __main__.py (the wizard/preset CLI entry)](#part-1--__main__py)
   - [Section A — Preset save (L28–48)](#4-section-a--preset-save-l2848)
   - [Section B — _validate_loaded_plan (L50–112)](#5-section-b--_validate_loaded_plan-l50112)
   - [Section C — _load_preset (L114–156)](#6-section-c--_load_preset-l114156)
   - [Section D — main (L159–205)](#7-section-d--main-l159205)
5. [PART 2 — run_benchmark_script.py (the direct-script path)](#part-2--run_benchmark_scriptpy)
   - [Section E — Args, logging & editable constants (L59–206)](#8-section-e--args-logging--editable-constants-l59206)
   - [Section F — Network & profile resolution (L208–300)](#9-section-f--network--profile-resolution-l208300)
   - [Section G — Scaling, DER/switch templates (L301–401)](#10-section-g--scaling-derswitch-templates-l301401)
   - [Section H — Config & run dispatch (L403–563)](#11-section-h--config--run-dispatch-l403563)
6. [PART 3 — helpers.py (console output)](#part-3--helperspy-l1210)
7. [Findings and flags](#13-findings-and-flags)
8. [Coverage checklist & Document 9 status](#14-coverage-checklist--document-9-status)

---

## 1. Overview

These three files are the **entry points** and their shared **console output**. The project
has **two ways in**, both landing on the same framework core:

```
INTERACTIVE:  python -m <pkg>  →  __main__.main()  →  run_wizard()/preset → RunPlan → executor.execute()  → run_benchmark()
DIRECT:       python run_benchmark_script.py [--network …] [--controller …]  →  (edit constants) → run_benchmark()/register_and_run()
```

- **`__main__.py`** is the **interactive CLI entry**: load a saved preset (with semantic
  re-validation) *or* run the wizard, confirm, optionally save a preset, then hand the
  `RunPlan` to `executor.execute` and propagate its typed exit code.
- **`run_benchmark_script.py`** is the **direct fast-iteration path**: an editable script with
  a network-selector block, DER-injection and switch-manipulation templates, and a
  `BenchmarkConfig` the researcher edits in place, then runs `run_benchmark` (or
  `register_and_run`) directly — bypassing the wizard/executor. It mirrors the executor's logic
  (PF-preserving scaling, the same `profile_factory`, the same run dispatch) but as code.
- **`helpers.py`** is the shared **Rich console output** (headers, the run-plan echo, error
  messages, the summary table) used by both the executor and the script.

**One asymmetry to flag (§13):** the direct-script path does **not** call
`apply_violation_limits` — so custom *thermal/angle* limits only take effect on the
wizard/executor path (the caller-property documented in Doc 11 §10 and 9B).

---

## 2. File logic flow — pseudocode

```
__main__.py:
  _save_preset(plan)          — RunPlan.to_dict → presets/<name>.json (sort_keys, indent=2)
  _validate_loaded_plan(plan) — SEMANTIC checks vs current codebase → warning list (no raise)
  _load_preset()              — read JSON → RunPlan.from_dict → validate → confirm-if-warnings
  main(): [load preset | run_wizard] → print_run_plan → confirm → [save preset] → sys.exit(execute(plan))

run_benchmark_script.py:  (top-level executable)
  argparse (--network, --controller, --q-ratio, --yes, …); editable constants (RUN_NAME, DER_*, LOAD_SCALE)
  resolve net+profiles (--network plugin  OR  network-selector block + build_annual_profiles)
  [optional] oversize/scale (p_mw & sn_mva together — PF-preserving); DER-injection & switch templates
  PublishHandle(+hc); BenchmarkConfig(edit in place); publish_topology_and_profiles
  if --controller: register_and_run  else: run_benchmark

helpers.py:  print_section_header / print_timestep_line / print_run_plan / print_error_message / print_summary_table
```

---

## 3. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `main` | function | `python -m <pkg>` (the `__main__` guard). The interactive entry. |
| `_load_preset`/`_save_preset`/`_validate_loaded_plan` | helpers | `main`. |
| `run_wizard` (9C), `executor.execute` (9B) | fns | `main`. |
| `run_benchmark_script.py` | script | `python run_benchmark_script.py …` — the direct path. |
| `run_benchmark` (9A), `register_and_run` (12B), `make_profile_factory` (12C), `PublishHandle`/`publish_*` (10A), `build_annual_profiles`, `plot_topology`/`plot_profiles` (10C) | fns | The script. |
| `helpers.print_*` | fns | `executor` (9B) + `__main__` + the script. |

---

# PART 1 — __main__.py

## 4. Section A — Preset save (L28–48)

```python
def _save_preset(plan):
    name = Prompt.ask("Preset name").strip()
    if not name: return   # empty → not saved
    (PRESET_DIR / f"{name}.json").write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
```

Serialises a `RunPlan` to `presets/<name>.json` via `plan.to_dict()` (9C), **sorted and
indented** for a human-diffable, version-controllable preset. **Design rationale:** a preset is
the reproducibility artefact — `sort_keys=True` makes two presets diff cleanly, and the
round-trip guarantee (9C) means the saved JSON reloads to an identical run.

## 5. Section B — _validate_loaded_plan (L50–112)

```python
    if source_type=="preset" and preset_name not in _preset_loaders(): warn("no longer exists")
    if source_type=="custom" and not Path(custom_path).exists():       warn("file gone")
    if source_type=="plugin" and not Path(plugin_path).exists():       warn("YAML gone")
    if switches_to_flip:  warn("verify indices still valid for the target network")
    if der_placements:    warn("verify those buses still exist")
    if hardware and not port: warn("hardware=True but no port")
    return warnings   # never raises
```

#### Line 50–112: Semantic re-validation of a loaded preset

A **third validation layer** distinct from `RunPlan.from_dict`'s *structural* safety (9C) and
the executor's *cross-field* checks (9B): this asks whether a preset saved **months ago still
makes sense against the current codebase** — the preset network still exists in the catalogue,
the custom/plugin file is still on disk, and (generically, without loading the net) the switch
indices and DER buses should be re-verified. **Design rationale (docstring):** `from_dict`
guarantees no `KeyError`/`TypeError`, but a stale preset can still reference a vanished file or
a network structure that changed; this returns **warnings, not exceptions**, so the caller
(`_load_preset`) can let the user decide. **Worked example:** a preset referencing
`custom_path="feeders/old.py"` that was since deleted → one warning "Custom network file … does
not exist at this path anymore"; the user is asked to proceed or abort.

## 6. Section C — _load_preset (L114–156)

Lists `presets/*.json`, loads the chosen one, reconstructs via `RunPlan.from_dict`, runs
`_validate_loaded_plan`, and — if there are warnings — asks the user to confirm before
returning the plan (else `None`). **Design rationale:** every failure mode (no presets, bad
name, unparseable JSON, declined warnings) returns `None` cleanly, so `main` can fall back to
the wizard.

## 7. Section D — main (L159–205)

```python
    if Confirm("Load a saved preset instead of the wizard?"): plan = _load_preset()
    if plan is None: plan = run_wizard()          # fallback
    print_run_plan(plan); if not Confirm("Proceed?"): sys.exit(0)
    if Confirm("Save as preset?"): _save_preset(plan)
    try: sys.exit(executor.execute(plan))         # propagate the TYPED exit code
    except KeyboardInterrupt: sys.exit(INTERRUPTED)
    except Exception: print_error_message(...); sys.exit(1)
```

#### Line 159–205: The interactive workflow

The entry orchestration: **preset-or-wizard** (a failed preset load falls back to the wizard),
echo the plan, confirm, optionally save, then **`sys.exit(executor.execute(plan))`** — crucially
**propagating the executor's typed exit code** (0/2/3/…/130, 9B) so shells and CI can branch on
the failure class. The surrounding `try/except` is explicitly only the net for a bug **in the
executor itself** (a normal run failure already returns a typed code, not an exception). **Design
rationale:** `main` is a thin conductor — all real work (validation, execution, error identity)
lives in `run_wizard`/`execute`; `main` just sequences them and forwards the exit code.

---

# PART 2 — run_benchmark_script.py

## 8. Section E — Args, logging & editable constants (L59–206)

`_PeriodicFilter` (L59) throttles per-timestep log lines to every N (default 96 = daily at
15-min). The **argparse** interface (L86–166) exposes `--network <yaml>` (plugin network),
`--controller <yaml>` (controller plugin), `--q-ratio`, `--yes` (skip validation prompts), and
others. The **editable module constants** (`_Q_RATIO` from `--q-ratio` or 0.25, `RUN_NAME`,
`OVERSIZE_FACTOR`, `LOAD_SCALE`, `DER_P_MW`, `DER_SN_MVA`) are the script's "control panel" — a
researcher edits these at the top and runs. **Design rationale:** the script mixes a small CLI
surface (the two plugin paths + a few knobs, so `--network`/`--controller` runs need no editing)
with in-file constants (for the deeper edits the wizard can't express) — the fast-iteration
counterpart to the wizard's structured prompts.

## 9. Section F — Network & profile resolution (L208–300)

```python
    if _args.network:                                    # PLUGIN path
        net, profiles = load_network_from_yaml(_args.network)     # net + profiles from the YAML (12C)
        _warnings = validate_network_plugin(net, profiles); prompt-to-proceed unless --yes
    else:                                                # NETWORK-SELECTOR block (edit only this)
        net = sb.get_simbench_net("1-MV-rural--2-sw"); net_name = "1-MV-rural--2-sw"; sb_code = "…"
        # …8 commented alternatives: Synthetic LV, CIGRE MV/LV, Kerber, Dickert, custom…
    if not _args.network:
        profiles = build_annual_profiles(net, net_name=net_name, data_dir=…/dwd, simbench_code=sb_code)
```

#### Line 208–300: Two network sources — plugin or the selector block

The `--network` plugin path calls `load_network_from_yaml` (12C — returns *both* net and
profiles) and echoes `validate_network_plugin`'s warnings (prompting unless `--yes`); otherwise
the **network-selector block** is the *one place the researcher edits*, with the primary
SimBench MV network active and **eight commented alternatives** (Synthetic LV suburb/village,
CIGRE MV/LV, Kerber, Dickert, a custom SimBench code) each carrying the `net`/`net_name`/`sb_code`
triple. `net_name` is load-bearing: it routes `build_annual_profiles`'s strategy (a name
containing "simbench"/"cigre" selects that path, else the DWD fallback — the `profile_builder`
name-routing the KB documents). **Design rationale:** the commented block is a menu-as-code — the
researcher uncomments one network rather than typing a wizard answer.

## 10. Section G — Scaling, DER/switch templates (L301–401)

The optional **oversize/scaling** block scales PV/wind `p_mw` **and** `sn_mva` **together by the
same factor**, so the power factor stays at the network's native value instead of drifting —
the docstring notes this **mirrors `executor.py`'s `_apply_scaling`/`der_scaling` exactly**
(9B). The **DER-injection template** (L356–376, commented) is the canonical
`pn.create_sgen(bus, p_mw, sn_mva, type="PV")` pattern at bus indices read from the labelled
topology plot — the same operation `NetworkConfig.der_placements` encodes (9C). The
**switch-manipulation template** (L377–401, commented) documents the `net.switch` columns and
the print→toggle→verify workflow — the same operation as `NetworkConfig.switches_to_flip`.
**Design rationale:** the script *is* the reference implementation these `NetworkConfig` fields
point at (9C's field comments say "the same pattern documented in run_benchmark_script.py") —
so the wizard path and the script path apply identical network modifications, one declaratively,
one by editing.

## 11. Section H — Config & run dispatch (L403–563)

```python
    handle = PublishHandle(output_dir=…/(net_name+RUN_NAME), update_every_k=6)   # + hc_stressed handle
    config = BenchmarkConfig(scenarios=[], dry_run=True, v_min=vd.V_MIN, v_max=vd.V_MAX,
        run_hc=True, run_hc_scenarios=False, hc_stress_scenarios=[1,2,3,4,5,10],
        profile_factory = make_profile_factory(_args.network) if _args.network else (lambda net_hc: build_annual_profiles(...)),
        publish_fn=handle, hc_publish_fn=handle_hc_stressed)
    publish_topology_and_profiles(net_os, profiles, …)
    if _args.controller: register_and_run(_args.controller, net_os, profiles, benchmark_config=config, …)
    else:                run_benchmark(net_os, profiles, network_id=net_name, config=config)
```

#### Line 403–563: The editable config and the run dispatch

The `PublishHandle` (live + HC-stressed streams, 10A), the **editable `BenchmarkConfig`**
(`scenarios=[]` meaning all six; the HC knobs; the `profile_factory` chosen by whether
`--network` is set — `make_profile_factory` for a plugin, else the DWD `build_annual_profiles`
lambda, exactly the two `profile_factory` sources of 9A/9B), a scaling-verification print,
`publish_topology_and_profiles`, then the **run dispatch** identical to the executor's Phase 3:
`register_and_run` when `--controller` is set (the custom controller runs *alongside* the
built-ins), else `run_benchmark`. **The key omission (§13):** `config.v_min/v_max` are set from
`vd.V_MIN/V_MAX`, but this path **never rebinds** the thermal/angle limit constants (no
`apply_violation_limits`), so those stay at framework defaults here. **Design rationale:** the
script deliberately reaches the same `run_benchmark`/`register_and_run` core the executor does —
it is a *thinner* path (no typed-exit-code phasing, no wizard) for a researcher who wants to edit
and run, accepting that a few executor-only conveniences (custom thermal limits, typed errors)
are not wired.

---

# PART 3 — helpers.py (L1–210)

Five Rich-console output functions shared across the entry points:

- **`print_section_header(run_plan, stage)`** (L16) — the stage banner (e.g. "Loading network
  and building profiles"), tagging output with the run context.
- **`print_timestep_line(t, algo, n_ov, n_uv, n_overload)`** (L37) — a per-timestep progress
  line (violation counts), throttled by the caller.
- **`print_run_plan(run_plan)`** (L60) — the **human-readable echo** of a `RunPlan` before a run
  (study, network, dataset, parameters, hardware) — what `main` shows for the "Proceed?"
  confirmation.
- **`print_error_message(error, context)`** (L171) — the typed-error renderer: the `context`
  (from an `ExecutorError`, 9B) plus the exception, formatted so the failure phase is legible.
- **`print_summary_table(summary)`** (L185) — the final results table renderer.

**Design rationale:** centralising console output in one module keeps the executor, the entry,
and the script visually consistent (the same headers, colours, and error framing) and keeps the
logic modules free of presentation code. **No standalone algorithmic content** — these are
presentation helpers, documented at function granularity for coverage.

---

## 13. Findings and flags

**Flagged, not edited:**

1. **Duplicate `print_run_plan(plan)` in `main` (L172 and L174).** The run-plan echo is printed
   **twice** before the "Proceed?" prompt — harmless (cosmetic double output), an obvious
   copy-paste slip. **Recommend:** delete one of the two lines.
2. **Direct-script thermal-limit asymmetry (documented, not a defect).**
   `run_benchmark_script.py` sets `config.v_min/v_max` from the `violation_detector` module
   constants but **never calls `apply_violation_limits`**, so on this path the *thermal/angle/
   unbalance* limits stay at framework defaults — custom thermal limits require the
   wizard/executor path (Doc 11 §10, 9B). Worth surfacing to the user: **YAML/parameter thermal
   overrides do nothing on the `run_benchmark_script.py` path**; use the CLI wizard for those.

**Not bugs (documented for clarity):** the two-entry-path design is intentional (interactive vs
fast-iteration); the network-selector "menu as code" is the script's deliberate edit surface;
the script reaching the same `run_benchmark`/`register_and_run` core as the executor is by
design (a thinner path, not a divergent one); the three validation layers (structural
`from_dict` / semantic `_validate_loaded_plan` / cross-field `validate_plan`) are complementary,
not redundant.

---

## 14. Coverage checklist & Document 9 status

| Entity | File · Lines | Kind | Documented | Notes |
|--------|--------------|------|:----------:|-------|
| `_save_preset` | __main__ 30–48 | fn | ✅ | preset persistence |
| `_validate_loaded_plan` | __main__ 50–112 | fn | ✅ | semantic re-validation |
| `_load_preset` | __main__ 114–156 | fn | ✅ | load + confirm |
| `main` | __main__ 159–205 | fn | ✅ | entry; propagates exit code; **dup print flagged** |
| `_PeriodicFilter` | script 59–83 | class | ✅ | log throttle |
| argparse + editable constants | script 86–206 | — | ✅ | CLI surface + control panel |
| network/profile resolution | script 208–300 | — | ✅ | plugin vs selector block |
| scaling + DER/switch templates | script 301–401 | — | ✅ | PF-preserving; the reference for `NetworkConfig` mods |
| config + run dispatch | script 403–563 | — | ✅ | editable config; **limits asymmetry flagged** |
| `print_section_header`/`print_timestep_line`/`print_run_plan`/`print_error_message`/`print_summary_table` | helpers 16–210 | fns | ✅ | console output |

**Also covered (not standalone entities):** the two entry paths and how they reach one core,
the three validation layers, the "menu as code" selector, the PF-preserving scaling, the
script-as-reference for the `NetworkConfig` DER/switch fields, and the thermal-limit asymmetry.

**Flags raised (not edited):** duplicate `print_run_plan` (cosmetic); direct-script thermal-limit
asymmetry (documented behaviour, surfaced for the user).

### Document 9 status

Docs **9A** (`benchmark_runner`), **9B** (`executor`), **9C** (`wizard` + `run_plan`), and
**9D** (this doc — `__main__` + `run_benchmark_script` + `helpers`) are complete. **Remaining:
9E** — the CLI support modules (`network_catalogue.py` 258, `_data.py` 448, `_console.py` 29,
`test_helpers.py` 117), which closes Document 9 and the Task 4 walkthrough set.

---

*End of Document 9D. Next: Document 9E — the CLI support modules, closing Document 9.*
