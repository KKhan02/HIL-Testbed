# Diagram ↔ File/Function Map — HIL Testbed Flowcharts (Task 1a)

Verified against the live `/mnt/project` tree and `/mnt/user-data/outputs`:
**26 SVG/PDF diagrams produced**, against **42 `.py` + 1 `.ino` + 6 `.yaml`**
source files. Every claim below was cross-checked file-by-file, not from a
summary.

Each diagram exists as three artefacts in `outputs/`: `<name>.svg`,
`<name>.pdf`, and the regeneratable `<name>.py` Graphviz generator
(generators are grouped: `flow_s1.py`…`flow_s5.py`, `flow_plugin.py`,
`flow_orch.py`, `flow_profile.py`, `flow_netload.py`, `flow_catalogue.py`,
`flow_publisher.py`, `flow_hc.py`, `flow_cli.py`; the `flow_s4.py` generator
also emits both s4 diagrams, etc.).

---

## 1. Produced diagrams (26) → files & functions depicted

| # | Diagram (`flow_*`) | Primary file | Function(s) / entry depicted | Secondary files shown in-lane |
|---|---|---|---|---|
| 1 | `s1_init` | `scenario_1_baseline.py` | scenario-1 setup: `DFData`, `ConstControl`, `OutputWriter` build | `scenario_result.py` (`oversize_inverters`, `adapt_profiles`) |
| 2 | `s1_exec` | `scenario_1_baseline.py` | `run_timeseries` sweep (`_timed_runpp`) + separate post-run violation pass | `violation_detector.py` (`detect_violations`) |
| 3 | `s2_init` | `scenario_2_oltc.py` | `_calibrate_tap_sign` probe + setup | — |
| 4 | `s2_loop` | `scenario_2_oltc.py` | per-timestep OLTC deadband tap loop (pre-PF → tap decision → post-tap PF/rollback) | `violation_detector.py` |
| 5 | `s3_init` | `scenario_3_svc.py` | `_compute_svc_params` (Q_MAX, k_q), `_select_svc_bus`, SVC sgen create | — |
| 6 | `s3_loop` | `scenario_3_svc.py` | `_droop_q` per-timestep loop + `finally` SVC removal | `violation_detector.py` |
| 7 | `s4_init` | `scenario_4_volt_var.py` | INIT/CFG/P handshake, DER index/MW resolution, dynamics init | `volt_var_controller.py` (`configure`, `set_qv_parameters`), `volt_var_arduino.ino`, `der_dynamics.py` |
| 8 | `s4_loop` | `scenario_4_volt_var.py` | `run_coordinated_timestep` full HIL loop (pre-PF → V:/Q: exchange → coordinate → dynamics → apply → post-PF → curtailment) | `sensitivity_coordinator.py` (`coordinate`), `volt_var_controller.py` (`exchange`, `QVCharacteristic`, `_clamp_to_net_limits`), `der_dynamics.py` (`DERDynamics.step`), `volt_var_arduino.ino` (`compute_q`), `violation_detector.py` |
| 9 | `s5_init` | `scenario_5_opf.py` | `_setup_opf`, `_prepare_ext_grid_for_opf`, `create_poly_cost` | — |
| 10 | `s5_loop` | `scenario_5_opf.py` | per-timestep `runopp` dispatch (p_bound clip, q_lim = min(VDE, circle), poly_cost rebuild) | `violation_detector.py` |
| 11 | `plugin_reg` | `plugin_runner.py` | `register_and_run`, `load_plugin`, `_import_controller_fn`, `_allocate_plugin_num`, HW routing → `HardwareControllerFn`, `_plugin_runner` | `benchmark_runner.py` (`run_benchmark`), `volt_var_arduino.ino` (HW mode) |
| 12 | `plugin_loop` | `custom_controller.py` | `run_custom_controller_scenario` per-timestep `[A]–[F]` loop; SW `controller_fn` vs HW path | `volt_var_controller.py` (metadata, clamp), `violation_detector.py` |
| 13 | `orch_script` | `run_benchmark_script.py` | top-level direct-script flow: args → overrides → net → profiles → `BenchmarkConfig` → dispatch → `publish_result` | `network_plugin.py`, `profile_builder.py`, `volt_var_controller.py`, `violation_detector.py` |
| 14 | `orch_runner` | `benchmark_runner.py` | `run_benchmark`, `_build_kwargs`, `SCENARIO_REGISTRY`, per-scenario deepcopy isolation, HC + HC-stressed recursion | `hosting_capacity.py`, scenario runners |
| 15 | `profile_simbench` | `profile_builder.py` | `build_annual_profiles` SimBench branch (`get_absolute_values`, 35 136-step reconstruction, masks) | simbench lib |
| 16 | `profile_fallback` | `profile_builder.py` | `build_annual_profiles` DWD/fallback branch: `load_dwd_solar/_wind/_temperature`, `compute_load_profiles_bdew`, `compute_pv_profile`, `compute_wind_profile` | pvlib, oemof.demand |
| 17 | `netload_load` | `network_plugin.py` | `load_network_from_yaml`, `_load_yaml_config`, `_load_net` (json/pickle/function), `validate_network_plugin` | pandapower/simbench |
| 18 | `netload_strat` | `network_plugin.py` | `_build_profiles_for_strategy` (4 strategies incl. `custom`→dwd alias), `_build_profiles_simbench_native/_dwd/_flat`, `make_profile_factory` | `profile_builder.py` |
| 19 | `catalogue` | `network_catalogue.py` | `_PRESET_CATALOGUE`, `get_preset_families`, `get_presets_for_family` (menu data only) | `wizard.py` (`_ask_network`), `executor.py` (`_preset_loaders`), pandapower.networks |
| 20 | `publisher_static` | `publisher.py` | `publish_result` + `build_topology`/`build_profiles_payload`/`build_hc_payload`/`build_scenario_payload` | — |
| 21 | `publisher_live` | `publisher.py` | `PublishHandle` (`on_scenario_start`/`on_timestep`/`on_scenario_end`), `build_live_frame` | scenario runners (callers) |
| 22 | `hc_baseline` | `hosting_capacity.py` | `run_baseline_hc` + `_infer_dist_voltage`/`_hc_params_for`/`_find_endoffeeder_bus`/`_set_worst_case_snapshot`/`_add_pv_at_bus` | `violation_detector.py` |
| 23 | `hc_voltvar` | `hosting_capacity.py` | `run_hc_with_volt_var`, `_qv_converge` fixed-point loop | `volt_var_controller.py` (`QVCharacteristic`, `_clamp_to_net_limits`), `violation_detector.py` |
| 24 | `cli_wizard` | `wizard.py` | `run_wizard` 9-step flow + `_ask_*` | `run_plan.py` (`RunPlan`/`NetworkConfig`/`DatasetConfig`/`ParameterConfig`), `__main__.py` |
| 25 | `cli_executor` | `executor.py` | `execute` 5 phases, `apply_qv_overrides`, `apply_violation_limits`, `build_benchmark_config` | `volt_var_controller.py`, `violation_detector.py`, framework |
| 26 | `cli_resolve` | `executor.py` | `build_net_and_profiles` (network 4-way × dataset 3-way dispatch), `_preset_loaders`, `_import_custom_network_fn` | `network_plugin.py`, `profile_builder.py`, simbench |

---

## 2. Coverage status of every source file in the tree

### 2a. Fully covered — a diagram's primary subject (22 files)

| File | Diagram(s) |
|---|---|
| `scenario_1_baseline.py` | s1_init, s1_exec |
| `scenario_2_oltc.py` | s2_init, s2_loop |
| `scenario_3_svc.py` | s3_init, s3_loop |
| `scenario_4_volt_var.py` | s4_init, s4_loop |
| `scenario_5_opf.py` | s5_init, s5_loop |
| `volt_var_controller.py` | s4_init, s4_loop (+ hc_voltvar, cli_executor) |
| `volt_var_arduino.ino` | s4_init, s4_loop (+ plugin_reg/loop HW) |
| `sensitivity_coordinator.py` | s4_loop |
| `custom_controller.py` | plugin_loop |
| `plugin_runner.py` | plugin_reg |
| `run_benchmark_script.py` | orch_script |
| `benchmark_runner.py` | orch_runner |
| `profile_builder.py` | profile_simbench, profile_fallback |
| `network_plugin.py` | netload_load, netload_strat |
| `network_catalogue.py` | catalogue |
| `publisher.py` | publisher_static, publisher_live |
| `hosting_capacity.py` | hc_baseline, hc_voltvar |
| `wizard.py` | cli_wizard (+ catalogue) |
| `executor.py` | cli_executor, cli_resolve (+ catalogue `_preset_loaders`) |
| `run_plan.py` | cli_wizard |
| `__main__.py` | cli_wizard |
| `der_dynamics.py` | shown inline in s4_init/s4_loop — see 2b (no standalone) |

### 2b. Leaf modules — shown inline, NO standalone diagram (3 files)

These appear inside the diagrams that call them but have no diagram of
their own. Fine for 1a as delivered; **still need Task 1b prose.**

| File | Where shown | Standalone diagram? |
|---|---|---|
| `violation_detector.py` (`detect_violations`, `detect_violations_3ph`) | every scenario loop | OPTIONAL — the only leaf with enough branching logic to justify a small one if you want exhaustive visuals |
| `der_dynamics.py` (`DERDynamics.step`) | s4_init, s4_loop | OPTIONAL — the PT1/ramp state machine could be a small standalone |
| `scenario_result.py` (`TimestepRecord`, `ScenarioResult`, `make_record_from_report`, `slice_profiles`, `oversize_inverters`) | s1_init, plugin_loop, publisher | LOW VALUE — mostly dataclasses; prose is enough |

### 2c. Not flowchart material — Task 1b prose only, NO diagram needed (11 files)

Output/util/example/test code with no multi-lane control flow to diagram.

| File | What it is |
|---|---|
| `network_plotter.py` | topology/profile plotting utility (matplotlib) |
| `plot_results.py` | results/figure plotting utility |
| `era5_to_csv.py` | one-shot ERA5 → CSV prep for the `custom` weather path |
| `network_export.py` | pandapower net → JSON/pickle export helper |
| `helpers.py` | CLI terminal-display helpers (`print_run_plan`, tables) |
| `_console.py` | Rich `Console` + theme singleton |
| `my_network_loader.py` | example `function`-source stub (`get_network` → CIGRE MV) |
| `deadband_controller.py` | example controller plugin (contract shown in plugin_loop) |
| `droop_controller.py` | example controller plugin (contract shown in plugin_loop) |
| `test_helpers.py` | smoke test for `helpers.py` |
| `__init__.py` | empty package marker |

### 2d. Diagnostics — EXCLUDED by standing user directive (never diagram) (8 files)

| File |
|---|
| `analyze_timing.py` |
| `diagnose_outliers.py` |
| `diagnose_qinitial_clip.py` |
| `compare_scenarios.py` |
| `classify_s2_violations.py` |
| `check_coordination_ground_truth.py` |
| `run_full_diagnostics.py` |
| `hil_setup_test.py` (hardware/serial setup test — diagnostic-adjacent; confirm intent, but not a benchmark flow) |

### 2e. Phantom — referenced in docs, ABSENT from tree (1)

| File | Note |
|---|---|
| `stress.py` | `apply_overvoltage_stress`/`apply_hw_synthetic_stress` described in `TASK1B_MODULE_DOCUMENTATION.md` + Knowledge Base docx, but **not present** in `/mnt/project`. Resolve in Task 2: un-uploaded working file, or removed + stale docs. |

### 2f. YAML schemas (6) — covered conceptually by netload/plugin diagrams; need 1b prose

| File | Role | Diagram context |
|---|---|---|
| `simbench_rural_mv.yaml` | network plugin: `json` + `simbench_native` | netload_load/strat |
| `custom_function.yaml` | network plugin: `function` + `custom` strategy | netload_load/strat, cli_resolve |
| `custom_lv_flat.yaml` | network plugin: `flat` | netload_strat |
| `Scenario_3.yaml` | direct-script SVC config | orch_script (YAML path) |
| `deadband_controller.yaml` | controller plugin config | plugin_reg |
| `droop_controller.yaml` | controller plugin config | plugin_reg |

---

## 3. Still left / could be done

### 3a. Mandatory for Task 1a: NONE
All core control, orchestration, and infrastructure flows are diagrammed.
1a is complete as delivered (flat per-module set).

### 3b. Optional standalone diagrams (only if exhaustive visual coverage wanted)
- `violation_detector.detect_violations` / `detect_violations_3ph` — small decision diagram (V band → line loading → trafo loading → angle → `ViolationReport`).
- `der_dynamics.DERDynamics.step` — small state diagram (reset guard → PT1 on Q → ramp on P → clip).
- (`scenario_result` dataclasses — low value; skip unless asked.)

### 3c. Descoped from the ORIGINAL brief §1a (available if the paper/presentation wants them)
The original Task 1a asked for two things the flat set intentionally
replaced (user-confirmed). They are NOT built; flag before assuming:
- **Top-level system-architecture diagram** (the "paper Figure 1 / slide"
  overview: two entry points → `benchmark_runner`, plugin paths, 6
  scenario types + HC, the HIL RPi↔Arduino subsystem, output landing).
  Maps to no single file — it's the whole-system view. Likely wanted for
  Task 3 (the course requires "a flow chart" for the model description).
- **Single-run end-to-end data-flow diagram** (RunPlan → net+profiles →
  TimestepRecords → ScenarioResult → publisher JSON → Streamlit/notebook,
  showing what is written to disk at each stage). Spans `run_plan` →
  `executor`/`run_benchmark_script` → runners → `scenario_result` →
  `publisher` → consumers.

### 3d. Not diagram work — belongs to the next tasks
- **Task 1b (module docs):** prose for EVERY file in §2a–2c and §2f
  (42 `.py` + `.ino` + 6 YAML). A ~112K `TASK1B_MODULE_DOCUMENTATION.md`
  already exists — continue/reconcile it, don't restart.
- **Task 2 (audit):** covers all files including the diagnostics in §2d;
  resolve the `stress.py` phantom (§2e) and the real `custom_function.yaml`
  absolute-`data_dir` finding.

---

## 4. One-line reconciliation with the earlier "coverage note"
The flowchart session's spoken coverage note was **incomplete**: it omitted
`network_plotter`, `plot_results`, `era5_to_csv`, `network_export`,
`deadband_controller`, `droop_controller`, `hil_setup_test`, and
`test_helpers`, and it **named `stress.py`, which does not exist**. This
document supersedes it. (It also predates the correction that
`network_plugin` `strategy: custom` is VALID, reflected in §1/§2f here.)
