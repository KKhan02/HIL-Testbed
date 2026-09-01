# Network Plugin System — Implementation Guide

**Files delivered:** `network_plugin.py` (new), `run_benchmark_script.py` (updated), `example_networks/` (3 YAMLs + 1 stub loader module)

**Constraint honoured:** pure addition. `network_catalogue.py`, `benchmark_runner.py`, `profile_builder.py`, and `scenario_result.py` are byte-for-byte untouched. The only modified file is `run_benchmark_script.py`, which was explicitly in scope.

---

## 1. What it does

A student drops a folder containing a YAML file and a network file (or loader script) next to the project and runs:

```
python run_benchmark_script.py --network example_networks/custom_lv_flat.yaml
```

`network_plugin.load_network_from_yaml()` loads the network from one of three sources, builds a benchmark-ready profiles dict with one of three strategies, and returns `(net, profiles)` — the exact pair `run_benchmark()` already consumes. `validate_network_plugin()` then screens the pair against the framework's known constraints and returns warning strings; the script prints them and asks for confirmation before running.

No knowledge of `profile_builder.py` internals, `network_catalogue.py`, or the 44-entry preset system is required.

---

## 2. YAML schema

```yaml
name: my_lv_feeder                      # required — becomes network_id
label: "My Custom LV Feeder"            # optional — defaults to name
source: json                            # json | pickle | function
path: networks/my_feeder.json           # json/pickle sources
# module: my_network_loader.py          # function source (a .py FILE PATH)
# function: get_network                 # function source
profiles:
  strategy: simbench_native             # simbench_native | dwd_pvlib | flat
  year: 2016
  # data_dir: ../data/dwd               # optional dwd_pvlib override
voltage_limits:
  v_min: 0.95
  v_max: 1.05
notes: "Real LV feeder from Stadtwerke Oldenburg, anonymised"
```

All relative paths (`path`, `module`, `profiles.data_dir`) resolve **relative to the YAML file's directory** — the same convention as the controller plugin, so a plugin folder is self-contained and portable between the laptop and the RPi over SCP.

### Sources

| Source | Mechanism | Notes |
| --- | --- | --- |
| `json` | `pandapower.from_json(path)` | **Recommended.** `pp.to_json(net, "networks/my_net.json")` preserves all element tables, `std_types`, and extra attributes — including SimBench's `net.profiles` dict (verified experimentally on pandapower 3.4.0, see §6). |
| `pickle` | `pandapower.from_pickle(path)` | Always through pandapower's own serialisation function, never the raw `pickle` module, so pandapower handles version compatibility itself. |
| `function` | importlib file-location import, then call a zero-argument function returning a `pandapowerNet` | Identical pattern to `plugin_runner._import_controller_fn()`: `spec_from_file_location`, mangled `sys.modules` key (hash of the resolved path) to prevent stem collisions, `sys.modules` cleanup on exec failure, `hasattr`/`callable` checks with a helpful listing of available callables on failure. |

The loaded object is type-checked against `pp.pandapowerNet` regardless of source, so a loader function returning the wrong thing fails immediately with a clear message rather than deep inside `adapt_profiles()`.

### Profile strategies

**`simbench_native`** — calls `sb.get_absolute_values(net, profiles_instead_of_study_cases=True)` **on the already-loaded net**, not via a `simbench_code` re-download. This is the key difference from `build_annual_profiles()`'s SimBench path, and it is what makes an exported SimBench JSON portable: the profiles metadata travels inside the file. The post-processing then mirrors `profile_builder`'s SimBench branch line-for-line in behaviour:

- `DatetimeIndex` reconstruction from the integer step index (`year` from the YAML, 15-min, Europe/Berlin);
- `pv_mask` includes `lv_res` (the "brings profiled DERs from ~7 to 98" invariant);
- night-time PV zeroing (hours ≥ 22 or ≤ 4);
- lower-clipping of load, PV, and wind frames;
- `extreme_days` via the existing `profile_builder.find_extreme_days()` (imported, not duplicated).

If the metadata is absent (`net.profiles` missing, empty, or containing no non-empty DataFrames), a warning is logged and the strategy **falls back to `dwd_pvlib`** — recorded in the returned metadata as `requested_strategy` vs `strategy` so the discrepancy is auditable.

**`dwd_pvlib`** — delegates entirely to `build_annual_profiles(net, net_name=…, data_dir=…, simbench_code=None)`, i.e. the documented DWD station 691 Bremen path (pvlib Erbs decomposition + NOCT correction + oemof.demand BDEW 2025 SLPs). Default `data_dir` is `<project_root>/data/dwd`, the same resolution `run_benchmark_script.py` already uses (`Path(__file__).resolve().parent.parent / "data" / "dwd"`).

One subtlety handled here: `detect_network_type()` routes **purely on the name string**. If a student names their plugin `simbench_rural_export`, `build_annual_profiles()` would take the SimBench branch and raise `simbench_code must be provided`. `_dwd_safe_name()` therefore sanitises any name that matches a `SIMBENCH_IDENTIFIERS` entry before passing it down (e.g. `simbench_rural_mv_export` → `sb_rural_mv_export`, `1-MV-rural--2-sw_copy` → `1_mv_rural--2-sw_copy`), guaranteeing the DWD path fires. Verified by test.

**`flat`** — constant profiles at rated capacity over a full leap-aware year at 15-min resolution: every load at `net.load.p_mw`, every PV/wind sgen at `net.sgen.p_mw`, on every timestep. This is the simultaneous-peak snapshot repeated all year — worst-case voltage-rise screening with zero data dependencies. Built with `np.tile` (one allocation per table, no per-timestep loop). *Design decision:* rated capacity means `p_mw`, consistent with `profile_builder`, which uses `p_mw` as `p_rated` in both `compute_pv_profile()` and `compute_wind_profile()`; `sn_mva` remains the inverter apparent-power limit used by the Q-clamping machinery and is not conflated with dispatch.

### Returned profiles dict

Schema-identical to `build_annual_profiles()`: `"load"`, `"pv"`, `"wind"`, `"times"`, `"extreme_days"`, `"net_type"` — so `adapt_profiles()`, `slice_profiles()`, `data_checks.py` (which asserts exactly this key set as *required*, not exclusive), the publisher, and the plotting stack consume it unchanged. One **additive** key is included: `"plugin_meta"` (see §5, improvement 1).

---

## 3. Validation matrix

`validate_network_plugin(net, profiles) -> list[str]` returns warnings, not exceptions — deliberately, because some networks legitimately fail individual checks (a DER-free Kerber feeder is fine for an HC-only or Scenarios 1–3 run). The caller decides.

| # | Check | Framework rationale |
| --- | --- | --- |
| 1 | `net.sgen` non-empty | Scenario 4 has no DERs to control otherwise; warning suggests `scenarios=[1, 2, 3]`. |
| 2 | ≥ 1 sgen with recognised type in `["PV", "WKA", "lv_res", "pv", "wind"]` (case-insensitive, plus `wp`/`solar` aliases matching profile_builder's own masks) | Otherwise the pv/wind masks match nothing and `der_p` ends up empty — Scenario 4 silently controls nothing. |
| 3 | ≥ 1 bus in MV (1.0–36.0 kV) or LV (< 1.0 kV) | The benchmark targets distribution networks; an HV-only net is out of scope. |
| 4 | ≥ 1 transformer | Scenario 2 operates on `net.trafo.tap_pos`; a separate softer warning fires when only `trafo3w` is present (OLTC still finds nothing to control). |
| 5 | No non-zero `const_i_percent` / `const_z_percent` on `net.load` | All `runpp()` calls use `voltage_depend_loads=False` (mandatory pandapower 3.2.0+ workaround), so ZIP shares would be **silently ignored** — the warning makes that explicit. |
| 6 | *(additive)* profiles/net column alignment, non-empty DER columns, NaN scan | Catches the three most common student failure modes before they surface as a cryptic `KeyError` inside `adapt_profiles()` or a mid-run non-convergence. |

---

## 4. `run_benchmark_script.py` changes (minimal, targeted)

Three edits, everything else untouched (all comment blocks, optional sections, publisher wiring, HC config preserved):

1. **argparse block** after the logging setup: `--network YAML` and `-y/--yes`. When `--network` is given, `load_network_from_yaml()` replaces both the hardcoded `sb.get_simbench_net()` call *and* the `build_annual_profiles()` call (the plugin already built the profiles per its strategy); `sb_code` is set to `None`. Validation warnings are printed in a numbered block and, unless `-y` is passed, an explicit `Proceed with the benchmark anyway? [y/N]` prompt gates the run — `sys.exit(1)` on anything other than y/yes.
2. **Profiles guard**: the existing `build_annual_profiles()` call is wrapped in `if not _args.network:` — no other change to it.
3. **Voltage limits wiring**: `BenchmarkConfig(v_min=…, v_max=…)` now reads from `plugin_meta` with 0.95/1.05 defaults, so the YAML's `voltage_limits` actually reach every scenario runner and the Rich summary highlighting.

Without `--network`, behaviour is bit-identical to before (defaults: `_args.network=None`, `_plugin_meta={}` → v_min/v_max fall back to 0.95/1.05, which were the implicit `BenchmarkConfig` defaults anyway).

Downstream compatibility: `slice_profiles()` explicitly passes non-indexed dict values through unchanged, so `profiles_run = slice_profiles(profiles, …)` on the plugin path works as-is and `plugin_meta` survives the slice (verified by test). `run_hc_scenarios` remains `False` by default; note that the HC `profile_factory` lambda still references `build_annual_profiles` with the plugin's `net_name`/`sb_code=None` — acceptable for a DWD-capable plugin net, but a flat-strategy plugin combined with `run_hc_scenarios=True` would rebuild DWD profiles rather than flat ones. Documented as a known limitation (§7).

---

## 5. Improvements beyond the literal spec — please confirm or reject

Each is additive and independently removable. All are implemented; say the word and I revert any of them.

1. **`plugin_meta` key in the profiles dict.** The spec's return contract is `(net, profiles)`, but `voltage_limits` from the YAML must reach `BenchmarkConfig` somehow. Rather than changing the return signature or a global, the parsed metadata (name, label, source, strategy actually used, requested strategy, year, v_min, v_max, notes, YAML path) rides inside the profiles dict as one extra key. `adapt_profiles()` reads only its four required keys and `slice_profiles()` passes extra scalars through, so this is invisible to every existing consumer — the same additive-key pattern already proven safe in the earlier network-plugin design work.
2. **`v_min`/`v_max` forwarded into `BenchmarkConfig`** in the script (consequence of 1). Otherwise the YAML's `voltage_limits` block would be decorative.
3. **`-y/--yes` flag** to auto-confirm warnings — needed for headless RPi runs over SSH where an interactive `input()` would hang a detached session.
4. **Optional `profiles.data_dir`, `file_map`, `col_map` keys** — forwards `build_annual_profiles()`'s existing custom-datasource hooks to plugin users (e.g. ERA5 CSVs) without them touching Python. Omitted keys behave exactly as the spec's minimal YAML.
5. **`module:` key for the `function` source** (with `path:` accepted as an alias) — matches the controller plugin's vocabulary so students learn one schema, while keeping the spec's `path`-only YAML valid.
6. **Validation checks beyond the required five** (row 6 in §3): profiles/net column alignment, empty-DER-columns, NaN scan. Strictly additional warning strings; the five required checks are implemented exactly as specified.
7. **`trafo3w` awareness** in the transformer check — a net with only three-winding transformers gets a *specific* warning (OLTC operates on `net.trafo`) instead of a false "no transformer" claim.
8. **`_dwd_safe_name()` sanitisation** — prevents the string-routing failure described in §2 when a plugin name collides with `SIMBENCH_IDENTIFIERS`.
9. **`label` optional, defaulting to `name`** — the spec YAML shows both; requiring `label` seemed needless friction for quick tests.
10. **`requested_strategy` recorded alongside `strategy`** in `plugin_meta`, so a simbench_native→dwd_pvlib fallback is visible after the fact, not just in a scrolled-past log line.

---

## 6. Verification performed (pandapower 3.4.0, simbench 1.6.2)

- **`pp.to_json` round-trip preserves `net.profiles`**: exported a SimBench net, reloaded via `pp.from_json`, ran `sb.get_absolute_values()` on the reloaded net → 35,136 steps, correct `(load, p_mw)` / `(sgen, p_mw)` frames. This is the empirical foundation of `simbench_native`.
- **All three strategies exercised**: `json + simbench_native` (35,136 steps, correct pv/load column counts, zero warnings); `json + flat` on a Kerber LV net with an added PV sgen (constant 0.03 MW verified at t=0); `function + dwd_pvlib` on CIGRE MV `pv_wind` using the project's DWD station 691 CSVs (52,698 aligned 10-min steps, 18 BDEW loads, 8 PV, 1 wind — matching the built-in CIGRE path output).
- **Fallback**: CIGRE JSON with `strategy: simbench_native` → warning logged, dwd_pvlib used, `requested_strategy`/`strategy` recorded correctly.
- **Warning matrix**: empty-sgen, ZIP-load (`const_z_percent=50`), and orphan-profile-column warnings all fire with the intended text.
- **Schema compatibility**: `adapt_profiles()` consumed a plugin profiles dict without modification (`der_p (35136, 1)`, `dt_s=900.0`); `slice_profiles(period="day")` sliced it and preserved `plugin_meta`.
- **End-to-end**: `python run_benchmark_script.py --network example_networks/custom_lv_flat.yaml -y` ran Scenarios 4A/4B on the sliced day — 96/96 converged on both, CSV written, publisher emitted all five JSON files under `outputs/publisher/my_lv_feeder_flat/`.
- **Prompt/abort path**: a warning-triggering plugin without `-y` printed the numbered warning block, prompted, and exited cleanly on `n`. `--help` renders both new arguments; the no-argument invocation follows the original hardcoded path untouched.
- **Name sanitiser**: `simbench_rural_mv_export` → routes to `fallback`; `1-MV-rural--2-sw_copy` → routes to `fallback`; clean names pass through unchanged.

*(Scenario 5 required a `cyipopt` stub in the sandbox only — the real RPi/laptop environments have IPOPT installed; nothing in the deliverables depends on the stub.)*

## 7. Known limitations

- ~~HC re-benchmark on plugin nets~~ **Resolved**: `make_profile_factory()` (§8) now supplies `BenchmarkConfig.profile_factory` on the plugin path, so the HC-stressed re-benchmark rebuilds profiles with the plugin's own strategy.
- **`simbench_native` year**: SimBench's internal reference year is 2016 (leap, 35,136 steps). Setting `year` to a non-leap year with a 35,136-step profile set will produce a time axis running 24 h into January of the following year. The default (2016) is correct; the field exists mainly for `flat`.
- **`flat` extreme days**: with constant profiles all daily means are equal, so `find_extreme_days()` returns the first day for every key — harmless, but zoom plots are unexciting by construction.
- **Pickle trust**: `pandapower.from_pickle()` still executes pickle bytecode; only load pickles from lab-internal sources. JSON remains the recommended format for exactly this reason too.

---

## 8. Follow-up additions (second iteration)

### 8.1 `make_profile_factory()` — HC re-benchmark uses the plugin's own strategy

`network_plugin.make_profile_factory(yaml_path)` returns a `callable(net) -> profiles` closure for `BenchmarkConfig.profile_factory`. Hosting-capacity analysis adds new sgens to a copy of the network; the stressed re-benchmark then needs fresh profiles that **cover those new sgens**. The strategy dispatch (including the simbench_native→dwd_pvlib fallback check) was factored into a shared `_build_profiles_for_strategy(net, cfg)` used by both `load_network_from_yaml()` and the factory, so outer run and re-benchmark are guaranteed identical semantics.

Per-strategy behaviour on the stressed net:

- **flat** — new HC sgens get constant rated-capacity columns automatically (the frame is rebuilt from `net.sgen`). *Verified:* stressed net with an added 0.5 MW PV sgen → profiles gained the new column at 0.5 MW.
- **dwd_pvlib** — new HC sgens get pvlib/BDEW columns automatically (the built-in DWD path iterates `net.sgen`).
- **simbench_native** — the stressed net is a deepcopy of the plugin net, so the SimBench metadata survives (*verified*); HC sgens have no native profile columns, matching the built-in SimBench HC behaviour.

The factory's `plugin_meta` records `name + "_hc_stressed"` so the publisher output directory naming stays consistent with the built-in convention.

**Script wiring:** on the `--network` path, `BenchmarkConfig.profile_factory` is now `make_profile_factory(_args.network)`; the original `build_annual_profiles` lambda is untouched on the default path.

### 8.2 `--controller` — network plugin × controller plugin combination

Direct answer to the question asked: **the two plugin systems were previously independent** — `plugin_runner.register_and_run()` existed but was only callable from Python; `run_benchmark_script.py` had no controller argument at all, on either the hardcoded or the `--network` path. Now:

```
python run_benchmark_script.py \
    --network    example_networks/custom_lv_flat.yaml \
    --controller example_plugins/droop_controller.yaml \
    -y
```

Implementation is one conditional in the Run section: when `--controller` is given, `plugin_runner.register_and_run(yaml, net, profiles_run, network_id=net_name, benchmark_config=config, return_benchmark=True)` replaces the direct `run_benchmark()` call. Everything composes for free by design:

- `register_and_run()` **extends** `config.scenarios` on a copy (`dataclasses.replace`), so the custom controller runs *alongside* the configured built-ins, registered at the first free number ≥ 10 with try/finally registry cleanup — exactly the existing controller-plugin machinery, no changes to `plugin_runner.py` needed.
- The plugin network's `(net, profiles_run)` flow through unchanged — sliced profiles, `plugin_meta`, and the YAML `voltage_limits` (already inside `config.v_min/v_max`) all reach the custom scenario, since `_build_kwargs()` injects `v_min`/`v_max` into every runner.
- `return_benchmark=True` hands back the full `BenchmarkResult`, so the downstream publisher / CSV / comparison-table code runs **unmodified** — the custom scenario appears as a row in the summary and as `scenarios/<name>.json` in the publisher output.
- A one-line summary for the custom controller is printed after the run.

*Verified end-to-end:* flat LV plugin network + `droop_controller.yaml` → custom scenario registered as **Scenario 10**, ran alongside 4A/4B, 96/96 converged, publisher emitted `scenarios/my_droop_controller.json` next to the built-in scenario JSONs.

**One interaction to be aware of:** `register_and_run()` builds the custom `ScenarioSpec` with `supports_lv=False`. `benchmark_runner`'s skip only fires for LV networks **with no PV/wind DERs** (`is_lv and not supports_lv and not _has_ders`), so any LV plugin network that actually has DERs — the only case where a Q controller is meaningful — runs the custom scenario normally. A DER-free LV network skips it with a logged reason, which is the correct outcome.
