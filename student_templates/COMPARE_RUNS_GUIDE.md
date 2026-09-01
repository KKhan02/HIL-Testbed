# `compare_runs.ipynb` — Implementation Guide

Companion document for the student comparison notebook in `student_templates/`.
Covers how the notebook works, the logic behind each cell, how it interacts with the
publisher JSON contract, how the custom-network and custom-controller plugin systems
are supported, and the improvements that were flagged and applied relative to the
original specification.

---

## 1. What the notebook is and where it sits in the architecture

The benchmark pipeline ends with `publisher.publish_result()`, which serialises a
completed `BenchmarkResult` into plain JSON under
`outputs/publisher/<net_name>/` (see `run_benchmark_script.py`, which sets
`output_dir = <project_root>/outputs/publisher/<net_name>`):

```
outputs/publisher/<net_name>/
├── topology.json          build_topology(net)
├── profiles.json          build_profiles_payload(profiles)
├── comparison.json        result.comparison_df.to_dict(orient="records")
├── hc.json                build_hc_payload(...)          (optional)
└── scenarios/
    ├── baseline.json      build_scenario_payload(ScenarioResult)
    ├── volt_var_local.json
    ├── volt_var_coord.json
    └── <plugin_id>.json   ← custom controller plugins land here too
```

The notebook consumes **only** these files. It deliberately mirrors the design
constraint of `plot_results.py`: *no pandapower, no simbench, no project imports* —
only `numpy`, `pandas`, `matplotlib`, `json` (plus stdlib `re`/`pathlib`/`math`). A
student can therefore copy a publisher folder onto any laptop and run the notebook
without installing the simulation stack, and the notebook can never mutate or depend on
the protected core modules. This is the same non-invasive, pure-addition principle used
for the controller and network plugin subsystems.

## 2. Cell-by-cell logic

### Cell 1 — Configuration
The only editable cell. `RUN_A_DIR` / `RUN_B_DIR` point at two publisher directories;
`STUDENT_LABEL_A/B` are the legend labels; `OUTPUT_DPI = 300` and
`OUTPUT_WIDTH_INCHES = 3.5` fix IEEE single-column geometry. A "do not edit below"
block then sets the matplotlib style once, globally: Times New Roman if installed
(checked via `font_manager.fontManager.ttflist`) with an explicit serif fallback,
8/7 pt font sizes (correct for a 3.5 in column — 10 pt text at 3.5 in prints
oversized), `savefig.dpi = 300`, `bbox_inches="tight"`. A `save_fig()` helper
guarantees every figure lands in `student_templates/figures/` with consistent settings,
and `IEEE_2COL_W = 7.16` is provided for the two heatmaps, which are unreadable at
single-column width. `VOLT_VAR_CONTROLLER_PATH` (used by Cell 8) also lives here so the
student edits paths in exactly one place.

### Cell 2 — Data loading
`load_run()` builds one dict per run: `comparison.json`, `topology.json` (for
`voltage_limits` and the `sgens` list), and **every** file in `scenarios/*.json` via
glob — the same auto-discovery as `load_publisher_dir()` in `plot_results.py`. Each
scenario's `timeseries` (a list of per-`TimestepRecord` dicts) is flattened into a
DataFrame of the *scalar* fields only (`max_vm_pu`, `t_total_ms`,
`coordination_active`, `curtailment_needed`, …), indexed by parsed timestamps; the
heavy dict-valued fields (`vm_pu_by_bus`, `q_mvar_by_sgen`, `p_mw_by_sgen`) stay in the
raw payload and are touched only by the cells that need them. One derived column is
precomputed during loading: `p_applied_sum_mw = Σ p_mw_by_sgen`, because two figures
need it and summing dicts per timestep is the expensive part.

Robustness decisions made here:

* **Missing files** → the scenario's entry is `None`, a warning is printed, and every
  downstream cell checks via `get_ts()` before plotting. Verified by executing the
  notebook against a run directory with `volt_var_coord.json`, `comparison.json` and
  `profiles.json` deleted — zero exceptions, correct skips.
* **Timestamp parsing** — `_parse_times()` first tries plain ISO parsing; if the run
  spans a DST change the mixed `+01:00/+02:00` offsets produce an object index, and it
  falls back to `utc=True`. A second fallback drops `format="ISO8601"` for
  pandas < 2.0.
* **Timestep resolution is inferred, never assumed** (`_step_hours()`): the publisher
  docstring says 10-min, the SimBench annual runs are 15-min (35,136 steps). Every
  energy integration and daily grouping uses the inferred value or the actual
  timestamps (`.groupby(index.normalize())`), so a hard-coded 144-steps/day bug (as in
  `fig09` of `plot_results.py`, which assumes 10-min) cannot occur.

### Cell 3 — Scalar comparison table
Builds one row per scenario per run with the 31 columns of
`ScenarioResult.summary_dict()`, copied verbatim (order included) from
`scenario_result.py`. Primary source is each scenario JSON's `summary` key — the
authoritative one, since `build_scenario_payload()` serialises `summary_dict()` NaN-safe
— with `comparison.json` used as fallback for scenarios whose per-timestep JSON is
absent. Conditional formatting is a per-column linear green→red gradient
(min = best = green) applied with `Styler.apply` on `n_violation_steps`,
`curtailment_steps`, `vdi`. The CSV (`comparison_table.csv`) is written *before*
styling, so the on-disk artefact never depends on Styler behaviour.

### Cell 4 — Voltage envelope
Two stacked, x-sharing subplots: `max_vm_pu` (top, with V_max dashed red) and
`min_vm_pu` (bottom, with V_min), Scenario 4B, RUN_A blue vs RUN_B orange. The band
limits come from each run's `topology.json → voltage_limits` (which
`build_topology()` fills from `violation_detector.V_MIN/V_MAX`, and which custom
networks can override) with a 0.95/1.05 fallback; if the two runs carry different
bands, both lines are drawn. Lines are `rasterized=True` so a 35k-point annual PDF
stays a few hundred kB instead of tens of MB, while axes/labels remain vector.

### Cell 5 — Monthly curtailment bars
`curtailment_needed` is a per-timestep boolean, so its monthly **sum is the count of
curtailed timesteps** — grouped with `index.to_period("M")` on the real timestamps, so
partial-year runs (from `slice_profiles()`) show only the months present. Four bars per
month in the order 4A_A, 4B_A, 4A_B, 4B_B: colour = scenario (the `plot_results.py`
greens `#4dac26`/`#1a9641`), hatch = run (solid A, `///` B), matching the spec's
"hatching distinguishes runs, colour distinguishes 4A/4B". The cell also prints annual
totals and the 4A→4B reduction per run — the timestep-count version of the paper's
`[X]%`.

### Cell 6 — Timing distribution
Histograms of steady-state `t_total_ms` (excluding `t = 0`, which carries pandapower
cold-start cost) for 4B in both runs on shared bins, with median (solid) and p95
(dotted) vertical lines labelled in run colour. `hil_latency_ms` — populated only on
live HIL runs, null on dry-run — is overlaid as a stepped dashed histogram when
present, which makes the "serial ≈ 66 % of the timestep budget" finding visible in one
figure. The printed summary reports median/p95 and, when latency exists, serial share
of the median timestep.

### Cell 7 — Coordination rate
`coordination_rate` is read from `summary` (`coordination_steps / n_converged`, a 0–1
fraction per `scenario_result.py`, converted to %) with a fallback computed from
`coordination_active` over converged timesteps. Two bars, value labels, and the
decision rule from the spec: if the runs differ by **> 20 percentage points**, the
quantisation-finding paragraph (Sec. 6.1 of the knowledge base: 4-decimal ASCII
truncation + AVR float32 → ~0.0013 MVAr/DER smaller `q_initial` → coordinator gate at
1.051 pu trips on ~99.9 % of violated timesteps on HIL vs ~9.5 % dry-run; analogous to
Modbus/SunSpec 16-bit registers) is printed as the auto-generated interpretation.
Otherwise a one-line "no divergence" message is printed instead.

### Cell 8 — Q(V) operating scatter
Reproduces the `fig10_qv_scatter` join: `q_mvar_by_sgen[sgen] ↔
vm_pu_by_bus[sgen_to_bus[sgen]]` using `topology.json → sgens`. RUN_A,
`volt_var_local` only (per spec, to avoid overplotting), converged timesteps only,
subsampled to 200 k points max with a seeded RNG so the PDF stays paper-sized and
reproducible. Breakpoints `U1_PU…U4_PU` and `Q_RATIO` are regex-parsed from
`volt_var_controller.py` at `VOLT_VAR_CONTROLLER_PATH` (the regex anchors on
`^NAME = <number>` so the trailing comment on the `Q_RATIO` line is ignored); if the
file is missing, the hard-coded VDE-AR-N 4110 defaults 0.96/0.99/1.01/1.04 and
Q_RATIO = 0.25 are used, with a comment in the code and a printed notice. The
theoretical overlay `qv_theoretical()` is a vectorised mirror of
`QVCharacteristic.compute_setpoint()`.

### Cell 9 — Export checklist + abstract
Lists the eight core outputs, checks each on disk, and prints/saves the abstract
template with `[X]` (4B-vs-4A `curtailment_steps` reduction, computed per run),
`[Y]`/`[Z]` (4B coordination rates of RUN_B and RUN_A respectively — a code comment
notes the RUN_A = dry-run / RUN_B = HIL convention and to swap if reversed) and
`[network]` (`network_id` from `topology.json`) auto-filled; `[CITE]` is left for the
student. The filled abstract is also written to `figures/abstract_template.txt`.

### Extensions 1–4 (from `plot_results.py`)
Selected from the 14 `plot_results.py` figures by two filters: (a) relevant to the
paper narrative (supervisor talking points: violation heatmap, curtailment reduction,
profiles documentation) and (b) computable from JSON alone in a **two-run comparison**
context. The network-map figures (fig01–04, 06) were excluded — they are single-run
topology documentation already covered by `plot_results.py`/`network_plotter.py`, and
need `networkx`, which would break the numpy/pandas/matplotlib-only constraint.

* **E1 Voltage heatmap** (fig05): day × bus daily-max `vm_pu_by_bus`, one panel per
  run, colour scale centred on each run's own band; PNG-only because a full raster
  heatmap PDF is enormous.
* **E2 Violation heatmap** (fig09): day × scenario `violation_flag`, **all** discovered
  scenarios — plugin scenarios appear as extra rows automatically; date-based
  day-grouping instead of fig09's hard-coded 144 steps/day.
* **E3 Curtailed energy** (fig13): per-timestep curtailed MW = `der_gen_mw` −
  Σ`p_mw_by_sgen` (fig13's exact definition: profile target minus applied), integrated
  with the inferred timestep length to MWh/day, 4A vs 4B × both runs; prints curtailed
  MWh totals and the energy-based 4A→4B reduction — the number most papers quote for
  `[X]` instead of timestep counts.
* **E4 Annual profiles** (fig07): hourly `load/pv/wind_total_mw` from `profiles.json`,
  loaded lazily (the file can be tens of MB) from RUN_A with RUN_B fallback.

## 3. Custom-plugin and custom-network support

**Controller plugins** (`custom_controller.py` / `plugin_runner.py`): a plugin run is
published exactly like a built-in scenario — `run_custom_controller_scenario()` returns
a `ScenarioResult` with the YAML's `scenario_id`, so `publish_result()` writes
`scenarios/<plugin_id>.json` with the identical schema. The notebook's glob-based
discovery therefore picks plugin scenarios up with zero configuration: they get rows in
the Cell 3 table (green/red grading includes them) and rows in the E2 violation
heatmap. `scenario_label()`/`scenario_sort_key()` fall back to the raw id and append
unknown scenarios after the built-in six. The degraded-run test included a plugin
scenario present in RUN_A but absent in RUN_B.

**Network plugins** (`network_plugin.py`): a plugin YAML may set
`voltage_limits: {v_min, v_max}` (forwarded through `plugin_meta` into the benchmark
and used by `run_custom_controller_scenario(..., v_min=..., v_max=...)`). The notebook
reads each run's band from `topology.json → voltage_limits` rather than hard-coding
0.95/1.05, so the envelope limits (Cell 4), the heatmap colour scales (E1), and the
Q(V) band markers (Cell 8) are correct for custom networks. `network_id` (the plugin's
`name`) flows from `topology.json` into the table, titles and abstract. Nothing else in
the JSON contract differs for plugin networks, so no other special-casing is needed.

## 4. Verification performed

The notebook was executed end-to-end (`jupyter nbconvert --execute`, restart-and-run-all
semantics) against synthetic two-run publisher directories generated to the exact
`publisher.py` schema (6,912 15-min timesteps spanning all 12 months, 14 buses, 6 DERs,
dry-run-like RUN_A with a plugin scenario, HIL-like RUN_B with `hil_latency_ms` ≈
151.8 ± 1.1 ms and 99.5 % coordination rate):

1. **Nominal run** — 0 errors; all 12 outputs written; quantisation branch fired at an
   87.4 pp gap; abstract auto-filled with plausible numbers (4B −32.8 % curtailment,
   99.5 % vs 12.1 % coordination).
2. **Degraded run** — RUN_B missing `volt_var_coord.json`, `comparison.json`,
   `profiles.json`, and `volt_var_controller.py` absent — 0 errors; every affected cell
   printed its warning and skipped only the missing half; Cell 8 fell back to the
   hard-coded VDE defaults; the checklist still reported the core outputs it could
   produce.
3. Figures visually inspected at 300 dpi for label sizing, legend collisions, and
   rasterisation quality.

## 5. Improvements flagged relative to the original specification

Each item below deviates from or extends the literal spec. All are implemented with the
listed default; say the word and any of them can be reverted to the literal reading.

1. **Voltage limits read from `topology.json`, not hard-coded** (Cells 4, 8, E1). The
   spec said "1.05 pu limit as a dashed red line"; hard-coding would silently draw the
   wrong band for network-plugin runs with custom `voltage_limits`. Fallback remains
   0.95/1.05.
2. **Q(V) scatter is normalised per DER** (Cell 8): `q / (Q_RATIO · P_inst)` with
   `sn_mva`-over-`p_mw` precedence as in `VoltVarController`. A single theoretical
   curve overlaid on raw MVAr is only meaningful if all DERs share one rating — on
   SimBench MV they span ~0.02 to ~2.4 MVA, so the raw plot would be a family of 100
   curves. Normalisation collapses them onto one dimensionless ±1 curve. (Revert =
   plot raw MVAr and drop the overlay's quantitative meaning.)
3. **Cell 3 table built from scenario `summary` blocks, `comparison.json` as
   fallback** (spec implied comparison.json as the source). The `summary` key is
   guaranteed to be `summary_dict()` verbatim per `build_scenario_payload()`, includes
   plugin scenarios, and survives even when `comparison.json` was not written
   (`comparison_df` empty). Both sources are still loaded, per spec.
4. **Eighth output = `abstract_template.txt`.** The spec's Cells 1–8 produce seven
   files; "the eight output files" in Cell 9 is satisfied by saving the auto-filled
   abstract as the eighth artefact (also the most useful one to persist).
5. **Timestep resolution inferred from timestamps** everywhere (energy integration,
   daily/monthly grouping) instead of assuming the publisher docstring's 10-min or the
   SimBench 15-min. Protects subset runs and future LV runs at other resolutions.
6. **`t = 0` exclusion generalised** in Cell 6 exactly as specified, plus percentile
   clipping of the bin range (0.1–99.5 %) so a single pandapower cache-growth outlier
   (Sec. 6.3 of the knowledge base) cannot flatten the histogram.
7. **Scatter subsampling cap (200 k points, seeded RNG)** in Cell 8 — an annual run ×
   102 DERs is ~3.6 M points; the full set makes a multi-hundred-MB PDF with zero
   added visual information. The cap is printed when it triggers.
8. **Monthly-count *and* energy-based curtailment metrics**: Cell 5 gives the
   spec-literal sum of `curtailment_needed` (timestep counts); Extension 3 adds
   curtailed MWh using the fig13 definition, since energy is the unit reviewers expect
   for the abstract's `[X]%`. Both reductions are printed so the student can choose.
9. **`hil_latency_ms` overlay drawn for both runs** if both are HIL (spec mentioned
   only "if present in the JSON") — dry-run vs HIL is the normal case, and the overlay
   simply doesn't appear for runs where the field is null.
10. **Extension figures** (E1–E4) selected from `plot_results.py` per the extension
    request, adapted to the two-run layout, with `fig09`'s hard-coded 144 steps/day
    replaced by timestamp-based day grouping.

Not implemented (needs your call): per-figure PNG twins for every PDF (only
`voltage_envelope` has both, per spec), and `hc.json`-based HC sweep comparison — HC
sweep-curve storage in `HCResult` is still listed as **pending** in the knowledge base,
so the figure would usually render the degenerate terminal-points fallback.

## 6. Known limits

* Loading two full annual runs holds the raw `timeseries` payloads (with all per-bus
  dicts) in RAM — roughly the same footprint as `plot_results.py`, ~2–4 GB for two
  35 k-step MV runs. On a constrained machine, delete unneeded scenario JSONs from the
  run folder copies or run months at a time via `slice_profiles()` subsets.
* Month grouping on DST-spanning runs falls back to UTC timestamps (≤ 1 h shift at
  month boundaries — irrelevant at monthly aggregation, noted for completeness).
* `[Y]`/`[Z]` in the abstract follow the RUN_A = dry-run, RUN_B = HIL convention; the
  code comment in Cell 9 says to swap if your runs are reversed.
