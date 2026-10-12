# TASK 4 — DOCUMENT 10B
## Output Subsystem, Part 2: Report Figures
### `plot_results.py` — the 14 matplotlib report figures (1315 lines)
### Complete walkthrough (figure catalogue)

> **Source verified:** read against the current `/mnt/project/plot_results.py`
> (1315 lines, the project tree). **Read alongside:** **10A** (`publisher.py` — this
> module consumes its JSON output), **10C** (`network_plotter.py` — the live/on-net
> plots, distinct from these post-run report figures).
>
> **⚠ Version-dependent flag:** `cm.get_cmap` (L208) is deprecated in matplotlib ≥3.7
> and **removed in ≥3.9** (§9).

---

## TABLE OF CONTENTS

1. [Overview — the report figure layer](#1-overview)
2. [Call-site map](#2-call-site-map)
3. [Section A — Style & scenario constants (L74–126)](#3-section-a--style--scenario-constants-l74126)
4. [Section B — Loading, coordinate & drawing infrastructure (L133–255)](#4-section-b--loading-coordinate--drawing-infrastructure-l133255)
5. [Section C — The 14-figure catalogue (L262–1181)](#5-section-c--the-14-figure-catalogue-l2621181)
6. [Section D — The plot_results orchestrator (L1188–1258)](#6-section-d--the-plot_results-orchestrator-l11881258)
7. [Findings and flags](#7-findings-and-flags)
8. [Coverage checklist](#8-coverage-checklist)

---

## 1. Overview

`plot_results.py` generates the **static report figures** — the SimBench/pandapower-style
figures for the paper and slides. Its defining architectural property is **decoupling**:
it reads *only* the JSON the publisher wrote (`load_publisher_dir`), never a live
pandapower net. So figures can be regenerated any time from a run's output directory,
on any machine, without re-running the benchmark — and the figure layer cannot perturb
the simulation.

Structure: a global IEEE style + consistent scenario colour/label maps (§3); shared
loading/coordinate/drawing helpers (§4); **14 figure functions** `fig01`–`fig14`, each a
pure `(payload dicts) -> saved PNG` (§5); and the `plot_results` orchestrator that detects
the coordinate type, runs the requested figures with **per-figure exception isolation**,
and handles the HC-only-run case (§6).

**Consistency by construction:** `SCENARIO_ORDER`/`SCENARIO_LABELS`/`SCENARIO_COLORS` are
defined once and used by every figure, so a scenario is the same colour and label across
all 14 — essential for a coherent figure set in one paper.

---

## 2. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `plot_results` | function | The CLI / a post-run figure step (Doc 9) and standalone (regenerate figures from a saved output dir). |
| `load_publisher_dir` | function | `plot_results` — reads `topology/profiles/hc/comparison/scenarios/*.json` (10A output). |
| `fig01`–`fig14` | functions | `plot_results`'s figure registry (dispatched by number). |
| `_draw_network_base`, `_get_bus_positions`, `_vn_color_map`, `_add_basemap`, `_save` | helpers | Shared by the network-map figures. |

---

## 3. Section A — Style & scenario constants (L74–126)

The module sets a global matplotlib **IEEE single-column style** (`rcParams`: serif, 10 pt,
7.16″ width, 300 dpi, tight bbox) so every figure is publication-ready without per-figure
styling. `SCENARIO_ORDER` (baseline→oltc→svc→volt_var_local→volt_var_coord→opf),
`SCENARIO_LABELS`, and `SCENARIO_COLORS` are the **single source of scenario styling**;
`SGEN_TYPE_COLORS` colours PV/wind/other; `_VN_PALETTE` + `_vn_color_map` colour buses by
voltage level (red HV slack, blue MV — immediately distinguishable). `V_MIN/V_MAX` (0.95/1.05)
draw the planning band. **Design rationale:** centralising style + colour makes the whole
figure set coherent and lets a reviewer read scenario identity by colour alone.

---

## 4. Section B — Loading, coordinate & drawing infrastructure (L133–255)

- **`load_publisher_dir`** (L138–155) — loads the published JSON into one dict
  (`topology`, `profiles`, `hc`/`comparison` if present, and every `scenarios/*.json`).
  **The decoupling boundary:** figures consume this, not a net.
- **`_is_geographic`** (L162–171) — detects WGS84 lat/lon coordinates (SimBench) vs
  schematic, by range-checking (`|x|≤180`, `|y|≤90`, span < 5°). Drives whether an OSM
  basemap is added.
- **`_get_bus_positions`** (L174–184) — bus coordinates, with a **networkx `spring_layout`
  fallback (`seed=42`, reproducible)** when a network has no coordinates (Kerber/Dickert) —
  so every network is drawable.
- **`_draw_network_base`** (L191–237) — the shared line/trafo renderer with a **3-priority
  line-colour rule**: a continuous colourmap from `line_values` (e.g. loading %), else a
  discrete `vn_kv` colour, else the default grey; trafos are dashed. Returns the norm/cmap
  for a colourbar.
- **`_add_basemap`** (L240–248) — adds an OSM basemap via `contextily` when geographic *and*
  the optional dependency is present (guarded, no-op otherwise).
- **`_save`** (L251–255) — `savefig` + `close` + a progress print.

**Design rationale:** the network-map figures (1/2/4/…) all reuse `_draw_network_base` +
`_get_bus_positions`, so their layout and colouring are identical and only the per-figure
overlay differs.

---

## 5. Section C — The 14-figure catalogue (L262–1181)

Each `figNN` is a pure function reading the payload(s) and saving one PNG. Grouped by role:

**Network maps (spatial):**
- **`fig01_network_generation_map`** (L262) — DER locations sized/coloured by type on the
  network base (SimBench Fig 2 left).
- **`fig02_network_line_loading_map`** (L313) — line loading % as a colourmap overlay, per
  scenario (worst-case snapshot).
- **`fig03_installed_capacity`** (L361) — installed DER capacity bar chart by type.
- **`fig04_network_topology`** (L404) — the network coloured by voltage level (the topology
  reference figure).

**Voltage quality (time / space):**
- **`fig05_voltage_heatmap`** (L452) — bus-vs-time voltage heatmap per scenario.
- **`fig06_voltage_vs_feeder_dist`** (L505) — voltage against topological feeder distance
  (uses `topology["feeder_dist"]`, 10A) — the classic droop-along-the-feeder view.

**Profiles & time series:**
- **`fig07_annual_profiles`** (L586) — the annual load/PV/wind profiles.
- **`fig08_timeseries_panels`** (L634, the largest) — multi-panel per-scenario time series
  (voltage extremes, loading, control action) over an extreme day, downsampled hourly;
  `_extreme_day_slice` picks the window from the profiles' extreme-day markers.

**Violations & control behaviour:**
- **`fig09_violation_heatmap`** (L780) — violation incidence across scenarios/time.
- **`fig10_qv_scatter`** (L822) — the realised Q vs V operating points (the Volt-Var
  characteristic as actually exercised).
- **`fig12_coordination_scatter`** (L932) — coordinator activity (per-DER Q adjustments).
- **`fig13_curtailment_timeseries`** (L992) — active-power curtailment over time.

**Hosting capacity & summary:**
- **`fig11_hc_sweep`** (L880) — the HC sweep (MW added vs binding voltage), baseline vs
  volt_var, from `hc.json`.
- **`fig14_benchmark_summary`** (L1062) — the headline two-panel summary **from the
  comparison CSV**: Panel A is **% improvement over baseline** (sign convention
  `(baseline − value)/|baseline|·100`, so positive = fewer violations / lower VDI / lower
  losses), capped at ±150% with annotation of out-of-range bars; Panel B is **absolute
  values** for metrics baseline doesn't define (curtailed/reactive energy). Panels
  auto-hide when their metric set is empty.

**Design rationale — figures map to paper figures:** the numbering and content mirror the
SimBench/pandapower reference figures the paper reproduces, so the deliverable is a drop-in
figure set. Each figure degrades gracefully (skips with a printed reason if its input —
`hc`, `comparison`, an extreme-day marker — is absent).

---

## 6. Section D — The plot_results orchestrator (L1188–1258)

```python
    data = load_publisher_dir(pub_dir); topology/profiles/hc/scenarios = data[...]
    if hc_pub_dir and not scenarios and hc_data["scenarios"]: scenarios = hc_data["scenarios"]  # HC-only merge
    geo = _is_geographic(buses)
    all_figs = {1: lambda: fig01(...), ..., 14: lambda: fig14(csv_path, out_dir)}
    for n in (figures or all keys):
        try: all_figs[n]()
        except Exception as exc: print(f"  fig{n:02d} ERROR: {exc}")
```

#### Line 1188–1258: Load, detect, dispatch with per-figure isolation

Loads the published dir, detects the **coordinate type** (geographic → OSM basemap if
`contextily` present; schematic; or none → networkx layout), and runs the requested figures
(all, or a caller-selected subset) from a **number → lambda registry**. **Two design
rationales:** (1) **per-figure `try/except`** — one figure failing (a missing payload, a
matplotlib issue) prints an error and continues, so a run never loses the other 13 figures
(the same isolation principle as `benchmark_runner`); (2) the **HC-only merge** — when the
outer run had `scenarios=[]` (a hosting-capacity-only run), it pulls the HC-stressed
scenarios from a second publisher dir so the scenario-dependent figures still render.

---

## 7. Findings and flags

**Flagged, not edited:**

1. **`cm.get_cmap` is deprecated/removed (L208, in `_draw_network_base`).** matplotlib
   deprecated `matplotlib.cm.get_cmap` in 3.7 and **removed** it in 3.9 (use
   `matplotlib.colormaps[name]` or `plt.get_cmap`). On a modern matplotlib this raises, and
   because the orchestrator isolates per figure, **every network-map figure that uses
   `_draw_network_base` (1/2/4/…) would fail with a printed error while the others succeed** —
   easy to miss. **Recommend:** `cmap_fn = plt.get_cmap(line_cmap)`. *(Depends on the
   installed matplotlib version — verify against the project's pinned version.)*

**Not bugs (documented for clarity):** the JSON-consuming decoupling is a deliberate design
strength (figures never touch the net); the per-figure exception isolation is intentional;
the networkx-layout fallback and graceful per-figure skips make the module robust across
networks with/without coordinates and runs with/without HC or comparison data.

---

## 8. Coverage checklist

| Entity | Lines | Documented | Notes |
|--------|-------|:----------:|-------|
| IEEE style + scenario/type/vn constants | 74–126 | ✅ | single source of styling |
| `_vn_color_map` | 123–126 | ✅ | |
| `_load` / `load_publisher_dir` | 133–155 | ✅ | the decoupling boundary |
| `_is_geographic` / `_get_bus_positions` | 162–184 | ✅ | geo detect + layout fallback |
| `_draw_network_base` | 191–237 | ✅ | `cm.get_cmap` flagged |
| `_add_basemap` / `_save` | 240–255 | ✅ | |
| `fig01`–`fig07` | 262–617 | ✅ (catalogue) | maps + profiles |
| `fig08_timeseries_panels` (+ `_extreme_day_slice`) | 622–775 | ✅ | largest; extreme-day panels |
| `fig09`–`fig13` | 780–1057 | ✅ (catalogue) | violations / qv / hc / coord / curtailment |
| `fig14_benchmark_summary` | 1062–1181 | ✅ | %-improvement + absolute panels |
| `plot_results` | 1188–1258 | ✅ | orchestrator, per-figure isolation |

**Also covered (not standalone entities):** the JSON-consuming decoupling, the centralised
scenario styling, the 3-priority line colouring, the geographic/schematic/networkx coordinate
handling, the figures-map-to-paper-figures rationale, the HC-only-run merge, and per-figure
graceful degradation.

**Flags raised (not edited):** `cm.get_cmap` deprecation/removal (L208, version-dependent).

---

*End of Document 10B. Next: Document 10C — `network_plotter.py` (967 lines): the network
topology/state plots (distinct from these post-run report figures).*
