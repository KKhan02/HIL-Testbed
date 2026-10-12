# TASK 4 — DOCUMENT 10C
## Output Subsystem, Part 3: Network Plots
### `network_plotter.py` — net-direct topology & profile figures (967 lines)
### Complete walkthrough (figure catalogue) — closes Document 10

> **Source verified:** read against the current `/mnt/project/network_plotter.py`
> (967 lines, the project tree). **Read alongside:** **10B** (`plot_results.py` — the
> *results* figures from published JSON; this module is different — see §1), **`profile_builder`**
> (profiles + extreme-day markers), **Document 9** (`run_benchmark_script` — the caller).

---

## TABLE OF CONTENTS

1. [Overview — net-direct documentation figures](#1-overview)
2. [Call-site map](#2-call-site-map)
3. [Section A — Constants & schematic lookup (L83–253)](#3-section-a--constants--schematic-lookup-l83253)
4. [Section B — Coordinate handling (L255–360)](#4-section-b--coordinate-handling-l255360)
5. [Section C — Plotting helpers (L362–500)](#5-section-c--plotting-helpers-l362500)
6. [Section D — plot_topology (L506–708)](#6-section-d--plot_topology-l506708)
7. [Section E — Profile & day figures (L710–967)](#7-section-e--profile--day-figures-l710967)
8. [Findings and flags](#8-findings-and-flags)
9. [Coverage checklist & Document 10 close-out](#9-coverage-checklist--document-10-close-out)

---

## 1. Overview

`network_plotter.py` is the **net-direct visualisation layer** — it draws figures straight
from a live pandapower net and a profiles dict, producing the "**document your network**"
artefacts: a topology figure, the annual profile figure, a single-day zoom, and the
extreme-day set. It is used at setup time (e.g. by `run_benchmark_script`) to verify and
illustrate the network being benchmarked.

**How it differs from 10B (`plot_results`).** They are complementary, not duplicative:

| | `network_plotter` (10C) | `plot_results` (10B) |
|--|------------------------|----------------------|
| Input | the **live net** + profiles dict | the **published JSON** |
| When | setup / documentation | post-run reporting |
| Content | topology, profiles, day zooms | voltage/violation/HC/summary results |
| Net access | reads the net directly | never touches the net |

**The defining feature is network-type awareness.** SimBench networks carry embedded
geodata (EPSG:31467), so `plot_topology` shows a geographic panel + a structured panel;
CIGRE/Kerber/Dickert/synthetic-LV networks have no geodata, so it shows a **reference
schematic PNG** (looked up by name) + a `simple_plot`, degrading to a single structured
plot when no PNG is found. Robust **3-tier coordinate extraction** (§4) handles the
pandapower 3.x geodata migration transparently, and the net is **never mutated** (structured
layouts are built on a deep copy).

---

## 2. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `plot_topology` | function | `run_benchmark_script.py` (the direct-script path, Doc 9) + standalone. |
| `plot_profiles` | function | `run_benchmark_script.py` + standalone. |
| `plot_network_and_profiles` | function | Convenience wrapper (backward compat). |
| `plot_day` / `plot_extreme_days` | functions | Standalone day-zoom diagnostics. |
| `get_bus_xy`, `find_schematic`, `auto_bus_size`, … | helpers | The figure functions. |

---

## 3. Section A — Constants & schematic lookup (L83–253)

`load_image` (L83) loads a schematic image (webp/png/jpg). Constants: the load/PV/wind
colours + fill alpha, `DEFAULT_SCHEMATIC_DIR = "data/schematics"`, `IMAGE_EXTENSIONS`, and
the two lookup tables — **`SCHEMATIC_MAP`** (net-name → schematic filename) and
**`SIMBENCH_KEYWORDS`**. `is_simbench` (L204) tests the net name against the keywords;
`find_schematic` (L209) resolves a non-SimBench net's reference PNG from `SCHEMATIC_MAP`
(with extension probing). **Design rationale:** SimBench is detected by name (it has
geodata, no PNG needed); every other network type maps to a hand-drawn schematic file, so
the topology figure always has a meaningful left panel.

---

## 4. Section B — Coordinate handling (L255–360)

The most substantive infrastructure — extracting bus coordinates across pandapower versions.

```python
def get_bus_xy(net) -> pd.DataFrame:      # 3-tier priority
    1. net.bus_geodata["x","y"]           (pandapower ≤2.x)
    2. net.bus["x","y"]                    (pandapower 3.x columns)
    3. net.bus["geo"] GeoJSON Point strings (CIGRE MV in pp 3.4)
```

`get_bus_xy` (L294–333) tries three coordinate sources in priority order, coercing to
numeric and dropping NaN, returning an empty frame if none yields ≥1 coordinate. **Design
rationale (docstring):** pandapower 3.x **migrated** bus coordinates — `bus_geodata` was
deprecated, coordinates moved to `net.bus["x"]/["y"]`, and CIGRE MV in 3.4 stores them as
**GeoJSON Point strings** in `net.bus["geo"]` — so a single robust extractor is needed for
the code to work across the networks and pandapower versions the project uses. `_parse_bus_geo`
(L265) decodes the GeoJSON. `refresh_geojson_from_xy` (L255) converts x/y back to `geo` for
pandapower-3.x `simple_plot` (with a try/except across the API's module move).
`needs_generic_coords` (L336, <2 real coords) and `auto_bus_size` (L340, marker size scaled
to the bounding box) build on it. **This is the same geodata-migration handling the KB flags
as a gotcha (`bus_geodata` removed → use `net.bus["x"]/["y"]`).**

---

## 5. Section C — Plotting helpers (L362–500)

Small shared helpers: `annotate_bus_ids` (bus-index labels), `apply_tiny_axis_padding` (pad
axes for tiny-extent nets), `aggregate` (sum a profile frame across elements → one series),
`topology_legend` (the element legend), `profile_axis` (draw one profile panel — a light
10-min fill + a bold daily-mean line, the readability pattern), and `annotate_extreme_days`
(mark the four extreme days on a profile axis). **Design rationale:** `profile_axis`'s
fill-plus-mean is the readability idiom — the raw 10-minute series as a translucent band with
the daily mean as a legible line, so a year of data is readable at a glance.

---

## 6. Section D — plot_topology (L506–708)

The network-type-aware topology figure — always two panels.

```python
    is_sb = is_simbench(net_name); show_schematic = (not is_sb) and find_schematic(...)
    # LEFT:  SimBench → geographic simple_plot (EPSG:31467); else → schematic PNG (or a "place PNG" note)
    # RIGHT: SimBench → deep-copy + create_generic_coordinates → structured simple_plot
    #        non-SimBench → simple_plot on the (coord-ensured) net
```

#### Line 506–708: The two-panel, type-aware topology

**SimBench:** left is the **geographic** layout from the embedded geodata (SimBench-paper
Fig 5 equivalent), right is a **structured** layout built on a **deep copy** with
`create_generic_coordinates(overwrite=True)` — the KB-mandated pattern, so the original net's
geodata is **never disturbed** (Fig 7 equivalent). **CIGRE/Kerber/etc:** left is the
**reference schematic PNG** if found, right is a pandapower `simple_plot`; with no PNG, a
"place a schematic here" note + a single structured plot. A post-draw pass restyles the sgen
marker patches (§8). **Design rationale:** the two-panel geographic-plus-structured (or
schematic-plus-structured) view gives both the *real spatial routing* and the *clean
topology* in one figure — and the deep-copy guarantees documentation plotting can never
perturb the network the benchmark then runs on.

---

## 7. Section E — Profile & day figures (L710–967)

- **`plot_profiles`** (L710–778) — the annual figure: three panels (aggregate load, PV, wind),
  each with the fill-plus-daily-mean idiom and the extreme-day markers annotated.
- **`plot_network_and_profiles`** (L784–814) — a convenience wrapper calling `plot_topology`
  then `plot_profiles` (kept for backward compatibility).
- **`plot_day`** (L820–924) — zooms to a single calendar day at **full 10-minute resolution**
  (any `YYYY-MM-DD`, or an `extreme_days` value), via a `_slice` helper.
- **`plot_extreme_days`** (L930–967) — calls `plot_day` for all four auto-detected extreme
  days (max/min load, max PV, max wind).

**Design rationale:** the annual view establishes the yearly pattern; the day zooms let a
reader inspect the 10-minute dynamics on the days that actually drive the worst voltages —
the same extreme-day markers `profile_builder`/`find_extreme_days` computed, so the
documentation and the results (10B fig08) reference the same days.

---

## 8. Findings and flags

**No functional bugs.** Minor notes only:

1. **The sgen-patch restyle heuristic (L580–587) is fragile.** After `simple_plot`, the code
   identifies sgen marker collections by a **linewidth heuristic** (`mean(lw) > 1.0`) to
   recolour them — which could restyle the wrong collection if pandapower's default linewidths
   change across versions. Cosmetic only (marker appearance); would fail safe (wrong colour,
   not a crash).
2. **CRLF line endings.** The file uses Windows CRLF (`\r\n`) — harmless on the RPi (Python
   tolerates it), a cleanliness note only.

**Not bugs (documented for clarity):** the network-type awareness is essential (SimBench has
geodata, others don't); the deep-copy for the structured panel is the mandatory no-mutation
pattern; the 3-tier `get_bus_xy` is the correct handling of the pandapower geodata migration;
the `refresh_geojson_from_xy` try/except defends against the pandapower geo-module API move.

---

## 9. Coverage checklist & Document 10 close-out

| Entity | Lines | Documented | Notes |
|--------|-------|:----------:|-------|
| `load_image` + constants + `SCHEMATIC_MAP`/`SIMBENCH_KEYWORDS` | 83–200 | ✅ | schematic lookup |
| `is_simbench` / `find_schematic` | 204–253 | ✅ | net-type + PNG resolve |
| `refresh_geojson_from_xy` / `_parse_bus_geo` / `get_bus_xy` | 255–333 | ✅ | 3-tier coord migration |
| `needs_generic_coords` / `auto_bus_size` | 336–360 | ✅ | |
| `annotate_bus_ids` / `apply_tiny_axis_padding` / `aggregate` / `topology_legend` / `profile_axis` / `annotate_extreme_days` | 362–500 | ✅ | plotting helpers |
| `plot_topology` | 506–708 | ✅ | type-aware two-panel |
| `plot_profiles` | 710–778 | ✅ | annual, fill+mean |
| `plot_network_and_profiles` | 784–814 | ✅ | wrapper |
| `plot_day` | 820–924 | ✅ | 10-min day zoom |
| `plot_extreme_days` | 930–967 | ✅ | four extreme days |

**Also covered (not standalone entities):** the net-direct-vs-JSON distinction from 10B, the
network-type-aware topology logic, the pandapower 3.x geodata-migration handling, the
no-mutation deep-copy pattern, the fill-plus-daily-mean readability idiom, and the shared
extreme-day markers.

**Flags raised (not edited):** fragile sgen-patch restyle heuristic (cosmetic); CRLF line
endings (cleanliness). No functional bugs.

---

### Document 10 — complete

Document 10 (the Output Subsystem) is fully documented across **10A–10C**:
- **10A** `publisher.py` — the JSON/JSONL output + crash-resume checkpointing (both NaN fixes
  and the held-open-handle SD-card fix confirmed applied).
- **10B** `plot_results.py` — the 14 report figures from published JSON (`cm.get_cmap`
  deprecation flagged).
- **10C** (this doc) — the net-direct topology/profile documentation figures.

*Remaining Task 4 queue: Document 9 — the orchestration layer (`benchmark_runner`, `executor`,
`wizard`, `run_plan`, `run_benchmark_script`, `helpers`, `__main__`).*
