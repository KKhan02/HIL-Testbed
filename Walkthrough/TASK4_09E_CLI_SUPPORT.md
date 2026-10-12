# TASK 4 — DOCUMENT 9E
## Orchestration Layer, Part 5: CLI Support Modules
### `network_catalogue.py` (258) + `_console.py` (29) + `_data.py` (448) + `test_helpers.py` (117)
### Complete line-by-line walkthrough — closes Document 9 & the Task 4 set

> **Source verified:** read against the current `/mnt/project/` copies. **Read
> alongside:** **9C** (`wizard` — reads the catalogue), **9B** (`executor` — resolves
> `preset_name`), **10A** (`publisher` — `_data.py` is the *read* side of its *write*
> side), **9D** (`helpers` — uses the shared `console`).

---

## TABLE OF CONTENTS

1. [Overview — the support layer](#1-overview)
2. [Call-site map](#2-call-site-map)
3. [Section A — network_catalogue.py (L1–258)](#3-section-a--network_cataloguepy-l1258)
4. [Section B — _console.py (L1–30)](#4-section-b--_consolepy-l130)
5. [Section C — _data.py (L1–448)](#5-section-c--_datapy-l1448)
6. [Section D — test_helpers.py (L1–117)](#6-section-d--test_helperspy-l1117)
7. [Findings and flags](#7-findings-and-flags)
8. [Coverage checklist & Document 9 / Task 4 close-out](#8-coverage-checklist--document-9--task-4-close-out)

---

## 1. Overview

Four small support modules that the CLI and dashboard lean on:

- **`network_catalogue.py`** — the wizard's **preset menu data** (family → preset entries).
  Data only; it does *not* load networks (the executor resolves `preset_name`).
- **`_console.py`** — the **shared Rich theme + single `console` instance** every
  terminal-facing module imports, so all CLI output is styled consistently.
- **`_data.py`** — the **dashboard's read layer**: the exact mirror of `publisher.py` (10A).
  Where the publisher *writes* topology/profiles/scenarios/hc/comparison JSON and the
  live/checkpoint JSONL streams, `_data.py` *discovers and reads* them for the Streamlit
  pages — including the **incremental live-stream follower** that pairs with the publisher's
  live writer.
- **`test_helpers.py`** — a **smoke test** that builds a dummy `RunPlan` and calls all five
  console helpers, so the presentation layer can be eyeballed without a real run.

The unifying theme is **read/produce mirrors of the write side**: the catalogue produces the
menu the executor consumes; `_data.py` reads exactly what the publisher writes; `_console.py`
centralises the styling the helpers render.

---

## 2. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `get_preset_families` / `get_presets_for_family` | fns | `wizard._ask_network` (9C) — builds the preset menu. |
| `_PRESET_CATALOGUE` | data | The two accessors. (`preset_name` → resolved by `executor._preset_loaders`, 9B.) |
| `console` / `CLI_THEME` | instance/theme | `helpers` (9D), `wizard` (9C), `executor` (9B), `__main__` (9D). |
| `discover_runs` / `load_*` / `read_jsonl_since` / `sidebar_run_selector` | fns | The Streamlit dashboard pages (`Home.py`, `1_Scenario_Comparison.py`, `5_Run_Health_and_Logs.py`, …). |
| `build_dummy_run_plan` / `main` | fns | `python -m …test_helpers` (dev smoke test). |

---

## 3. Section A — network_catalogue.py (L1–258)

```python
PresetEntry     = dict[str, str]                 # {"label", "preset_name", "preset_family"}
PresetCatalogue = dict[str, list[PresetEntry]]   # family → entries
_PRESET_CATALOGUE = { "Simbench": [{...}], "CIGRE": [{...},{...}], "Kerber": [...], "Dickert": [...], "Synthetic LV": [...] }
def get_preset_families() -> list[str]:                 return list(_PRESET_CATALOGUE.keys())
def get_presets_for_family(family) -> list[PresetEntry]: return _PRESET_CATALOGUE[family]
```

#### Line 1–258: The preset menu catalogue

A nested dict mapping a **network family** (SimBench, CIGRE, Kerber, Dickert, Synthetic LV) to
a list of **`PresetEntry`** dicts, each carrying a human `label` (e.g. "Simbench 1-MV-rural--2-sw,
primary MV HIL network, 99 buses, 102 DERs"), a machine `preset_name`, and the `preset_family`.
`get_preset_families` returns the families in menu order; `get_presets_for_family` returns one
family's entries. **The load-bearing design decision (docstring): this module stores menu data
ONLY — it does not import pandapower or load any network.** The `preset_name` string is later
resolved to an actual net by the executor's `_preset_loaders` (9B). **Design rationale:**
separating the *menu* (what the wizard shows) from the *loading* (what the executor does) keeps
the wizard free of the heavy pandapower/SimBench imports — the wizard can present a rich network
menu instantly, and only the executor (at run time) pays the network-construction cost. The
`label` strings pre-state each network's size and DER count so a user picks knowledgeably from
the menu alone. **Worked example:** `get_presets_for_family("CIGRE")` returns the two entries
(`cigre_mv_no_der`, `cigre_mv_pv_wind`); the wizard shows their labels, the user picks, and the
chosen `preset_name` lands in `NetworkConfig.preset_name` (9C) for the executor to resolve.

---

## 4. Section B — _console.py (L1–30)

```python
CLI_THEME = Theme({"header":"bold cyan","rule":"cyan","stage":"bold yellow","ok":"green",
                   "warning":"yellow","error":"bold red","muted":"grey50","algorithm":"cyan",
                   "overvoltage":"yellow","undervoltage":"blue","overload":"magenta","current":"bold default"})
console = Console(theme=CLI_THEME)
```

#### Line 1–30: The shared theme and single console

Defines the **`CLI_THEME`** (a Rich `Theme` mapping semantic style names to colours) and
constructs the **single `Console` instance** every terminal-facing module imports. **Code-to-UX
mapping:** the style names are *semantic*, not colour literals — `[stage]` for section banners,
`[ok]`/`[warning]`/`[error]` for status, and the domain-specific `[overvoltage]` (yellow),
`[undervoltage]` (blue), `[overload]` (magenta) matching the same colour language the plots use.
**Design rationale — one console, one theme:** importing a shared `console` (rather than each
module making its own) guarantees consistent styling, a single output stream, and that a change
to the palette propagates everywhere at once. The semantic names mean a module writes
`console.print("[overvoltage]…")` and never hardcodes a colour — the theme owns the mapping.

---

## 5. Section C — _data.py (L1–448)

The dashboard's data-access layer — the **read mirror of `publisher.py`** (10A). Walked by group.

#### Line 40–77: Shared scenario constants & helpers

`SCENARIO_ORDER`/`SCENARIO_LABELS`/`SCENARIO_COLORS`/`SGEN_TYPE_COLORS` — **the same scenario
styling as `plot_results`** (10B) — plus `scenario_label`/`scenario_color`/`sort_scenarios`.
**Design rationale:** the dashboard and the static figures must colour and order scenarios
identically; centralising the maps here (imported by the pages) guarantees it.

#### Line 80–138: Project-root detection & run discovery

`detect_project_root` resolves the root in priority order — **`$HIL_DASHBOARD_ROOT`** →
walk-up for a folder containing `outputs/` or `runs/` → grandparent → cwd — so the dashboard
finds runs regardless of where it is launched. `_discover`/`discover_runs` then **glob six
`_RUN_PATTERNS`** covering every layout the two entry paths produce (`outputs/publisher/*/` from
the script, `runs/*/publisher/*/` from the executor, plus legacy/defensive patterns), keying on
any `topology.json` (its parent is the run) and **de-duplicating by resolved path, keeping the
shortest label**. `@st.cache_data` memoises the scan. **Design rationale:** the script and
executor write to *different* directory shapes (9D), so the reader must recognise both — the
ordered pattern list is that reconciliation.

#### Line 144–191: mtime-keyed JSON loaders

`_load_json(path, mtime)` is `@st.cache_data`-memoised **keyed on path *and* mtime**, so
`_read` (which passes `path.stat().st_mtime`) returns the cached parse until the file changes,
then re-parses. `load_topology`/`load_profiles`/`load_hc`/`load_comparison`/`load_scenario`/
`list_scenarios`/`network_id_of` wrap it for each publisher artefact; `find_benchmark_csv`
locates fig14's CSV across the CLI and script locations (excluding the HC-stressed file). **The
mtime-keying is the key idea:** re-running a scenario rewrites its JSON with a new mtime, which
**invalidates exactly that cache entry** — so the dashboard live-refreshes a re-run scenario
without a manual cache clear and without re-reading unchanged files. **This is the precise read
counterpart to the publisher's per-scenario write** (10A §10).

#### Line 283–342: The live-stream follower — list_live_scenarios & read_jsonl_since

```python
def read_jsonl_since(path, offset) -> (frames, new_offset, reset):
    if path.stat().st_size < offset: offset, reset = 0, True     # file shrank → truncated → reset
    fh.seek(offset); chunk = fh.read()                          # binary mode: exact byte offsets
    last_nl = chunk.rfind(b"\n");  complete = chunk[:last_nl+1]  # only whole lines
    for line in complete.split(b"\n"): frames.append(json.loads(line))   # skip torn lines
    return frames, offset + len(complete), reset                # leave the partial tail for next tick
```

The **incremental live follower** — the read-side counterpart to the publisher's held-open live
writer (10A §9). It reads **only the bytes appended after `offset`**, in **binary mode** so byte
offsets are exact, parses **only complete newline-terminated lines** (a trailing partial line
from a mid-append writer is left for next tick), and skips a torn line (re-read intact next
poll). **Truncation detection:** if the file is now *smaller* than `offset`, the live file was
reset by `on_scenario_start` (the publisher truncates `live/<id>.jsonl` fresh each run, 10A), so
it returns **`reset=True`** telling the dashboard to discard its accumulated history and start
over. `list_live_scenarios` lists the live files (including **empty** ones — an empty file means
"streaming, first frame pending", since baseline/OPF flush only at the end). **Worked example:**
the dashboard holds `offset=1024`; the writer has appended 5 complete frames + 1 partial (file
now 2048 B). `read_jsonl_since` reads bytes 1024–2048, finds the last `\n`, parses the 5 frames,
returns `(5 frames, offset=~2000, reset=False)`, and leaves the ~48-byte partial for the next
poll. If instead `on_scenario_start` truncated the file to 0 B, `0 < 1024` → `reset=True` and the
dashboard clears its buffer. **Design rationale:** a live dashboard must tail an actively-written
file *safely* — never reading a half-written line, never re-reading old lines, and correctly
noticing a fresh run — which is exactly what the offset + last-newline + size-shrink logic
delivers.

#### Line 348–448: Log discovery/tail & the run selector

`discover_logs` finds candidate logs (CLI `session.log`, script `outputs/logs/session_*.log`,
optional `streamlit*.log`), most-relevant first, de-duplicated by resolved path. `tail_lines`
returns the last *n* lines by reading **at most the final `max_bytes`** (dropping the first,
possibly-partial line) — so tailing a huge annual log stays cheap. `sidebar_run_selector` is the
Streamlit widget that lists `discover_runs` and returns the chosen run's `Path`. **Design
rationale:** the same "read the tail cheaply, follow safely" discipline as the JSONL follower,
applied to logs — the dashboard never loads a multi-hundred-MB log in full.

---

## 6. Section D — test_helpers.py (L1–117)

```python
def build_dummy_run_plan() -> RunPlan:
    return RunPlan(parameters=ParameterConfig(der_scaling={None:1.25, 18:0.80}, load_scaling={None:1.10}, …),
                   network=NetworkConfig(source_type="simbench_code", simbench_code="1-MV-rural--2-sw", …),
                   dataset=DatasetConfig(…), study=…, …)
def main():
    plan = build_dummy_run_plan()
    print_run_plan(plan); print_section_header(plan, "…"); print_timestep_line(…)
    print_error_message(TimeoutError("No response from Arduino …"), context="…"); print_summary_table({…})
```

#### Line 1–117: The console-helpers smoke test

`build_dummy_run_plan` constructs a **representative** `RunPlan` — notably exercising the
scaling-dict sentinel (`der_scaling={None:1.25, 18:0.80}` — whole-network ×1.25 with bus 18
overridden, 9C §4) — and `main` calls **all five** `helpers.print_*` functions with realistic
data (including a simulated Arduino `TimeoutError` for `print_error_message` and a populated
summary dict). **Design rationale:** the presentation layer has no return values to unit-test —
its correctness is *visual* — so a smoke test that renders every helper once lets a developer
eyeball the formatting (colours, alignment, the run-plan echo) without standing up a real
network, hardware, or benchmark. It doubles as living documentation of each helper's expected
input shape.

---

## 7. Findings and flags

**No bugs found.** All four modules are clean. Points worth recording as *strengths*:

1. **`read_jsonl_since` is robust tail-following** — binary offsets, whole-lines-only, torn-line
   skipping, and size-shrink truncation detection make it safe against every state an
   actively-written live file passes through (mid-append, reset, gap).
2. **mtime-keyed caching** gives the dashboard automatic live-refresh of re-run scenarios with
   no manual invalidation — the correct pairing with the publisher's per-scenario writes.
3. **The catalogue's data-only discipline** keeps heavy imports out of the wizard.
4. **The six-pattern run discovery** correctly reconciles the two entry paths' differing output
   directory shapes (9D).

**Not bugs (documented for clarity):** including empty `live/<id>.jsonl` files in
`list_live_scenarios` is intentional (empty = "first frame pending"); the shared single
`console` is deliberate (consistent styling); `test_helpers` is a dev utility, not shipped
runtime.

---

## 8. Coverage checklist & Document 9 / Task 4 close-out

| Entity | File · Lines | Kind | Documented | Notes |
|--------|--------------|------|:----------:|-------|
| `PresetEntry`/`PresetCatalogue`/`_PRESET_CATALOGUE` | network_catalogue 10–248 | types/data | ✅ | menu data only |
| `get_preset_families` / `get_presets_for_family` | network_catalogue 250–256 | fns | ✅ | accessors |
| `CLI_THEME` / `console` | _console 12–30 | theme/instance | ✅ | shared styling |
| scenario constants + `scenario_label`/`color`/`sort` | _data 40–77 | data/fns | ✅ | shared with 10B |
| `detect_project_root` / `_discover` / `discover_runs` | _data 80–138 | fns | ✅ | 6-pattern discovery |
| `_load_json`/`_read`/`load_*`/`list_scenarios`/`network_id_of`/`find_benchmark_csv` | _data 144–219 | fns | ✅ | mtime-keyed loaders |
| `FIG_TITLES`/`KEY_FIGS`/`figure_title`/`find_figures_dir` | _data 220–282 | data/fns | ✅ | figure discovery |
| `list_live_scenarios` / `read_jsonl_since` | _data 283–342 | fns | ✅ | **live follower** |
| `discover_logs` / `tail_lines` / `sidebar_run_selector` | _data 348–448 | fns | ✅ | logs + selector |
| `build_dummy_run_plan` / `main` | test_helpers 20–117 | fns | ✅ | smoke test |

**Also covered (not standalone entities):** the read/produce-mirror theme, the catalogue's
data-only discipline, the mtime-keyed cache pairing with the publisher, the robust
byte-offset live follower and its truncation detection, and the two-entry-path run discovery.

**Flags raised:** none (no bugs). Design strengths recorded in §7.

### Document 9 — complete

Document 9 (the Orchestration Layer) is fully documented across **9A–9E**:
- **9A** `benchmark_runner.py` — the scenario orchestrator (registry, two-layer resume, HC recursion).
- **9B** `executor.py` — the RunPlan→run executor (typed failure identity, the late-binding limits rebind).
- **9C** `wizard.py` + `run_plan.py` — the interactive builder + the pure-data config classes.
- **9D** `__main__.py` + `run_benchmark_script.py` + `helpers.py` — the two entry paths + console output.
- **9E** (this doc) — the CLI support modules (catalogue, theme, dashboard data layer, smoke test).

### Task 4 — walkthrough set complete

With Document 9 closed, the Task 4 line-by-line walkthrough set now spans the **entire
codebase**: the control stack (1–5), the scenario runners (6A–6E), the orchestration layer
(9A–9E), the output subsystem (10A–10C), the violation detector (11), and the plugin subsystem
(12A–12D) — every module documented, every entity enumerated in a coverage checklist, all
against the current `/mnt/project` source, with the accumulated findings recorded in the master
index. The remaining known-open items are the previously-flagged **stale docstrings/comments**
and **cosmetic issues** (awaiting Khuzaim's go-ahead before any Python edit), not undocumented
code.

---

*End of Document 9E — and of the Task 4 walkthrough set. See `TASK4_00_MASTER_INDEX.md` for the
full document map, status tracker, and consolidated findings log.*
