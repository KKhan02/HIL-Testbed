# TASK 4 — DOCUMENT 9C
## Orchestration Layer, Part 3: The Wizard & Run Plan
### `run_plan.py` (466 lines) + `wizard.py` (958 lines)
### Complete line-by-line walkthrough

> **Source verified:** read against the current `/mnt/project/run_plan.py` (466 lines)
> and `/mnt/project/wizard.py` (958 lines). **Read alongside:** **9B** (`executor.py` —
> consumes the `RunPlan` this produces; `validate_plan` does the *cross-field* checks
> the wizard deliberately omits), **9D** (`__main__.py` — calls `run_wizard()` then
> `execute()`).

---

## TABLE OF CONTENTS

1. [Overview — the declarative-config layer + its interactive builder](#1-overview)
2. [File logic flow — pseudocode](#2-file-logic-flow--pseudocode)
3. [Call-site map](#3-call-site-map)
4. [PART 1 — run_plan.py](#part-1--run_planpy)
   - [Section A — ParameterConfig (L4–106)](#4-section-a--parameterconfig-l4106)
   - [Section B — NetworkConfig (L109–207)](#5-section-b--networkconfig-l109207)
   - [Section C — DatasetConfig (L210–326)](#6-section-c--datasetconfig-l210326)
   - [Section D — RunPlan (L328–466)](#7-section-d--runplan-l328466)
5. [PART 2 — wizard.py](#part-2--wizardpy)
   - [Section E — Back-navigation primitives (L13–132)](#8-section-e--back-navigation-primitives-l13132)
   - [Section F — The per-field prompt builders (L133–855)](#9-section-f--the-per-field-prompt-builders-l133855)
   - [Section G — run_wizard: the step machine (L856–959)](#10-section-g--run_wizard-the-step-machine-l856959)
6. [Findings and flags](#11-findings-and-flags)
7. [Coverage checklist](#12-coverage-checklist)

---

## 1. Overview

These two files are the **declarative-configuration layer** and its **interactive
builder**. Together with the executor (9B) they form a clean three-layer separation:

```
wizard.py  ──(interactive, per-field validation)──▶  RunPlan  ──(consumed)──▶  executor.py
                                                    (pure data)               (cross-field validation + run)
```

- **`run_plan.py`** defines four dataclasses — `ParameterConfig`, `NetworkConfig`,
  `DatasetConfig`, and the top-level `RunPlan` that nests them. A `RunPlan` is a **pure data
  container**: saving it to JSON and reloading via `from_dict` reproduces a run *identically*,
  with no code change. It **does not validate itself** — that is the wizard's job (per-field)
  and the executor's job (cross-field).
- **`wizard.py`** is the interactive TUI that *builds* a `RunPlan`, one field at a time, with
  **back-navigation** (any prompt can return to the previous step). Each prompt validates its
  own field in isolation; the cross-field checks are deferred to the executor (9B §9).

Two ideas recur: **reproducibility** (every config has `to_dict`/`from_dict` with
forward-compatible `.get()` defaults, so an old preset never breaks on reload) and
**separation of concerns** (data / interaction / validation live in three different places,
each simple and testable).

---

## 2. File logic flow — pseudocode

```
run_plan.py:
  @dataclass ParameterConfig  (v_min/v_max, thermal/angle/unbalance limits, der/load scaling dicts,
                               timestep_resolution, Q(V) overrides) + to_dict/from_dict
  @dataclass NetworkConfig    (source_type ∈ preset/simbench_code/custom/plugin + fields;
                               der_placements, switches_to_flip) + to_dict/from_dict
  @dataclass DatasetConfig    (source_type ∈ simbench_native/dwd/custom + data_dir/station/year/maps) + to_dict/from_dict
  @dataclass RunPlan          (nests the three + study/focus_buses/hardware/port/plugin/hc_stressed/
                               stream_every_k/run_id/output_dir) + to_dict/from_dict

wizard.py:
  BackRequested / BACK_TOKEN "<"
  _ask(menu) / _ask_value(free-text)      — both raise BackRequested on the back token
  _ask_study / _ask_network / _ask_dataset / _ask_parameters / _ask_hardware /
  _ask_controller_plugin / _ask_network_modifications / _ask_time_window   — per-field builders
  run_wizard(): ordered [step] list walked by index; BackRequested → index-1; assemble RunPlan
```

---

## 3. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `run_wizard` | function | `__main__.py` (9D) — the interactive entry; returns a `RunPlan`. |
| `RunPlan` (+ nested configs) | dataclasses | Produced by the wizard; consumed by `executor.execute` (9B); persisted by `_save_run_plan_copy`. |
| `RunPlan.from_dict` / `*.from_dict` | classmethods | Reloading a saved preset (reproducibility). |
| `RunPlan.to_dict` / `*.to_dict` | methods | `_save_run_plan_copy` (9B), preset saving. |
| `_ask` / `_ask_value` | helpers | Every per-field prompt builder. |
| `BackRequested` | exception | Raised by prompts, caught only by `run_wizard`'s loop. |

---

# PART 1 — run_plan.py

## 4. Section A — ParameterConfig (L4–106)

```python
@dataclass
class ParameterConfig:
    v_min=0.95; v_max=1.05; line_max_loading=100; trafo_max_loading=100
    va_diff_max_degree=30; unbalance_max_percent=2.0
    der_scaling  = field(default_factory=lambda: {None: 1.0})    # {bus→factor}, None key = whole network
    load_scaling = field(default_factory=lambda: {None: 1.0})
    timestep_resolution=15
    q_ratio=None; u1_pu=None; u2_pu=None; u3_pu=None; u4_pu=None  # Q(V) overrides; None = framework default
```

#### Line 4–47: The numerical-parameter record

Holds every numerical constraint and scaling parameter for a run, with **standards-anchored
defaults** (voltage 0.95–1.05 pu per EN 50160 / VDE-AR-N 4100; thermal 100 % of rated; angle
30° per VDE-AR-N 4110; unbalance 2 % per IEC 62749). **Code-to-physics — the scaling
sentinel:** `der_scaling`/`load_scaling` map `bus index → factor`, and the **`None` key means
"whole network"**. **Worked example:** `{None: 1.5}` scales all DERs ×1.5; `{12: 0.8, 15: 0.8}`
scales only buses 12 and 15; `{None: 1.5, 12: 0.8}` scales all ×1.5 then overrides bus 12 to
×0.8. The **Q(V) override fields** (`q_ratio`, `u1_pu`..`u4_pu`) are `None` by default (use the
`volt_var_controller` framework defaults `Q_RATIO=0.25`, `U1..U4=0.96/0.99/1.01/1.04`); when
set, the executor's `apply_qv_overrides` (9B) pushes them to the dry-run curve, the coordinator
sizing, **and** the Arduino CFG — "one value, three consumers" (the comment's phrase). **Design
rationale — `default_factory` for the scaling dicts:** a mutable default must use
`field(default_factory=...)` or every instance would share one dict; the frozen sentinel
`{None: 1.0}` means "no scaling".

#### Line 49–106: to_dict / from_dict — reproducibility with forward-compat

`to_dict` is `dataclasses.asdict(self)` (also invoked recursively by `RunPlan.to_dict`).
`from_dict` reconstructs with **`.get(key, default)` per field**, so an **older preset missing
a newly-added field does not `KeyError`** — a deliberate forward-compatibility guarantee.
**Two subtleties documented in the code:** (1) the scaling fields fall back to **`{None: 1.0}`
not `None`** (a `None` would `TypeError` when the executor iterates the scaling dict); (2) the
**`"null" → None` key coercion** — JSON serialises the `None` dict key as the string `"null"`,
so `from_dict` maps it back: `{(None if k == "null" else k): v for k, v in ...}`. **Worked
example:** a saved preset with `"der_scaling": {"null": 1.5, "12": 0.8}` reloads as
`{None: 1.5, "12": 0.8}` — the whole-network sentinel restored. **Design rationale:** a preset
must round-trip *exactly*, including the sentinel key that JSON can't natively represent.

---

## 5. Section B — NetworkConfig (L109–207)

```python
@dataclass
class NetworkConfig:
    source_type=None       # "preset" | "simbench_code" | "custom" | "plugin"
    preset_name=None; simbench_code=None; custom_path=None; custom_function_name="get_network"
    preset_family=None; simbench_selections=None; plugin_path=None
    der_placements=None; switches_to_flip=None    # pre-run modifications
```

#### Line 109–161: The network-source selection

Records *how to load* the network (not the network itself). **Four mutually-exclusive
`source_type`s**, each mapping to an executor loading path (9B §8): **`preset`** (a named
catalogue network → `PRESET_CATALOGUE[name]()`), **`simbench_code`** (a raw code the wizard
assembles from 4 questions → `sb.get_simbench_net(code)`), **`custom`** (a user `.py` with a
zero-arg `custom_function_name` → a `pandapowerNet`), and **`plugin`** (a network-plugin YAML →
`load_network_from_yaml`, which returns *both* net and profiles — so for `plugin` the
`DatasetConfig` is **ignored**, the YAML owns profile building). **`der_placements`** (list of
`{bus, p_mw, sn_mva}` → new PV sgens) and **`switches_to_flip`** (list of switch indices to
toggle) are **pre-run modifications applied before profile building** (so injected DERs receive
profile columns) — the same operations `run_benchmark_script.py` does inline (9D).

#### Line 163–207: Serialisation (with a rename alias)

Standard `asdict`/`.get()` pattern, plus one notable **backward-compat alias**: L206 reads
`switches_to_flip` but falls back to the **old field name `switches_to_open`**
(`data.get("switches_to_flip", data.get("switches_to_open", None))`) — so a preset saved under
the previous name still loads. **Design rationale:** field renames are a normal evolution;
aliasing in `from_dict` keeps old presets valid (the same reproducibility guarantee as the
`.get` defaults).

---

## 6. Section C — DatasetConfig (L210–326)

```python
@dataclass
class DatasetConfig:
    source_type=None       # "simbench_native" | "dwd" | "custom"
    data_dir=None; station_id=None; year=None; file_map=None; col_map=None; custom_path=None
```

#### Line 210–283: The profile-source selection

Records *how to build the profiles*. **Three `source_type`s** with documented **field
relevance**: **`simbench_native`** (use the SimBench net's embedded profiles via
`sb.get_absolute_values` — valid only with a `simbench_code` network; no extra fields);
**`dwd`** (build from local DWD CDC CSVs via `build_annual_profiles` — needs `data_dir`,
`station_id`, `year`); **`custom`** (user pre-built profile CSVs, e.g. ERA5 — needs `data_dir`,
`file_map`, `col_map`). The field comments encode the DWD subdirectory layout (`PV/`, `Wind/`,
`Temperature/`), the glob-by-`station_id` convention, and the note that `year` is **ignored for
`simbench_native`** (SimBench's internal reference is 2016, a leap year — the KB invariant).
**Design rationale — the field-relevance table (docstring):** the same flat dataclass serves
all three strategies; documenting which fields matter per `source_type` is what lets the wizard
ask only the relevant ones and the executor read only the relevant ones.

#### Line 284–326: Serialisation

The same `asdict`/`.get()` reproducibility pattern — no surprises; every field defaults to
`None` so a partially-specified strategy round-trips exactly.

---

## 7. Section D — RunPlan (L328–466)

```python
@dataclass
class RunPlan:
    parameters=None; network=None; dataset=None; study=None; focus_buses=None
    hardware=False; port=None; controller_plugin_path=None
    time_period=None; time_index=None; hc_stressed=False; stream_every_k=4
    run_id = field(default_factory=lambda: str(uuid.uuid4())); output_dir="runs"
```

#### Line 328–406: The top-level container

`RunPlan` nests the three focused configs and adds the run-control, streaming, and identity
fields (`study`, `focus_buses`, `hardware`/`port`, `controller_plugin_path`, the `time_period`/
`time_index` window, `hc_stressed`, `stream_every_k`, `output_dir`). **The explicit design
principle (docstring): `RunPlan` is a pure data container** — it "does not validate its own
fields, resolve incompatible combinations, or apply defaults based on other fields", because
that belongs to the wizard (interactive) and executor (runtime guards). This is the
separation-of-concerns that keeps `RunPlan` simple, testable, and reusable across entry points
— and it is *why* the executor's `validate_plan` exists (9B §9). **`run_id`** uses
`default_factory=lambda: str(uuid.uuid4())`, so **every instance gets a fresh UUID** (not a
shared class-level one) — it names the output dir `runs/<run_id>/` and matches dashboard SSE
events. `stream_every_k=4` publishes a live frame every simulated hour at 15-min resolution.

#### Line 408–466: to_dict / from_dict — the whole-plan round-trip

`to_dict` is `asdict` (recursively converting the three nested dataclasses → a fully
JSON-serialisable nested dict). `from_dict` reconstructs each nested config **via its own
`from_dict`**, guarded by **`data.get(key) or {}`** — which handles *both* a missing key
(`None`) and an explicit `null` in the JSON: either way the nested `from_dict` receives `{}`
and fills defaults. **Worked example:** a minimal preset `{"study": "opf_benchmark"}` reloads
to a `RunPlan` with `study="opf_benchmark"`, default `ParameterConfig`/`NetworkConfig`/
`DatasetConfig` (each built from `{}`), `hardware=False`, and a **fresh** `run_id` (the saved
one is reused if present, else a new UUID). **Design rationale — save/reload reproduces a run
identically:** this is the persistence contract `_save_run_plan_copy` (9B) relies on to make
every run auditable and repeatable.

---

# PART 2 — wizard.py

## 8. Section E — Back-navigation primitives (L13–132)

```python
class BackRequested(Exception): ...          # caught only by run_wizard's step loop
BACK_TOKEN = "<"                             # not a plausible path/ID/number/dict-literal
def _ask(prompt, choices, default=None, allow_back=False) -> str:   # numbered menu; option 0 = back
def _ask_value(prompt, default=None, cast=str, allow_back=True):    # free-text; "<" = back; re-ask on bad cast
```

#### Line 13–132: The two back-aware prompt primitives

Every wizard interaction goes through one of two primitives, both of which express "go back"
as a **`BackRequested` exception**. **`_ask`** renders a numbered menu of `(label, value)`
choices and returns the machine-readable value; with `allow_back=True` it adds "0 — go back"
which raises `BackRequested`, and it **re-asks on an out-of-range number** (never crashes on bad
input). **`_ask_value`** is the back-aware replacement for Rich's `Prompt`/`IntPrompt`: it reads
a raw string, treats the **`BACK_TOKEN "<"`** as `BackRequested`, casts to the requested type,
and **re-asks on a cast failure or an empty required answer**. **Design rationale — the back
token choice (docstring):** `"<"` is used because it is *not* a plausible file path, station ID,
number, or dict literal — so it can never collide with a real answer. **Worked example:** at the
parameters prompt a user types `abc` when an `int` is expected → `_ask_value` prints "Invalid
value. Expected int." and re-asks; the user types `<` → `BackRequested` propagates to
`run_wizard`, which steps back. **The exception-based design** means back-navigation needs no
special return-value plumbing through every prompt — a prompt just raises, and exactly one place
(the step loop, §10) handles it.

---

## 9. Section F — The per-field prompt builders (L133–855)

Eight builders, each assembling one config (or config fragment) via `_ask`/`_ask_value`, each
validating **only its own field**:

- **`_ask_study`** (L133) — selects the study protocol (`scenario_comparison`,
  `hosting_capacity`, `opf_benchmark`, …) → the `RunPlan.study` string.
- **`_ask_network`** (L163–402, the largest) — builds a **`NetworkConfig`** by first choosing
  the `source_type`, then branching: a preset menu; the **4-question SimBench-code assembly**
  (voltage level → grid type → DER level → switching, concatenated into the code); a custom
  `.py` path + function name; or a plugin YAML path. Each branch fills only the fields its
  `source_type` uses.
- **`_ask_dataset`** (L445–513) — builds a **`DatasetConfig`**, branching on strategy
  (`simbench_native`/`dwd`/`custom`) and asking only the relevant fields (per §6's relevance
  table); `_ask_optional_dict` (L403) parses a dict-literal answer (`file_map`/`col_map`).
- **`_ask_parameters`** (L524–656) — builds a **`ParameterConfig`**: the voltage/thermal/angle/
  unbalance limits, the optional Q(V) overrides, and the DER/load **scaling dicts** via
  `_ask_scaling_dict` (L514, which offers whole-network / single-bus / bus-set targeting and
  builds the `{bus→factor}` dict with the `None` sentinel).
- **`_ask_hc_stressed`** (L657) — the HC-stressed toggle (only reached for the HC study).
- **`_ask_hardware`** (L672) — the `(hardware, port)` pair (asks for a port only when hardware
  is chosen).
- **`_ask_controller_plugin`** (L698) — the optional controller-plugin YAML path.
- **`_ask_network_modifications`** (L747) + **`_ask_time_window`** (L823) — the pre-run DER/switch
  modifications and the time-window slice (`_load_preview_network`, L713, loads the net so the
  user can pick valid bus indices).

**Design rationale — per-field validation only:** each builder validates its *own* field
(menu ranges, dict-literal syntax, numeric casts) but deliberately performs **no cross-field
checks** (e.g. `v_min < v_max`, or `simbench_native` on a non-SimBench net) — those are the
executor's `validate_plan` (9B §9). This keeps each prompt independent and re-askable, and
concentrates cross-field logic in one place.

---

## 10. Section G — run_wizard: the step machine (L856–959)

```python
def run_wizard() -> RunPlan:
    state = {}
    def step_study(s): s["study"]=_ask_study(); s["hc_stressed"]=_ask_hc_stressed() if study=="hosting_capacity" else False
    def step_network(s): s["network"]=_ask_network()
    def step_network_mods(s): der,sw=_ask_network_modifications(s["network"]); s["network"].der_placements=der; ...
    def step_dataset(s):
        if s["network"].source_type=="plugin": s["dataset"]=DatasetConfig(source_type="plugin"); return   # SKIP
        s["dataset"]=_ask_dataset()
    ... step_time_window / step_parameters / step_hardware / step_stream_every_k / step_controller_plugin
    steps = [Study, Network, Network-mods, Dataset, Time-window, Parameters, Hardware, Streaming, Controller-plugin]
    index = 0
    while index < len(steps):
        try: fn(state)
        except BackRequested: index = max(index-1, 0); continue     # step BACK
        index += 1
    return RunPlan(**state)
```

#### Line 856–959: The ordered, back-navigable step machine

`run_wizard` is the only public function. It walks an **ordered list of nine steps by index**,
each step mutating a shared `state` dict, then assembles the `RunPlan`. **Back-navigation is
the loop's single responsibility:** any step raising `BackRequested` (from any `_ask`/`_ask_value`
inside it) is caught here → `index = max(index-1, 0)` → the **previous** step re-runs (re-asking
its questions with standard defaults); back on the first step just re-asks it. **Worked example:**
a user on Step 6 (Parameters) realises the network is wrong, types `<` → `BackRequested` →
`index 5→4` → Step 5 (Time window) re-asks; another `<` → Step 4 (Dataset); and so on back to
Network. **Two conditional steps** (both documented): `step_study` asks `hc_stressed` **only**
for the `hosting_capacity` study (else `False`), and `step_dataset` **skips** entirely for a
`plugin` network (printing why, and storing `DatasetConfig(source_type="plugin")`) because the
plugin YAML owns profile building. **Design rationale — re-ask, don't restore:** re-entering a
step re-asks from defaults rather than restoring the partial answers, which keeps the state
machine simple (no per-step undo stack) at the cost of re-typing — an acceptable trade for a
CLI wizard, and the reason the executor (not the wizard) holds the durable, reloadable `RunPlan`.

---

## 11. Findings and flags

**No bugs found.** Both files are clean and the separation-of-concerns is enforced
consistently. Points worth recording as *strengths*:

1. **Reproducibility is airtight** — every config's `from_dict` uses `.get()` defaults (old
   presets never `KeyError`), the scaling `"null" → None` coercion round-trips the sentinel key,
   the `switches_to_open` alias survives a field rename, and the `or {}` guard handles missing
   *and* null nested configs.
2. **Back-navigation via exceptions** is a clean design — prompts stay return-value-simple, and
   exactly one place (the step loop) implements "back".
3. **The wizard/executor validation split is deliberate and documented** — per-field in the
   wizard, cross-field in `executor.validate_plan` (9B) — so neither duplicates the other.

**Not bugs (documented for clarity):** `RunPlan` not validating itself is the intended design;
re-ask-don't-restore on back is a deliberate simplicity trade; the `plugin` network skipping the
dataset step is correct (the YAML owns profiles).

---

## 12. Coverage checklist

| Entity | File · Lines | Kind | Documented | Notes |
|--------|--------------|------|:----------:|-------|
| `ParameterConfig` (+ `to_dict`/`from_dict`) | run_plan 4–106 | dataclass | ✅ | scaling sentinel, null→None |
| `NetworkConfig` (+ `to_dict`/`from_dict`) | run_plan 109–207 | dataclass | ✅ | 4 sources, switches_to_open alias |
| `DatasetConfig` (+ `to_dict`/`from_dict`) | run_plan 210–326 | dataclass | ✅ | 3 sources, field relevance |
| `RunPlan` (+ `to_dict`/`from_dict`) | run_plan 328–466 | dataclass | ✅ | pure container, `or {}` guard |
| `BackRequested` / `BACK_TOKEN` | wizard 13–24 | exception/const | ✅ | back-nav mechanism |
| `_ask` / `_label_for_value` / `_ask_value` | wizard 26–132 | helpers | ✅ | back-aware primitives |
| `_ask_study` | wizard 133–162 | builder | ✅ | study |
| `_ask_network` (+ `_ask_optional_dict`) | wizard 163–444 | builder | ✅ | 4-question SimBench assembly |
| `_ask_dataset` / `_ask_scaling_dict` | wizard 445–523 | builders | ✅ | strategy branch, scaling dict |
| `_ask_parameters` | wizard 524–656 | builder | ✅ | limits + Q(V) + scaling |
| `_ask_hc_stressed` / `_ask_hardware` / `_ask_controller_plugin` | wizard 657–712 | builders | ✅ | |
| `_load_preview_network` / `_ask_network_modifications` / `_ask_time_window` | wizard 713–855 | builders | ✅ | pre-run mods + window |
| `run_wizard` | wizard 856–959 | function | ✅ | step machine + back-nav |

**Also covered (not standalone entities):** the three-layer separation, the reproducibility
contract (`to_dict`/`from_dict` round-trip incl the sentinel/alias gotchas), the scaling-dict
`None` sentinel, the four network + three dataset source types, the exception-based
back-navigation, the two conditional steps, and the wizard/executor per-field-vs-cross-field
validation split.

**Flags raised:** none (no bugs). Design strengths recorded in §11.

---

*End of Document 9C. Next: Document 9D — `run_benchmark_script.py` (563) + `helpers.py` (210) +
`__main__.py` (205): the direct-script entry point (the non-wizard path), the shared helpers,
and the CLI entry.*
