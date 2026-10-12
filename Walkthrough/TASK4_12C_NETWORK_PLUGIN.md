# TASK 4 — DOCUMENT 12C
## Custom Controller & Plugin Subsystem, Part 3: The Network Plugin
### `network_plugin.py` — load, profile, and validate a user network (775 lines)
### Complete line-by-line walkthrough

> **Source verified:** read against the current `/mnt/project/network_plugin.py`
> (**775 lines**, the project tree). Compiles. **Read alongside:** **12B**
> (`plugin_runner.py` — the parallel custom-*controller* subsystem), **`profile_builder`**
> (the SimBench/DWD profile paths this mirrors), **6A-i** (`adapt_profiles` — the
> consumer of the profiles built here), **Document 9** (the CLI/executor `--network`
> path that calls this).

---

## TABLE OF CONTENTS

1. [Overview — the custom-network subsystem](#1-overview)
2. [File logic flow — pseudocode](#2-file-logic-flow--pseudocode)
3. [Call-site map](#3-call-site-map)
4. [Section A — Constants (L135–153)](#4-section-a--constants-l135153)
5. [Section B — _load_yaml_config (L160–299)](#5-section-b--_load_yaml_config-l160299)
6. [Section C — Network loading (L306–367)](#6-section-c--network-loading-l306367)
7. [Section D — Profile strategies (L374–541)](#7-section-d--profile-strategies-l374541)
8. [Section E — load_network_from_yaml (L548–602)](#8-section-e--load_network_from_yaml-l548602)
9. [Section F — validate_network_plugin (L609–710)](#9-section-f--validate_network_plugin-l609710)
10. [Section G — make_profile_factory (L717–775)](#10-section-g--make_profile_factory-l717775)
11. [Findings and flags](#11-findings-and-flags)
12. [Coverage checklist](#12-coverage-checklist)

---

## 1. Overview

`network_plugin.py` lets a user benchmark **their own distribution network** — the
custom-*network* counterpart to 12B's custom-*controller*. It takes a YAML config + a
network artefact and produces the `(net, profiles)` pair `run_benchmark` needs. Three
responsibilities:

1. **Load the network** (§6) — from one of three sources: a pandapower **JSON**
   (recommended), a pandapower **pickle**, or a user **function** (a zero-arg loader
   returning a `pandapowerNet`).
2. **Build profiles** (§7) — via one of four strategies: **`simbench_native`** (reuse the
   loaded net's SimBench metadata), **`dwd_pvlib`** (DWD Bremen + pvlib + BDEW),
   **`flat`** (constant rated-capacity, worst-case simultaneity), or **`custom`** (DWD with
   user file maps). A `simbench_native → dwd_pvlib` fallback fires when the net lacks
   SimBench metadata.
3. **Validate** (§9) — 6 compatibility checks returning *warnings* (not exceptions), since
   some legitimate networks fail individual checks.

Plus **`make_profile_factory`** (§10) — a factory that rebuilds profiles for the
hosting-capacity re-benchmark with the *same* strategy as the outer run.

**The design principle mirrors profile_builder faithfully.** Each strategy reproduces
`profile_builder`'s exact semantics (SimBench pv-mask incl `lv_res`, night-time PV zeroing,
clipping) so a plugin network's numbers are comparable with the built-in SimBench/CIGRE
runs. The recurring theme is **portability**: a SimBench net exported to JSON carries its
`net.profiles`, so `simbench_native` rebuilds the profiles *without re-downloading* — the
network becomes a self-contained, shareable artefact.

---

## 2. File logic flow — pseudocode

```
CONSTANTS: valid sources (json/pickle/function), valid strategies
           (simbench_native/dwd_pvlib/flat/custom), recognised DER types, MV/LV voltage bands

_load_yaml_config(yaml_path) → cfg:
    validate name/label; source ∈ sources; path (json/pickle) OR module+function (function);
    profiles.strategy ∈ strategies (+ year, data_dir, file_map, col_map); voltage_limits

NETWORK LOADING:
  _import_loader_fn(module_path, fn)  — file-location import (mangled _hil_network_ name)
  _load_net(cfg) → net               — pp.from_json / pp.from_pickle / loader(); type-check pandapowerNet

PROFILE STRATEGIES:
  _has_simbench_metadata(net)              — net.profiles present & non-empty?
  _build_profiles_simbench_native(net,cfg) — sb.get_absolute_values on the loaded net (portable)
  _build_profiles_dwd(net,cfg)             — build_annual_profiles (DWD 691 Bremen + pvlib), _dwd_safe_name guard
  _build_profiles_flat(net,cfg)            — constant rated-capacity frames, no weather
  _build_profiles_for_strategy(net,cfg)    — dispatch + simbench_native→dwd_pvlib fallback

load_network_from_yaml(yaml_path) → (net, profiles + plugin_meta)    [public]
validate_network_plugin(net, profiles) → [warning strings]          [public]
make_profile_factory(yaml_path) → callable(net_hc) → profiles       [public, HC re-runs]
```

---

## 3. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `load_network_from_yaml` | function | `wizard.py`, `executor.py`, `run_benchmark_script.py` — the `--network <yaml>` path (Doc 9). |
| `validate_network_plugin` | function | `executor.py`, `run_benchmark_script.py` (after load, before run). |
| `make_profile_factory` | function | The HC-re-benchmark wiring (`BenchmarkConfig.profile_factory`). |
| `_load_yaml_config` | helper | `load_network_from_yaml`, `make_profile_factory`. |
| `_load_net`, `_import_loader_fn` | helpers | `load_network_from_yaml`. |
| `_build_profiles_*`, `_build_profiles_for_strategy` | helpers | Both public entry points. |
| `build_annual_profiles`, `find_extreme_days`, `detect_network_type` | infra | `profile_builder` / network detection. |

---

## 4. Section A — Constants (L135–153)

```python
_VALID_SOURCES    = ("json", "pickle", "function")
_VALID_STRATEGIES = ("simbench_native", "dwd_pvlib", "flat", "custom")
_RECOGNISED_SGEN_TYPES = frozenset(t.lower() for t in ("PV", "WKA", "lv_res", "pv", "wind", "wp", "solar"))
_PV_TYPE_PATTERN = "pv|solar|lv_res"; _WIND_TYPE_PATTERN = "wind|wp|wka"
V_MV_LOW, V_MV_HIGH = 1.0, 36.0; V_LV_HIGH = 1.0
```

#### Line 135–153: Sources, strategies, DER masks, voltage bands

The two enumerations define the plugin schema's `source` and `strategy` fields.
`_RECOGNISED_SGEN_TYPES` is matched **case-insensitively** so both SimBench conventions
(`PV`, `WKA`, `lv_res`) and CIGRE/manual conventions (`pv`, `wind`, `wp`) are accepted —
important because DER type strings drive the pv/wind profile masks. `V_MV_LOW/HIGH` (1–36
kV) and `V_LV_HIGH` (<1 kV) are the voltage bands the validator uses to confirm the network
is a *distribution* network (the benchmark's scope). **Design rationale:** centralising the
valid-value sets here makes the YAML validator's error messages authoritative and keeps the
DER masks consistent with `profile_builder`.

---

## 5. Section B — _load_yaml_config (L160–299)

The network-YAML validator (the counterpart to `plugin_runner.load_plugin`).

```python
    name = raw["name"].strip(); label = str(raw.get("label", name)).strip() or name
    source = str(raw.get("source", "")).strip().lower()   # ∈ _VALID_SOURCES
    if source in ("json","pickle"): net_path = (yaml_path.parent / raw["path"]).resolve()  # must exist
    if source == "function": module_path = (yaml_path.parent / raw["module"|"path"]).resolve(); function ...
    strategy = prof_raw["strategy"]  # ∈ _VALID_STRATEGIES; 'custom' requires file_map
    year ∈ [1990,2100]; data_dir (yaml-relative); v_min/v_max (0.5 ≤ v_min < v_max ≤ 1.5)
```

#### Line 160–299: Read and validate the network YAML

Validates the schema with field-naming errors, resolving **all paths relative to the YAML's
folder**. Source-specific requirements: `json`/`pickle` need a `path` to an existing file;
`function` needs a `module` (a `.py`, with `path` accepted as an alias for schema uniformity)
and a valid-identifier `function`. The `profiles` block requires a valid `strategy`
(`custom` additionally requires a `file_map`), a plausible `year` (default 2016, SimBench's
reference year), an optional YAML-relative `data_dir`, and optional `file_map`/`col_map` for
custom DWD files. `voltage_limits` default to 0.95/1.05 and are range-checked
(`0.5 ≤ v_min < v_max ≤ 1.5`). **Design rationale:** exhaustive up-front validation means a
misconfigured plugin fails at load with a precise message, not deep inside `run_benchmark`.

---

## 6. Section C — Network loading (L306–367)

```python
def _import_loader_fn(module_path, function_name):   # mangled _hil_network_<stem>_<hash> in sys.modules
    ... spec_from_file_location; exec_module; check callable ...

def _load_net(cfg):
    if source == "json":   net = pp.from_json(cfg["path"])
    elif source == "pickle": net = pp.from_pickle(cfg["path"])   # never raw pickle
    else: net = _import_loader_fn(cfg["module_path"], cfg["function"])()   # zero-arg loader
    if not isinstance(net, pp.pandapowerNet): raise TypeError(...)
```

#### Line 306–367: Load the network from one of three sources

`_import_loader_fn` is the same file-location import as `plugin_runner` (mangled unique
`sys.modules` name to avoid stem collisions). `_load_net` dispatches on `source`: **JSON**
via `pp.from_json`, **pickle** via `pp.from_pickle` (**never** the raw `pickle` module — so
pandapower handles its own version compatibility), **function** by importing and calling a
**zero-arg** loader. The result is type-checked to be a `pandapowerNet`, with a message
naming the loader contract. **Design rationale — JSON recommended:** `pp.to_json` preserves
every element table *including* SimBench's `net.profiles`, which is exactly what makes the
`simbench_native` strategy (§7) work on a portable, exported network.

---

## 7. Section D — Profile strategies (L374–541)

Four ways to attach a year of 15-minute profiles to the loaded network.

#### Line 389–438: _build_profiles_simbench_native — portable SimBench

```python
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    times = pd.date_range(f"{cfg['year']}-01-01", periods=len(times), freq="15min", tz="Europe/Berlin")
    pv_df.loc[(hour>=22)|(hour<=4)] = 0.0                    # night-time PV zeroing
    ... clip(lower=0.0) ... result["extreme_days"] = find_extreme_days(result)
```

Builds profiles by calling `sb.get_absolute_values` **on the already-loaded net** — not by
re-downloading via a `simbench_code`. **Design rationale — portability:** this is what makes
a JSON-exported SimBench network self-contained (its `net.profiles` metadata travels with
it). The path **mirrors `profile_builder`'s SimBench logic exactly**: the pv-mask includes
`lv_res`, the integer step index is reconstructed to a `DatetimeIndex` for the configured
year (SimBench's internal reference is 2016, a leap year, 35,136 steps), night-time PV
(22:00–04:00) is zeroed (SimBench's commercial networks carry spurious small night values),
and everything is lower-clipped. So `simbench_native` results match the built-in SimBench
runs.

#### Line 441–475: _dwd_safe_name & _build_profiles_dwd — the name-collision gotcha

```python
def _dwd_safe_name(name):
    if detect_network_type(name) != "simbench": return name
    ... replace SimBench identifiers; belt-and-braces → "plugin_custom_net" ...
```

`_build_profiles_dwd` routes to `build_annual_profiles` (DWD station 691 Bremen + pvlib +
BDEW). **The subtle gotcha `_dwd_safe_name` handles:** `profile_builder` routes purely on
the *net-name string* — so a plugin named `simbench_rural_export` would be detected as
SimBench and raise "`simbench_code must be provided`" on the DWD path. `_dwd_safe_name`
rewrites the name so it no longer trips SimBench detection (falling back to
`plugin_custom_net` if needed). **Design rationale:** a well-designed subsystem anticipates
that user names collide with framework heuristics and defuses it transparently.

#### Line 478–541: _build_profiles_flat & the dispatcher

```python
def _build_profiles_flat(net, cfg):   # constant rated-capacity, worst-case simultaneity, no weather
    load_df = tile(net.load.p_mw, T); pv/wind = tile(net.sgen.p_mw[mask], T); clip(lower=0)

def _build_profiles_for_strategy(net, cfg):
    if strategy=="simbench_native" and not _has_simbench_metadata(net): strategy = "dwd_pvlib"   # fallback
    dispatch → simbench_native | dwd(_pvlib/custom) | flat; return (profiles, strategy_used)
```

**`flat`** repeats rated capacity on every timestep — a worst-case simultaneity snapshot
needing no weather data (useful for a quick stress check). **`_build_profiles_for_strategy`**
is the dispatcher, and it implements the **`simbench_native → dwd_pvlib` fallback**: if
native SimBench profiles are requested but the net has no `net.profiles` metadata (a non-
SimBench JSON, or stripped metadata), it warns and falls back to DWD. **Design rationale —
shared dispatcher:** both `load_network_from_yaml` and `make_profile_factory` (§10) call
this, so the HC re-benchmark rebuilds profiles with *exactly* the same strategy and fallback
semantics as the outer run — no divergence.

---

## 8. Section E — load_network_from_yaml (L548–602)

```python
    cfg = _load_yaml_config(yaml_path); net = _load_net(cfg)
    profiles, strategy = _build_profiles_for_strategy(net, cfg)
    profiles["plugin_meta"] = {"name","label","source","strategy","requested_strategy","year","v_min","v_max","notes","yaml_path"}
    return net, profiles
```

#### Line 548–602: The public loader

Ties it together: parse the YAML, load the net, build the profiles, and attach an **additive
`plugin_meta` dict** carrying the plugin's identity and the *actually-used* strategy (vs the
requested one, so a fallback is visible downstream) plus the voltage limits. Returns
`(net, profiles)` in the `profile_builder` schema (`load`/`pv`/`wind`/`times`/`extreme_days`/
`net_type`) ready for `run_benchmark`. **Design rationale — `plugin_meta` is additive:** it
does not disturb the standard profile schema, so the plugin path is transparent to every
downstream consumer while still carrying the metadata the publisher/CLI want.

---

## 9. Section F — validate_network_plugin (L609–710)

Six compatibility checks, each appending a human-readable **warning** (not raising):

```python
    if len(net.sgen) == 0: warn("... Scenario 4 has no DERs ... skip 4/5")           # 1
    elif not types.isin(_RECOGNISED_SGEN_TYPES).any(): warn("... der_p will be empty")# 2
    if not (in_mv or in_lv): warn("... HV-only networks out of scope")               # 3
    if n_trafo==0 and n_trafo3w==0: warn("... Scenario 2 (OLTC) cannot run")         # 4
    for const_i/z_percent != 0: warn("... ZIP loads ignored under voltage_depend_loads=False")  # 5
    orphan load cols / zero DER cols / NaN in profiles: warn(...)                     # 6
```

#### Line 609–710: Warnings-not-exceptions compatibility validation

Checks: **(1)** non-empty `sgen`; **(2)** at least one recognised DER type (else the pv/wind
masks match nothing and `der_p` is empty); **(3)** a bus in the MV or LV band (HV-only is out
of scope); **(4)** at least one two-winding transformer (else OLTC can't run); **(5)**
ZIP-load compatibility — non-zero `const_i/z_percent` loads will be **silently ignored**
under the mandatory `voltage_depend_loads=False` (a genuine correctness caveat surfaced to
the user); **(6)** additive profile/net alignment — orphan load columns (would `KeyError` in
`adapt_profiles`), zero DER columns despite non-empty `sgen` (a type-string mismatch), and
NaN in profiles (would break `runpp`). **Design rationale — warnings, not exceptions
(docstring):** some networks legitimately fail individual checks — a DER-free Kerber feeder
for HC-only analysis has an empty `sgen` but is perfectly valid for Scenarios 1–3. So the
validator *informs* and lets the caller decide, rather than blocking. The messages are
prescriptive (they say how to fix or skip).

---

## 10. Section G — make_profile_factory (L717–775)

```python
def make_profile_factory(yaml_path):
    cfg = _load_yaml_config(yaml_path)                       # parse once
    def _factory(net_hc) -> dict:
        profiles, used = _build_profiles_for_strategy(net_hc, cfg)   # SAME strategy as outer run
        profiles["plugin_meta"] = {... "name": cfg["name"] + "_hc_stressed" ...}
        return profiles
    return _factory
```

#### Line 717–775: The hosting-capacity re-benchmark profile factory

Returns a `callable(net) -> profiles` for `BenchmarkConfig.profile_factory` when
`run_hc_scenarios=True`. **The gap it closes (docstring):** hosting-capacity analysis adds
new sgens to a *copy* of the network, and the re-benchmark then needs fresh profiles that
**cover those new sgens**. The built-in wiring hardcodes `build_annual_profiles` (the DWD
path) — wrong for a `flat`- or `simbench_native`-strategy plugin. This factory rebuilds the
stressed net's profiles with the **same strategy, year, data_dir, and fallback** as the outer
run. It documents each strategy's HC behaviour: **flat** → new sgens get constant rated
columns automatically; **dwd_pvlib** → pvlib/BDEW columns automatically; **simbench_native**
→ HC sgens have no native columns (they weren't in the SimBench dataset), matching the
built-in SimBench HC behaviour. **Design rationale:** parsing the YAML *once* at factory
creation (not per HC call) and reusing `_build_profiles_for_strategy` guarantees the HC
re-benchmark is a faithful continuation of the outer run.

---

## 11. Findings and flags

**No bugs found.** The module is clean and defensively designed. Points worth recording as
*strengths* rather than flags:

- The **`simbench_native → dwd_pvlib` fallback** and its sharing between the loader and the
  HC factory prevent silent strategy divergence.
- **`_dwd_safe_name`** anticipates the user-name-vs-SimBench-detection collision and defuses
  it transparently.
- **Warnings-not-exceptions** validation correctly distinguishes "incompatible" from
  "unusual-but-valid" (the DER-free Kerber case).
- **Pickle only via `pp.from_pickle`** (never the raw `pickle` module) is the right safety
  choice for version compatibility.

**Not bugs (documented for clarity):** the MV band top of 36 kV does not exclude SimBench MV
nets (their 20 kV MV buses satisfy the `in_mv` check; the 110 kV HV buses simply aren't
counted); `plugin_meta` is deliberately additive to the standard profile schema.

---

## 12. Coverage checklist

| Entity | Lines | Kind | Documented | Where-used stated |
|--------|-------|------|:----------:|:-----------------:|
| constants (sources/strategies/types/bands) | 135–153 | constants | ✅ | ✅ |
| `_load_yaml_config` | 160–299 | helper | ✅ | ✅ |
| `_import_loader_fn` | 306–339 | helper | ✅ | ✅ |
| `_load_net` | 342–367 | helper | ✅ | ✅ |
| `_has_simbench_metadata` | 374–379 | helper | ✅ | ✅ |
| `_annual_15min_index` | 382–386 | helper | ✅ | ✅ |
| `_build_profiles_simbench_native` | 389–438 | helper | ✅ | ✅ |
| `_dwd_safe_name` | 441–460 | helper | ✅ | ✅ |
| `_build_profiles_dwd` | 463–475 | helper | ✅ | ✅ |
| `_build_profiles_flat` | 478–512 | helper | ✅ | ✅ |
| `_build_profiles_for_strategy` | 515–541 | helper | ✅ | ✅ |
| `load_network_from_yaml` | 548–602 | function | ✅ | ✅ |
| `validate_network_plugin` | 609–710 | function | ✅ | ✅ |
| `make_profile_factory` | 717–775 | function | ✅ | ✅ |

**Also covered (not standalone entities):** the three-source loading (JSON recommended,
pickle via `pp.from_pickle`), the four profile strategies + native→DWD fallback, the
`_dwd_safe_name` collision gotcha, the portable-SimBench rationale, the warnings-not-exceptions
validation philosophy, the additive `plugin_meta`, and the HC profile factory's
strategy-faithful rebuild.

**Flags raised:** none (no bugs). Design strengths recorded in §11.

---

*End of Document 12C. Next: Document 12D — the small loaders (`my_network_loader.py`,
`network_export.py`) and the worked examples (`droop_controller.py`, `deadband_controller.py`,
the example networks + their YAMLs) — closing Document 12.*
