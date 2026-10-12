# TASK 4 — DOCUMENT 12D
## Custom Controller & Plugin Subsystem, Part 4: Loaders & Worked Examples
### `my_network_loader.py`, `network_export.py`, the example controllers & YAMLs
### Complete line-by-line walkthrough — closes Document 12

> **Source verified:** all files read in full from `/mnt/project/` (the project tree).
> **Read alongside:** **12A** (the `controller_fn` contract these controllers satisfy),
> **12B** (`plugin_runner` — consumes the controller YAMLs), **12C** (`network_plugin`
> — consumes the network YAMLs), **Document 1A** (the VDE Q(V) curve `deadband_controller`
> reproduces).
>
> **⚠ Contains stale/portability flags** in two YAMLs (§9).

---

## TABLE OF CONTENTS

1. [Overview — the worked examples](#1-overview)
2. [Section A — my_network_loader.py (26 lines)](#2-section-a--my_network_loaderpy-26-lines)
3. [Section B — network_export.py (9 lines)](#3-section-b--network_exportpy-9-lines)
4. [Section C — droop_controller.py (83 lines)](#4-section-c--droop_controllerpy-83-lines)
5. [Section D — deadband_controller.py (85 lines)](#5-section-d--deadband_controllerpy-85-lines)
6. [Section E — The controller YAMLs](#6-section-e--the-controller-yamls)
7. [Section F — The network YAMLs](#7-section-f--the-network-yamls)
8. [Findings and flags](#8-findings-and-flags)
9. [Coverage checklist & Document 12 close-out](#9-coverage-checklist--document-12-close-out)

---

## 1. Overview

This document covers the **worked examples** that make the plugin subsystem (12A–12C)
usable — the reference implementations a researcher copies and adapts:

- **Two example controllers** (`droop_controller.py`, `deadband_controller.py`) — reference
  `controller_fn` plugins, each self-contained (numpy only, no project imports).
- **One example network loader** (`my_network_loader.py`) — the `source: function`
  contract, plus **`network_export.py`**, a snippet showing the one-time JSON export.
- **Six example YAMLs** — two controller configs (12B) and four network configs (12C),
  spanning every source and profile strategy.

Their pedagogical value is that they are **minimal and honest**: the controllers import only
numpy so a student sees exactly the `controller_fn` contract with no framework noise, and the
YAMLs' comments document the run command and the design intent for each field.

---

## 2. Section A — my_network_loader.py (26 lines)

```python
import pandapower.networks as pn

def get_network():
    """Return the CIGRE MV benchmark network with PV + wind DERs."""
    return pn.create_cigre_network_mv(with_der="pv_wind")
```

The reference **`source: function`** loader: a **zero-argument** function returning a
`pandapowerNet`, imported by path (the importlib file-location machinery, 12C §6) and called
by `network_plugin`. The demo returns the CIGRE MV network with 9 DERs — the same network the
built-in OPF scenario uses. **Design rationale:** the docstring spells out the contract and
invites substitution (a parametric feeder generator, a stitched multi-feeder net, a synthetic
stress case) — the loader is the extension point for *programmatically constructed* networks,
complementing the JSON/pickle file sources.

---

## 3. Section B — network_export.py (9 lines)

```python
import simbench as sb, pandapower as pp
net = sb.get_simbench_net("1-MV-rural--2-sw")
# pp.to_json(net, "networks/simbench_1_mv_rural_2_sw.json")
net = sb.get_simbench_net("1-LV-semiurb4--2-sw")
pp.to_json(net, "networks/my_lv_feeder.json")
```

Not a module — a **copy-paste snippet** documenting the **one-time export step** that turns a
SimBench network into a portable JSON for the `source: json` plugins (`simbench_rural_mv.yaml`,
`custom_lv_flat.yaml`). **Design rationale:** `pp.to_json` preserves all element tables *and*
SimBench's `net.profiles` metadata, so the exported file works with the `simbench_native`
strategy (12C §7) without a `simbench_code` at run time. Included in the example folder so a
user can reproduce the exported artefacts the YAMLs reference.

---

## 4. Section C — droop_controller.py (83 lines)

The reference `controller_fn` — a proportional Q–P droop.

```python
def compute_setpoints(vm_pu, p_mw, droop_slope=5.0, deadband=0.01, q_ratio=0.25):
    dv     = vm_pu - 1.0
    dv_eff = np.sign(dv) * np.maximum(np.abs(dv) - deadband, 0.0)   # beyond the deadband edge
    q      = -droop_slope * dv_eff * p_mw
    q_max  = q_ratio * np.abs(p_mw)
    return np.clip(q, -q_max, q_max)
```

**The control law, derived.** Reactive power is proportional to the voltage deviation
*beyond* the deadband: `q = −slope · dv_eff · p_mw`, clipped to `±q_ratio·p_mw`. The
**deadband offset** (`|dv| − deadband`, floored at 0, sign-preserved) makes Q **continuous at
the deadband edge** — no step in the control law (the same continuity principle as the VDE
curve, Doc 1A §7, and the SVC droop, 6C §6). **Sign convention (code-to-physics):** `vm > 1`
→ `dv > 0` → `q < 0` (absorb, lowers voltage); `vm < 1` → `q > 0` (inject, raises voltage) —
pandapower's convention. **Units (docstring):** `droop_slope` is in 1/pu; to hit the
`±q_ratio·p_mw` clamp at deviation `dV_sat`, set `slope = q_ratio / dV_sat` (so 5.0 saturates
the 0.25 clamp at a 0.05 pu deviation). The docstring flags that the old spec's example
`droop_slope: 0.05` is far too weak (0.25% of `p_mw` at 5% deviation) — the shipped YAML uses
5.0. **Worked example** (`vm = 1.03`, `p = 1.5`, `slope = 5`, `deadband = 0.01`,
`q_ratio = 0.25`): `dv = 0.03`, `dv_eff = 0.02`, `q = −5·0.02·1.5 = −0.15`; `q_max = 0.375` →
`q = −0.15` MVAr (absorb, unsaturated). **Self-contained (numpy only)** — the reference the
`controller_fn` contract (12A §4) points to.

---

## 5. Section D — deadband_controller.py (85 lines)

A four-breakpoint piecewise-linear Q(V) — the VDE-AR-N 4110 family with every breakpoint
exposed as a kwarg.

```python
def compute_setpoints(vm_pu, p_mw, v_sat_low=0.96, v_db_low=0.99, v_db_high=1.01, v_sat_high=1.04, q_ratio=0.25):
    if not (v_sat_low < v_db_low <= v_db_high < v_sat_high): raise ValueError(...)
    q_max  = q_ratio * np.abs(p_mw)
    q_norm = np.interp(vm_pu, [v_sat_low, v_db_low, v_db_high, v_sat_high], [1.0, 0.0, 0.0, -1.0])
    return q_norm * q_max
```

**The characteristic, derived.** Full injection `+q_max` below `v_sat_low`, linear ramp to 0
across `[v_sat_low, v_db_low]`, **zero in the deadband** `[v_db_low, v_db_high]`, linear ramp
to `−q_max` across `[v_db_high, v_sat_high]`, full absorption `−q_max` above `v_sat_high`.
**The elegant implementation:** `np.interp` is monotone piecewise-linear with **flat
extrapolation** at both ends — which is *exactly* the clamped 5-segment characteristic, in one
call, on the normalised curve (`±1`), then scaled by `q_max` per DER. **Design rationale:** the
breakpoint-ordering guard (`v_sat_low < v_db_low ≤ v_db_high < v_sat_high`) fails fast on a
malformed curve; exposing every breakpoint as a kwarg lets a student sweep curve shapes from a
YAML **without touching framework code**. With the **default** breakpoints (0.96/0.99/1.01/1.04,
`q_ratio=0.25`) and `gate_clean_timesteps: true`, this plugin **closely tracks Scenario 4A**
(local Q(V) dry-run) — a strong end-to-end plumbing check — though not bit-identical (4A adds
the PT1 dynamics and curtailment backstop this omits, by design). **Worked example**
(`vm = 1.025`, defaults, `p = 1.5`): `q_norm = interp(1.025, …) = −0.5` (halfway across the
upper ramp), `q_max = 0.375` → `q = −0.1875` MVAr.

---

## 6. Section E — The controller YAMLs

**`droop_controller.yaml`** — configures the droop plugin: `name`, `label`, `module:
droop_controller.py` (relative to the YAML), `function: compute_setpoints`, `hardware: false`,
and `kwargs` (`droop_slope: 5.0`, `deadband: 0.01`, `q_ratio: 0.25`). The comments document
the run command, the `droop_slope` units, and the optional `gate_clean_timesteps`/
`clamp_to_net_limits` flags. **⚠ its `hardware: false` inline comment is stale** — it claims
`true` "raises `NotImplementedError`", contradicting the code (12B §11, §8).

**`deadband_controller.yaml`** — configures the deadband plugin with **non-default** kwargs
(`v_sat_low: 0.94`, `v_db_low: 0.985`, `v_db_high: 1.015`, `v_sat_high: 1.30`, `q_ratio: 0.42`)
and `gate_clean_timesteps: true`. **Design note:** it deliberately shows a *custom* curve
(wider deadband, a much wider `v_sat_high`, and a higher `q_ratio`) — the "sweep a curve from
YAML" workflow the plugin exists for, rather than just re-stating the defaults.

---

## 7. Section F — The network YAMLs

Four configs spanning every `source` and `strategy` (12C):

- **`simbench_rural_mv.yaml`** — `source: json`, `strategy: simbench_native`, `year: 2016`.
  The primary MV network (`1-MV-rural--2-sw`) exported to portable JSON; the comment documents
  the export step and the native→DWD fallback. The canonical portable-SimBench example.
- **`custom_lv_flat.yaml`** — `source: json`, `strategy: flat`, `year: 2016`. A user LV feeder
  with constant rated-capacity profiles (worst-case simultaneity screening, no weather data).
- **`custom_function.yaml`** — `source: function`, `module: my_network_loader.py`,
  `function: get_network`, `strategy: dwd_pvlib`, `year: 2024`. The programmatic-construction
  demo (CIGRE MV, profiles from DWD 691 Bremen + pvlib + BDEW).
- **`Scenario_3.yaml`** — `source: function`, `strategy: custom` with a 5-minute custom DWD
  `file_map` (station 998 CSVs) + `col_map`, `year: 2024`, tighter `v_min/v_max` (0.94/1.04).
  The custom-weather-data demo. **⚠ two issues (§8):** its header comment is a mis-pasted copy
  of `custom_function.yaml`'s, and its `data_dir: ..\..\data\custom` uses **Windows
  backslashes** that will not resolve on the Linux RPi.

**Design rationale — the example set is a coverage matrix:** json+native, json+flat,
function+dwd, function+custom — so a user finds a template for whichever (source, strategy)
combination matches their network, and can copy the closest one.

---

## 8. Findings and flags

**Flagged, not edited (cosmetic / portability — all in YAMLs, not the `.py` files):**

1. **`droop_controller.yaml` stale `hardware` comment.** The inline comment on
   `hardware: false` says `true` "is reserved … and raises `NotImplementedError`" — false; the
   code supports hardware plugins (12B §11). This is the **third location** of the stale
   `hardware: true` claim (with `plugin_runner.load_plugin`'s docstring L119 and L130).
   Recommend updating all three together.
2. **`Scenario_3.yaml` mis-pasted header.** Its top comment block is a verbatim copy of
   `custom_function.yaml`'s header ("custom_function.yaml — function source …") and does not
   describe this file's actual config (a CIGRE MV custom-5-min-weather stress demo). A
   copy-paste leftover; recommend rewriting the header to match the content.
3. **`Scenario_3.yaml` Windows-backslash `data_dir`.** `data_dir: ..\..\data\custom` uses
   backslash separators, which on the Linux RPi are literal filename characters, not path
   separators — the directory would not resolve. Recommend forward slashes
   (`../../data/custom`) for cross-platform portability. *(The file name `Scenario_3.yaml`
   also does not match its content — a minor naming nit.)*

**No bugs in the `.py` files.** `my_network_loader.py`, `droop_controller.py`, and
`deadband_controller.py` are clean, self-contained, and correct; `droop`/`deadband` both get
the deadband-continuity and clamp right, and `deadband`'s `np.interp` implementation is a neat,
correct expression of the clamped 5-segment curve.

---

## 9. Coverage checklist & Document 12 close-out

| Entity | File | Documented | Notes |
|--------|------|:----------:|-------|
| `get_network` | `my_network_loader.py` | ✅ | function-source contract |
| export snippet | `network_export.py` | ✅ | one-time JSON export |
| `compute_setpoints` (droop) | `droop_controller.py` | ✅ | proportional Q–P droop |
| `compute_setpoints` (deadband) | `deadband_controller.py` | ✅ | 4-breakpoint Q(V) via `np.interp` |
| controller config | `droop_controller.yaml` | ✅ | stale `hardware` comment flagged |
| controller config | `deadband_controller.yaml` | ✅ | custom-curve sweep example |
| network config | `simbench_rural_mv.yaml` | ✅ | json + simbench_native |
| network config | `custom_lv_flat.yaml` | ✅ | json + flat |
| network config | `custom_function.yaml` | ✅ | function + dwd_pvlib |
| network config | `Scenario_3.yaml` | ✅ | function + custom; header + backslash flagged |

**Flags raised (not edited):** stale `hardware` comment (`droop_controller.yaml`); mis-pasted
header + Windows-backslash `data_dir` (`Scenario_3.yaml`). No `.py` bugs.

---

### Document 12 — complete

Document 12 (the Custom Controller & Plugin Subsystem) is now fully documented across
**12A–12D**:
- **12A** `custom_controller.py` — the generic user-controller HIL loop.
- **12B** `plugin_runner.py` — loading/registering a controller plugin (software + hardware).
- **12C** `network_plugin.py` — loading/profiling/validating a user network.
- **12D** (this doc) — the loaders and worked examples.

**Subsystem-level findings:** one confirmed stale docstring (the `hardware: true` claim, now
located in **three** places — `plugin_runner` docstring ×2 and `droop_controller.yaml` ×1) and
two `Scenario_3.yaml` nits; otherwise the subsystem is clean and well-designed (faithful
`VoltVarController`/`profile_builder` reuse, defensive validation, temporary-registration
hygiene, strategy-faithful HC factory).

*This completes the scenario-runner and plugin documentation. Remaining Task 4 queue: Documents
9 (`benchmark_runner`/`executor`/`wizard`/`run_plan`/`run_benchmark_script`/`helpers`/`__main__`),
10 (`publisher`/`plot_results`/`network_plotter`), and 11 (`violation_detector`).*
