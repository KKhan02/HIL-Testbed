# TASK 4 — DOCUMENT 12B
## Custom Controller & Plugin Subsystem, Part 2: The Plugin Runner
### `plugin_runner.py` — load, register, and run a controller plugin (636 lines)
### Complete line-by-line walkthrough

> **Source verified:** read against the current `/mnt/project/plugin_runner.py`
> (**636 lines**, the project tree). Compiles. **Read alongside:** **12A**
> (`run_custom_controller_scenario`, which this drives), **Document 1B** (the
> `ArduinoSerialInterface` the hardware path uses), **Document 9** (`benchmark_runner`
> — `SCENARIO_REGISTRY`, `ScenarioSpec`, `run_benchmark`, `BenchmarkConfig`).
>
> **⚠ Contains a confirmed stale docstring** — `load_plugin`'s docstring claims
> `hardware: true` raises `NotImplementedError`, but the code **fully supports**
> hardware plugins (§7, §11).

---

## TABLE OF CONTENTS

1. [Overview — the plugin lifecycle](#1-overview)
2. [File logic flow — pseudocode](#2-file-logic-flow--pseudocode)
3. [Call-site map](#3-call-site-map)
4. [Section A — Contract & constants (L1–97)](#4-section-a--contract--constants-l197)
5. [Section B — load_plugin (L104–291)](#5-section-b--load_plugin-l104291)
6. [Section C — _import_controller_fn (L298–335)](#6-section-c--_import_controller_fn-l298335)
7. [Section D — _allocate_plugin_num (L342–344)](#7-section-d--_allocate_plugin_num-l342344)
8. [Section E — HardwareControllerFn (L347–411)](#8-section-e--hardwarecontrollerfn-l347411)
9. [Section F — register_and_run: import & hardware routing (L414–506)](#9-section-f--register_and_run-import--hardware-routing-l414506)
10. [Section G — _plugin_runner, registration, run & cleanup (L508–636)](#10-section-g--_plugin_runner-registration-run--cleanup-l508636)
11. [Findings and flags](#11-findings-and-flags)
12. [Coverage checklist](#12-coverage-checklist)

---

## 1. Overview

`plugin_runner.py` is the bridge between a **user's plugin** (a YAML config + a `.py`
controller) and the benchmark framework. It lets a researcher run their own controller
**alongside** the built-in scenarios in a single comparison table, without editing any
core file. The lifecycle:

1. **Load** — `load_plugin` reads and validates the YAML (§5).
2. **Import** — `_import_controller_fn` dynamically imports the named function from the
   named `.py` file (§6).
3. **Route** — software vs hardware (§9): a plugin's Q is computed either by the Python
   function (dry-run) or, if it declares `hardware: true` *and* a real run with a port,
   by the **researcher's own Arduino firmware** (real HIL).
4. **Register & run** — `register_and_run` temporarily inserts the plugin as a
   `ScenarioSpec` in `SCENARIO_REGISTRY` (number ≥ 10), runs the full benchmark so the
   custom row appears next to the built-ins, then **cleans the registry** in a `finally`
   (§10).

**Software vs hardware plugins.** A *software* plugin is a pure Python `controller_fn`
(the 12A contract). A *hardware* plugin additionally names an `.ino` firmware; when run
for real, `HardwareControllerFn` (§8) makes the Arduino compute Q via the standard V:/Q:
protocol — so a researcher can benchmark *their own algorithm on real silicon*, with the
Python function serving as the dry-run mirror. The routing rule is **"Python only ⇒ dry
run"**: hardware is used only when `hardware: true` **and** not dry-run **and** a port is
supplied.

**Note on the top-level import (L80).** `from custom_controller import
run_custom_controller_scenario` is at module top — so importing `plugin_runner` imports
`custom_controller` (the fact behind the "a broken `custom_controller` crashes at plugin
registration, not at the custom scenario" analysis; both now compile).

---

## 2. File logic flow — pseudocode

```
CONSTANTS: built-in scenario_ids (collision guard); required YAML fields; plugin-number floor (10)

load_plugin(yaml_path) → cfg dict:
    lazy-import PyYAML; read mapping; check required fields
    validate name (non-empty, not a built-in id), label, hardware (bool)
    hardware: true → require + resolve + check a 'firmware' .ino sketch
    resolve 'module' .py path (relative to the YAML folder); validate 'function' identifier
    validate kwargs (identifier keys), gate_clean_timesteps, clamp_to_net_limits

_import_controller_fn(module_path, function_name) → callable:
    importlib file-location import under a mangled unique sys.modules name; check attr is callable

_allocate_plugin_num() → first free registry key ≥ 10

class HardwareControllerFn(port, scenario_id):     # controller_fn backed by the Arduino
    __call__(vm_pu, p_mw): lazily open+configure the port; V:/Q: exchange → q
    close(): send END, release the port

register_and_run(yaml_path, net, profiles, network_id, benchmark_config, port):
    cfg = load_plugin; fn = _import_controller_fn; controller_fn = partial(fn, **kwargs)
    hardware routing: hw_fn only if hardware AND not dry_run AND port
    num = _allocate_plugin_num; spec = ScenarioSpec(_plugin_runner); config copy (dataclasses.replace)
    SCENARIO_REGISTRY[num] = spec
    try: bench = run_benchmark(...)   finally: pop registry entry; hw_fn.close()
    reconstruct a real ScenarioResult if a summary-dict shortcut leaked; return it
```

---

## 3. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `register_and_run` | function | `run_benchmark_script.py` / the CLI `--controller` path (Doc 9) — the public entry. |
| `load_plugin` | function | `register_and_run` (and callable standalone to validate a YAML). |
| `_import_controller_fn` | helper | `register_and_run`. |
| `_allocate_plugin_num` | helper | `register_and_run`. |
| `HardwareControllerFn` | class | Instantiated in `register_and_run` when routing to hardware. |
| `_plugin_runner` (nested) | function | Registered as `ScenarioSpec.runner`; called by `benchmark_runner` (Doc 9) — it calls `run_custom_controller_scenario` (12A). |
| `run_custom_controller_scenario` | function (12A) | Imported top-level (L80). |
| `ArduinoSerialInterface` | class (1B) | Lazily imported in `HardwareControllerFn.__call__`. |
| `SCENARIO_REGISTRY`, `ScenarioSpec`, `run_benchmark`, `BenchmarkConfig` | infra (Doc 9) | Registration + execution. |

---

## 4. Section A — Contract & constants (L1–97)

The module docstring restates the controller contract (the 12A `compute_setpoints(vm_pu,
p_mw) -> q_mvar` shape) and documents the YAML schema. Three constants govern registration:

- **`_BUILTIN_SCENARIO_IDS`** (L87) — the set of built-in `scenario_id`s, computed from
  `SCENARIO_REGISTRY`. **Design rationale:** the publisher writes `scenarios/<scenario_id>.json`,
  so a plugin whose `name` collided with a built-in would **silently overwrite** that
  scenario's output — `load_plugin` rejects the collision (§5).
- **`_REQUIRED_FIELDS`** (L91) — `name, label, module, function, hardware` — the mandatory
  YAML keys.
- **`_PLUGIN_NUM_FLOOR = 10`** (L97) — plugin registry numbers start at 10; built-ins
  occupy 1–6 and **7–9 are reserved** for future built-in scenarios.

---

## 5. Section B — load_plugin (L104–291)

The YAML validator — the largest function, all defensive schema checking with
field-naming error messages.

```python
    try: import yaml
    except ImportError: raise ImportError("plugin_runner requires PyYAML ...")
    yaml_path = Path(yaml_path).expanduser().resolve(); ... raw = yaml.safe_load(fh)
    missing = [f for f in _REQUIRED_FIELDS if f not in raw]; if missing: raise ValueError(...)
    name = raw["name"].strip(); if name in _BUILTIN_SCENARIO_IDS: raise ValueError("... collides ...")
    ... module_path = (yaml_path.parent / module_field).resolve(); if suffix != ".py": raise ...
    ... return {"name", "label", "module_path", "function", "hardware", "firmware_path", "kwargs", "gate_clean_timesteps", "clamp_to_net_limits"}
```

#### Line 104–291: Read and validate the plugin YAML

**Lazy PyYAML import** (L134) — PyYAML is only needed on the plugin path, so the rest of
the framework stays free of the dependency; a missing install gives an actionable message.
The validator then checks, each with a field-naming error: the top level is a mapping; all
required fields present; **`name` is non-empty and not a built-in id** (the collision guard,
§4); `label` non-empty; `hardware` is a real bool. **Path resolution is relative to the
YAML's own folder** (L227–230: `yaml_path.parent / module_field`) — the project's standing
rule that paths *inside* a YAML resolve against that YAML's location. `function` must be a
valid identifier; `kwargs` must be an identifier-keyed mapping; `gate_clean_timesteps`
(default `False`) and `clamp_to_net_limits` (default `True`) must be bools. Returns a
fully-resolved config dict.

**The hardware branch (L186–218) — the part the docstring contradicts.** When `hardware:
true`, the YAML **must** name a `firmware` field pointing to an existing `.ino` sketch
(resolved relative to the YAML, `.ino` suffix checked, file existence checked). **Design
rationale:** a hardware plugin runs the researcher's algorithm in *their own* firmware, and
naming the sketch makes a run **traceable to a specific firmware file** — the first of the
three flash failsafes (the CLI adds an interactive flash confirmation; the protocol
handshake in `configure()` is the third, failing loudly if the board doesn't speak the
protocol). **This directly contradicts the function's own docstring** (L119 "always False
(True raises)", L130 "NotImplementedError: hardware: true") — the docstring is **stale**;
the code supports hardware plugins. Flagged in §11.

---

## 6. Section C — _import_controller_fn (L298–335)

```python
    mod_name = f"_hil_plugin_{module_path.stem}_{abs(hash(str(module_path))) & 0xFFFFFF:06x}"
    spec = importlib.util.spec_from_file_location(mod_name, module_path)
    module = importlib.util.module_from_spec(spec); sys.modules[mod_name] = module
    try: spec.loader.exec_module(module)
    except Exception: sys.modules.pop(mod_name, None); raise
    if not hasattr(module, function_name): raise AttributeError("... no attribute ... Available callables: ...")
    fn = getattr(module, function_name); if not callable(fn): raise TypeError(...)
```

#### Line 298–335: Dynamic file-location import

Imports the plugin's function from a **file path** (not a dotted module name), so
`importlib.util.spec_from_file_location` is used rather than `import_module`. **Two design
rationales in the mangled module name** (`_hil_plugin_<stem>_<6-hex-of-path-hash>`): (a)
registering the module in `sys.modules` under a *unique* name makes dataclasses and pickling
inside the plugin work (they need the module resolvable), and (b) the path-hash suffix stops
two plugins whose files share a stem (`droop.py` in different folders) from clashing in
`sys.modules`. On an exec failure the entry is **popped** (no half-loaded module lingers).
The missing-attribute error **lists the available callables** — a helpful message for a
typo'd function name.

---

## 7. Section D — _allocate_plugin_num (L342–344)

```python
def _allocate_plugin_num() -> int:
    return max([_PLUGIN_NUM_FLOOR - 1, *SCENARIO_REGISTRY.keys()]) + 1
```

#### Line 342–344: Allocate the next free registry number

Returns the first free registry key **≥ 10** and above any existing key. **Design
rationale:** dynamic allocation (rather than a fixed number) lets sequential
`register_and_run` calls reuse numbers safely (the previous entry is removed in a `finally`,
§10), and keeps plugins clear of the built-in 1–6 and the reserved 7–9.

---

## 8. Section E — HardwareControllerFn (L347–411)

```python
class HardwareControllerFn:
    def __call__(self, vm_pu, p_mw):
        if self._iface is None: self._iface = ArduinoSerialInterface(self._port); self._iface.open(); ...
        if (self._configured_p is None or len != or not allclose): self._iface.configure(len(p), p); self._configured_p = p.copy()
        q, _latency_ms = self._iface.exchange_batched(vm_pu, p); return q
    def close(self): self._iface.close()   # sends END before closing the port
```

#### Line 347–411: The Arduino-backed controller function

A callable satisfying the same `fn(vm_pu, p_installed) -> q_mvar` contract as a Python
plugin — but the Q is computed by the **researcher's own firmware** on the Arduino. **Lazy
lifecycle:** the port is opened and the INIT/CFG/P handshake performed on the **first** call
(installed capacity is constant per run, passed every timestep by 12A); it **reconfigures
automatically** if `p_installed` changes (e.g. a hosting-capacity re-benchmark on a stressed
net). `close()` sends END (so the board resets, LED returns to steady) and releases the port
— called by `register_and_run`'s `finally`. **Design rationale — the firmware contract
(docstring):** the custom sketch must speak the exact `volt_var_arduino.ino` protocol
(Doc 5); a board without it fails `configure()` with `SerialConfigError` — the loudest flash
failsafe. `ArduinoSerialInterface` is imported *inside* `__call__` so the pyserial dependency
is only pulled on the hardware path.

---

## 9. Section F — register_and_run: import & hardware routing (L414–506)

```python
    cfg = load_plugin(yaml_path); fn = _import_controller_fn(cfg["module_path"], cfg["function"])
    controller_fn = functools.partial(fn, **cfg["kwargs"]) if cfg["kwargs"] else fn
    hw_fn = None
    if cfg["hardware"]:
        dry = benchmark_config.dry_run if benchmark_config else True
        if dry or port is None: logger.warning("... falling back to the Python dry-run mirror ...")
        else: hw_fn = HardwareControllerFn(port, cfg["name"]); controller_fn = hw_fn; logger.info("... HARDWARE mode ...")
```

#### Line 414–506: Load, bind kwargs, and route software/hardware

Loads and imports the plugin, then **binds the YAML `kwargs`** with `functools.partial` so
the framework always calls the fixed `fn(vm_pu, p_mw)` signature (the 12A contract). The
**hardware routing is a failsafe cascade** — a `HardwareControllerFn` is used **only** when
all three hold: `hardware: true`, **not** dry-run, **and** a serial port was supplied. In
every other combination it falls back to the Python function (the dry-run mirror the YAML is
required to name) with a warning naming the unused firmware. **Design rationale — "Python
only ⇒ dry run":** this makes it impossible to accidentally believe a dry run exercised the
hardware, and guarantees a hardware plugin always has a runnable software fallback.

---

## 10. Section G — _plugin_runner, registration, run & cleanup (L508–636)

```python
    def _plugin_runner(net_, profiles_, network_id="unknown", v_min=0.95, v_max=1.05,
                       publish_fn=None, enable_checkpointing=True, live_csv_rewrite_fn=None):
        return run_custom_controller_scenario(net_, profiles_, controller_fn=controller_fn,
            scenario_id=scenario_id, label=label, ..., gate_clean_timesteps=gate, clamp_to_net_limits=clamp)
    num = _allocate_plugin_num(); spec = ScenarioSpec(num, scenario_id, label, _plugin_runner, supports_lv=False)
    config_run = dataclasses.replace(benchmark_config, scenarios=run_scenarios + [num])
    SCENARIO_REGISTRY[num] = spec
    try: bench = run_benchmark(net, profiles, network_id=network_id, config=config_run)
    finally: SCENARIO_REGISTRY.pop(num, None); if hw_fn is not None: hw_fn.close()
    ... if isinstance(custom_result, dict): custom_result = ScenarioResult(...)   # reconstruct
```

#### Line 508–636: Register, run alongside the built-ins, and clean up

**`_plugin_runner`** is the `ScenarioSpec.runner` closure — it binds the plugin's
`controller_fn`, `scenario_id`, `label`, and the gate/clamp flags, and delegates to
`run_custom_controller_scenario` (12A). Its signature accepts exactly the kwargs
`benchmark_runner._build_kwargs` injects for a default-capability spec (`network_id`,
`v_min`, `v_max`, `publish_fn`, `enable_checkpointing`, `live_csv_rewrite_fn`).

**Three hygiene properties, each deliberate:**
- **Temporary registration.** The plugin is inserted at a dynamically-allocated number and
  **removed in a `finally`** — so the process-global `SCENARIO_REGISTRY` is clean for the
  next call even if `run_benchmark` raised. (Documented as *not thread-safe*; sequential
  calls are safe.)
- **Config immutability.** The scenario list is extended on a **copy** (`dataclasses.replace`)
  — the caller's `BenchmarkConfig` is never mutated. Running the full benchmark (not just the
  plugin) puts the custom row **next to the built-ins** in one comparison table.
- **Public-boundary reconstruction (L587–632).** If `benchmark_runner`'s "already complete"
  shortcut returned a plain summary **dict** (reloaded from a prior `scenarios/<id>.json`),
  `register_and_run` **rebuilds a real `ScenarioResult`** rather than leak the internal dict
  shape past its public API (external callers use `.scenario_id` etc.). A missing result
  raises `RuntimeError` with the isolated runner's traceback (benchmark isolation caught the
  plugin's failure; other scenarios' failures do not raise — they appear as failed rows).
  The `hw_fn.close()` in the `finally` releases the serial port on every exit path.

---

## 11. Findings and flags

**Flagged, not edited (stale docstring — confirmed):**

1. **`load_plugin` docstring contradicts the code on `hardware: true`.** The docstring says
   `hardware` is *"always False (True raises)"* (L119) and lists *"NotImplementedError:
   hardware: true"* (L130), but the code (L186–218) **fully supports** hardware plugins —
   validating the firmware sketch and routing to `HardwareControllerFn` (§8, §9). The
   **code is correct; the docstring is stale** (a leftover from when hardware was unimplemented).
   Recommend updating the docstring to describe the firmware requirement. *(Per the project's
   known-docstring-bug list, there may be a third such claim elsewhere in the module/CLI
   text — the ones in this function are the two load-bearing ones.)*

**Not bugs (documented for clarity):** the process-global registry is documented as
non-re-entrant (sequential-safe via `finally`); the summary-dict reconstruction is
deliberate public-boundary hygiene; the mangled `sys.modules` name is necessary for
plugin dataclasses/pickling and stem-collision avoidance; "Python only ⇒ dry run" is a
safety property, not a limitation.

---

## 12. Coverage checklist

| Entity | Lines | Kind | Documented | Where-used stated |
|--------|-------|------|:----------:|:-----------------:|
| module docstring / contract | 1–78 | doc | ✅ | ✅ |
| `_BUILTIN_SCENARIO_IDS` / `_REQUIRED_FIELDS` / `_PLUGIN_NUM_FLOOR` | 87–97 | constants | ✅ | ✅ |
| `load_plugin` | 104–291 | function | ✅ (stale hardware docstring flagged) | ✅ |
| `_import_controller_fn` | 298–335 | helper | ✅ | ✅ |
| `_allocate_plugin_num` | 342–344 | helper | ✅ | ✅ |
| `HardwareControllerFn` (`__init__`/`__call__`/`close`) | 347–411 | class | ✅ | ✅ |
| `register_and_run` | 414–636 | function | ✅ | ✅ |
| — import + kwargs bind + hardware routing | 468–506 | — | ✅ | — |
| — `_plugin_runner` (nested runner) | 512–536 | function | ✅ | ✅ |
| — registration + config copy + run | 538–573 | — | ✅ | — |
| — result retrieval + dict reconstruction | 579–636 | — | ✅ | — |

**Also covered (not standalone entities):** the plugin lifecycle, the built-in-id collision
guard, YAML-relative path resolution, the software-vs-hardware failsafe cascade ("Python
only ⇒ dry run"), the three flash failsafes, the mangled-`sys.modules` rationale, temporary
registration + config immutability, and public-boundary `ScenarioResult` reconstruction.

**Flags raised (not edited):** stale `hardware: true` docstring in `load_plugin` (code is
correct and supports hardware plugins).

---

*End of Document 12B. Next: Document 12C — `network_plugin.py` (775 lines): the custom-network
subsystem — loading, validating, and adapting a user-supplied pandapower network for the
benchmark.*
