"""
plugin_runner.py
================
YAML-configured custom controller plugins for the HIL benchmark framework.

Loads a controller function from an arbitrary Python file, wraps it into a
runner compatible with the ScenarioSpec.runner contract, registers it under
a dynamically allocated number (>= 10, never colliding with the built-in
scenarios 1–6), runs the full benchmark, and cleans the registry afterwards.
Pure addition — benchmark_runner.py, scenario_result.py,
sensitivity_coordinator.py, and volt_var_controller.py are untouched, and
SCENARIO_REGISTRY is restored in a try/finally even if the run raises.

YAML format
-----------
    name: my_droop_controller                      # required, unique id
    label: "Q-P Droop Controller (custom)"         # required, display name
    module: my_controllers/droop.py                # required, FILE PATH
    function: compute_setpoints                    # required, function name
    hardware: false                                # required, must be false
    kwargs:                                        # optional
      droop_slope: 0.05
      deadband: 0.01
    # Optional execution flags (see custom_controller.py):
    # gate_clean_timesteps: false
    # clamp_to_net_limits: true

`module` is a file path, NOT a dotted module name.  Relative paths are
resolved relative to the YAML file's directory, so a plugin folder
(controller .py + .yaml side by side) is self-contained and portable
between the laptop and the RPi.

`hardware: true` is reserved for a future Arduino-hosted plugin path and
raises NotImplementedError.

The named function must satisfy the controller_fn contract documented in
custom_controller.py:

    def compute_setpoints(vm_pu: np.ndarray, p_mw: np.ndarray,
                          **extra_kwargs) -> np.ndarray

Extra kwargs from the YAML are bound with functools.partial, so the
framework always calls the function as fn(vm_pu, p_mw).

Usage
-----
    from plugin_runner import register_and_run

    result = register_and_run(
        "example_plugins/droop_controller.yaml",
        net, profiles,
        network_id       = "1-MV-rural--2-sw",
        benchmark_config = config,
    )
    print(result.n_violation_steps)

    # If the caller also needs the full BenchmarkResult (comparison table,
    # publisher, CSV path):
    custom, bench = register_and_run(..., return_benchmark=True)
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

from benchmark_runner import (
    SCENARIO_REGISTRY,
    BenchmarkConfig,
    BenchmarkResult,
    ScenarioSpec,
    run_benchmark,
)
from scenario_result import ScenarioResult
from custom_controller import run_custom_controller_scenario

logger = logging.getLogger(__name__)

# scenario_ids reserved by the built-in registry.  The publisher writes
# scenarios/<scenario_id>.json, so a collision would silently overwrite a
# built-in scenario's output.
_BUILTIN_SCENARIO_IDS: frozenset[str] = frozenset(
    spec.scenario_id for spec in SCENARIO_REGISTRY.values()
)

_REQUIRED_FIELDS: tuple[str, ...] = (
    "name", "label", "module", "function", "hardware",
)

# First plugin number.  Built-ins occupy 1–6; 7–9 are left free for future
# built-in scenarios.
_PLUGIN_NUM_FLOOR: int = 10


# ===========================================================================
# load_plugin
# ===========================================================================

def load_plugin(yaml_path: Union[str, Path]) -> dict:
    """
    Read and validate a plugin YAML config file.

    Parameters
    ----------
    yaml_path : path to the YAML file.

    Returns
    -------
    dict with keys:
        name                 : str  — scenario_id for the custom run
        label                : str  — display name
        module_path          : Path — RESOLVED absolute path to the .py file
        function             : str  — function name inside the module
        hardware             : bool — always False (True raises)
        kwargs               : dict — extra keyword args ({} when absent)
        gate_clean_timesteps : bool — default False
        clamp_to_net_limits  : bool — default True

    Raises
    ------
    ImportError          : PyYAML not installed.
    FileNotFoundError    : YAML file or module file absent.
    ValueError           : any schema violation, with a message naming the
                           offending field.
    NotImplementedError  : hardware: true.
    """
    # Lazy import — PyYAML is only needed on the plugin path, and this keeps
    # the rest of the framework free of the dependency.
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "plugin_runner requires PyYAML to read plugin config files. "
            "Install it with:  pip install pyyaml   (inside hil_env on the RPi)."
        ) from exc

    yaml_path = Path(yaml_path).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Plugin YAML not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Plugin YAML {yaml_path} must contain a top-level mapping, "
            f"got {type(raw).__name__}."
        )

    missing = [f for f in _REQUIRED_FIELDS if f not in raw]
    if missing:
        raise ValueError(
            f"Plugin YAML {yaml_path} is missing required field(s): "
            f"{missing}. Required: {list(_REQUIRED_FIELDS)}."
        )

    # ---- name ----
    name = raw["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"'name' must be a non-empty string, got {name!r}.")
    name = name.strip()
    if name in _BUILTIN_SCENARIO_IDS:
        raise ValueError(
            f"'name' = {name!r} collides with a built-in scenario_id "
            f"({sorted(_BUILTIN_SCENARIO_IDS)}). Choose a different name — "
            "the publisher keys its per-scenario JSON files by scenario_id."
        )

    # ---- label ----
    label = raw["label"]
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"'label' must be a non-empty string, got {label!r}.")
    label = label.strip()

    # ---- hardware ----
    hardware = raw["hardware"]
    if not isinstance(hardware, bool):
        raise ValueError(
            f"'hardware' must be a boolean (true/false), got {hardware!r}."
        )
    firmware_path = None
    if hardware:
        # Hardware controller plugins: the researcher's algorithm runs in
        # their OWN Arduino firmware, which must implement the standard
        # serial protocol (INIT/CFG/P handshake + V:/Q: exchange + END).
        # The YAML must name the sketch so the run is traceable to a
        # specific firmware file, and the file must exist — the first of
        # the flash failsafes (the CLI adds an interactive flash
        # confirmation on top; the protocol handshake itself is the third:
        # a board without the expected protocol fails configure() loudly).
        fw_field = raw.get("firmware")
        if not isinstance(fw_field, str) or not fw_field.strip():
            raise ValueError(
                "hardware: true requires a 'firmware' field naming the "
                "Arduino sketch (.ino) implementing this controller. The "
                "named sketch must be flashed to the board before the run; "
                "the Python 'function' remains the dry-run mirror of the "
                "same algorithm."
            )
        firmware_path = Path(fw_field.strip()).expanduser()
        if not firmware_path.is_absolute():
            firmware_path = yaml_path.parent / firmware_path
        firmware_path = firmware_path.resolve()
        if firmware_path.suffix != ".ino":
            raise ValueError(
                f"'firmware' must point to a .ino sketch, got: {firmware_path}"
            )
        if not firmware_path.is_file():
            raise FileNotFoundError(
                f"Plugin firmware sketch not found: {firmware_path} "
                f"(resolved from 'firmware: {fw_field}' relative to "
                f"{yaml_path.parent})."
            )

    # ---- module (file path, resolved relative to the YAML's directory) ----
    module_field = raw["module"]
    if not isinstance(module_field, str) or not module_field.strip():
        raise ValueError(
            f"'module' must be a non-empty file path string, got "
            f"{module_field!r}."
        )
    module_path = Path(module_field.strip()).expanduser()
    if not module_path.is_absolute():
        module_path = yaml_path.parent / module_path
    module_path = module_path.resolve()
    if module_path.suffix != ".py":
        raise ValueError(
            f"'module' must point to a .py file, got: {module_path}"
        )
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Plugin module file not found: {module_path} "
            f"(resolved from 'module: {module_field}' relative to "
            f"{yaml_path.parent})."
        )

    # ---- function ----
    function = raw["function"]
    if not isinstance(function, str) or not function.isidentifier():
        raise ValueError(
            f"'function' must be a valid Python identifier, got {function!r}."
        )

    # ---- kwargs (optional) ----
    kwargs = raw.get("kwargs", {}) or {}
    if not isinstance(kwargs, dict):
        raise ValueError(
            f"'kwargs' must be a mapping of keyword arguments, got "
            f"{type(kwargs).__name__}."
        )
    bad_keys = [k for k in kwargs
                if not isinstance(k, str) or not k.isidentifier()]
    if bad_keys:
        raise ValueError(
            f"'kwargs' contains keys that are not valid Python identifiers: "
            f"{bad_keys}."
        )

    # ---- optional execution flags ----
    gate = raw.get("gate_clean_timesteps", False)
    if not isinstance(gate, bool):
        raise ValueError(
            f"'gate_clean_timesteps' must be a boolean, got {gate!r}."
        )
    clamp = raw.get("clamp_to_net_limits", True)
    if not isinstance(clamp, bool):
        raise ValueError(
            f"'clamp_to_net_limits' must be a boolean, got {clamp!r}."
        )

    logger.info(
        "load_plugin: '%s' (%s) — %s::%s | kwargs=%s | gate=%s | clamp=%s",
        name, label, module_path, function, kwargs, gate, clamp,
    )

    return {
        "name":                 name,
        "label":                label,
        "module_path":          module_path,
        "function":             function,
        "hardware":             hardware,
        "firmware_path":        firmware_path,   # None for software plugins
        "kwargs":               dict(kwargs),
        "gate_clean_timesteps": gate,
        "clamp_to_net_limits":  clamp,
    }


# ===========================================================================
# Module import
# ===========================================================================

def _import_controller_fn(module_path: Path, function_name: str) -> Callable:
    """
    Import `function_name` from the Python file at `module_path` using
    importlib file-location machinery (the 'module' field is a path, not a
    dotted name, so importlib.import_module does not apply).

    The module is registered in sys.modules under a mangled unique name so
    that (a) dataclasses/pickling inside the plugin work, and (b) two plugins
    whose files share a stem ('droop.py' in different folders) do not clash.
    """
    mod_name = f"_hil_plugin_{module_path.stem}_{abs(hash(str(module_path))) & 0xFFFFFF:06x}"

    spec = importlib.util.spec_from_file_location(mod_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build an import spec for {module_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise

    if not hasattr(module, function_name):
        raise AttributeError(
            f"Plugin module {module_path} has no attribute "
            f"'{function_name}'. Available callables: "
            f"{[n for n in dir(module) if not n.startswith('_') and callable(getattr(module, n))]}."
        )

    fn = getattr(module, function_name)
    if not callable(fn):
        raise TypeError(
            f"'{function_name}' in {module_path} is not callable "
            f"({type(fn).__name__})."
        )
    return fn


# ===========================================================================
# register_and_run
# ===========================================================================

def _allocate_plugin_num() -> int:
    """First free registry key >= 10, above any existing key."""
    return max([_PLUGIN_NUM_FLOOR - 1, *SCENARIO_REGISTRY.keys()]) + 1


class HardwareControllerFn:
    """
    Serial-backed controller_fn: satisfies the fn(vm_pu, p_installed_mw)
    contract of run_custom_controller_scenario() by delegating the Q
    computation to the researcher's OWN firmware on the Arduino, via the
    standard V:/Q: exchange.

    Lifecycle
    ---------
    Lazy: the port is opened and the INIT/CFG/P handshake performed on the
    FIRST call (p_installed is constant per run — custom_controller.py
    passes ctrl.p_installed_mw every timestep). Reconfiguration happens
    automatically if p_installed changes (HC re-benchmark on a stressed
    net). close() must be called by the owner (register_and_run's finally).

    Firmware contract (documented for plugin authors)
    -------------------------------------------------
    The custom sketch must speak the same protocol as volt_var_arduino.ino:
      INIT:<n>  -> ACK:INIT      CFG:<5 floats> -> ACK:CFG
      P:<n floats> -> ACK:P      V:<n floats>   -> Q:<n floats>
      END       -> reset
    The CFG values are the CLI-configured Q(V) parameters; a custom
    algorithm is free to ignore them, but must ACK the message. A board
    running firmware without this protocol fails configure() with
    SerialConfigError — the loudest of the flash failsafes.
    """

    def __init__(self, port: str, scenario_id: str):
        self._port        = port
        self._scenario_id = scenario_id
        self._iface       = None
        self._configured_p = None

    def __call__(self, vm_pu, p_mw):
        import numpy as _np
        from volt_var_controller import ArduinoSerialInterface
        if self._iface is None:
            self._iface = ArduinoSerialInterface(self._port)
            self._iface.open()
            logger.info(
                "[HW plugin '%s'] Serial port %s opened.",
                self._scenario_id, self._port,
            )
        p = _np.asarray(p_mw, dtype=float)
        if (self._configured_p is None
                or len(self._configured_p) != len(p)
                or not _np.allclose(self._configured_p, p)):
            self._iface.configure(len(p), p)
            self._configured_p = p.copy()
            logger.info(
                "[HW plugin '%s'] Arduino configured for %d DERs.",
                self._scenario_id, len(p),
            )
        q, _latency_ms = self._iface.exchange_batched(
            _np.asarray(vm_pu, dtype=float), p,
        )
        return q

    def close(self) -> None:
        if self._iface is not None:
            try:
                self._iface.close()   # sends END before closing the port
            finally:
                self._iface = None
                self._configured_p = None


def register_and_run(
        yaml_path:        Union[str, Path],
        net,
        profiles:         dict,
        network_id:       str,
        benchmark_config: Optional[BenchmarkConfig] = None,
        *,
        return_benchmark: bool = False,
        port:             Optional[str] = None,
) -> Union[ScenarioResult, Tuple[ScenarioResult, BenchmarkResult]]:
    """
    Load a plugin, register it in SCENARIO_REGISTRY, run the benchmark, and
    clean the registry afterwards.

    The custom scenario runs ALONGSIDE the scenarios already listed in
    benchmark_config.scenarios (a copy of the config is built with
    dataclasses.replace — the caller's config object is never mutated), so
    the comparison table printed by run_benchmark() includes the custom
    controller row next to the built-ins.

    Parameters
    ----------
    yaml_path        : path to the plugin YAML (see load_plugin()).
    net, profiles    : as for run_benchmark().  benchmark_runner deep-copies
                       net per scenario as usual.
    network_id       : human-readable network identifier.
    benchmark_config : BenchmarkConfig or None (None -> defaults, same as
                       run_benchmark()).
    return_benchmark : keyword-only.  When True, return
                       (ScenarioResult, BenchmarkResult) so the caller can
                       reuse the full comparison table, CSV path, and
                       publisher wiring.  Default False -> ScenarioResult
                       only.

    Returns
    -------
    ScenarioResult for the custom scenario (or the tuple above).

    Raises
    ------
    RuntimeError : if the custom scenario failed inside run_benchmark()
                   (benchmark isolation caught its exception).  The original
                   traceback text is included.  Failures of OTHER scenarios
                   do not raise — they appear as failed rows in the
                   comparison table, consistent with run_benchmark().
    Plus anything load_plugin() / _import_controller_fn() raise.

    Notes
    -----
    Registry mutation is process-global and not re-entrant: do not call
    register_and_run() concurrently from multiple threads.  Sequential
    repeated calls are safe — the entry is removed in a finally block and
    numbers are allocated dynamically.
    """
    cfg = load_plugin(yaml_path)
    fn  = _import_controller_fn(cfg["module_path"], cfg["function"])

    # Bind YAML kwargs so the framework always calls fn(vm_pu, p_mw).
    controller_fn = functools.partial(fn, **cfg["kwargs"]) if cfg["kwargs"] else fn

    # ---- Hardware routing (failsafe cascade) --------------------------------
    # A hardware plugin runs its algorithm in the researcher's own firmware.
    # It is used ONLY when all three hold: the YAML declares hardware: true,
    # the run is not a dry run, and a serial port was supplied. In every
    # other combination the Python function (the dry-run mirror the YAML is
    # required to name) is used instead — "Python only ⇒ dry run".
    hw_fn: Optional[HardwareControllerFn] = None
    if cfg["hardware"]:
        dry = benchmark_config.dry_run if benchmark_config is not None else True
        if dry or port is None:
            logger.warning(
                "[Plugin '%s'] hardware: true but %s — falling back to the "
                "Python dry-run mirror '%s'. The firmware sketch %s is NOT "
                "used in this run.",
                cfg["name"],
                "dry_run=True" if dry else "no serial port supplied",
                cfg["function"], cfg["firmware_path"],
            )
        else:
            hw_fn = HardwareControllerFn(port, cfg["name"])
            controller_fn = hw_fn
            logger.info(
                "[Plugin '%s'] HARDWARE mode: Q computed by the Arduino on "
                "%s. Expected firmware: %s (must be flashed beforehand — "
                "the protocol handshake will fail loudly if the board does "
                "not implement INIT/CFG/P + V:/Q:).",
                cfg["name"], port, cfg["firmware_path"],
            )

    scenario_id = cfg["name"]
    label       = cfg["label"]
    gate        = cfg["gate_clean_timesteps"]
    clamp       = cfg["clamp_to_net_limits"]

    # Runner matching the ScenarioSpec.runner contract.  With all
    # ScenarioSpec capability flags at their defaults, benchmark_runner's
    # _build_kwargs() injects exactly: network_id, v_min, v_max, publish_fn,
    # enable_checkpointing.
    def _plugin_runner(
            net_,
            profiles_,
            network_id: str = "unknown",
            v_min:      float = 0.95,
            v_max:      float = 1.05,
            publish_fn        = None,
            enable_checkpointing: bool = True,
            live_csv_rewrite_fn         = None,
    ) -> ScenarioResult:
        return run_custom_controller_scenario(
            net_,
            profiles_,
            controller_fn        = controller_fn,
            scenario_id          = scenario_id,
            label                = label,
            network_id           = network_id,
            v_min                = v_min,
            v_max                = v_max,
            publish_fn           = publish_fn,
            enable_checkpointing = enable_checkpointing,
            live_csv_rewrite_fn  = live_csv_rewrite_fn,
            gate_clean_timesteps = gate,
            clamp_to_net_limits  = clamp,
        )

    if benchmark_config is None:
        benchmark_config = BenchmarkConfig()

    num  = _allocate_plugin_num()
    spec = ScenarioSpec(
        num         = num,
        scenario_id = scenario_id,
        label       = label,
        runner      = _plugin_runner,
        supports_lv = False,   # a Q controller needs DERs; reuse the LV skip
    )

    # Extend the scenario list on a COPY of the config.
    run_scenarios = list(benchmark_config.scenarios)
    if num not in run_scenarios:
        run_scenarios.append(num)
    config_run = dataclasses.replace(benchmark_config, scenarios=run_scenarios)

    logger.info(
        "register_and_run: '%s' registered as scenario %d — running "
        "scenarios %s on %s.",
        scenario_id, num, sorted(run_scenarios), network_id,
    )

    SCENARIO_REGISTRY[num] = spec
    try:
        bench = run_benchmark(
            net,
            profiles,
            network_id = network_id,
            config     = config_run,
        )
    finally:
        # Registry must be clean for subsequent calls even if run_benchmark
        # raised (e.g. profile validation error).
        SCENARIO_REGISTRY.pop(num, None)
        # Hardware plugins own a serial port — release it (sends END so the
        # board resets and the LED returns to steady ON) even on failure.
        if hw_fn is not None:
            hw_fn.close()

    custom_result = bench.results.get(num)
    if custom_result is None:
        tb = bench.errors.get(num, "no traceback recorded")
        raise RuntimeError(
            f"Custom scenario '{scenario_id}' (num {num}) failed inside "
            f"run_benchmark(). Traceback from the isolated runner:\n{tb}"
        )

    if isinstance(custom_result, dict):
        # bench.results[num] holds a plain summary dict here when the
        # layer-1 "already complete" check in benchmark_runner.py's
        # scenario loop reloaded it from a prior scenarios/<id>.json --
        # that's fine for benchmark_runner.py's own internal use
        # (_ok_summary_row() handles both shapes), but register_and_run()
        # is a public boundary and external callers reasonably expect a
        # real ScenarioResult with real attributes (see e.g.
        # run_benchmark_script.py's `custom_result.scenario_id` usage).
        # Reconstruct one here rather than leaking the internal shortcut
        # past this boundary.
        _d = custom_result
        custom_result = ScenarioResult(
            scenario_id = _d.get("scenario_id", scenario_id),
            network_id  = _d.get("network_id", network_id),
            records     = [],
            elapsed_s   = _d.get("elapsed_s", 0.0),
            n_timesteps                  = _d.get("n_timesteps", 0),
            n_converged                  = _d.get("n_converged", 0),
            n_violation_steps            = _d.get("n_violation_steps", 0),
            violation_duration_h         = _d.get("violation_duration_h", 0.0),
            total_overvoltage_bus_steps  = _d.get("total_overvoltage_bus_steps", 0),
            total_undervoltage_bus_steps = _d.get("total_undervoltage_bus_steps", 0),
            total_overloaded_line_steps  = _d.get("total_overloaded_line_steps", 0),
            total_overloaded_trafo_steps = _d.get("total_overloaded_trafo_steps", 0),
            max_vm_pu                    = _d.get("max_vm_pu", float("nan")),
            min_vm_pu                    = _d.get("min_vm_pu", float("nan")),
            max_line_loading_pct         = _d.get("max_line_loading_pct", float("nan")),
            max_trafo_loading_pct        = _d.get("max_trafo_loading_pct", float("nan")),
            vdi                          = _d.get("vdi", float("nan")),
            q_total_mvar_abs             = _d.get("q_total_mvar_abs"),
            reactive_energy_mvarh        = _d.get("reactive_energy_mvarh"),
            curtailment_steps            = _d.get("curtailment_steps", 0),
            curtailed_energy_mwh         = _d.get("curtailed_energy_mwh"),
            curtail_exhausted_steps      = _d.get("curtail_exhausted_steps", 0),
            svc_bus                      = _d.get("svc_bus"),
            svc_q_max                    = _d.get("svc_q_max"),
            total_losses_mwh             = _d.get("total_losses_mwh"),
            grid_import_mwh              = _d.get("grid_import_mwh"),
            grid_export_mwh              = _d.get("grid_export_mwh"),
            der_gen_mwh                  = _d.get("der_gen_mwh"),
            load_demand_mwh              = _d.get("load_demand_mwh"),
            coordination_steps           = _d.get("coordination_steps"),
            coordination_rate            = _d.get("coordination_rate"),
            q_saturation_rate            = _d.get("q_saturation_rate"),
        )

    if return_benchmark:
        return custom_result, bench
    return custom_result
