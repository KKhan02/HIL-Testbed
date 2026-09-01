"""
executor.py — the runtime layer between a RunPlan and run_benchmark().

Responsibilities
----------------
1. Translate the wizard's RunPlan into the framework's BenchmarkConfig
   (build_benchmark_config).
2. Resolve the network and profiles from the four network source types
   (preset / simbench_code / custom / plugin) and three dataset source
   types (simbench_native / dwd / custom), applying the optional pre-run
   network modifications and scaling (build_net_and_profiles).
3. Validate everything the wizard COULD NOT validate at input time because
   it required filesystem, network, or serial-port access — and fail with
   one clear message and a distinct exit code BEFORE a multi-hour annual
   run starts.
4. Orchestrate the run: execute(plan) -> int is the public entry point
   __main__.py calls; the returned int is the process exit code.

Currently the focus_buses is display-only ando only stream_every_k IS
consumed: it drives PublishHandle.update_every_k for live streaming.

Separation of concerns
----------------------
This module is CLI plumbing only. It never modifies the protected
framework modules at runtime beyond two sanctioned, documented channels:
  - volt_var_controller.set_qv_parameters()  (per-run Q(V) characteristic,
    kept consistent across dry-run curve, coordinator sizing, and the CFG:
    message to the Arduino), and
  - rebinding violation_detector's module-level threshold constants, whose
    detect_violations() defaults are late-bound for exactly this purpose.
Students who script directly against run_benchmark() are unaffected: both
channels default to the historical constants unless a RunPlan asks
otherwise, and nothing here is imported by the framework.
"""

from __future__ import annotations

import enum
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple
import logging
from rich.logging import RichHandler

from ._console import console
from .run_plan import RunPlan
from .helpers import (
    print_run_plan,
    print_section_header,
    print_error_message,
    print_summary_table,
)

_logging_configured = False


def _configure_cli_logging(log_path: Path | None = None) -> None:
    """
    Attach Rich console logging (as before) and, when log_path is given,
    a plain-text FileHandler writing every INFO+ message to disk. This is
    what makes the CFG push/ACK handshake and any other console-only
    message durably recoverable after a run — previously it existed only
    in the terminal's scrollback, which tmux sessions can silently
    truncate on long HIL runs (confirmed: a ~27,000s run exceeded a
    1872-line tmux history-limit well before finishing).
    """
    global _logging_configured
    if _logging_configured:
        return
    
    handlers = [RichHandler(console=console, rich_tracebacks=True,
                             markup=False, show_path=True)]
    
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                         datefmt="[%H:%M:%S]", handlers=handlers)
    
    class _PeriodicFilter(logging.Filter):
        """
        Pass every Nth INFO message from a noisy logger, but always pass
        WARNING and above so genuine issues are never suppressed.

        every_n=96 means one message per simulated day at 15-min resolution.
        """
        def __init__(self, every_n: int = 96):
            super().__init__()
            self.every_n = every_n
            self._count  = 0

        def filter(self, record: logging.LogRecord) -> bool:
            if record.levelno >= logging.WARNING:
                return True          # warnings and errors always pass through
            self._count += 1
            return self._count % self.every_n == 0

    _sc_logger = logging.getLogger("sensitivity_coordinator")
    _sc_logger.addFilter(_PeriodicFilter(every_n=96))
    _logging_configured = True

# ===========================================================================
# Exit codes  (confirmed set — __main__.py calls sys.exit(execute(plan)))
# ===========================================================================

class ExitCode(enum.IntEnum):
    """
    Process exit codes for the CLI.

    Distinct codes let a student (or a shell script / CI job) tell apart
    "your configuration was invalid" from "your network file failed to
    load" from "the hardware did not respond" from "the simulation itself
    crashed" without reading the traceback. 130 follows the shell
    convention for SIGINT (Ctrl+C).
    """
    OK                 = 0
    CONFIG_ERROR       = 2   # invalid/contradictory RunPlan values
    NETWORK_LOAD_ERROR = 3   # network source failed to load / build
    DATASET_ERROR      = 4   # profile source missing / failed to build
    PLUGIN_ERROR       = 5   # plugin YAML / module invalid (either plugin type)
    HARDWARE_ERROR     = 6   # serial port absent / Arduino handshake failed
    SIMULATION_ERROR   = 7   # crash inside run_benchmark / plugin runner
    PUBLISH_ERROR      = 8   # run succeeded, publishing its results failed
    INTERRUPTED        = 130 # Ctrl+C


class ExecutorError(Exception):
    """
    Base class for all pre-run failures raised by this module.

    Each subclass carries the ExitCode that execute() returns and a
    context string for helpers.print_error_message(). The message must be
    actionable: name the offending value AND what to change.
    """
    exit_code: ExitCode = ExitCode.CONFIG_ERROR
    context:   str      = "Run configuration"

    def __init__(self, message: str, context: str = None):
        super().__init__(message)
        if context is not None:
            self.context = context


class ConfigError(ExecutorError):
    exit_code = ExitCode.CONFIG_ERROR
    context   = "Run configuration validation"


class NetworkLoadError(ExecutorError):
    exit_code = ExitCode.NETWORK_LOAD_ERROR
    context   = "Network loading"


class DatasetError(ExecutorError):
    exit_code = ExitCode.DATASET_ERROR
    context   = "Dataset / profile building"


class PluginError(ExecutorError):
    exit_code = ExitCode.PLUGIN_ERROR
    context   = "Plugin loading"


class HardwareError(ExecutorError):
    exit_code = ExitCode.HARDWARE_ERROR
    context   = "Arduino hardware pre-check"


class PublishError(ExecutorError):
    exit_code = ExitCode.PUBLISH_ERROR
    context   = "Result publishing"


# ===========================================================================
# Framework import path
# ===========================================================================
# The CLI package (this file) and the framework modules (benchmark_runner.py
# etc.) live in different directories; run_benchmark_script.py resolves this
# with a parent.parent sys.path insertion. The executor mirrors that but
# searches, so the CLI keeps working whether the package sits beside the
# framework files or one level below the project root (laptop vs RPi tree).

_FRAMEWORK_SENTINEL = "benchmark_runner.py"


def _ensure_framework_on_path() -> Path:
    """
    Locate the framework directory (the one containing benchmark_runner.py)
    and prepend it to sys.path. Returns the directory. Raises ConfigError
    with the searched locations if not found — a student who moved the CLI
    folder gets told exactly where the executor looked.
    """
    try:
        import benchmark_runner  # noqa: F401  (already importable)
        return Path(benchmark_runner.__file__).resolve().parent
    except ImportError:
        pass

    here = Path(__file__).resolve().parent
    candidates = [
        here,                    # flat layout (everything in one folder)
        here.parent,             # CLI package inside the project root
        here.parent.parent,      # CLI package one level deeper
    ]
    # Also scan the project root's immediate subdirectories (e.g. a
    # scenario_runners/ folder) without walking the whole tree.
    for base in (here.parent, here.parent.parent):
        if base.is_dir():
            candidates.extend(p for p in base.iterdir() if p.is_dir())

    seen = set()
    for cand in candidates:
        cand = cand.resolve()
        if cand in seen:
            continue
        seen.add(cand)
        if (cand / _FRAMEWORK_SENTINEL).is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand

    raise ConfigError(
        f"Could not locate the benchmark framework ({_FRAMEWORK_SENTINEL}). "
        f"Searched: {sorted(str(p) for p in seen)}. Keep the CLI package "
        f"inside (or one level below) the project root that contains the "
        f"framework modules.",
        context="Framework path resolution",
    )


# ===========================================================================
# Preset catalogue loaders
# ===========================================================================
# network_catalogue.py stores MENU DATA ONLY (its own docstring: "The
# executor is responsible for turning preset_name into an actual network").
# This mapping is that responsibility. Constructor names were verified
# against pandapower 3.4.0's pandapower.networks module; the Dickert preset
# names encode (feeders_range, linetype, customer, case) with the API's
# literal values ('average', not 'avg'; 'C&OHL' for the cable/overhead mix).

def _preset_loaders() -> dict:
    import simbench as sb
    import pandapower.networks as pn

    loaders: dict[str, Callable] = {
        # --- SimBench (native profiles available) ---
        "1-MV-rural--2-sw": lambda: sb.get_simbench_net("1-MV-rural--2-sw"),
        # --- CIGRE ---
        "cigre_mv_no_der":  lambda: pn.create_cigre_network_mv(with_der=False),
        "cigre_mv_pv_wind": lambda: pn.create_cigre_network_mv(with_der="pv_wind"),
        "cigre_lv":         lambda: pn.create_cigre_network_lv(),
        # --- Kerber standard ---
        "kerber_landnetz_kabel_1":        pn.create_kerber_landnetz_kabel_1,
        "kerber_landnetz_kabel_2":        pn.create_kerber_landnetz_kabel_2,
        "kerber_landnetz_freileitung_1":  pn.create_kerber_landnetz_freileitung_1,
        "kerber_landnetz_freileitung_2":  pn.create_kerber_landnetz_freileitung_2,
        "kerber_vorstadtnetz_kabel_1":    pn.create_kerber_vorstadtnetz_kabel_1,
        "kerber_vorstadtnetz_kabel_2":    pn.create_kerber_vorstadtnetz_kabel_2,
        "kerber_dorfnetz":                pn.create_kerber_dorfnetz,
        # --- Kerber extreme ---
        "kb_extrem_landnetz_kabel":         pn.kb_extrem_landnetz_kabel,
        "kb_extrem_landnetz_freileitung":   pn.kb_extrem_landnetz_freileitung,
        "kb_extrem_landnetz_kabel_trafo":   pn.kb_extrem_landnetz_kabel_trafo,
        "kb_extrem_landnetz_frltg_trafo":   pn.kb_extrem_landnetz_freileitung_trafo,
        "kb_extrem_dorfnetz":               pn.kb_extrem_dorfnetz,
        "kb_extrem_dorfnetz_trafo":         pn.kb_extrem_dorfnetz_trafo,
        "kb_extrem_vorstadtnetz_1":         pn.kb_extrem_vorstadtnetz_1,
        "kb_extrem_vorstadtnetz_2":         pn.kb_extrem_vorstadtnetz_2,
        "kb_extrem_vorstadtnetz_trafo_1":   pn.kb_extrem_vorstadtnetz_trafo_1,
        "kb_extrem_vorstadtnetz_trafo_2":   pn.kb_extrem_vorstadtnetz_trafo_2,
        # --- Synthetic voltage-control LV ---
        "synthetic_lv_rural_1":   lambda: pn.create_synthetic_voltage_control_lv_network("rural_1"),
        "synthetic_lv_rural_2":   lambda: pn.create_synthetic_voltage_control_lv_network("rural_2"),
        "synthetic_lv_village_1": lambda: pn.create_synthetic_voltage_control_lv_network("village_1"),
        "synthetic_lv_village_2": lambda: pn.create_synthetic_voltage_control_lv_network("village_2"),
        "synthetic_lv_suburb_1":  lambda: pn.create_synthetic_voltage_control_lv_network("suburb_1"),
    }

    # --- Dickert: 18 presets encode the four create_dickert_lv_network args ---
    _case_map = {"good": "good", "average": "average", "bad": "bad"}
    for rng in ("short", "middle", "long"):
        for lt_key, lt_api in (("cable", "cable"), ("cohl", "C&OHL")):
            for cust in ("single", "multiple"):
                for case_key, case_api in _case_map.items():
                    name = f"dickert_{rng}_{lt_key}_{cust}_{case_key}"
                    loaders[name] = (
                        lambda r=rng, l=lt_api, c=cust, k=case_api:
                        pn.create_dickert_lv_network(
                            feeders_range=r, linetype=l, customer=c, case=k,
                        )
                    )
    return loaders


# SimBench codes for presets whose native profiles exist (dataset
# source_type == "simbench_native" is only valid for these + raw codes).
_PRESET_SIMBENCH_CODES = {"1-MV-rural--2-sw": "1-MV-rural--2-sw"}

# CIGRE MV preset names — the only networks OPF (Scenario 6) is validated on.
_CIGRE_MV_PRESETS = {"cigre_mv_no_der", "cigre_mv_pv_wind"}


# ===========================================================================
# TASK 1 — RunPlan -> BenchmarkConfig
# ===========================================================================
# Study -> scenario mapping against the VERIFIED SCENARIO_REGISTRY numbering:
#   1 = baseline | 2 = oltc | 3 = svc | 4 = volt_var_local (4A)
#   5 = volt_var_coord (4B) | 6 = opf
# Confirmed decisions (July 2026):
#   scenario_comparison -> [1,2,3,4,5]  (OPF excluded: unreliable on
#                                        SimBench MV; it has its own study)
#   voltage_variation   -> [4,5]        (local + coordinated Volt-Var)
#   hosting_capacity    -> []           + run_hc=True; the stressed
#                          re-benchmark is opt-in via plan.hc_stressed
#   opf_benchmark       -> [6]          (NOT [5] — 5 is Volt-Var coordinated)

STUDY_SCENARIOS: dict[str, list[int]] = {
    "scenario_comparison": [1, 2, 3, 4, 5],
    "voltage_variation":   [4, 5],
    "hosting_capacity":    [],
    "opf_benchmark":       [6],
}

_HC_STRESS_SCENARIOS = [1, 2, 3, 4, 5]   # scenario set for the stressed re-benchmark


def _check_opf_network_compatibility(plan: RunPlan) -> None:
    """
    Guard the opf_benchmark study against networks runopp() cannot solve.

    OPF is validated on CIGRE MV only. On SimBench MV networks runopp()
    fails because the HV/MV transformer impedance contrast produces an
    ill-conditioned PYPOWER admittance matrix (Knowledge Base §5.3) — the
    error message names that mechanism, not just "unsupported". Custom and
    plugin networks are unknown territory: the user is asked to confirm
    explicitly rather than being hard-blocked (a researcher may have built
    an OPF-solvable net) or silently allowed (a student almost certainly
    has not).
    """
    net_cfg = plan.network

    if net_cfg.source_type == "preset":
        if net_cfg.preset_name in _CIGRE_MV_PRESETS:
            return
        if net_cfg.preset_name in _PRESET_SIMBENCH_CODES:
            raise ConfigError(
                f"Study 'opf_benchmark' cannot run on the SimBench preset "
                f"'{net_cfg.preset_name}': pandapower's runopp() fails on "
                f"SimBench MV networks because the HV/MV transformer "
                f"impedance contrast makes the PYPOWER admittance matrix "
                f"ill-conditioned. Select the CIGRE MV preset "
                f"(cigre_mv_pv_wind) for the OPF study."
            )
        raise ConfigError(
            f"Study 'opf_benchmark' is validated on CIGRE MV networks only; "
            f"preset '{net_cfg.preset_name}' is not one of "
            f"{sorted(_CIGRE_MV_PRESETS)}. Select cigre_mv_pv_wind (or "
            f"cigre_mv_no_der) for the OPF study."
        )

    if net_cfg.source_type == "simbench_code":
        raise ConfigError(
            f"Study 'opf_benchmark' cannot run on SimBench network "
            f"'{net_cfg.simbench_code}': runopp() fails on SimBench MV "
            f"networks — the HV/MV transformer impedance contrast makes "
            f"the PYPOWER admittance matrix ill-conditioned. Use the "
            f"CIGRE MV preset for the OPF study."
        )

    # custom / plugin: unknown net — explicit confirmation required.
    from rich.prompt import Confirm
    console.print(
        "[warning]OPF (runopp) is only validated on CIGRE MV in this "
        "framework. It is known to fail on SimBench MV (ill-conditioned "
        "admittance matrix from the HV/MV transformer impedance contrast) "
        "and is untested on custom/plugin networks.[/warning]"
    )
    if not Confirm.ask(
        "Attempt the OPF study on this custom network anyway?",
        default=False, console=console,
    ):
        raise ConfigError(
            "OPF study cancelled by user for an unvalidated custom/plugin "
            "network. Use the CIGRE MV preset for a supported OPF run."
        )


def build_benchmark_config(
        plan: RunPlan,
        profile_factory: Optional[Callable] = None,
        publish_fn=None,
):
    """
    Translate a RunPlan into the framework's BenchmarkConfig.

    Verified signature being targeted (benchmark_runner.py):
        run_benchmark(net, profiles, network_id, config) -> BenchmarkResult
    so this function produces only the config; build_net_and_profiles()
    produces the other inputs.

    Parameters
    ----------
    plan            : the wizard's RunPlan.
    profile_factory : callable(net) -> profiles dict, required when the
                      HC-stressed re-benchmark is requested
                      (plan.hc_stressed) — built by build_net_and_profiles
                      from the SAME dataset strategy as the outer run.
    publish_fn      : PublishHandle for live streaming, or None.
    """
    _ensure_framework_on_path()
    from benchmark_runner import BenchmarkConfig

    if plan.study not in STUDY_SCENARIOS:
        raise ConfigError(
            f"Unknown study '{plan.study}'. Valid studies: "
            f"{sorted(STUDY_SCENARIOS)}."
        )

    if plan.study == "opf_benchmark":
        _check_opf_network_compatibility(plan)

    scenarios = list(STUDY_SCENARIOS[plan.study])
    run_hc           = plan.study == "hosting_capacity"
    run_hc_scenarios = bool(run_hc and plan.hc_stressed)

    if run_hc_scenarios and profile_factory is None:
        # benchmark_runner._validate_inputs would reject this anyway, but
        # failing here names the CLI-level cause instead of the framework's.
        raise ConfigError(
            "The HC-stressed re-benchmark (hc_stressed=True) needs a "
            "profile factory rebuilt from the run's dataset strategy, and "
            "none could be constructed for this dataset source. Re-run "
            "with hc_stressed disabled, or use a dataset/plugin source "
            "that supports profile rebuilding."
        )

    out_dir = Path(plan.output_dir or "runs") / plan.run_id

    return BenchmarkConfig(
        scenarios           = scenarios,
        dry_run             = not plan.hardware,
        port                = plan.port,
        output_dir          = str(out_dir),
        write_csv           = True,
        verbose_opf         = False,
        v_min               = plan.parameters.v_min,
        v_max               = plan.parameters.v_max,
        run_hc              = run_hc,
        run_hc_scenarios    = run_hc_scenarios,
        hc_stress_scenarios = _HC_STRESS_SCENARIOS if run_hc_scenarios else None,
        profile_factory     = profile_factory if run_hc_scenarios else None,
        publish_fn          = publish_fn,
    )


# ===========================================================================
# Runtime parameter channels (confirmed D1/D2 + Q(V) overrides)
# ===========================================================================

def apply_qv_overrides(plan: RunPlan) -> None:
    """
    Apply per-run Q(V) characteristic overrides (ParameterConfig.q_ratio,
    u1_pu..u4_pu) via volt_var_controller.set_qv_parameters().

    One call updates every consumer consistently: the dry-run
    QVCharacteristic, the coordinator/dynamics q_max sizing (both read the
    module attribute at construction time), and the CFG: message pushed to
    the Arduino at configure() — so the Python/firmware mismatch the
    Knowledge Base §9.1 invariant warns about cannot occur. Validation is
    the firmware's own ERR:CFG_INVALID matrix; an invalid combination is a
    ConfigError BEFORE the run, not a silent mis-control during it.
    """
    p = plan.parameters
    if all(v is None for v in (p.q_ratio, p.u1_pu, p.u2_pu, p.u3_pu, p.u4_pu)):
        return

    _ensure_framework_on_path()
    import volt_var_controller as vvc
    try:
        applied = vvc.set_qv_parameters(
            q_ratio=p.q_ratio, u1=p.u1_pu, u2=p.u2_pu, u3=p.u3_pu, u4=p.u4_pu,
        )
    except ValueError as exc:
        raise ConfigError(
            f"Invalid Q(V) characteristic override: {exc} "
            f"Adjust the parameters (0 < q_ratio \u2264 1, strictly "
            f"increasing U1 < U2 < U3 < U4) or leave them unset to use "
            f"the framework defaults."
        ) from exc
    console.print(
        f"[stage]Q(V) characteristic for this run:[/stage] "
        f"q_ratio={applied['q_ratio']:.4f}, "
        f"U1..U4 = {applied['u1']:.4f}/{applied['u2']:.4f}/"
        f"{applied['u3']:.4f}/{applied['u4']:.4f} "
        f"[muted](applied to dry-run curve, coordinator sizing, and the "
        f"Arduino CFG message)[/muted]"
    )


def apply_violation_limits(plan: RunPlan) -> None:
    """
    Apply the RunPlan's thermal/angle/unbalance limits process-wide by
    rebinding violation_detector's module constants (whose
    detect_violations defaults are late-bound for exactly this purpose).

    Fallback semantics (confirmed): an invalid value keeps the framework
    default and prints a warning rather than aborting — the hardcoded
    constants are the fallbacks, user input the override.

    Scope note, stated honestly: these limits take effect everywhere
    detect_violations() is called with defaults — the Scenario 4 HIL loop
    (pre/post-PF reports and the curtailment gate), the coordinator, and
    hosting capacity. Scenarios 1–3 count VOLTAGE violations through their
    own inline vm checks driven by BenchmarkConfig.v_min/v_max (also set
    from this plan), so voltage limits are consistent everywhere; their
    per-record line/trafo loading fields, however, come from
    make_record_from_report and follow these module constants too.
    unbalance_max_percent is applied but currently reaches no result: no
    runner performs a 3-phase power flow (detect_violations_3ph has no
    caller in the benchmark stack).
    """
    _ensure_framework_on_path()
    import violation_detector as vd

    # Start from framework defaults so a prior run's overrides in the same
    # process (notebook / server) cannot leak into this one via the
    # None-keeps-current semantics of set_limit().
    vd.reset_limits()

    p = plan.parameters

    def _warn(msg: str) -> None:
        console.print(f"[warning]{msg}[/warning]")
    
    def _confirm(msg: str) -> None:
        console.print(f"[muted]{msg}[/muted]")
    
    vd.set_limit("LINE_MAX_LOADING",      p.line_max_loading,      0.0, vd.THERMAL_LOADING_PLAUSIBLE_MAX, "%",   warn=_warn, confirm=_confirm)
    vd.set_limit("TRAFO_MAX_LOADING",     p.trafo_max_loading,     0.0, vd.THERMAL_LOADING_PLAUSIBLE_MAX, "%",   warn=_warn, confirm=_confirm)
    vd.set_limit("VA_DIFF_MAX_DEGREE",    p.va_diff_max_degree,    0.0, 180.0,  "deg", warn=_warn, confirm=_confirm)
    vd.set_limit("UNBALANCE_MAX_PERCENT", p.unbalance_max_percent, 0.0, 100.0,  "%",   warn=_warn, confirm=_confirm)
    # V_MIN/V_MAX are ALSO rebound so detect_violations-based checks (the
    # coordinator's resid gate, curtailment, HC) agree with the inline
    # checks driven by BenchmarkConfig.v_min/v_max.
    if p.v_min is not None and p.v_max is not None and p.v_min < p.v_max:
        vd.V_MIN, vd.V_MAX = float(p.v_min), float(p.v_max)


# ===========================================================================
# TASK 2 — network and dataset resolution
# ===========================================================================

def _import_custom_network_fn(path_str: str, function_name: str) -> Callable:
    """
    Import the factory function from a bare custom network file
    (NetworkConfig.source_type == "custom").

    Every failure mode a student can hit is converted into a
    NetworkLoadError whose message names the exact problem — no raw
    ImportError/AttributeError traceback reaches the terminal.
    """
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise NetworkLoadError(
            f"Custom network file not found: {path}. Check the path you "
            f"entered in the wizard (relative paths resolve against the "
            f"directory you launched the CLI from: {Path.cwd()})."
        )
    if path.suffix != ".py":
        raise NetworkLoadError(
            f"Custom network file must be a .py Python file, got: {path}"
        )

    mod_name = f"_hil_custom_net_{path.stem}_{abs(hash(str(path))) & 0xFFFFFF:06x}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise NetworkLoadError(f"Could not build an import spec for {path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(mod_name, None)
        raise NetworkLoadError(
            f"Importing your custom network file {path.name} raised "
            f"{type(exc).__name__}: {exc}. Fix the error inside the file "
            f"and re-run (the file is executed on import)."
        ) from exc

    if not hasattr(module, function_name):
        available = [n for n in dir(module)
                     if not n.startswith("_") and callable(getattr(module, n))]
        raise NetworkLoadError(
            f"Your custom network file {path.name} has no function named "
            f"'{function_name}'. Callables found in the file: {available}. "
            f"Either rename your factory function or enter its actual name "
            f"in the wizard."
        )
    fn = getattr(module, function_name)
    if not callable(fn):
        raise NetworkLoadError(
            f"'{function_name}' in {path.name} exists but is not callable "
            f"({type(fn).__name__})."
        )
    return fn


def _looks_like_pandapower_net(obj) -> bool:
    """Duck-type check: the factory must return a pandapowerNet."""
    return all(hasattr(obj, attr) for attr in ("bus", "line", "load", "sgen"))


def _apply_network_modifications(net, network: "NetworkConfig") -> bool:
    """
    Apply the optional pre-run modifications recorded in NetworkConfig:
    DER injection and switch opening — the CLI equivalents of the manual
    blocks documented in run_benchmark_script.py. Returns True if anything
    changed (the plugin path uses this to trigger a profile rebuild so new
    sgens receive profile columns).
    """
    import pandapower as pp

    changed = False

    for placement in (network.der_placements or []):
        try:
            bus    = int(placement["bus"])
            p_mw   = float(placement["p_mw"])
            sn_mva = float(placement.get("sn_mva", p_mw * 1.1))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(
                f"Invalid DER placement entry {placement!r}: expected "
                f'{{"bus": int, "p_mw": float, "sn_mva": float}} ({exc}).'
            ) from exc
        if bus not in net.bus.index:
            raise ConfigError(
                f"DER placement bus {bus} does not exist in this network "
                f"(valid bus indices: {int(net.bus.index.min())}"
                f"..{int(net.bus.index.max())}). Use the topology plot to "
                f"identify bus indices."
            )
        pp.create_sgen(
            net, bus=bus, p_mw=p_mw, sn_mva=sn_mva,
            name=f"injected_PV_bus{bus}", type="PV", in_service=True,
        )
        console.print(
            f"[muted]Injected PV sgen at bus {bus}: P={p_mw} MW, "
            f"S={sn_mva} MVA[/muted]"
        )
        changed = True

    switches = network.switches_to_flip or []
    if switches:
        missing = [s for s in switches if s not in net.switch.index]
        if missing:
            raise ConfigError(
                f"switches_to_flip references indices not present in this "
                f"network: {missing} (valid: {list(net.switch.index)})"
            )
        net.switch.loc[switches, "closed"] = ~net.switch.loc[switches, "closed"]
        console.print(f"[muted]Flipped switches: {switches}[/muted]")
        changed = True

    return changed


def _resolve_scaling(scaling: dict, element_table) -> "object":
    """
    Turn a bus-keyed scaling dict ({None: global, bus: override}) into a
    per-element factor Series aligned to element_table.index (net.load or
    net.sgen). The None sentinel applies network-wide; per-bus entries
    override it for elements at that bus.
    """
    import pandas as pd
    base = float(scaling.get(None, 1.0)) if scaling else 1.0
    factors = pd.Series(base, index=element_table.index, dtype=float)
    for bus, factor in (scaling or {}).items():
        if bus is None:
            continue
        mask = element_table["bus"] == int(bus)
        factors[mask] = float(factor)
    return factors


def _apply_scaling(net, profiles: dict, plan: RunPlan) -> None:
    """
    Apply ParameterConfig.der_scaling / load_scaling (confirmed D1).

    Both the net tables AND the matching profile columns are scaled —
    the pattern documented in run_benchmark_script.py's load-scaling
    block ("add this line alongside the net.load scaling to scale the
    load profile too"). Profiles columns are ELEMENT indices, so the
    bus-keyed dict is first resolved to per-element factors.

    DER scaling scales p_mw AND sn_mva so the inverter Q ceiling
    (Q_RATIO x installed capacity) scales with the plant — scaling only
    p_mw would silently oversize every inverter.
    """
    p = plan.parameters

    def _is_noop(d):
        return not d or all(
            (k is None and float(v) == 1.0) or (k is not None and float(v) == 1.0)
            for k, v in d.items()
        )

    if not _is_noop(p.load_scaling):
        try:
            factors = _resolve_scaling(p.load_scaling, net.load)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Invalid load_scaling dict {p.load_scaling!r}: {exc}. "
                f"Expected {{None: factor}} and/or {{bus_index: factor}}."
            ) from exc
        net.load["p_mw"]   = net.load["p_mw"]   * factors
        net.load["q_mvar"] = net.load["q_mvar"] * factors
        load_df = profiles.get("load")
        if load_df is not None and not load_df.empty:
            common = [c for c in load_df.columns if c in factors.index]
            load_df[common] = load_df[common].mul(factors[common], axis=1)
        console.print(f"[muted]Load scaling applied: {p.load_scaling}[/muted]")

    if not _is_noop(p.der_scaling):
        try:
            factors = _resolve_scaling(p.der_scaling, net.sgen)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Invalid der_scaling dict {p.der_scaling!r}: {exc}. "
                f"Expected {{None: factor}} and/or {{bus_index: factor}}."
            ) from exc
        net.sgen["p_mw"] = net.sgen["p_mw"] * factors
        if "sn_mva" in net.sgen.columns:
            net.sgen["sn_mva"] = net.sgen["sn_mva"] * factors
        for key in ("pv", "wind"):
            df = profiles.get(key)
            if df is not None and not df.empty:
                common = [c for c in df.columns if c in factors.index]
                df[common] = df[common].mul(factors[common], axis=1)
        console.print(
            f"[muted]DER scaling verified: p_mw range="
            f"{net.sgen['p_mw'].min():.4f}-{net.sgen['p_mw'].max():.4f}, "
            f"sn_mva range={net.sgen['sn_mva'].min():.4f}-{net.sgen['sn_mva'].max():.4f}[/muted]"
        )


def _apply_time_window(profiles: dict, plan: RunPlan) -> dict:
    """
    FLAGGED IMPROVEMENT (accept or strip): slice the built annual profiles
    to plan.time_period/time_index via scenario_result.slice_profiles(),
    the same mechanism run_benchmark_script.py documents for fast
    iteration. Full annual (time_period None/"full") passes through
    untouched. Publishing uses the SLICED profiles so dashboard JSON and
    the benchmark cover the same window.
    """
    period = plan.time_period
    if not period or period == "full":
        return profiles
    from scenario_result import slice_profiles
    index = int(plan.time_index or 1)
    try:
        sliced = slice_profiles(profiles, period=period, index=index)
    except (KeyError, ValueError, IndexError) as exc:
        raise ConfigError(
            f"Invalid time window period={period!r}, index={index}: {exc}. "
            f"Valid periods: day (1-366), week (1-53), month (1-12), or "
            f"full."
        ) from exc
    console.print(
        f"[muted]Time window: {period} {index} — "
        f"{len(sliced['times'])} of {len(profiles['times'])} "
        f"timesteps[/muted]"
    )
    return sliced


def _check_timestep_resolution(profiles: dict, plan: RunPlan) -> None:
    """
    Confirmed D3: the plan's timestep_resolution is checked against the
    ACTUAL resolution of the built profiles and mismatches WARN rather
    than fail — future data sources (1-min weather stations, hourly ERA5)
    must not be blocked by a hard 15/10-minute whitelist. The actual
    resolution is what the simulation runs at; the plan value is display
    metadata.
    """
    times = profiles.get("times")
    if times is None or len(times) < 2:
        return
    actual_min = (times[1] - times[0]).total_seconds() / 60.0
    if abs(actual_min - float(plan.parameters.timestep_resolution)) > 1e-6:
        console.print(
            f"[warning]timestep_resolution in the plan is "
            f"{plan.parameters.timestep_resolution} min but the built "
            f"profiles are at {actual_min:g} min — the run uses the "
            f"profiles' actual resolution ({actual_min:g} min). Dynamics "
            f"(PT1/ramp) are parameterised from the actual dt.[/warning]"
        )


def build_net_and_profiles(
        plan: RunPlan,
) -> Tuple[object, dict, str, Optional[Callable]]:
    """
    Resolve plan.network + plan.dataset into (net, profiles, network_id,
    profile_factory).

    profile_factory is a callable(net) -> profiles dict rebuilt with the
    SAME strategy as the outer run — required by the HC-stressed
    re-benchmark (BenchmarkConfig.profile_factory) so new HC sgens get
    profile columns. None when the dataset source cannot rebuild (never
    the case for the four supported paths, but kept explicit).

    Order of operations (deliberate):
      1. load net           2. network modifications (DER injection,
      switch opening — BEFORE profiles so injected sgens get columns)
      3. build profiles     4. scaling (net tables + profile columns
      together)             5. resolution check
    """
    _ensure_framework_on_path()
    net_cfg, ds_cfg = plan.network, plan.dataset
    sb_code: Optional[str] = None
    profile_factory: Optional[Callable] = None

    # ------------------------------------------------------------------ #
    # 1. Network                                                          #
    # ------------------------------------------------------------------ #
    if net_cfg.source_type == "preset":
        loaders = _preset_loaders()
        if net_cfg.preset_name not in loaders:
            raise NetworkLoadError(
                f"Preset '{net_cfg.preset_name}' has no loader. Known "
                f"presets: {sorted(loaders)}. (network_catalogue.py and "
                f"the executor's loader map must list the same names.)"
            )
        network_id = net_cfg.preset_name
        sb_code    = _PRESET_SIMBENCH_CODES.get(net_cfg.preset_name)
        try:
            net = loaders[net_cfg.preset_name]()
        except Exception as exc:
            raise NetworkLoadError(
                f"Loading preset '{net_cfg.preset_name}' failed with "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    elif net_cfg.source_type == "simbench_code":
        import simbench as sb
        network_id = sb_code = net_cfg.simbench_code
        if not sb_code:
            raise ConfigError(
                "network.source_type is 'simbench_code' but no "
                "simbench_code is recorded in the plan."
            )
        try:
            net = sb.get_simbench_net(sb_code)
        except Exception as exc:
            raise NetworkLoadError(
                f"SimBench could not load code '{sb_code}' "
                f"({type(exc).__name__}: {exc}). Check the code against "
                f"the SimBench documentation — the wizard assembles it "
                f"from menu choices but not every combination exists."
            ) from exc

    elif net_cfg.source_type == "custom":
        fn = _import_custom_network_fn(
            net_cfg.custom_path, net_cfg.custom_function_name,
        )
        try:
            net = fn()
        except Exception as exc:
            raise NetworkLoadError(
                f"Your network factory "
                f"{net_cfg.custom_function_name}() raised "
                f"{type(exc).__name__}: {exc}. This is a bug inside your "
                f"custom network file, not in the framework."
            ) from exc
        if not _looks_like_pandapower_net(net):
            raise NetworkLoadError(
                f"{net_cfg.custom_function_name}() returned "
                f"{type(net).__name__}, not a pandapower network (missing "
                f"bus/line/load/sgen tables). The factory must return the "
                f"net object itself."
            )
        network_id = Path(net_cfg.custom_path).stem

    elif net_cfg.source_type == "plugin":
        from network_plugin import (
            load_network_from_yaml, validate_network_plugin,
            make_profile_factory,
        )
        from rich.prompt import Confirm
        try:
            net, profiles = load_network_from_yaml(net_cfg.plugin_path)
        except (FileNotFoundError, ValueError, ImportError,
                AttributeError, TypeError) as exc:
            raise PluginError(
                f"Network plugin '{net_cfg.plugin_path}' failed to load: "
                f"{exc}",
                context="Network plugin loading",
            ) from exc
        network_id = profiles["plugin_meta"]["name"]
        profile_factory = make_profile_factory(net_cfg.plugin_path)

        warnings = validate_network_plugin(net, profiles)
        if warnings:
            console.print("[warning]Network plugin compatibility warnings:[/warning]")
            for w in warnings:
                console.print(f"  [warning]- {w}[/warning]")
            if not Confirm.ask(
                "Proceed despite these warnings?", default=False,
                console=console,
            ):
                raise PluginError(
                    "Run cancelled after network-plugin compatibility "
                    "warnings.", context="Network plugin validation",
                )

        # Modifications before (re)building profiles: if DERs were injected
        # the loaded profiles lack their columns — rebuild with the SAME
        # YAML strategy via the factory.
        if _apply_network_modifications(net, net_cfg):
            console.print(
                "[muted]Network modified — rebuilding plugin profiles so "
                "new elements receive profile columns.[/muted]"
            )
            try:
                profiles = profile_factory(net)
            except Exception as exc:
                raise DatasetError(
                    f"Rebuilding plugin profiles after network "
                    f"modification failed ({type(exc).__name__}: {exc})."
                ) from exc

        _apply_scaling(net, profiles, plan)
        profiles = _apply_time_window(profiles, plan)
        _check_timestep_resolution(profiles, plan)
        _validate_focus_buses(net, plan)
        return net, profiles, network_id, profile_factory

    else:
        raise ConfigError(
            f"Unknown network.source_type {net_cfg.source_type!r}. Valid: "
            f"preset, simbench_code, custom, plugin."
        )

    # ------------------------------------------------------------------ #
    # 2. Modifications (non-plugin paths) — BEFORE profile building       #
    # ------------------------------------------------------------------ #
    _apply_network_modifications(net, net_cfg)

    # ------------------------------------------------------------------ #
    # 3. Profiles (non-plugin paths)                                      #
    # ------------------------------------------------------------------ #
    from profile_builder import build_annual_profiles

    if ds_cfg.source_type == "simbench_native":
        if sb_code is None:
            raise ConfigError(
                "Dataset source 'simbench_native' is only valid for "
                "SimBench networks (a SimBench preset or an assembled "
                "SimBench code) — the selected network has no native "
                "profiles. Choose the DWD or custom dataset source for "
                "this network."
            )
        builder_kwargs = dict(net_name=network_id, simbench_code=sb_code)

    elif ds_cfg.source_type == "dwd":
        data_dir = Path(ds_cfg.data_dir or "data/dwd").expanduser()
        _validate_dwd_dir(data_dir, ds_cfg.station_id)
        builder_kwargs = dict(
            net_name=network_id, data_dir=str(data_dir),
            simbench_code=sb_code,
        )

    elif ds_cfg.source_type == "custom":
        # Confirmed interpretation: custom_path IS the data directory
        # handed to profile_builder (build_annual_profiles has no
        # direct-load path), with the user's file_map/col_map overrides.
        if not ds_cfg.custom_path:
            raise DatasetError(
                "Dataset source 'custom' needs custom_path (the directory "
                "containing your profile source CSVs) — none was recorded."
            )
        data_dir = Path(ds_cfg.custom_path).expanduser()
        if not data_dir.is_dir():
            raise DatasetError(
                f"Custom dataset directory not found: {data_dir}. "
                f"custom_path must be the DIRECTORY containing the profile "
                f"CSVs (it is passed to profile_builder as data_dir), with "
                f"file_map naming the files inside it."
            )
        builder_kwargs = dict(
            net_name=network_id, data_dir=str(data_dir),
            simbench_code=sb_code,
            file_map=ds_cfg.file_map, col_map=ds_cfg.col_map,
        )

    else:
        raise ConfigError(
            f"Unknown dataset.source_type {ds_cfg.source_type!r} for a "
            f"non-plugin network. Valid: simbench_native, dwd, custom."
        )

    try:
        profiles = build_annual_profiles(net, **builder_kwargs)
    except Exception as exc:
        raise DatasetError(
            f"Profile building failed ({type(exc).__name__}: {exc}). "
            f"Dataset source: {ds_cfg.source_type}, network: {network_id}."
        ) from exc

    # HC-stressed re-benchmark factory: same strategy, stressed net.
    def profile_factory(net_hc, _kw=dict(builder_kwargs)):  # noqa: F811
        kw = dict(_kw)
        kw["net_name"] = str(kw["net_name"]) + "_hc_stressed"
        return _apply_time_window(build_annual_profiles(net_hc, **kw), plan)

    # ------------------------------------------------------------------ #
    # 4./5. Scaling, resolution check, focus-bus validation               #
    # ------------------------------------------------------------------ #
    _apply_scaling(net, profiles, plan)
    profiles = _apply_time_window(profiles, plan)
    _check_timestep_resolution(profiles, plan)
    _validate_focus_buses(net, plan)

    return net, profiles, network_id, profile_factory


# ===========================================================================
# TASK 3 — runtime validation
# ===========================================================================
# Cross-field checks the wizard does NOT perform (verified by reading
# wizard.py: every prompt validates its own field in isolation):
#   - v_min < v_max
#   - dataset <-> network compatibility (simbench_native on non-SimBench)
#   - study <-> network compatibility (OPF guard, in build_benchmark_config)
#   - timestep_resolution vs the profiles actually built (warn-only, D3)
#   - existence/importability of custom network files and plugin YAMLs
#   - focus_buses membership in net.bus.index
#   - serial-port availability for hardware runs
# Field-level wizard validation (menu ranges, dict-literal syntax) is NOT
# repeated here.

def _validate_focus_buses(net, plan: RunPlan) -> None:
    """focus_buses must exist in the loaded network (post-load check)."""
    if not plan.focus_buses:
        return
    missing = [b for b in plan.focus_buses if b not in net.bus.index]
    if missing:
        raise ConfigError(
            f"focus_buses {missing} do not exist in this network. Valid "
            f"bus indices: {int(net.bus.index.min())}"
            f"..{int(net.bus.index.max())}."
        )


def _validate_dwd_dir(data_dir: Path, station_id: Optional[str]) -> None:
    """
    DWD dataset pre-check: the directory must exist, and if a station_id
    is set, at least one CSV mentioning it must be present somewhere under
    it (profile_builder matches files by parameter code + glob; the
    station appears in the DWD CDC filenames, e.g.
    data_OBS_DEU_PT10M_RADG_691.csv).
    """
    if not data_dir.is_dir():
        raise DatasetError(
            f"DWD data directory not found: {data_dir.resolve()}. Create "
            f"the directory (expected layout: <dir>/PV, <dir>/Wind, "
            f"<dir>/Temperature with DWD CDC CSVs) or point the wizard at "
            f"the correct location."
        )
    if station_id:
        hits = list(data_dir.rglob(f"*{station_id}*.csv"))
        if not hits:
            raise DatasetError(
                f"No CSV file mentioning station '{station_id}' found "
                f"under {data_dir.resolve()}. DWD CDC filenames carry the "
                f"station ID (e.g. data_OBS_DEU_PT10M_RADG_"
                f"{station_id}.csv) — check the station ID and that the "
                f"files were copied into the directory."
            )


def validate_plan(plan: RunPlan) -> None:
    """
    Pre-load validation: everything checkable from the RunPlan alone,
    before any network/profile work is spent. Raises the taxonomy's
    typed errors; execute() maps them to exit codes and
    print_error_message().
    """
    p = plan.parameters

    # -- cross-field numeric sanity (the wizard asks these independently) --
    if not (p.v_min < p.v_max):
        raise ConfigError(
            f"v_min must be strictly less than v_max, got v_min={p.v_min}, "
            f"v_max={p.v_max}. Re-run the wizard (or edit the preset JSON) "
            f"with a valid voltage band, e.g. 0.95\u20131.05 pu."
        )
    if not (0.5 <= p.v_min and p.v_max <= 1.5):
        raise ConfigError(
            f"Voltage band [{p.v_min}, {p.v_max}] pu is outside the "
            f"plausible planning range [0.5, 1.5] pu — likely a typo."
        )

    # -- dataset <-> network compatibility --
    if (plan.dataset.source_type == "simbench_native"
            and plan.network.source_type not in ("simbench_code",)
            and plan.network.preset_name not in _PRESET_SIMBENCH_CODES):
        selected = (plan.network.preset_name or plan.network.custom_path
                    or plan.network.plugin_path or plan.network.source_type)
        raise ConfigError(
            "Dataset 'simbench_native' requires a SimBench network (the "
            "SimBench preset or an assembled SimBench code); the selected "
            f"network ({selected}) has "
            "no native profiles. Choose the DWD or custom dataset instead."
        )

    # -- file existence for paths the wizard accepted as plain text --
    if plan.network.source_type == "custom":
        cp = Path(plan.network.custom_path or "").expanduser()
        if not cp.is_file():
            raise NetworkLoadError(
                f"Custom network file not found: {cp.resolve()} (entered "
                f"in the wizard). Relative paths resolve against the "
                f"launch directory: {Path.cwd()}."
            )
    if plan.network.source_type == "plugin":
        yp = Path(plan.network.plugin_path or "").expanduser()
        if not yp.is_file():
            raise PluginError(
                f"Network plugin YAML not found: {yp.resolve()}.",
                context="Network plugin validation",
            )
    if plan.controller_plugin_path:
        cp = Path(plan.controller_plugin_path).expanduser()
        if not cp.is_file():
            raise PluginError(
                f"Controller plugin YAML not found: {cp.resolve()}.",
                context="Controller plugin validation",
            )

    # -- hardware plumbing --
    if plan.hardware and not plan.port:
        raise HardwareError(
            "Hardware mode is enabled but no serial port is set. Enter a "
            "port in the wizard (e.g. /dev/ttyACM0 on the RPi, COM3 on "
            "Windows) or switch to dry run."
        )


def _study_uses_hardware(plan: RunPlan) -> bool:
    """True when the run would actually open the serial port."""
    if not plan.hardware:
        return False
    scenario_set = set(STUDY_SCENARIOS.get(plan.study, []))
    hw_in_scenarios = bool(scenario_set & {4, 5})
    # HC-stressed re-benchmark includes scenarios 4/5; controller plugins
    # may be hardware-backed.
    if plan.study == "hosting_capacity" and plan.hc_stressed:
        hw_in_scenarios = True
    return hw_in_scenarios or bool(plan.controller_plugin_path)


def check_hardware_port(plan: RunPlan) -> None:
    """
    Fail BEFORE the multi-hour annual run, not at the first timestep's
    serial timeout.

    Verified behaviour of ArduinoSerialInterface (volt_var_controller.py):
    open() raises pyserial's raw serial.SerialException on a missing or
    permission-locked port and ImportError when pyserial is absent;
    handshake failures surface as SerialConfigError from configure().
    This pre-check therefore (a) confirms pyserial is installed, (b)
    checks the configured port against serial.tools.list_ports, and (c)
    probe-opens it once. The probe costs one extra DTR reset cycle, which
    is harmless — scenario 4's own open() waits the mandatory 2 s
    ARDUINO_RESET_DELAY_S regardless.
    """
    if not _study_uses_hardware(plan):
        return

    try:
        import serial
        from serial.tools import list_ports
    except ImportError as exc:
        raise HardwareError(
            "pyserial is not installed but hardware mode was requested. "
            "Install it with:  pip install pyserial  (inside hil_env on "
            "the RPi)."
        ) from exc

    available = {p.device for p in list_ports.comports()}
    if plan.port not in available:
        raise HardwareError(
            f"Serial port {plan.port} not found. Ports currently present: "
            f"{sorted(available) or 'none'}. Check the USB cable, that the "
            f"Arduino is powered, and the port name (RPi: /dev/ttyACM0 for "
            f"the Uno R3's USB CDC, /dev/ttyUSB0 for CH340 clones)."
        )

    try:
        probe = serial.Serial(port=plan.port, baudrate=115200, timeout=0.5)
        probe.close()
    except serial.SerialException as exc:
        raise HardwareError(
            f"Serial port {plan.port} exists but could not be opened: "
            f"{exc}. On Linux this is usually a permissions problem — add "
            f"your user to the dialout group (sudo usermod -a -G dialout "
            f"$USER, then log out/in) — or another process holds the port."
        ) from exc

    console.print(
        f"[ok]Hardware pre-check passed:[/ok] [muted]{plan.port} present "
        f"and openable. The Arduino resets on each port open; scenario 4 "
        f"waits the mandatory 2 s before the INIT/CFG/P handshake.[/muted]"
    )


def _confirm_plugin_firmware(plan: RunPlan) -> None:
    """
    Hardware controller-plugin flash failsafe (interactive layer).

    Cascade: (1) load_plugin requires the YAML to NAME the .ino and checks
    it exists; (2) this prompt makes the researcher confirm THAT sketch is
    what is flashed on THAT port — the protocol cannot distinguish
    firmwares that both ACK the handshake; (3) the INIT/CFG/P handshake
    itself fails loudly (SerialConfigError) on a board whose firmware does
    not implement the protocol at all.
    """
    if not (plan.controller_plugin_path and plan.hardware):
        return
    _ensure_framework_on_path()
    from plugin_runner import load_plugin
    from rich.prompt import Confirm

    try:
        cfg = load_plugin(plan.controller_plugin_path)
    except (FileNotFoundError, ValueError, ImportError,
            NotImplementedError) as exc:
        raise PluginError(
            f"Controller plugin '{plan.controller_plugin_path}' failed "
            f"validation: {exc}",
            context="Controller plugin validation",
        ) from exc

    if not cfg["hardware"]:
        console.print(
            "[muted]Controller plugin is software-only (hardware: false) — "
            "it runs as a Python dry-run mirror even in this hardware "
            "run.[/muted]"
        )
        return

    console.print(
        f"[stage]Hardware controller plugin:[/stage] '{cfg['name']}' "
        f"expects its own firmware on the Arduino:\n"
        f"  [current]{cfg['firmware_path']}[/current]\n"
        f"[muted]The board on {plan.port} must be flashed with this sketch "
        f"BEFORE the run. It must implement the standard protocol "
        f"(INIT/CFG/P handshake, V:->Q: exchange, END). The run uses "
        f"whatever firmware is actually on the board — the protocol "
        f"cannot verify the algorithm, only the message format.[/muted]"
    )
    if not Confirm.ask(
        f"Confirm: {Path(str(cfg['firmware_path'])).name} is flashed on "
        f"{plan.port}?",
        default=False, console=console,
    ):
        raise HardwareError(
            "Run cancelled: plugin firmware not confirmed as flashed. "
            "Flash the sketch named in the plugin YAML (Arduino IDE or "
            "arduino-cli) and re-run."
        )


# ===========================================================================
# TASK 4 — execute(plan) -> int
# ===========================================================================

def _build_summary(result, network_id: str, elapsed_s: float,
                   out_dir: Path, written: dict) -> dict:
    """Assemble the dict for helpers.print_summary_table()."""
    summary: dict = {
        "network": network_id,
        "total wall-clock [s]": f"{elapsed_s:.1f}",
    }
    df = result.comparison_df
    for _, row in df.iterrows():
        label  = str(row["scenario_label"])
        status = str(row["status"])
        if status == "ok":
            summary[label] = (
                f"ok | violations={int(row['n_violation_steps'])} "
                f"| max_vm={row['max_vm_pu']:.4f} pu"
            )
        else:
            summary[label] = f"{status.upper()} | {row['error_message']}"
    if result.hc_results:
        hc_b, hc_v = result.hc_results
        summary["hosting capacity (baseline)"] = f"{hc_b.hc_mw:.3f} MW"
        summary["hosting capacity (volt-var)"] = f"{hc_v.hc_mw:.3f} MW"
        summary["hosting capacity gain"] = f"{hc_v.hc_mw - hc_b.hc_mw:+.3f} MW"
    if result.hc_error:
        summary["hosting capacity"] = (
            "FAILED | " + result.hc_error.splitlines()[-1]
        )
    if result.hc_benchmark is not None:
        summary["HC-stressed re-benchmark"] = (
            f"done ({result.hc_benchmark.network_id})"
        )
    if result.csv_path:
        summary["comparison csv"] = str(result.csv_path)
    for label, path in (written or {}).items():
        summary[f"published {label}"] = str(path)
    summary["output directory"] = str(out_dir)
    return summary


def _save_run_plan_copy(plan: RunPlan, out_dir: Path) -> None:
    """
    Persist the exact RunPlan into the run's output directory so any run
    is reproducible from its own artefacts (independent of the optional
    preset save in __main__.py). Failures here never abort a run.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_plan.json").write_text(
            json.dumps(plan.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    except OSError as exc:
        console.print(
            f"[warning]Could not save run_plan.json to {out_dir}: "
            f"{exc}[/warning]"
        )


def execute(plan: RunPlan) -> int:
    """
    Public entry point: __main__.py calls sys.exit(execute(plan)).

    Phases, each with its own failure identity:
      configuration -> ExitCode.CONFIG_ERROR
      network       -> NETWORK_LOAD_ERROR   dataset -> DATASET_ERROR
      plugins       -> PLUGIN_ERROR         hardware -> HARDWARE_ERROR
      simulation    -> SIMULATION_ERROR     publishing -> PUBLISH_ERROR

    The run_benchmark()/register_and_run() call is wrapped in a broad
    except Exception as the last-resort net for crashes deep inside a
    student's custom controller/network code or pandapower itself; its
    message is worded so "your plugin code crashed" is immediately
    distinguishable from the typed pre-run validation failures above.
    """
    t_start = time.perf_counter()
    out_dir = Path(plan.output_dir or "runs") / plan.run_id
    _configure_cli_logging(log_path=out_dir / "session.log")
    out_dir = Path(plan.output_dir or "runs") / plan.run_id

    # ---------------------------------------------------------------- #
    # Phase 0 — configuration-level validation and runtime channels     #
    # ---------------------------------------------------------------- #
    try:
        print_section_header(plan, "Validating configuration")
        _ensure_framework_on_path()
        validate_plan(plan)
        apply_qv_overrides(plan)
        apply_violation_limits(plan)
        check_hardware_port(plan)
        _confirm_plugin_firmware(plan)
    except KeyboardInterrupt:
        console.print("\n[warning]Interrupted.[/warning]")
        return int(ExitCode.INTERRUPTED)
    except ExecutorError as exc:
        print_error_message(exc, context=exc.context)
        return int(exc.exit_code)

    # ---------------------------------------------------------------- #
    # Phase 1 — network + profiles                                      #
    # ---------------------------------------------------------------- #
    try:
        print_section_header(plan, "Loading network and building profiles")
        net, profiles, network_id, profile_factory = \
            build_net_and_profiles(plan)
        console.print(
            f"[ok]Network ready:[/ok] {network_id} "
            f"[muted]({len(net.bus)} buses, {len(net.sgen)} sgens, "
            f"{len(net.load)} loads | {len(profiles['times'])} "
            f"timesteps)[/muted]"
        )
    except KeyboardInterrupt:
        console.print("\n[warning]Interrupted.[/warning]")
        return int(ExitCode.INTERRUPTED)
    except ExecutorError as exc:
        print_error_message(exc, context=exc.context)
        return int(exc.exit_code)
    except Exception as exc:   # unexpected loader/builder crash
        print_error_message(
            exc,
            context="Unexpected failure while loading the network or "
                    "building profiles (see the type/message above; this "
                    "is a crash, not a configuration problem)",
        )
        return int(ExitCode.NETWORK_LOAD_ERROR)

    # ---------------------------------------------------------------- #
    # Phase 2 — benchmark configuration + live publisher                 #
    # ---------------------------------------------------------------- #
    try:
        from publisher import (
            PublishHandle,
            publish_topology_and_profiles,
            publish_hc_and_comparison,
        )
        pub_dir    = out_dir / "publisher" / network_id
        hc_pub_dir = out_dir / "publisher" / f"{network_id}_hc_stressed"
        handle = PublishHandle(
            output_dir     = str(pub_dir),
            # stream_every_k is the ONE RunPlan streaming field with a real
            # consumer: the live-frame cadence.
            update_every_k = int(plan.stream_every_k or 4),
        )
        hc_handle = PublishHandle(
            output_dir     = str(hc_pub_dir),
            update_every_k = int(plan.stream_every_k or 4),
        )
        config = build_benchmark_config(
            plan, profile_factory=profile_factory, publish_fn=handle,
        )
        config.hc_publish_fn = hc_handle   # restores HC-stressed dashboard output
        _save_run_plan_copy(plan, out_dir)
    except KeyboardInterrupt:
        console.print("\n[warning]Interrupted.[/warning]")
        return int(ExitCode.INTERRUPTED)
    except ExecutorError as exc:
        print_error_message(exc, context=exc.context)
        return int(exc.exit_code)

    # ---------------------------------------------------------------- #
    # Publish topology + profiles once, at network-load time.           #
    # Crash-resilient: a later scenario failure no longer loses these.  #
    # Non-fatal — a publish failure must not abort a not-yet-started run.#
    # ---------------------------------------------------------------- #
    try:
        publish_topology_and_profiles(
            net, profiles, output_dir=str(pub_dir), network_id=network_id,
        )
    except Exception as exc:
        console.print(f"[warning]Topology/profiles publish failed: {exc}[/warning]")

    # ---------------------------------------------------------------- #
    # Phase 3 — the run itself                                          #
    # ---------------------------------------------------------------- #
    try:
        print_section_header(plan, "Running benchmark")
        from benchmark_runner import run_benchmark

        if plan.controller_plugin_path:
            from plugin_runner import register_and_run
            _, result = register_and_run(
                plan.controller_plugin_path,
                net, profiles,
                network_id       = network_id,
                benchmark_config = config,
                return_benchmark = True,
                port             = plan.port if plan.hardware else None,
            )
        else:
            result = run_benchmark(
                net, profiles, network_id=network_id, config=config,
            )
    except KeyboardInterrupt:
        console.print("\n[warning]Interrupted.[/warning]")
        return int(ExitCode.INTERRUPTED)
    except (FileNotFoundError, ValueError, ImportError, AttributeError,
            NotImplementedError) as exc:
        # register_and_run's load_plugin/_import_controller_fn raise these
        # BEFORE any scenario starts — plugin definition problems, not
        # simulation crashes.
        print_error_message(
            exc,
            context="Controller plugin failed to load — fix the plugin "
                    "YAML / module (this happened before any simulation "
                    "started)",
        )
        return int(ExitCode.PLUGIN_ERROR)
    except RuntimeError as exc:
        # register_and_run raises RuntimeError when the CUSTOM scenario
        # failed inside the isolated runner.
        print_error_message(
            exc,
            context="YOUR CUSTOM CONTROLLER CRASHED inside the simulation "
                    "loop. The configuration and network were valid; the "
                    "traceback above comes from your plugin code",
        )
        return int(ExitCode.SIMULATION_ERROR)
    except Exception as exc:
        # Last-resort net: crashes deep inside pandapower / a custom
        # network's data / the framework. Deliberately distinguishable
        # from every typed validation failure above.
        print_error_message(
            exc,
            context="THE SIMULATION CRASHED after validation passed. This "
                    "is not a configuration problem: the most common "
                    "causes are a custom network/plugin whose data breaks "
                    "a pandapower assumption, or (hardware runs) the "
                    "Arduino handshake failing mid-run",
        )
        return int(ExitCode.SIMULATION_ERROR)

    # ---------------------------------------------------------------- #
    # Phase 4 — publish + summary                                       #
    # ---------------------------------------------------------------- #
    written: dict = {}
    exit_code = ExitCode.OK
    try:
        print_section_header(plan, "Publishing results")
        written = publish_hc_and_comparison(result, output_dir=str(pub_dir))
        if result.hc_benchmark is not None and result.net_hc is not None:
            written_hc = publish_hc_and_comparison(
                result.hc_benchmark, output_dir=str(hc_pub_dir),
            )
            written.update(
                {f"hc_stressed/{k}": v for k, v in written_hc.items()}
            )
    except KeyboardInterrupt:
        console.print("\n[warning]Interrupted during publishing.[/warning]")
        exit_code = ExitCode.INTERRUPTED
    except Exception as exc:
        print_error_message(
            exc,
            context="Publishing failed — THE SIMULATION ITSELF COMPLETED; "
                    "the comparison CSV (path in the summary below) is "
                    "intact, only the dashboard JSON export failed",
        )
        exit_code = ExitCode.PUBLISH_ERROR

    elapsed = time.perf_counter() - t_start
    print_summary_table(
        _build_summary(result, network_id, elapsed, out_dir, written)
    )

    # A run where scenarios failed still produced a comparison table (the
    # framework isolates failures); surface that in the exit code so
    # scripts notice, while keeping the table on screen.
    real_failures = {
        n: tb for n, tb in result.errors.items() if tb != "skipped"
    }
    if exit_code == ExitCode.OK and real_failures:
        console.print(
            f"[warning]{len(real_failures)} scenario(s) failed inside the "
            f"run (see the summary rows above) — exit code "
            f"{int(ExitCode.SIMULATION_ERROR)}.[/warning]"
        )
        exit_code = ExitCode.SIMULATION_ERROR

    return int(exit_code)
