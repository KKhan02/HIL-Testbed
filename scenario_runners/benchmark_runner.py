"""
benchmark_runner.py
===================
Orchestrates all five HIL comparison scenario runners on a single pandapower
network and produces a flat comparison table.

Responsibilities
----------------
- Run each scenario runner on an isolated deep-copy of the network.
- Collect ScenarioResult objects and build a comparison DataFrame.
- Export a timestamped CSV (optional) and print a Rich console summary.
- Isolate individual runner failures so one failed scenario does not abort
  the full benchmark.

This module is NOT responsible for network construction or profile building.
The caller (CLI, Flask, or script) provides a pre-built network and profiles
dict — consistent with how all five individual scenario runners work.

Usage
-----
    import simbench as sb
    from profile_builder import build_profiles
    from benchmark_runner import BenchmarkConfig, run_benchmark

    net      = sb.get_simbench_net("1-MV-rural--2-sw")
    profiles = build_profiles(net, ...)

    config = BenchmarkConfig(dry_run=True, write_csv=True)
    result = run_benchmark(net, profiles, network_id="1-MV-rural--2-sw",
                           config=config)

    print(result.comparison_df)
    print(result.csv_path)
    if result.hc_results:
        print(result.hc_results[0].summary_dict())  # baseline HC
        print(result.hc_results[1].summary_dict())  # HC with Volt-Var

Net isolation
-------------
The original net is never modified.  Each scenario runner receives a
copy.deepcopy(net).  This is mandatory because:
  - scenario_3_svc creates and then removes a sgen element.
  - scenario_5_opf calls create_continuous_bus_index() which renumbers all
    bus indices in place.
  - scenario_2_oltc modifies net.trafo.tap_pos.
Without per-scenario isolation, a later scenario would operate on a
structurally different network, producing silent index-alignment errors.
The deepcopy call is placed inside the try block so a copy failure produces
a failed-scenario row rather than aborting the whole benchmark.

Profile sharing
---------------
Profiles are NOT deep-copied.  Every runner calls adapt_profiles() internally,
which does its own .copy() on every DataFrame it uses.  Sharing the profiles
dict across runners is therefore safe and avoids five redundant full-year
copies in memory.

Failed scenario handling
------------------------
A runner that raises any Exception does not abort the benchmark.  The error
is caught, logged at ERROR level with the full traceback, and stored in
BenchmarkResult.errors[n].  The comparison DataFrame row for that scenario has
status="failed", NaN for all numeric fields, and the last traceback line in
error_message.  NaN (not 0) is used deliberately so the consumer can
distinguish "zero violations (clean run)" from "no data (runner failed)".

Rich theme
----------
Console output uses the shared console instance and CLI_THEME from
_console.py, so all terminal styling (colours, voltage highlighting) is
consistent with the rest of the CLI.  If Rich or _console.py is unavailable,
_print_summary_plain() is used as a fallback.
"""

from __future__ import annotations

import copy
import logging
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union
import json

import numpy as np
import pandas as pd

try:
    from _console import console as _console, CLI_THEME   # noqa: F401
    from rich.table import Table as RichTable
    from rich import box as rich_box
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

from scenario_1_baseline import run_scenario_1
from scenario_2_oltc     import run_scenario_2
from scenario_3_svc      import run_scenario_3
from scenario_4_volt_var import run_scenario_4
from scenario_5_opf      import run_scenario_5
from scenario_result     import ScenarioResult
from violation_detector  import V_MIN, V_MAX
from hosting_capacity import run_baseline_hc, run_hc_with_volt_var, HCResult
from publisher import publish_scenario_result, publish_topology_and_profiles


logger = logging.getLogger(__name__)


# ===========================================================================
# ScenarioSpec
# ===========================================================================

@dataclass(frozen=True)
class ScenarioSpec:
    """
    Immutable descriptor for one benchmark scenario.

    Attributes
    ----------
    num : int
        Canonical scenario number (1-5).
    scenario_id : str
        Machine-readable identifier.  Must match the string passed to
        ScenarioResult.from_records() inside the runner so that
        result.summary_dict()["scenario_id"] is always consistent with
        the registry.
    label : str
        Human-readable display name used in the Rich summary table.
    runner : Callable[..., ScenarioResult]
        The run_scenario_N() function for this scenario.
    supports_hardware : bool
        True for Scenario 4 only.  When True, _build_kwargs() injects
        dry_run and port into the runner call.
    supports_opf_verbose : bool
        True for Scenario 5 only.  When True, _build_kwargs() injects
        verbose_opf into the runner call.
    """
    num:                  int
    scenario_id:          str
    label:                str
    runner:               Callable[..., ScenarioResult]
    supports_hardware:    bool = False
    supports_opf_verbose: bool = False
    supports_coordination: bool = False # NEW
    supports_lv:          bool = True   # False for scenarios that require pre-existing DERs

SCENARIO_REGISTRY = {
    1: ScenarioSpec(num=1, scenario_id="baseline",  label="Baseline (no control)",        runner=run_scenario_1),
    2: ScenarioSpec(num=2, scenario_id="oltc",      label="OLTC-only",                    runner=run_scenario_2),
    3: ScenarioSpec(num=3, scenario_id="svc",       label="SVC",                          runner=run_scenario_3,
                    supports_lv=False),
    4: ScenarioSpec(num=4, scenario_id="volt_var_local", label="Volt-Var HIL (local Q(V))", runner=run_scenario_4,
                    supports_hardware=True, supports_coordination=False, supports_lv=False),  # 4A
    5: ScenarioSpec(num=5, scenario_id="volt_var_coord", label="Volt-Var HIL (+ coord)",    runner=run_scenario_4,
                    supports_hardware=True, supports_coordination=True,  supports_lv=False),  # 4B
    6: ScenarioSpec(num=6, scenario_id="opf",       label="OPF (theoretical bound)",       runner=run_scenario_5,
                    supports_opf_verbose=True, supports_lv=False),
}


# ===========================================================================
# BenchmarkConfig
# ===========================================================================

@dataclass(kw_only=True)
class BenchmarkConfig:
    """
    Configuration for a full benchmark run.

    All parameters are keyword-only (kw_only=True) to prevent accidental
    positional-argument errors during construction.

    Attributes
    ----------
    scenarios : list[int]
        Subset of [1, 2, 3, 4, 5] to run.  Always executed in ascending
        order regardless of the order passed here.  Default: all five.
    dry_run : bool
        If True, Scenario 4 skips Arduino serial communication and computes
        Q locally via the pure-Python QVCharacteristic.  Must be True when
        no Arduino is connected.  Default: True.
    port : str or None
        Serial port for Arduino hardware (e.g. "/dev/ttyACM0" on RPi,
        "COM3" on Windows).  Default: None -- the caller must supply an
        explicit port when hardware is needed.  Ignored when dry_run=True
        or scenario 4 is not in the run set.
    output_dir : str, Path, or None
        Directory for the comparison CSV.  Accepts str or Path so callers
        (CLI, Flask) do not need to wrap strings explicitly.
        None -> current working directory.
    write_csv : bool
        If True (default), write the comparison CSV after all scenarios
        complete.  Set False in tests or Flask in-memory use to avoid
        filesystem side-effects.
    verbose_opf : bool
        Forwarded to run_scenario_5() as verbose_opf.  Default: False.
    v_min : float
        Lower voltage planning limit (pu) forwarded to all five runners.
        Default: V_MIN (0.95) from violation_detector.
    v_max : float
        Upper voltage planning limit (pu) forwarded to all five runners.
        Default: V_MAX (1.05) from violation_detector.
    """
    scenarios:   list[int]               = field(default_factory=lambda: [1, 2, 3, 4, 5])
    dry_run:     bool                    = True
    port:        Optional[str]           = None
    output_dir:  Union[str, Path, None]  = None
    write_csv:   bool                    = True
    enable_checkpointing: bool           = True
    # When True, PublishHandle checkpoints full TimestepRecords (not just the
    # compact dashboard live frame) to checkpoint/<scenario_id>.jsonl at the
    # same cadence as update_every_k, and scenario runners will resume from
    # an existing checkpoint on the same output_dir instead of restarting
    # from t=0. Set False to force a clean run even if a stale checkpoint
    # file exists in output_dir (e.g. after confirming a completed run's
    # results are already published and you want a fresh comparison run).
    live_csv_path: Optional[Path] = None
    # Fixed-name CSV path, rewritten in place at update_every_k cadence with
    # a partial row for the currently-running scenario plus completed rows
    # for finished scenarios. None (default) disables this — only the final
    # timestamped CSV from _write_csv() is written, as today.
    verbose_opf: bool                    = False
    v_min:       float                   = V_MIN
    v_max:       float                   = V_MAX
    run_hc:      bool                    = True
    # If True, run hosting capacity analysis (baseline + volt_var) after
    # the scenario loop.  Set False to skip HC when time is limited.
    run_hc_scenarios: bool               = False
    # If True, run a second recursive benchmark on the HC-stressed network
    # (net at hc_mw capacity) after run_baseline_hc() completes.  Requires
    # profile_factory to be set.  run_hc must also be True.
    profile_factory: Optional[Callable]  = None
    # Callable(net) -> profiles dict.  Required when run_hc_scenarios=True.
    # Called with the HC-stressed net to produce time-series profiles for
    # the recursive benchmark.  Ignored when run_hc_scenarios=False.
    hc_stress_scenarios: Optional[list[int]] = None
    # Scenarios to run in the HC-stressed re-benchmark.
    # None = inherit config.scenarios (same list as the outer run).
    # Set explicitly to decouple outer and stressed scenario lists.
    # e.g. scenarios=[] (HC-only outer) + hc_stress_scenarios=[1,2,4,5]
    hc_publish_fn: Optional[Callable] = None
    # Separate PublishHandle for the HC-stressed re-benchmark's live/checkpoint
    # output. If None (default), the HC-stressed run gets NO live/checkpoint
    # output at all (matches current behavior) — set explicitly to give it
    # its own output_dir, distinct from the outer run's publish_fn, so the
    # two runs' checkpoint/live files never collide.
    publish_fn:  Optional[Callable]  = None
    # Optional PublishHandle for live streaming.  When set, on_scenario_start(),
    # on_timestep(), and on_scenario_end() are called from inside each runner.


# ===========================================================================
# BenchmarkResult
# ===========================================================================

@dataclass
class BenchmarkResult:
    """
    Output of run_benchmark().

    Attributes
    ----------
    network_id : str
        Human-readable network identifier.
    results : dict[int, ScenarioResult | dict | None]
        Keyed by scenario number (1-5).  A ScenarioResult for a scenario
        that ran this session.  A plain dict — shaped exactly like
        ScenarioResult.summary_dict()'s output — for a scenario skipped via
        the layer-1 "already complete" check and reloaded from a prior
        scenarios/<id>.json (no .records, no other methods; nothing else
        in this module reads anything from results[n] beyond what
        _ok_summary_row()'s isinstance-gated summary_dict()/dict access
        provides). None if the runner raised an exception.
    errors : dict[int, str]
        Full traceback string for each failed scenario.  Empty dict when
        all scenarios succeed.
    comparison_df : pd.DataFrame
        One row per scenario.  Columns: see _COMPARISON_COLS.
        Numeric fields for failed rows are np.nan.
        status column: "ok" or "failed".
        Written with index=False (scenario_num and scenario_id are columns).
    elapsed_s : float
        Total wall-clock time for all scenarios (seconds).
    csv_path : Path or None
        Absolute path of the written CSV file, or None when
        config.write_csv=False.
    """
    network_id:    str
    results:       dict[int, Optional[Union[ScenarioResult, dict]]]
    errors:        dict[int, str]
    comparison_df: pd.DataFrame
    elapsed_s:     float
    csv_path:      Optional[Path] = None
    hc_results:    Optional[list[HCResult]] = field(default=None)
    # List of two HCResult objects [baseline, volt_var], or None when
    # config.run_hc=False or HC raised an exception.
    hc_error:      Optional[str]            = field(default=None)
    # Full traceback string if HC analysis failed, else None.
    hc_benchmark:  Optional["BenchmarkResult"] = field(default=None)
    # BenchmarkResult from the recursive benchmark on the HC-stressed network.
    # Populated only when config.run_hc_scenarios=True and HC succeeded.
    net_hc: Optional[object] = field(default=None)
    # The pandapower network at baseline HC capacity (hc_mw), ready for export.
    # None when config.run_hc=False or HC failed.


# ===========================================================================
# Column schema
# ===========================================================================

# Ordered column list for the comparison DataFrame and CSV.
# Benchmark metadata columns come first; ScenarioResult metrics follow in
# the same order as summary_dict() so the two can be aligned by eye.
_COMPARISON_COLS: list[str] = [
    "scenario_num",
    "scenario_id",
    "scenario_label",
    "status",
    "network_id",
    # --- ScenarioResult metrics (same order as summary_dict()) ---
    "n_timesteps",
    "n_converged",
    "n_violation_steps",
    "violation_duration_h",
    "total_overvoltage_bus_steps",
    "total_undervoltage_bus_steps",
    "total_overloaded_line_steps",
    "total_overloaded_trafo_steps",
    "max_vm_pu",
    "min_vm_pu",
    "max_line_loading_pct",
    "max_trafo_loading_pct",
    "vdi",
    "q_total_mvar_abs",
    "reactive_energy_mvarh",
    "curtailment_steps",
    "curtailed_energy_mwh",
    "svc_bus",
    "svc_q_max",
    "elapsed_s",
    # --- Energy balance ---
    "total_losses_mwh",
    "grid_import_mwh",
    "grid_export_mwh",
    "der_gen_mwh",
    "load_demand_mwh",
    # --- Control effort (Scenario 4 only; NaN for others) ---
    "coordination_steps",
    "coordination_rate",
    "q_saturation_rate",
    # --- Failure metadata ---
    "error_message",
]

# Numeric columns that must be np.nan (not 0) on failed rows so the consumer
# can distinguish "zero violations because the run was clean" from "no data
# because the runner raised an exception".
_NUMERIC_COLS: frozenset[str] = frozenset({
    "n_timesteps",
    "n_converged",
    "n_violation_steps",
    "violation_duration_h",
    "total_overvoltage_bus_steps",
    "total_undervoltage_bus_steps",
    "total_overloaded_line_steps",
    "total_overloaded_trafo_steps",
    "max_vm_pu",
    "min_vm_pu",
    "max_line_loading_pct",
    "max_trafo_loading_pct",
    "vdi",
    "q_total_mvar_abs",
    "reactive_energy_mvarh",
    "curtailment_steps",
    "curtailed_energy_mwh",
    "curtail_exhausted_steps",
    "svc_bus",
    "svc_q_max",
    "elapsed_s",
    "total_losses_mwh",
    "grid_import_mwh",
    "grid_export_mwh",
    "der_gen_mwh",
    "load_demand_mwh",
    "coordination_steps",
    "coordination_rate",
    "q_saturation_rate",
})


# ===========================================================================
# Private helpers -- validation and kwargs
# ===========================================================================

def _safe_filename(s: str) -> str:
    """
    Replace characters that are awkward in filenames with underscores.

    Preserves alphanumerics, hyphens, and dots.
    "1-MV-rural--2-sw" -> "1-MV-rural--2-sw"  (unchanged)
    "net/v2 test"       -> "net_v2_test"
    """
    return re.sub(r"[^\w\-.]", "_", s)


def _validate_inputs(net, profiles: dict, config: BenchmarkConfig) -> None:
    """
    Validate inputs before the scenario loop.

    Raises ValueError early with a clear message rather than letting all
    five runners fail individually with the same root cause.

    Checks
    ------
    - profiles contains required keys "load", "pv", "wind", "times".
    - profiles["times"] is non-empty.
    - config.scenarios contains only valid numbers {1,2,3,4,5,6} with no
      duplicates (6 registry entries: 4A=4, 4B=5, OPF=6).
    - port is set when dry_run=False and scenario 4 is included.
    - v_min and v_max are finite and v_min < v_max.

    Column-alignment checks between profiles and net element indices are
    intentionally omitted.  adapt_profiles() inside each runner handles
    misalignment via reindex + fillna, and is better placed to diagnose it.
    """
    required_keys = {"load", "pv", "wind", "times"}
    missing = required_keys - set(profiles.keys())
    if missing:
        raise ValueError(
            f"profiles dict is missing required keys: {sorted(missing)}. "
            f"Keys present: {sorted(profiles.keys())}."
        )

    if len(profiles["times"]) == 0:
        raise ValueError(
            "profiles['times'] is empty. "
            "The profile must contain at least one timestep."
        )

    invalid = set(config.scenarios) - set(SCENARIO_REGISTRY)
    if invalid:
        raise ValueError(
            f"Invalid scenario numbers in config.scenarios: {sorted(invalid)}. "
            f"Valid values are {{1, 2, 3, 4, 5, 6}}."
        )

    if len(config.scenarios) != len(set(config.scenarios)):
        raise ValueError(
            f"config.scenarios contains duplicates: {config.scenarios}."
        )

    hardware_scenarios = {
        n for n in config.scenarios
        if SCENARIO_REGISTRY[n].supports_hardware
    }
    if hardware_scenarios and not config.dry_run and config.port is None:
        raise ValueError(
            f"config.port must be provided when config.dry_run=False and "
            f"hardware scenarios {sorted(hardware_scenarios)} are included. "
            "Example: config.port = '/dev/ttyACM0'"
        )

    if not np.isfinite(config.v_min) or not np.isfinite(config.v_max):
        raise ValueError(
            "config.v_min and config.v_max must be finite floats. "
            f"Got v_min={config.v_min}, v_max={config.v_max}."
        )

    if config.v_min >= config.v_max:
        raise ValueError(
            f"config.v_min must be strictly less than config.v_max. "
            f"Got v_min={config.v_min}, v_max={config.v_max}."
        )

    if config.run_hc_scenarios and config.profile_factory is None:
        raise ValueError(
            "config.profile_factory must be set when config.run_hc_scenarios=True. "
            "Provide a callable(net) -> profiles dict so the recursive benchmark "
            "can build time-series profiles for the HC-stressed network."
        )


def _is_lv_network(net) -> bool:
    """
    Return True if the network's distribution voltage level is LV (≤ 1.0 kV).

    Uses the statistical mode of net.bus.vn_kv — the same inference as
    hosting_capacity._infer_dist_voltage().  The minority HV slack bus does
    not influence the mode in any standard MV/LV or HV/MV network.
    """
    from statistics import mode as _mode
    return float(_mode(net.bus["vn_kv"].tolist())) <= 1.0


def _build_kwargs(
        spec:       ScenarioSpec,
        config:     BenchmarkConfig,
        network_id: str,
) -> dict:
    """
    Build the keyword argument dict to pass to a scenario runner.

    v_min and v_max are forwarded to all five runners -- all accept them and
    the benchmark must enforce the same voltage limits across every scenario
    for a fair comparison.

    dry_run and port are injected only for Scenario 4 (supports_hardware=True).
    verbose_opf is injected only for Scenario 5 (supports_opf_verbose=True).
    Passing unknown kwargs to Scenarios 1-3 would raise TypeError, so
    injection is gated on the ScenarioSpec flags rather than inspecting
    runner signatures at runtime.
    """
    kwargs: dict = {
        "network_id": network_id,
        "v_min":      config.v_min,
        "v_max":      config.v_max,
    }
    if spec.supports_hardware:
        kwargs["dry_run"] = config.dry_run
        kwargs["port"]    = config.port
        kwargs["coordination"] = spec.supports_coordination
    if spec.supports_opf_verbose:
        kwargs["verbose_opf"] = config.verbose_opf
    kwargs["publish_fn"] = config.publish_fn   # forwarded to all runners; None = no-op
    kwargs["enable_checkpointing"] = config.enable_checkpointing
    return kwargs



# ===========================================================================
# Private helpers -- comparison DataFrame construction
# ===========================================================================

def _ok_summary_row(
        n:          int,
        spec:       ScenarioSpec,
        result:     ScenarioResult,
        network_id: str,
) -> dict:
    """
    Build one comparison row from a successful ScenarioResult.

    `result` may also be a plain dict shaped like summary_dict()'s output —
    this happens when the scenario was skipped via the layer-1
    "already complete" check and results[n] holds the JSON's "summary"
    block directly rather than a reconstructed ScenarioResult (see the
    layer-1 block in the scenario loop above).

    The mapping from summary_dict() is explicit rather than derived
    dynamically.  If summary_dict() gains new fields they will not silently
    appear in the benchmark CSV until added here intentionally, keeping the
    column schema stable.
    """
    base = result.summary_dict() if isinstance(result, ScenarioResult) else result
    return {
        "scenario_num":                 n,
        "scenario_id":                  spec.scenario_id,
        "scenario_label":               spec.label,
        "status":                       "ok",
        "network_id":                   network_id,
        "n_timesteps":                  base.get("n_timesteps"),
        "n_converged":                  base.get("n_converged"),
        "n_violation_steps":            base.get("n_violation_steps"),
        "violation_duration_h":         base.get("violation_duration_h"),
        "total_overvoltage_bus_steps":  base.get("total_overvoltage_bus_steps"),
        "total_undervoltage_bus_steps": base.get("total_undervoltage_bus_steps"),
        "total_overloaded_line_steps":  base.get("total_overloaded_line_steps"),
        "total_overloaded_trafo_steps": base.get("total_overloaded_trafo_steps"),
        "max_vm_pu":                    base.get("max_vm_pu"),
        "min_vm_pu":                    base.get("min_vm_pu"),
        "max_line_loading_pct":         base.get("max_line_loading_pct"),
        "max_trafo_loading_pct":        base.get("max_trafo_loading_pct"),
        "vdi":                          base.get("vdi"),
        "q_total_mvar_abs":             base.get("q_total_mvar_abs"),
        "reactive_energy_mvarh":        base.get("reactive_energy_mvarh"),
        "curtailment_steps":            base.get("curtailment_steps"),
        "curtailed_energy_mwh":         base.get("curtailed_energy_mwh"),
        "curtail_exhausted_steps":           base.get("curtail_exhausted_steps"),
        "svc_bus":                      base.get("svc_bus"),
        "svc_q_max":                    base.get("svc_q_max"),
        "elapsed_s":                    base.get("elapsed_s"),
        "total_losses_mwh":             base.get("total_losses_mwh"),
        "grid_import_mwh":              base.get("grid_import_mwh"),
        "grid_export_mwh":              base.get("grid_export_mwh"),
        "der_gen_mwh":                  base.get("der_gen_mwh"),
        "load_demand_mwh":              base.get("load_demand_mwh"),
        "coordination_steps":           base.get("coordination_steps"),
        "coordination_rate":            base.get("coordination_rate"),
        "q_saturation_rate":            base.get("q_saturation_rate"),
        "error_message":                "",
    }


def _failed_summary_row(
        n:          int,
        spec:       ScenarioSpec,
        tb:         str,
        network_id: str,
) -> dict:
    """
    Build one comparison row for a scenario whose runner raised an exception.

    All numeric fields are np.nan -- not 0 -- so the consumer can distinguish
    "zero violations (clean run)" from "no data (runner failed)".

    error_message carries only the last line of the traceback.  The full
    traceback is stored in BenchmarkResult.errors[n] and logged at ERROR
    level; a full traceback in a CSV cell is unreadable.
    """
    last_line = tb.splitlines()[-1] if tb else "unknown error"
    row: dict = {
        "scenario_num":   n,
        "scenario_id":    spec.scenario_id,
        "scenario_label": spec.label,
        "status":         "failed",
        "network_id":     network_id,
        "error_message":  last_line,
    }
    for col in _NUMERIC_COLS:
        row[col] = np.nan
    return row


def _skipped_summary_row(
        n:          int,
        spec:       ScenarioSpec,
        network_id: str,
        reason:     str = "skipped: LV network",
) -> dict:
    """
    Build one comparison row for a scenario skipped due to network incompatibility.

    Distinct from "failed" (runner raised an exception) — "skipped" means the
    scenario was intentionally bypassed because the network type is unsupported.
    All numeric fields are np.nan for the same reason as _failed_summary_row.
    """
    row: dict = {
        "scenario_num":   n,
        "scenario_id":    spec.scenario_id,
        "scenario_label": spec.label,
        "status":         "skipped",
        "network_id":     network_id,
        "error_message":  reason,
    }
    for col in _NUMERIC_COLS:
        row[col] = np.nan
    return row


def _build_comparison_df(
        results:    dict[int, Optional[ScenarioResult]],
        errors:     dict[int, str],
        scenarios:  list[int],
        network_id: str,
) -> pd.DataFrame:
    """
    Assemble the full comparison DataFrame.

    Rows are always in ascending scenario-number order.
    Column order follows _COMPARISON_COLS exactly.
    No DataFrame index is set -- the CSV is written with index=False.
    """
    rows = []
    for n in sorted(scenarios):
        spec   = SCENARIO_REGISTRY[n]
        result = results.get(n)
        if result is not None:
            rows.append(_ok_summary_row(n, spec, result, network_id))
        elif n in errors and errors[n] == "skipped":
            rows.append(_skipped_summary_row(n, spec, network_id))
        else:
            rows.append(_failed_summary_row(n, spec, errors.get(n, ""), network_id))

    return pd.DataFrame(rows, columns=_COMPARISON_COLS)


# ===========================================================================
# Private helpers -- CSV export
# ===========================================================================

def _write_csv(
        df:         pd.DataFrame,
        network_id: str,
        output_dir: Union[str, Path, None],
) -> Path:
    """
    Write the comparison DataFrame to a timestamped CSV.

    Filename: <safe_network_id>_benchmark_<YYYYMMDD_HHMMSS_ffffff>.csv
    Microseconds (_ffffff) prevent overwrite when the benchmark is re-run
    within the same second (e.g. rapid test cycles).
    Written to output_dir (created if absent) or Path.cwd() when None.
    index=False -- scenario_num and scenario_id are normal columns.
    """
    out_dir = Path(output_dir) if output_dir is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"{_safe_filename(network_id)}_benchmark_{ts}.csv"
    path = out_dir / name

    df.to_csv(path, index=False)
    return path.resolve()


# ===========================================================================
# Private helpers -- console summary
# ===========================================================================

def _fmt_cell(val, fmt: str = ".4f") -> str:
    """
    Format one table cell value.  Returns "-" for missing data.

    pd.isna() is used instead of isinstance(val, float) + np.isnan() because
    it handles np.nan, pd.NA, pd.NaT, and None uniformly.  The try/except
    around pd.isna() handles the edge case where val is array-like (for which
    pd.isna returns an array that raises ValueError in boolean context).
    """
    try:
        if pd.isna(val):
            return "-"
    except (TypeError, ValueError):
        pass
    try:
        return f"{val:{fmt}}"
    except (ValueError, TypeError):
        return str(val)


def _print_summary_rich(
        comparison_df: pd.DataFrame,
        network_id:    str,
        errors:        dict[int, str],
        elapsed_s:     float,
        csv_path:      Optional[Path],
        v_min:         float,
        v_max:         float,
) -> None:
    """
    Print the benchmark summary table using the shared CLI Rich console.

    Voltage columns use CLI_THEME semantic tokens:
      - max_vm_pu > v_max  ->  [overvoltage] (yellow)
      - min_vm_pu < v_min  ->  [undervoltage] (blue)

    This matches the visual language used by other CLI output in the project.
    Falls back to _print_summary_plain() when Rich or _console is unavailable.

    Parameters
    ----------
    v_min, v_max : float
        The configured voltage planning limits forwarded from BenchmarkConfig.
        Highlighting is applied against these actual limits, not hardcoded
        defaults, so stricter or wider benchmark bands display correctly.
    """
    if not _RICH_AVAILABLE:
        _print_summary_plain(
            comparison_df, network_id, errors, elapsed_s, csv_path, v_min, v_max
        )
        return

    _console.print()
    _console.rule(f"[header]BENCHMARK SUMMARY -- {network_id}[/header]")

    table = RichTable(
        box=rich_box.SIMPLE_HEAVY,
        show_header=True,
        header_style="header",
        expand=False,
    )
    table.add_column("#",        style="muted",   width=3,  justify="right")
    table.add_column("Scenario", style="current", width=26)
    table.add_column("Status",                    width=8,  justify="center")
    table.add_column("Viol",                      width=6,  justify="right")
    table.add_column("Conv/Tot",                  width=10, justify="right")
    table.add_column("max_V",                     width=7,  justify="right")
    table.add_column("min_V",                     width=7,  justify="right")
    table.add_column("maxLL%",                    width=7,  justify="right")
    table.add_column("Q_tot",                     width=8,  justify="right")
    table.add_column("Curtail",                   width=10, justify="right")
    table.add_column("Time(s)",                   width=8,  justify="right")

    for _, row in comparison_df.iterrows():
        is_ok      = row["status"] == "ok"
        status_str = "[ok]ok[/ok]" if is_ok else "[error]FAILED[/error]"

        conv_tot = (
            f"{int(row['n_converged'])}/{int(row['n_timesteps'])}"
            if is_ok and pd.notna(row.get("n_converged"))
            else "-"
        )

        # Voltage cells: use CLI_THEME semantic tokens so highlighting is
        # consistent with violation reporting elsewhere in the CLI.
        # overvoltage -> yellow, undervoltage -> blue (per CLI_THEME).
        max_v_raw = row.get("max_vm_pu")
        min_v_raw = row.get("min_vm_pu")
        max_v = _fmt_cell(max_v_raw)
        min_v = _fmt_cell(min_v_raw)
        if is_ok and pd.notna(max_v_raw) and max_v_raw > v_max:
            max_v = f"[overvoltage]{max_v}[/overvoltage]"
        if is_ok and pd.notna(min_v_raw) and min_v_raw < v_min:
            min_v = f"[undervoltage]{min_v}[/undervoltage]"

        table.add_row(
            str(int(row["scenario_num"])),
            str(row["scenario_label"]),
            status_str,
            _fmt_cell(row.get("n_violation_steps"), ".0f"),
            conv_tot,
            max_v,
            min_v,
            _fmt_cell(row.get("max_line_loading_pct"), ".1f"),
            _fmt_cell(row.get("q_total_mvar_abs"),     ".2f"),
            _fmt_cell(row.get("curtailed_energy_mwh"), ".3f"),
            _fmt_cell(row.get("elapsed_s"),            ".1f"),
        )

    _console.print(table)
    _console.print(f"[muted]Total wall-clock: {elapsed_s:.1f} s[/muted]")
    if csv_path is not None:
        _console.print(f"[muted]CSV written to:   {csv_path}[/muted]")

    if errors:
        _console.print()
        _console.print("[error]Failed scenarios:[/error]")
        for n, tb in sorted(errors.items()):
            spec      = SCENARIO_REGISTRY[n]
            last_line = tb.splitlines()[-1] if tb else "unknown error"
            _console.print(f"  [{n}] {spec.label}: [error]{last_line}[/error]")

    _console.print()


def _print_summary_plain(
        comparison_df: pd.DataFrame,
        network_id:    str,
        errors:        dict[int, str],
        elapsed_s:     float,
        csv_path:      Optional[Path],
        v_min:         float,
        v_max:         float,
) -> None:
    """
    Plain-text fallback summary used when Rich or _console is unavailable.

    All output is ASCII-safe for Windows terminals and redirected log files.
    v_min and v_max are accepted for API consistency with _print_summary_rich
    but voltage highlighting is not applied in this fallback.
    """
    SEP = "=" * 80
    print(f"\n{SEP}")
    print(f"  BENCHMARK SUMMARY -- {network_id}")
    print(SEP)
    print(
        f"{'#':>2}  {'Scenario':<26} {'Status':^8} {'Viol':>5} "
        f"{'Conv/Tot':>10} {'maxV':>7} {'minV':>7} "
        f"{'maxLL%':>6} {'Q_tot':>8} {'Curtail':>9} {'Time':>7}"
    )
    print("-" * 80)
    for _, row in comparison_df.iterrows():
        is_ok    = row["status"] == "ok"
        conv_tot = (
            f"{int(row['n_converged'])}/{int(row['n_timesteps'])}"
            if is_ok and pd.notna(row.get("n_converged")) else "-"
        )
        print(
            f"{int(row['scenario_num']):>2}  "
            f"{str(row['scenario_label']):<26} "
            f"{'ok' if is_ok else 'FAILED':^8} "
            f"{_fmt_cell(row.get('n_violation_steps'), '.0f'):>5} "
            f"{conv_tot:>10} "
            f"{_fmt_cell(row.get('max_vm_pu')):>7} "
            f"{_fmt_cell(row.get('min_vm_pu')):>7} "
            f"{_fmt_cell(row.get('max_line_loading_pct'), '.1f'):>6} "
            f"{_fmt_cell(row.get('q_total_mvar_abs'), '.2f'):>8} "
            f"{_fmt_cell(row.get('curtailed_energy_mwh'), '.3f'):>9} "
            f"{_fmt_cell(row.get('elapsed_s'), '.1f'):>7}"
        )
    print(SEP)
    print(f"  Total wall-clock: {elapsed_s:.1f} s")
    if csv_path is not None:
        print(f"  CSV written to:   {csv_path}")
    if errors:
        print("  Failed scenarios:")
        for n, tb in sorted(errors.items()):
            spec = SCENARIO_REGISTRY[n]
            print(f"    [{n}] {spec.label}: {tb.splitlines()[-1] if tb else 'unknown'}")
    print()


# ===========================================================================
# Public API
# ===========================================================================

def run_benchmark(
        net,
        profiles:   dict,
        network_id: str                       = "unknown",
        config:     Optional[BenchmarkConfig] = None,
) -> BenchmarkResult:
    """
    Run all (or a subset of) the five HIL benchmark scenarios and compare.

    Parameters
    ----------
    net : pandapower network
        Deep-copied once per scenario internally.  The original is never
        modified.  The caller does not need to copy beforehand.
    profiles : dict
        Output of profile_builder.build_profiles().  Must contain keys
        "load", "pv", "wind", "times".  Shared across all runners --
        adapt_profiles() inside each runner does its own .copy().
    network_id : str
        Human-readable identifier used in log messages and the CSV filename.
    config : BenchmarkConfig or None
        Run configuration.  None -> BenchmarkConfig() defaults:
            scenarios = [1, 2, 3, 4, 5]
            dry_run   = True
            write_csv = True
            v_min / v_max = 0.95 / 1.05

    Returns
    -------
    BenchmarkResult
        results       - dict[int, ScenarioResult | None] keyed 1-5
        errors        - dict[int, str] with full tracebacks for failures
        comparison_df - pd.DataFrame with one row per scenario
        csv_path      - Path to the written CSV, or None
        elapsed_s     - total wall-clock time (s)

    Raises
    ------
    ValueError
        On invalid config or missing/empty profile keys.  See _validate_inputs
        for the full list of checks.

    Notes
    -----
    Scenarios always run in ascending order (1->5) regardless of the order
    passed in config.scenarios.  sorted() is applied unconditionally so
    benchmark output is always reproducible and CSV rows always follow the
    canonical ordering.

    copy.deepcopy(net) is placed inside the try block so a copy failure
    produces a failed-scenario row rather than aborting the whole benchmark.
    """
    if config is None:
        config = BenchmarkConfig()

    _validate_inputs(net, profiles, config)

    t_start = time.perf_counter()
    results: dict[int, Optional[ScenarioResult]] = {}
    errors:  dict[int, str]                      = {}

    is_lv = _is_lv_network(net)
    if is_lv:
        logger.info(
            "[Benchmark | %s] LV network detected — scenarios without "
            "supports_lv will be skipped.",
            network_id,
        )

    # Compute once before the loop — net does not change across scenarios
    # Check whether any PV/wind DERs are actually present in this network.
    # This matters for the HC-stressed re-benchmark where HC has added sgens
    # to an otherwise DER-free LV network (e.g. CIGRE LV net_hc).
    _type_col = net.sgen.get("type", pd.Series("", index=net.sgen.index)).fillna("")
    _has_ders = (
    not net.sgen.empty
    and net.sgen["in_service"].any()
    and _type_col.str.lower().str.contains("pv|solar|wind|wp", na=False).any()
    )

    _live_csv_lock_results: dict[int, Optional[ScenarioResult]] = {}
    # Populated by the runner-side callback below with a PARTIAL ScenarioResult
    # for the in-progress scenario. Merged with `results` (completed scenarios)
    # each time the CSV is rewritten.

    def _rewrite_live_csv(current_n: int, partial_result: ScenarioResult) -> None:
        if config.live_csv_path is None:
            return
        merged = dict(results)              # completed scenarios so far
        merged[current_n] = partial_result  # in-progress scenario's partial state
        df = _build_comparison_df(merged, errors, config.scenarios, network_id)
        Path(config.live_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.live_csv_path, index=False)

    for n in sorted(config.scenarios):
        spec = SCENARIO_REGISTRY[n]

        if is_lv and not spec.supports_lv and not _has_ders:
            logger.info(
                "[Benchmark | %s] Scenario %d (%s) skipped — LV network with no "
                "pre-existing DERs; nothing for the controller to act on.",
                network_id, n, spec.label,
            )
            errors[n]  = "skipped"
            results[n] = None
            continue
        # ------------------------------------------------------------------
        # Layer 1 — scenario-level "already complete" check. If
        # scenarios/<scenario_id>.json already exists in this output_dir,
        # this scenario finished successfully in a prior attempt against the
        # same output directory — skip the runner entirely and store its
        # "summary" dict directly in results[n]. Sits ABOVE the per-timestep
        # checkpoint (layer 2, inside each runner), which only handles a
        # scenario that is fresh or crashed partway through.
        #
        # results[n] here is a plain dict (summary_dict()-shaped), not a
        # ScenarioResult instance — see _ok_summary_row()'s isinstance check
        # and the BenchmarkResult.results docstring below.
        # ------------------------------------------------------------------
        if config.publish_fn is not None and getattr(config.publish_fn, "output_dir", None):
            _scenario_json = Path(config.publish_fn.output_dir) / "scenarios" / f"{spec.scenario_id}.json"
            if _scenario_json.exists():
                try:
                    with open(_scenario_json, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    results[n] = payload["summary"]
                    logger.info(
                        "[Benchmark | %s] Scenario %d (%s) already complete — "
                        "loaded summary from %s, skipping runner.",
                        network_id, n, spec.label, _scenario_json,
                    )
                    continue
                except Exception:
                    logger.warning(
                        "[Benchmark | %s] Scenario %d (%s): found %s but failed "
                        "to load it (%s) — re-running from scratch.",
                        network_id, n, spec.label, _scenario_json,
                        traceback.format_exc().splitlines()[-1],
                    )

        logger.info(
            "[Benchmark | %s] -- Scenario %d (%s) starting ...",
            network_id, n, spec.label,
        )

        try:
            # deepcopy is inside try so a copy failure produces a failed
            # row rather than aborting the benchmark.  See module docstring.
            net_copy = copy.deepcopy(net)
            kwargs   = _build_kwargs(spec, config, network_id)
            if config.live_csv_path is not None:
                kwargs["live_csv_rewrite_fn"] = (
                    lambda partial_result, _n=n: _rewrite_live_csv(_n, partial_result)
                )

            t_scen = time.perf_counter()
            result = spec.runner(net_copy, profiles, **kwargs)
            scen_elapsed = time.perf_counter() - t_scen

            logger.info(
                "[Benchmark | %s] Scenario %d (%s) done in %.1f s | "
                "violations=%d | converged=%d/%d",
                network_id, n, spec.label, scen_elapsed,
                result.n_violation_steps,
                result.n_converged, result.n_timesteps,
            )
            results[n] = result
            if config.publish_fn is not None and getattr(config.publish_fn, "output_dir", None):
                publish_scenario_result(result, output_dir=config.publish_fn.output_dir)

                # Archive (don't delete) the checkpoint now that the final
                # JSON is safely written. Renamed, not removed — output_dir
                # gets copied off the RPi to a laptop for analysis anyway,
                # so the disk-space motivation for deleting doesn't apply.
                # Renaming keeps the raw checkpoint stream around for
                # audit/debugging, while making sure get_resume_records() —
                # which only ever looks for the exact
                # "checkpoint/<scenario_id>.jsonl" path first — can't
                # mistake a finished scenario's checkpoint for one that's
                # still live.
                _ckpt = Path(config.publish_fn.output_dir) / "checkpoint" / f"{spec.scenario_id}.jsonl"
                if _ckpt.exists():
                    _archived = _ckpt.parent / (_ckpt.name + ".completed")
                    _ckpt.rename(_archived)
                    logger.info(
                        "[Benchmark | %s] Scenario %d (%s) checkpoint archived -> %s",
                        network_id, n, spec.label, _archived,
                    )

        except Exception:
            tb = traceback.format_exc()
            errors[n]  = tb
            results[n] = None
            logger.error(
                "[Benchmark | %s] Scenario %d (%s) FAILED:\n%s",
                network_id, n, spec.label, tb,
            )
            # Do not re-raise.  Continue to next scenario.
    # ------------------------------------------------------------------
    # Hosting capacity analysis — runs after all scenario loops complete.
    # Uses a fresh deepcopy of the original net (pre-scenario mutations).
    # ------------------------------------------------------------------
    hc_results_list: Optional[list[HCResult]] = None
    hc_error: Optional[str] = None
    hc_benchmark: Optional[BenchmarkResult] = None
    net_hc: Optional[object] = None

    if config.run_hc:
        logger.info("[Benchmark | %s] -- Hosting capacity analysis starting ...", network_id)
        try:
            t_hc = time.perf_counter()
            hc_baseline, net_hc = run_baseline_hc(net, network_id)
            hc_voltvar           = run_hc_with_volt_var(net, network_id)
            hc_results_list = [hc_baseline, hc_voltvar]
            logger.info(
                "[Benchmark | %s] HC done in %.1f s | "
                "baseline=%.3f MW | volt_var=%.3f MW | gain=%.3f MW",
                network_id,
                time.perf_counter() - t_hc,
                hc_baseline.hc_mw,
                hc_voltvar.hc_mw,
                hc_voltvar.hc_mw - hc_baseline.hc_mw,
            )
        except Exception:
            hc_error = traceback.format_exc()
            net_hc   = None
            logger.error(
                "[Benchmark | %s] Hosting capacity analysis FAILED:\n%s",
                network_id, hc_error,
            )

    if config.run_hc_scenarios and net_hc is not None:
        hc_network_id = f"{network_id}_hc_stressed"
        logger.info(
            "[Benchmark | %s] -- HC-stressed re-benchmark starting on %s ...",
            network_id, hc_network_id,
        )
        try:
            profiles_hc = config.profile_factory(net_hc)

            # Write topology + profiles for the HC-stressed network NOW —
            # net_hc and profiles_hc are both final at this point, and this
            # is genuinely "network-load time" for this network, before any
            # of its scenarios run. Uses hc_publish_fn's output_dir so this
            # lands in the same folder as that handle's live/checkpoint
            # files and the eventual per-scenario writes below.
            if config.hc_publish_fn is not None and getattr(config.hc_publish_fn, "output_dir", None):
                publish_topology_and_profiles(
                    net_hc, profiles_hc,
                    output_dir = config.hc_publish_fn.output_dir,
                    network_id = hc_network_id,
                )
            config_hc   = BenchmarkConfig(
                scenarios = config.hc_stress_scenarios if config.hc_stress_scenarios is not None else config.scenarios,
                dry_run          = config.dry_run,
                port             = config.port,
                output_dir       = config.output_dir,
                write_csv        = config.write_csv,
                verbose_opf      = config.verbose_opf,
                v_min            = config.v_min,
                v_max            = config.v_max,
                run_hc           = False,   # no recursion into HC again
                run_hc_scenarios = False,   # prevent infinite recursion
                profile_factory  = None,
                publish_fn       = config.hc_publish_fn,
                enable_checkpointing  = config.enable_checkpointing,
                live_csv_path         = config.live_csv_path,
            )
            hc_benchmark = run_benchmark(
                net_hc,
                profiles_hc,
                network_id = hc_network_id,
                config     = config_hc,
            )
        except Exception:
            logger.error(
                "[Benchmark | %s] HC-stressed re-benchmark FAILED:\n%s",
                network_id, traceback.format_exc(),
            )
    comparison_df = _build_comparison_df(results, errors, config.scenarios, network_id)

    csv_path: Optional[Path] = None
    if config.write_csv:
        csv_path = _write_csv(comparison_df, network_id, config.output_dir)
        logger.info("[Benchmark | %s] CSV written to: %s", network_id, csv_path)

    elapsed = time.perf_counter() - t_start

    _print_summary_rich(
        comparison_df, network_id, errors, elapsed, csv_path,
        config.v_min, config.v_max,
    )

    return BenchmarkResult(
            network_id    = network_id,
            results       = results,
            errors        = errors,
            comparison_df = comparison_df,
            elapsed_s     = elapsed,
            csv_path      = csv_path,
            hc_results    = hc_results_list,
            hc_error      = hc_error,
            hc_benchmark  = hc_benchmark,
            net_hc = net_hc,
        )
