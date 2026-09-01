"""
Terminal presentation helpers for the HIL CLI.

This module owns console display only. It does not run simulations,
extract data, publish to Flask, persist files, or mutate run state.

All functions receive already-prepared values and format them for the
terminal using Rich.
"""

from __future__ import annotations
from ._console import console
from rich.table import Table
from .run_plan import RunPlan
 
def print_section_header(run_plan: RunPlan, current_stage: str) -> None:
    """Print a visually distinct section header for the current run stage."""
    
    console.print()
    console.print("[header]HIL CLI[/header] [muted]| Hardware-in-the-Loop Voltage Control[/muted]")
    console.rule(f"[stage]{current_stage}[/stage]", style="rule")

    console.print(
        f"[muted]Study: {run_plan.study or 'unknown'} | "
        f"Stream every k: {run_plan.stream_every_k} | "
        f"Target: {'HARDWARE (' + run_plan.port + ')' if run_plan.hardware else 'dry run'}[/muted]"

    )
    console.print(
        f"[muted]Run ID: {run_plan.run_id} | "
        f"Network: {run_plan.network.preset_name or run_plan.network.simbench_code or run_plan.network.custom_path or run_plan.network.source_type or 'unknown'} | "
        f"Dataset: {run_plan.dataset.source_type or 'unknown'} | "
        f"Δt: {run_plan.parameters.timestep_resolution} min[/muted]"
    )
    console.rule("[muted]end of section[/muted]",style="rule_end")

def print_timestep_line(timestep: int, algorithm: str, n_ov: int, n_uv: int, n_overload: int) -> None:
    """
    Print a compact per-timestep violation status line.

    Returns OK when no violations exist.
    """

    if n_overload + n_ov + n_uv == 0:
        console.print(
            f"[header]t={timestep}[/header] | "
            f"[rule]{algorithm}[/rule] | "
            f"[ok] OK [/ok]"
        )
        return
    
    console.print(
        f"[error]t={timestep}[/error] | "
        f"[rule]{algorithm}[/rule] | "
        f"[warning]OV: {n_ov}[/warning] | "
        f"[undervoltage]UV: {n_uv}[/undervoltage] | "
        f"[overload]Overload: {n_overload}[/overload]"
    )

def print_run_plan(run_plan: RunPlan) -> None:
    """
    Display the selected run configuration before execution starts.

    This function only formats values already stored in RunPlan. It does not
    validate, compute, load networks, build profiles, or mutate the run plan.
    """
    parameters = run_plan.parameters
    network  = run_plan.network
    dataset = run_plan.dataset

    network_label = (
        network.preset_name
        or network.simbench_code
        or network.custom_path
        or network.source_type
        or "unknown"
    )

    dataset_label = (
        dataset.custom_path
        or dataset.source_type
        or "unknown"
    )

    table_run = Table(
        title="Run Configuration Confirmation",
        show_header=True,
        header_style="header"
    )
    table_network = Table(
        title="Network Configuration Confirmation",
        show_header=True,
        header_style="header"
    )

    table_dataset = Table(
        title="Dataset Configuration Confirmation",
        show_header=True,
        header_style="header"
    )

    table_limits = Table(
        title="Limits Parameters Confirmation",
        show_header=True,
        header_style="header"
    )

    console.rule(f"[stage]Run Plan Configuration Confirmation[/stage]", style="rule")
    table_run.add_column("Section", style="stage",no_wrap=True)
    table_run.add_column("Field", style="muted")
    table_run.add_column("Value")

    table_network.add_column("Section", style="stage",no_wrap=True)
    table_network.add_column("Field", style="muted")
    table_network.add_column("Value")

    table_dataset.add_column("Section", style="stage",no_wrap=True)
    table_dataset.add_column("Field", style="muted")
    table_dataset.add_column("Value")

    table_limits.add_column("Section", style="stage",no_wrap=True)
    table_limits.add_column("Field", style="muted")
    table_limits.add_column("Value")

    table_run.add_row("Run", "Run ID", run_plan.run_id)
    table_run.add_row("Run", "Study", run_plan.study or "unknown")
    table_run.add_row("Run", "Output Directory", run_plan.output_dir or "unknown")
    table_run.add_row("Run", "Time Window", (f"{run_plan.time_period} {run_plan.time_index}" if run_plan.time_period and run_plan.time_period != "full" else "full annual"))
    table_run.add_row("Run", "Execution Target", (f"HARDWARE (Arduino on {run_plan.port})" if run_plan.hardware else "dry run (pure-Python Q(V))"))
    table_run.add_row("Run", "Controller Plugin", run_plan.controller_plugin_path or "none")
    if run_plan.study == "hosting_capacity":
        table_run.add_row("Run", "HC Stressed Re-benchmark", "yes" if run_plan.hc_stressed else "no")

    table_dataset.add_row("Dataset", "Source Type", dataset.source_type or "unknown")
    table_dataset.add_row("Dataset", "Selected Dataset", dataset_label)
    table_dataset.add_row("Dataset", "Station ID", dataset.station_id or "unknown")
    table_dataset.add_row("Dataset", "Year", str(dataset.year) if dataset.year is not None else "unknown")
    table_dataset.add_row("Dataset", "File Mapping", str(dataset.file_map) if dataset.file_map is not None else "unknown")
    table_dataset.add_row("Dataset", "Column Mapping", str(dataset.col_map) if dataset.col_map is not None else "Default Dataset used")

    table_network.add_row("Network", "Source Type", network.source_type or "unknown")
    table_network.add_row("Network", "Selected Network", network_label)
    table_network.add_row("Network", "Preset Family", network.preset_family or "unknown")
    table_network.add_row("Network", "Custom Network File", network.custom_path or "Default Networks used")
    table_network.add_row("Network", "Network Plugin (YAML)", network.plugin_path or "none")
    table_network.add_row("Network", "Injected DERs", str(network.der_placements) if network.der_placements else "none")
    table_network.add_row("Network", "Switches To Flip", str(run_plan.network.switches_to_flip) if run_plan.network.switches_to_flip else "none")

    table_limits.add_row("Limits", "Voltage Range",f"{parameters.v_min:.3f}–{parameters.v_max:.3f} pu")
    table_limits.add_row("Limits", "Line Loading Max", f"{parameters.line_max_loading:.1f} %")
    table_limits.add_row("Limits", "Transformer Loading Max",f"{parameters.trafo_max_loading:.1f} %")
    table_limits.add_row("Limits", "Angle Difference Max", f"{parameters.va_diff_max_degree:.1f}°")
    table_limits.add_row("Limits", "Unbalance Max", f"{parameters.unbalance_max_percent:.1f} %")
    table_limits.add_row("Timing", "Timestep Resolution", f"{parameters.timestep_resolution} min")
    _qv = (f"q_ratio={parameters.q_ratio}, U1..U4={parameters.u1_pu}/{parameters.u2_pu}/{parameters.u3_pu}/{parameters.u4_pu}"
           if parameters.q_ratio is not None else "framework defaults (Q_RATIO=0.25)")
    table_limits.add_row("Q(V)", "Characteristic", _qv)
    table_limits.add_row("Network (Run Plan)", "Focused buses", str(run_plan.focus_buses) if run_plan.focus_buses is not None else "All")
    table_limits.add_row("Network (Parameter)", "DER Scaling Factor", str(parameters.der_scaling) if parameters.der_scaling is not None else "Default DER scaling used")
    table_limits.add_row("Network (Parameter)", "Load Scaling Factor", str(parameters.load_scaling) if parameters.load_scaling is not None else "Default Load scaling used")
    table_limits.add_row("Streaming", "Stream every k (Flask)", str(run_plan.stream_every_k) if run_plan.stream_every_k is not None else "unknown")

    console.print()
    console.print("[header]HIL CLI[/header] [muted]| Run configuration preview[/muted]")
    console.print(table_run)
    console.print(table_dataset)
    console.print(table_network)
    console.print(table_limits)
    console.rule("[muted]end of section[/muted]",style="rule_end")
    
def print_error_message(error: Exception, context: str) -> None:
    """
    Print a generic error message with caller-provided context.

    The caller decides where this function is used. This helper only formats the
    exception type, exception message, and the context string for terminal output.
    """
    console.print()
    console.rule("[error]Error[/error]", style="error")
    console.print(f"[stage]Context:[/stage] {context}")
    console.print(f"[stage]Type:[/stage] {type(error).__name__}")
    console.print(f"[stage]Message:[/stage] {str(error) or 'No Error Message provided'}")
    console.rule("[muted]end of section[/muted]",style="rule_end")

def print_summary_table(summary:dict) -> None:
    """
    Print the final run summary table.

    The caller must provide a pre-computed summary dictionary. This function does
    not calculate metrics, inspect result files, or modify the summary.
    """
    table = Table(
        title="Run Summary",
        show_header=True,
        header_style="header",
    )

    table.add_column("Metric", style="muted")
    table.add_column("Value")

    for key,value in summary.items():
        table.add_row(
            str(key).replace("_", " ").title(),
            str(value) if value is not None else "Not Available"
        )
    
    console.print()
    console.rule("[stage]Run completed[/stage]",style="rule")
    console.print(table)
    console.rule("[muted]end of section[/muted]",style="rule_end")
