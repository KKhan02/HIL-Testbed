"""
Standalone smoke test for cli/helpers.py.

This script uses dummy RunPlan values and calls all five terminal
presentation helper functions.
"""

from __future__ import annotations

from .run_plan import DatasetConfig, NetworkConfig, ParameterConfig, RunPlan
from .helpers import (
    print_error_message,
    print_run_plan,
    print_section_header,
    print_summary_table,
    print_timestep_line,
)


def build_dummy_run_plan() -> RunPlan:
    """Build a representative dummy RunPlan for helper display testing."""

    parameters = ParameterConfig(
        v_min=0.95,
        v_max=1.05,
        line_max_loading=100,
        trafo_max_loading=100,
        va_diff_max_degree=30,
        unbalance_max_percent=2.0,
        der_scaling={None: 1.25, 18: 0.80},
        load_scaling={None: 1.10},
        timestep_resolution=15,
    )

    network = NetworkConfig(
        source_type="simbench_code",
        preset_name=None,
        simbench_code="1-MV-rural--2-sw",
        custom_path=None,
        custom_function_name="get_network",
        preset_family="MV rural",
        simbench_selections={
            "voltage_level": "MV",
            "grid_type": "rural",
            "der_level": "sw",
        },
    )

    dataset = DatasetConfig(
        source_type="dwd",
        data_dir="data/dwd",
        station_id="691",
        year=2016,
        file_map=None,
        col_map=None,
        custom_path=None,
    )

    return RunPlan(
        parameters=parameters,
        network=network,
        dataset=dataset,
        study="scenario_comparison",
        focus_buses=[12, 18, 24],
        stream_every_k=4,
        output_dir="runs",
    )


def main() -> None:
    """Run all helper display functions once."""

    run_plan = build_dummy_run_plan()

    print_run_plan(run_plan)

    print_section_header(run_plan, "Pre-control power flow")
    print_timestep_line(
        timestep=1,
        algorithm="baseline",
        n_ov=0,
        n_uv=0,
        n_overload=0,
    )

    print_section_header(run_plan, "Arduino Volt-Var exchange")
    print_timestep_line(
        timestep=42,
        algorithm="volt_var_hil",
        n_ov=3,
        n_uv=1,
        n_overload=2,
    )

    try:
        raise TimeoutError("No response from Arduino after 3 retries.")
    except Exception as exc:
        print_error_message(
            error=exc,
            context="Arduino exchange during timestep 42",
        )

    summary = {
        "run_id": run_plan.run_id,
        "study": run_plan.study,
        "timesteps_completed": 96,
        "max_overvoltage_count": 3,
        "max_undervoltage_count": 1,
        "max_overload_count": 2,
        "failed_timesteps": 0,
        "output_dir": f"{run_plan.output_dir}/{run_plan.run_id}",
    }

    print_summary_table(summary)


if __name__ == "__main__":
    main()