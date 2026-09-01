from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path

import pandas as pd
import simbench as sb

# ---------------------------------------------------------------------
# Make local scenario files importable.
# Supports both:
#   project_root/scenario_runners/scenario_result.py
# and:
#   project_root/scenario_result.py
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SCENARIO_DIR = PROJECT_ROOT / "scenario_runners"

if SCENARIO_DIR.exists():
    sys.path.insert(0, str(SCENARIO_DIR))
else:
    sys.path.insert(0, str(PROJECT_ROOT))

from profile_builder import build_annual_profiles
from scenario_1_baseline import run_scenario_1
from scenario_4_volt_var import run_scenario_4
from scenario_5_opf_rewritten import run_scenario_5


NET_ID = "1-MV-rural--2-sw"


def choose_test_day(profiles: dict) -> str:
    """
    Prefer profile_builder's max_der day.
    Fallback: choose the day with the highest total PV + wind generation.
    """
    extreme_days = profiles.get("extreme_days", {})
    if extreme_days.get("max_der"):
        return extreme_days["max_der"]

    times = profiles["times"]
    pv = profiles.get("pv", pd.DataFrame(index=times))
    wind = profiles.get("wind", pd.DataFrame(index=times))

    total_der = pd.Series(0.0, index=times)
    if pv is not None and not pv.empty:
        total_der = total_der.add(pv.sum(axis=1), fill_value=0.0)
    if wind is not None and not wind.empty:
        total_der = total_der.add(wind.sum(axis=1), fill_value=0.0)

    if total_der.max() <= 0:
        return str(times[0].date())

    return total_der.resample("D").sum().idxmax().strftime("%Y-%m-%d")


def slice_profiles_one_day(profiles: dict, day: str) -> dict:
    """
    Return a shallow copy of the profile dict sliced to one calendar day.
    Works for timezone-aware SimBench profile indices.
    """
    times = profiles["times"]
    tz = times.tz

    start = pd.Timestamp(day, tz=tz)
    end = start + pd.Timedelta(days=1)

    mask = (times >= start) & (times < end)
    sliced_times = times[mask]

    if len(sliced_times) == 0:
        raise ValueError(f"No timesteps found for test day {day}.")

    out = dict(profiles)
    out["times"] = sliced_times

    for key in ["load", "pv", "wind"]:
        df = profiles.get(key)
        if df is None:
            out[key] = pd.DataFrame(index=sliced_times)
        elif df.empty:
            out[key] = pd.DataFrame(index=sliced_times, columns=df.columns, dtype=float)
        else:
            out[key] = df.reindex(sliced_times).fillna(0.0)

    out["extreme_days"] = {"max_der": day}
    return out

def slice_profiles_one_timestep(profiles: dict, step: int = 0) -> dict:
    """
    Return a shallow copy of profiles containing exactly one timestep.
    Keeps load, pv, and wind aligned to the selected timestamp.
    """
    times = profiles["times"]

    if step < 0 or step >= len(times):
        raise IndexError(f"step={step} out of range for {len(times)} timesteps.")

    selected_time = times[step:step + 1]

    out = dict(profiles)
    out["times"] = selected_time

    for key in ["load", "pv", "wind"]:
        df = profiles.get(key)

        if df is None:
            out[key] = pd.DataFrame(index=selected_time)
        elif df.empty:
            out[key] = pd.DataFrame(index=selected_time, columns=df.columns, dtype=float)
        else:
            out[key] = df.reindex(selected_time).fillna(0.0)

    out["extreme_days"] = {
        "single_timestep": str(selected_time[0]),
    }

    return out

def compact_summary(result) -> dict:
    """
    Keep only the metrics we need for first validation.
    Missing fields become None.
    """
    d = result.summary_dict()

    keys = [
        "scenario_id",
        "network_id",
        "n_timesteps",
        "n_converged",
        "n_violation_steps",
        "total_overvoltage_bus_steps",
        "total_undervoltage_bus_steps",
        "total_overloaded_line_steps",
        "total_overloaded_trafo_steps",
        "max_vm_pu",
        "min_vm_pu",
        "max_line_loading_pct",
        "max_trafo_loading_pct",
        "q_total_mvar_abs",
        "curtailment_steps",
        "curtailed_energy_mwh",
        "elapsed_s",
    ]

    return {k: d.get(k, None) for k in keys}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    logging.getLogger("sensitivity_coordinator").setLevel(logging.ERROR)
    logging.getLogger("pandapower").setLevel(logging.WARNING)

    print(f"\nLoading network: {NET_ID}")
    net_base = sb.get_simbench_net(NET_ID)

    print("Building annual profiles...")
    profiles_full = build_annual_profiles(
        net_base,
        NET_ID,
        simbench_code=NET_ID,
    )

    test_day = choose_test_day(profiles_full)
    profiles_day = slice_profiles_one_day(profiles_full, test_day)
    profiles_one = slice_profiles_one_timestep(profiles_day,step=0)

    print(f"\nRunning one-day scenario test: {test_day}")
    print(f"Timesteps: {len(profiles_day['times'])}")

    results = []

    print("\n[1/3] Scenario 1 baseline")
#    r1 = run_scenario_1(
#        copy.deepcopy(net_base),
#        profiles_day,
#        network_id=NET_ID,
#    )
#    results.append(compact_summary(r1))

    print("\n[2/3] Scenario 4 Volt-Var dry-run")
#    r4 = run_scenario_4(
#        copy.deepcopy(net_base),
#        profiles_day,
#        network_id=NET_ID,
#        dry_run=True,
#    )
#    results.append(compact_summary(r4))

    print("\n[3/3] Scenario 5 OPF")
    r5 = run_scenario_5(
        copy.deepcopy(net_base),
        profiles_one,
        network_id=NET_ID,
        verbose_opf=True,
        opf_init="flat",
        line_limit_percent=None,
        trafo_limit_percent=None,
        debug_opf_task=True,
    )
    results.append(compact_summary(r5))

    df = pd.DataFrame(results)

    print("\n=== Short-slice comparison ===")
    print(df.to_string(index=False))

    out_path = PROJECT_ROOT / "scenario_short_test_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved summary: {out_path}")


if __name__ == "__main__":
    main()