"""
Run the same benchmark flow as scenario_runners/run_benchmark_script.py,
but choose the network from a dictionary for HC sweep runs.

This file intentionally does not modify run_benchmark_script.py.

Usage examples:
    python tests/test_benchmark_hc_sweep.py --network-key mv_rural
    python tests/test_benchmark_hc_sweep.py --all
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandapower.networks as pn
import pandas as pd
import simbench as sb
from rich.logging import RichHandler


# Keep import behavior aligned with scenario_runners/run_benchmark_script.py.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Command_Line_Interface"))
sys.path.insert(0, str(ROOT / "scenario_runners"))

from benchmark_runner import BenchmarkConfig, run_benchmark
from profile_builder import build_annual_profiles
from publisher import PublishHandle, publish_hc_and_comparison, publish_topology_and_profiles
import violation_detector as vd
import volt_var_controller as vvc
from test_suite import (
    ALL_DICKERT_CASES,
    ALL_KERBER_CASES,
    ALL_SYNTHETIC_LV_CASES,
    IN_SCOPE_SIMBENCH_CODES,
)


@dataclass(frozen=True)
class NetworkSpec:
    key: str
    net_name: str
    loader: Callable[[], object]
    simbench_code: str | None


def configure_logging() -> None:
    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"hc_sweep_session_{datetime.now():%Y%m%d_%H%M%S}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True, markup=False, show_path=True), file_handler],
    )


def apply_default_controller_and_violation_limits() -> None:
    # Same defaults as run_benchmark_script.py.
    vvc.set_qv_parameters(q_ratio=0.25, u1=0.96, u2=0.99, u3=1.01, u4=1.04)

    vd.V_MIN = 0.95
    vd.V_MAX = 1.05
    vd.set_limit("LINE_MAX_LOADING", None, 0.0, vd.THERMAL_LOADING_PLAUSIBLE_MAX, "%", confirm=print)
    vd.set_limit("TRAFO_MAX_LOADING", None, 0.0, vd.THERMAL_LOADING_PLAUSIBLE_MAX, "%", confirm=print)
    vd.VA_DIFF_MAX_DEGREE = 22.0
    vd.UNBALANCE_MAX_PERCENT = 2.0


def default_networks() -> dict[str, NetworkSpec]:
    networks: dict[str, NetworkSpec] = {}

    # 1) SimBench in-scope dictionary
    for code in IN_SCOPE_SIMBENCH_CODES:
        networks[code] = NetworkSpec(
            key=code,
            net_name=code,
            loader=lambda c=code: sb.get_simbench_net(c),
            simbench_code=code,
        )

    # 2) Dickert dictionary
    for name, feeders_range, linetype, customer, case in ALL_DICKERT_CASES:
        networks[name] = NetworkSpec(
            key=name,
            net_name=name,
            loader=lambda fr=feeders_range, lt=linetype, cu=customer, ca=case: pn.create_dickert_lv_network(fr, lt, cu, ca),
            simbench_code=None,
        )

    # 3) Synthetic LV dictionary
    for network_class in ALL_SYNTHETIC_LV_CASES:
        name = f"synthetic_lv_{network_class}"
        networks[name] = NetworkSpec(
            key=name,
            net_name=name,
            loader=lambda nc=network_class: pn.create_synthetic_voltage_control_lv_network(nc),
            simbench_code=None,
        )

    # 4) Kerber dictionary
    for name, fn_name in ALL_KERBER_CASES:
        if not hasattr(pn, fn_name):
            continue
        networks[name] = NetworkSpec(
            key=name,
            net_name=name,
            loader=lambda f=fn_name: getattr(pn, f)(),
            simbench_code=None,
        )

    return networks


def print_hc_gain_summary(results: dict[str, object]) -> None:
    rows = []
    for key, result in results.items():
        if result.hc_results:
            hc_b, hc_v = result.hc_results
            gain = hc_v.hc_mw - hc_b.hc_mw
            rows.append((key, hc_b.hc_mw, hc_v.hc_mw, gain))
        else:
            rows.append((key, None, None, None))

    ranked = [r for r in rows if r[3] is not None]
    ranked.sort(key=lambda r: r[3], reverse=True)

    print("\n=== HC GAIN SUMMARY (ranked) ===")
    print(f"{'Network':45s} {'Baseline MW':>12s} {'Volt-Var MW':>12s} {'Gain MW':>10s}")
    print("-" * 85)
    for key, base, vv, gain in ranked:
        print(f"{key:45s} {base:12.3f} {vv:12.3f} {gain:10.3f}")

    missing = [r for r in rows if r[3] is None]
    if missing:
        print("\nNo HC result returned for:")
        for key, _, _, _ in missing:
            print(f"  - {key}")

    out_path = ROOT / "outputs" / "benchmarks" / "hc_gain_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["network", "baseline_hc_mw", "volt_var_hc_mw", "hc_gain_mw"])
        for key, base, vv, gain in ranked:
            writer.writerow([key, base, vv, gain])
        for key, _, _, _ in missing:
            writer.writerow([key, "", "", ""])
    print(f"\nHC gain summary CSV: {out_path}")


def write_master_comparison_csv(results: dict[str, object]) -> Path:
    frames = []

    for key, result in results.items():
        df = result.comparison_df.copy()
        df.insert(0, "network", key)

        hc_baseline_mw = None
        hc_volt_var_mw = None
        hc_gain_mw = None
        if result.hc_results:
            hc_b, hc_v = result.hc_results
            hc_baseline_mw = hc_b.hc_mw
            hc_volt_var_mw = hc_v.hc_mw
            hc_gain_mw = hc_v.hc_mw - hc_b.hc_mw

        df["hc_baseline_mw"] = hc_baseline_mw
        df["hc_volt_var_mw"] = hc_volt_var_mw
        df["hc_gain_mw"] = hc_gain_mw
        df["hc_error"] = result.hc_error
        df["failed_scenarios"] = ",".join(map(str, sorted(result.errors.keys()))) if result.errors else ""

        frames.append(df)

    if frames:
        master_df = pd.concat(frames, ignore_index=True)
    else:
        master_df = pd.DataFrame()

    out_path = ROOT / "outputs" / "benchmarks" / "hc_sweep_master.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(out_path, index=False)
    return out_path


def run_one_network(spec: NetworkSpec, run_name: str = "_HC_Run"):
    oversize_factor = 1.5
    load_scale = 1.0

    net = spec.loader()
    profiles = build_annual_profiles(
        net,
        net_name=spec.net_name,
        data_dir=str(ROOT / "data" / "dwd"),
        simbench_code=spec.simbench_code,
    )

    if oversize_factor is not None:
        net_os = copy.deepcopy(net)
        net_os.sgen["p_mw"] = net_os.sgen["p_mw"] * oversize_factor
        net_os.sgen["sn_mva"] = net_os.sgen["sn_mva"] * oversize_factor
        for key in ("pv", "wind"):
            if key in profiles and not profiles[key].empty:
                common = [c for c in profiles[key].columns if c in net_os.sgen.index]
                profiles[key][common] = profiles[key][common] * oversize_factor
    else:
        net_os = net

    net_os.load["p_mw"] *= load_scale
    net_os.load["q_mvar"] *= load_scale
    profiles["load"] *= load_scale

    handle = PublishHandle(
        output_dir=str(ROOT / "outputs" / "publisher" / (spec.net_name + run_name)),
        update_every_k=6,
    )
    handle_hc = PublishHandle(
        output_dir=str(ROOT / "outputs" / "publisher" / (spec.net_name + "_hc_stressed" + run_name)),
        update_every_k=6,
    )

    config = BenchmarkConfig(
        scenarios=[],
        dry_run=True,
        write_csv=False,
        output_dir=str(ROOT / "outputs" / "benchmarks"),
        v_min=vd.V_MIN,
        v_max=vd.V_MAX,
        run_hc=True,
        run_hc_scenarios=False,
        hc_stress_scenarios=[1, 2, 3, 4, 5, 10],
        profile_factory=lambda net_hc: build_annual_profiles(
            net_hc,
            net_name=spec.net_name + "_hc_stressed" + run_name,
            data_dir=str(ROOT / "data" / "dwd"),
            simbench_code=spec.simbench_code,
        ),
        publish_fn=handle,
        hc_publish_fn=handle_hc,
    )

    publish_topology_and_profiles(
        net_os,
        profiles,
        output_dir=str(ROOT / "outputs" / "publisher" / (spec.net_name + run_name)),
        network_id=spec.net_name,
    )

    result = run_benchmark(net_os, profiles, network_id=spec.net_name, config=config)

    publish_hc_and_comparison(
        result=result,
        output_dir=str(ROOT / "outputs" / "publisher" / (spec.net_name + run_name)),
    )
    if result.hc_benchmark is not None and result.net_hc is not None:
        publish_hc_and_comparison(
            result=result.hc_benchmark,
            output_dir=str(ROOT / "outputs" / "publisher" / (spec.net_name + "_hc_stressed" + run_name)),
        )

    return result


def run_hc_sweep(network_dict: dict[str, NetworkSpec], keys: list[str]) -> dict[str, object]:
    results: dict[str, object] = {}
    for key in keys:
        if key not in network_dict:
            raise KeyError(f"Unknown network key: {key}. Available: {sorted(network_dict)}")

        spec = network_dict[key]
        print(f"\n=== Running HC sweep benchmark for: {spec.key} -> {spec.net_name} ===")
        result = run_one_network(spec)
        results[key] = result

        print(result.comparison_df[[
            "scenario_id",
            "n_violation_steps",
            "violation_duration_h",
            "vdi",
            "total_losses_mwh",
            "elapsed_s",
        ]].to_string())

        if result.hc_results:
            hc_b, hc_v = result.hc_results
            print("\nHosting Capacity:")
            print(f"  Baseline : {hc_b.hc_mw:.3f} MW")
            print(f"  Volt-Var : {hc_v.hc_mw:.3f} MW")
            print(f"  HC gain  : {hc_v.hc_mw - hc_b.hc_mw:+.3f} MW")

        if result.hc_error:
            print(f"\nHC analysis failed: {result.hc_error.splitlines()[-1]}")

        if result.errors:
            print(f"\nFailed scenarios: {sorted(result.errors.keys())}")

    print_hc_gain_summary(results)
    master_path = write_master_comparison_csv(results)
    print(f"Master run CSV: {master_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark script flow from tests, changing only network by dictionary key."
    )
    parser.add_argument(
        "--network-key",
        action="append",
        dest="network_keys",
        help="Network dictionary key to run. Can be repeated.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all entries from the network dictionary.",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    apply_default_controller_and_violation_limits()

    nets = default_networks()
    args = parse_args()

    if args.all:
        keys = list(nets.keys())
    elif args.network_keys:
        keys = args.network_keys
    else:
        keys = ["1-LV-urban6--1-sw"]

    run_hc_sweep(nets, keys)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
