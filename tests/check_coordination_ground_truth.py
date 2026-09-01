"""
check_coordination_ground_truth.py

Purpose
-------
The coordinator's activation decision (coordination_active) is based on a
LINEAR sensitivity-matrix prediction of vm_pu (computed from q_initial,
BEFORE any power flow), checked against a 1-millipu margin. This script
does NOT validate that pre-decision prediction directly -- that's not
reconstructable after the fact, since q_initial's hypothetical "local-only"
outcome is not what gets solved once coordination has already been applied.

Instead, this checks something adjacent and still useful: for timesteps
where the coordinator DID activate, was the REAL, converged, post-
coordination vm_pu_by_bus (already persisted in scenarios/*.json) actually
close to or inside the violation band? This tells you whether an
oversensitive trigger is at least landing near real, physically meaningful
territory, versus firing in situations where the real outcome was
comfortably safe regardless.

Uses the exact same thresholds as sensitivity_coordinator.py itself
(V_BAND_LOWER=0.95, V_BAND_UPPER=1.05, MIN_VIOLATION_DV=1e-3) so the
comparison is apples-to-apples with the coordinator's own trigger logic.

Usage
-----
Edit RPI_RUN_DIR / LAPTOP_RUN_DIR below if they've changed, then:

    python check_coordination_ground_truth.py

Read-only. No files modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Optional

DEFAULT_RPI_RUN_DIR    = r"D:\My Files\Personal Projects\HIL-Testbed\outputs (RPi)\SB 1-MV--sw-2\outputs\publisher\1-MV-rural--2-sw"
DEFAULT_LAPTOP_RUN_DIR = r"D:\My Files\Personal Projects\HIL-Testbed\outputs\Simbench 1-MV--2-sw run\publisher\1-MV-rural--2-sw"
SCENARIO_ID = "volt_var_coord"

# Same constants as sensitivity_coordinator.py — kept in sync manually since
# this is a read-only analysis script, not an import of the project module.
V_BAND_LOWER      = 0.95
V_BAND_UPPER      = 1.05
MIN_VIOLATION_DV  = 1e-3


def load_scenario(run_dir: Path, scenario_id: str) -> list[dict]:
    path = run_dir / "scenarios" / f"{scenario_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario JSON not found: {path}")
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "timeseries" in payload:
        return payload["timeseries"]
    if isinstance(payload, list):
        return payload
    raise ValueError(
        f"Unrecognised scenario JSON shape at {path}: "
        f"top-level type {type(payload)}, keys={list(payload.keys()) if isinstance(payload, dict) else 'n/a'}"
    )


def index_by_t(records: list[dict]) -> dict[int, dict]:
    return {r["t"]: r for r in records if r.get("t") is not None}


def closest_margin_to_band(vm_pu_by_bus: dict) -> float:
    """
    Smallest signed distance from the violation band, across all buses.
    Positive => inside the safe band (larger = safer margin from either edge).
    Negative => genuinely outside the band (real violation).
    Zero-ish  => sitting right at the edge.

    For each bus: margin = min(vm - V_BAND_LOWER, V_BAND_UPPER - vm)
    The timestep's overall margin is the MIN across buses (worst-case bus).
    """
    if not vm_pu_by_bus:
        return float("nan")
    margins = []
    for v in vm_pu_by_bus.values():
        if v is None:
            continue
        margins.append(min(v - V_BAND_LOWER, V_BAND_UPPER - v))
    return min(margins) if margins else float("nan")


def analyse(env_name: str, records: list[dict], emit=print) -> None:
    emit("\n" + "-" * 78)
    emit(f"{env_name} — ground-truth check on coordination_active timesteps")
    emit("-" * 78)

    active_recs = [r for r in records if r.get("coordination_active") is True]
    emit(f"  Timesteps with coordination_active=True: {len(active_recs)} of {len(records)}")

    if not active_recs:
        emit("  No active timesteps to check.")
        return

    margins = []
    real_violations = 0        # margin < 0: genuinely outside V_BAND after coordination
    near_edge        = 0        # 0 <= margin < MIN_VIOLATION_DV: within the trigger's own noise floor
    comfortably_safe = 0        # margin >= MIN_VIOLATION_DV: real state was safely inside the band

    for r in active_recs:
        m = closest_margin_to_band(r.get("vm_pu_by_bus") or {})
        if m != m:  # NaN check without importing math
            continue
        margins.append(m)
        if m < 0:
            real_violations += 1
        elif m < MIN_VIOLATION_DV:
            near_edge += 1
        else:
            comfortably_safe += 1

    n = len(margins)
    emit(f"\n  Of {n} active timesteps with usable vm_pu_by_bus data:")
    emit(f"    Real violation remained (margin < 0)                : "
          f"{real_violations} ({100*real_violations/n:.1f}%)")
    emit(f"    Near the trigger's own noise floor (0 <= margin < {MIN_VIOLATION_DV})"
          f" : {near_edge} ({100*near_edge/n:.1f}%)")
    emit(f"    Comfortably inside the band (margin >= {MIN_VIOLATION_DV})   : "
          f"{comfortably_safe} ({100*comfortably_safe/n:.1f}%)")

    emit(f"\n  Margin distribution (pu, worst bus per timestep):")
    emit(f"    mean={mean(margins):.5f}  median={median(margins):.5f}  "
          f"min={min(margins):.5f}  max={max(margins):.5f}")

    if real_violations == 0:
        emit("\n  >> No real post-coordination violations found on any active timestep.")
        emit("  >> The coordinator's corrections (even when triggered by a")
        emit("     margin-sensitive linear prediction) are landing in a safe")
        emit("     final state every time it activates.")
    else:
        emit(f"\n  >> {real_violations} timestep(s) show a REAL violation persisting")
        emit("     even after coordination. Dumping details below to check")
        emit("     correlation with curtailment_needed / curtail_exhausted /")
        emit("     q_saturated_count (i.e. was Q headroom already maxed out).")

        emit(f"\n  Full detail on the {real_violations} real-violation timestep(s):")
        checked = 0
        for r in active_recs:
            m = closest_margin_to_band(r.get("vm_pu_by_bus") or {})
            if m != m or m >= 0:
                continue
            checked += 1
            emit(f"    t={r.get('t')}  timestamp={r.get('timestamp')}  margin={m:.5f}")
            emit(f"      q_saturated_count      = {r.get('q_saturated_count')}")
            emit(f"      curtailment_needed     = {r.get('curtailment_needed')}")
            emit(f"      curtail_exhausted      = {r.get('curtail_exhausted')}")
            emit(f"      max_vm_pu / min_vm_pu  = {r.get('max_vm_pu')} / {r.get('min_vm_pu')}")
            emit(f"      over_voltage_buses     = {r.get('over_voltage_buses')}")
            emit(f"      under_voltage_buses    = {r.get('under_voltage_buses')}")
        if checked != real_violations:
            emit(f"    (NOTE: dumped {checked} of {real_violations} — count mismatch, "
                  f"check for duplicate/NaN handling)")


def run(rpi_dir: Path, laptop_dir: Path, scenario_id: str = SCENARIO_ID,
        out: Optional[Path] = None) -> tuple[list[str], bool]:
    """
    Runs the coordination ground-truth check and returns (report_lines, ran).
    ran=False if the scenario JSON is missing on either side (e.g. a run
    that never exercised Scenario 5 / volt_var_coord) — lets the
    consolidated runner skip this diagnostic gracefully.
    """
    report_lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        report_lines.append(s)

    emit("=" * 78)
    emit(f"COORDINATION GROUND-TRUTH CHECK — scenario: {scenario_id}")
    emit(f"Thresholds: V_BAND=[{V_BAND_LOWER}, {V_BAND_UPPER}]  "
         f"MIN_VIOLATION_DV={MIN_VIOLATION_DV}")
    emit("=" * 78)

    try:
        rpi_records    = load_scenario(rpi_dir, scenario_id)
        laptop_records = load_scenario(laptop_dir, scenario_id)
    except FileNotFoundError as exc:
        emit(f"\nSKIPPED — {exc}")
        emit("(This diagnostic only applies to runs that included a Volt-Var")
        emit(" Coordinated scenario. Skipping is expected for Baseline/OLTC/SVC-only runs.)")
        if out:
            out.write_text("\n".join(report_lines), encoding="utf-8")
        return report_lines, False

    analyse("RPi", rpi_records, emit=emit)
    analyse("Laptop", laptop_records, emit=emit)

    emit("\n" + "=" * 78)
    emit("END OF REPORT")
    emit("=" * 78)

    if out:
        out.write_text("\n".join(report_lines), encoding="utf-8")
    return report_lines, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rpi-dir", type=Path, default=DEFAULT_RPI_RUN_DIR,
                         help="Directory with RPi/HIL publisher output (contains scenarios/)")
    parser.add_argument("--laptop-dir", type=Path, default=DEFAULT_LAPTOP_RUN_DIR,
                         help="Directory with laptop dry-run publisher output (contains scenarios/)")
    parser.add_argument("--scenario-id", default=SCENARIO_ID)
    parser.add_argument("--out", type=Path, default=None,
                         help="Optional path to also write the report to a text file")
    args = parser.parse_args()

    run(args.rpi_dir, args.laptop_dir, args.scenario_id, out=args.out)


if __name__ == "__main__":
    main()