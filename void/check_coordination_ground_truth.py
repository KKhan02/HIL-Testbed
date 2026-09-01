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

import json
from pathlib import Path
from statistics import mean, median

# ============================================================================
# EDIT THESE TWO PATHS IF THEY'VE CHANGED
# ============================================================================
RPI_RUN_DIR    = Path(r"D:\My Files\Personal Projects\HIL-Testbed\outputs (RPi)\SB 1-MV--sw-2\outputs\publisher\1-MV-rural--2-sw")
LAPTOP_RUN_DIR = Path(r"D:\My Files\Personal Projects\HIL-Testbed\outputs\Simbench 1-MV--2-sw run\publisher\1-MV-rural--2-sw")
SCENARIO_ID    = "volt_var_coord"
# ============================================================================

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


def analyse(env_name: str, records: list[dict]) -> None:
    print("\n" + "-" * 78)
    print(f"{env_name} — ground-truth check on coordination_active timesteps")
    print("-" * 78)

    active_recs = [r for r in records if r.get("coordination_active") is True]
    print(f"  Timesteps with coordination_active=True: {len(active_recs)} of {len(records)}")

    if not active_recs:
        print("  No active timesteps to check.")
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
    print(f"\n  Of {n} active timesteps with usable vm_pu_by_bus data:")
    print(f"    Real violation remained (margin < 0)                : "
          f"{real_violations} ({100*real_violations/n:.1f}%)")
    print(f"    Near the trigger's own noise floor (0 <= margin < {MIN_VIOLATION_DV})"
          f" : {near_edge} ({100*near_edge/n:.1f}%)")
    print(f"    Comfortably inside the band (margin >= {MIN_VIOLATION_DV})   : "
          f"{comfortably_safe} ({100*comfortably_safe/n:.1f}%)")

    print(f"\n  Margin distribution (pu, worst bus per timestep):")
    print(f"    mean={mean(margins):.5f}  median={median(margins):.5f}  "
          f"min={min(margins):.5f}  max={max(margins):.5f}")

    if real_violations == 0:
        print("\n  >> No real post-coordination violations found on any active timestep.")
        print("  >> The coordinator's corrections (even when triggered by a")
        print("     margin-sensitive linear prediction) are landing in a safe")
        print("     final state every time it activates.")
    else:
        print(f"\n  >> {real_violations} timestep(s) show a REAL violation persisting")
        print("     even after coordination. Dumping details below to check")
        print("     correlation with curtailment_needed / curtail_exhausted /")
        print("     q_saturated_count (i.e. was Q headroom already maxed out).")

        print(f"\n  Full detail on the {real_violations} real-violation timestep(s):")
        checked = 0
        for r in active_recs:
            m = closest_margin_to_band(r.get("vm_pu_by_bus") or {})
            if m != m or m >= 0:
                continue
            checked += 1
            print(f"    t={r.get('t')}  timestamp={r.get('timestamp')}  margin={m:.5f}")
            print(f"      q_saturated_count      = {r.get('q_saturated_count')}")
            print(f"      curtailment_needed     = {r.get('curtailment_needed')}")
            print(f"      curtail_exhausted      = {r.get('curtail_exhausted')}")
            print(f"      max_vm_pu / min_vm_pu  = {r.get('max_vm_pu')} / {r.get('min_vm_pu')}")
            print(f"      over_voltage_buses     = {r.get('over_voltage_buses')}")
            print(f"      under_voltage_buses    = {r.get('under_voltage_buses')}")
        if checked != real_violations:
            print(f"    (NOTE: dumped {checked} of {real_violations} — count mismatch, "
                  f"check for duplicate/NaN handling)")


def main() -> None:
    print("=" * 78)
    print(f"COORDINATION GROUND-TRUTH CHECK — scenario: {SCENARIO_ID}")
    print(f"Thresholds: V_BAND=[{V_BAND_LOWER}, {V_BAND_UPPER}]  "
          f"MIN_VIOLATION_DV={MIN_VIOLATION_DV}")
    print("=" * 78)

    rpi_records    = load_scenario(RPI_RUN_DIR, SCENARIO_ID)
    laptop_records = load_scenario(LAPTOP_RUN_DIR, SCENARIO_ID)

    analyse("RPi", rpi_records)
    analyse("Laptop", laptop_records)

    print("\n" + "=" * 78)
    print("END OF REPORT")
    print("=" * 78)


if __name__ == "__main__":
    main()