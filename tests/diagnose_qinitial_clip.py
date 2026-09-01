"""
diagnose_qinitial_clip.py

Purpose
-------
q_initial (the raw Q(V)-curve output, pre-clip) is never persisted to the
scenario JSON — only q_applied_mvar (post-clip, i.e. q_adjusted) and the
derived coordination_active / q_saturated_count fields are. This script
uses those persisted proxies to distinguish two hypotheses for the
RPi coordination_rate=1.0000 vs Laptop coordination_rate=0.0947 gap on
volt_var_coord:

  H1 (intended mechanism): serial ASCII truncation of vm_pu on the RPi
     path pushes q_initial fractionally over +-q_max more often than the
     full-precision laptop dry-run path. Expect: q_saturated_count only
     MARGINALLY higher on RPi at matching timesteps, and q_mvar_by_sgen
     magnitudes nearly identical between RPi/laptop (agreeing to ~3-4
     decimal places) except at the specific DERs/timesteps where the clip
     bit on one side and not the other.

  H2 (residual bug / leftover mismatch): Q_RATIO (or another parameter)
     still differs between the RPi Arduino and the laptop dry-run path.
     Expect: q_mvar_by_sgen magnitudes SYSTEMATICALLY larger on RPi
     across most DERs/timesteps (e.g. ~roughly proportional to the ratio
     of old/new Q_RATIO), not just at the margin.

Usage
-----
Edit RPI_RUN_DIR and LAPTOP_RUN_DIR below to point at the two run
directories (the folder containing scenarios/volt_var_coord.json),
then run:

    python diagnose_qinitial_clip.py

No files are modified. Read-only diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Optional

import numpy as np

# Defaults match the paths used throughout this project's session history.
# Override with --rpi-dir / --laptop-dir when pointing at a different run.
DEFAULT_RPI_RUN_DIR    = r"D:\My Files\Personal Projects\HIL-Testbed\outputs (RPi)\SB 1-MV--sw-2\outputs\publisher\1-MV-rural--2-sw"
DEFAULT_LAPTOP_RUN_DIR = r"D:\My Files\Personal Projects\HIL-Testbed\outputs\Simbench 1-MV--2-sw run\publisher\1-MV-rural--2-sw"
SCENARIO_ID = "volt_var_coord"   # 4B — the scenario showing the coordination_rate gap


def load_scenario(run_dir: Path, scenario_id: str) -> list[dict]:
    path = run_dir / "scenarios" / f"{scenario_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario JSON not found: {path}")
    with open(path) as f:
        payload = json.load(f)
    # build_scenario_payload() (publisher.py) returns:
    #   {"scenario_id":..., "network_id":..., "elapsed_s":...,
    #    "summary": {...}, "timeseries": [ {t, ...}, ... ]}
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


def run(rpi_dir: Path, laptop_dir: Path, scenario_id: str = SCENARIO_ID,
        out: Optional[Path] = None) -> tuple[list[str], bool]:
    """
    Runs the Q_initial clip diagnostic and returns (report_lines, ran).
    ran=False (with a one-line explanatory message) if the scenario JSON
    is missing on either side — e.g. a run that only exercised
    Baseline/OLTC/SVC and never ran Scenario 4/5, so volt_var_coord.json
    does not exist. This lets the consolidated runner skip this
    diagnostic gracefully rather than crashing the whole suite.
    """
    report_lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        report_lines.append(s)

    emit("=" * 78)
    emit(f"Q_INITIAL CLIP DIAGNOSTIC — scenario: {scenario_id}")
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

    rpi_by_t       = index_by_t(rpi_records)
    laptop_by_t    = index_by_t(laptop_records)

    common_t = sorted(set(rpi_by_t) & set(laptop_by_t))
    emit(f"\nRPi records:    {len(rpi_records)}")
    emit(f"Laptop records: {len(laptop_records)}")
    emit(f"Common timesteps: {len(common_t)}")
    if not common_t:
        emit("No overlapping timesteps found -- cannot compare. Check paths/scenario_id.")
        if out:
            out.write_text("\n".join(report_lines), encoding="utf-8")
        return report_lines, True

    # ------------------------------------------------------------------
    # PART A — q_saturated_count comparison
    # ------------------------------------------------------------------
    emit("\n" + "-" * 78)
    emit("PART A — q_saturated_count (out of up to 98 DERs), RPi vs Laptop")
    emit("-" * 78)

    rpi_sat    = [rpi_by_t[t].get("q_saturated_count") for t in common_t]
    laptop_sat = [laptop_by_t[t].get("q_saturated_count") for t in common_t]
    rpi_sat    = [v for v in rpi_sat if v is not None]
    laptop_sat = [v for v in laptop_sat if v is not None]

    emit(f"  RPi    : mean={mean(rpi_sat):.2f}  median={median(rpi_sat):.1f}  "
          f"max={max(rpi_sat)}  min={min(rpi_sat)}")
    emit(f"  Laptop : mean={mean(laptop_sat):.2f}  median={median(laptop_sat):.1f}  "
          f"max={max(laptop_sat)}  min={min(laptop_sat)}")

    diffs = [
        rpi_by_t[t].get("q_saturated_count", 0) - laptop_by_t[t].get("q_saturated_count", 0)
        for t in common_t
        if rpi_by_t[t].get("q_saturated_count") is not None
        and laptop_by_t[t].get("q_saturated_count") is not None
    ]
    n_rpi_higher    = sum(1 for d in diffs if d > 0)
    n_laptop_higher = sum(1 for d in diffs if d < 0)
    n_equal         = sum(1 for d in diffs if d == 0)
    emit(f"\n  Per-timestep comparison (n={len(diffs)}):")
    emit(f"    RPi higher    : {n_rpi_higher} ({100*n_rpi_higher/len(diffs):.1f}%)")
    emit(f"    Laptop higher : {n_laptop_higher} ({100*n_laptop_higher/len(diffs):.1f}%)")
    emit(f"    Equal         : {n_equal} ({100*n_equal/len(diffs):.1f}%)")
    emit(f"    Mean diff (RPi - Laptop): {mean(diffs):+.3f} DERs")

    if mean(rpi_sat) > 2 * mean(laptop_sat):
        emit("\n  >> LARGE, roughly-proportional gap in saturation counts.")
        emit("  >> Consistent with H2 (residual Q_RATIO/parameter mismatch),")
        emit("     not simple truncation noise at the margin.")
    else:
        emit("\n  >> Saturation counts are in the same order of magnitude.")
        emit("  >> Consistent with H1 (truncation pushing values over the")
        emit("     clip boundary only marginally more often on RPi).")

    # ------------------------------------------------------------------
    # PART B — per-DER q_mvar magnitude comparison at common timesteps
    # ------------------------------------------------------------------
    emit("\n" + "-" * 78)
    emit("PART B — per-DER |q_mvar| magnitude comparison (q_mvar_by_sgen)")
    emit("-" * 78)

    # Sample every Nth common timestep to keep this fast on a full annual run.
    sample_every = max(1, len(common_t) // 2000)
    sample_t = common_t[::sample_every]
    emit(f"  Sampling {len(sample_t)} of {len(common_t)} common timesteps "
          f"(every {sample_every}th)")

    ratios = []          # |q_rpi| / |q_laptop| where laptop value is non-negligible
    abs_diffs = []        # |q_rpi - q_laptop|
    clip_mismatch_count = 0   # times one side is at/near a limit and the other isn't
    total_der_t_pairs = 0

    for t in sample_t:
        rpi_q    = rpi_by_t[t].get("q_mvar_by_sgen") or {}
        laptop_q = laptop_by_t[t].get("q_mvar_by_sgen") or {}
        common_ders = set(rpi_q) & set(laptop_q)
        for der in common_ders:
            qr = rpi_q[der]
            ql = laptop_q[der]
            if qr is None or ql is None:
                continue
            total_der_t_pairs += 1
            abs_diffs.append(abs(qr - ql))
            if abs(ql) > 1e-4:
                ratios.append(abs(qr) / abs(ql))

    if abs_diffs:
        emit(f"\n  |q_rpi - q_laptop| across {total_der_t_pairs} (DER, t) pairs:")
        emit(f"    mean={mean(abs_diffs):.6f}  median={median(abs_diffs):.6f}  "
              f"max={max(abs_diffs):.6f}  (MVAr)")
    if ratios:
        emit(f"\n  |q_rpi| / |q_laptop| ratio (excluding near-zero laptop values):")
        emit(f"    mean={mean(ratios):.4f}  median={median(ratios):.4f}")
        # A ratio near 1.0 => magnitudes agree (H1-consistent).
        # A ratio systematically >> 1.0 (e.g. ~1.9-2x) => proportional
        # mismatch, consistent with a leftover Q_RATIO discrepancy (H2).
        near_one = sum(1 for r in ratios if 0.9 <= r <= 1.1)
        emit(f"    fraction of ratios within +-10% of 1.0: "
              f"{100*near_one/len(ratios):.1f}%")

        if mean(ratios) > 1.5:
            emit("\n  >> Ratio is systematically >> 1.0 across most (DER, t) pairs.")
            emit("  >> Consistent with H2 (a genuine magnitude mismatch, e.g.")
            emit("     leftover Q_RATIO discrepancy), NOT truncation noise.")
        elif near_one / len(ratios) > 0.85:
            emit("\n  >> The vast majority of (DER, t) pairs agree to within +-10%.")
            emit("  >> Consistent with H1 (truncation only bites at the margin")
            emit("     for a small subset of DERs/timesteps).")
        else:
            emit("\n  >> Mixed signal -- neither cleanly H1 nor H2. Inspect the")
            emit("     largest individual |q_rpi - q_laptop| cases below by hand.")

    # ------------------------------------------------------------------
    # PART C — worst individual (DER, t) disagreements, for manual inspection
    # ------------------------------------------------------------------
    emit("\n" + "-" * 78)
    emit("PART C — top 15 largest individual |q_rpi - q_laptop| disagreements")
    emit("-" * 78)

    worst = []
    for t in sample_t:
        rpi_q    = rpi_by_t[t].get("q_mvar_by_sgen") or {}
        laptop_q = laptop_by_t[t].get("q_mvar_by_sgen") or {}
        for der in set(rpi_q) & set(laptop_q):
            qr, ql = rpi_q[der], laptop_q[der]
            if qr is None or ql is None:
                continue
            worst.append((abs(qr - ql), t, der, qr, ql))

    worst.sort(reverse=True)
    for diff, t, der, qr, ql in worst[:15]:
        rpi_sat_t    = rpi_by_t[t].get("q_saturated_count")
        laptop_sat_t = laptop_by_t[t].get("q_saturated_count")
        emit(f"  t={t:>6}  DER={der:>4}  "
              f"q_rpi={qr:+.5f}  q_laptop={ql:+.5f}  diff={diff:.5f}  "
              f"[sat_count RPi={rpi_sat_t} Laptop={laptop_sat_t}]")

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