#!/usr/bin/env python3
"""
diagnose_outliers.py
=====================
Follow-up diagnostic for severe t_total_ms outliers found in the annual run.

Investigates:
    1. Periodicity — do a scenario's slow timesteps fall at suspiciously
       regular intervals (e.g. every ~7100-7900 steps)?
    2. Cross-scenario clustering — do different scenarios go slow at the
       same (or nearby) timestamps, suggesting a shared cause (specific
       network/profile condition) rather than scenario-specific logic?
    3. Root-cause signature — for OLTC specifically: are the slowest
       timesteps ones where tap_attempted=False (i.e. the slowdown is in
       the base runpp() call itself, not OLTC's own control logic)?
    4. The exact-zero t_total_ms anomaly in Scenario 4 — locates the
       offending timestep(s) and dumps their full raw record alongside
       neighbouring records for context.
    5. Calendar/seasonal placement — converts slow timesteps' indices to
       actual dates (via the 'timestamp' field) to check for seasonal
       clustering (e.g. all in summer, all in winter, all at month
       boundaries).

This script does NOT assert a root cause. It surfaces patterns in the data
(timing, clustering, correlation) and reports them plainly so a human can
decide what's actually happening. Anywhere a pattern looks structural
(e.g. near-exact periodic spacing) it is reported as an observation with
the supporting numbers shown, not as a diagnosis.

Usage
-----
    python3 diagnose_outliers.py --rpi-dir <dir> --laptop-dir <dir> \\
        [--top-n 20] [--out report.txt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SCENARIOS = ["baseline", "oltc", "svc", "volt_var_local", "volt_var_coord"]
SCENARIO4_IDS = {"volt_var_local", "volt_var_coord"}

# Same IQR multiplier as analyze_timing.py, kept consistent across both
# scripts so "outlier" means the same thing in both reports.
IQR_FENCE_MULTIPLIER = 10


# ===========================================================================
# Loading (mirrors analyze_timing.py)
# ===========================================================================

def _find_scenario_file(directory: Path, scenario_id: str) -> Optional[Path]:
    candidates = [
        directory / f"{scenario_id}.json",
        directory / f"{scenario_id}_dry_run.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_all(directory: Path) -> dict[str, dict]:
    out = {}
    for sid in SCENARIOS:
        path = _find_scenario_file(directory, sid)
        if path is None:
            print(f"  [WARN] {sid}: no file found in {directory}", file=sys.stderr)
            continue
        with open(path) as f:
            out[sid] = json.load(f)
    return out


def timeseries_df(scenario_json: dict) -> pd.DataFrame:
    return pd.DataFrame(scenario_json["timeseries"])


def get_outliers(df: pd.DataFrame, exclude_t0: bool = True) -> pd.DataFrame:
    """Same IQR-fence logic as analyze_timing.py's check_sanity_bounds,
    returning the actual outlier rows rather than just a count."""
    vals = df["t_total_ms"].dropna()
    if vals.empty:
        return df.iloc[0:0]
    q1, q3 = vals.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr <= 0:
        return df.iloc[0:0]
    upper_fence = q3 + IQR_FENCE_MULTIPLIER * iqr
    out = df[df["t_total_ms"] > upper_fence].copy()
    if exclude_t0 and "t" in df.columns and not out.empty:
        t_min = df["t"].min()
        out = out[out["t"] != t_min]
    return out.sort_values("t_total_ms", ascending=False)


# ===========================================================================
# 1. Periodicity check
# ===========================================================================

def check_periodicity(env_name: str, sid: str, outliers: pd.DataFrame) -> str:
    lines = [f"  [{env_name}/{sid}] Periodicity check:"]
    if len(outliers) < 3:
        lines.append(f"    Only {len(outliers)} outlier(s) — too few to assess periodicity.")
        return "\n".join(lines)

    t_sorted = np.sort(outliers["t"].values.astype(float))
    diffs = np.diff(t_sorted)
    lines.append(f"    n={len(t_sorted)} outliers, t range [{t_sorted[0]:.0f}, {t_sorted[-1]:.0f}]")
    lines.append(
        f"    Gaps between consecutive outlier timesteps: "
        f"mean={diffs.mean():.1f}  std={diffs.std():.1f}  "
        f"min={diffs.min():.0f}  max={diffs.max():.0f}"
    )
    # Coefficient of variation of gaps — low CV implies regular spacing
    cv = diffs.std() / diffs.mean() if diffs.mean() > 0 else float("inf")
    lines.append(f"    Coefficient of variation of gaps: {cv:.3f}")
    if cv < 0.15:
        lines.append(
            "    OBSERVATION: gaps are highly regular (CV < 0.15) — outliers "
            "occur at near-evenly-spaced intervals. This pattern is consistent "
            "with a periodic cause (e.g. a recurring data/profile boundary, a "
            "fixed-interval condition in the input data), but this script does "
            "not determine the cause itself — only that the spacing is regular."
        )
    elif cv < 0.4:
        lines.append(
            "    OBSERVATION: gaps show moderate regularity (0.15 <= CV < 0.4) "
            "— a weak periodic signal may be present alongside other irregular causes."
        )
    else:
        lines.append(
            "    OBSERVATION: gaps are irregular (CV >= 0.4) — no strong evidence "
            "of periodicity; outliers look more scattered/random in time."
        )
    return "\n".join(lines)


# ===========================================================================
# 2. Cross-scenario clustering
# ===========================================================================

def check_cross_scenario_clustering(env_name: str, data: dict[str, dict],
                                     all_outliers: dict[str, pd.DataFrame],
                                     window: int = 5) -> str:
    """For each pair of scenarios, count how many outlier timesteps in one
    scenario fall within `window` steps of an outlier timestep in another.
    A high overlap suggests a shared, timestep-specific cause rather than
    independent per-scenario behavior."""
    lines = [f"\n  [{env_name}] Cross-scenario outlier clustering (within +/-{window} timesteps):"]
    sids = [s for s in SCENARIOS if s in all_outliers and not all_outliers[s].empty]
    if len(sids) < 2:
        lines.append("    Fewer than 2 scenarios have outliers — nothing to compare.")
        return "\n".join(lines)

    for i, sid_a in enumerate(sids):
        for sid_b in sids[i + 1:]:
            t_a = set(all_outliers[sid_a]["t"].values.astype(int))
            t_b = all_outliers[sid_b]["t"].values.astype(int)
            matches = 0
            matched_pairs = []
            for tb in t_b:
                nearby = [ta for ta in t_a if abs(ta - tb) <= window]
                if nearby:
                    matches += 1
                    matched_pairs.append((tb, nearby[0]))
            pct_of_b = (matches / len(t_b) * 100) if len(t_b) else 0.0
            lines.append(
                f"    {sid_a} vs {sid_b}: {matches}/{len(t_b)} of {sid_b}'s outliers "
                f"({pct_of_b:.1f}%) fall within {window} steps of an {sid_a} outlier"
            )
            if matches >= 3:
                examples = ", ".join(f"t={tb}~{ta}" for tb, ta in matched_pairs[:5])
                lines.append(f"      Example matches: {examples}")
    return "\n".join(lines)


# ===========================================================================
# 3. OLTC root-cause signature
# ===========================================================================

def check_oltc_signature(env_name: str, data: dict[str, dict],
                          outliers: pd.DataFrame) -> str:
    lines = [f"\n  [{env_name}/oltc] Root-cause signature check:"]
    if outliers.empty:
        lines.append("    No outliers to inspect.")
        return "\n".join(lines)
    if "tap_attempted" not in outliers.columns:
        lines.append("    'tap_attempted' field not present — cannot check signature.")
        return "\n".join(lines)

    n_total = len(outliers)
    n_tap_attempted = int((outliers["tap_attempted"] == True).sum())   # noqa: E712
    n_no_tap = n_total - n_tap_attempted
    lines.append(
        f"    Of {n_total} outlier timesteps: {n_tap_attempted} had tap_attempted=True, "
        f"{n_no_tap} had tap_attempted=False"
    )
    if n_no_tap > 0 and n_no_tap / n_total > 0.5:
        lines.append(
            "    OBSERVATION: majority of slow timesteps did NOT involve a tap "
            "attempt — the slowdown is occurring in the base power-flow solve "
            "itself (the single pre-action runpp() call), not in OLTC's own "
            "tap-stepping logic. This suggests the cause is shared with other "
            "scenarios' base runpp() calls, not OLTC-specific."
        )
    elif n_tap_attempted == n_total:
        lines.append(
            "    OBSERVATION: every slow timestep involved a tap attempt — "
            "consistent with OLTC's own multi-runpp() tap-stepping logic being "
            "the cause, not a shared base-solve issue."
        )
    return "\n".join(lines)


# ===========================================================================
# 4. Exact-zero anomaly
# ===========================================================================

def check_zero_anomaly(env_name: str, data: dict[str, dict]) -> str:
    lines = [f"\n  [{env_name}] Exact-zero t_total_ms anomaly check:"]
    found_any = False
    for sid in SCENARIO4_IDS:
        if sid not in data:
            continue
        df = timeseries_df(data[sid])
        if "t_total_ms" not in df.columns:
            continue
        zero_rows = df[df["t_total_ms"] == 0.0]
        if zero_rows.empty:
            continue
        found_any = True
        lines.append(f"    {sid}: {len(zero_rows)} record(s) with t_total_ms == 0.0")
        for _, row in zero_rows.iterrows():
            t_val = int(row["t"])
            lines.append(f"      t={t_val}, full record:")
            for col in df.columns:
                if col == "vm_pu_by_bus" or col.endswith("_by_bus") or col.endswith("_by_sgen"):
                    continue  # skip large nested fields for readability
                lines.append(f"        {col} = {row[col]}")
            # Show neighbouring records for context
            idx = df.index[df["t"] == t_val]
            if len(idx):
                pos = df.index.get_loc(idx[0])
                lines.append("      Neighbouring t_total_ms values (t-2..t+2):")
                for offset in range(-2, 3):
                    p = pos + offset
                    if 0 <= p < len(df):
                        lines.append(
                            f"        t={int(df.iloc[p]['t'])}: "
                            f"t_total_ms={df.iloc[p]['t_total_ms']}"
                        )
    if not found_any:
        lines.append("    None found in Scenario 4 files.")
    return "\n".join(lines)


# ===========================================================================
# 5. Calendar / seasonal placement
# ===========================================================================

def check_calendar_placement(env_name: str, sid: str, outliers: pd.DataFrame) -> str:
    lines = [f"  [{env_name}/{sid}] Calendar placement of outliers:"]
    if outliers.empty or "timestamp" not in outliers.columns:
        lines.append("    No outliers or no timestamp field — skipped.")
        return "\n".join(lines)
    try:
        ts = pd.to_datetime(outliers["timestamp"], utc=True)
    except Exception as e:
        lines.append(f"    Could not parse timestamps: {e}")
        return "\n".join(lines)
    months = ts.dt.month
    month_counts = months.value_counts().sort_index()
    lines.append("    Outlier count by month:")
    for month, count in month_counts.items():
        lines.append(f"      Month {month:2d}: {count}")
    # Flag if heavily concentrated in one or two months
    if len(month_counts) > 0:
        top_share = month_counts.max() / month_counts.sum()
        if top_share > 0.5:
            top_month = month_counts.idxmax()
            lines.append(
                f"    OBSERVATION: month {top_month} alone accounts for "
                f"{top_share*100:.0f}% of outliers — possible seasonal concentration."
            )
    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rpi-dir", type=Path, default=None, help="Directory with RPi/HIL scenario JSONs")
    parser.add_argument("--laptop-dir", type=Path, default=None, help="Directory with laptop dry-run scenario JSONs")
    parser.add_argument("--top-n", type=int, default=20, help="How many top outliers to list explicitly per scenario")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to also write the report to a text file")
    args = parser.parse_args()

    report_lines = []

    def emit(s: str):
        print(s)
        report_lines.append(s)

    emit("=" * 78)
    emit("OUTLIER DIAGNOSTIC FOLLOW-UP")
    emit("=" * 78)

    emit(f"\nLoading RPi data from: {args.rpi_dir}")
    rpi_data = load_all(args.rpi_dir)
    emit(f"Loading laptop data from: {args.laptop_dir}")
    if args.laptop_dir is not None and args.laptop_dir.exists():
        laptop_data = load_all(args.laptop_dir)
    else:
        if args.laptop_dir is not None:
            print(f"  [INFO] --laptop-dir {args.laptop_dir} not found — "
                  f"RPi-only analysis.", file=sys.stderr)
        laptop_data = {}          # RPi-only: comparison sections skip cleanly
    for env_name, data in [("RPi", rpi_data), ("Laptop", laptop_data)]:
        emit("\n" + "=" * 78)
        emit(f"ENVIRONMENT: {env_name}")
        emit("=" * 78)

        all_outliers: dict[str, pd.DataFrame] = {}

        for sid in SCENARIOS:
            if sid not in data:
                continue
            df = timeseries_df(data[sid])
            if "t_total_ms" not in df.columns:
                continue
            outliers = get_outliers(df, exclude_t0=True)
            all_outliers[sid] = outliers

            emit(f"\n--- {sid}: {len(outliers)} outlier(s) (t=0 excluded) ---")
            if outliers.empty:
                continue

            # Top-N explicit list
            top = outliers.head(args.top_n)
            cols_to_show = ["t", "t_total_ms", "timestamp"]
            for extra in ("converged", "tap_changed", "tap_attempted", "svc_saturated",
                          "coordination_active", "curtailment_needed"):
                if extra in outliers.columns:
                    cols_to_show.append(extra)
            emit(f"  Top {min(args.top_n, len(top))} outliers:")
            for _, row in top.iterrows():
                parts = [f"{c}={row[c]}" for c in cols_to_show if c in row]
                emit("    " + ", ".join(parts))

            emit("\n" + check_periodicity(env_name, sid, outliers))

            if sid == "oltc":
                emit(check_oltc_signature(env_name, data, outliers))

            emit("\n" + check_calendar_placement(env_name, sid, outliers))

        emit(check_cross_scenario_clustering(env_name, data, all_outliers))
        emit(check_zero_anomaly(env_name, data))

    emit("\n" + "=" * 78)
    emit("END OF REPORT")
    emit("=" * 78)

    if args.out:
        args.out.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"\n[Report also written to {args.out}]")


if __name__ == "__main__":
    main()
