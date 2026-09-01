#!/usr/bin/env python3
"""
analyze_timing.py
==================
Validation + analysis of per-timestep t_total_ms / hil_latency_ms data from
publisher scenario JSON files.

Usage
-----
    python3 analyze_timing.py --rpi-dir <dir> --laptop-dir <dir> [--out report.txt]

Each directory should contain the five scenario JSON files:
    baseline.json, oltc.json, svc.json, volt_var_local.json, volt_var_coord.json
(laptop dir files may instead be suffixed _dry_run.json — both naming
conventions are auto-detected).

What this script does
----------------------
VALIDATION (must pass before committing to a full annual run):
    1. Field presence       — t_total_ms exists in every record
    2. Coverage count       — non-null count == n_timesteps (or documented gaps)
    3. Sanity bounds        — no negative / zero / absurd-outlier values
    4. Scenario 4 ordering  — t_total_ms >= hil_latency_ms at every timestep
    5. OLTC tap correlation — tap_changed=True steps show elevated t_total_ms
    6. No regressions       — other known fields still present / well-formed

ANALYSIS (once validation passes):
    A. Per-scenario t_total_ms distribution (mean/std/min/max/percentiles)
    B. Scenario 4 decomposition: t_total_ms - hil_latency_ms
    C. HIL vs dry-run t_total_ms comparison, per scenario
    D. Correlation with control activity (OLTC tap, SVC saturation,
       Scenario 4 coordination/curtailment)
    E. Outlier scan (single slowest timesteps) + temporal trend check
       (is t_total_ms drifting upward over the run, not just spiking?)
    F. One-line annual-representativeness caveat (1-month window limits)

Notes on honesty of output
---------------------------
This script reports what is actually in the data. It does not invent
thresholds dressed up as ground truth — "sanity bounds" and "elevated" are
statistical heuristics (e.g. IQR-based outlier fences, percentile gaps),
flagged as such in the printed output, not hard pass/fail engineering specs
unless the check is structurally guaranteed (e.g. check 4, which must hold
by construction since hil_latency_ms is a sub-interval of t_total_ms).
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

# Scenario 4 only — fields relevant to coordination/curtailment activity check
SCENARIO4_IDS = {"volt_var_local", "volt_var_coord"}


# ===========================================================================
# Loading
# ===========================================================================

def _find_scenario_file(directory: Path, scenario_id: str) -> Optional[Path]:
    """Locate a scenario JSON file under either naming convention."""
    candidates = [
        directory / f"{scenario_id}.json",
        directory / f"{scenario_id}_dry_run.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_all(directory: Path) -> dict[str, dict]:
    """Load all five scenario JSON files from a directory into a dict."""
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
    """Flatten a scenario's timeseries list into a DataFrame."""
    return pd.DataFrame(scenario_json["timeseries"])


# ===========================================================================
# VALIDATION CHECKS
# ===========================================================================

def check_field_presence(env_name: str, data: dict[str, dict]) -> list[str]:
    """Check 1: t_total_ms key exists in every record."""
    issues = []
    for sid, sjson in data.items():
        ts = sjson.get("timeseries", [])
        if not ts:
            issues.append(f"[{env_name}/{sid}] timeseries is empty")
            continue
        missing = sum(1 for rec in ts if "t_total_ms" not in rec)
        if missing:
            issues.append(
                f"[{env_name}/{sid}] {missing}/{len(ts)} records missing "
                f"'t_total_ms' key entirely"
            )
    return issues


def check_coverage(env_name: str, data: dict[str, dict]) -> list[str]:
    """Check 2: non-null t_total_ms count == n_timesteps."""
    issues = []
    for sid, sjson in data.items():
        df = timeseries_df(sjson)
        if "t_total_ms" not in df.columns:
            continue  # already reported by check_field_presence
        n_total = len(df)
        n_nonnull = df["t_total_ms"].notna().sum()
        n_converged = int(df["converged"].sum()) if "converged" in df.columns else None
        if n_nonnull != n_total:
            note = ""
            if n_converged is not None and n_nonnull == n_converged:
                note = " (matches n_converged — null only on non-converged steps, may be expected)"
            issues.append(
                f"[{env_name}/{sid}] t_total_ms populated for {n_nonnull}/{n_total} "
                f"timesteps{note}"
            )
    return issues


def check_sanity_bounds(env_name: str, data: dict[str, dict]) -> list[str]:
    """Check 3: no negative/zero/absurd-outlier t_total_ms values."""
    issues = []
    for sid, sjson in data.items():
        df = timeseries_df(sjson)
        if "t_total_ms" not in df.columns:
            continue
        vals = df["t_total_ms"].dropna()
        if vals.empty:
            continue
        n_negative = (vals < 0).sum()
        n_zero = (vals == 0).sum()
        if n_negative:
            issues.append(f"[{env_name}/{sid}] {n_negative} negative t_total_ms values (impossible)")
        if n_zero:
            issues.append(f"[{env_name}/{sid}] {n_zero} exactly-zero t_total_ms values (suspicious)")
        # IQR-based outlier fence — heuristic, not a hard spec
        q1, q3 = vals.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            upper_fence = q3 + 10 * iqr  # generous fence: 10x IQR, not the usual 1.5x,
                                          # since legitimate multi-runpp steps (OLTC tap
                                          # rollback, SVC post-control) are expected to be
                                          # genuinely slower, not just statistical noise
            outliers = df.loc[df["t_total_ms"] > upper_fence, ["t", "t_total_ms"]] if "t" in df.columns else None
            n_extreme = (vals > upper_fence).sum()
            if n_extreme:
                t_min = df["t"].min() if "t" in df.columns else None
                n_first_step = 0
                if outliers is not None and t_min is not None:
                    n_first_step = int((outliers["t"] == t_min).sum())
                n_other = n_extreme - n_first_step
                note = ""
                if n_first_step:
                    note = (
                        f" — includes t={t_min} (likely first-call warm-up, not a bug); "
                        f"{n_other} OTHER outlier(s) beyond t={t_min} warrant a closer look"
                        if n_other else f" — this is just t={t_min} (likely first-call warm-up, not a bug)"
                    )
                issues.append(
                    f"[{env_name}/{sid}] {n_extreme} values exceed 10x-IQR fence "
                    f"({upper_fence:.2f} ms) — max={vals.max():.2f} ms "
                    f"(heuristic flag, not necessarily a bug — inspect before annual run){note}"
                )
    return issues


def check_scenario4_ordering(env_name: str, data: dict[str, dict]) -> list[str]:
    """Check 4: t_total_ms >= hil_latency_ms at every timestep (structural)."""
    issues = []
    for sid in SCENARIO4_IDS:
        if sid not in data:
            continue
        df = timeseries_df(data[sid])
        if "t_total_ms" not in df.columns or "hil_latency_ms" not in df.columns:
            issues.append(f"[{env_name}/{sid}] cannot check ordering — missing column(s)")
            continue
        sub = df.dropna(subset=["t_total_ms", "hil_latency_ms"])
        violations = sub[sub["t_total_ms"] < sub["hil_latency_ms"]]
        if not violations.empty:
            issues.append(
                f"[{env_name}/{sid}] {len(violations)} timesteps where t_total_ms < "
                f"hil_latency_ms — STRUCTURAL VIOLATION (hil_latency_ms must be a "
                f"sub-interval of t_total_ms by construction). First offending t={violations.iloc[0]['t']}."
            )
    return issues


def check_oltc_tap_correlation(env_name: str, data: dict[str, dict]) -> list[str]:
    """Check 5 (soft): tap_changed=True steps should show elevated t_total_ms."""
    issues = []
    if "oltc" not in data:
        return issues
    df = timeseries_df(data["oltc"])
    if "t_total_ms" not in df.columns or "tap_changed" not in df.columns:
        issues.append(f"[{env_name}/oltc] cannot check tap correlation — missing column(s)")
        return issues
    sub = df.dropna(subset=["t_total_ms"])
    changed = sub[sub["tap_changed"] == True]["t_total_ms"]   # noqa: E712
    unchanged = sub[sub["tap_changed"] == False]["t_total_ms"]  # noqa: E712
    if changed.empty or unchanged.empty:
        issues.append(
            f"[{env_name}/oltc] insufficient data to compare tap-changed vs "
            f"tap-unchanged timing (changed n={len(changed)}, unchanged n={len(unchanged)})"
        )
        return issues
    if changed.mean() <= unchanged.mean():
        issues.append(
            f"[{env_name}/oltc] SOFT CHECK: tap_changed=True mean t_total_ms "
            f"({changed.mean():.3f} ms) is NOT higher than tap_changed=False mean "
            f"({unchanged.mean():.3f} ms) — expected tap actions (extra runpp calls) "
            f"to be slower. Worth a look, not necessarily a bug."
        )
    return issues


def check_no_regressions(env_name: str, data: dict[str, dict]) -> list[str]:
    """Check 6: other known fields still present and well-formed."""
    issues = []
    expected_always = ["t", "timestamp", "converged", "losses_mw", "grid_import_mw",
                        "der_gen_mw", "load_mw"]
    for sid, sjson in data.items():
        df = timeseries_df(sjson)
        for field in expected_always:
            if field not in df.columns:
                issues.append(f"[{env_name}/{sid}] expected field '{field}' is missing")
    return issues


def run_validation(env_name: str, data: dict[str, dict]) -> list[str]:
    all_issues = []
    all_issues += check_field_presence(env_name, data)
    all_issues += check_coverage(env_name, data)
    all_issues += check_sanity_bounds(env_name, data)
    all_issues += check_scenario4_ordering(env_name, data)
    all_issues += check_oltc_tap_correlation(env_name, data)
    all_issues += check_no_regressions(env_name, data)
    return all_issues


# ===========================================================================
# ANALYSIS
# ===========================================================================

def _dist_stats(vals: pd.Series) -> dict:
    if vals.empty:
        return {}
    return {
        "n": len(vals),
        "mean": vals.mean(),
        "std": vals.std(),
        "min": vals.min(),
        "max": vals.max(),
        "median": vals.median(),
        "p5": vals.quantile(0.05),
        "p25": vals.quantile(0.25),
        "p75": vals.quantile(0.75),
        "p95": vals.quantile(0.95),
        "p99": vals.quantile(0.99),
    }


def _fmt_dist(stats: dict) -> str:
    if not stats:
        return "    (no data)"
    return (
        f"    n={stats['n']}  mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
        f"min={stats['min']:.3f}  max={stats['max']:.3f}\n"
        f"    median={stats['median']:.3f}  p5={stats['p5']:.3f}  p25={stats['p25']:.3f}  "
        f"p75={stats['p75']:.3f}  p95={stats['p95']:.3f}  p99={stats['p99']:.3f}"
    )


def analysis_a_per_scenario_distribution(env_name: str, data: dict[str, dict]) -> str:
    """Reports the full distribution (every timestep, t=0 included) and,
    separately, the steady-state distribution (t>=1). Neither replaces the
    other — t=0 is a real, legitimate data point (likely first-call warm-up
    cost: Ybus/Jacobian construction, possible JIT compilation on the first
    runpp() call) and is never excluded from the full-distribution numbers.
    The steady-state breakout exists only to answer the separate question
    "what does a typical, already-warm timestep cost", without hiding the
    cold-start number anywhere."""
    lines = [f"\n--- A. Per-scenario t_total_ms distribution [{env_name}] (ms) ---"]
    for sid in SCENARIOS:
        if sid not in data:
            continue
        df = timeseries_df(data[sid])
        if "t_total_ms" not in df.columns or "t" not in df.columns:
            continue
        full_vals = df["t_total_ms"].dropna()
        if full_vals.empty:
            continue

        lines.append(f"  {sid}:")
        lines.append("    [Full distribution, all timesteps incl. t=0]")
        lines.append(_fmt_dist(_dist_stats(full_vals)))

        # First-timestep cost, called out explicitly rather than buried
        first_row = df[df["t"] == df["t"].min()]
        if not first_row.empty and pd.notna(first_row["t_total_ms"].iloc[0]):
            t0_val = first_row["t_total_ms"].iloc[0]
            steady_vals = df[df["t"] != df["t"].min()]["t_total_ms"].dropna()
            if not steady_vals.empty and steady_vals.median() > 0:
                ratio = t0_val / steady_vals.median()
                lines.append(
                    f"    First timestep (t={int(first_row['t'].iloc[0])}): "
                    f"{t0_val:.3f} ms  ({ratio:.1f}x steady-state median)"
                )
            else:
                lines.append(f"    First timestep (t={int(first_row['t'].iloc[0])}): {t0_val:.3f} ms")

        # Steady-state distribution, t != min(t) — separate, not a replacement
        steady_vals = df[df["t"] != df["t"].min()]["t_total_ms"].dropna()
        if not steady_vals.empty:
            lines.append("    [Steady-state distribution, excludes first timestep only]")
            lines.append(_fmt_dist(_dist_stats(steady_vals)))
    return "\n".join(lines)


def analysis_b_scenario4_decomposition(env_name: str, data: dict[str, dict]) -> str:
    lines = [f"\n--- B. Scenario 4 decomposition: t_total_ms - hil_latency_ms [{env_name}] (ms) ---"]
    for sid in SCENARIO4_IDS:
        if sid not in data:
            continue
        df = timeseries_df(data[sid])
        if "t_total_ms" not in df.columns or "hil_latency_ms" not in df.columns:
            continue
        sub = df.dropna(subset=["t_total_ms", "hil_latency_ms"])
        if sub.empty:
            continue
        non_serial = sub["t_total_ms"] - sub["hil_latency_ms"]
        lines.append(f"  {sid} (non-serial cost = PF solves + Schur solve + dynamics):")
        lines.append(_fmt_dist(_dist_stats(non_serial)))
        lines.append(
            f"    share of t_total_ms that is serial latency: "
            f"{(sub['hil_latency_ms'].sum() / sub['t_total_ms'].sum() * 100):.1f}%"
        )
    return "\n".join(lines)


def analysis_c_hil_vs_dryrun(rpi_data: dict, laptop_data: dict) -> str:
    lines = ["\n--- C. HIL vs dry-run t_total_ms comparison (ms) ---"]
    for sid in SCENARIOS:
        if sid not in rpi_data or sid not in laptop_data:
            lines.append(f"  {sid}: missing in one environment, skipped")
            continue
        rpi_df = timeseries_df(rpi_data[sid])
        lap_df = timeseries_df(laptop_data[sid])
        if "t_total_ms" not in rpi_df.columns or "t_total_ms" not in lap_df.columns:
            continue
        rpi_vals = rpi_df["t_total_ms"].dropna()
        lap_vals = lap_df["t_total_ms"].dropna()
        if rpi_vals.empty or lap_vals.empty:
            continue
        lines.append(f"  {sid}:")
        lines.append(f"    RPi    : mean={rpi_vals.mean():.3f} ms  median={rpi_vals.median():.3f} ms")
        lines.append(f"    Laptop : mean={lap_vals.mean():.3f} ms  median={lap_vals.median():.3f} ms")
        ratio = rpi_vals.mean() / lap_vals.mean() if lap_vals.mean() else float("nan")
        lines.append(f"    RPi/Laptop ratio (mean): {ratio:.2f}x")
    return "\n".join(lines)


def analysis_d_control_activity_correlation(env_name: str, data: dict[str, dict]) -> str:
    lines = [f"\n--- D. Correlation with control activity [{env_name}] ---"]

    # OLTC: tap_changed
    if "oltc" in data:
        df = timeseries_df(data["oltc"])
        if "t_total_ms" in df.columns and "tap_changed" in df.columns:
            sub = df.dropna(subset=["t_total_ms"])
            changed = sub[sub["tap_changed"] == True]["t_total_ms"]    # noqa: E712
            unchanged = sub[sub["tap_changed"] == False]["t_total_ms"]  # noqa: E712
            lines.append("  OLTC — tap_changed vs tap_unchanged t_total_ms:")
            lines.append(f"    tap_changed=True  : n={len(changed)}  mean={changed.mean():.3f} ms" if not changed.empty else "    tap_changed=True  : n=0")
            lines.append(f"    tap_changed=False : n={len(unchanged)}  mean={unchanged.mean():.3f} ms" if not unchanged.empty else "    tap_changed=False : n=0")

    # SVC: svc_saturated
    if "svc" in data:
        df = timeseries_df(data["svc"])
        if "t_total_ms" in df.columns and "svc_saturated" in df.columns:
            sub = df.dropna(subset=["t_total_ms"])
            sat = sub[sub["svc_saturated"] == True]["t_total_ms"]      # noqa: E712
            unsat = sub[sub["svc_saturated"] == False]["t_total_ms"]   # noqa: E712
            lines.append("  SVC — svc_saturated vs not t_total_ms:")
            lines.append(f"    saturated=True  : n={len(sat)}  mean={sat.mean():.3f} ms" if not sat.empty else "    saturated=True  : n=0")
            lines.append(f"    saturated=False : n={len(unsat)}  mean={unsat.mean():.3f} ms" if not unsat.empty else "    saturated=False : n=0")

    # Scenario 4: coordination_active, curtailment_needed
    for sid in SCENARIO4_IDS:
        if sid not in data:
            continue
        df = timeseries_df(data[sid])
        if "t_total_ms" not in df.columns:
            continue
        sub = df.dropna(subset=["t_total_ms"])
        lines.append(f"  {sid} — coordination_active / curtailment_needed:")
        if "coordination_active" in sub.columns:
            ca = sub[sub["coordination_active"] == True]["t_total_ms"]    # noqa: E712
            nca = sub[sub["coordination_active"] == False]["t_total_ms"]  # noqa: E712
            lines.append(f"    coordination_active=True  : n={len(ca)}  mean={ca.mean():.3f} ms" if not ca.empty else "    coordination_active=True  : n=0")
            lines.append(f"    coordination_active=False : n={len(nca)}  mean={nca.mean():.3f} ms" if not nca.empty else "    coordination_active=False : n=0")
        if "curtailment_needed" in sub.columns:
            cn = sub[sub["curtailment_needed"] == True]["t_total_ms"]     # noqa: E712
            ncn = sub[sub["curtailment_needed"] == False]["t_total_ms"]   # noqa: E712
            lines.append(f"    curtailment_needed=True   : n={len(cn)}  mean={cn.mean():.3f} ms" if not cn.empty else "    curtailment_needed=True   : n=0")
            lines.append(f"    curtailment_needed=False  : n={len(ncn)}  mean={ncn.mean():.3f} ms" if not ncn.empty else "    curtailment_needed=False  : n=0")

    return "\n".join(lines)


def analysis_e_outliers_and_trend(env_name: str, data: dict[str, dict], top_n: int = 5) -> str:
    """Outlier scan (isolated spikes) + temporal trend check (gradual drift)."""
    lines = [f"\n--- E. Outlier scan + temporal trend check [{env_name}] ---"]
    for sid in SCENARIOS:
        if sid not in data:
            continue
        df = timeseries_df(data[sid])
        if "t_total_ms" not in df.columns:
            continue
        sub = df.dropna(subset=["t_total_ms", "t"]).sort_values("t").reset_index(drop=True)
        if sub.empty:
            continue

        lines.append(f"  {sid}:")

        # --- Outlier scan: top N slowest timesteps ---
        slowest = sub.nlargest(top_n, "t_total_ms")[["t", "t_total_ms"]]
        lines.append(f"    Top {top_n} slowest timesteps:")
        for _, row in slowest.iterrows():
            extra_cols = []
            for c in ("converged", "tap_changed", "tap_attempted", "svc_saturated",
                      "coordination_active", "curtailment_needed", "n_retries"):
                if c in sub.columns:
                    val = sub.loc[sub["t"] == row["t"], c]
                    if not val.empty and pd.notna(val.iloc[0]):
                        extra_cols.append(f"{c}={val.iloc[0]}")
            extra_str = (", " + ", ".join(extra_cols)) if extra_cols else ""
            lines.append(f"      t={int(row['t'])}  t_total_ms={row['t_total_ms']:.3f}{extra_str}")

        # --- Temporal trend check: linear regression of t_total_ms vs t ---
        x = sub["t"].values.astype(float)
        y = sub["t_total_ms"].values.astype(float)
        if len(x) >= 10:
            slope, intercept = np.polyfit(x, y, 1)
            # Pearson correlation as a normalized trend-strength indicator
            corr = np.corrcoef(x, y)[0, 1] if np.std(y) > 0 else 0.0
            predicted_change_over_run = slope * (x.max() - x.min())
            pct_change = (
                (predicted_change_over_run / y.mean()) * 100 if y.mean() else float("nan")
            )
            lines.append(
                f"    Temporal trend: slope={slope:.6f} ms/timestep, "
                f"corr(t, t_total_ms)={corr:.4f}"
            )
            lines.append(
                f"      Implied drift over this run: {predicted_change_over_run:+.3f} ms "
                f"({pct_change:+.1f}% relative to mean)"
            )
            # Heuristic interpretation — flagged as heuristic, not asserted as fact
            if abs(corr) > 0.3:
                direction = "increasing" if slope > 0 else "decreasing"
                lines.append(
                    f"      FLAG: |corr| > 0.3 — t_total_ms shows a {direction} trend "
                    f"over the run, not just isolated spikes. Worth investigating before "
                    f"the annual run (e.g. memory growth, accumulating state, thermal "
                    f"throttling on RPi)."
                )
            else:
                lines.append(
                    "      No strong linear trend detected (|corr| <= 0.3) — variation "
                    "looks more like noise/isolated spikes than systematic drift."
                )

            # Rolling-window check: split run into 4 quartile blocks, compare means
            sub["_block"] = pd.qcut(sub["t"], 4, labels=False, duplicates="drop")
            block_means = sub.groupby("_block")["t_total_ms"].mean()
            if len(block_means) == 4:
                lines.append(
                    "      Quartile-block means (chronological): "
                    + ", ".join(f"Q{i+1}={v:.3f}" for i, v in enumerate(block_means))
                )
                if block_means.iloc[-1] > 1.5 * block_means.iloc[0]:
                    lines.append(
                        "      FLAG: last quartile mean is >1.5x first quartile mean — "
                        "possible progressive slowdown within this run."
                    )
        else:
            lines.append("    (too few points for trend analysis)")

    return "\n".join(lines)


def analysis_f_annual_caveat(env_name: str, data: dict[str, dict]) -> str:
    lines = [f"\n--- F. Run coverage [{env_name}] ---"]
    n_steps = None
    dt_min = None
    span_days = None

    for sid in SCENARIOS:
        if sid not in data:
            continue
        ts_list = data[sid].get("timeseries", [])
        if not ts_list:
            continue
        n_steps = len(ts_list)
        # Infer resolution from the gap between first two timestamps
        if len(ts_list) >= 2:
            try:
                t0 = pd.to_datetime(ts_list[0]["timestamp"], utc=True)
                t1 = pd.to_datetime(ts_list[1]["timestamp"], utc=True)
                dt_min = (t1 - t0).total_seconds() / 60.0
            except Exception:
                dt_min = None
        # Infer span from first and last timestamp
        if len(ts_list) >= 2:
            try:
                t_first = pd.to_datetime(ts_list[0]["timestamp"], utc=True)
                t_last  = pd.to_datetime(ts_list[-1]["timestamp"], utc=True)
                span_days = (t_last - t_first).total_seconds() / 86400.0
            except Exception:
                span_days = None
        break

    # Build a human-readable description of the run window
    if n_steps and dt_min and span_days:
        res_str = f"{dt_min:.0f}-min resolution"
        if span_days >= 350:
            window_str = f"full year ({span_days:.0f} days)"
            caveat = (
                "All seasons are represented. Distribution shapes here are "
                "as complete as this dataset allows."
            )
        elif span_days >= 25:
            window_str = f"~{span_days:.0f} days"
            caveat = (
                "Seasonal variation in PV/wind profiles could change "
                "coordination-iteration frequency (Scenario 4) and "
                "tap-action frequency (Scenario 2) in ways this window "
                "cannot fully capture. Treat distribution shapes here as "
                "indicative, not final, until a full annual run completes."
            )
        else:
            window_str = f"{span_days:.1f} days"
            caveat = (
                "This is a short window. Seasonal effects are not represented "
                "at all. Treat all statistics as provisional."
            )
        lines.append(
            f"  This run covers {n_steps} timesteps "
            f"({window_str}, {res_str}). {caveat}"
        )
    else:
        lines.append(
            f"  This run covers {n_steps if n_steps else '?'} timesteps "
            f"(resolution and span could not be inferred from timestamp fields)."
        )
    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rpi-dir", type=Path, default=None, help="Directory with RPi/HIL scenario JSONs")
    parser.add_argument("--laptop-dir", type=Path,default=None, help="Directory with laptop dry-run scenario JSONs")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to also write the report to a text file")
    args = parser.parse_args()

    report_lines = []

    def emit(s: str):
        print(s)
        report_lines.append(s)

    emit("=" * 78)
    emit("TIMING DATA VALIDATION + ANALYSIS")
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

    # ---------------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------------
    emit("\n" + "=" * 78)
    emit("VALIDATION CHECKS")
    emit("=" * 78)

    rpi_issues = run_validation("RPi", rpi_data)
    laptop_issues = run_validation("Laptop", laptop_data)
    all_issues = rpi_issues + laptop_issues

    if not all_issues:
        emit("\n  All validation checks passed with no issues found.")
    else:
        emit(f"\n  {len(all_issues)} issue(s) found:\n")
        for issue in all_issues:
            emit(f"  - {issue}")

    structural_failures = [i for i in all_issues if "STRUCTURAL VIOLATION" in i]
    if structural_failures:
        emit(
            "\n  *** STRUCTURAL VIOLATIONS DETECTED — these indicate a real wiring "
            "bug, not noise. Recommend fixing before the annual run. ***"
        )

    # ---------------------------------------------------------------------
    # ANALYSIS
    # ---------------------------------------------------------------------
    emit("\n" + "=" * 78)
    emit("ANALYSIS")
    emit("=" * 78)

    emit(analysis_a_per_scenario_distribution("RPi", rpi_data))
    emit(analysis_a_per_scenario_distribution("Laptop", laptop_data))

    emit(analysis_b_scenario4_decomposition("RPi", rpi_data))
    emit(analysis_b_scenario4_decomposition("Laptop", laptop_data))

    emit(analysis_c_hil_vs_dryrun(rpi_data, laptop_data))

    emit(analysis_d_control_activity_correlation("RPi", rpi_data))
    emit(analysis_d_control_activity_correlation("Laptop", laptop_data))

    emit(analysis_e_outliers_and_trend("RPi", rpi_data))
    emit(analysis_e_outliers_and_trend("Laptop", laptop_data))

    emit(analysis_f_annual_caveat("RPi", rpi_data))

    emit("\n" + "=" * 78)
    emit("END OF REPORT")
    emit("=" * 78)

    if args.out:
        args.out.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"\n[Report also written to {args.out}]")


if __name__ == "__main__":
    main()