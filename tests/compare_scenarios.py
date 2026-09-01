#!/usr/bin/env python3
"""
compare_scenarios.py
=====================
Electrical performance comparison across all five scenarios, derived entirely
from publisher JSON files. No pandapower or simbench dependency.

Analyses
--------
C.  Extreme state timeseries
      max_vm_pu and max_line_loading_pct overlaid across all scenarios,
      showing how each control scheme affects the absolute worst-case
      operating point at each timestep. Also reports the distribution of
      the voltage envelope (how often and how far above nominal the network
      sits), not just the single annual maximum.

D.  Reactive power distribution (Scenario 4)
      Per-DER q_mvar_by_sgen statistics: which DERs are chronically at
      Q-limit (q_saturated), how much of the total reactive budget is used,
      and whether the coordination redistributes Q differently from local
      Q(V) alone. Also reports q_saturated_count distribution.

E.  Active power curtailment (Scenario 4)
      Per-DER p_mw_by_sgen compared to the uncurtailed DER profile:
      how much curtailment is concentrated on specific DERs, when in the
      year it occurs, and whether curtailment is uniformly distributed or
      falls predominantly on a few nodes.
      NOTE: this analysis requires the baseline JSON to provide the
      uncurtailed DER profile, since p_mw_by_sgen in baseline is None
      (no curtailment logic runs there). The baseline der_gen_mw is used
      as the reference uncurtailed output.

F.  Export/import balance
      grid_import_mw sign distribution across scenarios: how many timesteps
      the network is exporting (grid_import_mw < 0) vs importing, and
      whether Scenario 4 changes the net export/import balance relative to
      baseline. Also computes net annual grid exchange energy.

G.  Violation frequency map (bus-level)
      over_voltage_buses and under_voltage_buses: which specific buses
      appear most frequently across the year per scenario, producing a
      ranked "chronic violator" list. Intentionally does NOT expand
      vm_pu_by_bus (per-bus per-timestep) into a full matrix — that would
      be ~3.5M values per scenario and is not needed to answer which buses
      are most problematic.

H.  Summary comparison table
      Side-by-side scalar comparison of all key summary-block metrics
      across scenarios: total_losses_mwh, vdi, n_violation_steps,
      reactive_energy_mvarh, curtailed_energy_mwh, der_gen_mwh,
      grid_export_mwh. Losses increase and reactive energy cost
      expressed as absolute and percentage deltas vs baseline.

Usage
-----
    python3 compare_scenarios.py --rpi-dir <dir> [--laptop-dir <dir>]
                                 [--top-n-buses 10] [--top-n-ders 10]
                                 [--out report.txt]

--laptop-dir is optional; if omitted, only RPi data is analysed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SCENARIOS      = ["baseline", "oltc", "svc", "volt_var_local", "volt_var_coord"]
SCENARIO_LABEL = {
    "baseline":       "Baseline",
    "oltc":           "OLTC",
    "svc":            "SVC",
    "volt_var_local": "Volt-Var Local (4A)",
    "volt_var_coord": "Volt-Var Coord (4B)",
}
SCENARIO4 = {"volt_var_local", "volt_var_coord"}


# ===========================================================================
# Loading
# ===========================================================================

def _find(directory: Path, sid: str) -> Optional[Path]:
    for name in [f"{sid}.json", f"{sid}_dry_run.json"]:
        p = directory / name
        if p.exists():
            return p
    return None


def load_all(directory: Path) -> dict[str, dict]:
    out = {}
    for sid in SCENARIOS:
        p = _find(directory, sid)
        if p is None:
            print(f"  [WARN] {sid}: not found in {directory}")
            continue
        with open(p) as f:
            out[sid] = json.load(f)
    return out


def ts_df(sjson: dict) -> pd.DataFrame:
    df = pd.DataFrame(sjson["timeseries"])
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def infer_dt_h(sjson: dict) -> float:
    """Infer timestep duration in hours from first two records."""
    ts = sjson.get("timeseries", [])
    if len(ts) < 2:
        return 0.25   # fallback: 15-min
    try:
        t0 = pd.to_datetime(ts[0]["timestamp"], utc=True)
        t1 = pd.to_datetime(ts[1]["timestamp"], utc=True)
        return (t1 - t0).total_seconds() / 3600.0
    except Exception:
        return 0.25


# ===========================================================================
# Formatting helpers
# ===========================================================================

def _pct_delta(val: float, ref: float) -> str:
    if ref == 0 or ref is None or val is None:
        return "n/a"
    return f"{(val - ref) / abs(ref) * 100:+.1f}%"


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _header(title: str) -> str:
    return f"\n{'=' * 78}\n{title}\n{'=' * 78}"


def _sub(title: str) -> str:
    return f"\n--- {title} ---"


# ===========================================================================
# C. Extreme state timeseries
# ===========================================================================

def analysis_c(env_name: str, data: dict[str, dict]) -> str:
    lines = [_header(f"C. Extreme State Timeseries [{env_name}]")]

    lines.append(_sub("C1. max_vm_pu distribution across scenarios"))
    lines.append(
        f"  {'Scenario':<25} {'mean':>8} {'median':>8} {'p95':>8} "
        f"{'p99':>8} {'max':>8} {'> 1.05 (%)':>12} {'> 1.08 (%)':>12}"
    )
    for sid in SCENARIOS:
        if sid not in data:
            continue
        df = ts_df(data[sid])
        v = df["max_vm_pu"].dropna()
        if v.empty:
            continue
        gt105 = (v > 1.05).sum() / len(v) * 100
        gt108 = (v > 1.08).sum() / len(v) * 100
        lines.append(
            f"  {SCENARIO_LABEL[sid]:<25} "
            f"{v.mean():>8.4f} {v.median():>8.4f} "
            f"{v.quantile(0.95):>8.4f} {v.quantile(0.99):>8.4f} "
            f"{v.max():>8.4f} {gt105:>11.1f}% {gt108:>11.1f}%"
        )

    lines.append(_sub("C2. max_line_loading_pct distribution across scenarios"))
    lines.append(
        f"  {'Scenario':<25} {'mean':>8} {'median':>8} {'p95':>8} "
        f"{'p99':>8} {'max':>8} {'> 100% (steps)':>16}"
    )
    for sid in SCENARIOS:
        if sid not in data:
            continue
        df = ts_df(data[sid])
        v = df["max_line_loading_pct"].dropna()
        if v.empty:
            continue
        n_overloaded = (v > 100.0).sum()
        lines.append(
            f"  {SCENARIO_LABEL[sid]:<25} "
            f"{v.mean():>8.2f} {v.median():>8.2f} "
            f"{v.quantile(0.95):>8.2f} {v.quantile(0.99):>8.2f} "
            f"{v.max():>8.2f} {n_overloaded:>15d}"
        )

    lines.append(_sub("C3. Voltage envelope: how often does max_vm_pu exceed thresholds?"))
    thresholds = [1.02, 1.05, 1.08, 1.10]
    header = f"  {'Scenario':<25}" + "".join(f"  {f'>= {th:.2f}':>12}" for th in thresholds)
    lines.append(header)
    for sid in SCENARIOS:
        if sid not in data:
            continue
        df = ts_df(data[sid])
        v = df["max_vm_pu"].dropna()
        if v.empty:
            continue
        row = f"  {SCENARIO_LABEL[sid]:<25}"
        for th in thresholds:
            pct = (v >= th).sum() / len(v) * 100
            row += f"  {pct:>10.1f}%  "
        lines.append(row)

    return "\n".join(lines)


# ===========================================================================
# D. Reactive power distribution (Scenario 4)
# ===========================================================================

def analysis_d(env_name: str, data: dict[str, dict]) -> str:
    lines = [_header(f"D. Reactive Power Distribution — Scenario 4 [{env_name}]")]

    for sid in ["volt_var_local", "volt_var_coord"]:
        if sid not in data:
            continue
        lines.append(_sub(f"D1. {SCENARIO_LABEL[sid]} — per-DER Q statistics"))

        df = ts_df(data[sid])
        # Build a T×N DataFrame from q_mvar_by_sgen dicts
        q_records = df["q_mvar_by_sgen"].dropna()
        if q_records.empty:
            lines.append("  No q_mvar_by_sgen data.")
            continue

        q_df = pd.DataFrame(q_records.tolist(), index=q_records.index)
        q_df.columns = q_df.columns.astype(str)

        lines.append(f"  {len(q_df.columns)} DERs, {len(q_df)} timesteps with Q data")
        lines.append(
            f"  {'DER':>6} {'mean Q':>10} {'std Q':>10} {'min Q':>10} "
            f"{'max Q':>10} {'% at limit (>0.95*max)':>24}"
        )

        # Q saturation: a DER is "at limit" when |q| is within 5% of its own max
        q_abs = q_df.abs()
        q_max_per_der = q_abs.max()

        # Report top-N most saturated DERs by saturation rate, not all 98
        sat_rate = pd.Series({
            col: (q_abs[col] >= 0.95 * q_max_per_der[col]).mean() * 100
            for col in q_df.columns
            if q_max_per_der[col] > 1e-6
        })
        top_sat = sat_rate.nlargest(10)

        for der_id in top_sat.index:
            col_vals = q_df[der_id].dropna()
            lines.append(
                f"  {der_id:>6} "
                f"{col_vals.mean():>10.4f} "
                f"{col_vals.std():>10.4f} "
                f"{col_vals.min():>10.4f} "
                f"{col_vals.max():>10.4f} "
                f"{top_sat[der_id]:>22.1f}%"
            )

        lines.append(
            f"\n  Overall Q saturation: "
            f"mean q_saturated_count = {df['q_saturated_count'].dropna().mean():.1f} "
            f"of {len(q_df.columns)} DERs per timestep"
        )
        lines.append(
            f"  Total reactive energy (from summary): "
            f"{data[sid]['summary'].get('reactive_energy_mvarh', 'n/a')} MVar·h"
        )

        lines.append(_sub(f"D2. {SCENARIO_LABEL[sid]} — Q budget utilisation over time"))
        q_total_abs = q_df.abs().sum(axis=1)
        lines.append(
            f"  Total |Q| per timestep: "
            f"mean={q_total_abs.mean():.3f} MVAr  "
            f"std={q_total_abs.std():.3f}  "
            f"max={q_total_abs.max():.3f}  "
            f"min={q_total_abs.min():.3f}"
        )

        # Monthly breakdown of Q budget usage
        if "timestamp" in df.columns:
            df2 = df.copy()
            df2["q_total_abs"] = q_total_abs.reindex(df2.index).values
            df2["month"] = df2["timestamp"].dt.month
            monthly = df2.groupby("month")["q_total_abs"].mean()
            lines.append("  Monthly mean total |Q| (MVAr):")
            for month, val in monthly.items():
                lines.append(f"    Month {month:2d}: {val:.3f}")

    return "\n".join(lines)


# ===========================================================================
# E. Active power curtailment (Scenario 4)
# ===========================================================================

def analysis_e(env_name: str, data: dict[str, dict]) -> str:
    lines = [_header(f"E. Active Power Curtailment — Scenario 4 [{env_name}]")]

    dt_h = infer_dt_h(next(iter(data.values()))) if data else 0.25

    for sid in ["volt_var_local", "volt_var_coord"]:
        if sid not in data:
            continue
        lines.append(_sub(f"E1. {SCENARIO_LABEL[sid]} — curtailment overview"))

        df = ts_df(data[sid])
        summary = data[sid]["summary"]

        curtailment_steps = summary.get("curtailment_steps", 0) or 0
        curtailed_mwh = summary.get("curtailed_energy_mwh") or 0.0
        der_gen_mwh = summary.get("der_gen_mwh") or 0.0

        lines.append(f"  Curtailment steps: {curtailment_steps}")
        lines.append(f"  Curtailed energy: {curtailed_mwh:.4f} MWh")
        if der_gen_mwh > 0:
            lines.append(
                f"  Curtailment as % of total DER generation: "
                f"{curtailed_mwh / der_gen_mwh * 100:.3f}%"
            )

        # Per-DER curtailment breakdown.
        #
        # The correct uncurtailed reference for a DER at a curtailment step is
        # what that DER would naturally produce at that moment without control
        # intervention. baseline p_mw_by_sgen is None (baseline logs no per-DER
        # P), so we use each DER's own median P across NON-curtailment steps as
        # its natural operating level. On non-curtailment steps the DER runs at
        # its profile output without curtailment logic touching it, so those
        # steps are the cleanest available proxy for uncurtailed output.
        #
        # Curtailed energy per DER:
        #   sum over curtailment steps of max(0, natural_p_median - actual_p) * dt_h
        #
        # This intentionally uses median (not mean) for the natural-level
        # reference, because mean is pulled upward by the high-generation
        # curtailment steps themselves if any non-curtailment steps are
        # included in a mixed index — median is more robust to that skew.
        p_records = df["p_mw_by_sgen"].dropna()
        if p_records.empty:
            lines.append("  No p_mw_by_sgen data for per-DER breakdown.")
            continue

        p_df = pd.DataFrame(p_records.tolist(), index=p_records.index)
        p_df.columns = p_df.columns.astype(str)

        curt_steps_mask = df["curtailment_needed"] == True    # noqa: E712
        non_curt_mask   = df["curtailment_needed"] == False   # noqa: E712

        curt_idx     = df.index[curt_steps_mask]
        non_curt_idx = df.index[non_curt_mask]

        lines.append(f"\n  Per-DER curtailment breakdown (top 10 most curtailed):")

        if len(curt_idx) == 0:
            lines.append("  No curtailment steps found in timeseries.")
        elif p_df.empty:
            lines.append("  p_mw_by_sgen DataFrame is empty.")
        else:
            # Natural level: median P on non-curtailment steps
            p_non_curt = p_df.loc[p_df.index.intersection(non_curt_idx)]
            if p_non_curt.empty:
                lines.append(
                    "  All timesteps are curtailment steps — cannot derive "
                    "uncurtailed reference from non-curtailment steps."
                )
            else:
                natural_p = p_non_curt.median()   # Series: DER → natural MW level

                # Actual P on curtailment steps
                p_curt = p_df.loc[p_df.index.intersection(curt_idx)]

                # Curtailed energy per DER (MWh)
                shortfall = (natural_p - p_curt).clip(lower=0.0)  # T_curt × N
                curtailed_mwh_per_der = shortfall.sum() * dt_h

                # Number of steps each DER was genuinely curtailed below
                # its natural level (>1% shortfall to exclude numerical noise)
                n_curt_steps_per_der = (shortfall > natural_p * 0.01).sum()

                top_curtailed = curtailed_mwh_per_der.nlargest(10)

                lines.append(
                    f"  {'DER':>6} {'natural P (MW)':>16} "
                    f"{'mean P on curt steps':>22} "
                    f"{'curtailed MWh':>15} {'curt steps':>12}"
                )
                for der_id in top_curtailed.index:
                    nat   = natural_p.get(der_id, float("nan"))
                    act   = p_curt[der_id].mean() if der_id in p_curt else float("nan")
                    cmwh  = curtailed_mwh_per_der.get(der_id, 0.0)
                    nstep = int(n_curt_steps_per_der.get(der_id, 0))
                    lines.append(
                        f"  {der_id:>6} "
                        f"{nat:>16.4f} "
                        f"{act:>22.4f} "
                        f"{cmwh:>15.4f} "
                        f"{nstep:>12d}"
                    )

                total_curtailed_check = curtailed_mwh_per_der.sum()
                # The summary's curtailed_energy_mwh is computed as
                # sum over ALL converged steps of (p_target - p_applied).clip(0),
                # which includes PT1-dynamics-induced P lag on non-curtailment
                # steps as well as deliberate curtailment steps. The per-DER
                # sum here covers only the deliberate curtailment steps
                # (curtailment_needed=True) against the natural-level reference,
                # so a gap between the two is expected and is not a bug.
                lines.append(
                    f"\n  Sum of per-DER curtailed energy (curtailment steps only): "
                    f"{total_curtailed_check:.4f} MWh"
                )
                lines.append(
                    f"  Summary curtailed_energy_mwh: {curtailed_mwh:.4f} MWh "
                    f"(includes PT1-dynamics P lag on all steps, not just curtailment steps — "
                    f"gap of {abs(total_curtailed_check - curtailed_mwh):.4f} MWh is expected)"
                )

        # Monthly curtailment distribution
        if "timestamp" in df.columns and curtailment_steps > 0:
            df2 = df[curt_steps_mask].copy()
            if not df2.empty:
                df2["month"] = df2["timestamp"].dt.month
                monthly_curt = df2.groupby("month").size()
                lines.append("\n  Curtailment steps by month:")
                for month, count in monthly_curt.items():
                    lines.append(f"    Month {month:2d}: {count} steps")

    return "\n".join(lines)


# ===========================================================================
# F. Export / import balance
# ===========================================================================

def analysis_f(env_name: str, data: dict[str, dict]) -> str:
    lines = [_header(f"F. Export / Import Balance [{env_name}]")]
    dt_h = infer_dt_h(next(iter(data.values()))) if data else 0.25

    lines.append(
        "  Note: grid_import_mw < 0 means the network is NET EXPORTING "
        "(DER generation exceeds local load + losses)."
    )
    lines.append(
        f"\n  {'Scenario':<25} {'Export steps':>14} {'Import steps':>14} "
        f"{'Export %':>10} {'Net export MWh':>16} {'Net import MWh':>16}"
    )

    baseline_export_mwh = None
    for sid in SCENARIOS:
        if sid not in data:
            continue
        df = ts_df(data[sid])
        if "grid_import_mw" not in df.columns:
            continue
        g = df["grid_import_mw"].dropna()
        n_export = (g < 0).sum()
        n_import = (g >= 0).sum()
        pct_export = n_export / len(g) * 100 if len(g) else 0.0
        # Net export: sum of negative values × dt → positive number = MWh exported
        net_export_mwh = (-g[g < 0]).sum() * dt_h
        net_import_mwh = g[g >= 0].sum() * dt_h
        if sid == "baseline":
            baseline_export_mwh = net_export_mwh
        lines.append(
            f"  {SCENARIO_LABEL[sid]:<25} "
            f"{n_export:>14d} {n_import:>14d} "
            f"{pct_export:>9.1f}% "
            f"{net_export_mwh:>16.2f} "
            f"{net_import_mwh:>16.2f}"
        )

    if baseline_export_mwh is not None and baseline_export_mwh > 0:
        lines.append(f"\n  Export change vs baseline:")
        for sid in SCENARIOS:
            if sid == "baseline" or sid not in data:
                continue
            df = ts_df(data[sid])
            if "grid_import_mw" not in df.columns:
                continue
            g = df["grid_import_mw"].dropna()
            net_export_mwh = (-g[g < 0]).sum() * dt_h
            delta = net_export_mwh - baseline_export_mwh
            pct = delta / baseline_export_mwh * 100
            lines.append(
                f"    {SCENARIO_LABEL[sid]:<25}: "
                f"{delta:+.2f} MWh ({pct:+.1f}% vs baseline)"
            )

    # Monthly breakdown for baseline vs Scenario 4 variants
    lines.append(_sub("F2. Monthly net export comparison: baseline vs Scenario 4"))
    sids_to_compare = [s for s in ["baseline", "volt_var_local", "volt_var_coord"] if s in data]
    if sids_to_compare:
        # Build monthly net export table
        monthly_data: dict[str, dict[int, float]] = {}
        for sid in sids_to_compare:
            df = ts_df(data[sid])
            if "grid_import_mw" not in df.columns or "timestamp" not in df.columns:
                continue
            df2 = df.copy()
            df2["month"] = df2["timestamp"].dt.month
            df2["net_export"] = (-df2["grid_import_mw"]).clip(lower=0) * dt_h
            monthly_data[sid] = df2.groupby("month")["net_export"].sum().to_dict()

        if monthly_data:
            header = f"  {'Month':>6}" + "".join(
                f"  {SCENARIO_LABEL[s]:>22}" for s in sids_to_compare
            )
            lines.append(header)
            all_months = sorted(set(
                m for md in monthly_data.values() for m in md
            ))
            for month in all_months:
                row = f"  {month:>6}"
                for sid in sids_to_compare:
                    val = monthly_data.get(sid, {}).get(month, 0.0)
                    row += f"  {val:>22.2f}"
                lines.append(row)
            lines.append(f"  (values in MWh exported per month)")

    return "\n".join(lines)


# ===========================================================================
# G. Violation frequency map
# ===========================================================================

def analysis_g(env_name: str, data: dict[str, dict], top_n: int = 10) -> str:
    lines = [_header(f"G. Violation Frequency Map [{env_name}]")]

    for sid in SCENARIOS:
        if sid not in data:
            continue
        df = ts_df(data[sid])
        ov_counter: Counter = Counter()
        uv_counter: Counter = Counter()

        for _, row in df.iterrows():
            ov = row.get("over_voltage_buses") or []
            uv = row.get("under_voltage_buses") or []
            if isinstance(ov, list):
                ov_counter.update(ov)
            if isinstance(uv, list):
                uv_counter.update(uv)

        total_ov = sum(ov_counter.values())
        total_uv = sum(uv_counter.values())

        lines.append(_sub(f"G. {SCENARIO_LABEL[sid]}"))
        lines.append(
            f"  Total overvoltage bus-steps: {total_ov}  |  "
            f"Total undervoltage bus-steps: {total_uv}"
        )

        if total_ov > 0:
            lines.append(f"  Top {top_n} most frequently overvoltage buses:")
            lines.append(f"  {'Bus':>8}  {'Count':>8}  {'% of all OV steps':>20}")
            for bus, count in ov_counter.most_common(top_n):
                lines.append(
                    f"  {str(bus):>8}  {count:>8}  {count/total_ov*100:>19.1f}%"
                )
        else:
            lines.append("  No overvoltage bus-steps recorded.")

        if total_uv > 0:
            lines.append(f"  Top {top_n} most frequently undervoltage buses:")
            lines.append(f"  {'Bus':>8}  {'Count':>8}  {'% of all UV steps':>20}")
            for bus, count in uv_counter.most_common(top_n):
                lines.append(
                    f"  {str(bus):>8}  {count:>8}  {count/total_uv*100:>19.1f}%"
                )
        else:
            lines.append("  No undervoltage bus-steps recorded.")

    return "\n".join(lines)


# ===========================================================================
# H. Summary comparison table
# ===========================================================================

def analysis_h(env_name: str, data: dict[str, dict]) -> str:
    lines = [_header(f"H. Summary Comparison Table [{env_name}]")]

    fields = [
        ("n_violation_steps",      "Violation steps"),
        ("total_overvoltage_bus_steps", "OV bus-steps"),
        ("total_undervoltagE_bus_steps", "UV bus-steps"),  # kept even if zero
        ("vdi",                    "VDI"),
        ("total_losses_mwh",       "Total losses (MWh)"),
        ("reactive_energy_mvarh",  "Reactive energy (MVAr·h)"),
        ("curtailed_energy_mwh",   "Curtailed energy (MWh)"),
        ("der_gen_mwh",            "DER generation (MWh)"),
        ("grid_export_mwh",        "Grid export (MWh)"),
        ("max_vm_pu",              "Peak max_vm_pu"),
        ("min_vm_pu",              "Worst min_vm_pu"),
        ("max_line_loading_pct",   "Peak line loading (%)"),
        ("coordination_rate",      "Coordination rate"),
        ("q_saturation_rate",      "Q saturation rate"),
        ("elapsed_s",              "Elapsed (s)"),
    ]

    # Get baseline values for delta computation
    baseline_summary = data.get("baseline", {}).get("summary", {})

    col_width = 22
    header = f"  {'Metric':<35}" + "".join(
        f"{SCENARIO_LABEL.get(s, s):>{col_width}}" for s in SCENARIOS if s in data
    )
    lines.append(header)
    lines.append("  " + "-" * (35 + col_width * len([s for s in SCENARIOS if s in data])))

    for field_key, field_label in fields:
        row = f"  {field_label:<35}"
        for sid in SCENARIOS:
            if sid not in data:
                continue
            val = data[sid]["summary"].get(field_key)
            # Some field names had a typo above — try underscore variant
            if val is None:
                val = data[sid]["summary"].get(field_key.replace("E_", "e_"))
            row += f"{_fmt(val):>{col_width}}"
        lines.append(row)

    # Loss penalty vs baseline
    lines.append(_sub("H2. Loss penalty and reactive cost vs baseline"))
    baseline_losses = baseline_summary.get("total_losses_mwh")
    if baseline_losses:
        for sid in SCENARIOS:
            if sid == "baseline" or sid not in data:
                continue
            s_losses = data[sid]["summary"].get("total_losses_mwh")
            s_reactive = data[sid]["summary"].get("reactive_energy_mvarh")
            loss_delta = _pct_delta(s_losses, baseline_losses)
            lines.append(
                f"  {SCENARIO_LABEL[sid]:<25}: "
                f"losses {loss_delta} vs baseline"
                + (f", reactive energy = {s_reactive:.2f} MVAr·h" if s_reactive else "")
            )

    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rpi-dir",    type=Path, default=r"D:\My Files\Personal Projects\HIL-Testbed\outputs (RPi)\SB 1-MV--sw-2\outputs\publisher\1-MV-rural--2-sw\scenarios")
    parser.add_argument("--laptop-dir", type=Path, default=r"D:\My Files\Personal Projects\HIL-Testbed\outputs\Simbench 1-MV--2-sw run\publisher\1-MV-rural--2-sw\scenarios")
    parser.add_argument("--top-n-buses", type=int, default=10)
    parser.add_argument("--top-n-ders",  type=int, default=10)
    parser.add_argument("--out",         type=Path, default=None)
    args = parser.parse_args()

    report_lines = []

    def emit(s: str):
        print(s)
        report_lines.append(s)

    environments = [("RPi", args.rpi_dir)]
    if args.laptop_dir:
        environments.append(("Laptop", args.laptop_dir))

    for env_name, directory in environments:
        emit(f"\n{'#' * 78}")
        emit(f"# ENVIRONMENT: {env_name}  —  {directory}")
        emit(f"{'#' * 78}")

        data = load_all(directory)
        if not data:
            emit("  No data found, skipping.")
            continue

        emit(analysis_c(env_name, data))
        emit(analysis_d(env_name, data))
        emit(analysis_e(env_name, data))
        emit(analysis_f(env_name, data))
        emit(analysis_g(env_name, data, top_n=args.top_n_buses))
        emit(analysis_h(env_name, data))

    emit("\n" + "=" * 78)
    emit("END OF REPORT")
    emit("=" * 78)

    if args.out:
        args.out.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"\n[Report also written to {args.out}]")


if __name__ == "__main__":
    main()