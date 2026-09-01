"""
hiltest/sections/volt_var.py
=============================
Section: volt_var_control

Subsections
-----------
[1/4]  QVCharacteristic unit tests
[2/4]  VoltVarController dry_run  — 9 representative networks
[2b/4] Weather-driven (extreme days only)
[3/4]  VoltVarController dry_run  — all 199+ networks
[4/4]  Hardware (Arduino) — 9 representative networks

Changes from original test_suite.py (blockers applied)
--------------------------------------------------------
B1  Pre-PF non-convergence → FAIL not SKIP.
    Previously: tc.record("pre_converged", False) then tc.skipped = True.
    Now: tc.record("pre_converged", False) and return normally so the failed
    check propagates through TestCase.passed.

B4  Weather-driven load q_mvar preserves power factor.
    Previously: net.load.q_mvar = 0.0 (unity PF, understates reactive stress).
    Now: net.load.q_mvar scaled proportionally with p_mw (same ratio).

B4b Weather-driven timeseries alignment uses .reindex() not .loc[ts].
    Previously: pv_day.loc[ts] raised KeyError on timestamp mismatch.
    Now: pv and wind frames are reindexed to load_day.index before the loop.

Summary header now imports stress_description() from stress.py so it stays
in sync with the actual applied stress rather than being hardcoded.

Hardware timing assertion uses HIL_MAX_EXCHANGE_MS from constants.py.
"""

import traceback
import time
import warnings

import numpy as np
import pandas as pd
import pandapower.networks as pn
import simbench as sb

from hiltest.framework    import TestCase, print_case
from hiltest.catalogues import (
    ALL_KERBER_CASES, ALL_SYNTHETIC_LV_CASES,
    ALL_DICKERT_CASES, IN_SCOPE_SIMBENCH_CODES,
)
from hiltest.constants  import (
    DWD_DATA_DIR, HIL_MAX_EXCHANGE_MS,
)
from hiltest.networks     import get_representative_networks
from hiltest.stress       import (
    apply_overvoltage_stress, apply_hw_synthetic_stress,
    stress_description,
)
from hiltest.runpp_utils import (
    run_controller_until_converged,
    runpp_candidates,
)


# ---------------------------------------------------------------------------
# Summary printers
# ---------------------------------------------------------------------------

def _print_volt_var_summary(records: list, top_n: int = 10) -> None:
    """Ranked tables of Q(V) control effectiveness across networks."""
    valid = [r for r in records if r.get("n_ders") is not None]
    if not valid:
        print("\n  [volt_var summary] No valid results.")
        return

    W = 74
    print(f"\n{'='*W}")
    print("  VOLT-VAR CONTROL SUMMARY")
    print(stress_description())    # single source of truth — not hardcoded
    print(f"{'='*W}")

    def _table(title, rows, val_key, val_fmt=".4f", val_label="value",
               reverse=True, filter_fn=None):
        filtered = [r for r in rows if r.get(val_key) is not None]
        if filter_fn:
            filtered = [r for r in filtered if filter_fn(r)]
        if not filtered:
            print(f"\n  {title}: no data.")
            return
        ranked = sorted(
            filtered,
            key=lambda r: (r[val_key] if r[val_key] == r[val_key] else 0.0),
            reverse=reverse,
        )
        n = min(top_n, len(ranked))
        print(f"\n  Top {n} -- {title}:")
        print(f"  {'#':<4} {'Network':<42} {val_label:>10} {'DERs':>6}")
        print(f"  {'-'*W}")
        for i, r in enumerate(ranked[:n], 1):
            print(f"  {i:<4} {r['name']:<42} "
                  f"{r[val_key]:>10{val_fmt}} "
                  f"{r.get('n_ders') or 0:>6}")

    _table("Most overvoltage buses pre-control",
           valid, "n_ov_pre", val_fmt="d", val_label="buses OV", reverse=True)
    _table("Highest total |Q| applied (MVAr)",
           valid, "q_total", val_fmt=".3f", val_label="|Q| total", reverse=True)
    _table("Highest single DER |Q| applied (MVAr)",
           valid, "q_max", val_fmt=".3f", val_label="|Q| max", reverse=True)

    n_resolved  = sum(1 for r in valid if r.get("violations_resolved"))
    n_reduced   = sum(1 for r in valid
                      if r.get("v_reduced") and not r.get("violations_resolved"))
    n_no_effect = sum(1 for r in valid
                      if r.get("n_ov_pre")
                      and not r.get("v_reduced")
                      and not r.get("violations_resolved"))
    n_no_viol   = sum(1 for r in valid if not r.get("n_ov_pre"))
    n_failed    = sum(1 for r in records if r.get("n_ders") is None)

    print("\n  Voltage violation outcomes:")
    print(f"  {'No voltage violations pre-control:':<42} {n_no_viol:>4} / {len(valid)}")
    print(f"  {'Fully resolved by Q(V):':<42} {n_resolved:>4} / {len(valid)}")
    print(f"  {'Partially reduced by Q(V):':<42} {n_reduced:>4} / {len(valid)}")
    print(f"  {'No effect (DERs in deadband or no headroom):':<42} {n_no_effect:>4} / {len(valid)}")
    print(f"  {'Failed (PF non-convergence):':<42} {n_failed:>4} / {len(records)}")
    print(f"{'='*W}\n")


def _print_weather_vv_summary(records: list, top_n: int = 10) -> None:
    """Ranked tables for weather-driven extreme-day results."""
    if not records:
        print("\n  [weather summary] No records.")
        return

    W = 88
    print(f"\n{'='*W}")
    print("  WEATHER-DRIVEN VOLT-VAR SUMMARY (extreme days only)")
    print(f"{'='*W}")

    def _safe_key(r, key, default=-1):
        v = r.get(key)
        return v if isinstance(v, (int, float)) else default

    ranked = sorted(records, key=lambda r: _safe_key(r, "total_reduction", -1),
                    reverse=True)
    print(f"\n  Top {min(top_n, len(ranked))} — Biggest total violation reduction:")
    print(f"  {'#':<3} {'Network':<28} {'Day':<10} {'Pre':>6} {'Post':>6} {'Δ':>6} {'ms':>8}")
    print(f"  {'-'*W}")
    for i, r in enumerate(ranked[:top_n], 1):
        print(f"  {i:<3} {r['name']:<28} {r['day']:<10} "
              f"{r['total_pre']:>6} {r['total_post']:>6} {r['total_reduction']:>6} "
              f"{r['t_ms']:>8.1f}")

    ranked_post = sorted(records,
                         key=lambda r: _safe_key(r, "total_post", -1), reverse=True)
    print(f"\n  Top {min(top_n, len(ranked_post))} — Most violations remaining post-control:")
    print(f"  {'#':<3} {'Network':<28} {'Day':<10} {'Post':>6} {'OV':>4} {'UV':>4} {'ms':>8}")
    print(f"  {'-'*W}")
    for i, r in enumerate(ranked_post[:top_n], 1):
        print(f"  {i:<3} {r['name']:<28} {r['day']:<10} "
              f"{r['total_post']:>6} {r['ov_post']:>4} {r['uv_post']:>4} "
              f"{r['t_ms']:>8.1f}")

    n = len(records)
    n_reduced = sum(1 for r in records if r["total_reduction"] > 0)
    mean_ms   = sum(r["t_ms"] for r in records) / n
    print(f"\n  {'Records:':<25} {n}")
    print(f"  {'Days with reduction:':<25} {n_reduced}")
    print(f"  {'Mean per-day runtime:':<25} {mean_ms:.1f} ms")


# ---------------------------------------------------------------------------
# Section entry point
# ---------------------------------------------------------------------------

def run_volt_var_tests(
        verbose:      bool = False,
        only:         list = None,
        arduino_port: str  = None,
        only_hw:      bool = False,
) -> list:
    """
    Volt-Var Q(V) control tests.

    Run hardware:
        python -m hiltest --section volt_var_control --arduino-port /dev/ttyACM0
    """
    from volt_var_controller import (
        QVCharacteristic, VoltVarController,
        ArduinoSerialInterface,
        U1_PU, U2_PU, U3_PU, U4_PU, Q_RATIO,
    )

    cases         = []
    vv_rep_records = []

    # Build representative network list here — deferred import of pn/sb
    REPRESENTATIVE_NETWORKS = get_representative_networks()

    if not only_hw:

        # -------------------------------------------------------------------
        # [1/4]  QVCharacteristic unit tests
        # -------------------------------------------------------------------
        print("\n  [1/4] QVCharacteristic unit tests")
        name = "qv_characteristic"
        if not (only and not any(s in name for s in only)):
            tc = TestCase(name)
            t0 = time.perf_counter()
            try:
                qv    = QVCharacteristic
                p     = 1.0
                q_max = Q_RATIO * p
                tol   = 1e-9

                tc.record("sat_inject_at_U1",
                    abs(qv.compute_setpoint(U1_PU, p) - q_max) < tol,
                    f"got {qv.compute_setpoint(U1_PU, p):.9f}")
                tc.record("sat_inject_below_U1",
                    abs(qv.compute_setpoint(0.90, p) - q_max) < tol,
                    f"got {qv.compute_setpoint(0.90, p):.9f}")
                tc.record("deadband_at_U2",
                    abs(qv.compute_setpoint(U2_PU, p)) < tol,
                    f"got {qv.compute_setpoint(U2_PU, p):.9f}")
                tc.record("deadband_centre",
                    abs(qv.compute_setpoint(1.00, p)) < tol,
                    f"got {qv.compute_setpoint(1.00, p):.9f}")
                tc.record("deadband_at_U3",
                    abs(qv.compute_setpoint(U3_PU, p)) < tol,
                    f"got {qv.compute_setpoint(U3_PU, p):.9f}")
                tc.record("sat_absorb_at_U4",
                    abs(qv.compute_setpoint(U4_PU, p) + q_max) < tol,
                    f"got {qv.compute_setpoint(U4_PU, p):.9f}")
                tc.record("sat_absorb_above_U4",
                    abs(qv.compute_setpoint(1.10, p) + q_max) < tol,
                    f"got {qv.compute_setpoint(1.10, p):.9f}")
                mid_inj = (U1_PU + U2_PU) / 2.0
                tc.record("inject_ramp_midpoint",
                    abs(qv.compute_setpoint(mid_inj, p) - q_max / 2.0) < tol,
                    f"got {qv.compute_setpoint(mid_inj, p):.9f}")
                mid_abs = (U3_PU + U4_PU) / 2.0
                tc.record("absorb_ramp_midpoint",
                    abs(qv.compute_setpoint(mid_abs, p) + q_max / 2.0) < tol,
                    f"got {qv.compute_setpoint(mid_abs, p):.9f}")
                tc.record("zero_p_installed",
                    abs(qv.compute_setpoint(0.90, 0.0)) < tol)

                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    r_nan = qv.compute_setpoint(0.90, float("nan"))
                tc.record("nan_p_returns_zero", abs(r_nan) < tol,
                          f"got {r_nan!r}")

                vm_s = pd.Series([0.94, 1.00, 1.06], index=[10, 11, 12])
                p_s  = pd.Series([1.0,  2.0,  1.5],  index=[10, 11, 12])
                q_s  = qv.compute_setpoints(vm_s, p_s)
                tc.record("vectorised_type", isinstance(q_s, pd.Series))
                tc.record("vectorised_index",
                    list(q_s.index) == [10, 11, 12], f"got {list(q_s.index)}")
                tc.record("vectorised_signs",
                    q_s[10] > 0 and abs(q_s[11]) < tol and q_s[12] < 0,
                    f"q_s = {q_s.values}")

                vm_bad = pd.Series([0.94, 1.00], index=[0, 1])
                p_bad  = pd.Series([1.0,  2.0],  index=[1, 2])
                try:
                    qv.compute_setpoints(vm_bad, p_bad)
                    tc.record("index_mismatch_raises", False,
                              "expected ValueError, none raised")
                except ValueError:
                    tc.record("index_mismatch_raises", True)

                tc.record("slope_inject_positive", qv.slope_inject() > 0)
                tc.record("slope_absorb_negative", qv.slope_absorb() < 0)

            except Exception:
                tc.error = traceback.format_exc()
            tc.duration = time.perf_counter() - t0
            cases.append(tc)
            print_case(tc, verbose)

        # -------------------------------------------------------------------
        # [2/4]  VoltVarController dry_run — 9 representative networks
        # -------------------------------------------------------------------
        print(f"\n  [2/4] VoltVarController dry_run -- "
              f"{len(REPRESENTATIVE_NETWORKS)} representative networks")

        for net_name, loader, label in REPRESENTATIVE_NETWORKS:
            if only and not any(s in net_name for s in only):
                continue
            tc  = TestCase(f"vv_dry_{net_name}")
            t0  = time.perf_counter()
            rec = {"name": label}
            try:
                net = loader()
                apply_overvoltage_stress(net)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ctrl = VoltVarController(net, interface=None, dry_run=True)
                    ctrl.configure()

                if ctrl.n_ders == 0:
                    tc.record("skipped_no_der", True,
                              "No controllable DERs in network")
                    tc.skipped = True
                    cases.append(tc)
                    vv_rep_records.append(rec)
                    print_case(tc, verbose)
                    continue

                tc.record("n_ders_positive", ctrl.n_ders > 0,
                          f"n_ders={ctrl.n_ders}")

                # Replaced previous lv_runpp_kwargs calling with new one

                result, used_runpp_kw, tried_runpp = run_controller_until_converged(ctrl,label)
                
                tc.record(
                    "runpp_candidate selected",
                    used_runpp_kw is not None,
                    f"no runpp candidate converged; tried = {tried_runpp}",
                )

                # FIX (blocker 1): pre-PF non-convergence is a FAIL, not a SKIP.
                # Previously: tc.record(..., False) then tc.skipped = True.
                # Now: record the failure and let it propagate through passed.
                tc.record("pre_converged",  
                          result is not None and result.converged_pre,
                          f"runpp() did not converge at baseline state; tried = {tried_runpp}")
                if result is not None or not result.converged_pre:
                    # Nothing further is meaningful — append and move on.
                    cases.append(tc)
                    vv_rep_records.append(rec)
                    tc.duration = time.perf_counter() - t0
                    print_case(tc, verbose)
                    continue

                tc.record("post_converged", result.converged_post)
                tc.record("q_length",
                          len(result.q_setpoints) == ctrl.n_ders)
                tc.record("q_finite",
                          np.all(np.isfinite(result.q_setpoints.values)))
                tc.record("q_applied",
                    np.allclose(
                        net.sgen.loc[ctrl.sgen_indices, "q_mvar"].values,
                        result.q_setpoints.values,
                    ))

                rec.update({
                    "n_ders":              ctrl.n_ders,
                    "n_ov_pre":            result.report_pre.n_over_voltage,
                    "n_ov_post":           (result.report_post.n_over_voltage
                                            if result.report_post else None),
                    "worst_v_pre":         result.report_pre.worst_over_voltage,
                    "worst_v_post":        (result.report_post.worst_over_voltage
                                            if result.report_post else None),
                    "violations_resolved": result.violations_resolved,
                    "v_reduced":           result.voltage_violations_reduced,
                    "q_max":  float(np.abs(result.q_setpoints.values).max()),
                    "q_total":float(np.abs(result.q_setpoints.values).sum()),
                })
            except Exception:
                tc.error = traceback.format_exc()
                rec.update({k: None for k in [
                    "n_ders","n_ov_pre","n_ov_post","worst_v_pre","worst_v_post",
                    "violations_resolved","v_reduced","q_max","q_total",
                ]})
            tc.duration = time.perf_counter() - t0
            cases.append(tc)
            vv_rep_records.append(rec)
            print_case(tc, verbose)

        _print_volt_var_summary(vv_rep_records)

        # -------------------------------------------------------------------
        # [2b/4]  Weather-driven (extreme days only)
        # -------------------------------------------------------------------
        print("\n  [2b/4] VoltVarController weather-driven -- "
              "extreme days (9 representative networks)")

        from profile_builder import build_annual_profiles
        weather_records = []

        extreme_keys = [
            ("max_der",  "Max DER generation day"),
            ("min_der",  "Min DER generation day"),
            ("max_load", "Peak load day"),
            ("min_load", "Min load day"),
        ]

        for net_name, loader, label in REPRESENTATIVE_NETWORKS:
            if only and not any(s in net_name for s in only):
                continue
            tc = TestCase(f"vv_weather_{net_name}")
            t0 = time.perf_counter()
            try:
                net = loader()
                prof = build_annual_profiles(
                    net, label,
                    data_dir=DWD_DATA_DIR,
                    simbench_code=label if "1-" in label else None,
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ctrl = VoltVarController(net, interface=None, dry_run=True)
                    ctrl.configure()

                if ctrl.n_ders == 0:
                    tc.record("skipped_no_der", True,
                              "No controllable DERs")
                    tc.skipped = True
                    cases.append(tc)
                    print_case(tc, verbose)
                    continue

                ed = prof.get("extreme_days", {})

                # Capture base load values ONCE from the freshly loaded network,
                # before any timestep mutations. Used for power-factor
                # preservation across all four extreme days. Captured here —
                # not inside the extreme_keys loop — so that day N+1 does not
                # inherit the mutated load state from the last timestep of day N.
                base_p_mw   = net.load["p_mw"].copy()
                base_q_mvar = net.load["q_mvar"].copy()

                def _slice_day(df, day_str):
                    if df is None or df.empty:
                        return None
                    tz        = df.index.tz
                    day_start = pd.Timestamp(day_str, tz=tz)
                    day_end   = day_start + pd.Timedelta(days=1)
                    return df.loc[
                        (df.index >= day_start) & (df.index < day_end)
                    ]

                for key, desc in extreme_keys:
                    day_str  = ed.get(key)
                    if day_str is None:
                        tc.record(f"{key}_missing", False,
                                  f"{desc} not found in profiles")
                        continue

                    load_day = _slice_day(prof.get("load"), day_str)
                    if load_day is None or load_day.empty:
                        tc.record(f"{key}_load_empty", False,
                                  f"No load data for {day_str}")
                        continue

                    # Reindex PV/wind to load index once per day — not per
                    # timestep — to avoid per-step DataFrame allocation cost
                    # and KeyError on timestamp mismatch (DST, resampling).
                    pv_day   = _slice_day(prof.get("pv"),   day_str)
                    wind_day = _slice_day(prof.get("wind"), day_str)

                    if pv_day is not None and not pv_day.empty:
                        pv_day = pv_day.reindex(load_day.index, fill_value=0.0)
                    else:
                        pv_day = pd.DataFrame(
                            0.0, index=load_day.index,
                            columns=net.sgen.index,
                        )

                    if wind_day is not None and not wind_day.empty:
                        wind_day = wind_day.reindex(load_day.index, fill_value=0.0)
                    else:
                        wind_day = pd.DataFrame(
                            0.0, index=load_day.index,
                            columns=net.sgen.index,
                        )

                    t_start_day = time.perf_counter()
                    ov_pre = uv_pre = ov_post = uv_post = 0

                    for ts in load_day.index:
                        # pv_day and wind_day are already aligned — simple .loc
                        p_sgen = (
                            pv_day.loc[ts].reindex(net.sgen.index, fill_value=0.0)
                            + wind_day.loc[ts].reindex(net.sgen.index, fill_value=0.0)
                        )
                        net.sgen["p_mw"]   = p_sgen.values
                        net.sgen["q_mvar"] = 0.0

                        # fillna(0.0) handles NaN slots in DWD data before ratio
                        p_load = (
                            load_day.loc[ts]
                            .reindex(net.load.index, fill_value=0.0)
                            .fillna(0.0)
                        )
                        net.load["p_mw"] = p_load.values

                        # Power-factor preservation using np.divide with where=
                        # to avoid evaluating division on zero denominators.
                        # np.where would still evaluate both branches (UB on /0);
                        # np.divide(..., where=) writes result only where True,
                        # leaving out=ratio (initialised to 0.0) untouched elsewhere.
                        ratio = np.zeros_like(p_load.values, dtype=float)
                        np.divide(
                            p_load.values,
                            base_p_mw.values,
                            out=ratio,
                            where=np.abs(base_p_mw.values) > 1e-9,
                        )
                        net.load["q_mvar"] = base_q_mvar.values * ratio

                        result = ctrl.run_timestep()
                        if result.report_pre:
                            ov_pre += result.report_pre.n_over_voltage
                            uv_pre += result.report_pre.n_under_voltage
                        if result.report_post:
                            ov_post += result.report_post.n_over_voltage
                            uv_post += result.report_post.n_under_voltage

                    t_ms            = (time.perf_counter() - t_start_day) * 1e3
                    total_pre       = ov_pre + uv_pre
                    total_post      = ov_post + uv_post
                    total_reduction = total_pre - total_post

                    weather_records.append({
                        "name":            label,
                        "day":             day_str,
                        "key":             key,
                        "t_ms":            t_ms,
                        "ov_pre":          ov_pre,
                        "uv_pre":          uv_pre,
                        "ov_post":         ov_post,
                        "uv_post":         uv_post,
                        "total_pre":       total_pre,
                        "total_post":      total_post,
                        "total_reduction": total_reduction,
                    })

                tc.record("weather_extreme_days_complete", True)

            except Exception:
                tc.error = traceback.format_exc()

            tc.duration = time.perf_counter() - t0
            cases.append(tc)
            print_case(tc, verbose)

        _print_weather_vv_summary(weather_records)

        # -------------------------------------------------------------------
        # [3/4]  VoltVarController dry_run — all in-scope networks
        # -------------------------------------------------------------------
        print("\n  [3/4] VoltVarController dry_run -- all in-scope networks")
        vv_all_records = []

        def _run_vv_one(tc, name, loader_fn):
            try:
                net = loader_fn()
                apply_overvoltage_stress(net)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ctrl = VoltVarController(net, interface=None, dry_run=True)
                    ctrl.configure()

                if ctrl.n_ders == 0:
                    tc.record("skipped_no_der", True,
                              "No controllable DERs")
                    tc.skipped = True
                    return {
                        "name": name, "n_ders": 0,
                        "n_ov_pre": None, "n_ov_post": None,
                        "worst_v_pre": None, "worst_v_post": None,
                        "violations_resolved": None, "v_reduced": None,
                        "q_max": 0.0, "q_total": 0.0,
                    }
                
                # Replaced lv_runpp_kwargs block with a dynamic runpp algorithm block

                result, used_runpp_kw, tried_runpp = run_controller_until_converged(ctrl, name) 
                tc.record(
                    "runpp_candidate_selected",
                    used_runpp_kw is not None,
                    f"no runpp candidate converged; tried={tried_runpp}",
                )

                # Record convergence first. If pre-PF failed, accessing
                # result.report_pre.n_over_voltage would raise AttributeError.
                # Return early with a partial stats dict so the failed check
                # propagates cleanly through TestCase.passed.
                tc.record("pre_converged", 
                          result is not None and result.converged_pre,
                          f"runpp() did not converge; tried={tried_runpp}")
                if result is not None or not result.converged_pre:
                    return {"name": name, **{k: None for k in [
                        "n_ders","n_ov_pre","n_ov_post","worst_v_pre",
                        "worst_v_post","violations_resolved","v_reduced",
                        "q_max","q_total",
                    ]}}

                tc.record("q_length",
                          len(result.q_setpoints) == ctrl.n_ders)

                return {
                    "name":                name,
                    "n_ders":              ctrl.n_ders,
                    "n_ov_pre":            result.report_pre.n_over_voltage,
                    "n_ov_post":           (result.report_post.n_over_voltage
                                            if result.report_post else None),
                    "worst_v_pre":         result.report_pre.worst_over_voltage,
                    "worst_v_post":        (result.report_post.worst_over_voltage
                                            if result.report_post else None),
                    "violations_resolved": result.violations_resolved,
                    "v_reduced":           result.voltage_violations_reduced,
                    "q_max":  float(np.abs(result.q_setpoints.values).max()),
                    "q_total":float(np.abs(result.q_setpoints.values).sum()),
                }
            except Exception:
                tc.error = traceback.format_exc()
                return {"name": name, **{k: None for k in [
                    "n_ders","n_ov_pre","n_ov_post","worst_v_pre","worst_v_post",
                    "violations_resolved","v_reduced","q_max","q_total",
                ]}}

        for code in IN_SCOPE_SIMBENCH_CODES:
            if only and not any(s in code for s in only):
                continue
            tc = TestCase(f"vv_{code}")
            t0 = time.perf_counter()
            rec = _run_vv_one(tc, code, lambda c=code: sb.get_simbench_net(c))
            tc.duration = time.perf_counter() - t0
            cases.append(tc); vv_all_records.append(rec)
            print_case(tc, verbose)

        for name, loader in [
            ("cigre_mv_with_der",
             lambda: pn.create_cigre_network_mv(with_der="pv_wind")),
            ("cigre_lv", lambda: pn.create_cigre_network_lv()),
        ]:
            if only and not any(s in name for s in only):
                continue
            tc = TestCase(f"vv_{name}")
            t0 = time.perf_counter()
            rec = _run_vv_one(tc, name, loader)
            tc.duration = time.perf_counter() - t0
            cases.append(tc); vv_all_records.append(rec)
            print_case(tc, verbose)

        for name, fn_name in ALL_KERBER_CASES:
            if only and not any(s in name for s in only):
                continue
            tc = TestCase(f"vv_{name}")
            t0 = time.perf_counter()
            rec = _run_vv_one(tc, name,
                              lambda f=fn_name: getattr(pn, f)())
            tc.duration = time.perf_counter() - t0
            cases.append(tc); vv_all_records.append(rec)
            print_case(tc, verbose)

        for network_class in ALL_SYNTHETIC_LV_CASES:
            name = f"synthetic_lv_{network_class}"
            if only and not any(s in name for s in only):
                continue
            tc = TestCase(f"vv_{name}")
            t0 = time.perf_counter()
            rec = _run_vv_one(
                tc, name,
                lambda c=network_class:
                    pn.create_synthetic_voltage_control_lv_network(c),
            )
            tc.duration = time.perf_counter() - t0
            cases.append(tc); vv_all_records.append(rec)
            print_case(tc, verbose)

        for name, feeders_range, linetype, customer, case in ALL_DICKERT_CASES:
            if only and not any(s in name for s in only):
                continue
            tc = TestCase(f"vv_{name}")
            t0 = time.perf_counter()
            try:
                rec = _run_vv_one(
                    tc, name,
                    lambda fr=feeders_range, lt=linetype, cu=customer, ca=case:
                        pn.create_dickert_lv_network(fr, lt, cu, ca),
                )
            except ValueError as e:
                if "no dickert network" in str(e).lower():
                    tc.skipped = True
                rec = {"name": name, **{k: None for k in [
                    "n_ders","n_ov_pre","n_ov_post","worst_v_pre","worst_v_post",
                    "violations_resolved","v_reduced","q_max","q_total",
                ]}}
            tc.duration = time.perf_counter() - t0
            cases.append(tc); vv_all_records.append(rec)
            print_case(tc, verbose)

        _print_volt_var_summary(vv_all_records)

    # -----------------------------------------------------------------------
    # [4/4]  Hardware — 9 representative networks
    # -----------------------------------------------------------------------
    hw_label = arduino_port if arduino_port else "SKIPPED — pass --arduino-port"
    print(f"\n  [4/4] VoltVarController hardware  ({hw_label})")

    if arduino_port:
        for net_name, loader, label in REPRESENTATIVE_NETWORKS:
            if only and not any(s in net_name for s in only):
                continue
            tc = TestCase(f"vv_hw_{net_name}")
            t0 = time.perf_counter()
            try:
                net = loader()
                apply_overvoltage_stress(net)

                if "synthetic" in label.lower():
                    apply_hw_synthetic_stress(net)

                with ArduinoSerialInterface(port=arduino_port) as arduino:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ctrl = VoltVarController(net, interface=arduino,
                                                 dry_run=False)
                        ctrl.configure()

                    if ctrl.n_ders == 0:
                        tc.record("hw_no_der", True,
                                  "No controllable DERs")
                        tc.skipped = True
                        cases.append(tc)
                        print_case(tc, verbose)
                        continue

                    result = None
                    used_runpp_kw = None
                    tried_runpp = []
                    for kwargs in runpp_candidates(label):
                        tried_runpp.append(kwargs)
                        result = ctrl.run_timestep(runpp_kwargs=kwargs)
                        if result.converged_pre:
                            used_runpp_kw = kwargs
                            break
                tc.record(
                    "hw_runpp_candidate_selected",
                    used_runpp_kw is not None,
                    f"no runpp candidate converged; tried={tried_runpp}",
                )
                tc.record("hw_pre_converged",  result.converged_pre)
                tc.record("hw_post_converged", result.converged_post)
                tc.record("hw_q_length",
                          len(result.q_setpoints) == ctrl.n_ders)
                tc.record("hw_t_exchange_positive",
                          result.t_exchange_ms > 0,
                          f"t_exchange_ms={result.t_exchange_ms:.1f}")
                tc.record(
                    "hw_t_exchange_within_budget",
                    result.t_exchange_ms < HIL_MAX_EXCHANGE_MS,
                    f"t_exchange_ms={result.t_exchange_ms:.1f}ms "
                    f"exceeds HIL budget {HIL_MAX_EXCHANGE_MS:.0f}ms",
                )
                tc.record("hw_n_retries_reported", result.n_retries >= 0)
                print(f"         {label}: exchange={result.t_exchange_ms:.0f}ms  "
                      f"n_ders={ctrl.n_ders}  retries={result.n_retries}  "
                      f"resolved={result.violations_resolved}")
            except Exception:
                tc.error = traceback.format_exc()
            tc.duration = time.perf_counter() - t0
            cases.append(tc)
            print_case(tc, verbose)

    return cases
