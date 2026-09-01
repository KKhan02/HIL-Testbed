"""
hiltest/sections/violation.py
==============================
Sections: violation_detector, violation_detector_all
Pure refactor — timing switched to perf_counter.
"""
import traceback
import time
import pandapower as pp
import pandapower.networks as pn
import simbench as sb

from hiltest.framework  import TestCase, print_case
from hiltest.catalogues  import (
    IN_SCOPE_SIMBENCH_CODES, ALL_KERBER_CASES,
    ALL_SYNTHETIC_LV_CASES, ALL_DICKERT_CASES,
)
from hiltest.networks   import get_representative_networks
from violation_detector import (
    ViolationReport, detect_violations,
    V_MAX, V_MIN, LINE_MAX_LOADING, TRAFO_MAX_LOADING,
)


def run_violation_detector_tests(verbose: bool = False,
                                  only: list = None) -> list:
    cases = []
    REPRESENTATIVE_NETWORKS = get_representative_networks()
    for test_name, loader, label in REPRESENTATIVE_NETWORKS:
        if only and not any(s in test_name for s in only):
            continue
        tc = TestCase(test_name)
        t0 = time.perf_counter()
        try:
            net    = loader()
            pp.runpp(net, voltage_depend_loads=False)
            report = detect_violations(net)

            tc.record("returns_violation_report",
                      isinstance(report, ViolationReport),
                      "detect_violations() did not return a ViolationReport")
            tc.record("converged", report.converged, "runpp() did not converge")
            tc.record("any_violations_is_bool",
                      isinstance(report.any_violations, bool),
                      "any_violations is not bool")

            for attr, col in [
                ("over_voltage",      "deviation_pu"),
                ("under_voltage",     "deviation_pu"),
                ("overloaded_lines",  "loading_percent"),
                ("overloaded_trafos", "loading_percent"),
                ("angle_violations",  "va_diff_degree"),
            ]:
                df = getattr(report, attr)
                tc.record(f"{attr}_has_columns", col in df.columns,
                    f"'{col}' missing from report.{attr} "
                    f"(columns: {list(df.columns)})")

            derived = (
                not report.over_voltage.empty
                or not report.under_voltage.empty
                or not report.overloaded_lines.empty
                or not report.overloaded_trafos.empty
                or not report.angle_violations.empty
            )
            tc.record("any_violations_consistent",
                      report.any_violations == derived,
                      f"any_violations={report.any_violations} derived={derived}")

            print(f"         {report.summary()}")
        except Exception:
            tc.error = traceback.format_exc()
        tc.duration = time.perf_counter() - t0
        cases.append(tc)
        print_case(tc, verbose)
    return cases


def run_violation_detector_all_tests(verbose: bool = False,
                                      only: list = None) -> list:
    cases   = []
    records = []

    def _run_one(tc, net_name, loader_fn):
        try:
            net    = loader_fn()
            pp.runpp(net, voltage_depend_loads=False)
            report = detect_violations(net)
            tc.record("converged",        report.converged)
            tc.record("report_valid",     hasattr(report, "any_violations"))
            tc.record("angle_check_ran",  hasattr(report, "angle_violations"))
            max_vm   = float(net.res_bus["vm_pu"].max()) if report.converged else None
            min_vm   = float(net.res_bus["vm_pu"].min()) if report.converged else None
            def _res_max(table, col):
                t = getattr(net, table, None)
                if report.converged and t is not None and not t.empty \
                        and col in t.columns:
                    return float(t[col].max())
                return None

            def _res_min(table, col):
                t = getattr(net, table, None)
                if report.converged and t is not None and not t.empty \
                        and col in t.columns:
                    return float(t[col].min())
                return None

            return {
                "name":               net_name,
                "n_over_voltage":     report.n_over_voltage,
                "n_under_voltage":    report.n_under_voltage,
                "max_vm_pu":          max_vm,
                "min_vm_pu":          min_vm,
                "n_overloaded_trafo": report.n_overloaded_trafos,
                "n_overloaded_line":  report.n_overloaded_lines,
                "max_trafo_loading":  _res_max("res_trafo", "loading_percent"),
                "min_trafo_loading":  _res_min("res_trafo", "loading_percent"),
                "max_line_loading":   _res_max("res_line",  "loading_percent"),
                "min_line_loading":   _res_min("res_line",  "loading_percent"),
                "any_violations":     report.any_violations,
            }
        except Exception:
            tc.error = traceback.format_exc()
            return {"name": net_name, **{k: None for k in [
                "n_over_voltage","n_under_voltage","max_vm_pu","min_vm_pu",
                "n_overloaded_trafo","n_overloaded_line",
                "max_trafo_loading","min_trafo_loading",
                "max_line_loading","min_line_loading","any_violations",
            ]}}

    print(f"\n  [1/6] SimBench  ({len(IN_SCOPE_SIMBENCH_CODES)} networks)")
    for code in IN_SCOPE_SIMBENCH_CODES:
        if only and not any(s in code for s in only):
            continue
        tc = TestCase(code)
        t0 = time.perf_counter()
        rec = _run_one(tc, code, lambda c=code: sb.get_simbench_net(c))
        tc.duration = time.perf_counter() - t0
        cases.append(tc); records.append(rec)
        print_case(tc, verbose)

    print("\n  [2/6] CIGRE networks  (2 networks)")
    for name, loader in [
        ("cigre_mv_with_der", lambda: pn.create_cigre_network_mv(with_der="pv_wind")),
        ("cigre_lv",          lambda: pn.create_cigre_network_lv()),
    ]:
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(name)
        t0 = time.perf_counter()
        rec = _run_one(tc, name, loader)
        tc.duration = time.perf_counter() - t0
        cases.append(tc); records.append(rec)
        print_case(tc, verbose)

    print(f"\n  [3/6] Kerber  ({len(ALL_KERBER_CASES)} variants)")
    for name, fn_name in ALL_KERBER_CASES:
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(name)
        t0 = time.perf_counter()
        rec = _run_one(tc, name, lambda f=fn_name: getattr(pn, f)())
        tc.duration = time.perf_counter() - t0
        cases.append(tc); records.append(rec)
        print_case(tc, verbose)

    print(f"\n  [4/6] Synthetic LV  ({len(ALL_SYNTHETIC_LV_CASES)} classes)")
    for network_class in ALL_SYNTHETIC_LV_CASES:
        name = f"synthetic_lv_{network_class}"
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(name)
        t0 = time.perf_counter()
        rec = _run_one(tc, name,
            lambda c=network_class: pn.create_synthetic_voltage_control_lv_network(c))
        tc.duration = time.perf_counter() - t0
        cases.append(tc); records.append(rec)
        print_case(tc, verbose)

    print(f"\n  [5/6] Dickert LV  ({len(ALL_DICKERT_CASES)} combinations)")
    for name, feeders_range, linetype, customer, case in ALL_DICKERT_CASES:
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(name)
        t0 = time.perf_counter()
        try:
            net = pn.create_dickert_lv_network(feeders_range, linetype, customer, case)
            rec = _run_one(tc, name, lambda n=net: n)
        except ValueError as e:
            if "no dickert network" in str(e):
                tc.skipped = True
                rec = {"name": name, **{k: None for k in [
                    "n_over_voltage","n_under_voltage","max_vm_pu","min_vm_pu",
                    "n_overloaded_trafo","n_overloaded_line",
                    "max_trafo_loading","min_trafo_loading",
                    "max_line_loading","min_line_loading","any_violations",
                ]}}
            else:
                tc.error = traceback.format_exc()
                rec = {"name": name, **{k: None for k in [
                    "n_over_voltage","n_under_voltage","max_vm_pu","min_vm_pu",
                    "n_overloaded_trafo","n_overloaded_line",
                    "max_trafo_loading","min_trafo_loading",
                    "max_line_loading","min_line_loading","any_violations",
                ]}}
        except Exception:
            tc.error = traceback.format_exc()
            rec = {"name": name, **{k: None for k in [
                "n_over_voltage","n_under_voltage","max_vm_pu","min_vm_pu",
                "n_overloaded_trafo","n_overloaded_line",
                "max_trafo_loading","min_trafo_loading",
                "max_line_loading","min_line_loading","any_violations",
            ]}}
        tc.duration = time.perf_counter() - t0
        cases.append(tc); records.append(rec)
        print_case(tc, verbose)

    print("\n  [6/6] ERA5 datasource  (1 network — runpp only)")
    name = "cigre_mv_era5"
    if not (only and not any(s in name for s in only)):
        tc = TestCase(name)
        t0 = time.perf_counter()
        rec = _run_one(tc, name,
            lambda: pn.create_cigre_network_mv(with_der="pv_wind"))
        tc.duration = time.perf_counter() - t0
        cases.append(tc); records.append(rec)
        print_case(tc, verbose)

    _print_violations_summary(records)
    return cases


def _print_violations_summary(records: list, top_n: int = 10) -> None:
    """
    Six ranked tables covering all key violation dimensions.
    Restored from original test_suite.py — this was incorrectly reduced to
    two tables during the refactor.
    """
    valid = [r for r in records if r.get("max_vm_pu") is not None]
    if not valid:
        print("\n  [violations summary] No valid results.")
        return

    W = 74
    print(f"\n{'='*W}")
    print("  VIOLATIONS SUMMARY")
    print("  Nominal loading only — no DWD profiles applied.")
    print("  Identifies structurally weak networks, not worst-case scenarios.")
    print(f"{'='*W}")

    def _table(title, rows, val_key, count_key, threshold, higher_is_bad,
               val_fmt=".4f", val_label="value", count_label="count"):
        filtered = [r for r in rows if r.get(val_key) is not None]
        if not filtered:
            print(f"\n  {title}: no data available.")
            return
        ranked = sorted(filtered,
                        key=lambda r: r[val_key] or 0.0,
                        reverse=higher_is_bad)
        n = min(top_n, len(ranked))
        print(f"\n  Top {n} — {title}:")
        print(f"  {'#':<4} {'Network':<44} {val_label:>12} {count_label:>10}")
        print(f"  {'-'*W}")
        for i, r in enumerate(ranked[:n], 1):
            val = r[val_key]
            cnt = r.get(count_key) or 0
            flag = "  ← VIOLATION" if (
                (higher_is_bad and val > threshold)
                or (not higher_is_bad and val < threshold)
            ) else ""
            print(f"  {i:<4} {r['name']:<44} {val:>12{val_fmt}} "
                  f"{cnt:>10}{flag}")

    _table("Highest bus voltage (overvoltage risk)",
           valid, "max_vm_pu", "n_over_voltage",
           threshold=V_MAX, higher_is_bad=True,
           val_fmt=".4f", val_label="max vm_pu", count_label="OV buses")

    _table("Lowest bus voltage (undervoltage risk)",
           valid, "min_vm_pu", "n_under_voltage",
           threshold=V_MIN, higher_is_bad=False,
           val_fmt=".4f", val_label="min vm_pu", count_label="UV buses")

    _table("Highest transformer loading (thermal risk)",
           valid, "max_trafo_loading", "n_overloaded_trafo",
           threshold=TRAFO_MAX_LOADING, higher_is_bad=True,
           val_fmt=".1f", val_label="max load %", count_label="OL trafos")

    _table("Lowest transformer loading (informational)",
           valid, "min_trafo_loading", "n_overloaded_trafo",
           threshold=0.0, higher_is_bad=False,
           val_fmt=".1f", val_label="min load %", count_label="OL trafos")

    _table("Highest line loading (thermal risk)",
           valid, "max_line_loading", "n_overloaded_line",
           threshold=LINE_MAX_LOADING, higher_is_bad=True,
           val_fmt=".1f", val_label="max load %", count_label="OL lines")

    _table("Lowest line loading (informational)",
           valid, "min_line_loading", "n_overloaded_line",
           threshold=0.0, higher_is_bad=False,
           val_fmt=".1f", val_label="min load %", count_label="OL lines")

    n_any = sum(1 for r in valid if r.get("any_violations"))
    n_ov  = sum(1 for r in valid if (r.get("n_over_voltage")  or 0) > 0)
    n_uv  = sum(1 for r in valid if (r.get("n_under_voltage") or 0) > 0)
    n_olt = sum(1 for r in valid if (r.get("n_overloaded_trafo") or 0) > 0)
    n_oll = sum(1 for r in valid if (r.get("n_overloaded_line")  or 0) > 0)
    print(f"\n  {'Networks with any violation:':<42} {n_any:>4} / {len(valid)}")
    print(f"  {'  overvoltage:':<42} {n_ov:>4}")
    print(f"  {'  undervoltage:':<42} {n_uv:>4}")
    print(f"  {'  overloaded transformer:':<42} {n_olt:>4}")
    print(f"  {'  overloaded line:':<42} {n_oll:>4}")
    print(f"{'='*W}\n")
