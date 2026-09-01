"""
hiltest/sections/profile_builder.py
=====================================
Section: profile_builder

Pure refactor — no logic changes from original test_suite.py.
Timing switched to time.perf_counter().
"""
import traceback
import time
import pandapower.networks as pn
import simbench as sb

from hiltest.framework   import TestCase, print_case
from hiltest.catalogues  import (
    IN_SCOPE_SIMBENCH_CODES, ALL_KERBER_CASES, ALL_SYNTHETIC_LV_CASES,
    ALL_DICKERT_CASES,
)
from hiltest.constants   import (
    DWD_DATA_DIR, ERA5_DATA_DIR, ERA5_FILE_MAP, ERA5_COL_MAP,
)
from hiltest.data_checks import check_profiles


def run_profile_builder_tests(verbose: bool = False, only: list = None) -> list:
    from profile_builder import build_annual_profiles
    cases = []

    print(f"\n  [1/6] SimBench  ({len(IN_SCOPE_SIMBENCH_CODES)} networks)")
    for code in IN_SCOPE_SIMBENCH_CODES:
        if only and not any(s in code for s in only):
            continue
        tc = TestCase(code)
        t0 = time.perf_counter()
        try:
            net    = sb.get_simbench_net(code)
            result = build_annual_profiles(net, code, simbench_code=code)
            check_profiles(tc, result, check_pv_night=True)
        except Exception:
            tc.error = traceback.format_exc()
        tc.duration = time.perf_counter() - t0
        cases.append(tc)
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
        try:
            net    = loader()
            result = build_annual_profiles(net, name, data_dir=DWD_DATA_DIR)
            check_profiles(tc, result, check_pv_night=True)
        except Exception:
            tc.error = traceback.format_exc()
        tc.duration = time.perf_counter() - t0
        cases.append(tc)
        print_case(tc, verbose)

    print(f"\n  [3/6] Kerber networks  ({len(ALL_KERBER_CASES)} variants)")
    for name, fn_name in ALL_KERBER_CASES:
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(name)
        t0 = time.perf_counter()
        try:
            net    = getattr(pn, fn_name)()
            result = build_annual_profiles(net, name, data_dir=DWD_DATA_DIR)
            check_profiles(tc, result, check_pv_night=False)
        except Exception:
            tc.error = traceback.format_exc()
        tc.duration = time.perf_counter() - t0
        cases.append(tc)
        print_case(tc, verbose)

    print(f"\n  [4/6] Synthetic Voltage Control LV  ({len(ALL_SYNTHETIC_LV_CASES)} classes)")
    for network_class in ALL_SYNTHETIC_LV_CASES:
        name = f"synthetic_lv_{network_class}"
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(name)
        t0 = time.perf_counter()
        try:
            net    = pn.create_synthetic_voltage_control_lv_network(network_class)
            result = build_annual_profiles(net, name, data_dir=DWD_DATA_DIR)
            check_profiles(tc, result, check_pv_night=True)
        except Exception:
            tc.error = traceback.format_exc()
        tc.duration = time.perf_counter() - t0
        cases.append(tc)
        print_case(tc, verbose)

    print(f"\n  [5/6] Dickert LV  ({len(ALL_DICKERT_CASES)} combinations)")
    for name, feeders_range, linetype, customer, case in ALL_DICKERT_CASES:
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(name)
        t0 = time.perf_counter()
        try:
            net    = pn.create_dickert_lv_network(feeders_range, linetype, customer, case)
            result = build_annual_profiles(net, name, data_dir=DWD_DATA_DIR)
            check_profiles(tc, result, check_pv_night=False)
        except ValueError as e:
            if "no dickert network" in str(e):
                tc.skipped = True
            else:
                tc.error = traceback.format_exc()
        except Exception:
            tc.error = traceback.format_exc()
        tc.duration = time.perf_counter() - t0
        cases.append(tc)
        print_case(tc, verbose)

    print("\n  [6/6] ERA5 datasource  (CIGRE MV)")
    if not (only and not any(s in "cigre_mv_era5" for s in only)):
        tc = TestCase("cigre_mv_era5")
        t0 = time.perf_counter()
        try:
            net    = pn.create_cigre_network_mv(with_der="pv_wind")
            result = build_annual_profiles(
                net, "cigre_mv",
                data_dir=ERA5_DATA_DIR,
                file_map=ERA5_FILE_MAP,
                col_map=ERA5_COL_MAP,
            )
            check_profiles(tc, result, check_pv_night=True)
        except Exception:
            tc.error = traceback.format_exc()
        tc.duration = time.perf_counter() - t0
        cases.append(tc)
        print_case(tc, verbose)

    return cases
