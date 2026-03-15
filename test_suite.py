"""
HIL Testbed — Master Test Framework
=====================================
Extensible stress test suite for the full HIL testbed project.
Each module gets its own test section. Add new sections as modules
are implemented. Run the full suite or a single section at any time.

Current test sections:
    1. profile_builder   — annual profile generation
                           - 156 in-scope SimBench networks
                             (MVLV coupled + MV single + LV single)
                           - CIGRE MV + LV
                           - Kerber (6 variants)
                           - Synthetic Voltage Control LV (5 classes)
                           - Dickert LV (all 18 combinations)
                           - ERA5 datasource (CIGRE MV)

Out of scope (skipped — EHV/HV/HVMV networks exceed project scope
and require >9 GB RAM per network for profile allocation):
    - complete_data       (3 codes)
    - EHVHVMVLV coupled   (3 codes)
    - EHVHV coupled       (18 codes)
    - EHV single          (6 codes)
    - HVMV coupled        (48 codes)
    - HV single           (12 codes)

Planned sections (uncomment and implement as modules are ready):
    2. network_plotter   — topology and profile visualisation
    3. volt_var_control  — Tier 1 Volt-VAr Q(V) algorithm
    4. baseline_scenario — Scenario 1 baseline timeseries run
    5. oltc_scenario     — Scenario 2 OLTC-only
    6. svc_scenario      — Scenario 3 SVC
    7. hil_scenario      — Scenario 4 Rule-based Volt-VAr HIL loop
    8. opf_scenario      — Scenario 5 OPF benchmark
    9. hosting_capacity  — Hosting capacity analysis (with/without Volt-VAr)

Usage:
    Run all sections:
        python test_suite.py

    Run a single section:
        python test_suite.py --section profile_builder

    Run with verbose output (prints full tracebacks for all failures):
        python test_suite.py --verbose

Structure for adding a new section:
    1. Define a run_{section_name}_tests() function below
    2. Register it in SECTIONS dict at the bottom of this file
    3. The framework handles timing, pass/fail counting, and summary

Dependencies:
    pip install pandapower simbench pvlib oemof-demand workalendar
"""

import argparse
import traceback
import time
import numpy as np
import pandas as pd

# ===========================================================================
# Paths — update if your folder structure differs
# ===========================================================================
DWD_DATA_DIR  = "data/dwd"
ERA5_DATA_DIR = "data/era5"

ERA5_FILE_MAP = {
    "RAD-G": "era5_solar.csv",
    "F":     "era5_wind.csv",
    "T2M":   "era5_temp.csv",
}

ERA5_COL_MAP = {
    "timestamp": "timestamp",
    "solar":     "GHI_Wm2",
    "wind":      "WS_ms",
    "temp":      "AT_degC",
    "sep":       ",",
}

# ===========================================================================
# In-scope SimBench network codes  (156 total)
# Scope: MV/LV distribution networks relevant to voltage regulation studies
# Excluded: EHV, HV, HVMV, complete_data — transmission level, out of scope
#           and require >9 GB RAM per network for profile allocation
# ===========================================================================
IN_SCOPE_SIMBENCH_CODES = [

    # ------------------------------------------------------------------
    # MV+LV coupled — rural  (12 codes)
    # ------------------------------------------------------------------
    "1-MVLV-rural-all-0-sw",     "1-MVLV-rural-all-0-no_sw",
    "1-MVLV-rural-all-1-sw",     "1-MVLV-rural-all-1-no_sw",
    "1-MVLV-rural-all-2-sw",     "1-MVLV-rural-all-2-no_sw",
    "1-MVLV-rural-1.108-0-sw",   "1-MVLV-rural-1.108-0-no_sw",
    "1-MVLV-rural-1.108-1-sw",   "1-MVLV-rural-1.108-1-no_sw",
    "1-MVLV-rural-1.108-2-sw",   "1-MVLV-rural-1.108-2-no_sw",
    "1-MVLV-rural-2.107-0-sw",   "1-MVLV-rural-2.107-0-no_sw",
    "1-MVLV-rural-2.107-1-sw",   "1-MVLV-rural-2.107-1-no_sw",
    "1-MVLV-rural-2.107-2-sw",   "1-MVLV-rural-2.107-2-no_sw",
    "1-MVLV-rural-4.101-0-sw",   "1-MVLV-rural-4.101-0-no_sw",
    "1-MVLV-rural-4.101-1-sw",   "1-MVLV-rural-4.101-1-no_sw",
    "1-MVLV-rural-4.101-2-sw",   "1-MVLV-rural-4.101-2-no_sw",

    # ------------------------------------------------------------------
    # MV+LV coupled — semiurban  (12 codes)
    # ------------------------------------------------------------------
    "1-MVLV-semiurb-all-0-sw",   "1-MVLV-semiurb-all-0-no_sw",
    "1-MVLV-semiurb-all-1-sw",   "1-MVLV-semiurb-all-1-no_sw",
    "1-MVLV-semiurb-all-2-sw",   "1-MVLV-semiurb-all-2-no_sw",
    "1-MVLV-semiurb-3.202-0-sw", "1-MVLV-semiurb-3.202-0-no_sw",
    "1-MVLV-semiurb-3.202-1-sw", "1-MVLV-semiurb-3.202-1-no_sw",
    "1-MVLV-semiurb-3.202-2-sw", "1-MVLV-semiurb-3.202-2-no_sw",
    "1-MVLV-semiurb-4.201-0-sw", "1-MVLV-semiurb-4.201-0-no_sw",
    "1-MVLV-semiurb-4.201-1-sw", "1-MVLV-semiurb-4.201-1-no_sw",
    "1-MVLV-semiurb-4.201-2-sw", "1-MVLV-semiurb-4.201-2-no_sw",
    "1-MVLV-semiurb-5.220-0-sw", "1-MVLV-semiurb-5.220-0-no_sw",
    "1-MVLV-semiurb-5.220-1-sw", "1-MVLV-semiurb-5.220-1-no_sw",
    "1-MVLV-semiurb-5.220-2-sw", "1-MVLV-semiurb-5.220-2-no_sw",

    # ------------------------------------------------------------------
    # MV+LV coupled — urban  (10 codes)
    # ------------------------------------------------------------------
    "1-MVLV-urban-all-0-sw",     "1-MVLV-urban-all-0-no_sw",
    "1-MVLV-urban-all-1-sw",     "1-MVLV-urban-all-1-no_sw",
    "1-MVLV-urban-all-2-sw",     "1-MVLV-urban-all-2-no_sw",
    "1-MVLV-urban-5.303-0-sw",   "1-MVLV-urban-5.303-0-no_sw",
    "1-MVLV-urban-5.303-1-sw",   "1-MVLV-urban-5.303-1-no_sw",
    "1-MVLV-urban-5.303-2-sw",   "1-MVLV-urban-5.303-2-no_sw",
    "1-MVLV-urban-6.305-0-sw",   "1-MVLV-urban-6.305-0-no_sw",
    "1-MVLV-urban-6.305-1-sw",   "1-MVLV-urban-6.305-1-no_sw",
    "1-MVLV-urban-6.305-2-sw",   "1-MVLV-urban-6.305-2-no_sw",
    "1-MVLV-urban-6.309-0-sw",   "1-MVLV-urban-6.309-0-no_sw",
    "1-MVLV-urban-6.309-1-sw",   "1-MVLV-urban-6.309-1-no_sw",
    "1-MVLV-urban-6.309-2-sw",   "1-MVLV-urban-6.309-2-no_sw",

    # ------------------------------------------------------------------
    # MV+LV coupled — commercial  (12 codes)
    # ------------------------------------------------------------------
    "1-MVLV-comm-all-0-sw",      "1-MVLV-comm-all-0-no_sw",
    "1-MVLV-comm-all-1-sw",      "1-MVLV-comm-all-1-no_sw",
    "1-MVLV-comm-all-2-sw",      "1-MVLV-comm-all-2-no_sw",
    "1-MVLV-comm-3.403-0-sw",    "1-MVLV-comm-3.403-0-no_sw",
    "1-MVLV-comm-3.403-1-sw",    "1-MVLV-comm-3.403-1-no_sw",
    "1-MVLV-comm-3.403-2-sw",    "1-MVLV-comm-3.403-2-no_sw",
    "1-MVLV-comm-4.416-0-sw",    "1-MVLV-comm-4.416-0-no_sw",
    "1-MVLV-comm-4.416-1-sw",    "1-MVLV-comm-4.416-1-no_sw",
    "1-MVLV-comm-4.416-2-sw",    "1-MVLV-comm-4.416-2-no_sw",
    "1-MVLV-comm-5.401-0-sw",    "1-MVLV-comm-5.401-0-no_sw",
    "1-MVLV-comm-5.401-1-sw",    "1-MVLV-comm-5.401-1-no_sw",
    "1-MVLV-comm-5.401-2-sw",    "1-MVLV-comm-5.401-2-no_sw",

    # ------------------------------------------------------------------
    # MV single level — rural, semiurban, urban, commercial  (24 codes)
    # ------------------------------------------------------------------
    "1-MV-rural--0-sw",          "1-MV-rural--0-no_sw",
    "1-MV-rural--1-sw",          "1-MV-rural--1-no_sw",
    "1-MV-rural--2-sw",          "1-MV-rural--2-no_sw",
    "1-MV-semiurb--0-sw",        "1-MV-semiurb--0-no_sw",
    "1-MV-semiurb--1-sw",        "1-MV-semiurb--1-no_sw",
    "1-MV-semiurb--2-sw",        "1-MV-semiurb--2-no_sw",
    "1-MV-urban--0-sw",          "1-MV-urban--0-no_sw",
    "1-MV-urban--1-sw",          "1-MV-urban--1-no_sw",
    "1-MV-urban--2-sw",          "1-MV-urban--2-no_sw",
    "1-MV-comm--0-sw",           "1-MV-comm--0-no_sw",
    "1-MV-comm--1-sw",           "1-MV-comm--1-no_sw",
    "1-MV-comm--2-sw",           "1-MV-comm--2-no_sw",

    # ------------------------------------------------------------------
    # LV single level — rural1/2/3, semiurb4/5, urban6  (36 codes)
    # ------------------------------------------------------------------
    "1-LV-rural1--0-sw",         "1-LV-rural1--0-no_sw",
    "1-LV-rural1--1-sw",         "1-LV-rural1--1-no_sw",
    "1-LV-rural1--2-sw",         "1-LV-rural1--2-no_sw",
    "1-LV-rural2--0-sw",         "1-LV-rural2--0-no_sw",
    "1-LV-rural2--1-sw",         "1-LV-rural2--1-no_sw",
    "1-LV-rural2--2-sw",         "1-LV-rural2--2-no_sw",
    "1-LV-rural3--0-sw",         "1-LV-rural3--0-no_sw",
    "1-LV-rural3--1-sw",         "1-LV-rural3--1-no_sw",
    "1-LV-rural3--2-sw",         "1-LV-rural3--2-no_sw",
    "1-LV-semiurb4--0-sw",       "1-LV-semiurb4--0-no_sw",
    "1-LV-semiurb4--1-sw",       "1-LV-semiurb4--1-no_sw",
    "1-LV-semiurb4--2-sw",       "1-LV-semiurb4--2-no_sw",
    "1-LV-semiurb5--0-sw",       "1-LV-semiurb5--0-no_sw",
    "1-LV-semiurb5--1-sw",       "1-LV-semiurb5--1-no_sw",
    "1-LV-semiurb5--2-sw",       "1-LV-semiurb5--2-no_sw",
    "1-LV-urban6--0-sw",         "1-LV-urban6--0-no_sw",
    "1-LV-urban6--1-sw",         "1-LV-urban6--1-no_sw",
    "1-LV-urban6--2-sw",         "1-LV-urban6--2-no_sw",
]

# All 18 Dickert LV combinations (3 feeder lengths × 3 customer types × 2 line types)
ALL_DICKERT_CASES = [
    # short / cable / single
    ("dickert_short_cable_single_good",        "short", "cable",  "single",   "good"),
    ("dickert_short_cable_single_average",     "short", "cable",  "single",   "average"),
    ("dickert_short_cable_single_bad",         "short", "cable",  "single",   "bad"),
    # short / cable / multiple
    ("dickert_short_cable_multiple_good",      "short", "cable",  "multiple", "good"),
    ("dickert_short_cable_multiple_average",   "short", "cable",  "multiple", "average"),
    ("dickert_short_cable_multiple_bad",       "short", "cable",  "multiple", "bad"),
    # middle / cable / multiple
    ("dickert_middle_cable_multiple_good",     "middle","cable",  "multiple", "good"),
    ("dickert_middle_cable_multiple_average",  "middle","cable",  "multiple", "average"),
    ("dickert_middle_cable_multiple_bad",      "middle","cable",  "multiple", "bad"),
    # middle / C&OHL / multiple
    ("dickert_middle_cohl_multiple_good",      "middle","C&OHL",  "multiple", "good"),
    ("dickert_middle_cohl_multiple_average",   "middle","C&OHL",  "multiple", "average"),
    ("dickert_middle_cohl_multiple_bad",       "middle","C&OHL",  "multiple", "bad"),
    # long / cable / multiple
    ("dickert_long_cable_multiple_good",       "long",  "cable",  "multiple", "good"),
    ("dickert_long_cable_multiple_average",    "long",  "cable",  "multiple", "average"),
    ("dickert_long_cable_multiple_bad",        "long",  "cable",  "multiple", "bad"),
    # long / C&OHL / multiple
    ("dickert_long_cohl_multiple_good",        "long",  "C&OHL",  "multiple", "good"),
    ("dickert_long_cohl_multiple_average",     "long",  "C&OHL",  "multiple", "average"),
    ("dickert_long_cohl_multiple_bad",         "long",  "C&OHL",  "multiple", "bad"),
]

# All 5 Synthetic Voltage Control LV classes
ALL_SYNTHETIC_LV_CASES = [
    "rural_1", "rural_2", "village_1", "village_2", "suburb_1"
]

#17 Kerber variants total — 7 standard + 10 extreme
ALL_KERBER_CASES = [
    # Standard Landnetze
    ("kerber_landnetz_kabel_1",          "create_kerber_landnetz_kabel_1"),
    ("kerber_landnetz_kabel_2",          "create_kerber_landnetz_kabel_2"),
    ("kerber_landnetz_freileitung_1",    "create_kerber_landnetz_freileitung_1"),
    ("kerber_landnetz_freileitung_2",    "create_kerber_landnetz_freileitung_2"),
    # Standard Vorstadtnetze
    ("kerber_vorstadtnetz_kabel_1",      "create_kerber_vorstadtnetz_kabel_1"),
    ("kerber_vorstadtnetz_kabel_2",      "create_kerber_vorstadtnetz_kabel_2"),
    # Standard Dorfnetz
    ("kerber_dorfnetz",                  "create_kerber_dorfnetz"),
    # Extreme Landnetze
    ("kb_extrem_landnetz_kabel",         "kb_extrem_landnetz_kabel"),
    ("kb_extrem_landnetz_freileitung",   "kb_extrem_landnetz_freileitung"),
    ("kb_extrem_landnetz_kabel_trafo",   "kb_extrem_landnetz_kabel_trafo"),
    ("kb_extrem_landnetz_frltg_trafo",   "kb_extrem_landnetz_freileitung_trafo"),
    # Extreme Dorfnetze
    ("kb_extrem_dorfnetz",               "kb_extrem_dorfnetz"),
    ("kb_extrem_dorfnetz_trafo",         "kb_extrem_dorfnetz_trafo"),
    # Extreme Vorstadtnetze
    ("kb_extrem_vorstadtnetz_1",         "kb_extrem_vorstadtnetz_1"),
    ("kb_extrem_vorstadtnetz_2",         "kb_extrem_vorstadtnetz_2"),
    ("kb_extrem_vorstadtnetz_trafo_1",   "kb_extrem_vorstadtnetz_trafo_1"),
    ("kb_extrem_vorstadtnetz_trafo_2",   "kb_extrem_vorstadtnetz_trafo_2"),
]

# ===========================================================================
# Core check engine  (shared across all test sections)
# ===========================================================================

class TestCase:
    """Represents a single named test with pass/fail/error state."""

    def __init__(self, name: str):
        self.name     = name
        self.checks   = []    # list of (check_name, passed, detail)
        self.error    = None  # full traceback string if test crashed
        self.duration = 0.0
        self.skipped = False

    def record(self, check_name: str, condition: bool, detail: str = ""):
        self.checks.append((check_name, condition, detail))

    @property
    def passed(self) -> bool:
        return not self.error and not self.skipped and all(ok for _, ok, _ in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for _, ok, _ in self.checks if ok)

    @property
    def n_total(self) -> int:
        return len(self.checks)


def print_case(tc: TestCase, verbose: bool = False):
    if tc.skipped:
        print(f"  SKIP  {tc.name:<60}")
        return
    
    status = "PASS" if tc.passed else "FAIL"
    print(f"  {status}  {tc.name:<60}  "
          f"({tc.n_passed}/{tc.n_total})  [{tc.duration:.1f}s]")
    if not tc.passed:
        if tc.error:
            last_line = tc.error.strip().splitlines()[-1]
            print(f"         ERROR: {last_line}")
        for name, ok, detail in tc.checks:
            if not ok:
                print(f"         FAIL check '{name}': {detail}")
        if verbose and tc.error:
            print(tc.error)


# ===========================================================================
# Shared profile_builder sanity checks
# ===========================================================================

def check_profiles(tc: TestCase, result: dict, check_pv_night: bool = False):
    """Populates a TestCase with standard profile_builder sanity checks."""

    required = {"load", "pv", "wind", "times", "net_type", "extreme_days"}
    missing  = required - set(result.keys())
    tc.record("required_keys", not missing,
              f"Missing: {missing}" if missing else "")

    times = result.get("times", pd.DatetimeIndex([]))
    tc.record("timestep_count", len(times) > 100,
              f"Only {len(times)} timesteps")

    # Extreme days — all four keys must be present
    extreme = result.get("extreme_days", {})
    tc.record("extreme_days_exists",
              isinstance(extreme, dict), "Not a dict")
    tc.record("extreme_days_keys",
              all(k in extreme for k in
                  ("max_der", "min_der", "max_load", "min_load")),
              f"Keys present: {list(extreme.keys())}")

    # Load checks
    load_df = result.get("load")
    if load_df is not None and not load_df.empty:
        bad = load_df.columns[load_df.isna().all()].tolist()
        tc.record("load_no_all_nan",   not bad,
                  f"All-NaN cols: {bad}")
        tc.record("load_sum_positive", load_df.sum().sum() > 0,
                  "All load values are zero")
        tc.record("load_no_negative",  (load_df >= 0).all().all(),
                  "Negative load values found")
    else:
        tc.record("load_exists", False, "Load DataFrame missing or empty")

    # PV checks
    pv_df = result.get("pv")
    if pv_df is not None and not pv_df.empty:
        bad = pv_df.columns[pv_df.isna().all()].tolist()
        tc.record("pv_no_all_nan",  not bad,
                  f"All-NaN cols: {bad}")
        tc.record("pv_no_negative", (pv_df >= 0).all().all(),
                  "Negative PV values found")
        if check_pv_night and isinstance(times, pd.DatetimeIndex) \
                and len(times) > 0:
            night = (times.hour >= 22) | (times.hour <= 4)
            tc.record("pv_zero_at_night",
                      (pv_df.loc[night] < 0.001).all().all(),
                      "Non-zero PV found at night hours")

    # Wind checks
    wind_df = result.get("wind")
    if wind_df is not None and not wind_df.empty:
        bad = wind_df.columns[wind_df.isna().all()].tolist()
        tc.record("wind_no_all_nan",  not bad,
                  f"All-NaN cols: {bad}")
        tc.record("wind_no_negative", (wind_df >= 0).all().all(),
                  "Negative wind values found")
        tc.record("wind_max_plausible", wind_df.max().max() < 1000,
                  f"Max wind {wind_df.max().max():.1f} MW exceeds 1000 MW")


# ===========================================================================
# SECTION 1 — profile_builder
# ===========================================================================

def run_profile_builder_tests(verbose: bool = False, only:list =None) -> list:
    """
    Tests build_annual_profiles() across all in-scope networks.

    Network coverage:
        156  SimBench (MVLV coupled + MV single + LV single)
          2  CIGRE (MV with DER + LV)
          6  Kerber variants
          5  Synthetic Voltage Control LV classes
         18  Dickert LV (all feeder/customer/line-type combinations)
          1  ERA5 datasource (CIGRE MV)
        ---
        188  total test cases
    """
    from profile_builder import build_annual_profiles
    import simbench as sb
    import pandapower.networks as pn

    cases = []

    # -----------------------------------------------------------------------
    # SimBench — 156 in-scope codes
    # -----------------------------------------------------------------------
    print(f"\n  [1/6] SimBench  ({len(IN_SCOPE_SIMBENCH_CODES)} networks)")
    for code in IN_SCOPE_SIMBENCH_CODES:
        if only and not any(s in code for s in only):
            continue
        tc = TestCase(code)
        t0 = time.time()
        try:
            net    = sb.get_simbench_net(code)
            result = build_annual_profiles(net, code, simbench_code=code)
            check_profiles(tc, result, check_pv_night=True)
        except Exception:
            tc.error = traceback.format_exc()
        tc.duration = time.time() - t0
        cases.append(tc)
        print_case(tc, verbose)

    # -----------------------------------------------------------------------
    # CIGRE MV + LV
    # -----------------------------------------------------------------------
    print(f"\n  [2/6] CIGRE networks  (2 networks)")
    for name, loader in [
        ("cigre_mv_with_der", lambda: pn.create_cigre_network_mv(with_der="pv_wind")),
        ("cigre_lv",          lambda: pn.create_cigre_network_lv()),
    ]:
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(name)
        t0 = time.time()
        try:
            net    = loader()
            result = build_annual_profiles(net, name, data_dir=DWD_DATA_DIR)
            check_profiles(tc, result, check_pv_night=True)
        except Exception:
            tc.error = traceback.format_exc()
        tc.duration = time.time() - t0
        cases.append(tc)
        print_case(tc, verbose)

    # -----------------------------------------------------------------------
    # Kerber — all 6 variants
    # -----------------------------------------------------------------------
    print(f"\n  [3/6] Kerber networks  ({len(ALL_KERBER_CASES)} variants)")
    for name, fn_name in ALL_KERBER_CASES:
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(name)
        t0 = time.time()
        try:
            fn     = getattr(pn, fn_name)
            net    = fn()
            result = build_annual_profiles(net, name, data_dir=DWD_DATA_DIR)
            check_profiles(tc, result, check_pv_night=False)
        except Exception:
            tc.error = traceback.format_exc()
        tc.duration = time.time() - t0
        cases.append(tc)
        print_case(tc, verbose)

    # -----------------------------------------------------------------------
    # Synthetic Voltage Control LV — all 5 classes
    # -----------------------------------------------------------------------
    print(f"\n  [4/6] Synthetic Voltage Control LV  "
          f"({len(ALL_SYNTHETIC_LV_CASES)} classes)")
    for network_class in ALL_SYNTHETIC_LV_CASES:
        name = f"synthetic_lv_{network_class}"
        if only and not any(s in name for s in only):
            continue
        tc   = TestCase(name)
        t0   = time.time()
        try:
            net    = pn.create_synthetic_voltage_control_lv_network(
                        network_class
                     )
            result = build_annual_profiles(net, name, data_dir=DWD_DATA_DIR)
            check_profiles(tc, result, check_pv_night=True)
        except Exception:
            tc.error = traceback.format_exc()
        tc.duration = time.time() - t0
        cases.append(tc)
        print_case(tc, verbose)

    # -----------------------------------------------------------------------
    # Dickert LV — all 18 combinations
    # -----------------------------------------------------------------------
    print(f"\n  [5/6] Dickert LV  ({len(ALL_DICKERT_CASES)} combinations)  "
          f"[3 feeder lengths × 3 customer types × 2 line types]")
    for name, feeders_range, linetype, customer, case in ALL_DICKERT_CASES:
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(name)
        t0 = time.time()
        try:
            net = pn.create_dickert_lv_network(feeders_range, linetype, customer, case)
            result = build_annual_profiles(net, name, data_dir=DWD_DATA_DIR)
            check_profiles(tc, result, check_pv_night=False)
        except ValueError as e:
            if "no dickert network" in str(e):
                tc.skipped = True   # mark as skipped, not failed
            else:
                tc.error = traceback.format_exc()
        except Exception:
            tc.error = traceback.format_exc()
        tc.duration = time.time() - t0
        cases.append(tc)
        print_case(tc, verbose)
    # -----------------------------------------------------------------------
    # ERA5 datasource — CIGRE MV as representative case
    # -----------------------------------------------------------------------
    print(f"\n  [6/6] ERA5 datasource  (CIGRE MV)")
    if only and not any(s in "cigre_mv_era5" for s in only):
        pass
    else:
        tc = TestCase("cigre_mv_era5")
        t0 = time.time()
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
        tc.duration = time.time() - t0
        cases.append(tc)
        print_case(tc, verbose)

    return cases


# ===========================================================================
# SECTION 2 — network_plotter  (placeholder)
# ===========================================================================

# def run_network_plotter_tests(verbose: bool = False) -> list:
#     from network_plotter import plot_network_and_profiles, plot_day
#     cases = []
#     # implement when network_plotter.py is ready
#     return cases


# ===========================================================================
# SECTION 3 — volt_var_control  (placeholder)
# ===========================================================================

# def run_volt_var_tests(verbose: bool = False) -> list:
#     from volt_var_control import VoltVarController
#     cases = []
#     # implement when volt_var_control.py is ready
#     return cases


# ===========================================================================
# SECTIONS 4–8 — scenario tests  (placeholders)
# ===========================================================================

# def run_baseline_tests(verbose: bool = False) -> list: ...
# def run_oltc_tests(verbose: bool = False) -> list: ...
# def run_svc_tests(verbose: bool = False) -> list: ...
# def run_hil_tests(verbose: bool = False) -> list: ...
# def run_opf_tests(verbose: bool = False) -> list: ...


# ===========================================================================
# SECTION 9 — hosting_capacity  (placeholder)
# ===========================================================================

# def run_hosting_capacity_tests(verbose: bool = False) -> list: ...


# ===========================================================================
# Section registry
# ===========================================================================

SECTIONS = {
    "profile_builder":  run_profile_builder_tests,
    # "network_plotter":  run_network_plotter_tests,
    # "volt_var_control": run_volt_var_tests,
    # "baseline":         run_baseline_tests,
    # "oltc":             run_oltc_tests,
    # "svc":              run_svc_tests,
    # "hil":              run_hil_tests,
    # "opf":              run_opf_tests,
    # "hosting_capacity": run_hosting_capacity_tests,
}


# ===========================================================================
# Summary printer
# ===========================================================================

def print_summary(section_results: dict):
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    grand_pass = grand_total = 0
    for section, cases in section_results.items():
        n_pass  = sum(1 for tc in cases if tc.passed)
        n_total = len(cases)
        grand_pass  += n_pass
        grand_total += n_total
        status = "PASS" if n_pass == n_total else "FAIL"
        print(f"  {status}  {section:<35}  {n_pass}/{n_total} cases")
    print("-" * 70)
    print(f"  {'PASS' if grand_pass == grand_total else 'FAIL'}  "
          f"{'TOTAL':<35}  {grand_pass}/{grand_total} cases")
    print("=" * 70)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HIL Testbed master test suite"
    )
    parser.add_argument(
        "--section",
        choices=list(SECTIONS.keys()),
        default=None,
        help="Run a single section only (default: all sections)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full tracebacks for all failures"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only test cases whose names contain any of these substrings. "
            "e.g. --only cigre kerber dickert era5"
    )
    args = parser.parse_args()

    t_start = time.time()
    section_results = {}

    to_run = (
        {args.section: SECTIONS[args.section]}
        if args.section
        else SECTIONS
    )

    for section_name, run_fn in to_run.items():
        print(f"\n{'='*70}")
        print(f"  SECTION: {section_name.upper()}")
        print(f"{'='*70}")
        cases = run_fn(verbose=args.verbose, only=args.only)
        section_results[section_name] = cases

    print_summary(section_results)
    print(f"\n  Total time: {time.time() - t_start:.1f}s")
