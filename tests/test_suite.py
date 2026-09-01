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
                           - Kerber (17 variants)
                           - Synthetic Voltage Control LV (5 classes)
                           - Dickert LV (all 18 combinations)
                           - ERA5 datasource (CIGRE MV)
    2. network_plotter   — topology and profile visualisation
                           - 9 representative networks (one per family)
                           - plot_topology, plot_profiles, plot_day × 4
                           - SHOW_PLOTS flag controls interactive display
    3. violation_detector  - Tests detect_violations() across either one 
                            representative network per family or the 
                            entire test suite.
    4.                      
    5.

Out of scope (skipped — EHV/HV/HVMV networks exceed project scope
and require >9 GB RAM per network for profile allocation):
    - complete_data       (3 codes)
    - EHVHVMVLV coupled   (3 codes)
    - EHVHV coupled       (18 codes)
    - EHV single          (6 codes)
    - HVMV coupled        (48 codes)
    - HV single           (12 codes)

Planned sections (uncomment and implement as modules are ready):
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
import warnings
import argparse
import traceback
import time
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import simbench as sb
from violation_detector import ViolationReport, detect_violations
from violation_detector import V_MAX, V_MIN, LINE_MAX_LOADING, TRAFO_MAX_LOADING
from sensitivity_coordinator import (SensitivityCoordinator, CoordinatorResult, run_coordinated_timestep)
from volt_var_controller import VoltVarController, Q_RATIO

# ===========================================================================
# Global display flag
# Set SHOW_PLOTS = True  to view figures interactively (press any key to advance)
# Set SHOW_PLOTS = False for RPi / headless / GitHub CI (figures never open)
# ===========================================================================x
SHOW_PLOTS = False

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
          17  Kerber variants
          5  Synthetic Voltage Control LV classes
         18  Dickert LV (all feeder/customer/line-type combinations)
          1  ERA5 datasource (CIGRE MV)
        ---
        199  total test cases
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
# SECTION 2 — network_plotter
# ===========================================================================
 
def run_network_plotter_tests(verbose: bool = False, only: list = None) -> list:
    """
    Tests plot_topology(), plot_profiles(), and all four plot_day() calls
    across a representative subset of networks — one per network family.
 
    Running all 199 networks through the plotter would generate ~1400 figures
    and take hours. One representative per family exercises the same code paths
    while keeping the suite practical.
 
    Coverage:
        1  SimBench MV rural     (primary demonstration network)
        1  SimBench LV rural     (LV single-level)
        1  SimBench MVLV coupled (coupled voltage level)
        1  CIGRE MV with_der     (secondary MV validation)
        1  CIGRE LV
        1  Kerber standard       (radial LV, no geodata)
        1  Kerber extreme
        1  Synthetic Voltage Control LV
        1  Dickert LV
        ---
        9  total test cases
 
    Each test case runs:
        - plot_topology()         → topology figure
        - plot_profiles()         → annual profiles figure
        - plot_day() × 4          → four extreme day zooms
    
    SHOW_PLOTS controls whether figures are displayed:
        True  — figures appear, press any key to advance (interactive dev)
        False — figures never open (RPi / headless / CI)
    """
    try:
        import matplotlib
        import matplotlib.figure
        import matplotlib.pyplot as plt
        from network_plotter import plot_topology, plot_profiles, plot_day
        from profile_builder import build_annual_profiles
        import simbench as sb
        import pandapower.networks as pn
    except ImportError as exc:
        if verbose:
            print(
                "Skipping network plotter tests because an optional plotting "
                f"dependency is missing: {exc}"
            )
        return []
    # Representative networks: (test_name, loader_fn, net_name, simbench_code)
    REPRESENTATIVE_NETWORKS = [
        ("sb_mv_rural",
            lambda: sb.get_simbench_net("1-MV-rural--2-sw"),
            "1-MV-rural--2-sw",      "1-MV-rural--2-sw"),
        ("sb_lv_rural",
            lambda: sb.get_simbench_net("1-LV-rural1--0-sw"),
            "1-LV-rural1--0-sw",     "1-LV-rural1--0-sw"),
        ("sb_mvlv_rural",
            lambda: sb.get_simbench_net("1-MVLV-rural-all-0-sw"),
            "1-MVLV-rural-all-0-sw", "1-MVLV-rural-all-0-sw"),
        ("cigre_mv",
            lambda: pn.create_cigre_network_mv(with_der="pv_wind"),
            "cigre_mv_with_der",     None),
        ("cigre_lv",
            lambda: pn.create_cigre_network_lv(),
            "cigre_lv",              None),
        ("kerber_standard",
            lambda: pn.create_kerber_landnetz_kabel_1(),
            "kerber_landnetz_kabel_1", None),
        ("kerber_extreme",
            lambda: pn.kb_extrem_landnetz_kabel(),
            "kb_extrem_landnetz_kabel", None),
        ("synthetic_lv",
            lambda: pn.create_synthetic_voltage_control_lv_network("rural_1"),
            "synthetic_lv_rural_1",  None),
        ("dickert",
            lambda: pn.create_dickert_lv_network("short", "cable", "single", "good"),
            "dickert_short_cable_single_good", None),
    ]
 
    cases = []
    extreme_day_keys = [
        ("max_der",  "Max DER generation day"),
        ("min_der",  "Min DER generation day"),
        ("max_load", "Peak load day"),
        ("min_load", "Min load day"),
    ]
 
    for test_name, loader, net_name, sb_code in REPRESENTATIVE_NETWORKS:
        if only and not any(s in test_name for s in only):
            continue
 
        tc = TestCase(test_name)
        t0 = time.time()
        try:
            net = loader()
 
            # Build profiles
            kwargs = dict(data_dir=DWD_DATA_DIR)
            if sb_code:
                kwargs["simbench_code"] = sb_code
            prof = build_annual_profiles(net, net_name, **kwargs)
 
            # --- check 1: plot_topology ---
            fig_topo = plot_topology(net, net_name, show=SHOW_PLOTS)
            tc.record("topology_returns_figure",
                      isinstance(fig_topo, matplotlib.figure.Figure),
                      "plot_topology did not return a Figure")
 
            # --- check 2: plot_profiles ---
            fig_prof = plot_profiles(net_name, prof, show=SHOW_PLOTS)
            tc.record("profiles_returns_figure",
                      isinstance(fig_prof, matplotlib.figure.Figure),
                      "plot_profiles did not return a Figure")
 
            # --- checks 3–6: plot_day for all four extreme days ---
            ed = prof.get("extreme_days", {})
            for day_key, day_label in extreme_day_keys:
                day_str = ed.get(day_key)
                if day_str is None:
                    # No DER units in network — skip DER day checks gracefully
                    tc.record(f"plot_day_{day_key}_skipped_no_der", True,
                              f"No {day_label} — network has no relevant DER")
                    continue
                fig_day = plot_day(prof, day_str, net_name,
                                   day_label=day_label, show=SHOW_PLOTS)
                tc.record(f"plot_day_{day_key}",
                          isinstance(fig_day, matplotlib.figure.Figure),
                          f"plot_day({day_key}) did not return a Figure")
 
        except Exception:
            tc.error = traceback.format_exc()
 
        tc.duration = time.time() - t0
        cases.append(tc)
        print_case(tc, verbose)
 
    return cases

# ===========================================================================
# SECTION 3 — violation_detector  (representative networks)
# ===========================================================================
 
def run_violation_detector_tests(verbose: bool = False,
                                  only: list = None) -> list:
    """
    Tests detect_violations() across one representative network per family.
 
    Uses the same nine networks as Section 2 (network_plotter) so results
    are directly comparable. Each test:
        - loads the network
        - runs runpp(net, voltage_depend_loads=False)
        - calls detect_violations(net)
        - verifies the report structure and field types
        - prints violation summary for human inspection
 
    Coverage:
        1  SimBench MV rural     (primary HIL network)
        1  SimBench LV rural
        1  SimBench MVLV coupled
        1  CIGRE MV with_der     (secondary MV)
        1  CIGRE LV
        1  Kerber standard
        1  Kerber extreme
        1  Synthetic Voltage Control LV
        1  Dickert LV
        ---
        9  total
    """
    import pandapower as pp
    import pandapower.networks as pn
    import simbench as sb
    from violation_detector import ViolationReport, detect_violations
 
    REPRESENTATIVE_NETWORKS = [
        ("sb_mv_rural",
            lambda: sb.get_simbench_net("1-MV-rural--2-sw"),
            "1-MV-rural--2-sw"),
        ("sb_lv_rural",
            lambda: sb.get_simbench_net("1-LV-rural1--0-sw"),
            "1-LV-rural1--0-sw"),
        ("sb_mvlv_rural",
            lambda: sb.get_simbench_net("1-MVLV-rural-all-0-sw"),
            "1-MVLV-rural-all-0-sw"),
        ("cigre_mv",
            lambda: pn.create_cigre_network_mv(with_der="pv_wind"),
            "cigre_mv_with_der"),
        ("cigre_lv",
            lambda: pn.create_cigre_network_lv(),
            "cigre_lv"),
        ("kerber_standard",
            lambda: pn.create_kerber_landnetz_kabel_1(),
            "kerber_landnetz_kabel_1"),
        ("kerber_extreme",
            lambda: pn.kb_extrem_landnetz_kabel(),
            "kb_extrem_landnetz_kabel"),
        ("synthetic_lv",
            lambda: pn.create_synthetic_voltage_control_lv_network("rural_1"),
            "synthetic_lv_rural_1"),
        ("dickert",
            lambda: pn.create_dickert_lv_network("short", "cable", "single", "good"),
            "dickert_short_cable_single_good"),
    ]
 
    cases = []
 
    for test_name, loader, net_name in REPRESENTATIVE_NETWORKS:
        if only and not any(s in test_name for s in only):
            continue
 
        tc = TestCase(test_name)
        t0 = time.time()
        try:
            net = loader()
            pp.runpp(net, voltage_depend_loads=False)
            report = detect_violations(net)
 
            # --- Structural checks ---
            tc.record("returns_violation_report",
                      isinstance(report, ViolationReport),
                      "detect_violations() did not return a ViolationReport")
 
            tc.record("converged",
                      report.converged,
                      "runpp() did not converge")
 
            tc.record("any_violations_is_bool",
                      isinstance(report.any_violations, bool),
                      "any_violations is not bool")
 
            # --- DataFrame column schema checks (safe indexing) ---
            for attr, col in [
                ("over_voltage",      "deviation_pu"),
                ("under_voltage",     "deviation_pu"),
                ("overloaded_lines",  "loading_percent"),
                ("overloaded_trafos", "loading_percent"),
                ("angle_violations",  "va_diff_degree"),
            ]:
                df = getattr(report, attr)
                tc.record(
                    f"{attr}_has_columns",
                    col in df.columns,
                    f"'{col}' column missing from report.{attr} "
                    f"(DataFrame has columns: {list(df.columns)})"
                )
 
            # --- Consistency check: any_violations matches frame states ---
            derived = (
                not report.over_voltage.empty
                or not report.under_voltage.empty
                or not report.overloaded_lines.empty
                or not report.overloaded_trafos.empty
                or not report.angle_violations.empty
            )
            tc.record("any_violations_consistent",
                      report.any_violations == derived,
                      f"any_violations={report.any_violations} but "
                      f"derived={derived} from frame states")
 
            # Print summary for human inspection (always visible regardless of SHOW_PLOTS)
            print(f"         {report.summary()}")
 
        except Exception:
            tc.error = traceback.format_exc()
 
        tc.duration = time.time() - t0
        cases.append(tc)
        print_case(tc, verbose)
 
    return cases
 
 
# ===========================================================================
# SECTION 4 — violation_detector_all  (all networks + ranked summary)
# ===========================================================================
 
def run_violation_detector_all_tests(verbose: bool = False,
                                      only: list = None) -> list:
    """
    Runs detect_violations() across all 199 in-scope networks plus CIGRE,
    Kerber, Synthetic LV, Dickert, and ERA5. Collects violation statistics
    and prints a ranked summary at the end showing which networks produce the
    most severe voltage and transformer violations at nominal operating point.
 
    NOTE: These are results at default loading (no DWD profiles applied).
    The ranking identifies structurally weak networks, not worst-case
    operating scenarios. Profile-driven violation studies belong in the
    scenario test sections (Sections 4–8, placeholders).
 
    Coverage: same as Section 1 (profile_builder) — 199 networks + 18 + 17
              + 5 + 2 + 1 = 242 total.
    """
    cases   = []
    records = []   # violation statistics per network for ranked summary
 
    def _run_one(tc: TestCase,
                 net_name: str,
                 loader_fn) -> dict:
        """
        Load, run, detect, record. Returns a stats dict for the summary.
        Mutates tc with check results.
        """
        try:
            net = loader_fn()
            pp.runpp(net, voltage_depend_loads=False)
            report = detect_violations(net)
 
            tc.record("converged",        report.converged,
                      "runpp() did not converge")
            tc.record("report_valid",
                      hasattr(report, "any_violations"),
                      "ViolationReport malformed")
            tc.record("angle_check_ran",
                      hasattr(report, "angle_violations"),
                      "angle_violations field missing from report")
 
            # Collect statistics even when no violations
            max_vm   = float(net.res_bus["vm_pu"].max()) if report.converged else None
            min_vm   = float(net.res_bus["vm_pu"].min()) if report.converged else None
            max_traf = (
                float(net.res_trafo["loading_percent"].max())
                if report.converged
                and hasattr(net, "res_trafo")
                and not net.res_trafo.empty
                and "loading_percent" in net.res_trafo.columns
                else None
            )
            min_traf = (
                float(net.res_trafo["loading_percent"].min())
                if report.converged
                and hasattr(net, "res_trafo")
                and not net.res_trafo.empty
                and "loading_percent" in net.res_trafo.columns
                else None
            )
            max_line = (
                float(net.res_line["loading_percent"].max())
                if report.converged
                and hasattr(net, "res_line")
                and not net.res_line.empty
                and "loading_percent" in net.res_line.columns
                else None
            )
            min_line = (
                float(net.res_line["loading_percent"].min())
                if report.converged
                and hasattr(net, "res_line")
                and not net.res_line.empty
                and "loading_percent" in net.res_line.columns
                else None
            )
            return {
                "name":               net_name,
                "n_over_voltage":     report.n_over_voltage,
                "n_under_voltage":    report.n_under_voltage,
                "max_vm_pu":          max_vm,
                "min_vm_pu":          min_vm,
                "n_overloaded_trafo": report.n_overloaded_trafos,
                "n_overloaded_line":  report.n_overloaded_lines,
                "max_trafo_loading":  max_traf,
                "min_trafo_loading":  min_traf,
                "max_line_loading":   max_line,
                "min_line_loading":   min_line,
                "any_violations":     report.any_violations,
            }
        except Exception:
            tc.error = traceback.format_exc()
            return {
                "name": net_name,
                "n_over_voltage": None,    "n_under_voltage": None,
                "max_vm_pu": None,         "min_vm_pu": None,
                "n_overloaded_trafo": None,"n_overloaded_line": None,
                "max_trafo_loading": None, "min_trafo_loading": None,
                "max_line_loading": None,  "min_line_loading": None,
                "any_violations": None,
            }
 
    # ------------------------------------------------------------------
    # SimBench — 156 networks
    # ------------------------------------------------------------------
    print(f"\n  [1/6] SimBench  ({len(IN_SCOPE_SIMBENCH_CODES)} networks)")
    for code in IN_SCOPE_SIMBENCH_CODES:
        if only and not any(s in code for s in only):
            continue
        tc = TestCase(code)
        t0 = time.time()
        rec = _run_one(tc, code, lambda c=code: sb.get_simbench_net(c))
        tc.duration = time.time() - t0
        cases.append(tc)
        records.append(rec)
        print_case(tc, verbose)
 
    # ------------------------------------------------------------------
    # CIGRE MV + LV — 2 networks
    # ------------------------------------------------------------------
    print(f"\n  [2/6] CIGRE networks  (2 networks)")
    for name, loader in [
        ("cigre_mv_with_der",
            lambda: pn.create_cigre_network_mv(with_der="pv_wind")),
        ("cigre_lv",
            lambda: pn.create_cigre_network_lv()),
    ]:
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(name)
        t0 = time.time()
        rec = _run_one(tc, name, loader)
        tc.duration = time.time() - t0
        cases.append(tc)
        records.append(rec)
        print_case(tc, verbose)
 
    # ------------------------------------------------------------------
    # Kerber — 17 variants
    # ------------------------------------------------------------------
    print(f"\n  [3/6] Kerber networks  ({len(ALL_KERBER_CASES)} variants)")
    for name, fn_name in ALL_KERBER_CASES:
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(name)
        t0 = time.time()
        fn = getattr(pn, fn_name)
        rec = _run_one(tc, name, fn)
        tc.duration = time.time() - t0
        cases.append(tc)
        records.append(rec)
        print_case(tc, verbose)
 
    # ------------------------------------------------------------------
    # Synthetic Voltage Control LV — 5 classes
    # ------------------------------------------------------------------
    print(f"\n  [4/6] Synthetic Voltage Control LV  ({len(ALL_SYNTHETIC_LV_CASES)} classes)")
    for network_class in ALL_SYNTHETIC_LV_CASES:
        name = f"synthetic_lv_{network_class}"
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(name)
        t0 = time.time()
        rec = _run_one(
            tc, name,
            lambda c=network_class: pn.create_synthetic_voltage_control_lv_network(c)
        )
        tc.duration = time.time() - t0
        cases.append(tc)
        records.append(rec)
        print_case(tc, verbose)
 
    # ------------------------------------------------------------------
    # Dickert LV — 18 combinations
    # ------------------------------------------------------------------
    print(f"\n  [5/6] Dickert LV  ({len(ALL_DICKERT_CASES)} combinations)")
    for name, feeders_range, linetype, customer, case in ALL_DICKERT_CASES:
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(name)
        t0 = time.time()
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
                    "max_line_loading","min_line_loading","any_violations"]}}
            else:
                tc.error = traceback.format_exc()
                rec = {"name": name, **{k: None for k in [
                    "n_over_voltage","n_under_voltage","max_vm_pu","min_vm_pu",
                    "n_overloaded_trafo","n_overloaded_line",
                    "max_trafo_loading","min_trafo_loading",
                    "max_line_loading","min_line_loading","any_violations"]}}
        except Exception:
            tc.error = traceback.format_exc()
            rec = {"name": name, **{k: None for k in [
                "n_over_voltage","n_under_voltage","max_vm_pu","min_vm_pu",
                "n_overloaded_trafo","n_overloaded_line",
                "max_trafo_loading","min_trafo_loading",
                "max_line_loading","min_line_loading","any_violations"]}}
        tc.duration = time.time() - t0
        cases.append(tc)
        records.append(rec)
        print_case(tc, verbose)
 
    # ------------------------------------------------------------------
    # ERA5 datasource — 1 network
    # ------------------------------------------------------------------
    print(f"\n  [6/6] ERA5 datasource  (1 network — runpp only, no profiles)")
    name = "cigre_mv_era5"
    if not (only and not any(s in name for s in only)):
        tc = TestCase(name)
        t0 = time.time()
        rec = _run_one(
            tc, name,
            lambda: pn.create_cigre_network_mv(with_der="pv_wind")
        )
        tc.duration = time.time() - t0
        cases.append(tc)
        records.append(rec)
        print_case(tc, verbose)
 
    _print_violations_summary(records)
 
    return cases
 
 
def _print_violations_summary(records: list, top_n: int = 10):
    """
    Prints six ranked tables covering all key violation dimensions:
        1. Highest bus voltage   — overvoltage risk
        2. Lowest bus voltage    — undervoltage risk
        3. Highest trafo loading — trafo thermal risk
        4. Lowest trafo loading  — (informational, identifies lightly loaded trafos)
        5. Highest line loading  — line thermal risk
        6. Lowest line loading   — (informational)
 
    Violation flags use thresholds imported directly from violation_detector,
    so this summary stays in sync if thresholds are ever changed there.
 
    NOTE: Results are at default pandapower loading (no DWD profiles applied).
    This is a structural network scan — it identifies networks that are close
    to limits even at nominal operating point. Profile-driven worst-case
    scenarios belong in the comparison scenario test sections.
 
    Only includes networks with valid (non-None) results.
    """
    valid = [r for r in records if r.get("max_vm_pu") is not None]
    if not valid:
        print("\n  [violations summary] No valid results to summarise.")
        return
 
    W = 74  # table width
 
    print(f"\n{'='*W}")
    print(f"  VIOLATIONS SUMMARY")
    print(f"  Nominal loading only — no DWD profiles applied.")
    print(f"  Identifies structurally weak networks, not worst-case scenarios.")
    print(f"{'='*W}")
 
    def _table(title, rows, val_key, count_key, threshold, higher_is_bad,
               val_fmt=".4f", val_label="value", count_label="viol. buses"):
        """Generic ranked table printer."""
        filtered = [r for r in rows if r.get(val_key) is not None]
        if not filtered:
            print(f"\n  {title}: no data available.")
            return
        ranked = sorted(filtered,
                        key=lambda r: r[val_key] or 0.0,
                        reverse=higher_is_bad)
        n = min(top_n, len(ranked))
        print(f"\n  Top {n} — {title}:")
        print(f"  {'#':<4} {'Network':<44} {val_label:>12} {count_label:>13}")
        print(f"  {'-'*W}")
        for i, r in enumerate(ranked[:top_n], 1):
            val = r[val_key]
            cnt = r.get(count_key) or 0
            if higher_is_bad:
                flag = f"  <-- {'VIOLATION' if val > threshold else ''}"
            else:
                flag = f"  <-- {'VIOLATION' if val < threshold else ''}"
            flag = flag.rstrip()
            print(
                f"  {i:<4} {r['name']:<44} "
                f"{val:>12{val_fmt}} "
                f"{cnt:>13}{flag}"
            )
 
    # Table 1: Highest bus voltage
    _table(
        "Highest bus voltage (overvoltage risk)",
        valid, "max_vm_pu", "n_over_voltage",
        threshold=V_MAX, higher_is_bad=True,
        val_fmt=".4f", val_label="max vm_pu", count_label="over-V buses",
    )
 
    # Table 2: Lowest bus voltage
    _table(
        "Lowest bus voltage (undervoltage risk)",
        valid, "min_vm_pu", "n_under_voltage",
        threshold=V_MIN, higher_is_bad=False,
        val_fmt=".4f", val_label="min vm_pu", count_label="under-V buses",
    )
 
    # Table 3: Highest trafo loading
    trafo_valid = [r for r in valid if r.get("max_trafo_loading") is not None]
    _table(
        "Highest trafo loading (thermal risk)",
        trafo_valid, "max_trafo_loading", "n_overloaded_trafo",
        threshold=TRAFO_MAX_LOADING, higher_is_bad=True,
        val_fmt=".1f", val_label="max trafo %", count_label="overloaded",
    )
 
    # Table 4: Lowest trafo loading (informational)
    _table(
        "Lowest trafo loading (informational)",
        trafo_valid, "min_trafo_loading", "n_overloaded_trafo",
        threshold=0.0, higher_is_bad=False,
        val_fmt=".1f", val_label="min trafo %", count_label="overloaded",
    )
 
    # Table 5: Highest line loading
    line_valid = [r for r in valid if r.get("max_line_loading") is not None]
    _table(
        "Highest line loading (thermal risk)",
        line_valid, "max_line_loading", "n_overloaded_line",
        threshold=LINE_MAX_LOADING, higher_is_bad=True,
        val_fmt=".1f", val_label="max line %", count_label="overloaded",
    )
 
    # Table 6: Lowest line loading (informational)
    _table(
        "Lowest line loading (informational)",
        line_valid, "min_line_loading", "n_overloaded_line",
        threshold=0.0, higher_is_bad=False,
        val_fmt=".1f", val_label="min line %", count_label="overloaded",
    )
 
    # --- Quick count summary ---
    n_any  = sum(1 for r in valid if r.get("any_violations"))
    n_ov   = sum(1 for r in valid if (r.get("n_over_voltage")  or 0) > 0)
    n_uv   = sum(1 for r in valid if (r.get("n_under_voltage") or 0) > 0)
    n_traf = sum(1 for r in valid if (r.get("n_overloaded_trafo") or 0) > 0)
    n_line = sum(1 for r in valid if (r.get("n_overloaded_line")  or 0) > 0)
 
    print(f"\n  {'Networks with any violation:':<40} {n_any:>4} / {len(valid)}")
    print(f"  {'  of which overvoltage:':<40} {n_ov:>4}")
    print(f"  {'  of which undervoltage:':<40} {n_uv:>4}")
    print(f"  {'  of which trafo overload:':<40} {n_traf:>4}")
    print(f"  {'  of which line overload:':<40} {n_line:>4}")
    print(f"{'='*W}\n")
 

# ===========================================================================
# SECTION 5 - volt_var_control
# ===========================================================================
def _print_volt_var_summary(records: list, top_n: int = 10):
    """
    Prints ranked tables of Q(V) control effectiveness across networks.
    Called after both representative (3.2) and full (3.3) dry-run sweeps.
    """
    valid = [r for r in records if r.get("n_ders") is not None]
    if not valid:
        print("\n  [volt_var summary] No valid results.")
        return

    W = 74
    print(f"\n{'='*W}")
    print(f"  VOLT-VAR CONTROL SUMMARY")
    print("  Stressed loading: adaptive by network family")
    print("    - MV/MVLV: sgen 90% of sn_mva, load 20% of nominal")
    print("    - LV/Synthetic/Dickert: sgen 50% of sn_mva, load 20% of nominal")
    print(f"{'='*W}")

    def _table(title, rows, val_key, val_fmt=".4f", val_label="value",
               reverse=True, filter_fn=None):
        filtered = [r for r in rows if r.get(val_key) is not None]
        if filter_fn:
            filtered = [r for r in filtered if filter_fn(r)]
        if not filtered:
            print(f"\n  {title}: no data.")
            return
        ranked = sorted(filtered,
            key=lambda r: (r[val_key] if (r[val_key] == r[val_key]) else 0.0),
            reverse=reverse)
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

    # Outcome counts
    n_resolved = sum(1 for r in valid if r.get("violations_resolved"))
    n_reduced  = sum(1 for r in valid if r.get("v_reduced") and not r.get("violations_resolved"))
    n_no_effect = sum(1 for r in valid
                      if r.get("n_ov_pre") and not r.get("v_reduced") and not r.get("violations_resolved"))
    n_no_viol  = sum(1 for r in valid if not r.get("n_ov_pre"))
    n_failed   = sum(1 for r in records if r.get("n_ders") is None)

    print(f"\n  Voltage violation outcomes:")
    print(f"  {'No voltage violations pre-control:':<42} {n_no_viol:>4} / {len(valid)}")
    print(f"  {'Fully resolved by Q(V):':<42} {n_resolved:>4} / {len(valid)}")
    print(f"  {'Partially reduced by Q(V):':<42} {n_reduced:>4} / {len(valid)}")
    print(f"  {'No effect (DERs in deadband or no headroom):':<42} {n_no_effect:>4} / {len(valid)}")
    print(f"  {'Failed (PF non-convergence):':<42} {n_failed:>4} / {len(records)}")
    print(f"{'='*W}\n")

def _print_weather_vv_summary(records: list, top_n: int = 10):
    """
    Summarises weather-driven Volt-Var results for extreme days.
    records: list of dicts, one per (network, extreme day).
    """
    if not records:
        print("\n  [weather summary] No records.")
        return

    W = 88
    print(f"\n{'='*W}")
    print("  WEATHER-DRIVEN VOLT-VAR SUMMARY (extreme days only)")
    print(f"{'='*W}")

    # Helper: sort with safe fallback
    def _safe_key(r, key, default=-1):
        v = r.get(key)
        return v if isinstance(v, (int, float)) else default

    # Table 1: biggest total violation reduction (pre_total - post_total)
    ranked = sorted(records,
                    key=lambda r: _safe_key(r, "total_reduction", -1),
                    reverse=True)
    print(f"\n  Top {min(top_n, len(ranked))} — Biggest total violation reduction:")
    print(f"  {'#':<3} {'Network':<28} {'Day':<10} {'Pre':>6} {'Post':>6} {'Δ':>6} {'ms':>8}")
    print(f"  {'-'*W}")
    for i, r in enumerate(ranked[:top_n], 1):
        print(f"  {i:<3} {r['name']:<28} {r['day']:<10} "
              f"{r['total_pre']:>6} {r['total_post']:>6} {r['total_reduction']:>6} "
              f"{r['t_ms']:>8.1f}")

    # Table 2: worst remaining violations post-control
    ranked_post = sorted(records,
                         key=lambda r: _safe_key(r, "total_post", -1),
                         reverse=True)
    print(f"\n  Top {min(top_n, len(ranked_post))} — Most violations remaining post-control:")
    print(f"  {'#':<3} {'Network':<28} {'Day':<10} {'Post':>6} {'OV':>4} {'UV':>4} {'ms':>8}")
    print(f"  {'-'*W}")
    for i, r in enumerate(ranked_post[:top_n], 1):
        print(f"  {i:<3} {r['name']:<28} {r['day']:<10} "
              f"{r['total_post']:>6} {r['ov_post']:>4} {r['uv_post']:>4} "
              f"{r['t_ms']:>8.1f}")

    # Aggregate stats
    n = len(records)
    n_reduced = sum(1 for r in records if r["total_reduction"] > 0)
    mean_ms = sum(r["t_ms"] for r in records) / n
    print(f"\n  {'Records:':<25} {n}")
    print(f"  {'Days with reduction:':<25} {n_reduced}")
    print(f"  {'Mean per-day runtime:':<25} {mean_ms:.1f} ms")

def _print_coordinator_summary(records: list, top_n: int = 10):
    """
    Prints ranked tables of sensitivity coordinator effectiveness.
    Called after the representative-network and all-199-network sweeps.
 
    Each record dict must contain:
        name, n_ders, n_ov_pre, n_ov_post, violations_resolved,
        max_dq_corr, q_adj_max, t_ms, post_pf_ok
    """
    valid = [r for r in records if r.get("n_ders") is not None]
    if not valid:
        print("\n  [coord summary] No valid results.")
        return
 
    W = 88
    print(f"\n{'='*W}")
    print("  SENSITIVITY COORDINATOR SUMMARY")
    print("  Stress: sgen 90% of sn_mva, load 20% of nominal (all families)")
    print(f"{'='*W}")
     
    def _table(title, rows, val_key, val_fmt, val_label, reverse=True):
        filtered = [r for r in rows if r.get(val_key) is not None]
        if not filtered:
            print(f"\n  {title}: no data.")
            return
        ranked = sorted(
            filtered,
            key=lambda r: (r[val_key] if r[val_key] == r[val_key] else 0.0),
            reverse=reverse,
        )
        n = min(top_n, len(ranked))
        print(f"\n  Top {n} — {title}:")
        print(f"  {'#':<4} {'Network':<42} {val_label:>12} {'DERs':>6}")
        print(f"  {'-'*W}")
        for i, r in enumerate(ranked[:n], 1):
            print(
                f"  {i:<4} {r['name']:<42} "
                f"{r[val_key]:>12{val_fmt}} "
                f"{r.get('n_ders') or 0:>6}"
            )
 
    _table(
        "Most overvoltage buses pre-control",
        valid, "n_ov_pre", "d", "buses OV", reverse=True,
    )
    _table(
        "Largest coordinator correction over Item 2 (max|dQ|)",
        valid, "max_dq_corr", ".4f", "dQ MVAr", reverse=True,
    )
    _table(
        "Slowest timesteps (wall-clock)",
        valid, "t_ms", ".1f", "t_ms", reverse=True,
    )
 
    # Outcome counts
    has_pre      = [r for r in valid if (r.get("n_ov_pre") or 0) > 0]
    n_resolved   = sum(1 for r in has_pre if r.get("violations_resolved"))
    n_partial    = sum(1 for r in has_pre
                       if not r.get("violations_resolved")
                       and r.get("post_pf_ok")
                       and (r.get("n_ov_post") or 0) < r["n_ov_pre"])
    n_no_imp     = sum(1 for r in has_pre
                       if not r.get("violations_resolved")
                       and r.get("post_pf_ok")
                       and (r.get("n_ov_post") or 0) >= r["n_ov_pre"])
    n_no_viol    = sum(1 for r in valid if not (r.get("n_ov_pre") or 0))
    n_pf_failed  = sum(1 for r in records if r.get("n_ders") is None)
 
    print(f"\n  Voltage violation outcomes:")
    print(f"  {'No pre-violations:':<44} {n_no_viol:>4} / {len(valid)}")
    print(f"  {'Fully resolved by coordinator:':<44} {n_resolved:>4} / {len(valid)}")
    print(f"  {'Partially reduced:':<44} {n_partial:>4} / {len(valid)}")
    print(f"  {'No improvement:':<44} {n_no_imp:>4} / {len(valid)}")
    print(f"  {'Failed (PF non-convergence or exception):':<44} {n_pf_failed:>4} / {len(records)}")
    print(f"{'='*W}\n")

def run_volt_var_tests(
        verbose:      bool = False,
        only:         list = None,
        arduino_port: str  = None,
        only_hw: bool = False,
) -> list:
    """
    Section 5 -- Volt-Var Q(V) control tests.
 
    Subsections
    -----------
    5.1  QVCharacteristic unit tests -- no hardware, no pandapower.
    5.2  VoltVarController dry_run   -- 9 representative networks + summary.
    5.3  VoltVarController dry_run   -- all 199+ networks + ranked summary.
    5.4  Hardware                    -- 9 representative networks (skipped if no --arduino-port).
 
    Run hardware section:
        python test_suite.py --section volt_var_control --arduino-port /dev/ttyACM0
    """
    from volt_var_controller import (
        QVCharacteristic, VoltVarController, VoltVarResult,
        ArduinoSerialInterface, ArduinoProtocolError, SerialTimeoutError,
        U1_PU, U2_PU, U3_PU, U4_PU, Q_RATIO,
    )
    import simbench as sb
 
    cases = []
    vv_rep_records = []
    REPRESENTATIVE_NETWORKS_VV = [
            ("sb_mv_rural",    lambda: sb.get_simbench_net("1-MV-rural--2-sw"),              "1-MV-rural--2-sw"),
            ("sb_lv_rural",    lambda: sb.get_simbench_net("1-LV-rural1--0-sw"),             "1-LV-rural1--0-sw"),
            ("sb_mvlv_rural",  lambda: sb.get_simbench_net("1-MVLV-rural-all-0-sw"),         "1-MVLV-rural-all-0-sw"),
            ("cigre_mv",       lambda: pn.create_cigre_network_mv(with_der="pv_wind"),       "cigre_mv"),
            ("cigre_lv",       lambda: pn.create_cigre_network_lv(),                         "cigre_lv"),
            ("kerber_std",     lambda: pn.create_kerber_landnetz_kabel_1(),                  "kerber_landnetz_kabel_1"),
            ("kerber_extreme", lambda: pn.kb_extrem_landnetz_kabel(),                        "kb_extrem_landnetz_kabel"),
            ("synthetic_lv",   lambda: pn.create_synthetic_voltage_control_lv_network("rural_1"), "synthetic_lv_rural_1"),
            ("dickert",        lambda: pn.create_dickert_lv_network("short","cable","single","good"), "dickert"),
        ]

    if not only_hw:

        # -------------------------------------------------------------------
        # 5.1  QVCharacteristic unit tests
        # -------------------------------------------------------------------
        print("\n  [1/4] QVCharacteristic unit tests")
        name = "qv_characteristic"
        if not (only and not any(s in name for s in only)):
            tc = TestCase(name)
            t0 = time.time()
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
    
                import warnings as _w
                with _w.catch_warnings(record=True):
                    _w.simplefilter("always")
                    r_nan = qv.compute_setpoint(0.90, float("nan"))
                tc.record("nan_p_returns_zero", abs(r_nan) < tol, f"got {r_nan!r}")
    
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
            tc.duration = time.time() - t0
            cases.append(tc)
            print_case(tc, verbose)
    
        # -------------------------------------------------------------------
        # 5.2  VoltVarController dry_run -- 9 representative networks
        # -------------------------------------------------------------------

        print(f"\n  [2/4] VoltVarController dry_run -- {len(REPRESENTATIVE_NETWORKS_VV)} representative networks")

        for net_name, loader, label in REPRESENTATIVE_NETWORKS_VV:
            if only and not any(s in net_name for s in only):
                continue
            tc = TestCase(f"vv_dry_{net_name}")
            t0 = time.time()
            rec = {"name": label}
            try:
                net  = loader()
                net.sgen.p_mw   = net.sgen.sn_mva * 0.90
                net.load.p_mw   = net.load.p_mw   * 0.20
                net.load.q_mvar = net.load.q_mvar  * 0.20
                ctrl = VoltVarController(net, interface=None, dry_run=True)
                ctrl.configure()
                if ctrl.n_ders == 0:
                    tc.record("skipped_no_der", True, "No controllable DERs in network")
                    tc.skipped = True
                    cases.append(tc)
                    vv_rep_records.append(rec)
                    print_case(tc, verbose)
                    continue

                tc.record("n_ders_positive", ctrl.n_ders > 0, f"n_ders={ctrl.n_ders}")
                runpp_kwargs = {}
                if "synthetic" in label or "lv" in label.lower() or "dickert" in label.lower():
                    runpp_kwargs = {
                        "algorithm": "bfsw",
                        "max_iteration": 30,
                        "init": "flat",
                    }

                result = ctrl.run_timestep(runpp_kwargs=runpp_kwargs)
                if not result.converged_pre:
                    tc.record("pre_converged", False, "runpp() did not converge")
                    tc.skipped = True
                    cases.append(tc)
                    vv_rep_records.append(rec)
                    print_case(tc, verbose)
                    continue

                tc.record("pre_converged",  result.converged_pre)
                tc.record("post_converged", result.converged_post)
                tc.record("q_length",       len(result.q_setpoints) == ctrl.n_ders)
                tc.record("q_finite",       np.all(np.isfinite(result.q_setpoints.values)))
                tc.record("q_applied",
                    np.allclose(
                        net.sgen.loc[ctrl.sgen_indices, "q_mvar"].values,
                        result.q_setpoints.values))

                rec.update({
                    "n_ders":             ctrl.n_ders,
                    "n_ov_pre":           result.report_pre.n_over_voltage,
                    "n_ov_post":          result.report_post.n_over_voltage if result.report_post else None,
                    "worst_v_pre":        result.report_pre.worst_over_voltage,
                    "worst_v_post":       result.report_post.worst_over_voltage if result.report_post else None,
                    "violations_resolved": result.violations_resolved,
                    "v_reduced":          result.voltage_violations_reduced,
                    "q_max":              float(np.abs(result.q_setpoints.values).max()) if ctrl.n_ders > 0 else 0.0,
                    "q_total":            float(np.abs(result.q_setpoints.values).sum()) if ctrl.n_ders > 0 else 0.0,
                })
            except Exception:
                tc.error = traceback.format_exc()
                rec.update({"n_ders": None, "n_ov_pre": None, "n_ov_post": None,
                            "worst_v_pre": None, "worst_v_post": None,
                            "violations_resolved": None, "v_reduced": None,
                            "q_max": None, "q_total": None})
            tc.duration = time.time() - t0
            cases.append(tc)
            vv_rep_records.append(rec)
            print_case(tc, verbose)

        _print_volt_var_summary(vv_rep_records)

        # -------------------------------------------------------------------
        # 5.2b  VoltVarController weather-driven (extreme days only)
        # -------------------------------------------------------------------
        print("\n  [2b/4] VoltVarController weather-driven -- extreme days (9 representative networks)")

        from profile_builder import build_annual_profiles
        weather_records = []

        def _slice_day(df: pd.DataFrame, day_str: str):
            if df is None or df.empty or day_str is None:
                return None
            tz = df.index.tz
            day_start = pd.Timestamp(day_str, tz=tz)
            day_end = day_start + pd.Timedelta(days=1)
            return df.loc[(df.index >= day_start) & (df.index < day_end)]

        extreme_keys = [
            ("max_der",  "Max DER day"),
            ("min_der",  "Min DER day"),
            ("max_load", "Max load day"),
            ("min_load", "Min load day"),
        ]

        for net_name, loader, label in REPRESENTATIVE_NETWORKS_VV:
            if only and not any(s in net_name for s in only):
                continue

            tc = TestCase(f"vv_weather_{net_name}")
            t0 = time.time()
            try:
                net = loader()

                # Build annual profiles (DWD default)
                prof = build_annual_profiles(
                    net, label,
                    data_dir=DWD_DATA_DIR,
                    simbench_code=label if "1-" in label else None
                )

                ctrl = VoltVarController(net, interface=None, dry_run=True)
                ctrl.configure()

                if ctrl.n_ders == 0:
                    tc.record("skipped_no_der", True, "No controllable DERs in network")
                    tc.skipped = True
                    cases.append(tc)
                    print_case(tc, verbose)
                    continue

                ed = prof.get("extreme_days", {})

                # Run only the four extreme days
                for key, desc in extreme_keys:
                    day_str = ed.get(key)
                    if day_str is None:
                        tc.record(f"{key}_missing", False, f"{desc} not found in profiles")
                        continue

                    pv_day   = _slice_day(prof.get("pv"), day_str)
                    wind_day = _slice_day(prof.get("wind"), day_str)
                    load_day = _slice_day(prof.get("load"), day_str)

                    if load_day is None or load_day.empty:
                        tc.record(f"{key}_load_empty", False, f"No load data for {day_str}")
                        continue

                    # --- timing + violation metrics for this extreme day ---
                    t_start_day = time.perf_counter()
                    ov_pre = uv_pre = ov_post = uv_post = 0

                    for ts in load_day.index:
                        pv_row   = pv_day.loc[ts] if pv_day is not None and not pv_day.empty else pd.Series(dtype=float)
                        wind_row = wind_day.loc[ts] if wind_day is not None and not wind_day.empty else pd.Series(dtype=float)

                        p_sgen = (
                            pv_row.reindex(net.sgen.index, fill_value=0.0)
                            + wind_row.reindex(net.sgen.index, fill_value=0.0)
                        )
                        net.sgen.p_mw = p_sgen.values
                        net.sgen.q_mvar = 0.0

                        p_load = load_day.loc[ts].reindex(net.load.index, fill_value=0.0)
                        net.load.p_mw = p_load.values
                        net.load.q_mvar = 0.0

                        result = ctrl.run_timestep()
                        if result.report_pre:
                            ov_pre += result.report_pre.n_over_voltage
                            uv_pre += result.report_pre.n_under_voltage
                        if result.report_post:
                            ov_post += result.report_post.n_over_voltage
                            uv_post += result.report_post.n_under_voltage

                    t_ms = (time.perf_counter() - t_start_day) * 1e3
                    total_pre = ov_pre + uv_pre
                    total_post = ov_post + uv_post
                    total_reduction = total_pre - total_post

                    weather_records.append({
                        "name": label,
                        "day": day_str,
                        "key": key,
                        "t_ms": t_ms,
                        "ov_pre": ov_pre,
                        "uv_pre": uv_pre,
                        "ov_post": ov_post,
                        "uv_post": uv_post,
                        "total_pre": total_pre,
                        "total_post": total_post,
                        "total_reduction": total_reduction,
                    })
                tc.record("weather_extreme_days_complete", True)

            except Exception:
                tc.error = traceback.format_exc()

            tc.duration = time.time() - t0
            cases.append(tc)
            print_case(tc, verbose)
        
        _print_weather_vv_summary(weather_records)

        # -------------------------------------------------------------------
        # 5.3  VoltVarController dry_run -- all 199+ networks + summary
        # -------------------------------------------------------------------
        print(f"\n  [3/4] VoltVarController dry_run -- all in-scope networks")
        import simbench as sb
        vv_all_records = []

        def _run_vv_one(tc, name, loader_fn):
            """Load network, run one dry_run timestep, return stats dict."""
            try:
                net  = loader_fn()
                net.sgen.p_mw   = net.sgen.sn_mva * 0.90
                net.load.p_mw   = net.load.p_mw   * 0.20
                net.load.q_mvar = net.load.q_mvar  * 0.20
                ctrl = VoltVarController(net, interface=None, dry_run=True)
                ctrl.configure()
                if ctrl.n_ders == 0:
                    tc.record("skipped_no_der", True, "No controllable DERs in network")
                    tc.skipped = True
                    return {
                        "name": name,
                        "n_ders": 0,
                        "n_ov_pre": None, "n_ov_post": None,
                        "worst_v_pre": None, "worst_v_post": None,
                        "violations_resolved": None, "v_reduced": None,
                        "q_max": 0.0, "q_total": 0.0
                    }
                runpp_kwargs = {}
                if "synthetic" in name or "lv" in name.lower():
                    runpp_kwargs = {"algorithm": "bfsw", "max_iteration": 30, "init": "flat"}

                result = ctrl.run_timestep(runpp_kwargs=runpp_kwargs)
                tc.record("pre_converged",  result.converged_pre)
                tc.record("q_length",       len(result.q_setpoints) == ctrl.n_ders)

                return {
                    "name":                name,
                    "n_ders":              ctrl.n_ders,
                    "n_ov_pre":            result.report_pre.n_over_voltage,
                    "n_ov_post":           result.report_post.n_over_voltage if result.report_post else None,
                    "worst_v_pre":         result.report_pre.worst_over_voltage,
                    "worst_v_post":        result.report_post.worst_over_voltage if result.report_post else None,
                    "violations_resolved": result.violations_resolved,
                    "v_reduced":           result.voltage_violations_reduced,
                    "q_max":               float(np.abs(result.q_setpoints.values).max()) if ctrl.n_ders > 0 else 0.0,
                    "q_total":             float(np.abs(result.q_setpoints.values).sum()) if ctrl.n_ders > 0 else 0.0,
                }
            except Exception:
                tc.error = traceback.format_exc()
                return {"name": name, **{k: None for k in [
                    "n_ders","n_ov_pre","n_ov_post","worst_v_pre","worst_v_post",
                    "violations_resolved","v_reduced","q_max","q_total"]}}

        # SimBench 156
        for code in IN_SCOPE_SIMBENCH_CODES:
            if only and not any(s in code for s in only):
                continue
            tc = TestCase(f"vv_{code}")
            t0 = time.time()
            rec = _run_vv_one(tc, code, lambda c=code: sb.get_simbench_net(c))
            tc.duration = time.time() - t0
            cases.append(tc)
            vv_all_records.append(rec)
            print_case(tc, verbose)

        # CIGRE MV + LV
        for name, loader in [
            ("cigre_mv_with_der", lambda: pn.create_cigre_network_mv(with_der="pv_wind")),
            ("cigre_lv",          lambda: pn.create_cigre_network_lv()),
        ]:
            if only and not any(s in name for s in only):
                continue
            tc = TestCase(f"vv_{name}")
            t0 = time.time()
            rec = _run_vv_one(tc, name, loader)
            tc.duration = time.time() - t0
            cases.append(tc)
            vv_all_records.append(rec)
            print_case(tc, verbose)

        # Kerber 17
        for name, fn_name in ALL_KERBER_CASES:
            if only and not any(s in name for s in only):
                continue
            tc = TestCase(f"vv_{name}")
            t0 = time.time()
            rec = _run_vv_one(tc, name, lambda f=fn_name: getattr(pn, f)())
            tc.duration = time.time() - t0
            cases.append(tc)
            vv_all_records.append(rec)
            print_case(tc, verbose)

        # Synthetic LV 5
        for network_class in ALL_SYNTHETIC_LV_CASES:
            name = f"synthetic_lv_{network_class}"
            if only and not any(s in name for s in only):
                continue
            tc = TestCase(f"vv_{name}")
            t0 = time.time()
            rec = _run_vv_one(tc, name,
                lambda c=network_class: pn.create_synthetic_voltage_control_lv_network(c))
            tc.duration = time.time() - t0
            cases.append(tc)
            vv_all_records.append(rec)
            print_case(tc, verbose)

        # Dickert 18
        for name, feeders_range, linetype, customer, case in ALL_DICKERT_CASES:
            if only and not any(s in name for s in only):
                continue
            tc = TestCase(f"vv_{name}")
            t0 = time.time()
            try:
                rec = _run_vv_one(tc, name,
                    lambda fr=feeders_range, lt=linetype, cu=customer, ca=case:
                        pn.create_dickert_lv_network(fr, lt, cu, ca))
            except ValueError as e:
                if "no dickert network" in str(e).lower():
                    tc.skipped = True
                rec = {"name": name, **{k: None for k in [
                    "n_ders","n_ov_pre","n_ov_post","worst_v_pre","worst_v_post",
                    "violations_resolved","v_reduced","q_max","q_total"]}}
            tc.duration = time.time() - t0
            cases.append(tc)
            vv_all_records.append(rec)
            print_case(tc, verbose)

        _print_volt_var_summary(vv_all_records)

    # -------------------------------------------------------------------
    # 5.5  VoltVarController hardware -- 9 representative networks
    #      Skipped entirely if --arduino-port not supplied.
    # -------------------------------------------------------------------
    hw_label = arduino_port if arduino_port else "SKIPPED -- pass --arduino-port"
    print(f"\n  [4/4] VoltVarController hardware  ({hw_label})")

    if arduino_port:
        for net_name, loader, label in REPRESENTATIVE_NETWORKS_VV:
            if only and not any(s in net_name for s in only):
                continue
            tc = TestCase(f"vv_hw_{net_name}")
            t0 = time.time()
            try:
                net = loader()
                net.sgen.p_mw   = net.sgen.sn_mva * 0.90
                net.load.p_mw   = net.load.p_mw   * 0.20
                net.load.q_mvar = net.load.q_mvar  * 0.20

                with ArduinoSerialInterface(port=arduino_port) as arduino:
                    ctrl = VoltVarController(net, interface=arduino, dry_run=False)
                    ctrl.configure()
                    if ctrl.n_ders == 0:
                        tc.record("hw_no_der", True, "No controllable DERs in network")
                        tc.skipped = True
                        cases.append(tc)
                        print_case(tc, verbose)
                        continue
                    result = None

                    if "synthetic" in label.lower():
                        # reset q_mvar to zero to avoid NaN propagation
                        net.sgen["q_mvar"] = 0.0
                        net.load["q_mvar"] = net.load["q_mvar"].fillna(0.0)

                        # remove any NaNs/Inf in p_mw/sn_mva before scaling
                        net.sgen["p_mw"] = net.sgen["p_mw"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                        net.load["p_mw"] = net.load["p_mw"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                        net.sgen["sn_mva"] = net.sgen["sn_mva"].replace([np.inf, -np.inf], np.nan).fillna(net.sgen["p_mw"])

                        # lower stress for synthetic LV
                        net.sgen.p_mw = net.sgen.sn_mva * 0.20
                        net.load.p_mw = net.load.p_mw * 0.40

                        runpp_candidates = [
                            {"algorithm": "bfsw", "max_iteration": 80, "init": "flat"},
                            {"algorithm": "nr", "max_iteration": 50, "init": "flat"},
                        ]
                    else:
                        runpp_candidates = [None]   # run once with default settings

                    for kwargs in runpp_candidates:
                        result = ctrl.run_timestep(runpp_kwargs=kwargs if kwargs else None)
                        if result.converged_pre:
                            break
                tc.record("hw_pre_converged",  result.converged_pre)
                tc.record("hw_post_converged", result.converged_post)
                tc.record("hw_q_length",       len(result.q_setpoints) == ctrl.n_ders)
                tc.record("hw_t_exchange_positive", result.t_exchange_ms > 0,
                    f"t_exchange_ms={result.t_exchange_ms:.1f}")
                tc.record("hw_t_exchange_reasonable", result.t_exchange_ms < 2000,
                    f"t_exchange_ms={result.t_exchange_ms:.1f}ms")
                tc.record("hw_n_retries_reported", result.n_retries >= 0)
                print(f"         {label}: exchange={result.t_exchange_ms:.0f}ms  "
                      f"n_ders={ctrl.n_ders}  retries={result.n_retries}  "
                      f"resolved={result.violations_resolved}")
            except Exception:
                tc.error = traceback.format_exc()
            tc.duration = time.time() - t0
            cases.append(tc)
            print_case(tc, verbose)
        return cases

    return cases

# ===========================================================================
# SECTION 6 - Sensitivity Coordinator
# ===========================================================================

def run_sensitivity_coordinator_tests(
        verbose:      bool = False,
        only:         list = None,
        arduino_port: str  = None,
        only_hw:      bool = False,
) -> list:
    """
    Section 6 — Sensitivity Coordinator tests.
 
    Subsections
    -----------
    6.1  CoordinatorResult unit tests      — no hardware, no pandapower.
    6.2  Per-network dry-run               — 9 representative networks.
         Combines: construction checks, coordinate() guard conditions,
         output validity, and run_coordinated_timestep() integration.
         Prints coordinator summary.
    6.3  Hardware                          — 9 representative networks
                                            (skipped if no --arduino-port).
 
    Run with hardware:
        python test_suite.py --section sensitivity_coordinator
                             --arduino-port /dev/ttyACM0
    """
    import warnings as _warnings
    from sensitivity_coordinator import (
        SensitivityCoordinator,
        CoordinatorResult,
        run_coordinated_timestep,
    )
    from volt_var_controller import (
        VoltVarController, Q_RATIO,
        ArduinoSerialInterface,
    )
    from violation_detector import ViolationReport
 
    cases            = []
    coord_rep_records = []
 
    REPRESENTATIVE_NETWORKS_COORD = [
        ("sb_mv_rural",    lambda: sb.get_simbench_net("1-MV-rural--2-sw"),                    "1-MV-rural--2-sw"),
        ("sb_lv_rural",    lambda: sb.get_simbench_net("1-LV-rural1--0-sw"),                   "1-LV-rural1--0-sw"),
        ("sb_mvlv_rural",  lambda: sb.get_simbench_net("1-MVLV-rural-all-0-sw"),               "1-MVLV-rural-all-0-sw"),
        ("cigre_mv",       lambda: pn.create_cigre_network_mv(with_der="pv_wind"),             "cigre_mv"),
        ("cigre_lv",       lambda: pn.create_cigre_network_lv(),                               "cigre_lv"),
        ("kerber_std",     lambda: pn.create_kerber_landnetz_kabel_1(),                        "kerber_landnetz_kabel_1"),
        ("kerber_extreme", lambda: pn.kb_extrem_landnetz_kabel(),                              "kb_extrem_landnetz_kabel"),
        ("synthetic_lv",   lambda: pn.create_synthetic_voltage_control_lv_network("rural_1"), "synthetic_lv_rural_1"),
        ("dickert",        lambda: pn.create_dickert_lv_network("short","cable","single","good"), "dickert"),
    ]
 
    if not only_hw:
 
        # -------------------------------------------------------------------
        # 6.1  CoordinatorResult unit tests  (no network, no pandapower)
        # -------------------------------------------------------------------
        print("\n  [1/3] CoordinatorResult unit tests")
 
        name = "coordinator_result_unit"
        if not (only and not any(s in name for s in only)):
            tc = TestCase(name)
            t0 = time.time()
            try:
                # Fabricate two minimal ViolationReport objects.
                # over_voltage DataFrame requires columns vm_pu, deviation_pu.
                _ov_df    = pd.DataFrame(
                    {"vm_pu": [1.06], "deviation_pu": [0.01]}, index=[5]
                )
                _clean_df = pd.DataFrame(columns=["vm_pu", "deviation_pu"])
                _rep_viol  = ViolationReport(
                    over_voltage=_ov_df, any_violations=True, converged=True
                )
                _rep_clean = ViolationReport(
                    over_voltage=_clean_df, any_violations=False, converged=True
                )
 
                q_ini = np.array([0.10, -0.20, 0.05])
                q_adj = np.array([0.15, -0.18, 0.08])
 
                cr = CoordinatorResult(
                    report_pre=_rep_viol, report_post=_rep_clean,
                    q_initial=q_ini, q_adjusted=q_adj,
                    curtailment_needed=False, post_pf_ok=True,
                )
 
                # dq_correction arithmetic
                tc.record(
                    "dq_correction_arithmetic",
                    np.allclose(cr.dq_correction, q_adj - q_ini),
                    f"expected {q_adj - q_ini}, got {cr.dq_correction}",
                )
 
                # violations_resolved True  (pre violated, post clean, post_pf_ok=True)
                tc.record(
                    "violations_resolved_true",
                    cr.violations_resolved is True,
                    "pre=violated, post=clean, post_pf_ok=True → must be True",
                )
 
                # violations_resolved False when post_pf_ok=False  (branch 1)
                _cr_nopf = CoordinatorResult(
                    report_pre=_rep_viol, report_post=_rep_clean,
                    q_initial=q_ini, q_adjusted=q_adj,
                    curtailment_needed=False, post_pf_ok=False,
                )
                tc.record(
                    "violations_resolved_false_post_pf_failed",
                    _cr_nopf.violations_resolved is False,
                    "post_pf_ok=False → must return False",
                )
 
                # violations_resolved False when report_post=None  (branch 2)
                _cr_nopost = CoordinatorResult(
                    report_pre=_rep_viol, report_post=None,
                    q_initial=q_ini, q_adjusted=q_adj,
                    curtailment_needed=False, post_pf_ok=True,
                )
                tc.record(
                    "violations_resolved_false_report_post_none",
                    _cr_nopost.violations_resolved is False,
                    "report_post=None → must return False",
                )
 
                # violations_resolved False when pre has no violations (edge case)
                _cr_nopre = CoordinatorResult(
                    report_pre=_rep_clean, report_post=_rep_clean,
                    q_initial=q_ini, q_adjusted=q_adj,
                    curtailment_needed=False, post_pf_ok=True,
                )
                tc.record(
                    "violations_resolved_false_no_pre_violations",
                    _cr_nopre.violations_resolved is False,
                    "no pre-violations → violations_resolved must be False",
                )
 
                # summary() returns non-empty string containing "CoordResult"
                summary_str = cr.summary()
                tc.record(
                    "summary_returns_string",
                    isinstance(summary_str, str)
                    and "CoordResult" in summary_str
                    and len(summary_str) > 0,
                    f"summary()={summary_str!r}",
                )
 
                # Default field values
                _cr_def = CoordinatorResult(
                    report_pre=_rep_clean, report_post=_rep_clean,
                    q_initial=np.zeros(2), q_adjusted=np.zeros(2),
                    curtailment_needed=False, post_pf_ok=True,
                )
                tc.record(
                    "default_n_retries_zero",
                    _cr_def.n_retries == 0,
                    f"n_retries={_cr_def.n_retries}",
                )
                tc.record(
                    "default_t_total_ms_zero",
                    _cr_def.t_total_ms == 0.0,
                    f"t_total_ms={_cr_def.t_total_ms}",
                )
 
            except Exception:
                tc.error = traceback.format_exc()
            tc.duration = time.time() - t0
            cases.append(tc)
            print_case(tc, verbose)
 
        # -------------------------------------------------------------------
        # 6.2  Per-network dry-run — 9 representative networks
        #      Each TestCase covers: construction checks, guard conditions,
        #      output validity, and run_coordinated_timestep() integration.
        # -------------------------------------------------------------------
        print(f"\n  [2/3] Coordinator dry_run — "
              f"{len(REPRESENTATIVE_NETWORKS_COORD)} representative networks")
 
        for net_name, loader, label in REPRESENTATIVE_NETWORKS_COORD:
            if only and not any(s in net_name for s in only):
                continue
            tc = TestCase(f"coord_dry_{net_name}")
            t0 = time.time()
            rec = {"name": label}
            try:
                net = loader()
 
                # Overvoltage stress: high generation + low demand.
                # Same condition as volt_var Section 5.2 for comparability.
                net.sgen.p_mw   = net.sgen.sn_mva * 0.90
                net.load.p_mw   = net.load.p_mw   * 0.20
                net.load.q_mvar = net.load.q_mvar  * 0.20
                net.sgen["q_mvar"] = 0.0   # q=0 invariant for coordinate()
 
                # LV runpp_kwargs — bfsw for radial LV networks
                runpp_kwargs = {}
                if ("synthetic" in label or "lv" in label.lower()
                        or "dickert" in label.lower()
                        or "kerber" in label.lower()):
                    runpp_kwargs = {
                        "algorithm": "bfsw",
                        "max_iteration": 30,
                        "init": "flat",
                    }
 
                # Suppress UserWarning for p_installed <= 0 on some sgens
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore", UserWarning)
                    ctrl = VoltVarController(net, interface=None, dry_run=True)
 
                if ctrl.n_ders == 0:
                    tc.record("skipped_no_der", True,
                              "No controllable DERs in network")
                    tc.skipped = True
                    cases.append(tc)
                    coord_rep_records.append(rec)
                    print_case(tc, verbose)
                    continue
 
                # --- Construction checks -----------------------------------
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore", UserWarning)
                    coord = SensitivityCoordinator(net, ctrl)
 
                tc.record(
                    "curtailment_init",
                    coord.curtailment_needed is False,
                    f"curtailment_needed={coord.curtailment_needed}",
                )
                tc.record(
                    "qmax_formula",
                    np.allclose(coord._q_max, Q_RATIO * ctrl.p_installed_mw,
                                atol=1e-9),
                    f"max|_q_max - Q_RATIO×p_inst|="
                    f"{np.abs(coord._q_max - Q_RATIO * ctrl.p_installed_mw).max():.2e}",
                )
                tc.record(
                    "qmax_shape",
                    coord._q_max.shape == (ctrl.n_ders,),
                    f"shape={coord._q_max.shape}, expected=({ctrl.n_ders},)",
                )
 
                # Pre-PF needed for Jacobian (guard tests + integration below)
                pp.runpp(net, voltage_depend_loads=False, **runpp_kwargs)
 
                # --- Guard 1: wrong-length q_initial → ValueError -----------
                _raised_ve = False
                try:
                    coord.coordinate(np.zeros(ctrl.n_ders + 1))
                except ValueError:
                    _raised_ve = True
                except Exception:
                    pass
                tc.record(
                    "guard_valueerror_wrong_length",
                    _raised_ve,
                    f"expected ValueError for length {ctrl.n_ders + 1} != {ctrl.n_ders}",
                )
 
                # --- Guard 2: non-zero q_mvar → RuntimeError ---------------
                net.sgen.loc[ctrl.sgen_indices, "q_mvar"] = 1.0
                _raised_rte = False
                try:
                    coord.coordinate(np.zeros(ctrl.n_ders))
                except RuntimeError:
                    _raised_rte = True
                except Exception:
                    pass
                tc.record(
                    "guard_runtimeerror_nonzero_qmvar",
                    _raised_rte,
                    "expected RuntimeError when q_mvar != 0 before coordinate()",
                )
                # Restore clean state before integration
                net.sgen.loc[ctrl.sgen_indices, "q_mvar"] = 0.0
 
                # --- Integration: run_coordinated_timestep() ---------------
                kw = {"voltage_depend_loads": False}
                kw.update(runpp_kwargs)
 
                result = run_coordinated_timestep(net, ctrl, coord,
                                                  runpp_kwargs=kw)
 
                tc.record(
                    "post_pf_ok",
                    result.post_pf_ok,
                    "post-PF runpp() must converge",
                )
                tc.record(
                    "q_adj_finite",
                    bool(np.isfinite(result.q_adjusted).all()),
                    f"{int((~np.isfinite(result.q_adjusted)).sum())} non-finite values",
                )
                _q_max_arr = Q_RATIO * ctrl.p_installed_mw
                tc.record(
                    "q_adj_clip_bound",
                    bool((np.abs(result.q_adjusted) <= _q_max_arr + 1e-9).all()),
                    f"max excess = "
                    f"{float(np.maximum(0.0, np.abs(result.q_adjusted) - _q_max_arr).max()):.6f} MVAr",
                )
                tc.record(
                    "t_ms_positive",
                    result.t_total_ms > 0.0,
                    f"t_ms={result.t_total_ms:.2f}",
                )
 
                rec.update({
                    "n_ders":              ctrl.n_ders,
                    "n_ov_pre":            result.report_pre.n_over_voltage,
                    "n_ov_post":           (result.report_post.n_over_voltage
                                            if result.report_post else None),
                    "violations_resolved": result.violations_resolved,
                    "max_dq_corr":         float(np.abs(result.dq_correction).max()),
                    "q_adj_max":           float(np.abs(result.q_adjusted).max()),
                    "t_ms":                result.t_total_ms,
                    "post_pf_ok":          result.post_pf_ok,
                })
 
            except Exception:
                tc.error = traceback.format_exc()
                rec.update({k: None for k in [
                    "n_ders", "n_ov_pre", "n_ov_post", "violations_resolved",
                    "max_dq_corr", "q_adj_max", "t_ms", "post_pf_ok",
                ]})
 
            tc.duration = time.time() - t0
            cases.append(tc)
            coord_rep_records.append(rec)
            print_case(tc, verbose)
 
        _print_coordinator_summary(coord_rep_records)
 
    # -----------------------------------------------------------------------
    # 6.3  Hardware — 9 representative networks
    #      Skipped entirely if --arduino-port not supplied.
    # -----------------------------------------------------------------------
    hw_label = arduino_port if arduino_port else "SKIPPED — pass --arduino-port"
    print(f"\n  [3/3] Coordinator hardware  ({hw_label})")
 
    if arduino_port:
        from sensitivity_coordinator import (
            SensitivityCoordinator,
            run_coordinated_timestep,
        )
        from volt_var_controller import (
            VoltVarController, Q_RATIO, ArduinoSerialInterface,
        )
 
        for net_name, loader, label in REPRESENTATIVE_NETWORKS_COORD:
            if only and not any(s in net_name for s in only):
                continue
            tc = TestCase(f"coord_hw_{net_name}")
            t0 = time.time()
            try:
                net = loader()
 
                net.sgen.p_mw   = net.sgen.sn_mva * 0.90
                net.load.p_mw   = net.load.p_mw   * 0.20
                net.load.q_mvar = net.load.q_mvar  * 0.20
 
                # Synthetic LV: clean NaN/Inf, reduce stress, bfsw fallback
                if "synthetic" in label.lower():
                    net.sgen["q_mvar"] = 0.0
                    net.load["q_mvar"] = net.load["q_mvar"].fillna(0.0)
                    net.sgen["p_mw"]   = (net.sgen["p_mw"]
                                          .replace([np.inf, -np.inf], np.nan)
                                          .fillna(0.0))
                    net.load["p_mw"]   = (net.load["p_mw"]
                                          .replace([np.inf, -np.inf], np.nan)
                                          .fillna(0.0))
                    net.sgen["sn_mva"] = (net.sgen["sn_mva"]
                                          .replace([np.inf, -np.inf], np.nan)
                                          .fillna(net.sgen["p_mw"]))
                    net.sgen.p_mw      = net.sgen.sn_mva * 0.20
                    net.load.p_mw      = net.load.p_mw   * 0.40
                    runpp_candidates = [
                        {"algorithm": "bfsw", "max_iteration": 80, "init": "flat"},
                        {"algorithm": "nr",   "max_iteration": 50, "init": "flat"},
                    ]
                elif ("lv" in label.lower() or "dickert" in label.lower()
                      or "kerber" in label.lower()):
                    runpp_candidates = [
                        {"algorithm": "bfsw", "max_iteration": 30, "init": "flat"},
                        {"algorithm": "nr",   "max_iteration": 50, "init": "flat"},
                    ]
                else:
                    runpp_candidates = [{}]
 
                with ArduinoSerialInterface(port=arduino_port) as arduino:
                    with _warnings.catch_warnings():
                        _warnings.simplefilter("ignore")
                        ctrl_hw = VoltVarController(net, interface=arduino,
                                                    dry_run=False)
                        ctrl_hw.configure()
 
                    if ctrl_hw.n_ders == 0:
                        tc.record("hw_no_der", True,
                                  "No controllable DERs in network")
                        tc.skipped = True
                        cases.append(tc)
                        print_case(tc, verbose)
                        continue
 
                    coord = SensitivityCoordinator(net, ctrl_hw)
 
                    result = None
                    for kw in runpp_candidates:
                        full_kw = {"voltage_depend_loads": False}
                        full_kw.update(kw)
                        result = run_coordinated_timestep(
                            net, ctrl_hw, coord, runpp_kwargs=full_kw
                        )
                        if result.post_pf_ok:
                            break
 
                tc.record("hw_post_pf_ok", result.post_pf_ok,
                          "post-PF must converge after hardware Q exchange")
                tc.record("hw_q_adj_finite",
                          bool(np.isfinite(result.q_adjusted).all()),
                          f"max|q_adj|={float(np.abs(result.q_adjusted).max()):.4f} MVAr")
                _qmax = Q_RATIO * ctrl_hw.p_installed_mw
                tc.record("hw_q_adj_clipped",
                          bool((np.abs(result.q_adjusted) <= _qmax + 1e-9).all()),
                          "q_adjusted must respect ±q_max bounds")
                tc.record("hw_t_ms_positive", result.t_total_ms > 0.0,
                          f"t_ms={result.t_total_ms:.2f}")
                print(f"         {label}: t={result.t_total_ms:.0f}ms  "
                      f"n_ders={ctrl_hw.n_ders}  retries={result.n_retries}  "
                      f"resolved={result.violations_resolved}")
 
            except Exception:
                tc.error = traceback.format_exc()
            tc.duration = time.time() - t0
            cases.append(tc)
            print_case(tc, verbose)
 
    return cases

# ===========================================================================
# run_sensitivity_coordinator_all_tests
# ===========================================================================

def run_sensitivity_coordinator_all_tests(
        verbose:      bool = False,
        only:         list = None,
        arduino_port: str  = None,
        only_hw:      bool = False,
) -> list:
    """
    Section 6 (all-199) — Sensitivity Coordinator sweep across all in-scope
    networks.  One TestCase per network, one assertion (post_pf_ok).
    Prints ranked coordinator summary at the end.
 
    Run:
        python test_suite.py --section sensitivity_coordinator_all
    """
    import warnings as _warnings
    from sensitivity_coordinator import (
        SensitivityCoordinator,
        run_coordinated_timestep,
    )
    from volt_var_controller import VoltVarController, Q_RATIO
 
    cases             = []
    coord_all_records = []
 
    print(f"\n  [1/1] Coordinator dry_run — all in-scope networks")
 
    def _run_coord_one(tc: TestCase, name: str, loader_fn) -> dict:
        """
        Load network, apply stress, run one coordinated timestep.
        Records checks on tc. Returns a stats dict for the summary.
        """
        try:
            net = loader_fn()
 
            net.sgen.p_mw   = net.sgen.sn_mva * 0.90
            net.load.p_mw   = net.load.p_mw   * 0.20
            net.load.q_mvar = net.load.q_mvar  * 0.20
            net.sgen["q_mvar"] = 0.0
 
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", UserWarning)
                ctrl = VoltVarController(net, interface=None, dry_run=True)
 
            if ctrl.n_ders == 0:
                tc.record("skipped_no_der", True, "No controllable DERs")
                tc.skipped = True
                return {"name": name, **{k: None for k in [
                    "n_ders", "n_ov_pre", "n_ov_post", "violations_resolved",
                    "max_dq_corr", "q_adj_max", "t_ms", "post_pf_ok",
                ]}}
 
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", UserWarning)
                coord = SensitivityCoordinator(net, ctrl)
 
            runpp_kwargs = {}
            if ("synthetic" in name or "lv" in name.lower()
                    or "dickert" in name.lower()
                    or "kerber" in name.lower()):
                runpp_kwargs = {
                    "algorithm": "bfsw",
                    "max_iteration": 30,
                    "init": "flat",
                }
 
            kw = {"voltage_depend_loads": False}
            kw.update(runpp_kwargs)
 
            result = run_coordinated_timestep(net, ctrl, coord, runpp_kwargs=kw)
 
            tc.record("post_pf_ok", result.post_pf_ok,
                      f"n_ov_pre={result.report_pre.n_over_voltage} → "
                      f"n_ov_post="
                      f"{result.report_post.n_over_voltage if result.report_post else '?'}")
 
            return {
                "name":               name,
                "n_ders":             ctrl.n_ders,
                "n_ov_pre":           result.report_pre.n_over_voltage,
                "n_ov_post":          (result.report_post.n_over_voltage
                                       if result.report_post else None),
                "violations_resolved": result.violations_resolved,
                "max_dq_corr":        float(np.abs(result.dq_correction).max()),
                "q_adj_max":          float(np.abs(result.q_adjusted).max()),
                "t_ms":               result.t_total_ms,
                "post_pf_ok":         result.post_pf_ok,
            }
 
        except Exception:
            tc.error = traceback.format_exc()
            return {"name": name, **{k: None for k in [
                "n_ders", "n_ov_pre", "n_ov_post", "violations_resolved",
                "max_dq_corr", "q_adj_max", "t_ms", "post_pf_ok",
            ]}}
 
    # SimBench — 156 codes
    for code in IN_SCOPE_SIMBENCH_CODES:
        if only and not any(s in code for s in only):
            continue
        tc = TestCase(f"coord_{code}")
        t0 = time.time()
        rec = _run_coord_one(tc, code, lambda c=code: sb.get_simbench_net(c))
        tc.duration = time.time() - t0
        cases.append(tc)
        coord_all_records.append(rec)
        print_case(tc, verbose)
 
    # CIGRE MV + LV — 2 networks
    for name, loader in [
        ("cigre_mv_with_der", lambda: pn.create_cigre_network_mv(with_der="pv_wind")),
        ("cigre_lv",          lambda: pn.create_cigre_network_lv()),
    ]:
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(f"coord_{name}")
        t0 = time.time()
        rec = _run_coord_one(tc, name, loader)
        tc.duration = time.time() - t0
        cases.append(tc)
        coord_all_records.append(rec)
        print_case(tc, verbose)
 
    # Kerber — 17 variants
    for name, fn_name in ALL_KERBER_CASES:
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(f"coord_{name}")
        t0 = time.time()
        rec = _run_coord_one(tc, name,
                             lambda f=fn_name: getattr(pn, f)())
        tc.duration = time.time() - t0
        cases.append(tc)
        coord_all_records.append(rec)
        print_case(tc, verbose)
 
    # Synthetic LV — 5 classes
    for network_class in ALL_SYNTHETIC_LV_CASES:
        name = f"synthetic_lv_{network_class}"
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(f"coord_{name}")
        t0 = time.time()
        rec = _run_coord_one(
            tc, name,
            lambda c=network_class: pn.create_synthetic_voltage_control_lv_network(c),
        )
        tc.duration = time.time() - t0
        cases.append(tc)
        coord_all_records.append(rec)
        print_case(tc, verbose)
 
    # Dickert — 18 combinations
    for name, feeders_range, linetype, customer, case in ALL_DICKERT_CASES:
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(f"coord_{name}")
        t0 = time.time()
        try:
            rec = _run_coord_one(
                tc, name,
                lambda fr=feeders_range, lt=linetype, cu=customer, ca=case:
                    pn.create_dickert_lv_network(fr, lt, cu, ca),
            )
        except ValueError as e:
            if "no dickert network" in str(e).lower():
                tc.skipped = True
            rec = {"name": name, **{k: None for k in [
                "n_ders", "n_ov_pre", "n_ov_post", "violations_resolved",
                "max_dq_corr", "q_adj_max", "t_ms", "post_pf_ok",
            ]}}
        tc.duration = time.time() - t0
        cases.append(tc)
        coord_all_records.append(rec)
        print_case(tc, verbose)
 
    _print_coordinator_summary(coord_all_records)
    return cases


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
    "network_plotter":  run_network_plotter_tests,
    "violation_detector": run_violation_detector_tests,
    "violation_detector_all": run_violation_detector_all_tests,
    "volt_var_control": run_volt_var_tests,
    "sensitivity_coordinator": run_sensitivity_coordinator_tests,
    "sensitivity_coordinator_all": run_sensitivity_coordinator_all_tests,
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
        "--arduino-port",
        default=None,
        help="Serial port for hardware Volt-Var tests e.g. /dev/ttyACM0"
             "If omitted, Section 5.4 (hardware) is skipped automatically "
    )
    parser.add_argument(
        "--only-hw",
        action="store_true",
        help="Run hardware (Arduino) Volt-Var tests only (skip dry-run sections)."
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

    # after args are parsed and before to_run is built
    if args.only_hw:
        if not args.arduino_port:
            raise SystemExit("--only-hw requires --arduino-port")

    # wherever you build to_run
    if args.only_hw:
        # keep only the volt-var section (hardware block inside it)
        to_run = {"volt_var_control": run_volt_var_tests}
    
    else:
        to_run = (
            {args.section: SECTIONS[args.section]}
            if args.section
            else SECTIONS
        )

    for section_name, run_fn in to_run.items():
        print(f"\n{'='*70}")
        print(f"  SECTION: {section_name.upper()}")
        print(f"{'='*70}")
        kwargs = {"verbose": args.verbose, "only": args.only}
        if section_name == "volt_var_control":
            kwargs.update({"arduino_port": args.arduino_port, "only_hw": args.only_hw})
        if section_name in ("sensitivity_coordinator",
                            "sensitivity_coordinator_all"):
            kwargs.update({"arduino_port": args.arduino_port,
                           "only_hw":      args.only_hw})
        
        cases = run_fn(**kwargs)
        section_results[section_name] = cases

    print_summary(section_results)
    print(f"\n  Total time: {time.time() - t_start:.1f}s")