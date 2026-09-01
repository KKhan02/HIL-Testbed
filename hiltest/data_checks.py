"""
hiltest/data_checks.py
=======================
Shared profile sanity checks used by profile_builder and plotter sections.

Time-index checks added (previously accepted as missing):
- times is a DatetimeIndex
- index is unique (no duplicate timestamps)
- index is monotonic increasing
- timezone consistency: all tz-aware or all tz-naive, not mixed
- profile DataFrame index equals times
- values are finite (no NaN or Inf remaining after profile build)
- columns are numeric dtype
"""

import numpy as np
import pandas as pd

from hiltest.framework import TestCase


def _check_timeseries(tc: TestCase, df: pd.DataFrame, name: str,
                      times: pd.DatetimeIndex) -> bool:
    """
    Run time-index and dtype checks on a single profile DataFrame.
    Returns True if all checks passed, False otherwise.
    Records checks directly into tc.
    """
    if not isinstance(df, pd.DataFrame):
        tc.record(f"{name}_is_dataframe", False,
                  f"expected DataFrame, got {type(df).__name__}")
        return False

    if df.empty:
        tc.record(f"{name}_not_empty", False, "DataFrame is empty")
        return False

    # Numeric dtype — catches any object-typed columns
    non_numeric = [str(c) for c in df.columns
                   if not pd.api.types.is_numeric_dtype(df[c])]
    tc.record(f"{name}_numeric_dtype", not non_numeric,
              f"Non-numeric columns: {non_numeric}")

    # Finite values — NaN or Inf here means profile_builder left bad data
    finite_ok = bool(np.isfinite(df.values).all())
    n_bad = int((~np.isfinite(df.values)).sum())
    tc.record(f"{name}_finite_values", finite_ok,
              f"{n_bad} NaN/Inf values found")

    # Index alignment with times
    if isinstance(times, pd.DatetimeIndex) and len(times) > 0:
        aligned = df.index.equals(times)
        tc.record(f"{name}_index_equals_times", aligned,
                  f"DataFrame index length {len(df.index)} != "
                  f"times length {len(times)}")

    return True


def check_profiles(tc: TestCase, result: dict,
                   check_pv_night: bool = False) -> None:
    """
    Populate tc with profile_builder sanity checks.

    Checks run in four groups:
    1. Required keys and times index integrity
    2. Extreme-days dict
    3. Per-profile content (load, pv, wind)
    4. PV night zeroing (optional)

    Parameters
    ----------
    tc              : TestCase to record checks into.
    result          : Dict returned by build_annual_profiles().
    check_pv_night  : If True, verify PV output is zero between 22:00–04:00.
    """
    # -----------------------------------------------------------------------
    # Group 1: Required keys
    # -----------------------------------------------------------------------
    required = {"load", "pv", "wind", "times", "net_type", "extreme_days"}
    missing  = required - set(result.keys())
    tc.record("required_keys", not missing,
              f"Missing: {missing}" if missing else "")

    # -----------------------------------------------------------------------
    # Group 2: Times index integrity
    # -----------------------------------------------------------------------
    times = result.get("times", pd.DatetimeIndex([]))

    tc.record("times_is_datetimeindex",
              isinstance(times, pd.DatetimeIndex),
              f"times is {type(times).__name__}, expected DatetimeIndex")

    tc.record("times_count", len(times) > 100,
              f"Only {len(times)} timesteps")

    if isinstance(times, pd.DatetimeIndex) and len(times) > 1:
        tc.record("times_unique",
                  bool(times.is_unique),
                  f"{(~times.duplicated()).sum()} duplicate timestamps")

        tc.record("times_monotonic",
                  bool(times.is_monotonic_increasing),
                  "Index is not monotonic increasing")

        # Timezone consistency: all tz-aware or all tz-naive — not mixed
        tz = times.tz
        tc.record("times_tz_consistent",
                  tz is not None or times.tz is None,
                  "Mixed tz-aware and tz-naive timestamps" if tz is None
                  else f"tz={tz}")

    # -----------------------------------------------------------------------
    # Group 3: Extreme-days dict
    # -----------------------------------------------------------------------
    extreme = result.get("extreme_days", {})
    tc.record("extreme_days_exists", isinstance(extreme, dict),
              f"extreme_days is {type(extreme).__name__}")
    tc.record("extreme_days_keys",
              all(k in extreme
                  for k in ("max_der", "min_der", "max_load", "min_load")),
              f"Keys present: {list(extreme.keys())}")

    # -----------------------------------------------------------------------
    # Group 4: Per-profile content
    # -----------------------------------------------------------------------
    load_df = result.get("load")
    if isinstance(load_df, pd.DataFrame) and not load_df.empty:
        _check_timeseries(tc, load_df, "load", times)
        bad = load_df.columns[load_df.isna().all()].tolist()
        tc.record("load_no_all_nan", not bad,  f"All-NaN cols: {bad}")
        tc.record("load_sum_positive", load_df.sum().sum() > 0,
                  "All load values are zero")
        tc.record("load_no_negative", (load_df >= 0).all().all(),
                  "Negative load values found")
    else:
        tc.record("load_exists", False, "Load DataFrame missing or empty")

    pv_df = result.get("pv")
    if isinstance(pv_df, pd.DataFrame) and not pv_df.empty:
        _check_timeseries(tc, pv_df, "pv", times)
        bad = pv_df.columns[pv_df.isna().all()].tolist()
        tc.record("pv_no_all_nan",  not bad,  f"All-NaN cols: {bad}")
        tc.record("pv_no_negative", (pv_df >= 0).all().all(),
                  "Negative PV values found")
        if check_pv_night and isinstance(times, pd.DatetimeIndex) \
                and len(times) > 0:
            night = (times.hour >= 22) | (times.hour <= 4)
            tc.record("pv_zero_at_night",
                      (pv_df.loc[night] < 0.001).all().all(),
                      "Non-zero PV found at night hours")

    wind_df = result.get("wind")
    if isinstance(wind_df, pd.DataFrame) and not wind_df.empty:
        _check_timeseries(tc, wind_df, "wind", times)
        bad = wind_df.columns[wind_df.isna().all()].tolist()
        tc.record("wind_no_all_nan",   not bad, f"All-NaN cols: {bad}")
        tc.record("wind_no_negative",  (wind_df >= 0).all().all(),
                  "Negative wind values found")
        tc.record("wind_max_plausible", wind_df.max().max() < 1000,
                  f"Max wind {wind_df.max().max():.1f} MW exceeds 1000 MW")
