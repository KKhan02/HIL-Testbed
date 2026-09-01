"""
hiltest/sections/plotter.py
============================
Section: network_plotter

Changes from previous version
------------------------------
- Figures closed after each check when SHOW_PLOTS=False to prevent memory
  accumulation during batch runs (54 open figure objects over 9 networks).
- Missing-dependency path now returns one skipped TestCase instead of [].
  [] made the section show 0/0 in the summary, which is ambiguous.
- Removed unused matplotlib.pyplot import (plt was never called after
  SHOW_PLOTS was delegated to plot_* functions).
"""
import traceback
import time

from hiltest.framework import TestCase, print_case
from hiltest.constants import DWD_DATA_DIR, SHOW_PLOTS
from hiltest.networks  import get_representative_networks_plotter


def run_network_plotter_tests(verbose: bool = False, only: list = None) -> list:
    try:
        import matplotlib
        import matplotlib.figure
        import matplotlib.pyplot as plt   # needed for plt.close()
        from network_plotter import plot_topology, plot_profiles, plot_day
        from profile_builder import build_annual_profiles
    except ImportError as exc:
        # Return one skipped TestCase so the section shows SKIP in the summary
        # rather than the ambiguous 0/0 from returning [].
        # Record the check (dependency_available=False) so the detail is
        # visible in verbose output, consistent with other early-exit paths.
        tc = TestCase("plotter_dependency_check")
        tc.record("dependency_available", False, str(exc))
        tc.skipped = True
        print(f"  SKIP  plotter_dependency_check  (missing dep: {exc})")
        return [tc]

    REPRESENTATIVE_NETWORKS_PLOTTER = get_representative_networks_plotter()

    cases = []
    extreme_day_keys = [
        ("max_der",  "Max DER generation day"),
        ("min_der",  "Min DER generation day"),
        ("max_load", "Peak load day"),
        ("min_load", "Min load day"),
    ]

    for test_name, loader, net_name, sb_code in REPRESENTATIVE_NETWORKS_PLOTTER:
        if only and not any(s in test_name for s in only):
            continue
        tc = TestCase(test_name)
        t0 = time.perf_counter()
        try:
            net    = loader()
            kwargs = dict(data_dir=DWD_DATA_DIR)
            if sb_code:
                kwargs["simbench_code"] = sb_code
            prof = build_annual_profiles(net, net_name, **kwargs)

            fig_topo = plot_topology(net, net_name, show=SHOW_PLOTS)
            tc.record("topology_returns_figure",
                      isinstance(fig_topo, matplotlib.figure.Figure),
                      "plot_topology did not return a Figure")
            if not SHOW_PLOTS:
                plt.close(fig_topo)

            fig_prof = plot_profiles(net_name, prof, show=SHOW_PLOTS)
            tc.record("profiles_returns_figure",
                      isinstance(fig_prof, matplotlib.figure.Figure),
                      "plot_profiles did not return a Figure")
            if not SHOW_PLOTS:
                plt.close(fig_prof)

            ed = prof.get("extreme_days", {})
            for day_key, day_label in extreme_day_keys:
                day_str = ed.get(day_key)
                if day_str is None:
                    tc.record(f"plot_day_{day_key}_skipped_no_der", True,
                              f"No {day_label} — network has no relevant DER")
                    continue
                fig_day = plot_day(prof, day_str, net_name,
                                   day_label=day_label, show=SHOW_PLOTS)
                tc.record(f"plot_day_{day_key}",
                          isinstance(fig_day, matplotlib.figure.Figure),
                          f"plot_day({day_key}) did not return a Figure")
                if not SHOW_PLOTS:
                    plt.close(fig_day)

        except Exception:
            tc.error = traceback.format_exc()
        tc.duration = time.perf_counter() - t0
        cases.append(tc)
        print_case(tc, verbose)

    return cases
