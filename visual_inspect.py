"""
visual_inspect.py
=================
Iterates through all in-scope networks, showing topology + profiles
for each. Press any key to advance to the next network.

Usage:
    python visual_inspect.py                    # all 199 networks
    python visual_inspect.py --only cigre       # filter by name substring
    python visual_inspect.py --section simbench # simbench only
    python visual_inspect.py --save outputs/    # save all figures, no display
"""

import argparse
import pandapower.networks as pn
import simbench as sb
from profile_builder import build_annual_profiles
from network_plotter import plot_topology, plot_profiles, plot_day

from test_suite import (
    IN_SCOPE_SIMBENCH_CODES,
    ALL_KERBER_CASES,
    ALL_SYNTHETIC_LV_CASES,
    ALL_DICKERT_CASES,
    DWD_DATA_DIR,
)

DWD_DATA_DIR = "data/dwd"

def inspect_network(net, net_name, sb_code=None, show=True, save_dir=None):
    import os
    save_topo = save_prof = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe = net_name.replace("/", "-")
        save_topo = f"{save_dir}/{safe}_topology.png"
        save_prof = f"{save_dir}/{safe}_profiles.png"

    kwargs = dict(data_dir=DWD_DATA_DIR)
    if sb_code:
        kwargs["simbench_code"] = sb_code

    prof = build_annual_profiles(net, net_name, **kwargs)

    plot_topology(net, net_name, save_path=save_topo, show=show)
    plot_profiles(net_name, prof, save_path=save_prof, show=show)

    ed = prof.get("extreme_days", {})
    for key, label in [("max_der",  "Max DER generation day"),
                       ("min_der",  "Min DER generation day"),
                       ("max_load", "Peak load day"),
                       ("min_load", "Min load day")]:
        day_str = ed.get(key)
        if day_str is None:
            continue
        save_day = f"{save_dir}/{safe}_{key}.png" if save_dir else None
        plot_day(prof, day_str, net_name,
                 day_label=label, save_path=save_day, show=show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only",    nargs="+", default=None)
    parser.add_argument("--section", choices=["simbench","cigre","kerber",
                                               "synthetic","dickert","all"],
                        default="all")
    parser.add_argument("--save",    default=None,
                        help="Save figures to this directory instead of showing")
    args = parser.parse_args()

    show = args.save is None   # show interactively unless saving

    def should_run(name):
        return not args.only or any(s in name for s in args.only)

    # SimBench
    if args.section in ("simbench", "all"):
        for code in IN_SCOPE_SIMBENCH_CODES:
            if not should_run(code): continue
            print(f"[inspect] {code}")
            net = sb.get_simbench_net(code)
            inspect_network(net, code, sb_code=code,
                            show=show, save_dir=args.save)

    # CIGRE
    if args.section in ("cigre", "all"):
        for name, loader in [
            ("cigre_mv_with_der", lambda: pn.create_cigre_network_mv(with_der="pv_wind")),
            ("cigre_lv",          lambda: pn.create_cigre_network_lv()),
        ]:
            if not should_run(name): continue
            print(f"[inspect] {name}")
            inspect_network(loader(), name, show=show, save_dir=args.save)

    # Kerber
    if args.section in ("kerber", "all"):
        for name, fn_name in ALL_KERBER_CASES:
            if not should_run(name): continue
            print(f"[inspect] {name}")
            fn  = getattr(pn, fn_name)
            inspect_network(fn(), name, show=show, save_dir=args.save)

    # Synthetic LV
    if args.section in ("synthetic", "all"):
        for cls in ALL_SYNTHETIC_LV_CASES:
            name = f"synthetic_lv_{cls}"
            if not should_run(name): continue
            print(f"[inspect] {name}")
            net = pn.create_synthetic_voltage_control_lv_network(cls)
            inspect_network(net, name, show=show, save_dir=args.save)

    # Dickert
    if args.section in ("dickert", "all"):
        for name, feeders, linetype, customer, case in ALL_DICKERT_CASES:
            if not should_run(name): continue
            print(f"[inspect] {name}")
            net = pn.create_dickert_lv_network(feeders, linetype, customer, case)
            inspect_network(net, name, show=show, save_dir=args.save)

    print("[inspect] Done.")
