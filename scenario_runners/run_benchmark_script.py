"""
run_benchmark_script.py
-----------------------
Minimal script to run the full benchmark (five scenarios + hosting capacity
analysis) on one network.

Place this in the same directory as scenario_result.py, benchmark_runner.py,
profile_builder.py, etc. and run:

    python run_benchmark_script.py

Or on the RPi:

    source hil_env/bin/activate
    python run_benchmark_script.py
"""
import logging
import simbench as sb
import pandapower.networks as pn
from datetime import datetime
from pathlib import Path
import sys
import copy

# ── Path setup ─────────────────────────────────────────────────────────────
# This file lives in scenario_runners/. We need:
#   project_root/  → der_dynamics, violation_detector, profile_builder, etc.
#   Command_Line_Interface/ → _console (Rich theme); falls back to plain if absent
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "Command_Line_Interface"))
# ───────────────────────────────────────────────────────────────────────────

from profile_builder import build_annual_profiles
from benchmark_runner import BenchmarkConfig, run_benchmark
from scenario_result import slice_profiles, oversize_inverters
from rich.logging import RichHandler
from network_plotter import plot_topology, plot_profiles
import matplotlib.pyplot as plt
from publisher import publish_topology_and_profiles, publish_hc_and_comparison, PublishHandle
import volt_var_controller as vvc          # for set_qv_parameters() — Q_RATIO/U1-U4 override
import violation_detector as vd            # for module-constant override — voltage/thermal/angle limits

# ── Configure logging so progress messages appear in the terminal ──────────
_log_dir = _root / "outputs" / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
_log_path = _log_dir / f"session_{datetime.now():%Y%m%d_%H%M%S}.log"

_file_handler = logging.FileHandler(_log_path, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))

logging.basicConfig(
    level  = logging.INFO,
    format = "%(message)s",
    datefmt= "[%H:%M:%S]",
    handlers = [RichHandler(rich_tracebacks=True, markup=False, show_path=True), _file_handler],
)

class _PeriodicFilter(logging.Filter):
    """
    Pass every Nth INFO message from a noisy logger, but always pass
    WARNING and above so genuine issues are never suppressed.

    every_n=96 means one message per simulated day at 15-min resolution.
    """
    def __init__(self, every_n: int = 96):
        super().__init__()
        self.every_n = every_n
        self._count  = 0

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True          # warnings and errors always pass through
        self._count += 1
        return self._count % self.every_n == 0

_sc_logger = logging.getLogger("sensitivity_coordinator")
_sc_logger.addFilter(_PeriodicFilter(every_n=96))

# ── Command-line arguments ───────────────────────────────────────────────────
# --network <yaml>  : load a network plugin YAML (network_plugin.py) instead
#                     of the hardcoded selector block below.  The plugin
#                     builds the profiles itself, so build_annual_profiles()
#                     is skipped on this path.
# -y / --yes        : auto-confirm validation warnings (headless RPi runs).
import argparse

_parser = argparse.ArgumentParser(
    description="Run the full HIL benchmark on one network.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=(
        "Examples:\n"
        "  # Defaults (canonical VDE-AR-N 4110 Q(V), 0.95-1.05 pu band):\n"
        "  python run_benchmark_script.py\n\n"
        "  # A single override from the RPi console (no file edit):\n"
        "  python run_benchmark_script.py --q-ratio 0.30\n\n"
        "  python run_benchmark_script.py \\\n"
        "      --u1 0.95 --u2 0.99 --u3 1.01 --u4 1.04 \\\n"
        "      --v-min 0.95 --v-max 1.05 \\\n"
        "      --line-max-loading 100 --trafo-max-loading 100\n\n"
        "  # A custom network + controller plugin together:\n"
        "  python run_benchmark_script.py \\\n"
        "      --network example_networks/custom_function.yaml \\\n"
        "      --controller example_plugins/droop_controller.yaml\n\n"
        "Any flag left unset uses the canonical default shown in its help "
        "text below. Q(V) breakpoints must satisfy U1 < U2 < U3 < U4 and "
        "0 < q_ratio <= 1, or set_qv_parameters() raises (mirrors the "
        "firmware's ERR:CFG_INVALID guard)."
    ),
)
_parser.add_argument(
    "--network", metavar="YAML", default=None,
    help="Path to a network plugin YAML (see network_plugin.py and "
         "example_networks/). Replaces the hardcoded network selector.",
)
_parser.add_argument(
    "--controller", metavar="YAML", default=None,
    help="Path to a controller plugin YAML (see plugin_runner.py and "
         "example_plugins/). The custom controller scenario runs ALONGSIDE "
         "the scenarios in BenchmarkConfig. Combinable with --network.",
)
_parser.add_argument(
    "-y", "--yes", action="store_true",
    help="Proceed without prompting when validation warnings are present.",
)

_parser.add_argument(
    "--q-ratio", type=float, default=None, metavar="Q",
    help="Override Q_RATIO (Q_max / P_installed, 0 < q <= 1). "
         "Leave unset to use the hardcoded value below.",
)
_parser.add_argument(
    "--v-min", type=float, default=None, metavar="PU",
    help="Override voltage_detector.V_MIN (lower voltage planning limit, pu).",
)
_parser.add_argument(
    "--v-max", type=float, default=None, metavar="PU",
    help="Override violation_detector.V_MAX (upper voltage planning limit, pu).",
)
_parser.add_argument(
    "--line-max-loading", type=float, default=None, metavar="PCT",
    help="Override violation_detector.LINE_MAX_LOADING (line thermal limit, %%).",
)
_parser.add_argument(
    "--trafo-max-loading", type=float, default=None, metavar="PCT",
    help="Override violation_detector.TRAFO_MAX_LOADING (trafo thermal limit, %%).",
)
_parser.add_argument(
    "--u1", type=float, default=None, metavar="PU",
    help="Override Q(V) lower saturation breakpoint U1 (full +Q at/below, "
         "pu). Default 0.96. Must satisfy U1 < U2 < U3 < U4.",
)
_parser.add_argument(
    "--u2", type=float, default=None, metavar="PU",
    help="Override Q(V) deadband lower edge U2 (pu). Default 0.99.",
)
_parser.add_argument(
    "--u3", type=float, default=None, metavar="PU",
    help="Override Q(V) deadband upper edge U3 (pu). Default 1.01.",
)
_parser.add_argument(
    "--u4", type=float, default=None, metavar="PU",
    help="Override Q(V) upper saturation breakpoint U4 (full -Q at/above, "
         "pu). Default 1.04.",
)
_args = _parser.parse_args()

# ===========================================================================
# ── Parametric overrides: Q(V) curve shape + violation limits
# ===========================================================================
# Canonical VDE-AR-N 4110 Bild 8 defaults. Each --flag overrides one value
# without editing this file (e.g. `python run_benchmark_script.py --q-ratio 0.30`).
# NOTE: set_qv_parameters() must be called BEFORE run_benchmark() / any
# scenario runner — the controller and coordinator read these values at
# construction time, and the Arduino CFG message is built from them too.

_Q_RATIO = _args.q_ratio if _args.q_ratio is not None else 0.25    # 0 < q <= 1
_U1_PU   = _args.u1 if _args.u1 is not None else 0.96   # lower saturation [pu]
_U2_PU   = _args.u2 if _args.u2 is not None else 0.99   # deadband lower edge [pu]
_U3_PU   = _args.u3 if _args.u3 is not None else 1.01   # deadband upper edge [pu]
_U4_PU   = _args.u4 if _args.u4 is not None else 1.04   # upper saturation [pu]

vvc.set_qv_parameters(
    q_ratio = _Q_RATIO,
    u1      = _U1_PU,
    u2      = _U2_PU,
    u3      = _U3_PU,
    u4      = _U4_PU,
)   # raises ValueError if invalid (mirrors firmware ERR:CFG_INVALID)

# Violation-detector module constants — plain assignment, same mechanism
# executor.py's apply_violation_limits() uses internally. Every call to
# detect_violations()/detect_violations_3ph() with no explicit kwargs picks
# these up at CALL time (late-bound defaults), so this reaches Scenario 4's
# bare detect_violations(net) with no other code changes needed.
vd.V_MIN              = _args.v_min if _args.v_min is not None else 0.95   # pu, lower voltage limit
vd.V_MAX              = _args.v_max if _args.v_max is not None else 1.05   # pu, upper voltage limit
vd.set_limit("LINE_MAX_LOADING",  _args.line_max_loading,  0.0, vd.THERMAL_LOADING_PLAUSIBLE_MAX, "%", confirm = print)
vd.set_limit("TRAFO_MAX_LOADING", _args.trafo_max_loading, 0.0, vd.THERMAL_LOADING_PLAUSIBLE_MAX, "%", confirm = print)
vd.VA_DIFF_MAX_DEGREE = 22.0    # deg — max angle difference across a line, DEFAULT 30.0
vd.UNBALANCE_MAX_PERCENT = 2.0  # % — IEC 62749 voltage unbalance limit (currently reaches no
                                 # result: no runner in the benchmark stack calls detect_violations_3ph())

RUN_NAME = "_HC_Run"
OVERSIZE_FACTOR = None   # set to None to disable oversizing for this run
LOAD_SCALE = 1.0    # 0.5 = halve all loads

# ===========================================================================

_plugin_meta: dict = {}

if _args.network:
    # ── Plugin network path ──────────────────────────────────────────────
    from network_plugin import (
        load_network_from_yaml,
        make_profile_factory,
        validate_network_plugin,
    )

    net, profiles = load_network_from_yaml(_args.network)
    _plugin_meta  = profiles.get("plugin_meta", {})
    net_name      = _plugin_meta.get("name", Path(_args.network).stem)
    sb_code       = None    # plugin path never re-downloads via simbench_code
    _base_pf = (net.sgen["p_mw"] / net.sgen["sn_mva"]).round(4)
    print(f"[BASE CHECK] pre-scaling p_mw sum={net.sgen['p_mw'].sum():.4f}, "
      f"sn_mva sum={net.sgen['sn_mva'].sum():.4f}, PF values={_base_pf.unique().tolist()}")

    _warnings = validate_network_plugin(net, profiles)
    if _warnings:
        print("\n" + "=" * 78)
        print(f"  NETWORK PLUGIN VALIDATION — {net_name}: "
              f"{len(_warnings)} warning(s)")
        print("=" * 78)
        for i, w in enumerate(_warnings, 1):
            print(f"  [{i}] {w}")
        print("=" * 78)
        if not _args.yes:
            _answer = input("Proceed with the benchmark anyway? [y/N] ").strip().lower()
            if _answer not in ("y", "yes"):
                print("Aborted by user after validation warnings.")
                sys.exit(1)
    else:
        print(f"[network_plugin] Validation passed — no warnings "
              f"({net_name}).")
else:
    # ── Network selector — change only this block ───────────────────────────
    #
    # SimBench MV rural (primary HIL network 1-MV-rural--2-sw)
    net        = sb.get_simbench_net("1-MV-rural--2-sw")
    net_name   = "1-MV-rural--2-sw"         # triggers SimBench profile path
    sb_code    = "1-MV-rural--2-sw"         # required for SimBench only

    # ── Synthetic LV suburb (best for coordinator validation — dense DERs) ──────
    # net        = pn.create_synthetic_voltage_control_lv_network("suburb_1")
    # net_name   = "synthetic_lv_suburb_1"
    # sb_code    = None

    # ── Synthetic LV village ────────────────────────────────────────────────────
    # net        = pn.create_synthetic_voltage_control_lv_network("rural_2")
    # net_name = "synthetic_lv_rural_2"
    # sb_code    = None

    # CIGRE MV with DER  — uncomment to use
    # net        = pn.create_cigre_network_mv(with_der="pv_wind")
    # net_name   = "cigre_mv_pv_wind"       # "cigre" in name → DWD path
    # sb_code    = None

    # CIGRE LV  — uncomment to use
    # net        = pn.create_cigre_network_lv()
    # net_name   = "cigre_lv"       # "cigre" in name → DWD path
    # sb_code    = None

    # Kerber Landnetz Kabel 1 — uncomment to use 
    # For lV networks with no DERs, skip scenario 4 in the benchmark config
    # net        = pn.create_kerber_landnetz_kabel_1()
    # net_name   = "kerber_landnetz_kabel_1"  # no "simbench"/"cigre" → DWD fallback path
    # sb_code    = None

    # Dickert short cable, single feeder, bad case — uncomment to use
    # net        = pn.create_dickert_lv_network("short", "cable", "single", "bad")
    # net_name   = "dickert_short_cable_single_bad"
    # sb_code    = None

    # Custom — uncomment to use
    # net        = sb.get_simbench_net("1-LV-urban6--1-sw")
    # net_name   = "1-LV-urban6--1-sw"         # triggers SimBench profile path
    # sb_code    = "1-LV-urban6--1-sw"         # required for SimBench only
# ─────────────────────────────────────────────────────────────────────────────

# ── Profiles ─────────────────────────────────────────────────────────────────
# On the --network plugin path the profiles were already built by
# load_network_from_yaml() according to the YAML's profiles.strategy.
if not _args.network:
    profiles = build_annual_profiles(
        net,
        net_name      = net_name,
        data_dir      = str(_root / "data" / "dwd"),   # ignored for SimBench
        simbench_code = sb_code,                        # None is fine for non-SimBench
    )

# ===========================================================================
# ── Optional: oversize inverters (increases Q_max for Scenarios 4A/4B)
#    None = no change | 1.1 = 10% oversize | 1.3 = 30% oversize
#    Only scales PV and wind sgens — load-sgens are not affected.
#    Set OVERSIZE_FACTOR = None to skip oversizing entirely — net_os then
#    just points at net, so every call site below that already reads
#    net_os keeps working unchanged.
# ── Optional: DER scaling (p_mw AND sn_mva together, PF preserved)
#    Mirrors executor.py's _apply_scaling()/der_scaling exactly — scales
#    BOTH fields by the same factor so PF stays at the network's native
#    value instead of drifting. Set OVERSIZE_FACTOR = None to disable.
# ===========================================================================
if OVERSIZE_FACTOR is not None:
    net_os = copy.deepcopy(net)
#    oversize_inverters(net_os, factor=OVERSIZE_FACTOR)
    net_os.sgen["p_mw"]   = net_os.sgen["p_mw"]   * OVERSIZE_FACTOR
    net_os.sgen["sn_mva"] = net_os.sgen["sn_mva"] * OVERSIZE_FACTOR
    for key in ("pv", "wind"):
        if key in profiles and not profiles[key].empty:
            common = [c for c in profiles[key].columns if c in net_os.sgen.index]
            profiles[key][common] = profiles[key][common] * OVERSIZE_FACTOR
    print(f"[DER scaling] p_mw and sn_mva scaled by {OVERSIZE_FACTOR} "
          f"({len(net_os.sgen)} sgens, PF preserved)")
else:
    net_os = net

# ===========================================================================
# ── Optional: time-slice (faster iteration — comment out for full annual run)
# ===========================================================================
# profiles_run = slice_profiles(profiles, period="month", index=6)
# profiles_run = slice_profiles(profiles, period="week", index=28)
# profiles_run = slice_profiles(profiles, period="day", index=172)

# ===========================================================================
# ── Network inspection — topology + profiles
#    Runs before any benchmark modification so you see the unmodified network.
#    Set show=False to suppress interactive windows (e.g. on RPi headless).
# ===========================================================================
# plot_topology(net, net_name, show=True, show_bus_ids=True) 
# plot_profiles(net_name, profiles, show=True)

# ===========================================================================
# ── Optional: scale down loads (e.g. to reduce baseline congestion)
#    Modifies net.load.p_mw and net.load.q_mvar in place.
#    Profiles are unaffected — adapt_profiles() re-reads net.load indices only.
# ===========================================================================
net_os.load["p_mw"]  *= LOAD_SCALE
net_os.load["q_mvar"] *= LOAD_SCALE
profiles["load"] *= LOAD_SCALE   # add this line alongside the net.load scaling to scale the load profile too

# ===========================================================================
# ── Optional: add DER sgens at specific buses to study network response
#    Call plot_topology(net, net_name, show_bus_ids=True) first to
#    identify bus indices, then place DERs below.
#    Runs a single-timestep power flow to verify placement before benchmark.
# ===========================================================================

'''DER_BUS_LIST = [3, 8, 14, 11]    # bus indices from the labelled topology plot
DER_P_MW     = 0.500           # 500 kW per injected DER
DER_SN_MVA   = DER_P_MW * 1.1 # 10% oversized inverter
#
for bus in DER_BUS_LIST:
    pn.create_sgen(
        net_os,
        bus        = bus,
        p_mw       = DER_P_MW,
        sn_mva     = DER_SN_MVA,
        name       = f"injected_PV_bus{bus}",
        type       = "PV",
        in_service = True,
    )
logging.info(
    "Injected %d DERs at buses %s | total P = %.2f MW",
    len(DER_BUS_LIST), DER_BUS_LIST, len(DER_BUS_LIST) * DER_P_MW,
)
'''
# ===========================================================================
# ── Optional: inspect and manipulate network switches
#    Uncomment the print block first to see what switches exist.
#    Then uncomment the manipulation block to open/close specific ones.
#
#    net.switch columns:
#      bus      — bus index the switch is connected to
#      element  — index of the connected element (line, bus, or trafo)
#      et       — element type: 'l'=line, 'b'=bus, 't'=trafo
#      closed   — True = closed (conducting) | False = open (isolating)
#      name     — switch name from SimBench
# ===========================================================================

# ── Step 1: print all switches so you can identify indices ─────────────────
# print(net_os.switch[["name", "bus", "element", "et", "closed"]].to_string())

# ── Step 2: selectively open switches by index ─────────────────────────────
# Replace [0, 1, 2] with the actual switch indices from the print above.
# SWITCHES_TO_TOGGLE = [1, 2, 4]
# net_os.switch.loc[SWITCHES_TO_TOGGLE, "closed"] = True

# ── Step 3: verify the result ──────────────────────────────────────────────
# print("\nSwitch state after manipulation:")
# print(net_os.switch[["name", "bus", "element", "et", "closed"]].to_string())
# ===========================================================================

# ===========================================================================
# ── Publisher Live Handle configuration
# ===========================================================================
handle = PublishHandle(
    output_dir     = str(_root / "outputs" / "publisher" / (net_name + RUN_NAME)),
    update_every_k = 6,   # one JSONL frame per hour at 10-min resolution
)

handle_hc_stressed = PublishHandle(
    output_dir     = str(_root / "outputs" / "publisher" / (net_name + "_hc_stressed" + RUN_NAME)),
    update_every_k = 6,   # one JSONL frame per hour at 10-min resolution
)
# ===========================================================================
# ── Benchmark configuration
# ===========================================================================
config = BenchmarkConfig(
    scenarios   = [],   # all six; pass [1, 5] to run a subset
    dry_run     = True,               # no Arduino; set False + port= for hardware
    write_csv   = False,               # writes CSV to current directory
    output_dir = str(_root / "outputs" / "benchmarks"),
    # voltage_limits from the plugin YAML (defaults 0.95/1.05 otherwise)
    v_min = vd.V_MIN,
    v_max = vd.V_MAX,

    # port        = "/dev/ttyACM0", # uncomment if doing on hardware and make dry run = False. 
                                    # /dev/ttyACM0 for RPi, COM for Windows
    run_hc           = True,
    run_hc_scenarios = False,
    hc_stress_scenarios = [1, 2, 3, 4, 5, 10],  # scenarios to run on the HC-stressed net
    
    # For a full annual HC re-benchmark.  On the --network plugin path the
    # factory rebuilds profiles with the YAML's own strategy (flat /
    # simbench_native / dwd_pvlib) instead of forcing the DWD path.
    profile_factory = (
        make_profile_factory(_args.network)
        if _args.network else
        lambda net_hc: build_annual_profiles(
            net_hc,
            net_name      = net_name + "_hc_stressed" + RUN_NAME,
            data_dir      = str(_root / "data" / "dwd"),
            simbench_code = sb_code,
        )
    ),

    # For a sliced HC re-benchmark:
    # profile_factory  = lambda net_hc: slice_profiles(
    #     build_annual_profiles(
    #         net_hc,
    #         net_name      = net_name + "_hc_stressed" + RUN_NAME,
    #         data_dir      = str(_root / "data" / "dwd"),
    #         simbench_code = sb_code,
    #     ),
    #     period="month", index=6,   # mirror the outer slice
    # ),

    publish_fn = handle,  #Uncomment for Live transmission
    hc_publish_fn = handle_hc_stressed,
)

# ===========================================================================
# ── Verification: confirm DER scaling actually took effect
# ===========================================================================
_pf_check = (net_os.sgen["p_mw"] / net_os.sgen["sn_mva"]).round(4)
print(f"[SCALING CHECK] {net_name}: "
      f"p_mw sum={net_os.sgen['p_mw'].sum():.4f}, "
      f"sn_mva sum={net_os.sgen['sn_mva'].sum():.4f}, "
      f"PF values={_pf_check.unique().tolist()}")

# ===========================================================================
# ── Save Network Topolgy and Profiles as JSON
# ===========================================================================

# ── Publish topology + profiles once, at network-load time ──────────────
publish_topology_and_profiles(
    net_os, profiles,
    output_dir = str(_root / "outputs" / "publisher" / (net_name + RUN_NAME)),
    network_id = net_name,
)

# ===========================================================================
# ── Run ────────────────────────────────────────────────────────────────────
# ===========================================================================

if _args.controller:
    # Custom controller plugin: registers the YAML-configured controller as
    # an extra scenario (>= 10) and runs it ALONGSIDE config.scenarios.
    # Fully combinable with --network — the plugin net + plugin profiles
    # flow through unchanged.  The registry is cleaned in a finally block
    # inside register_and_run().
    from plugin_runner import register_and_run

    custom_result, result = register_and_run(
        _args.controller,
        net_os,         
        profiles,     # profiles_run for a slice of time series
        network_id       = net_name,
        benchmark_config = config,
        return_benchmark = True,
    )
    print(f"\n[plugin_runner] Custom controller '{custom_result.scenario_id}' "
          f"done: {custom_result.n_violation_steps} violation steps | "
          f"{custom_result.n_converged}/{custom_result.n_timesteps} converged")
else:
    result = run_benchmark(
        net_os,
        profiles, # profiles_run for a slice of time series
        network_id = net_name,
        config     = config,
    )


# ===========================================================================
# ── Publisher Post Run output ─────────────────────────────────────────────────────────
# ===========================================================================

written = publish_hc_and_comparison(
    result     = result,
    output_dir = str(_root / "outputs" / "publisher" / (net_name + RUN_NAME)),
)

if result.hc_benchmark is not None and result.net_hc is not None:
    written_hc = publish_hc_and_comparison(
        result     = result.hc_benchmark,
        output_dir = str(_root / "outputs" / "publisher" / (net_name + "_hc_stressed" + RUN_NAME)),
    )
else:
    written_hc = {}

print("\nPublisher wrote:")
for label, path in written.items():
    print(f"  {label:35s} → {path}")
for label, path in written_hc.items():
    print(f"  {label:35s} → {path}")

# ===========================================================================
# ── Optional: inspect results in Python ───────────────────────────────────
# ===========================================================================
print(result.comparison_df[["scenario_id", "n_violation_steps",
                             "violation_duration_h", "vdi",
                             "total_losses_mwh", "elapsed_s"]].to_string())

if result.csv_path:
    print(f"\nFull CSV: {result.csv_path}")

if result.hc_results:
    hc_b, hc_v = result.hc_results
    print(f"\nHosting Capacity:")
    print(f"  Baseline : {hc_b.hc_mw:.3f} MW (binding bus {hc_b.binding_bus}, {hc_b.binding_vm_pu:.4f} pu)")
    print(f"  Volt-Var : {hc_v.hc_mw:.3f} MW (binding bus {hc_v.binding_bus}, {hc_v.binding_vm_pu:.4f} pu)")
    print(f"  HC gain  : {hc_v.hc_mw - hc_b.hc_mw:+.3f} MW")

if result.hc_error:
    print(f"\nHC analysis failed: {result.hc_error.splitlines()[-1]}")

if result.errors:
    print(f"\nFailed scenarios: {sorted(result.errors.keys())}")

if result.hc_benchmark:
    print(f"\nHC-stressed re-benchmark ({result.hc_benchmark.network_id}):")
    print(result.hc_benchmark.comparison_df[
        ["scenario_id", "n_violation_steps", "vdi", "curtailment_steps", "elapsed_s"]
    ].to_string())