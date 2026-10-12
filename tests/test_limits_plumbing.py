"""
test_limits_plumbing.py
=======================
Regression guards for the runtime-override plumbing.

Every check here corresponds to a defect that was live in the codebase and is
invisible to the rest of tests/: those are characterisation and smoke scripts
that print results for a human to read, so a silently wrong limit produced a
plausible-looking number and nothing failed.

The central one is test_record_agrees_with_report(). Before the fix, the
scenario runners from-imported LINE_MAX_LOADING / TRAFO_MAX_LOADING, which
freezes the import-time value, while detect_violations() resolved them at call
time. On any run with an overridden thermal limit the control decisions used
the override and the recorded violation lists used 100 %, so the published
counts disagreed with the behaviour that produced them.

Runs under pytest, or standalone:

    python tests/test_limits_plumbing.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "scenario_runners"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

import violation_detector as vd


def _solved_mv_net():
    net = pn.create_cigre_network_mv(with_der="pv_wind")
    pp.runpp(net, voltage_depend_loads=False)
    return net


# ---------------------------------------------------------------------------
# The core invariant
# ---------------------------------------------------------------------------

def test_record_agrees_with_report():
    """
    A TimestepRecord's violation lists must be built against the SAME limits
    detect_violations() used. Any from-import of a limit constant in a runner
    breaks this.
    """
    from scenario_result import make_record_from_report

    net = _solved_mv_net()
    worst = float(net.res_line["loading_percent"].max())

    vd.reset_limits()
    try:
        # A limit strictly below the worst line, so something IS overloaded.
        vd.set_limit("LINE_MAX_LOADING", worst * 0.5,
                     0.0, vd.THERMAL_LOADING_PLAUSIBLE_MAX, "%", warn=print)

        report = vd.detect_violations(net)
        record = make_record_from_report(
            0, pd.Timestamp("2016-01-01"), net, converged=True,
        )

        assert report.n_overloaded_lines > 0, (
            "test setup is wrong: the tightened limit produced no overload"
        )
        assert len(record.overloaded_lines) == report.n_overloaded_lines, (
            f"record counted {len(record.overloaded_lines)} overloaded lines "
            f"but the report counted {report.n_overloaded_lines} — a limit "
            f"constant is frozen at import somewhere in the record path"
        )
    finally:
        vd.reset_limits()


def test_no_runner_from_imports_a_thermal_limit():
    """
    Static guard: a from-import of LINE_MAX_LOADING / TRAFO_MAX_LOADING freezes
    the value, so the override silently stops propagating. Read them through
    the module instead (`import violation_detector as _vd`).
    """
    offenders = []
    for path in sorted((_ROOT / "scenario_runners").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "from violation_detector import" not in text:
            continue
        block = text.split("from violation_detector import", 1)[1].split(")", 1)[0]
        for name in ("LINE_MAX_LOADING", "TRAFO_MAX_LOADING"):
            if name in block:
                offenders.append(f"{path.name}: {name}")

    assert not offenders, (
        "these modules freeze a thermal limit at import time: "
        + ", ".join(offenders)
    )


def test_reset_limits_restores_everything():
    """set_limit() mutates module globals process-wide; reset must undo all."""
    # Named explicitly rather than iterating _LIMIT_DEFAULTS: that dict is the
    # thing under test, so looping over it would pass by construction for any
    # constant missing from it.
    guarded = (
        "V_MIN", "V_MAX", "LINE_MAX_LOADING", "TRAFO_MAX_LOADING",
        "VA_DIFF_MAX_DEGREE", "UNBALANCE_MAX_PERCENT",
        "THERMAL_LOADING_PLAUSIBLE_MAX",
    )

    vd.reset_limits()
    before = {k: getattr(vd, k) for k in guarded}
    try:
        vd.set_limit("LINE_MAX_LOADING", 55.0, 0.0, 300.0, "%", warn=print)
        vd.set_limit("VA_DIFF_MAX_DEGREE", 11.0, 0.0, 180.0, "deg", warn=print)
        vd.THERMAL_LOADING_PLAUSIBLE_MAX = 999.0
        vd.reset_limits()
        after = {k: getattr(vd, k) for k in guarded}
        missing = [k for k in guarded if k not in vd._LIMIT_DEFAULTS]
        assert not missing, f"_LIMIT_DEFAULTS does not cover: {missing}"
        assert after == before, f"reset_limits() left {after} != {before}"
    finally:
        vd.reset_limits()


def test_benchmark_config_voltage_defaults_are_late_bound():
    """
    A bare dataclass default is evaluated at class-definition time, so it would
    pin the import-time voltage band and ignore a later override.
    """
    from benchmark_runner import BenchmarkConfig

    vd.reset_limits()
    try:
        vd.V_MIN, vd.V_MAX = 0.90, 1.10
        cfg = BenchmarkConfig()
        assert (cfg.v_min, cfg.v_max) == (0.90, 1.10), (
            f"BenchmarkConfig froze the band at {cfg.v_min}/{cfg.v_max}"
        )
    finally:
        vd.reset_limits()


def test_scenario_5_reads_q_ratio_late():
    """set_qv_parameters() must reach the OPF reactive bound."""
    import scenario_5_opf as s5

    assert not hasattr(s5, "Q_RATIO"), (
        "scenario_5_opf from-imports Q_RATIO, so a per-run override never "
        "reaches q_vde in _write_timestep_opf_state()"
    )
    assert hasattr(s5, "_vvc")


def test_cyipopt_import_is_guarded():
    """
    benchmark_runner imports scenario_5_opf at top level, so an unguarded
    cyipopt ImportError would make all six scenarios unrunnable.
    """
    import scenario_5_opf as s5
    assert hasattr(s5, "_HAS_CYIPOPT")


# ---------------------------------------------------------------------------
# DER dynamics parameterisation
# ---------------------------------------------------------------------------

def test_dynamics_presets_are_reachable_and_differ():
    """
    The SLOW / FAST presets in der_dynamics.py had no caller at all: the ramp
    parameters were pinned inside _build_dynamics().
    """
    import numpy as np
    from der_dynamics import DERDynamics, MV_T95_Q_S, MV_T95_Q_SLOW_S, MV_RAMP_RATE_P_BASE

    kw = dict(dt_s=10.0, ramp_rate_p_frac_s=MV_RAMP_RATE_P_BASE,
              q_max_mvar=np.array([1.0]), p_rated_mw=np.array([2.0]))
    base = DERDynamics(t95_q_s=MV_T95_Q_S, **kw)
    slow = DERDynamics(t95_q_s=MV_T95_Q_SLOW_S, **kw)

    assert slow.alpha < base.alpha / 2, (
        f"the slow preset barely changed the filter: {slow.alpha:.4f} vs "
        f"{base.alpha:.4f} — is t95_q_s actually being used?"
    )


def test_scenario_4_accepts_dynamics_kwargs():
    import scenario_4_volt_var as s4

    sig = inspect.signature(s4.run_scenario_4).parameters
    for name in ("t95_q_s", "ramp_rate_p_frac_s"):
        assert name in sig, f"run_scenario_4 does not accept {name}"


def test_build_kwargs_forwards_dynamics_only_where_supported():
    """
    Scenarios 1-3 would raise TypeError on an unexpected kwarg, so the
    injection must stay gated on the ScenarioSpec flag.
    """
    from benchmark_runner import BenchmarkConfig, SCENARIO_REGISTRY, _build_kwargs

    cfg = BenchmarkConfig(t95_q_s=60.0, ramp_rate_p_frac_s=0.0033)
    assert _build_kwargs(SCENARIO_REGISTRY[4], cfg, "n")["t95_q_s"] == 60.0
    assert "t95_q_s" not in _build_kwargs(SCENARIO_REGISTRY[1], cfg, "n")


def test_scenario_1_clears_its_timing_accumulator():
    """
    _RUNPP_TIMING is module-level and _timed_runpp accumulates into it, so a
    second baseline run in one process reported the sum of both.
    """
    import scenario_1_baseline as s1
    assert "_RUNPP_TIMING.clear()" in inspect.getsource(s1.run_scenario_1)


def test_profile_builder_does_not_silence_warnings_globally():
    """
    An unqualified filterwarnings("ignore") here hid the Arduino fallback and
    empty-result-table warnings from every other module.
    """
    import warnings
    import profile_builder  # noqa: F401  (importing it is the point)

    # Inspect the installed filters directly. A catch_warnings block with
    # simplefilter("always") would reset them and pass regardless, which is
    # exactly how this defect stayed invisible.
    catch_all = [
        f for f in warnings.filters
        if f[0] == "ignore" and f[2] is Warning and f[1] is None and not f[3]
    ]
    assert not catch_all, (
        "an unqualified filterwarnings('ignore') is installed — it silences "
        "the Arduino fallback and empty-result-table warnings from every "
        "other module, process-wide"
    )


def test_svc_placement_is_centralised_and_paths_agree():
    """
    Scenario 3 benchmarks a single centralised SVC at the substation MV
    busbar, so every selection path must target that bus. The old chain
    ranked its fallback by largest voltage deviation, which placed the
    device at the most remote feeder bus whenever the stress module was
    unavailable — a different configuration reported under the same name.
    """
    import scenario_3_svc as s3

    net      = pn.create_cigre_network_mv(with_der="pv_wind")
    mv_buses = s3._eligible_mv_buses(net)
    busbars  = set(net.trafo["lv_bus"].astype(int))

    assert not hasattr(s3, "_select_svc_bus_first_violation"), (
        "the largest-deviation selector is back; it places the SVC at a "
        "remote feeder bus, not at the substation"
    )

    chosen = s3._select_svc_bus(net)
    assert chosen in busbars, (
        f"SVC placed at bus {chosen}, which is not a transformer LV busbar "
        f"{sorted(busbars)} — placement is no longer centralised"
    )

    by_trafo = s3._select_svc_bus_substation(net, mv_buses)
    by_hops  = s3._select_svc_bus_nearest_slack(net, mv_buses)
    assert by_trafo in busbars
    assert by_hops in busbars, (
        f"the fallback chose bus {by_hops}, outside the busbar set "
        f"{sorted(busbars)} — it disagrees with the primary path"
    )

    # A voltage ranking must not come back as the fallback: under the
    # overvoltage stress case CIGRE MV's load-dominated feeder 2 puts the
    # minimum at bus 14, the far end, not at a busbar.
    assert not hasattr(s3, "_select_svc_bus_stress"), (
        "placement is ranking buses by voltage again"
    )

    # The last resort must not reach for the far end of the feeder either.
    assert int(mv_buses[0]) < int(mv_buses[-1])
    assert "mv_buses[0]" in inspect.getsource(s3._select_svc_bus), (
        "the last-resort branch still takes the highest-index (most remote) "
        "MV bus"
    )


def test_hc_snapshot_leaves_the_qv_headroom_it_promises():
    """
    The worst-case snapshot set p_mw = sn_mva, i.e. P = S, so the controller's
    apparent-power clamp left |Q| <= sqrt(S^2 - P^2) = 0 on every pre-placed
    DER. Case B then had no reactive capability to deploy and was identical to
    Case A by construction, on any network.
    """
    import math

    import hosting_capacity as hc
    import volt_var_controller as vvc

    net = pn.create_cigre_network_mv(with_der="pv_wind")
    hc._set_worst_case_snapshot(net)

    sn = net.sgen["sn_mva"].astype(float).to_numpy()
    p  = net.sgen["p_mw"].astype(float).to_numpy()
    ok = (sn > 0) & ~pd.isna(sn) & ~pd.isna(p)
    assert ok.any(), "test setup: CIGRE MV should have rated sgens"

    q_lim = (sn[ok] ** 2 - p[ok] ** 2).clip(min=0.0) ** 0.5
    assert q_lim.sum() > 0.0, (
        "the snapshot leaves zero fleet Q headroom — p_mw is still being set "
        "to the apparent rating"
    )

    # Headroom must be exactly the Q(V) characteristic's own ratio, and must
    # follow a per-run override of it rather than a hardcoded factor.
    expected = float(vvc.Q_RATIO) * p[ok]
    assert np.allclose(q_lim, expected, rtol=1e-6), (
        "fleet Q headroom is not Q_RATIO x P — the derating factor and the "
        "Q(V) characteristic have drifted apart"
    )
    assert math.isclose(
        hc._p_at_max_irradiance(1.0),
        1.0 / math.sqrt(1.0 + float(vvc.Q_RATIO) ** 2),
        rel_tol=1e-12,
    )


def test_hc_gain_is_measurable_where_the_method_applies():
    """
    CIGRE MV has pre-placed DER and a compliant snapshot, so it is the case
    where an HC gain is meaningful. Before the headroom fix both sweeps were
    byte-identical and the gain was exactly +0.000 MW.
    """
    import hosting_capacity as hc

    net = pn.create_cigre_network_mv(with_der="pv_wind")
    base, _ = hc.run_baseline_hc(net, "test")
    vv      = hc.run_hc_with_volt_var(net, "test")

    assert base.hc_mw == base.hc_mw, "baseline HC is NaN: snapshot infeasible"
    assert vv.hc_mw > base.hc_mw, (
        f"HC gain is {vv.hc_mw - base.hc_mw:+.3f} MW — Q(V) bought nothing, "
        f"so the controlled fleet has no reactive headroom"
    )


def test_hc_reports_nan_not_zero_when_the_snapshot_already_violates():
    """
    hc_mw was left at params["start"] (0.0) when the very first step violated,
    so a network already non-compliant under the snapshot published a flat
    0.00 MW that read like a measurement.
    """
    import math

    import hosting_capacity as hc

    net = pn.create_cigre_network_mv(with_der="pv_wind")
    # Force the premise to fail: a band this tight is violated by the snapshot.
    vd.reset_limits()
    try:
        vd.set_limit("V_MAX", 1.0001, 0.5, 1.5, "pu", warn=print)
        res, _ = hc.run_baseline_hc(net, "test")
        assert math.isnan(res.hc_mw), (
            f"hc_mw came back as {res.hc_mw}, not NaN — an infeasible "
            f"snapshot is still reported as a measured hosting capacity"
        )
        assert "already violating" in res.binding_element, res.binding_element
        assert res.n_steps == 0
    finally:
        vd.reset_limits()


def test_dwd_discovery_handles_five_minute_files():
    """The default glob was pinned to PT10M and to the hyphenated RAD-G code."""
    import profile_builder as pb

    src = inspect.getsource(pb.find_dwd_file)
    assert "PT*M" in src, "the DWD glob still cannot see PT5M files"
    assert 'replace("-", "")' in src, "RAD-G will not match a ...RADG_<id>.csv file"


# ---------------------------------------------------------------------------

def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed.append(fn.__name__)
            print(f"  FAIL  {fn.__name__}\n        {exc}")
        except Exception as exc:                     # noqa: BLE001
            failed.append(fn.__name__)
            print(f"  ERROR {fn.__name__}\n        {type(exc).__name__}: {exc}")
    print("-" * 60)
    print(f"{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
