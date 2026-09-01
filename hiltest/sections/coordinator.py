"""
hiltest/sections/coordinator.py
================================
Section: sensitivity_coordinator / sensitivity_coordinator_all

Changes from previous test_suite_section5.py (blockers applied)
----------------------------------------------------------------
B3  Canonical import: SensitivityCoordinator, CoordinatorResult, and
    run_coordinated_timestep are imported from `sensitivity_coordinator`
    only.  The `sensitivity_coordinator_claude` alias is gone.
B9  run_sensitivity_coordinator_all_tests: two additional assertions per
    network beyond post_pf_ok — q_adj_finite and n_ov_not_worsened —
    so a coordinator that converges but increases violations fails.
"""

import traceback
import time
import warnings

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import simbench as sb

from hiltest.framework  import TestCase, print_case
from hiltest.catalogues import (
    ALL_KERBER_CASES, ALL_SYNTHETIC_LV_CASES, ALL_DICKERT_CASES,
    IN_SCOPE_SIMBENCH_CODES,
)
from hiltest.constants  import HIL_MAX_CYCLE_MS, HIL_MAX_EXCHANGE_MS
from hiltest.networks   import get_representative_networks
from hiltest.stress     import (
    apply_overvoltage_stress, apply_hw_synthetic_stress,
    stress_description,
)
from hiltest.runpp_utils import select_jacobian_runpp_kwargs

# B3: single canonical import — no _claude alias
from sensitivity_coordinator import (
    SensitivityCoordinator,
    CoordinatorResult,
    run_coordinated_timestep,
)
from violation_detector import ViolationReport
from volt_var_controller import VoltVarController, Q_RATIO


def _case_name(prefix: str, net_name: str, label: str | None = None) -> str:
    """
    Build a readable test-case name.

    net_name stays short for filtering, for example sb_mvlv_rural.
    label shows the actual network, for example 1-MVLV-rural-all-0-sw.
    """
    if label and label != net_name:
        return f"{prefix}_{net_name} [{label}]"
    return f"{prefix}_{net_name}"

# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_coordinator_summary(records: list, top_n: int = 10) -> None:
    """Ranked tables of sensitivity coordinator effectiveness."""
    valid = [r for r in records if r.get("n_ders") is not None]
    if not valid:
        print("\n  [coord summary] No valid results.")
        return

    W = 88
    print(f"\n{'='*W}")
    print("  SENSITIVITY COORDINATOR SUMMARY")
    print(stress_description())
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
            print(f"  {i:<4} {r['name']:<42} "
                  f"{r[val_key]:>12{val_fmt}} "
                  f"{r.get('n_ders') or 0:>6}")

    _table("Most overvoltage buses pre-control",
           valid, "n_ov_pre", "d", "buses OV", reverse=True)
    _table("Largest coordinator correction over Item 2 (max|dQ|)",
           valid, "max_dq_corr", ".4f", "dQ MVAr", reverse=True)
    _table("Slowest timesteps (wall-clock)",
           valid, "t_ms", ".1f", "t_ms", reverse=True)

    has_pre     = [r for r in valid if (r.get("n_ov_pre") or 0) > 0]
    n_resolved  = sum(1 for r in has_pre if r.get("violations_resolved"))
    n_partial   = sum(1 for r in has_pre
                      if not r.get("violations_resolved")
                      and r.get("post_pf_ok")
                      and (r.get("n_ov_post") or 0) < r["n_ov_pre"])
    n_no_imp    = sum(1 for r in has_pre
                      if not r.get("violations_resolved")
                      and r.get("post_pf_ok")
                      and (r.get("n_ov_post") or 0) >= r["n_ov_pre"])
    n_no_viol   = sum(1 for r in valid if not (r.get("n_ov_pre") or 0))
    n_pf_failed = sum(1 for r in records if r.get("n_ders") is None)

    print("\n  Voltage violation outcomes:")
    print(f"  {'No pre-violations:':<44} {n_no_viol:>4} / {len(valid)}")
    print(f"  {'Fully resolved by coordinator:':<44} {n_resolved:>4} / {len(valid)}")
    print(f"  {'Partially reduced:':<44} {n_partial:>4} / {len(valid)}")
    print(f"  {'No improvement:':<44} {n_no_imp:>4} / {len(valid)}")
    print(f"  {'Failed (PF non-convergence or exception):':<44} {n_pf_failed:>4} / {len(records)}")
    print(f"{'='*W}\n")


# ---------------------------------------------------------------------------
# Section: sensitivity_coordinator
# ---------------------------------------------------------------------------

def run_sensitivity_coordinator_tests(
        verbose:      bool = False,
        only:         list = None,
        arduino_port: str  = None,
        only_hw:      bool = False,
) -> list:
    """
    Sensitivity Coordinator tests — 9 representative networks.

    [1/3]  CoordinatorResult unit tests  (no network)
    [2/3]  Per-network dry-run — construction, guards, output, integration
    [3/3]  Hardware (Arduino) — skipped if no --arduino-port
    """
    cases              = []
    coord_rep_records  = []

    # Build representative network list here — deferred import of pn/sb
    REPRESENTATIVE_NETWORKS = get_representative_networks()

    if not only_hw:

        # -------------------------------------------------------------------
        # [1/3]  CoordinatorResult unit tests
        # -------------------------------------------------------------------
        print("\n  [1/3] CoordinatorResult unit tests")
        name = "coordinator_result_unit"
        if not (only and not any(s in name for s in only)):
            tc = TestCase(name)
            t0 = time.perf_counter()
            try:
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
                cr    = CoordinatorResult(
                    report_pre=_rep_viol, report_post=_rep_clean,
                    q_initial=q_ini, q_adjusted=q_adj,
                    curtailment_needed=False, post_pf_ok=True,
                )

                tc.record("dq_correction_arithmetic",
                    np.allclose(cr.dq_correction, q_adj - q_ini),
                    f"expected {q_adj - q_ini}, got {cr.dq_correction}")
                tc.record("violations_resolved_true",
                    cr.violations_resolved is True,
                    "pre=violated, post=clean, post_pf_ok=True → must be True")

                _cr_nopf = CoordinatorResult(
                    report_pre=_rep_viol, report_post=_rep_clean,
                    q_initial=q_ini, q_adjusted=q_adj,
                    curtailment_needed=False, post_pf_ok=False,
                )
                tc.record("violations_resolved_false_post_pf_failed",
                    _cr_nopf.violations_resolved is False,
                    "post_pf_ok=False → must return False")

                _cr_nopost = CoordinatorResult(
                    report_pre=_rep_viol, report_post=None,
                    q_initial=q_ini, q_adjusted=q_adj,
                    curtailment_needed=False, post_pf_ok=True,
                )
                tc.record("violations_resolved_false_report_post_none",
                    _cr_nopost.violations_resolved is False,
                    "report_post=None → must return False")

                _cr_nopre = CoordinatorResult(
                    report_pre=_rep_clean, report_post=_rep_clean,
                    q_initial=q_ini, q_adjusted=q_adj,
                    curtailment_needed=False, post_pf_ok=True,
                )
                tc.record("violations_resolved_false_no_pre_violations",
                    _cr_nopre.violations_resolved is False,
                    "no pre-violations → violations_resolved must be False")

                summary_str = cr.summary()
                tc.record("summary_returns_string",
                    isinstance(summary_str, str)
                    and "CoordResult" in summary_str
                    and len(summary_str) > 0,
                    f"summary()={summary_str!r}")

                _cr_def = CoordinatorResult(
                    report_pre=_rep_clean, report_post=_rep_clean,
                    q_initial=np.zeros(2), q_adjusted=np.zeros(2),
                    curtailment_needed=False, post_pf_ok=True,
                )
                tc.record("default_n_retries_zero",
                    _cr_def.n_retries == 0, f"n_retries={_cr_def.n_retries}")
                tc.record("default_t_total_ms_zero",
                    _cr_def.t_total_ms == 0.0,
                    f"t_total_ms={_cr_def.t_total_ms}")

            except Exception:
                tc.error = traceback.format_exc()
            tc.duration = time.perf_counter() - t0
            cases.append(tc)
            print_case(tc, verbose)

        # -------------------------------------------------------------------
        # [2/3]  Per-network dry-run
        # -------------------------------------------------------------------
        print(f"\n  [2/3] Coordinator dry_run — "
              f"{len(REPRESENTATIVE_NETWORKS)} representative networks")

        for net_name, loader, label in REPRESENTATIVE_NETWORKS:
            if only and not any(s in net_name for s in only):
                continue
            tc = TestCase(_case_name("coord_dry", net_name, label))
            t0  = time.perf_counter()
            rec = {"name": label}
            try:
                net = loader()
                apply_overvoltage_stress(net)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ctrl = VoltVarController(net, interface=None, dry_run=True)
                    ctrl.configure()   # populates n_ders, sgen_indices, p_installed_mw
                    coord = SensitivityCoordinator(net, ctrl)

                if ctrl.n_ders == 0:
                    tc.record("skipped_no_der", True, "n_ders=0 — SKIP")
                    tc.skipped = True
                    cases.append(tc)
                    coord_rep_records.append(rec)
                    print_case(tc, verbose)
                    continue

                # Construction checks
                tc.record("curtailment_init",
                    coord.curtailment_needed is False,
                    f"curtailment_needed={coord.curtailment_needed}")
                tc.record("qmax_formula",
                    np.allclose(coord._q_max, Q_RATIO * ctrl.p_installed_mw,
                                atol=1e-9),
                    "max|_q_max - Q_RATIO×p_inst| out of tolerance")
                tc.record("qmax_shape",
                    coord._q_max.shape == (ctrl.n_ders,),
                    f"shape={coord._q_max.shape}, expected=({ctrl.n_ders},)")

                # Pre-PF for Jacobian population. Coordinator requires NR/Iwamoto Jacobian
                selected_runpp_kw, tried_runpp, last_exc = select_jacobian_runpp_kwargs(net,label)

                tc.record(
                    "pre_pf_jacobian_available",
                    selected_runpp_kw is not None,
                    f"No Jacobian-capable runpp candidate converged; "
                    f"tried={tried_runpp}; "
                    f"last_exc={type(last_exc).__name__ if last_exc else None}: {last_exc}",
                )

                if selected_runpp_kw is None:
                    rec.update({k: None for k in [
                        "n_ders","n_ov_pre","n_ov_post","violations_resolved",
                        "max_dq_corr","q_adj_max","t_ms","post_pf_ok",
                    ]})
                    tc.duration = time.perf_counter() - t0
                    cases.append(tc)
                    coord_rep_records.append(rec)
                    print_case(tc, verbose)
                    continue

                # Guard 1: wrong-length q_initial → ValueError
                # Capture unexpected exception type for debugging.
                _raised_ve = False
                _unexpected_ve: str = ""
                try:
                    coord.coordinate(np.zeros(ctrl.n_ders + 1))
                except ValueError:
                    _raised_ve = True
                except Exception as exc:
                    _unexpected_ve = f" (got {type(exc).__name__}: {exc})"
                tc.record("guard_valueerror_wrong_length", _raised_ve,
                    f"expected ValueError for length {ctrl.n_ders + 1}"
                    f"{_unexpected_ve}")

                # Guard 2: non-zero q_mvar → RuntimeError
                net.sgen.loc[ctrl.sgen_indices, "q_mvar"] = 1.0
                _raised_rte = False
                _unexpected_rte: str = ""
                try:
                    coord.coordinate(np.zeros(ctrl.n_ders))
                except RuntimeError:
                    _raised_rte = True
                except Exception as exc:
                    _unexpected_rte = f" (got {type(exc).__name__}: {exc})"
                tc.record("guard_runtimeerror_nonzero_qmvar", _raised_rte,
                    f"expected RuntimeError when q_mvar != 0{_unexpected_rte}")
                net.sgen.loc[ctrl.sgen_indices, "q_mvar"] = 0.0

                # Integration
                result = run_coordinated_timestep(
                    net, ctrl, coord, runpp_kwargs=selected_runpp_kw
                )
                tc.record("post_pf_ok", result.post_pf_ok,
                    "post-PF runpp() must converge")
                tc.record("q_adj_finite",
                    bool(np.isfinite(result.q_adjusted).all()),
                    f"{int((~np.isfinite(result.q_adjusted)).sum())} non-finite values")
                _qmax = Q_RATIO * ctrl.p_installed_mw
                tc.record("q_adj_clip_bound",
                    bool((np.abs(result.q_adjusted) <= _qmax + 1e-9).all()),
                    f"max excess = "
                    f"{float(np.maximum(0, np.abs(result.q_adjusted)-_qmax).max()):.6f}")
                tc.record("t_ms_positive", result.t_total_ms > 0.0,
                    f"t_ms={result.t_total_ms:.2f}")

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
                    "n_ders","n_ov_pre","n_ov_post","violations_resolved",
                    "max_dq_corr","q_adj_max","t_ms","post_pf_ok",
                ]})
            tc.duration = time.perf_counter() - t0
            cases.append(tc)
            coord_rep_records.append(rec)
            print_case(tc, verbose)

        _print_coordinator_summary(coord_rep_records)

    # -----------------------------------------------------------------------
    # [3/3]  Hardware
    # -----------------------------------------------------------------------
    hw_label = arduino_port if arduino_port else "SKIPPED — pass --arduino-port"
    print(f"\n  [3/3] Coordinator hardware  ({hw_label})")

    if arduino_port:
        from volt_var_controller import ArduinoSerialInterface

        for net_name, loader, label in REPRESENTATIVE_NETWORKS:
            if only and not any(s in net_name for s in only):
                continue
            tc = TestCase(_case_name("coord_hw", net_name, label))
            t0 = time.perf_counter()
            try:
                net = loader()
                apply_overvoltage_stress(net)
                if "synthetic" in label.lower():
                    apply_hw_synthetic_stress(net)

                with ArduinoSerialInterface(port=arduino_port) as arduino:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ctrl_hw = VoltVarController(net, interface=arduino,
                                                    dry_run=False)
                        ctrl_hw.configure()

                    if ctrl_hw.n_ders == 0:
                        tc.record("hw_no_der", True, "n_ders=0 — SKIP")
                        tc.skipped = True
                        cases.append(tc)
                        print_case(tc, verbose)
                        continue

                    coord = SensitivityCoordinator(net, ctrl_hw)
                    net.sgen.loc[ctrl_hw.sgen_indices, "q_mvar"] = 0.0
                    selected_runpp_kw, tried_runpp, last_exc = select_jacobian_runpp_kwargs(net, label)

                    tc.record(
                        "hw_pre_pf_jacobian_available",
                        selected_runpp_kw is not None,
                        f"no Jacobian-capable runpp candidate converged; "
                        f"tried={tried_runpp}; "
                        f"last_exc={type(last_exc).__name__ if last_exc else None}: {last_exc}",
                    )

                    if selected_runpp_kw is None:
                        tc.duration = time.perf_counter() - t0
                        cases.append(tc)
                        print_case(tc, verbose)
                        continue

                    result = run_coordinated_timestep(
                        net, ctrl_hw, coord, runpp_kwargs=selected_runpp_kw
                    )

                tc.record("hw_post_pf_ok", result.post_pf_ok,
                    "post-PF must converge after hardware Q exchange")
                tc.record("hw_q_adj_finite",
                    bool(np.isfinite(result.q_adjusted).all()),
                    f"max|q_adj|={float(np.abs(result.q_adjusted).max()):.4f}")
                _qmax = Q_RATIO * ctrl_hw.p_installed_mw
                tc.record("hw_q_adj_clipped",
                    bool((np.abs(result.q_adjusted) <= _qmax + 1e-9).all()),
                    "q_adjusted exceeds ±q_max bounds")
                # t_total_ms covers the full cycle: 2× runpp + Schur + serial.
                # Use HIL_MAX_CYCLE_MS (800 ms), not the serial-only budget.
                tc.record("hw_cycle_within_budget",
                    result.t_total_ms < HIL_MAX_CYCLE_MS,
                    f"cycle={result.t_total_ms:.1f}ms "
                    f"exceeds HIL cycle budget {HIL_MAX_CYCLE_MS:.0f}ms"
                    f" (serial budget is {HIL_MAX_EXCHANGE_MS:.0f}ms)")
                print(f"         {label}: t={result.t_total_ms:.0f}ms  "
                      f"n_ders={ctrl_hw.n_ders}  retries={result.n_retries}  "
                      f"resolved={result.violations_resolved}")
            except Exception:
                tc.error = traceback.format_exc()
            tc.duration = time.perf_counter() - t0
            cases.append(tc)
            print_case(tc, verbose)

    return cases


# ---------------------------------------------------------------------------
# Section: sensitivity_coordinator_all
# ---------------------------------------------------------------------------

def run_sensitivity_coordinator_all_tests(
        verbose:      bool = False,
        only:         list = None,
        arduino_port: str  = None,
        only_hw:      bool = False,
) -> list:
    """
    Coordinator sweep across all 199 in-scope networks.

    B9 fix: each network now asserts three checks (not just post_pf_ok):
    - post_pf_ok         : PF must converge after Q application
    - q_adj_finite       : no NaN/Inf in the setpoint vector
    - n_ov_not_worsened  : coordinator must not increase violation count
    """
    cases              = []
    coord_all_records  = []

    print("\n  [1/1] Coordinator dry_run — all in-scope networks")

    def _run_coord_one(tc: TestCase, name: str, loader_fn) -> dict:
        try:
            net = loader_fn()
            apply_overvoltage_stress(net)
            selected_runpp_kw, tried_runpp, last_exc = select_jacobian_runpp_kwargs(net, name)

            tc.record(
                "pre_pf_jacobian_available",
                selected_runpp_kw is not None,
                f"no Jacobian-capable runpp candidate converged; "
                f"tried={tried_runpp}; "
                f"last_exc={type(last_exc).__name__ if last_exc else None}: {last_exc}",
            )

            if selected_runpp_kw is None:
                return {"name": name, **{k: None for k in [
                    "n_ders","n_ov_pre","n_ov_post","violations_resolved",
                    "max_dq_corr","q_adj_max","t_ms","post_pf_ok",
                ]}}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                ctrl = VoltVarController(net, interface=None, dry_run=True)
                ctrl.configure()   # populates n_ders, sgen_indices, p_installed_mw
                coord = SensitivityCoordinator(net, ctrl)

            if ctrl.n_ders == 0:
                tc.record("skipped_no_der", True, "n_ders=0 — SKIP")
                tc.skipped = True
                return {"name": name, **{k: None for k in [
                    "n_ders","n_ov_pre","n_ov_post","violations_resolved",
                    "max_dq_corr","q_adj_max","t_ms","post_pf_ok",
                ]}}

            result = run_coordinated_timestep(
                net, ctrl, coord, runpp_kwargs=selected_runpp_kw
            )

            # B9: three assertions — convergence, finiteness, no worsening
            tc.record("post_pf_ok", result.post_pf_ok,
                f"n_ov_pre={result.report_pre.n_over_voltage}")
            tc.record("q_adj_finite",
                bool(np.isfinite(result.q_adjusted).all()),
                "NaN/Inf in q_adjusted")

            n_pre  = result.report_pre.n_over_voltage
            n_post = (result.report_post.n_over_voltage
                      if result.report_post else n_pre)
            tc.record("n_ov_not_worsened", n_post <= n_pre,
                f"coordinator worsened: n_ov_pre={n_pre} → n_ov_post={n_post}")

            return {
                "name":               name,
                "n_ders":             ctrl.n_ders,
                "n_ov_pre":           n_pre,
                "n_ov_post":          n_post,
                "violations_resolved":result.violations_resolved,
                "max_dq_corr":        float(np.abs(result.dq_correction).max()),
                "q_adj_max":          float(np.abs(result.q_adjusted).max()),
                "t_ms":               result.t_total_ms,
                "post_pf_ok":         result.post_pf_ok,
            }
        except Exception:
            tc.error = traceback.format_exc()
            return {"name": name, **{k: None for k in [
                "n_ders","n_ov_pre","n_ov_post","violations_resolved",
                "max_dq_corr","q_adj_max","t_ms","post_pf_ok",
            ]}}

    # SimBench 156
    for code in IN_SCOPE_SIMBENCH_CODES:
        if only and not any(s in code for s in only):
            continue
        tc = TestCase(_case_name("coord", code))
        t0 = time.perf_counter()
        rec = _run_coord_one(tc, code, lambda c=code: sb.get_simbench_net(c))
        tc.duration = time.perf_counter() - t0
        cases.append(tc); coord_all_records.append(rec)
        print_case(tc, verbose)

    # CIGRE 2
    for name, loader in [
        ("cigre_mv_with_der",
         lambda: pn.create_cigre_network_mv(with_der="pv_wind")),
        ("cigre_lv", lambda: pn.create_cigre_network_lv()),
    ]:
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(_case_name("coord", name))
        t0 = time.perf_counter()
        rec = _run_coord_one(tc, name, loader)
        tc.duration = time.perf_counter() - t0
        cases.append(tc); coord_all_records.append(rec)
        print_case(tc, verbose)

    # Kerber 17
    for name, fn_name in ALL_KERBER_CASES:
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(_case_name("coord", name))
        t0 = time.perf_counter()
        rec = _run_coord_one(tc, name, lambda f=fn_name: getattr(pn, f)())
        tc.duration = time.perf_counter() - t0
        cases.append(tc); coord_all_records.append(rec)
        print_case(tc, verbose)

    # Synthetic LV 5
    for network_class in ALL_SYNTHETIC_LV_CASES:
        name = f"synthetic_lv_{network_class}"
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(_case_name("coord", name))
        t0 = time.perf_counter()
        rec = _run_coord_one(
            tc, name,
            lambda c=network_class:
                pn.create_synthetic_voltage_control_lv_network(c),
        )
        tc.duration = time.perf_counter() - t0
        cases.append(tc); coord_all_records.append(rec)
        print_case(tc, verbose)

    # Dickert 18
    for name, feeders_range, linetype, customer, case in ALL_DICKERT_CASES:
        if only and not any(s in name for s in only):
            continue
        tc = TestCase(_case_name("coord", name))
        t0 = time.perf_counter()
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
                "n_ders","n_ov_pre","n_ov_post","violations_resolved",
                "max_dq_corr","q_adj_max","t_ms","post_pf_ok",
            ]}}
        tc.duration = time.perf_counter() - t0
        cases.append(tc); coord_all_records.append(rec)
        print_case(tc, verbose)

    _print_coordinator_summary(coord_all_records)
    return cases
