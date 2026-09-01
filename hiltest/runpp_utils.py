"""
hiltest/runpp_utils.py
======================
Shared pandapower.runpp candidate helpers.

Rules:
- NR is always tried first because it is pandapower's default.
- BFSW is only a fallback for general load-flow tests.
- SensitivityCoordinator must use Jacobian-capable solvers only, so no BFSW.
"""

from __future__ import annotations

from typing import Any
import pandapower as pp


BASE_RUNPP_KWARGS: dict[str, Any] = {
    "voltage_depend_loads": False,
}


def runpp_candidates(label: str) -> list[dict[str, Any]]:
    """
    General AC power-flow candidates.

    Use this for Volt-Var.
    """
    name = label.lower()

    candidates = [
        {**BASE_RUNPP_KWARGS, "algorithm": "nr", "max_iteration": 50, "init": "auto"},
        {**BASE_RUNPP_KWARGS, "algorithm": "nr", "max_iteration": 80, "init": "dc"},
        {**BASE_RUNPP_KWARGS, "algorithm": "iwamoto_nr", "max_iteration": 100, "init": "auto"},
        {**BASE_RUNPP_KWARGS, "algorithm": "iwamoto_nr", "max_iteration": 100, "init": "dc"},
    ]

    lv_like = any(
        marker in name
        for marker in ("1-lv-", "1-mvlv-", "synthetic", "dickert", "kerber", "cigre_lv")
    )

    if lv_like:
        candidates.append(
            {**BASE_RUNPP_KWARGS, "algorithm": "bfsw", "max_iteration": 100, "init": "flat"}
        )

    return candidates


def jacobian_runpp_candidates(label: str) -> list[dict[str, Any]]:
    """
    Runpp candidates for SensitivityCoordinator.

    Do not include BFSW. The coordinator needs the internal Newton Jacobian.
    """
    return [
        {**BASE_RUNPP_KWARGS, "algorithm": "nr", "max_iteration": 50, "init": "auto"},
        {**BASE_RUNPP_KWARGS, "algorithm": "nr", "max_iteration": 80, "init": "dc"},
        {**BASE_RUNPP_KWARGS, "algorithm": "iwamoto_nr", "max_iteration": 100, "init": "auto"},
        {**BASE_RUNPP_KWARGS, "algorithm": "iwamoto_nr", "max_iteration": 100, "init": "dc"},
    ]


def has_internal_jacobian(net) -> bool:
    return (
        getattr(net, "_ppc", None) is not None
        and "internal" in net._ppc
        and "J" in net._ppc["internal"]
    )


def run_controller_until_converged(ctrl, label: str):
    """
    Try VoltVarController.run_timestep() with candidate runpp kwargs.

    Returns:
        result, selected_kwargs, tried_kwargs
    """
    last_result = None
    tried: list[dict[str, Any]] = []

    for kwargs in runpp_candidates(label):
        tried.append(kwargs)
        last_result = ctrl.run_timestep(runpp_kwargs=kwargs)
        if getattr(last_result, "converged_pre", False):
            return last_result, kwargs, tried

    return last_result, None, tried


def select_jacobian_runpp_kwargs(net, label: str):
    """
    Run pp.runpp() until the network converges and the internal Jacobian exists.

    Returns:
        selected_kwargs, tried_kwargs, last_exception
    """
    tried: list[dict[str, Any]] = []
    last_exc: Exception | None = None

    for kwargs in jacobian_runpp_candidates(label):
        tried.append(kwargs)
        try:
            pp.runpp(net, **kwargs)
            if getattr(net, "converged", False) and has_internal_jacobian(net):
                return kwargs, tried, None
        except Exception as exc:
            last_exc = exc

    return None, tried, last_exc