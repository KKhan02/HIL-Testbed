"""
hiltest/sections/__init__.py
=============================
Section registry — lazy import pattern.

Fix (Blocker 2 — lazy imports): previously all section modules were imported
at module level, meaning `python -m hiltest --help` pulled in pandapower,
simbench, and all project modules unconditionally.

Now SECTIONS maps names to (module_path, function_name) strings. The actual
import happens in resolve_section(), called only when a section is about to
run. Broken or missing dependencies in one section no longer prevent other
sections from loading.

Fix (HW_SECTIONS): sensitivity_coordinator_all removed. That function is
dry-run only and ignores arduino_port/only_hw. Accepting --only-hw for it
would silently run the full network sweep instead of hardware tests.
"""
from __future__ import annotations
from typing import Callable

# Registry: section_name → (dotted_module_path, function_name)
# No heavy imports happen here.
_REGISTRY: dict[str, tuple[str, str]] = {
    "profile_builder":             ("hiltest.sections.profile_builder",
                                    "run_profile_builder_tests"),
    "network_plotter":             ("hiltest.sections.plotter",
                                    "run_network_plotter_tests"),
    "violation_detector":          ("hiltest.sections.violation",
                                    "run_violation_detector_tests"),
    "violation_detector_all":      ("hiltest.sections.violation",
                                    "run_violation_detector_all_tests"),
    "volt_var_control":            ("hiltest.sections.volt_var",
                                    "run_volt_var_tests"),
    "sensitivity_coordinator":     ("hiltest.sections.coordinator",
                                    "run_sensitivity_coordinator_tests"),
    "sensitivity_coordinator_all": ("hiltest.sections.coordinator",
                                    "run_sensitivity_coordinator_all_tests"),
    # "baseline":          ("hiltest.sections.scenarios", "run_baseline_tests"),
    # "oltc":              ("hiltest.sections.scenarios", "run_oltc_tests"),
    # "svc":               ("hiltest.sections.scenarios", "run_svc_tests"),
    # "hil":               ("hiltest.sections.scenarios", "run_hil_tests"),
    # "opf":               ("hiltest.sections.scenarios", "run_opf_tests"),
    # "hosting_capacity":  ("hiltest.sections.scenarios", "run_hosting_capacity_tests"),
}

# Public names list — used by argparse choices without importing anything heavy
SECTION_NAMES: list[str] = list(_REGISTRY.keys())

# Sections that accept arduino_port and only_hw kwargs.
# sensitivity_coordinator_all is NOT here — it is dry-run only.
HW_SECTIONS: set[str] = {"volt_var_control", "sensitivity_coordinator"}


def resolve_section(name: str) -> Callable:
    """
    Import and return the run function for the given section name.
    Called at run time, not at import time.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown section: {name!r}. "
                       f"Available: {', '.join(SECTION_NAMES)}")
    module_path, fn_name = _REGISTRY[name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)
