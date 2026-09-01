"""
hiltest/cli.py
===============
Argument parsing and section routing.

Uses SECTION_NAMES and HW_SECTIONS from the lazy registry so that --help
never imports pandapower or any project module.
"""
from __future__ import annotations

import argparse

from hiltest.sections import SECTION_NAMES, HW_SECTIONS, resolve_section


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HIL Testbed master test suite",
        prog="python -m hiltest",
    )
    parser.add_argument(
        "--section",
        choices=SECTION_NAMES,
        default=None,
        help="Run a single section only (default: all sections)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full tracebacks for all failures",
    )
    parser.add_argument(
        "--arduino-port",
        default=None,
        help="Serial port for hardware tests e.g. /dev/ttyACM0. "
             "If omitted, hardware subsections are skipped automatically.",
    )
    parser.add_argument(
        "--only-hw",
        action="store_true",
        help=(
            "Run hardware subsection only (skip dry-run). "
            "Requires --arduino-port and --section to name a hardware-capable "
            "section: " + ", ".join(sorted(HW_SECTIONS))
        ),
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only test cases whose names contain any of these substrings. "
             "e.g. --only cigre kerber",
    )
    return parser


def build_to_run(args: argparse.Namespace) -> dict[str, str]:
    """
    Return {section_name: section_name} for sections to execute.
    Callers resolve the actual function via resolve_section(name).

    Fix (Blocker 1 — --only-hw honours --section):
    Previously hardcoded to volt_var_control regardless of --section.
    Now --only-hw requires --section to name a HW_SECTION explicitly,
    and rejects sensitivity_coordinator_all (dry-run only, no hw block).
    """
    if args.only_hw:
        if not args.arduino_port:
            raise SystemExit("--only-hw requires --arduino-port")
        if not args.section:
            raise SystemExit(
                "--only-hw requires --section naming a hardware-capable section: "
                + ", ".join(sorted(HW_SECTIONS))
            )
        if args.section not in HW_SECTIONS:
            raise SystemExit(
                f"--only-hw: '{args.section}' has no hardware subsection. "
                f"Hardware-capable sections: {', '.join(sorted(HW_SECTIONS))}"
            )
        return {args.section: args.section}

    if args.section:
        return {args.section: args.section}

    return {name: name for name in SECTION_NAMES}


def build_kwargs(section_name: str, args: argparse.Namespace) -> dict:
    """Return kwargs for the section run function."""
    kwargs: dict = {"verbose": args.verbose, "only": args.only}
    if section_name in HW_SECTIONS:
        kwargs["arduino_port"] = args.arduino_port
        kwargs["only_hw"]      = args.only_hw
    return kwargs
