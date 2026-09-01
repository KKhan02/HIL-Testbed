"""
hiltest/__main__.py
====================
Entry point: python -m hiltest [options]

Sections are resolved lazily via resolve_section() so that --help and
--section profile_builder never import coordinator or volt_var modules.
"""
import sys
import time

from hiltest.cli      import build_parser, build_to_run, build_kwargs
from hiltest.sections import resolve_section
from hiltest.framework import print_summary


def main() -> None:
    parser  = build_parser()
    args    = parser.parse_args()
    to_run  = build_to_run(args)
    t_start = time.perf_counter()

    section_results: dict = {}

    for section_name in to_run:
        print(f"\n{'='*70}")
        print(f"  SECTION: {section_name.upper()}")
        print(f"{'='*70}")

        run_fn = resolve_section(section_name)   # import happens here only
        kwargs = build_kwargs(section_name, args)
        cases  = run_fn(**kwargs)
        section_results[section_name] = cases

    any_failure = print_summary(section_results)
    print(f"\n  Total time: {time.perf_counter() - t_start:.1f}s")

    if any_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
