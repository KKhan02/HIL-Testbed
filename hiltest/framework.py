"""
hiltest/framework.py
====================
Core test primitives shared by all sections.

Changes from original test_suite.py
-------------------------------------
- TestCase.passed: added `and len(self.checks) > 0` (blocker 5 — zero-check
  tests previously returned True due to all([]) == True).
- print_summary: now returns a bool (any_failure) so __main__ can call
  sys.exit(1) when the run contains failures (CI silent-exit blocker).
- Timing uses time.perf_counter() throughout for monotonic wall clock.
"""

import traceback
import time


class TestCase:
    """Represents a single named test with pass/fail/error state."""

    def __init__(self, name: str):
        self.name     = name
        self.checks   = []    # list of (check_name, passed, detail)
        self.error    = None  # full traceback string if test crashed
        self.duration = 0.0
        self.skipped  = False

    def record(self, check_name: str, condition: bool, detail: str = ""):
        self.checks.append((check_name, bool(condition), detail))

    @property
    def passed(self) -> bool:
        # FIX (blocker 5): require at least one check — all([]) is True, which
        # silently passes empty TestCases from early-exit or placeholder paths.
        return (
            not self.error
            and not self.skipped
            and len(self.checks) > 0
            and all(ok for _, ok, _ in self.checks)
        )

    @property
    def n_passed(self) -> int:
        return sum(1 for _, ok, _ in self.checks if ok)

    @property
    def n_total(self) -> int:
        return len(self.checks)


def print_case(tc: TestCase, verbose: bool = False) -> None:
    if tc.skipped:
        print(f"  SKIP  {tc.name:<60}")
        return

    status = "PASS" if tc.passed else "FAIL"
    print(
        f"  {status}  {tc.name:<60}  "
        f"({tc.n_passed}/{tc.n_total})  [{tc.duration:.1f}s]"
    )
    if not tc.passed:
        if tc.error:
            last_line = tc.error.strip().splitlines()[-1]
            print(f"         ERROR: {last_line}")
        for name, ok, detail in tc.checks:
            if not ok:
                print(f"         FAIL check '{name}': {detail}")
        if verbose and tc.error:
            print(tc.error)


def print_summary(section_results: dict) -> bool:
    """
    Print the grand summary table.

    Returns True if any test case failed (so __main__ can sys.exit(1)).
    Skipped cases are counted separately and not treated as failures.
    """
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    grand_pass = grand_fail = grand_skip = grand_total = 0

    for section, cases in section_results.items():
        n_pass  = sum(1 for tc in cases if tc.passed)
        n_skip  = sum(1 for tc in cases if tc.skipped)
        n_fail  = sum(1 for tc in cases if not tc.passed and not tc.skipped)
        n_total = len(cases)

        grand_pass  += n_pass
        grand_fail  += n_fail
        grand_skip  += n_skip
        grand_total += n_total

        status = "PASS" if n_fail == 0 else "FAIL"
        skip_note = f"  ({n_skip} skipped)" if n_skip else ""
        print(f"  {status}  {section:<35}  {n_pass}/{n_total} cases{skip_note}")

    print("-" * 70)
    any_failure = grand_fail > 0
    grand_status = "PASS" if not any_failure else "FAIL"
    skip_note = f"  ({grand_skip} skipped)" if grand_skip else ""
    print(
        f"  {grand_status}  {'TOTAL':<35}  "
        f"{grand_pass}/{grand_total} cases{skip_note}"
    )
    print("=" * 70)
    return any_failure
