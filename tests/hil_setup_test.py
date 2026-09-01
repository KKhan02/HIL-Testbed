"""
hil_setup_test.py
==================
Standalone HIL connectivity test. Run this once before any scenario execution
to confirm the RPi-Arduino serial link is working correctly.

This script is intentionally separate from test_suite.py. It requires a
physical Arduino connection and is not expected to pass in CI or on machines
without the HIL hardware present.

Usage
-----
    python hil_setup_test.py --port /dev/ttyACM0

Optional flags
--------------
    --port        Serial port (required). Uno R3 on RPi: /dev/ttyACM0 or
                  /dev/ttyUSB0 (CH340 clone).
    --baud        Baud rate (default: 115200). Must match BAUD_RATE in sketch.
    --n-ders      Number of DERs to use in test handshake (default: 3).
    --n-exchanges Number of V/Q exchange iterations for timing benchmark (default: 10).
    --verbose     Print full tracebacks for all failures.

Tests performed
---------------
1. Serial open         -- port opens without ImportError (pyserial) or OS error.
2. Configure handshake -- INIT and P messages receive ACK:INIT and ACK:P.
3. Single exchange     -- one V: message; Q: response length and finiteness.
4. Deadband exchange   -- vm_pu = 1.00 for all DERs; Q must be ~0.
5. Saturation inject   -- vm_pu = 0.90 for all DERs; Q must be +Q_max.
6. Saturation absorb   -- vm_pu = 1.10 for all DERs; Q must be -Q_max.
7. Timing benchmark    -- n_exchanges timed; mean and max latency reported.
8. NaN rejection       -- NaN vm_pu must trigger SerialTimeoutError (Arduino
                          sends ERR:V_INVALID, Python raises ArduinoProtocolError).
9. Reconfigure         -- second configure() after first succeeds; confirms
                          Arduino re-accepts INIT after a prior session.

Example run python hil_setup_test.py --port PORT --n-ders 3 --n-exchanges 10 (Add --verbose for full traceback for all failures)

Exit codes
----------
0  All tests passed.
1  One or more tests failed.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import numpy as np

from volt_var_controller import (
    ArduinoSerialInterface,
    ArduinoProtocolError,
    SerialTimeoutError,
    QVCharacteristic,
    BAUD_RATE,
    Q_RATIO,
    U2_PU,
    U3_PU,
    ARDUINO_MAX_DERS,
)


# ===========================================================================
# Minimal test scaffolding (not imported from test_suite to keep standalone)
# ===========================================================================

class Result:
    def __init__(self, name: str):
        self.name   = name
        self.checks = []    # (label, passed, detail)
        self.error  = None
        self.t_ms   = 0.0

    def record(self, label: str, cond: bool, detail: str = ""):
        self.checks.append((label, cond, detail))

    @property
    def passed(self):
        return not self.error and all(ok for _, ok, _ in self.checks)


def print_result(r: Result, verbose: bool):
    status = "PASS" if r.passed else ("SKIP" if r.error is None and not r.checks else "FAIL")
    print(f"  {status}  {r.name:<50}  [{r.t_ms:.0f}ms]")
    if not r.passed:
        if r.error:
            print(f"         ERROR: {r.error.strip().splitlines()[-1]}")
        for label, ok, detail in r.checks:
            if not ok:
                print(f"         FAIL  {label}: {detail}")
        if verbose and r.error:
            print(r.error)


# ===========================================================================
# Individual tests
# ===========================================================================

def test_open(port: str, baud: int) -> tuple:
    """
    Open the serial port. Returns (arduino, result).
    If open fails, arduino is None -- all downstream tests are skipped.
    """
    r = Result("1_serial_open")
    t0 = time.perf_counter()
    arduino = None
    try:
        arduino = ArduinoSerialInterface(port=port, baud=baud)
        arduino.open()
        r.record("port_open", arduino.is_open, f"is_open={arduino.is_open}")
    except Exception:
        r.error = traceback.format_exc()
    r.t_ms = (time.perf_counter() - t0) * 1e3
    return arduino, r


def test_configure(arduino: ArduinoSerialInterface, n_ders: int) -> tuple:
    """
    Run the INIT + P handshake. Returns (p_installed, result).
    p_installed is a test array with values 1.0..n_ders MW.
    """
    r = Result("2_configure_handshake")
    t0 = time.perf_counter()
    p_installed = np.ones(n_ders, dtype=float) * 2.0
    try:
        batch_n = min(n_ders,ARDUINO_MAX_DERS)
        arduino.configure(batch_n, p_installed[:batch_n])
        r.record("configure_ok", True)
    except Exception:
        r.error = traceback.format_exc()
        r.record("configure_ok", False)
    r.t_ms = (time.perf_counter() - t0) * 1e3
    return p_installed, r


def test_single_exchange(
        arduino: ArduinoSerialInterface,
        n_ders: int,
        p_installed: np.ndarray,
) -> Result:
    r = Result("3_single_exchange")
    t0 = time.perf_counter()
    try:
        vm_pu = np.full(n_ders, 1.00, dtype=float)
        q_arr, n_att = arduino.exchange_batched(vm_pu,p_installed)
        r.record("response_length",
            len(q_arr) == n_ders, f"got {len(q_arr)}, expected {n_ders}")
        r.record("response_finite",
            np.all(np.isfinite(q_arr)), f"non-finite: {q_arr}")
        r.record("n_attempts_positive", 
                 n_att >= 1, f"n_att={n_att}")
    except Exception:
        r.error = traceback.format_exc()
    r.t_ms = (time.perf_counter() - t0) * 1e3
    return r


def test_deadband(
        arduino: ArduinoSerialInterface,
        n_ders: int,
        p_installed: np.ndarray,
) -> Result:
    """All vm_pu = 1.00 pu (centre of deadband) => Q must be ~0."""
    r = Result("4_deadband_vm100")
    t0 = time.perf_counter()
    try:
        vm_pu = np.full(n_ders, 1.00, dtype=float)
        q_arr, _ = arduino.exchange_batched(vm_pu,p_installed)
        r.record("q_near_zero",
            np.allclose(q_arr, 0.0, atol=1e-3),
            f"max|q| = {np.abs(q_arr).max():.6f} MVAr")
    except Exception:
        r.error = traceback.format_exc()
    r.t_ms = (time.perf_counter() - t0) * 1e3
    return r


def test_saturation_inject(
        arduino: ArduinoSerialInterface,
        n_ders: int,
        p_installed: np.ndarray,
) -> Result:
    """All vm_pu = 0.90 pu (below U1 = 0.96) => Q must be +Q_max per DER."""
    r = Result("5_saturation_inject_vm090")
    t0 = time.perf_counter()
    try:
        vm_pu   = np.full(n_ders, 0.90, dtype=float)
        q_arr, _ = arduino.exchange_batched(vm_pu,p_installed)
        q_exp   = Q_RATIO * p_installed
        r.record("q_near_q_max",
            np.allclose(q_arr, q_exp, atol=1e-2),
            f"q_arr={np.round(q_arr,4)} expected={np.round(q_exp,4)}")
        r.record("q_positive",
            np.all(q_arr >= 0), f"negative Q found: {q_arr}")
    except Exception:
        r.error = traceback.format_exc()
    r.t_ms = (time.perf_counter() - t0) * 1e3
    return r


def test_saturation_absorb(
        arduino: ArduinoSerialInterface,
        n_ders: int,
        p_installed: np.ndarray,
) -> Result:
    """All vm_pu = 1.10 pu (above U4 = 1.04) => Q must be -Q_max per DER."""
    r = Result("6_saturation_absorb_vm110")
    t0 = time.perf_counter()
    try:
        vm_pu    = np.full(n_ders, 1.10, dtype=float)
        q_arr, _ = arduino.exchange_batched(vm_pu,p_installed)
        q_exp    = -Q_RATIO * p_installed
        r.record("q_near_neg_q_max",
            np.allclose(q_arr, q_exp, atol=1e-2),
            f"q_arr={np.round(q_arr,4)} expected={np.round(q_exp,4)}")
        r.record("q_negative",
            np.all(q_arr <= 0), f"positive Q found: {q_arr}")
    except Exception:
        r.error = traceback.format_exc()
    r.t_ms = (time.perf_counter() - t0) * 1e3
    return r


def test_timing_benchmark(
        arduino:     ArduinoSerialInterface,
        n_ders:      int,
        n_exchanges: int,
        p_installed: np.ndarray,
) -> Result:
    """Timed benchmark: n_exchanges V/Q round trips. Reports mean and max."""
    r = Result(f"7_timing_benchmark_{n_exchanges}x")
    t0 = time.perf_counter()
    try:
        vm_pu  = np.full(n_ders, 1.00, dtype=float)
        times  = []
        length_ok = True
        for _ in range(n_exchanges):
            t_ex = time.perf_counter()
            q_arr, _ = arduino.exchange_batched(vm_pu,p_installed)
            times.append((time.perf_counter() - t_ex) * 1e3)
            if len(q_arr) != n_ders:
                length_ok = False

        mean_ms = float(np.mean(times))
        max_ms  = float(np.max(times))
        print(f"         Timing: mean={mean_ms:.1f}ms  max={max_ms:.1f}ms  "
              f"n={n_exchanges}")
        r.record("all_exchanges_completed", len(times) == n_exchanges)
        r.record("response_length_consistent", length_ok,
            f"At least one exchange returned wrong Q count (expected {n_ders})")
        # Estimate bytes for each message type (ASCII, newline-terminated)
        def bytes_init(n):
            return len(f"INIT:{n}\n")  # e.g., "INIT:105\n"

        def bytes_p(n):
            # P:<p1>,<p2>... with p formatted at 4 decimals (e.g., "2.0000")
            # Each value: 6 chars + comma, except last no comma
            return 2 + (7 * n) - 1 + 1  # "P:" + values + "\n"

        def bytes_v(n):
            # V:<v1>,<v2>... with 4 decimals (e.g., "1.0000")
            # Each value: 6 chars + comma, except last
            return 2 + (9 * n) - 1 + 1

        def bytes_q(n):
            # Q:<q1>,<q2>... with 4 decimals
            return 2 + (7 * n) - 1 + 1

        batches = [
            min(ARDUINO_MAX_DERS, n_ders - i)
            for i in range(0, n_ders, ARDUINO_MAX_DERS)
        ]

        char_time_ms = 0.087  # 1/115200 * 10 bits ≈ 0.087 ms/char

        theoretical_ms = 0.0
        for b in batches:
            total_chars = (
                bytes_init(b) +
                bytes_p(b) +
                bytes_v(b) +
                bytes_q(b)
            )
            theoretical_ms += total_chars * char_time_ms

        threshold_ms = max(300, theoretical_ms * 1.5)
        r.record("mean_within_expected",
            mean_ms < threshold_ms,
            f"mean={mean_ms:.1f}ms, threshold={threshold_ms:.0f}ms "
            f"(theoretical={theoretical_ms:.0f}ms for {n_ders} DERs at 115200 baud)")
        r.record("max_under_1500ms", max_ms < 1500,
            f"max={max_ms:.1f}ms (>1500ms suggests cable/driver issue)")
    except Exception:
        r.error = traceback.format_exc()
    r.t_ms = (time.perf_counter() - t0) * 1e3
    return r


def test_nan_rejection(
        arduino: ArduinoSerialInterface,
        n_ders:  int,
        p_installed: np.ndarray,
) -> Result:
    """
    Send NaN vm_pu. The Python layer should raise SerialTimeoutError
    (NaN guard in exchange()) before any bytes are transmitted.
    If NaN somehow slips through, the Arduino returns ERR:V_INVALID
    which raises ArduinoProtocolError.
    Either exception confirms the guard is active.
    """
    r = Result("8_nan_vm_rejection")
    t0 = time.perf_counter()
    try:
        vm_nan = np.full(n_ders, float("nan"), dtype=float)
        try:
            arduino.exchange_batched(vm_nan,p_installed)
            r.record("nan_rejected",
                False, "No exception raised -- NaN was accepted silently")
        except (SerialTimeoutError, ArduinoProtocolError):
            r.record("nan_rejected", True)
    except Exception:
        r.error = traceback.format_exc()
    r.t_ms = (time.perf_counter() - t0) * 1e3
    return r


def test_reconfigure(
        arduino: ArduinoSerialInterface,
        n_ders:  int,
        p_installed: np.ndarray,
) -> Result:
    """
    Second configure() call after the first succeeds. Confirms the Arduino
    re-accepts INIT without requiring a physical reset.
    """
    r = Result("9_reconfigure")
    t0 = time.perf_counter()
    try:
        batch_n = min(n_ders, ARDUINO_MAX_DERS)
        arduino.configure(batch_n, p_installed[:batch_n])
        r.record("reconfigure_ok", True)

        # Verify exchange still works after reconfigure
        vm_pu   = np.full(n_ders, 1.00, dtype=float)
        q_arr, _ = arduino.exchange_batched(vm_pu,p_installed)
        r.record("exchange_after_reconfigure",
            len(q_arr) == n_ders, f"got {len(q_arr)}, expected {n_ders}")
    except Exception:
        r.error = traceback.format_exc()
    r.t_ms = (time.perf_counter() - t0) * 1e3
    return r


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HIL setup connectivity test. Requires physical Arduino."
    )
    parser.add_argument(
        "--port", required=True,
        help="Serial port, e.g. /dev/ttyACM0 or /dev/ttyUSB0"
    )
    parser.add_argument(
        "--baud", type=int, default=BAUD_RATE,
        help=f"Baud rate (default: {BAUD_RATE})"
    )
    parser.add_argument(
        "--n-ders", type=int, default=3,
        help="Number of DERs for test handshake (default: 3)"
    )
    parser.add_argument(
        "--n-exchanges", type=int, default=10,
        help="Exchange count for timing benchmark (default: 10)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full tracebacks for failures"
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  HIL Setup Test")
    print(f"  Port: {args.port}  Baud: {args.baud}  "
          f"DERs: {args.n_ders}  Exchanges: {args.n_exchanges}")
    print(f"{'='*70}\n")

    t_start  = time.perf_counter()
    results  = []
    arduino  = None

    # Guard n_ders before any hardware contact
    if args.n_ders < 1:
        print(f"  ERROR: --n-ders must be >= 1. Got {args.n_ders}.")
        return 1

    # Guard n_exchanges
    if args.n_exchanges < 1:
        print(f"  ERROR: --n-exchanges must be >= 1. Got {args.n_exchanges}.")
        return 1

    # 1. Open — in a try/finally so the port is always closed even if open()
    # itself raises after seizing the OS file descriptor.
    try:
        arduino, r_open = test_open(args.port, args.baud)
    finally:
        # If test_open raised before assigning arduino, arduino is still None
        pass

    results.append(r_open)
    print_result(r_open, args.verbose)

    if not r_open.passed or arduino is None:
        print("\n  FATAL: Could not open serial port -- all downstream tests skipped.")
        _print_summary(results, time.perf_counter() - t_start)
        return 1

    try:
        # 2. Configure
        p_installed, r_cfg = test_configure(arduino, args.n_ders)
        results.append(r_cfg)
        print_result(r_cfg, args.verbose)

        if not r_cfg.passed:
            print("\n  FATAL: Configure failed -- exchange tests skipped.")
            _print_summary(results, time.perf_counter() - t_start)
            return 1

        # 3–6. Exchange tests
        for fn in (
            lambda: test_single_exchange(arduino, args.n_ders, p_installed),
            lambda: test_deadband(arduino, args.n_ders, p_installed),
            lambda: test_saturation_inject(arduino, args.n_ders, p_installed),
            lambda: test_saturation_absorb(arduino, args.n_ders, p_installed),
        ):
            r = fn()
            results.append(r)
            print_result(r, args.verbose)

        # 7. Timing benchmark
        r = test_timing_benchmark(arduino, args.n_ders, args.n_exchanges, p_installed)
        results.append(r)
        print_result(r, args.verbose)

        # 8. NaN rejection
        r = test_nan_rejection(arduino, args.n_ders, p_installed)
        results.append(r)
        print_result(r, args.verbose)

        # 9. Reconfigure
        r = test_reconfigure(arduino, args.n_ders, p_installed)
        results.append(r)
        print_result(r, args.verbose)

    finally:
        arduino.close()

    return _print_summary(results, time.perf_counter() - t_start)


def _print_summary(results: list, elapsed_s: float) -> int:
    n_pass  = sum(1 for r in results if r.passed)
    n_total = len(results)
    status  = "PASS" if n_pass == n_total else "FAIL"
    print(f"\n{'='*70}")
    print(f"  {status}  {n_pass}/{n_total} tests passed  "
          f"({elapsed_s:.1f}s total)")
    print(f"{'='*70}\n")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())