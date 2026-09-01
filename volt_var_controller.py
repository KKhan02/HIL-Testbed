"""
volt_var_controller.py
=======================
Phase 1 Item 2 — Q(V) Volt-Var control module.

Implements the VDE-AR-N 4110 Bild 8 Q(U) piecewise linear characteristic
and the HIL coordination layer that bridges pandapower and the Arduino.

Architecture
------------
Two co-designed components share the same characteristic constants.
Constants defined here are the authoritative reference and are transmitted
to the firmware at session startup via the CFG: protocol message, so only
this file needs editing when the parameter VALUES change. The curve SHAPE
(piecewise linear with deadband) is still implemented independently on both
sides -- structural changes to the characteristic must be made in both files.

    RPi (this module)                    Arduino (volt_var_arduino.ino)
    ─────────────────────────────────    ────────────────────────────────
    1. pp.runpp() → res_bus.vm_pu    ──► receives "V:" message
    2. QVCharacteristic  (dry-run)        runs compute_q() per DER
    3. ArduinoSerialInterface        ◄──  returns "Q:" message
    4. apply q_mvar to net.sgen
    5. second pp.runpp()
    6. detect_violations() × 2

Q(U) Characteristic — VDE-AR-N 4110 Bild 8
--------------------------------------------
Piecewise linear; clamped outside [U1_PU, U4_PU]:

    U <= 0.96 pu   =>  Q = +Q_max          (inject, raise voltage)
    0.96-0.99 pu   =>  linear +Q_max -> 0
    0.99-1.01 pu   =>  Q = 0               (deadband)
    1.01-1.04 pu   =>  linear 0 -> -Q_max
    U >= 1.04 pu   =>  Q = -Q_max          (absorb, lower voltage)

    Q_max = Q_RATIO x P_b_inst   (rated/installed capacity in MW)

Sign convention follows pandapower:
    q_mvar > 0  =>  inject reactive power  (capacitive, raises voltage)
    q_mvar < 0  =>  absorb reactive power  (inductive, lowers voltage)

Serial Protocol
---------------
Baud rate: 115200. All messages are ASCII, newline-terminated.
Framing: prefix token identifies message type; values comma-separated.

Startup handshake (once per session, via configure()):

    RPi  ->  Arduino :  "INIT:<n>\\n"          n = number of DERs
    Arduino  ->  RPi :  "ACK:INIT\\n"
    RPi  ->  Arduino :  "CFG:<q_ratio>,<u1>,<u2>,<u3>,<u4>\\n"
                                               Q(V) characteristic parameters
    Arduino  ->  RPi :  "ACK:CFG\\n"
    RPi  ->  Arduino :  "P:<p1>,<p2>,...\\n"   p_installed_mw per DER (MW)
    Arduino  ->  RPi :  "ACK:P\\n"

CFG carries Q_RATIO, U1_PU, U2_PU, U3_PU, U4_PU from THIS module (the single
source of truth) to the firmware at session startup, replacing the old
requirement that the numeric constants be manually mirrored in
volt_var_arduino.ino. Firmware compile-time constants are boot defaults only;
the firmware rejects P: with ERR:CFG_BEFORE_P until a valid CFG is received.

Per timestep (via exchange()):

    RPi  ->  Arduino :  "V:<v1>,<v2>,...\\n"   vm_pu per DER
    Arduino  ->  RPi :  "Q:<q1>,<q2>,...\\n"   q_mvar per DER

Error responses from Arduino:  "ERR:<reason>\\n"

DER ordering: sgen_indices sorted ascending (default). Position-based;
ordering must be consistent across INIT, P, and V messages.

Dependencies
------------
    RPi   :  pyserial  (pip install pyserial)
    Arduino: standard AVR libraries only

Usage
-----
Scenario 4 HIL loop::

    import pandapower as pp
    import simbench as sb
    from volt_var_controller import VoltVarController, ArduinoSerialInterface

    net = sb.get_simbench_net("1-MV-rural--2-sw")

    with ArduinoSerialInterface(port="/dev/ttyACM0") as arduino:
        ctrl = VoltVarController(net, arduino)
        ctrl.configure()

        for t in time_steps:
            net.sgen.p_mw   = p_profiles.iloc[t]
            net.load.p_mw   = p_load.iloc[t]
            net.load.q_mvar = q_load.iloc[t]
            result = ctrl.run_timestep()

Dry-run (no Arduino -- unit tests and offline analysis)::

    ctrl = VoltVarController(net, interface=None, dry_run=True)
    ctrl.configure()
    result = ctrl.run_timestep()
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pandapower as pp

from violation_detector import ViolationReport, detect_violations

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"simbench\.converter\.csv_pp_converter",
)


# ===========================================================================
# Characteristic constants -- VDE-AR-N 4110 Bild 8
# SINGLE SOURCE OF TRUTH: these values are pushed to the Arduino at session
# startup via the CFG: message (ArduinoSerialInterface.configure()). The
# constants in volt_var_arduino.ino are boot defaults only. Edit here only.
# ===========================================================================

U1_PU   = 0.96    # lower saturation  -- full Q injection at or below this
U2_PU   = 0.99    # deadband lower edge
U3_PU   = 1.01    # deadband upper edge
U4_PU   = 1.04    # upper saturation  -- full Q absorption at or above this
Q_RATIO = 0.25    # Q_max / P_b_inst  (VDE-AR-N 4110 Bild 8). Modifying to 0.15 from 0.48 to check control response
PLAUSIBLE_PU_RANGE = (0.5, 2.5)


# Compile-time defaults, kept for reset_qv_parameters() and provenance.
_QV_DEFAULTS = {"q_ratio": Q_RATIO, "u1": U1_PU, "u2": U2_PU,
                "u3": U3_PU, "u4": U4_PU}


def set_qv_parameters(
        q_ratio: float = None,
        u1:      float = None,
        u2:      float = None,
        u3:      float = None,
        u4:      float = None,
) -> dict:
    """
    Runtime override of the Q(V) characteristic parameters.

    Rebinds the module-level constants (Q_RATIO, U1_PU..U4_PU) that every
    consumer reads at call time: QVCharacteristic (dry-run curve),
    ArduinoSerialInterface.configure()'s CFG: message (hardware curve), and
    — via their call-time module-attribute reads — the coordinator q_max
    sizing and scenario 4's saturation ceiling. This preserves the single-
    source-of-truth invariant (Knowledge Base §9.1): one call changes the
    dry-run curve, the coordination sizing, AND the value pushed to the
    Arduino over CFG, so a Python/firmware mismatch cannot occur.

    Call BEFORE run_benchmark() / run_scenario_4() — the values are read
    when the controller and coordinator are constructed and when CFG is
    sent at configure() time.

    Parameters left as None keep their current value. Validation checks
    two things: (1) mirrors the firmware's ERR:CFG_INVALID matrix exactly
    — all values finite, 0 < q_ratio <= 1, and strictly increasing
    u1 < u2 < u3 < u4 (keeps both ramp denominators positive, the same
    division-by-zero guard the AVR side enforces); (2) a plausibility
    check NOT present in the firmware — u1..u4 must fall within
    [0.5, 2.5] pu, since strict monotonicity alone does not catch a
    breakpoint that is technically increasing but physically nonsensical
    (e.g. u4=6.0 typed instead of u4=1.06).

    Returns
    -------
    dict  {"q_ratio", "u1", "u2", "u3", "u4"} — the values now in effect.

    Raises
    ------
    ValueError  on any value the firmware would reject with ERR:CFG_INVALID.
    """
    global Q_RATIO, U1_PU, U2_PU, U3_PU, U4_PU

    new_q  = float(q_ratio) if q_ratio is not None else Q_RATIO
    new_u1 = float(u1)      if u1      is not None else U1_PU
    new_u2 = float(u2)      if u2      is not None else U2_PU
    new_u3 = float(u3)      if u3      is not None else U3_PU
    new_u4 = float(u4)      if u4      is not None else U4_PU

    vals = (new_q, new_u1, new_u2, new_u3, new_u4)
    if not all(np.isfinite(v) for v in vals):
        raise ValueError(
            f"Q(V) parameters must all be finite, got "
            f"q_ratio={new_q}, u1..u4={vals[1:]} (firmware ERR:CFG_INVALID)."
        )
    if not (0.0 < new_q <= 1.0):
        raise ValueError(
            f"q_ratio must satisfy 0 < q_ratio <= 1, got {new_q} "
            f"(firmware ERR:CFG_INVALID)."
        )
    if not (new_u1 < new_u2 < new_u3 < new_u4):
        raise ValueError(
            f"Voltage breakpoints must be strictly increasing "
            f"u1 < u2 < u3 < u4, got {new_u1}, {new_u2}, {new_u3}, {new_u4} "
            f"(firmware ERR:CFG_INVALID)."
        )

    # Plausibility range check — separate from the monotonicity check above.
    # u1<u2<u3<u4 alone does not catch a value that is technically strictly
    # increasing but physically nonsensical for a per-unit voltage breakpoint
    # (e.g. a typo like 6.0 instead of 1.06 still satisfies u3 < u4). Real
    # MV/LV networks operate within roughly +-20% of nominal even under
    # significant stress, so anything outside [0.5, 2.5] pu is almost
    # certainly a data-entry error, not an intended operating point.
    breakpoint_names = ("u1", "u2", "u3", "u4")
    for name, v in zip(breakpoint_names, (new_u1, new_u2, new_u3, new_u4)):
        if not (PLAUSIBLE_PU_RANGE[0] <= v <= PLAUSIBLE_PU_RANGE[1]):
            raise ValueError(
                f"{name}={v} is outside the plausible per-unit voltage "
                f"range {PLAUSIBLE_PU_RANGE} — this is very likely a typo "
                f"(e.g. 6.0 instead of 1.06). If this value is genuinely "
                f"intended, adjust PLAUSIBLE_PU_RANGE in "
                f"volt_var_controller.py's set_qv_parameters()."
            )
    
    # Parse-then-commit, mirroring the firmware: nothing is written until
    # every check has passed, so a bad call can never half-update the curve.
    Q_RATIO, U1_PU, U2_PU, U3_PU, U4_PU = vals
    return {"q_ratio": Q_RATIO, "u1": U1_PU, "u2": U2_PU,
            "u3": U3_PU, "u4": U4_PU}


def reset_qv_parameters() -> dict:
    """Restore the compile-time default Q(V) parameters (see _QV_DEFAULTS)."""
    return set_qv_parameters(**_QV_DEFAULTS)


# ===========================================================================
# Serial protocol constants
# ===========================================================================

BAUD_RATE              = 115200
ARDUINO_MAX_DERS       = 105    # hard ceiling from ATmega328P SRAM budget
DEFAULT_TIMEOUT_S      = 1.0    # per-read timeout for exchange()
ACK_TIMEOUT_S          = 5.0    # longer timeout for configure() ACK responses
DEFAULT_MAX_RETRIES    = 3      # exchange() retry limit before fallback to Q=0
ARDUINO_RESET_DELAY_S  = 3.0    # Uno R3 resets on USB serial open -- mandatory wait


# ===========================================================================
# Custom exceptions
# ===========================================================================

class SerialTimeoutError(RuntimeError):
    """Arduino did not respond within the timeout window, or readline() returned
    a truncated message (no terminating newline byte received)."""
    pass

class ArduinoProtocolError(RuntimeError):
    """Arduino returned an explicit ERR: response. Distinct from a timeout --
    the Arduino is alive but rejected the message (wrong DER count, invalid
    voltage value, buffer overflow, etc.). The reason string from the Arduino
    is preserved in the exception message for diagnostics."""
    pass


class SerialConfigError(RuntimeError):
    """Startup handshake failed. Either the port is not open, or the Arduino
    returned an unexpected response to INIT or P messages."""
    pass


# ===========================================================================
# QVCharacteristic -- pure Q(U) logic, hardware-independent
# ===========================================================================

class QVCharacteristic:
    """
    VDE-AR-N 4110 Bild 8 Q(U) piecewise linear characteristic.

    Stateless static methods only. No hardware or network dependency.
    Mirrors compute_q() in volt_var_arduino.ino exactly -- any change here
    must be propagated to the Arduino sketch and vice versa.

    Used by VoltVarController in dry_run mode and directly by the test suite.

    Sign convention (pandapower):
        q_mvar > 0  =>  inject Q  (capacitive, raises voltage)
        q_mvar < 0  =>  absorb Q  (inductive, lowers voltage)
    """

    @staticmethod
    def compute_setpoint(vm_pu: float, p_installed_mw: float) -> float:
        """
        Q setpoint for a single DER at bus voltage vm_pu.

        Parameters
        ----------
        vm_pu           : Bus voltage magnitude in pu.
        p_installed_mw  : Rated/installed active power capacity in MW.
                          Must be finite; NaN or inf => returns 0.0 with warning.
                          Negative values: abs() taken (sign has no physical meaning
                          for installed capacity; guard is defensive only).
                          Pass 0 for non-controllable DERs (returns Q = 0).

        Returns
        -------
        float  q_mvar. Positive = injection. Negative = absorption.
        """
        # Guard 1: non-finite installed capacity
        # Returns false for NaN, +Inf and -Inf
        if not np.isfinite(p_installed_mw):
            warnings.warn(
                f"[QVCharacteristic] p_installed_mw={p_installed_mw!r} is not "
                f"finite; returning Q=0.",
                RuntimeWarning, stacklevel=2,
            )
            return 0.0
        # Guard 2: non-finite voltage
        if not np.isfinite(vm_pu):
            warnings.warn(
                f"[QVCharacteristic] vm_pu={vm_pu!r} is not finite; returning Q=0.",
                RuntimeWarning, stacklevel=2,
            )
            return 0.0
        # abs incase somehow a value of p_installed_mw is -ve (due to faulty dataset)
        q_max = Q_RATIO * abs(p_installed_mw)

        # Inclusive boundary at saturation points avoids unnecessary
        # division-by-zero risk at exact floating-point boundary values.
        # Both branches yield the same result at the boundaries (the function
        # is continuous), so the choice only affects which branch executes.
        if vm_pu <= U1_PU:
            return q_max                                            # saturate inject
        elif vm_pu < U2_PU:
            return q_max * (U2_PU - vm_pu) / (U2_PU - U1_PU)     # ramp -> 0
        elif vm_pu <= U3_PU:
            return 0.0                                              # deadband
        elif vm_pu < U4_PU:
            return -q_max * (vm_pu - U3_PU) / (U4_PU - U3_PU)    # ramp -> absorb
        else:
            return -q_max                                           # saturate absorb

    @staticmethod
    def compute_setpoints(
            vm_pu:          pd.Series,
            p_installed_mw: pd.Series,
    ) -> pd.Series:
        """
        Vectorised Q(U) setpoints for multiple DERs.

        Parameters
        ----------
        vm_pu           : Bus voltages, indexed by sgen index.
        p_installed_mw  : Installed capacities (MW), indexed by sgen index.
                          Index must exactly match vm_pu -- raises ValueError
                          if not. A positional mismatch silently assigns wrong
                          Q to wrong DERs and cannot be caught downstream.

        Returns
        -------
        pd.Series  q_mvar, indexed by sgen index, dtype float.

        Raises
        ------
        ValueError  If indices do not match.
        """
        # Checks if the index of DER in vm_pu and p_installed_mw are same or not
        # They are both indexed by sgen so it should be same. If not return an error 
        if not vm_pu.index.equals(p_installed_mw.index):
            raise ValueError(
                "compute_setpoints: vm_pu and p_installed_mw must have the "
                "same index. "
                f"vm_pu.index={list(vm_pu.index)}, "
                f"p_installed_mw.index={list(p_installed_mw.index)}. "
                "Call p_installed_mw = p_installed_mw.reindex(vm_pu.index) "
                "if ordering differs."
            )
        # np.vectorize is a loop wrapper that avoids using a for loop
        #  otypes tells numpy the return type is float
        q = np.vectorize(QVCharacteristic.compute_setpoint, otypes=[float])(
            vm_pu.values, p_installed_mw.values
        )
        return pd.Series(q, index=vm_pu.index, dtype=float, name="q_mvar")

    @staticmethod
    def q_max_mvar(p_installed_mw: float) -> float:
        """Maximum reactive power (MVAr) for the given installed capacity."""
        return Q_RATIO * abs(p_installed_mw)

    @staticmethod
    def slope_inject() -> float:
        """dQ/dV slope in the injection ramp [U1_PU, U2_PU]. Positive.
        Units: pu_Q per (pu_V * MW_installed).
        slope_inject() = 16.0 means each 1 MW of installed capacity can 
        change Q by 16 MVAr per pu of voltage change in the ramp zone"""
        return Q_RATIO / (U2_PU - U1_PU)

    @staticmethod
    def slope_absorb() -> float:
        """dQ/dV slope in the absorption ramp [U3_PU, U4_PU]. Negative.
        Units: pu_Q per (pu_V * MW_installed)."""
        return -Q_RATIO / (U4_PU - U3_PU)


# ===========================================================================
# VoltVarResult -- structured output from run_timestep()
# ===========================================================================

@dataclass
class VoltVarResult:
    """
    Structured output from VoltVarController.run_timestep().

    Attributes
    ----------
    report_pre      : ViolationReport from runpp() before Q action.
    report_post     : ViolationReport from runpp() after Q action.
                      None if pre-control runpp() did not converge.
    q_setpoints     : pd.Series -- q_mvar applied to net.sgen.q_mvar,
                      indexed by sgen index. Zero-series if pre-PF failed.
    t_exchange_ms   : Serial round-trip time in ms.
                      0.0 in dry_run mode.
    t_total_ms      : Total timestep wall-clock time in ms.
    n_retries       : Serial retry count this timestep.
                      0 = first attempt succeeded.
                      max_retries = total failure (Q=0 applied).
                      0 always in dry_run mode.
    """

    report_pre    : ViolationReport
    report_post   : Optional[ViolationReport]
    q_setpoints   : pd.Series
    t_exchange_ms : float = 0.0
    t_total_ms    : float = 0.0
    n_retries     : int   = 0

    @property
    def converged_pre(self) -> bool:
        return self.report_pre.converged

    @property
    def converged_post(self) -> bool:
        return self.report_post is not None and self.report_post.converged

    @property
    def violations_resolved(self) -> bool:
        """
        True if any_violations was True pre-control and False post-control.

        Explicitly returns False if post-control runpp() did not converge.
        A diverged solver produces an empty ViolationReport (all DataFrames
        empty => any_violations=False), which would otherwise appear as
        "all violations resolved" and mask the solver crash in logs.
        """
        if not self.converged_post:
            return False
        return self.report_pre.any_violations and not self.report_post.any_violations

    @property
    def voltage_violations_reduced(self) -> bool:
        """
        True if the count of voltage-violating buses decreased.

        Explicitly returns False if post-control runpp() did not converge.
        An empty post report has n_post=0, which would falsely satisfy
        n_post < n_pre whenever there were pre-control violations.
        """
        if not self.converged_post:
            return False
        n_pre  = self.report_pre.n_over_voltage  + self.report_pre.n_under_voltage
        n_post = self.report_post.n_over_voltage + self.report_post.n_under_voltage
        return n_post < n_pre

    def summary(self) -> str:
        """One-line summary for logging."""
        pre  = self.report_pre.summary()
        post = self.report_post.summary() if self.report_post else "no post-PF"
        return (
            f"VoltVarResult | pre: {pre} | post: {post} | "
            f"exchange: {self.t_exchange_ms:.1f}ms | retries: {self.n_retries}"
        )


# ===========================================================================
# ArduinoSerialInterface -- serial wrapper for V/Q exchange
# ===========================================================================

class ArduinoSerialInterface:
    """
    Thin pyserial wrapper implementing the V/Q exchange protocol.

    Lifecycle
    ---------
    1. Instantiate with port and parameters.
    2. Call open() or use as context manager.
    3. Call configure(n_ders, p_installed_mw) once.
    4. Call exchange(vm_pu) each timestep.
    5. Call close() or let the context manager exit.

    Thread safety: not thread-safe. Single HIL loop only.

    Parameters
    ----------
    port        : Serial port path. Uno R3 on RPi is typically /dev/ttyACM0
                  (USB CDC) or /dev/ttyUSB0 (CH340 clone).
    baud        : Baud rate. Must match BAUD_RATE in volt_var_arduino.ino.
    timeout_s   : Read timeout for exchange() in seconds.
    max_retries : exchange() retry limit before raising SerialTimeoutError.
    """

    def __init__(
            self,
            port:        str,
            baud:        int   = BAUD_RATE,
            timeout_s:   float = DEFAULT_TIMEOUT_S,
            max_retries: int   = DEFAULT_MAX_RETRIES,   
    ):
        self.port        = port
        self.baud        = baud
        self.timeout_s   = timeout_s
        self.max_retries = max_retries
        self._ser        = None
        self._configured_batch_start = None   # tracks which batch is currently loaded

    def __enter__(self) -> ArduinoSerialInterface:
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def open(self) -> None:
        """
        Open the serial port and wait for Arduino reset to complete.

        The Arduino Uno R3 asserts DTR on USB serial open, triggering a
        hardware reset. ARDUINO_RESET_DELAY_S (2 s) is required before
        sending data. reset_input_buffer() discards all boot output
        (including "READY\\n" and any bootloader noise) so configure()
        begins from a clean state.
        Arduino does Hardware reset (via 100nF Capacitor) when the Data 
        Terminal Ready line pulses which takes 1-1.5 sec to complete 
        """
        try:
            import serial as _serial
        except ImportError:
            raise ImportError(
                "pyserial is required for ArduinoSerialInterface. "
                "Install with: pip install pyserial"
            ) from None

        self._ser = _serial.Serial(
            port     = self.port,
            baudrate = self.baud,
            timeout  = self.timeout_s,
        )
        time.sleep(ARDUINO_RESET_DELAY_S)
        self._ser.reset_input_buffer() # Discard any data received during the wait time

    def close(self) -> None:
        if self._ser is not None and self._ser.is_open:
            try:
                self._ser.write(b"END\n")
                self._ser.flush()
            except Exception:
                pass
            self._ser.close()
        self._ser = None

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    # ------------------------------------------------------------------
    # configure() -- startup handshake
    # ------------------------------------------------------------------

    def configure(self, n_ders: int, p_installed_mw: np.ndarray) -> None:
        """
        Send INIT, CFG and P messages to the Arduino. Call once after open().

        Protocol sequence (all under ACK_TIMEOUT_S per read):

            1. "INIT:<n>\\n"                          -> expect "ACK:INIT"
            2. "CFG:<q_ratio>,<u1>,<u2>,<u3>,<u4>\\n" -> expect "ACK:CFG"
            3. "P:<p1>,<p2>,...\\n"                   -> expect "ACK:P"

        The CFG step (step 2) transmits the Q(V) characteristic parameters
        Q_RATIO, U1_PU, U2_PU, U3_PU, U4_PU from this module's constants --
        the single source of truth -- so the firmware no longer relies on
        its own compile-time copies. Values are formatted at 4 decimal
        places, matching the firmware's float parsing resolution.

        Boot noise resilience: open() calls reset_input_buffer() after the
        2-second reset wait. configure() calls it again immediately before
        sending INIT to discard any bytes that arrived in the gap. The
        _read_ack() helper additionally skips non-ACK/ERR lines, handling
        any residual bootloader output that slipped through both flushes.
        CFG uses the same _read_ack() timeout-and-skip mechanism as INIT.

        Backward compatibility: firmware that predates the CFG step replies
        "ERR:UNKNOWN" (or stays silent) to the CFG message. Both outcomes
        raise SerialConfigError with a message explicitly identifying the
        CFG protocol step and pointing at a likely firmware version
        mismatch. SerialConfigError is used for ALL handshake-phase
        failures (INIT, CFG, P) by design; ArduinoProtocolError is reserved
        for the per-timestep exchange() V->Q loop. The failed step is
        identified by the message text ("INIT handshake", "CFG protocol
        step", "P handshake").

        Parameters
        ----------
        n_ders          : Number of DERs.
        p_installed_mw  : Installed capacity per DER (MW), shape (n_ders,).

        Raises
        ------
        SerialConfigError
            If port is not open, or the Arduino returns an unexpected
            response to the INIT, CFG, or P messages. For CFG the message
            explicitly names the CFG step and the likely cause (firmware
            without CFG support), so callers can diagnose version mismatch.
        ValueError
            If len(p_installed_mw) != n_ders.
        """
        if not self.is_open:
            raise SerialConfigError(
                "Serial port not open. Call open() or use as context manager."
            )
        if len(p_installed_mw) != n_ders:
            raise ValueError(
                f"p_installed_mw has {len(p_installed_mw)} values but n_ders={n_ders}."
            )

        original_timeout  = self._ser.timeout
        self._ser.timeout = ACK_TIMEOUT_S

        try:
            # --- THE WINDOWS IDLE GLITCH FIX ---
            self._ser.write(b'\n')
            self._ser.flush()
            time.sleep(0.1)
            
            # Drain any bytes that arrived since open()'s flush
            self._ser.reset_input_buffer()

            self._ser.write(f"INIT:{n_ders}\n".encode())
            ack = self._read_ack("INIT")
            if ack != "ACK:INIT":
                raise SerialConfigError(
                    f"INIT handshake failed -- expected 'ACK:INIT', got '{ack}'."
                )

            # ---- CFG: push the Q(V) characteristic to the firmware ----
            # Values come from this module's constants (single source of
            # truth). Order MUST match the firmware parser:
            #   q_ratio, u1, u2, u3, u4
            cfg_msg = (
                f"CFG:{Q_RATIO:.4f},{U1_PU:.4f},{U2_PU:.4f},"
                f"{U3_PU:.4f},{U4_PU:.4f}\n"
            )
            self._ser.write(cfg_msg.encode())
            try:
                # Same ACK_TIMEOUT_S + boot-noise-skipping read as INIT/P.
                # _read_ack raises SerialConfigError on ERR: (old firmware
                # answers ERR:UNKNOWN) and on silence (5 timed-out reads) --
                # re-raised with augmented context so the caller can identify
                # the CFG step specifically. SerialConfigError (not
                # ArduinoProtocolError) keeps the codebase taxonomy intact:
                # handshake-phase failures -> SerialConfigError; per-timestep
                # exchange() failures -> ArduinoProtocolError.
                ack = self._read_ack("CFG")
            except SerialConfigError as exc:
                raise SerialConfigError(
                    "CFG protocol step failed: no 'ACK:CFG' received for "
                    f"message '{cfg_msg.strip()}'. Cause: {exc} "
                    "Likely the Arduino is running firmware that predates "
                    "the CFG handshake (responds ERR:UNKNOWN or ignores it). "
                    "Re-flash volt_var_arduino.ino with CFG support, or roll "
                    "back this controller version."
                ) from exc
            if ack != "ACK:CFG":
                raise SerialConfigError(
                    f"CFG protocol step failed -- expected 'ACK:CFG', got "
                    f"'{ack}'. Firmware/controller protocol version mismatch."
                )

            p_str = ",".join(f"{p:.3f}" for p in p_installed_mw)
            self._ser.write(f"P:{p_str}\n".encode())
            ack = self._read_ack("P")
            if ack != "ACK:P":
                raise SerialConfigError(
                    f"P handshake failed -- expected 'ACK:P', got '{ack}'."
                )

        finally:
            self._ser.timeout = original_timeout

    def _read_ack(self, context: str) -> str:
        """
        Read lines until ACK: or ERR: is found, skipping boot noise.

        Consumes at most 5 lines to prevent an infinite loop if the
        Arduino is misbehaving. Raises SerialConfigError on ERR: or if
        no ACK/ERR is found within 5 lines.
        decode(errors="replace") handles any non-UTF-8 bytes from bootloader 
        noise by substituting the replacement character instead of raising UnicodeDecodeError.
        """
        for _ in range(5):
            raw  = self._ser.readline()
            line = raw.decode(errors="replace").strip()
            if line.startswith("ACK:") or line.startswith("ERR:"):
                if line.startswith("ERR:"):
                    raise SerialConfigError(
                        f"{context} handshake: Arduino returned '{line}'."
                    )
                return line
            # Non-ACK/ERR line (e.g. "READY" or bootloader noise) -- skip
        raise SerialConfigError(
            f"{context} handshake: no ACK received after 5 lines. "
            f"Confirm the Arduino is running volt_var_arduino.ino."
        )

    # ------------------------------------------------------------------
    # exchange() -- per-timestep V -> Q round trip
    # ------------------------------------------------------------------

    def exchange(self, vm_pu: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Send vm_pu values to Arduino; receive q_mvar setpoints.

        Safety guarantees
        -----------------
        1. NaN/inf pre-check: invalid floats are rejected before transmission;
           they would produce garbled ASCII and trigger ERR:V_INVALID, wasting
           a retry cycle.
        2. Message completeness: readline() returns without a trailing b'\\n'
           if the timeout fires mid-transmission (e.g. "Q:1.23,2." instead of
           "Q:1.23,2.45\\n"). The raw bytes are checked for b'\\n' before
           decoding. A truncated message is treated as a failed attempt.
        3. Response length validation: the parsed Q array must have exactly
           len(vm_pu) elements. A length mismatch means inconsistent DER
           counts between RPi and Arduino -- misapplying setpoints to the
           wrong DERs is a safety issue that must be caught explicitly.

        Parameters
        ----------
        vm_pu : Bus voltages, shape (n_ders,). Must be finite.

        Returns
        -------
        (q_arr, n_attempts) tuple.
            q_arr      : np.ndarray, shape (n_ders,), dtype float64.
            n_attempts : int, 1..max_retries. 1 = first try succeeded.
                         The caller stores n_attempts - 1 as n_retries.

        Raises
        ------
        SerialTimeoutError
            After max_retries failed attempts, or if vm_pu is non-finite.
        ArduinoProtocolError
            If Arduino returns ERR:<reason> other than ERR:UNKNOWN.
            ERR:UNKNOWN is retried (it indicates prefix corruption / a
            phantom byte, which a flush-and-resend can clear); all other
            ERR: codes are deterministic for the same message and are
            raised immediately without retry.
        """
        if not self.is_open:
            raise SerialTimeoutError("Serial port not open.")

        if not np.all(np.isfinite(vm_pu)):
            bad = np.where(~np.isfinite(vm_pu))[0].tolist()
            raise SerialTimeoutError(
                f"vm_pu contains non-finite values at positions {bad}. "
                f"Withholding transmission."
            )

        n_ders = len(vm_pu)
        vm_str = ",".join(f"{v:.4f}" for v in vm_pu)
        msg    = f"V:{vm_str}\n".encode()

        # Flush stale bytes once before the first attempt only.
        # Resetting before every retry risks discarding a valid Q: response
        # that arrives in the gap between reset_input_buffer() and the next read.
        self._ser.reset_input_buffer()

        for attempt in range(1, self.max_retries + 1):
            self._ser.write(msg)
            raw = self._ser.readline()    # bytes; includes '\n' if complete

            # Completeness check: readline() times out without '\n'
            if not raw.endswith(b"\n"):
                # Truncated response -- flush and retry
                self._ser.reset_input_buffer()
                continue

            line = raw.decode(errors="replace").strip()

            if line.startswith("ERR:"):
                if line == "ERR:UNKNOWN":
                    # Prefix corruption or phantom byte. Flush and retry.
                    self._ser.reset_input_buffer()
                    continue
                raise ArduinoProtocolError(
                    f"Arduino returned '{line}' on attempt {attempt}."
                )

            if line.startswith("Q:"):
                try:
                    q_arr = np.array(
                        [float(x) for x in line[2:].split(",")],
                        dtype=float,
                    )
                except ValueError:
                    self._ser.reset_input_buffer()
                    continue   # malformed float -- retry

                if len(q_arr) != n_ders:
                    # DER count mismatch is a protocol error, not a timeout.
                    # The Arduino responded correctly but with the wrong count,
                    # indicating a configuration inconsistency (INIT sent with
                    # wrong n_ders). Retrying the same V: message will get the
                    # same wrong-length response -- raise immediately.
                    raise ArduinoProtocolError(
                        f"Response length mismatch: expected {n_ders} Q values, "
                        f"got {len(q_arr)}. Re-run configure() if DER count changed."
                    )

                return q_arr, attempt

            # Unexpected prefix -- flush and retry
            self._ser.reset_input_buffer()

        raise SerialTimeoutError(
            f"No valid Q: response after {self.max_retries} attempts "
            f"on port '{self.port}'."
        )

    def exchange_batched(
            self,
            vm_pu:         np.ndarray,
            p_installed_mw: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Handle any number of DERs by splitting into batches of ARDUINO_MAX_DERS.

        For n_ders <= ARDUINO_MAX_DERS this is identical to exchange().
        For n_ders > ARDUINO_MAX_DERS the Arduino is reconfigured once per batch.

        Each batch is a complete INIT+P+V/Q cycle. Since Q(V) is stateless
        (each DER's Q depends only on its own vm_pu and p_installed), batching
        is mathematically identical to processing all DERs in one message.

        Latency at 115200 baud for 210 DERs (2 batches of 105):
            2 × (INIT+P handshake ~20ms + V:/Q: ~175ms) = ~390ms

        Parameters
        ----------
        vm_pu          : shape (n_ders,)  — voltages for ALL DERs
        p_installed_mw : shape (n_ders,)  — capacities for ALL DERs

        Returns
        -------
        (q_arr, total_attempts)
            q_arr: shape (n_ders,) — assembled Q setpoints for all DERs
        """
        n_total = len(vm_pu)

        if n_total <= ARDUINO_MAX_DERS:
            return self.exchange(vm_pu)

        q_all          = np.zeros(n_total, dtype=float)
        total_attempts = 0

        for start in range(0, n_total, ARDUINO_MAX_DERS):
            end      = min(start + ARDUINO_MAX_DERS, n_total)
            batch_vm = vm_pu[start:end]
            batch_p  = p_installed_mw[start:end]

            # Reconfigure Arduino for this batch's DER count and capacities
            if start != self._configured_batch_start:
                self.configure(len(batch_vm), batch_p)
                self._configured_batch_start = start

            q_batch, n_att  = self.exchange(batch_vm)
            q_all[start:end] = q_batch
            total_attempts  += n_att

        return q_all, total_attempts

# ===========================================================================
# VoltVarController -- main HIL coordination class
# ===========================================================================

class VoltVarController:
    """
    Coordinates the Q(V) Volt-Var control loop between pandapower and Arduino.

    Used exclusively for Scenario 4 (rule-based Volt-Var HIL). Not used in
    Scenarios 1 (baseline ConstControl + run_timeseries) or 5 (OPF runopp).

    Parameters
    ----------
    net             : pandapower network. Modified in place by run_timestep().
    interface       : ArduinoSerialInterface, or None for dry_run mode.
    sgen_indices    : Iterable of sgen indices to control.
                      Default: all in-service sgens, ascending order.
    p_installed_mw  : Installed active power per DER (MW), shape (n_ders,).
                      Default: net.sgen.sn_mva where finite and positive,
                      else net.sgen.p_mw.
    dry_run         : If True, compute Q locally -- no serial communication.
                      Automatically True when interface is None.
    """

    def __init__(
            self,
            net,
            interface:      Optional[ArduinoSerialInterface] = None,
            sgen_indices                                     = None,
            p_installed_mw                                   = None,
            dry_run:        bool                             = False,
    ):
        self._net   = net
        self._iface = interface
        self._dry   = dry_run or (interface is None)

        # Resolve sgen indices
        if sgen_indices is None:
            self._sgen_idx = (
                net.sgen.index[net.sgen["in_service"] == True]
                .sort_values().copy()
            )
        else:
            self._sgen_idx = pd.Index(sgen_indices)

        # Resolve installed capacities
        if p_installed_mw is None:
            self._p_installed = self._resolve_p_installed()
        else:
            self._p_installed = np.asarray(p_installed_mw, dtype=float)

        if len(self._p_installed) != len(self._sgen_idx):
            raise ValueError(
                f"p_installed_mw length ({len(self._p_installed)}) does not "
                f"match sgen_indices length ({len(self._sgen_idx)})."
            )

        zero_mask = self._p_installed <= 0.0
        if zero_mask.any():
            warnings.warn(
                f"[VoltVarController] p_installed_mw <= 0 for sgen indices "
                f"{self._sgen_idx[zero_mask].tolist()}. "
                f"Q_max = 0 for these DERs.",
                UserWarning, stacklevel=2,
            )

        self._sgen_buses = net.sgen.loc[self._sgen_idx, "bus"].values.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_p_installed(self) -> np.ndarray:
        '''
        The installed capacity (used to compute Q_max) should be the rated apparent power sn_mva, 
        not the current active power output p_mw. sn_mva represents the inverter’s maximum capability 
        and is fixed; p_mw varies with solar irradiance and changes every timestep. 
        use_sn = np.isfinite(sn) & (sn > 0.0) masks rows where sn_mva is NaN, zero, or negative. 
        For those rows, p_mw is used as a fallback. np.where(use_sn, sn, p_mw) selects 
        element-wise: sn where the mask is True, p_mw otherwise.
        '''
        sgens  = self._net.sgen.loc[self._sgen_idx]
        sn     = sgens["sn_mva"].values.astype(float)
        p_mw   = sgens["p_mw"].values.astype(float)
        use_sn = np.isfinite(sn) & (sn > 0.0)
        return np.where(use_sn, sn, p_mw).astype(float)

    def _get_vm_pu(self) -> np.ndarray:
        """
        Read vm_pu at each DER's connection bus from net.res_bus.

        Raises RuntimeError if any bus is absent from res_bus -- this would
        otherwise produce a KeyError (pandas) or NaN (older pandas), both of
        which would propagate silently into Q calculations.
        self._sgen_buses is a NumPy array of bus indices (copied from net.sgen.loc[_sgen_idx, "bus"] in the constructor). 
        res_bus_set = set(net.res_bus.index) converts the DataFrame index to a Python set for O(1) membership lookup
        """
        res_bus_set = set(self._net.res_bus.index)
        missing = [
            int(b) for b in self._sgen_buses
            if b not in res_bus_set
        ]
        if missing:
            raise RuntimeError(
                f"[VoltVarController] Buses {missing} absent from net.res_bus "
                f"after runpp(). Check sgens are on in-service buses."
            )
        return self._net.res_bus.loc[self._sgen_buses, "vm_pu"].values.astype(float)

    def _clamp_to_net_limits(self, q_arr: np.ndarray) -> np.ndarray:
        """
        Clamp Q setpoints against two layers:

        Layer 1 -- explicit min/max_q_mvar on net.sgen (where finite).
        Layer 2 -- apparent power limit: |Q| <= sqrt(sn_mva^2 - p_mw^2).
            Q_RATIO * P_b_inst can exceed sn_mva capability if sn_mva was
            set lower than p_installed_mw. This layer is the safety backstop.

        Returns a new array.
        """
        q_out = q_arr.copy()
        sgens = self._net.sgen
        # Layer 1: explicit min/max_q_mvar columns (where finite)
        if "max_q_mvar" in sgens.columns:
            max_q = sgens.loc[self._sgen_idx, "max_q_mvar"].values.astype(float)
            fin   = np.isfinite(max_q)
            if fin.any():
                # where the limit is finite, apply it; where it is NaN/Inf (not set for this DER), leave q_out unchanged.
                q_out = np.where(fin, np.minimum(q_out, max_q), q_out)
        # Layer 2: apparent power limit |Q| <= sqrt(sn_mva^2 - p_mw^2)
        if "min_q_mvar" in sgens.columns:
            min_q = sgens.loc[self._sgen_idx, "min_q_mvar"].values.astype(float)
            fin   = np.isfinite(min_q)
            if fin.any():
                q_out = np.where(fin, np.maximum(q_out, min_q), q_out)

        # Apparent power cap
        sn    = sgens.loc[self._sgen_idx, "sn_mva"].values.astype(float)
        p_now = sgens.loc[self._sgen_idx, "p_mw"].values.astype(float)
        fin   = np.isfinite(sn) & (sn > 0.0) & np.isfinite(p_now)
        if fin.any():
            p_clamp = np.minimum(np.abs(p_now), sn)
            q_lim   = np.sqrt(np.maximum(0.0, sn ** 2 - p_clamp ** 2))
            q_out   = np.where(fin, np.clip(q_out, -q_lim, q_lim), q_out)

        return q_out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_ders(self) -> int:
        return len(self._sgen_idx)

    @property
    def sgen_indices(self) -> pd.Index:
        return self._sgen_idx

    @property
    def p_installed_mw(self) -> np.ndarray:
        return self._p_installed.copy()

    def configure(self) -> None:
        """
        Send INIT and P messages to Arduino. No-op in dry_run mode.
        Call once after open() and before the first run_timestep().
        """
        if self._dry:
            return
        if self._iface is None:
            raise SerialConfigError(
                "interface is None but dry_run=False."
            )
        # ---- Guard for unsupported DER counts ----
        if self.n_ders == 0:
            warnings.warn(
                "[VoltVarController] No DERs in network; skipping Arduino INIT.",
                RuntimeWarning, stacklevel=2,
            )
            return

        if self.n_ders > ARDUINO_MAX_DERS:
            warnings.warn(
                f"[VoltVarController] n_ders={self.n_ders} exceeds "
                f"ARDUINO_MAX_DERS={ARDUINO_MAX_DERS}. "
                "Skipping initial INIT and using exchange_batched().",
                RuntimeWarning, stacklevel=2,
            )
            return
        self._iface.configure(self.n_ders, self._p_installed)

    def run_timestep(
            self,
            runpp_kwargs: Optional[dict] = None,
    ) -> VoltVarResult:
        """
        Execute one HIL timestep: pre-PF -> Q exchange -> apply -> post-PF.

        Caller must set net.sgen.p_mw, net.load.p_mw, net.load.q_mvar before
        calling this method.

        Convergence safety: if pp.runpp() raises, a local pf_pre_ok flag is
        set False. detect_violations() is always called (it independently
        checks net.res_bus), but if pf_pre_ok=False the method returns early
        without attempting Q exchange. This avoids proceeding on a stale
        net.converged=True from a prior timestep.

        Serial failure handling: ArduinoProtocolError (ERR: from Arduino)
        and SerialTimeoutError (timeout / truncation) are both caught and
        fall back to Q=0 with a RuntimeWarning. The distinction is preserved
        in the warning text for operator diagnostics.

        Parameters
        ----------
        runpp_kwargs : Forwarded to pp.runpp(). voltage_depend_loads=False
                       is always enforced regardless of kwargs.

        Returns
        -------
        VoltVarResult
        """
        t0 = time.perf_counter()

        # Guard: no controlled DERs -- nothing to do
        if self.n_ders == 0:
            empty_q = pd.Series([], dtype=float, name="q_mvar")
            pp.runpp(self._net, **{"voltage_depend_loads": False, **(runpp_kwargs or {})})
            report = detect_violations(self._net)
            return VoltVarResult(
                report_pre  = report,
                report_post = report,
                q_setpoints = empty_q,
                t_total_ms  = (time.perf_counter() - t0) * 1e3,
            )

        # Reset q_mvar to 0 before the pre-control power flow.
        # Without this reset, the pre-control PF for timestep T inherits
        # the Q setpoints applied in timestep T-1, contaminating report_pre
        # with the partially-controlled state of the previous step. report_pre
        # must reflect the genuinely uncontrolled network for benchmarking
        # Scenario 4 against the Scenario 1 baseline (where q_mvar=0 always).
        self._net.sgen.loc[self._sgen_idx, "q_mvar"] = 0.0

        kwargs = {"voltage_depend_loads": False}
        if runpp_kwargs:
            kwargs.update(runpp_kwargs)
            kwargs["voltage_depend_loads"] = False

        # ---- Pre-control power flow ----
        pf_pre_ok = True
        try:
            pp.runpp(self._net, **kwargs)
        except Exception as exc:
            pf_pre_ok = False
            warnings.warn(
                f"[VoltVarController] Pre-control runpp() raised: {exc}",
                RuntimeWarning, stacklevel=2,
            )

        report_pre = detect_violations(self._net)

        if not pf_pre_ok or not report_pre.converged:
            empty_q = pd.Series(
                np.zeros(self.n_ders, dtype=float),
                index=self._sgen_idx,
                name="q_mvar",
            )
            return VoltVarResult(
                report_pre  = report_pre,
                report_post = None,
                q_setpoints = empty_q,
                t_total_ms  = (time.perf_counter() - t0) * 1e3,
            )

        # ---- Q(V) exchange ----
        vm_pu     = self._get_vm_pu()
        n_retries = 0

        if self._dry:
            vm_s = pd.Series(vm_pu,             index=self._sgen_idx)
            p_s  = pd.Series(self._p_installed, index=self._sgen_idx)
            q_arr         = QVCharacteristic.compute_setpoints(vm_s, p_s).values
            t_exchange_ms = 0.0

        else:
            t_exch = time.perf_counter()
            try:
                q_arr, n_attempts = self._iface.exchange_batched(vm_pu, self._p_installed)
                n_retries = n_attempts - 1
            except ArduinoProtocolError as exc:
                warnings.warn(
                    f"[VoltVarController] Arduino protocol error: {exc}. "
                    f"Falling back to Q=0 this timestep.",
                    RuntimeWarning, stacklevel=2,
                )
                q_arr     = np.zeros(self.n_ders, dtype=float)
                n_retries = self._iface.max_retries
            except SerialTimeoutError as exc:
                warnings.warn(
                    f"[VoltVarController] Serial timeout/truncation: {exc}. "
                    f"Falling back to Q=0 this timestep.",
                    RuntimeWarning, stacklevel=2,
                )
                q_arr     = np.zeros(self.n_ders, dtype=float)
                n_retries = self._iface.max_retries
            t_exchange_ms = (time.perf_counter() - t_exch) * 1e3

        # ---- Clamp and apply ----
        q_arr       = self._clamp_to_net_limits(q_arr)
        q_setpoints = pd.Series(q_arr, index=self._sgen_idx, name="q_mvar")
        self._net.sgen.loc[self._sgen_idx, "q_mvar"] = q_arr

        # ---- Post-control power flow ----
        try:
            pp.runpp(self._net, **kwargs)
        except Exception as exc:
            warnings.warn(
                f"[VoltVarController] Post-control runpp() raised: {exc}",
                RuntimeWarning, stacklevel=2,
            )

        report_post = detect_violations(self._net)

        return VoltVarResult(
            report_pre    = report_pre,
            report_post   = report_post,
            q_setpoints   = q_setpoints,
            t_exchange_ms = t_exchange_ms,
            t_total_ms    = (time.perf_counter() - t0) * 1e3,
            n_retries     = n_retries,
        )
