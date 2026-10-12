# TASK 4 — DOCUMENT 1B
## Serial Protocol & Arduino Interface — `volt_var_controller.py`
### Serial constants (L243–252), exceptions (L259–275), `ArduinoSerialInterface` (L494–895)
### Complete line-by-line walkthrough

> **Source verified:** lines 243–275 and 494–895 of the current
> `volt_var_controller.py` were read directly before writing. The quantisation
> figures in §9 are cross-checked against the project's
> `QUANTISATION_MECHANISM_FINDINGS.md`. If the file is edited, re-verify line
> numbers.
>
> **Read alongside:** Master Index §5–§8; **Document 1A** (the Q(U) law whose
> exact float64 outputs this layer quantises to 4-decimal ASCII); **Document 5**
> (the Arduino firmware — the *other* end of every message here, and the source
> of the Q-formatting leg of the quantisation finding); **Document 1C** (the
> loop that calls `exchange_batched` and catches these exceptions).
>
> **Flowchart:** `flow_volt_var_exchange.*` and `flow_arduino_handshake.*`
> (Task 1a).

---

## TABLE OF CONTENTS

1. [Overview — the serial transport layer](#1-overview)
2. [File logic flow — pseudocode (this slice)](#2-file-logic-flow--pseudocode-this-slice)
3. [Call-site map — where every entity is used](#3-call-site-map--where-every-entity-is-used)
4. [Section A — Serial protocol constants (L243–252)](#4-section-a--serial-protocol-constants-l243252)
5. [Section B — Custom exceptions & the taxonomy (L259–275)](#5-section-b--custom-exceptions--the-taxonomy-l259275)
6. [Section C — ArduinoSerialInterface (L494–895)](#6-section-c--arduinoserialinterface-l494895)
   - [Class, `__init__`, context manager (L494–536)](#class-init-context-manager-l494536)
   - [open / close / is_open (L538–578)](#open--close--is_open-l538578)
   - [configure — the handshake (L584–703)](#configure--the-handshake-l584703)
   - [_read_ack (L705–728)](#_read_ack-l705728)
   - [exchange — the per-timestep round trip (L734–843)](#exchange--the-per-timestep-round-trip-l734843)
   - [exchange_batched (L845–895)](#exchange_batched-l845895)
7. [The serial-protocol quantisation finding](#7-the-serial-protocol-quantisation-finding)
8. [Failure modes and the exception taxonomy](#8-failure-modes-and-the-exception-taxonomy)
9. [Coverage checklist](#9-coverage-checklist)

---

## 1. Overview

If Document 1A is the *law* (given a voltage, what Q?), Document 1B is the
*wire*. `ArduinoSerialInterface` is a thin pyserial wrapper that carries bus
voltages from the RPi to the Arduino and reactive-power setpoints back, framed
as newline-terminated ASCII at 115200 baud. It is used **only** on the hardware
path of Scenario 4; the dry-run path never touches this class (it calls 1A's
`compute_setpoints` directly).

Three things live in this slice:

- **Serial protocol constants** (L243–252) — baud, board capacity, timeouts,
  retry limit, reset delay.
- **Three custom exceptions** (L259–275) — a deliberate taxonomy that keeps
  handshake failures, protocol errors and timeouts distinguishable, so a
  configure-phase failure can never be silently swallowed by the per-timestep
  fallback.
- **`ArduinoSerialInterface`** (L494–895) — port lifecycle (`open`/`close`/
  context manager), the once-per-session `configure()` handshake
  (INIT → CFG → P), the per-timestep `exchange()` V→Q round trip with retry and
  validation, and `exchange_batched()` for DER counts above one board's SRAM
  budget.

**Why this layer matters for the paper.** Every value that crosses this wire is
formatted to a fixed number of ASCII decimals — voltages and the CFG curve to
4 (`%.4f`), installed capacities to 3 (`%.3f`). That 4-decimal formatting of
the voltage is one leg of the **serial-protocol quantisation** finding (§7);
the other leg is the Arduino's `%.4f` formatting of Q on the return trip
(Document 5). The dry-run path uses full float64, so the two paths disagree by
a small but *coordinator-visible* amount — the mechanism the paper reports.

**How 1B connects to 1A and 1C.** 1C's `VoltVarController.run_timestep` runs a
power flow, reads the DER-bus voltages, and — on the hardware path — hands them
to `exchange_batched` here; the returned Q is clamped and applied, then a second
power flow runs. `configure()` transmits 1A's characteristic constants
(`Q_RATIO`, `U1_PU..U4_PU`) to the firmware via `CFG:`, which is what makes 1A
the single source of truth even for the hardware curve.

---

## 2. File logic flow — pseudocode (this slice)

Plain-language flow — *what happens in what order*, no Python.

```
SERIAL CONSTANTS (module load):
    baud rate; max DERs one board can hold; per-read timeout; longer ACK timeout;
    retry limit; mandatory post-open reset delay

EXCEPTIONS (three distinct types, never conflated):
    SerialConfigError   — the startup handshake failed (port shut, bad ACK)
    ArduinoProtocolError— the board answered but rejected/mis-answered a timestep
    SerialTimeoutError  — no answer in time, or a truncated answer

CLASS ArduinoSerialInterface — owns the pyserial port, speaks the wire protocol:

    open():
        open the port
        wait out the Arduino's power-on reset (opening the port resets the board)
        discard the boot noise that arrived during the wait

    close():
        send END so the board knows the session is over
        close the port

    configure(n_ders, capacities):          [once per session, or once per batch]
        widen the read timeout for slow handshake ACKs
        nudge-and-drain the line (Windows idle-glitch workaround)
        send  INIT:<n>              → require ACK:INIT
        send  CFG:<curve params>    → require ACK:CFG     (push the 1A curve to firmware)
        send  P:<capacities>        → require ACK:P
        restore the original timeout no matter what
        any bad/absent ACK          → SerialConfigError naming the step

    exchange(voltages):                       [once per timestep]
        refuse to transmit if ANY voltage is non-finite (would garble the ASCII)
        format voltages to 4 decimals; send V:<...>
        flush the input once (not per retry — a valid reply could arrive in the gap)
        UP TO max_retries times:
            read one line
            arrived truncated (no newline)?   → flush, retry
            ERR:UNKNOWN (transient corruption)?→ flush, retry
            any other ERR:?                    → raise ArduinoProtocolError (deterministic)
            Q:<...>?
                parse floats
                wrong count?                   → raise ArduinoProtocolError
                else                            → return (Q values, attempt number)
            anything else?                     → flush, retry
        exhausted all retries                  → raise SerialTimeoutError

    exchange_batched(voltages, capacities):
        fits one board? → just exchange()
        else: split into batches of ARDUINO_MAX_DERS,
              reconfigure the board once per batch (Q(V) is stateless, so batching
              is mathematically identical to one big message),
              exchange each batch, stitch the Q arrays back together
```

---

## 3. Call-site map — where every entity is used

Verified by grep. "No caller" entities are flagged with a verdict per the
Master Index §5 convention.

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `BAUD_RATE` (L247) | constant | `__init__` default `baud` (L520). Must match `volt_var_arduino.ino`'s serial setup (Doc 5). |
| `ARDUINO_MAX_DERS` (L248) | constant | `exchange_batched` batching (L875, L881–882); `VoltVarController.configure` guard (L1079, Doc 1C). The hard 105 ceiling from the ATmega328P SRAM budget. |
| `DEFAULT_TIMEOUT_S` (L249) | constant | `__init__` default `timeout_s` (L521). |
| `ACK_TIMEOUT_S` (L250) | constant | `configure()` — temporarily widens the port timeout for handshake ACKs (L642). |
| `DEFAULT_MAX_RETRIES` (L251) | constant | `__init__` default `max_retries` (L522). |
| `ARDUINO_RESET_DELAY_S` (L252) | constant | `open()` post-open sleep (L563). Referenced in `executor.py` (Doc 9). |
| `SerialTimeoutError` (L259–262) | exception | Raised in `exchange` (L775, L779, L840) and `VoltVarController.run_timestep` catches it (L1195). Caught in `sensitivity_coordinator.py` L1024. Imported by coordinator L100. |
| `ArduinoProtocolError` (L264–269) | exception | Raised in `exchange` (L810, L830). Caught in `run_timestep` L1187 and coordinator L1024. Imported by coordinator L99. |
| `SerialConfigError` (L272–275) | exception | Raised throughout `configure` (L633, L656, L680, L689, L698, L720, L725) and `VoltVarController.configure` L1068. Referenced in `executor.py` and `plugin_runner.py` docstrings as the "loudest flash failsafe." |
| `ArduinoSerialInterface` (class) | class | **Instantiated** in `scenario_4_volt_var.py` L495 (context manager) and `plugin_runner.py` L384. Imported by scenario_4 L87, coordinator L98, plugin_runner L382. |
| `__init__` (L517–529) | method | Via the instantiations above. |
| `__enter__` / `__exit__` (L531–536) | methods | Context-manager use: scenario_4 L495, coordinator L53. |
| `open` (L538–564) | method | `__enter__` (L532); direct in `plugin_runner.py` L385. |
| `close` (L566–574) | method | `__exit__` (L536); direct in `plugin_runner.py` L408. |
| `is_open` (L576–578) | property | Internal guards in `configure` (L632) and `exchange` (L774). |
| `configure` (L584–703) | method | `VoltVarController.configure` L1087; `exchange_batched` L888 (per batch); `plugin_runner.py` L394. (`VoltVarController.configure` is in turn called from hosting_capacity L639, scenario_4 L244, coordinator L55/L68, custom_controller L248.) |
| `_read_ack` (L705–728) | method | `configure` only — once each for INIT (L654), CFG (L678), P (L696). |
| `exchange` (L734–843) | method | `exchange_batched` only (L876, L891). **Not called directly anywhere else** — `exchange_batched` is the public entry point even for small DER counts. |
| `exchange_batched` (L845–895) | method | `VoltVarController.run_timestep` L1185 (hardware path); `sensitivity_coordinator.py` L1020 (coordinator `q_initial` hardware path); `plugin_runner.py` L400. |

**No uncalled public entities in this slice** — everything here is on a live
path. (`exchange` is "internal-only public" — reachable but always routed
through `exchange_batched`; that is by design, not an oversight, so no verdict
is needed.)

---

## 4. Section A — Serial protocol constants (L243–252)

```python
# Lines 243-252
# ===========================================================================
# Serial protocol constants
# ===========================================================================

BAUD_RATE              = 115200
ARDUINO_MAX_DERS       = 105    # hard ceiling from ATmega328P SRAM budget
DEFAULT_TIMEOUT_S      = 1.0    # per-read timeout for exchange()
ACK_TIMEOUT_S          = 5.0    # longer timeout for configure() ACK responses
DEFAULT_MAX_RETRIES    = 3      # exchange() retry limit before fallback to Q=0
ARDUINO_RESET_DELAY_S  = 3.0    # Uno R3 resets on USB serial open -- mandatory wait
```

#### Line 247: `BAUD_RATE = 115200`

The serial line speed in bits/second. **Must match** the Arduino's
`Serial.begin(...)` in `volt_var_arduino.ino` (Doc 5) — a mismatch produces
garbage on both ends, not an error. 115200 is the fast standard rate the Uno R3
reliably sustains; at this rate the per-timestep V/Q round trip for ~100 DERs is
well under the control interval. **Where used:** `__init__` default (L520).

#### Line 248: `ARDUINO_MAX_DERS = 105`

The maximum number of DERs one Arduino can hold in a single `configure()`. **The
number is a memory budget, not a protocol limit:** the ATmega328P has 2 KB of
SRAM; each DER needs a float for `p_installed` plus working space, and 105 is the
largest count that fits with headroom for the serial buffers and stack.
`exchange_batched` exists precisely to handle counts above this by splitting into
batches. **Where used:** `exchange_batched` (L875, L881–882) and the
`VoltVarController.configure` guard (L1079, Doc 1C). **Worked example:** the
primary network has ~100 DERs — one batch, no splitting; a 210-DER network would
be 2 batches of 105.

#### Line 249: `DEFAULT_TIMEOUT_S = 1.0`

The per-read timeout for the per-timestep `exchange()` — how long a single
`readline()` waits for a Q response before returning what it has (possibly
truncated). 1 s is generous for a round trip that normally completes in tens of
ms; it is long enough to absorb an occasional scheduling hiccup on the RPi
without stalling the loop. **Where used:** `__init__` default (L521).

#### Line 250: `ACK_TIMEOUT_S = 5.0`

A *longer* timeout used only during `configure()`. Handshake ACKs can be slow:
the firmware may still be settling after reset, and the CFG/P parsing does real
work. 5 s avoids a spurious handshake failure on a board that is simply slow to
answer the first message. `configure()` widens the port timeout to this value and
restores the original in a `finally` (L641–642, L702–703). **Where used:**
`configure()` (L642).

#### Line 251: `DEFAULT_MAX_RETRIES = 3`

How many times `exchange()` re-sends a `V:` message before giving up and raising
`SerialTimeoutError` (which the loop turns into a fallback-to-Q=0 for that
timestep, Doc 1C). 3 balances resilience against transient line noise with not
stalling the loop on a genuinely dead board. **Where used:** `__init__` default
(L522).

#### Line 252: `ARDUINO_RESET_DELAY_S = 3.0`

The mandatory wait after opening the port. **Derivation of the need:** the Uno
R3 asserts a hardware reset when the USB CDC serial port is opened (DTR pulses a
100 nF cap on the reset line); the board reboots and runs its bootloader before
`setup()`. Sending data during this window is lost. 3 s comfortably covers the
~1–1.5 s reset+bootloader. **Note the docstring drift:** `open()`'s docstring
(L543) says "2 s", but the constant is 3.0 — a stale docstring number, **flagged
not edited**. **Where used:** `open()` (L563); referenced in `executor.py`.

---

## 5. Section B — Custom exceptions & the taxonomy (L259–275)

```python
# Lines 259-275
class SerialTimeoutError(RuntimeError):
    """Arduino did not respond within the timeout window, or readline() returned
    a truncated message (no terminating newline byte received)."""
    pass

class ArduinoProtocolError(RuntimeError):
    """Arduino returned an explicit ERR: response. Distinct from a timeout --
    the Arduino is alive but rejected the message ..."""
    pass

class SerialConfigError(RuntimeError):
    """Startup handshake failed. Either the port is not open, or the Arduino
    returned an unexpected response to INIT or P messages."""
    pass
```

#### Line 259–275: Three exceptions — one per failure *phase*, deliberately

**Design rationale — why three types, not one.** The three map to three
distinct failure *phases*, and keeping them distinct is a safety property:

- **`SerialConfigError`** — a **handshake-phase** failure (port not open, wrong
  ACK to INIT/CFG/P). Raised by `configure()`. A configure failure is fatal to
  the run — the board is not set up, so no timestep can proceed.
- **`ArduinoProtocolError`** — a **per-timestep** failure where the board is
  *alive but rejected or mis-answered* a `V:` message (an `ERR:` other than the
  transient `ERR:UNKNOWN`, or a wrong-length Q). Deterministic for the same
  message → not worth retrying.
- **`SerialTimeoutError`** — a **per-timestep** failure where the board *did not
  answer in time* or answered *truncated*. Possibly transient → the loop falls
  back to Q=0 for that timestep and continues.

**The safety point (why this is not over-engineering).** `VoltVarController.
run_timestep` (Doc 1C) wraps the per-timestep exchange in
`except ArduinoProtocolError` / `except SerialTimeoutError`, both falling back to
Q=0. If `configure()` raised the *same* type, a handshake failure would be caught
by that per-timestep fallback and **silently swallowed** — the run would proceed
timestep after timestep applying Q=0, looking like a working-but-ineffective
controller, when in fact the board was never configured. Reserving
`SerialConfigError` for the handshake phase makes that misclassification
impossible: a configure failure propagates out of the timestep loop and stops
the run loudly. This taxonomy is stated explicitly in `configure()`'s docstring
(L610–615).

**Where used:** see §3. All three are `RuntimeError` subclasses, so a caller that
does not care about the distinction can still `except RuntimeError`.

---

## 6. Section C — ArduinoSerialInterface (L494–895)

### Class, `__init__`, context manager (L494–536)

```python
# Lines 517-529
    def __init__(self, port, baud=BAUD_RATE, timeout_s=DEFAULT_TIMEOUT_S,
                 max_retries=DEFAULT_MAX_RETRIES):
        self.port        = port
        self.baud        = baud
        self.timeout_s   = timeout_s
        self.max_retries = max_retries
        self._ser        = None
        self._configured_batch_start = None   # tracks which batch is currently loaded
```

#### Line 517–529: Construction — store parameters, defer the port

**Parameters:** `port` (str, e.g. `/dev/ttyACM0` on the RPi, `/dev/ttyUSB0` for a
CH340 clone); `baud` (int, default `BAUD_RATE`); `timeout_s` (float, default
`DEFAULT_TIMEOUT_S`); `max_retries` (int, default `DEFAULT_MAX_RETRIES`).

**Design rationale.** `__init__` stores parameters and sets `_ser = None` — it
does **not** open the port. Opening is deferred to `open()` so the object can be
constructed cheaply and opened via the context manager (or explicitly), and so
construction cannot fail on a busy/absent port. `_configured_batch_start`
remembers which batch's INIT/P is currently loaded on the board, so
`exchange_batched` only re-configures when the batch actually changes (L887).

**Where used:** instantiated in scenario_4 L495 and plugin_runner L384.

```python
# Lines 531-536
    def __enter__(self) -> ArduinoSerialInterface:
        self.open()
        return self
    def __exit__(self, *_) -> None:
        self.close()
```

#### Line 531–536: Context-manager protocol

`with ArduinoSerialInterface(port) as arduino:` opens on entry and — crucially —
`close()`s on exit **even if an exception propagates**, so the port is always
released and the `END` message always sent. `__exit__(*_)` ignores the exception
triple (it does not suppress exceptions — returning `None` is falsy). **Where
used:** scenario_4 L495, coordinator L53.

---

### open / close / is_open (L538–578)

```python
# Lines 538-564 (body)
    def open(self) -> None:
        try:
            import serial as _serial
        except ImportError:
            raise ImportError("pyserial is required ...") from None
        self._ser = _serial.Serial(port=self.port, baudrate=self.baud,
                                   timeout=self.timeout_s)
        time.sleep(ARDUINO_RESET_DELAY_S)
        self._ser.reset_input_buffer()   # Discard boot noise
```

#### Line 538–564: Open the port and survive the Arduino's reset

**Derivation of the sequence.** (1) Import pyserial *inside* the method, not at
module top — so the whole module (and the pure-Python dry-run path in 1A/1C) can
be imported and used on a machine without pyserial installed; the dependency is
only required when hardware is actually opened. The `from None` suppresses the
chained `ImportError` for a clean message. (2) Open the `Serial` port — this
asserts DTR and **resets the Uno R3** (§4, L252). (3) `time.sleep(
ARDUINO_RESET_DELAY_S)` waits out the reset+bootloader. (4)
`reset_input_buffer()` discards everything received during the wait —
`"READY\n"`, bootloader chatter, line noise — so `configure()` starts from a
clean input buffer.

**Design rationale.** Without the sleep, the first `INIT:` is sent into a
rebooting board and lost → handshake timeout. Without the flush, boot noise sits
in the buffer and the first `readline()` returns it instead of `ACK:INIT` (the
`_read_ack` skip logic is the second line of defence for anything that slips
through). **Where used:** `__enter__` (L532), plugin_runner L385.

```python
# Lines 566-574
    def close(self) -> None:
        if self._ser is not None and self._ser.is_open:
            try:
                self._ser.write(b"END\n")
                self._ser.flush()
            except Exception:
                pass
            self._ser.close()
        self._ser = None
```

#### Line 566–574: Close politely, but never fail on close

Sends `END\n` so the firmware can return to its idle state, flushes, then closes.
**Design rationale — the bare `except Exception: pass`.** Close must be
idempotent and must never raise: it runs from `__exit__`, often while another
exception is already propagating. If the write fails (board already gone, port
already half-closed), swallowing it lets the real error surface and still closes
the port. `_ser = None` afterwards makes a second `close()` a no-op and flips
`is_open` to `False`. **Where used:** `__exit__` (L536), plugin_runner L408.

```python
# Lines 576-578
    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open
```

#### Line 576–578: `is_open` — the guard predicate

Two-part check: the port object exists *and* pyserial reports it open. Used by
`configure()` and `exchange()` to fail fast with a clear message rather than
`AttributeError` on `None`. **Where used:** `configure` L632, `exchange` L774.

---

### configure — the handshake (L584–703)

The once-per-session (or once-per-batch) setup: INIT → CFG → P. This is where 1A's
characteristic constants are pushed to the firmware, making 1A the single source
of truth for the hardware curve.

```python
# Lines 632-642
        if not self.is_open:
            raise SerialConfigError("Serial port not open. Call open() ...")
        if len(p_installed_mw) != n_ders:
            raise ValueError(f"p_installed_mw has {len(p_installed_mw)} values "
                             f"but n_ders={n_ders}.")
        original_timeout  = self._ser.timeout
        self._ser.timeout = ACK_TIMEOUT_S
```

#### Line 632–642: Preconditions and timeout widening

Fail fast if the port is shut (`SerialConfigError`) or the capacity array length
disagrees with `n_ders` (`ValueError` — a caller bug, distinct from a wire
failure). Then save the current timeout and widen it to `ACK_TIMEOUT_S` (5 s) for
the handshake; the `finally` at L702–703 restores it no matter what. **Design
rationale for the length check being `ValueError`, not `SerialConfigError`:** a
mismatched array is a programming error on the RPi side, caught before any byte
is sent — categorically different from the board misbehaving.

```python
# Lines 645-651
            # --- THE WINDOWS IDLE GLITCH FIX ---
            self._ser.write(b'\n')
            self._ser.flush()
            time.sleep(0.1)
            self._ser.reset_input_buffer()
```

#### Line 645–651: The Windows idle-glitch workaround

**What it does.** Writes a lone newline, flushes, waits 100 ms, and drains the
input buffer *again* (open() already drained once). **Design rationale.** On
Windows hosts (the RPi is driven from a Windows Mobile Hotspot in this testbed,
and the laptop-with-Arduino leg runs on Windows), an idle USB-CDC port can
deliver a stale phantom byte on the first real write. Sending a throwaway `\n`
absorbs that glitch, and the second drain clears anything it dislodged, so the
first *meaningful* message (`INIT:`) starts clean. Cheap insurance against a
class of first-message handshake failures that are otherwise maddening to
reproduce.

```python
# Lines 653-658
            self._ser.write(f"INIT:{n_ders}\n".encode())
            ack = self._read_ack("INIT")
            if ack != "ACK:INIT":
                raise SerialConfigError(f"INIT handshake failed -- expected "
                                        f"'ACK:INIT', got '{ack}'.")
```

#### Line 653–658: INIT — announce the DER count

Sends `INIT:<n>\n`; the firmware allocates its per-DER arrays for `n` DERs and
replies `ACK:INIT`. `_read_ack` (below) reads past any residual boot noise. Any
other reply → `SerialConfigError` naming the INIT step. **Worked example:** 100
DERs → `INIT:100\n` → `ACK:INIT`.

```python
# Lines 660-692
            cfg_msg = (f"CFG:{Q_RATIO:.4f},{U1_PU:.4f},{U2_PU:.4f},"
                       f"{U3_PU:.4f},{U4_PU:.4f}\n")
            self._ser.write(cfg_msg.encode())
            try:
                ack = self._read_ack("CFG")
            except SerialConfigError as exc:
                raise SerialConfigError("CFG protocol step failed: no 'ACK:CFG' "
                    f"received for message '{cfg_msg.strip()}'. Cause: {exc} "
                    "Likely the Arduino is running firmware that predates the CFG "
                    "handshake ...") from exc
            if ack != "ACK:CFG":
                raise SerialConfigError(f"CFG protocol step failed -- expected "
                    f"'ACK:CFG', got '{ack}'. Firmware/controller protocol "
                    f"version mismatch.")
```

#### Line 660–692: CFG — push the single-source-of-truth curve *(quantisation leg 1)*

**The most important lines in the handshake.** `CFG:` transmits 1A's five
characteristic constants — `Q_RATIO`, `U1_PU..U4_PU` — to the firmware, in the
exact order the firmware parser expects. This is the mechanism that retired the
old "keep the numbers identical in two files by hand" hazard: the firmware's
compile-time copies are now only boot defaults, overwritten by whatever CFG
sends.

**Format detail — `%.4f`, and its link to the paper (§7).** Each constant is
formatted to **4 decimal places**. For the breakpoints and `Q_RATIO=0.2500`
that is lossless (they have ≤2 significant decimals). But `%.4f` is the *same*
formatting used for the per-timestep voltages (L785), and that one is **not**
lossless — it is the first leg of the quantisation finding. Documenting it here
so the format choice is traced to one decision: the wire protocol is 4-decimal
ASCII throughout.

**Backward-compatibility design.** A firmware without CFG support answers
`ERR:UNKNOWN` or stays silent. `_read_ack` turns both into `SerialConfigError`,
which is caught and **re-raised with augmented context** — the message names the
CFG step and the likely cause (firmware predating CFG) and tells the operator to
re-flash. The taxonomy is preserved: `SerialConfigError` (not
`ArduinoProtocolError`) for this handshake-phase failure. **Worked example:**
`CFG:0.2500,0.9600,0.9900,1.0100,1.0400\n` → `ACK:CFG`.

```python
# Lines 694-703
            p_str = ",".join(f"{p:.3f}" for p in p_installed_mw)
            self._ser.write(f"P:{p_str}\n".encode())
            ack = self._read_ack("P")
            if ack != "ACK:P":
                raise SerialConfigError(f"P handshake failed -- expected "
                                        f"'ACK:P', got '{ack}'.")
        finally:
            self._ser.timeout = original_timeout
```

#### Line 694–703: P — send installed capacities, then restore the timeout

Sends `P:<p1>,<p2>,...\n` with each capacity at **3 decimals** (`%.3f` — MW to
kW resolution, adequate for a fixed rated value that never changes during a run,
unlike the voltages). The firmware stores these as each DER's `p_installed` for
its `q_max = Q_RATIO·P` computation. The `finally` restores the original
per-read timeout so subsequent `exchange()` calls use `DEFAULT_TIMEOUT_S`, not
the 5 s ACK timeout. **Worked example (DER-A, DER-B):** `P:0.500,1.500\n` →
`ACK:P`.

---

### _read_ack (L705–728)

```python
# Lines 715-728
        for _ in range(5):
            raw  = self._ser.readline()
            line = raw.decode(errors="replace").strip()
            if line.startswith("ACK:") or line.startswith("ERR:"):
                if line.startswith("ERR:"):
                    raise SerialConfigError(f"{context} handshake: Arduino "
                                            f"returned '{line}'.")
                return line
            # Non-ACK/ERR line (e.g. "READY" or bootloader noise) -- skip
        raise SerialConfigError(f"{context} handshake: no ACK received after 5 "
                                f"lines. Confirm the Arduino is running ...")
```

#### Line 705–728: Read an ACK, skipping boot noise, bounded to 5 lines

**What it does.** Reads up to 5 lines; returns the first `ACK:`; raises
`SerialConfigError` on the first `ERR:`; skips anything else (`READY`, bootloader
chatter). If 5 lines pass with no ACK/ERR, raises.

**Design rationale.**
- **The 5-line cap** prevents an infinite loop on a chatty or wedged board — a
  bounded number of skips, then a definite failure.
- **`decode(errors="replace")`** turns non-UTF-8 bootloader bytes into the
  replacement character instead of raising `UnicodeDecodeError` mid-handshake —
  a noisy byte becomes a skipped line, not a crash.
- **The `context` argument** ("INIT"/"CFG"/"P") is threaded into every error
  message so a handshake failure says *which* step failed — essential for the
  CFG backward-compat diagnostics above.

**Where used:** `configure` only, once per handshake step (L654, L678, L696).

---

### exchange — the per-timestep round trip (L734–843)

The hot path: one `V:` out, one `Q:` back, with three safety guarantees
(non-finite pre-check, completeness check, length validation) and a bounded
retry loop.

```python
# Lines 774-786
        if not self.is_open:
            raise SerialTimeoutError("Serial port not open.")
        if not np.all(np.isfinite(vm_pu)):
            bad = np.where(~np.isfinite(vm_pu))[0].tolist()
            raise SerialTimeoutError(f"vm_pu contains non-finite values at "
                                     f"positions {bad}. Withholding transmission.")
        n_ders = len(vm_pu)
        vm_str = ",".join(f"{v:.4f}" for v in vm_pu)
        msg    = f"V:{vm_str}\n".encode()
```

#### Line 774–782: Guards — port open, and no non-finite voltage

Port-shut → `SerialTimeoutError`. **The non-finite pre-check** rejects `NaN`/
`inf` voltages *before* transmission, reporting the offending positions. **Design
rationale:** a `NaN` would format as the literal ASCII `"nan"`, which the firmware
cannot parse — it would answer `ERR:V_INVALID` and waste a whole retry cycle.
Catching it here withholds the transmission entirely and surfaces a precise
diagnostic. (1A's `compute_setpoint` also guards `NaN` voltages by returning
Q=0; this is the transport-layer counterpart on the hardware path.)

#### Line 785: `vm_str = ",".join(f"{v:.4f}" for v in vm_pu)` *(quantisation leg 2 — the paper's headline)*

**This single line is the origin of the serial-protocol quantisation finding.**
Each float64 bus voltage from `pandapower.runpp()` is formatted to **4 decimal
places** of ASCII before transmission. That is a genuine precision reduction
relative to the dry-run path, which feeds full float64 straight into 1A's
`compute_setpoint`. §7 derives the consequence in full; the short version:

```
worst-case rounding error   δv = 0.5 × 10⁻⁴ = 5 × 10⁻⁵ pu
Q(V) ramp sensitivity        dQ/dV = Q_RATIO·P/(U2−U1) = 0.25·0.5/0.03 = 4.1667 MVAr/pu
worst-case Q error           δQ = 4.1667 × 5×10⁻⁵ = 2.08 × 10⁻⁴ MVAr
vs coordinator tolerance     δQ / SATURATION_TOL = 2.08×10⁻⁴ / 1×10⁻⁶ ≈ 208×
```

**Terminology note (for the paper).** `%.4f` performs round-half-to-even, so the
precise term is "4-decimal ASCII **rounding**", not "truncation" — the project's
informal "truncation" wording is a slight misnomer. **Flagged, not edited.**

**Units check.** `%.4f` prints a dimensionless pu value; the quantum is 10⁻⁴ pu.
Multiplying by `dQ/dV [MVAr/pu]` gives MVAr — the Q-domain error. ✓

```python
# Lines 788-795
        self._ser.reset_input_buffer()
        for attempt in range(1, self.max_retries + 1):
            self._ser.write(msg)
            raw = self._ser.readline()    # bytes; includes '\n' if complete
```

#### Line 788–795: Flush *once*, then the retry loop

**Design rationale — flush before the loop, not inside it.** The input buffer is
drained once before the first attempt. Doing it *per retry* would risk discarding
a valid `Q:` response that arrives in the gap between `reset_input_buffer()` and
the next `readline()` — a race that would turn a good reply into a spurious
timeout. The retries below have their *own* targeted flushes only on the paths
that need them (truncation, `ERR:UNKNOWN`, unexpected prefix). `attempt` runs
`1..max_retries` so the returned attempt count is human-meaningful (1 = first try).

```python
# Lines 797-812
            if not raw.endswith(b"\n"):
                self._ser.reset_input_buffer()
                continue
            line = raw.decode(errors="replace").strip()
            if line.startswith("ERR:"):
                if line == "ERR:UNKNOWN":
                    self._ser.reset_input_buffer()
                    continue
                raise ArduinoProtocolError(f"Arduino returned '{line}' on "
                                           f"attempt {attempt}.")
```

#### Line 797–812: Completeness check, then ERR triage

**Completeness (guarantee 2).** `readline()` returns without a trailing `\n` if
the 1 s timeout fires mid-message (`"Q:1.23,2."` instead of `"Q:1.23,2.45\n"`).
Checking the raw *bytes* for `b"\n"` before decoding catches this — a truncated
message is flushed and retried, never parsed.

**ERR triage — the transient/deterministic split.** `ERR:UNKNOWN` means prefix
corruption / a phantom byte — a *transient* fault a flush-and-resend can clear,
so it retries. **Every other** `ERR:` (e.g. `ERR:V_INVALID`, `ERR:COUNT`) is
*deterministic* for the same message: retrying would get the same error, so it
raises `ArduinoProtocolError` immediately. **Design rationale:** distinguishing
transient from deterministic avoids both wasted retries (on deterministic errors)
and premature failure (on transient ones).

```python
# Lines 814-838
            if line.startswith("Q:"):
                try:
                    q_arr = np.array([float(x) for x in line[2:].split(",")],
                                     dtype=float)
                except ValueError:
                    self._ser.reset_input_buffer()
                    continue   # malformed float -- retry
                if len(q_arr) != n_ders:
                    raise ArduinoProtocolError(f"Response length mismatch: "
                        f"expected {n_ders} Q values, got {len(q_arr)}. "
                        f"Re-run configure() if DER count changed.")
                return q_arr, attempt
            self._ser.reset_input_buffer()   # unexpected prefix -- flush and retry
```

#### Line 814–838: Parse Q, validate length, return

Strips the `Q:` prefix, splits on commas, parses floats. **A malformed float**
(a mangled byte producing `"2.4x"`) raises `ValueError` locally → treated as a
transient corruption → flush and retry (not a hard failure).

**Length validation (guarantee 3) — a safety property.** The parsed Q array must
have exactly `n_ders` elements. A mismatch means the RPi and Arduino disagree on
the DER count (e.g. `configure()` sent a different `n`), which would cause Q
setpoints to be applied to the *wrong DERs* — a silent, physically wrong result.
This is raised as `ArduinoProtocolError` **immediately, without retry**: the same
`V:` message would get the same wrong-length answer, and the fix is to re-run
`configure()`, not to resend.

An **unexpected prefix** (neither `ERR:` nor `Q:`) is flushed and retried.

```python
# Lines 840-843
        raise SerialTimeoutError(f"No valid Q: response after "
            f"{self.max_retries} attempts on port '{self.port}'.")
```

#### Line 840–843: Retries exhausted → SerialTimeoutError

Falling off the loop means `max_retries` attempts all failed transiently
(truncation / `ERR:UNKNOWN` / malformed / unexpected). Raise `SerialTimeoutError`
— which `run_timestep` (Doc 1C) catches and converts to a Q=0 fallback for that
timestep, logging the port. **Worked example:** on a disconnected board all 3
attempts time out (truncated reads) → `SerialTimeoutError` → that timestep runs
uncontrolled, the loop continues.

---

### exchange_batched (L845–895)

```python
# Lines 873-895
        n_total = len(vm_pu)
        if n_total <= ARDUINO_MAX_DERS:
            return self.exchange(vm_pu)
        q_all          = np.zeros(n_total, dtype=float)
        total_attempts = 0
        for start in range(0, n_total, ARDUINO_MAX_DERS):
            end      = min(start + ARDUINO_MAX_DERS, n_total)
            batch_vm = vm_pu[start:end]
            batch_p  = p_installed_mw[start:end]
            if start != self._configured_batch_start:
                self.configure(len(batch_vm), batch_p)
                self._configured_batch_start = start
            q_batch, n_att   = self.exchange(batch_vm)
            q_all[start:end] = q_batch
            total_attempts  += n_att
        return q_all, total_attempts
```

#### Line 845–895: Split large DER counts across batches

**The public entry point for the exchange** — even small counts go through here
(the fast path `n_total <= ARDUINO_MAX_DERS` just delegates to `exchange`, which
is why `exchange` has no other callers).

**Derivation — why batching is exact, not an approximation.** Q(V) is
*stateless*: each DER's Q depends only on its own `vm_pu` and `p_installed`,
nothing else. So processing DERs 0–104 in one INIT+P+V/Q cycle and 105–209 in a
second is **mathematically identical** to processing all 210 at once. The loop
slices `[start:end]` in blocks of `ARDUINO_MAX_DERS`, reconfigures the board only
when the batch changes (`start != self._configured_batch_start` — avoids a
redundant INIT/P when the same batch is re-exchanged), exchanges, and writes each
batch's Q into the right slice of `q_all`.

**Units / worked example.** 210 DERs → 2 batches of 105. Latency at 115200 baud
per the docstring: `2 × (~20 ms handshake + ~175 ms V/Q) ≈ 390 ms`. The primary
network (~100 DERs) never batches — one `exchange`, `_configured_batch_start`
untouched. **Where used:** `run_timestep` L1185, coordinator L1020, plugin_runner
L400.

---

## 7. The serial-protocol quantisation finding

This is the paper's headline result, and this document is where its **cause**
lives (the `%.4f` formatting at L785 and L665). The full derivation, verified
against the live constants in `QUANTISATION_MECHANISM_FINDINGS.md`:

**Step 1 — the Q(V) ramp sensitivity.** On either ramp segment,
`dQ/dV = Q_RATIO·P_installed/(U2−U1)`. With `U2−U1 = U4−U3 = 0.03` pu exactly
(the symmetric breakpoints), both ramps share the same slope. For the project's
default injected-DER size `P = 0.5 MW`:

```
dQ/dV = 0.25 × 0.5 / 0.03 = 4.1667 MVAr per pu
```

**Step 2 — the rounding error from `%.4f`.** Formatting a float64 voltage to 4
decimals (L785) is round-half-to-even; the worst-case error is half the last
place:

```
δv_max = 0.5 × 10⁻⁴ = 0.00005 pu
```

**Step 3 — the resulting Q error, versus the coordinator's tolerance.**

```
δQ_max = dQ/dV × δv_max = 4.1667 × 0.00005 = 2.08 × 10⁻⁴ MVAr
δQ_max / SATURATION_TOL = 2.08×10⁻⁴ / 1×10⁻⁶ ≈ 208×
```

A single worst-case rounding event is **~208× the coordinator's own
saturation-detection tolerance** (`SATURATION_TOL = 1e-6` MVAr, Docs 2/3). It is
not marginal — it is two-to-three orders of magnitude above the detection floor,
which is why `coordination_active` fires on nearly every timestep on the serial
path but only ~9% of the time on the dry-run path.

**Cross-checks (from the mechanism findings).**
- *Empirical:* the measured average `q_initial` disagreement between the
  RPi/serial and laptop/dry-run paths is ~0.0013 MVAr/DER — larger than the
  0.5 MW single-step bound because it averages over larger DERs (a ~3.12 MW DER
  alone would produce 0.0013 MVAr from one rounding step) and includes the
  Arduino's AVR float32 re-computation on top of the ASCII rounding.
- *Numerical:* sweeping 200,000 voltages and comparing `compute_q(v)` against
  `compute_q(round(v,4))` gives a max discrepancy of 0.00020833 MVAr, matching
  the analytical 0.00020833 to five significant figures — confirming 4-decimal
  rounding *alone* accounts for the maximum disagreement.

**Two suppression levers (not equivalent).**
- **Lever A — raise `SATURATION_TOL`** past ~2×10⁻⁴. This changes only whether
  the *detector* reacts; the underlying ~2×10⁻⁴ MVAr disagreement is still
  there. A detection-threshold change, **not** a root-cause fix.
- **Lever B — round the dry-run `vm_pu` to 4 decimals** before feeding
  `compute_setpoint`, so both paths quantise identically. This attacks the
  *source* of the asymmetry: with identically rounded inputs there is nothing
  left for any coordinator, at any tolerance, to detect. This is the honest fix
  and the basis for the proposed follow-up experiment that decomposes the
  ASCII-rounding term from the AVR-float32 term.

**Why the finding is "serial-protocol", not "hardware", quantisation.** The 2×2
mode×platform decomposition showed RPi-dry matches laptop-dry to 10+ significant
figures — the RPi's own arithmetic contributes nothing. The effect is tied to the
4-decimal ASCII *protocol* (and the AVR float32 recompute), not to the Raspberry
Pi. Hence the paper's framing.

---

## 8. Failure modes and the exception taxonomy

| Condition | Where | Exception | Retry? | Downstream |
|-----------|-------|-----------|--------|------------|
| Port not open (configure) | L632 | `SerialConfigError` | — | Run stops loudly |
| Capacity array length ≠ n_ders | L636 | `ValueError` | — | Caller bug, pre-transmit |
| Bad/absent ACK to INIT/CFG/P | L656/689/698/725 | `SerialConfigError` | 5-line skip in `_read_ack` | Run stops loudly (never swallowed by timestep fallback) |
| Firmware predates CFG | L680 | `SerialConfigError` (augmented) | — | Message says "re-flash" |
| Port not open (exchange) | L775 | `SerialTimeoutError` | — | Q=0 fallback (Doc 1C) |
| Non-finite `vm_pu` | L779 | `SerialTimeoutError` | — | Withheld pre-transmit |
| Truncated reply (no `\n`) | L798 | — | flush + retry | Q=0 after max_retries |
| `ERR:UNKNOWN` (transient) | L806 | — | flush + retry | Q=0 after max_retries |
| Other `ERR:` (deterministic) | L810 | `ArduinoProtocolError` | no | Q=0 this timestep |
| Malformed Q float | L820 | — | flush + retry | Q=0 after max_retries |
| Q length ≠ n_ders | L830 | `ArduinoProtocolError` | no | Q=0; re-run configure() |
| All retries exhausted | L840 | `SerialTimeoutError` | — | Q=0 this timestep |

The taxonomy's safety guarantee (§5): handshake failures (`SerialConfigError`)
propagate out of the timestep loop; only per-timestep failures
(`ArduinoProtocolError`/`SerialTimeoutError`) are caught and converted to Q=0, so
a misconfigured board can never masquerade as a working-but-idle one.

---

## 9. Coverage checklist

Every named entity in L243–275 and L494–895, each confirmed documented.

| Entity | Lines | Kind | Documented | Where-used stated |
|--------|-------|------|:----------:|:-----------------:|
| `BAUD_RATE` | 247 | constant | ✅ | ✅ |
| `ARDUINO_MAX_DERS` | 248 | constant | ✅ | ✅ |
| `DEFAULT_TIMEOUT_S` | 249 | constant | ✅ | ✅ |
| `ACK_TIMEOUT_S` | 250 | constant | ✅ | ✅ |
| `DEFAULT_MAX_RETRIES` | 251 | constant | ✅ | ✅ |
| `ARDUINO_RESET_DELAY_S` | 252 | constant | ✅ (stale "2 s" docstring flagged) | ✅ |
| `SerialTimeoutError` | 259–262 | exception | ✅ | ✅ |
| `ArduinoProtocolError` | 264–269 | exception | ✅ | ✅ |
| `SerialConfigError` | 272–275 | exception | ✅ | ✅ |
| `ArduinoSerialInterface` | 494–895 | class | ✅ | ✅ |
| `__init__` | 517–529 | method | ✅ | ✅ |
| `__enter__` | 531–533 | method | ✅ | ✅ |
| `__exit__` | 535–536 | method | ✅ | ✅ |
| `open` | 538–564 | method | ✅ | ✅ |
| `close` | 566–574 | method | ✅ | ✅ |
| `is_open` | 576–578 | property | ✅ | ✅ |
| `configure` | 584–703 | method | ✅ | ✅ |
| `_read_ack` | 705–728 | method | ✅ | ✅ |
| `exchange` | 734–843 | method | ✅ | ✅ (internal-only public) |
| `exchange_batched` | 845–895 | method | ✅ | ✅ |

**Also covered (not standalone entities):** the Windows idle-glitch workaround
(L645–651), the flush-once-not-per-retry rationale (L788–791), the three
`exchange()` safety guarantees (non-finite pre-check, completeness, length
validation), and the `%.4f`/`%.3f` formatting decisions (L665, L694, L785).

**No uncalled public entities in this slice** (see §3).

---

*End of Document 1B. Next: Document 1C — `VoltVarController` + `VoltVarResult`
(L901–1229), the HIL loop that runs the pre-/post-control power flows, chooses
the dry-run (1A) or hardware (1B) path, clamps against network limits, and
catches the exceptions defined here.*
