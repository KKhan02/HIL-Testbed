# TASK 4 — DOCUMENT 5
## Arduino Firmware — `volt_var_arduino.ino`
### The device-side mirror of the Q(V) law (1A) and serial protocol (1B)
### Complete line-by-line walkthrough

> **Source verified:** `volt_var_arduino.ino` (578 lines) was read in full before
> writing. This is C for the ATmega328P (Arduino Uno R3); the same walkthrough
> standard applies. If the file is edited, re-verify line numbers.
>
> **Read alongside — this is the counterpart of two documents:** **Document 1A**
> (`compute_q` here is the line-for-line mirror of `QVCharacteristic.compute_setpoint`)
> and **Document 1B** (`handle_message` here is the other end of every `INIT`/`CFG`/
> `P`/`V`/`END` message `ArduinoSerialInterface` sends, and the source of every
> `ACK:`/`ERR:` it reads). The recommended reading order (Master Index §3) pairs
> this document with 1A.
>
> **Flowchart:** `flow_arduino_handshake.*`, `flow_volt_var_exchange.*` (Task 1a).

---

## TABLE OF CONTENTS

1. [Overview — the device side of the loop](#1-overview)
2. [File logic flow — pseudocode](#2-file-logic-flow--pseudocode)
3. [Mirror & correspondence map](#3-mirror--correspondence-map)
4. [Section A — Includes & constants (L163–196)](#4-section-a--includes--constants-l163196)
5. [Section B — Global state & the SRAM budget (L202–223)](#5-section-b--global-state--the-sram-budget-l202223)
6. [Section C — compute_q: the mirror of 1A (L245–262)](#6-section-c--compute_q-the-mirror-of-1a-l245262)
7. [Section D — parse_float_array (L285–299)](#7-section-d--parse_float_array-l285299)
8. [Section E — handle_message: the protocol handler (L315–497)](#8-section-e--handle_message-the-protocol-handler-l315497)
9. [Section F — The Q-formatting quantisation leg (L459–484)](#9-section-f--the-q-formatting-quantisation-leg-l459484)
10. [Section G — setup & loop (L504–578)](#10-section-g--setup--loop-l504578)
11. [Coverage checklist](#11-coverage-checklist)

---

## 1. Overview

This firmware is the *silicon half* of the Hardware-in-the-Loop path. On the RPi
side (Documents 1A/1B), the controller computes voltages, frames them as ASCII,
and sends them over USB serial; **this firmware receives them, computes Q with its
own copy of the VDE-AR-N 4110 characteristic, frames the answer as ASCII, and
sends it back.** In dry-run mode none of this runs — the RPi computes Q locally
(1A). The whole point of the HIL path, and of the paper's finding, is that this
device-side round trip introduces *quantisation* the dry-run path does not.

Three responsibilities, each mirroring a Python element:

| Firmware | Mirrors | Role |
|----------|---------|------|
| `compute_q` (L245) | `QVCharacteristic.compute_setpoint` (1A) | the Q(V) law, on-device |
| `handle_message` (L315) | `ArduinoSerialInterface.configure`/`exchange` (1B) | the protocol, on-device |
| `loop`/`setup` (L504+) | (no Python analogue) | non-blocking serial + lifecycle |

**The two quantisation legs (the paper's finding, Doc 1B §7).** The 4-decimal ASCII
format is used in *both* directions: the RPi formats voltages to 4 decimals
(1B L785, the V leg), and **this firmware formats Q to 4 decimals** (L465,
`FLOAT_DEC = 4`, the Q leg). On top of the Q leg, the ATmega328P recomputes
`compute_q` in **float32**, adding the AVR-float32 term. The dry-run path has
neither — it uses float64 end to end. That asymmetry, ~208× `SATURATION_TOL` in
the worst case, is what the coordinator detects on the HIL path but not the
dry-run path.

**The mirror invariant.** `compute_q` and `QVCharacteristic.compute_setpoint`
implement the *same* five-segment curve, independently, in two languages. The
`CFG:` message keeps the *parameter values* in sync automatically (the RPi pushes
`Q_RATIO`/`U1..U4`, overwriting the firmware's compile-time defaults). But the
*shape* — the five branches, the boundary conditions — lives in two places and
must be changed in both. This document verifies the firmware side matches 1A
exactly.

---

## 2. File logic flow — pseudocode

```
CONSTANTS: the Q(V) breakpoints + Q_RATIO (compile-time defaults, overwritten by CFG);
           protocol sizes (baud, max DERs, buffer size, float decimals = 4)

GLOBAL STATE (static, not stack — SRAM budget):
    per-DER installed capacity; a voltage buffer (also reused as CFG parse scratch);
    DER count; two "configured" flags; the CFG-supplied curve params;
    one buffer used for BOTH receiving and responding

compute_q(vm, p):                         [mirror of 1A]
    q_max = cfg_q_ratio · |p|
    five-branch piecewise curve using the CFG params → signed q

parse_float_array(text, out, max_n):
    strtod each comma-separated token; stop on the first malformed token; return count

handle_message(msg):                      [mirror of 1B's device end]
    INIT:<n>  → validate n (strtol, range); set DER count; ACK:INIT / ERR:INIT_RANGE
    CFG:<...> → parse 5 floats (ask for 6 to catch "too many"); validate
                (finite, 0<q_ratio≤1, strictly increasing); parse-then-commit;
                ACK:CFG / ERR:CFG_COUNT / ERR:CFG_INVALID
    P:<...>   → require INIT then CFG first; parse capacities; ACK:P / ERR:*
    V:<...>   → require configured; parse voltages; reject non-finite;
                compute_q per DER; format each Q to 4 decimals; send "Q:<...>"
    END       → clear configured state (no response)
    else      → ERR:UNKNOWN   (the transient error the RPi retries on)

setup():  start serial; print "READY" (RPi discards it); LED on
loop():   non-blocking — accumulate bytes into buf; on '\n' dispatch; discard '\r';
          on buffer-full-without-newline → ERR:BUF_OVERFLOW + skip to next '\n';
          heartbeat-blink the LED when configured
```

---

## 3. Mirror & correspondence map

Firmware is C, so "where used" means (a) the internal call graph and (b) the
Python element each piece mirrors.

**Internal call graph:** `loop()` → (on `\n`) `handle_message()` → `compute_q()`
and `parse_float_array()`; `setup()`/`loop()` are the Arduino entry points.

**`compute_q` ↔ `QVCharacteristic.compute_setpoint` (1A) — branch by branch:**

| Firmware (L251–261) | Python 1A (L338–345) | Segment |
|---------------------|----------------------|---------|
| `vm_pu <= cfg_u1 → q_max` | `vm_pu <= U1_PU → q_max` | saturate inject |
| `vm_pu < cfg_u2 → q_max*(cfg_u2-vm)/(cfg_u2-cfg_u1)` | `q_max*(U2_PU-vm)/(U2_PU-U1_PU)` | inject ramp |
| `vm_pu <= cfg_u3 → 0` | `→ 0.0` | deadband |
| `vm_pu < cfg_u4 → -q_max*(vm-cfg_u3)/(cfg_u4-cfg_u3)` | `-q_max*(vm-U3_PU)/(U4_PU-U3_PU)` | absorb ramp |
| `else → -q_max` | `else → -q_max` | saturate absorb |

**Identical.** The only differences are cosmetic (C `float` literals, `fabsf`) and
that the firmware reads `cfg_*` runtime params (set by `CFG:`) where 1A reads
module globals (set by `set_qv_parameters`) — both the single-source-of-truth
mechanism, each on its own side.

**`handle_message` ↔ 1B messages/errors:**

| Firmware handler | 1B sender (Doc 1B) | Success | Errors → 1B taxonomy |
|------------------|--------------------|---------|-----------------------|
| `INIT:<n>` (L318) | `configure` INIT (L654) | `ACK:INIT` | `ERR:INIT_RANGE` → `SerialConfigError` |
| `CFG:<...>` (L341) | `configure` CFG `.4f` (L665) | `ACK:CFG` | `ERR:CFG_COUNT`/`ERR:CFG_INVALID` → `SerialConfigError` |
| `P:<...>` (L384) | `configure` P `.3f` (L694) | `ACK:P` | `ERR:P_BEFORE_INIT`/`ERR:CFG_BEFORE_P`/`ERR:P_COUNT` → `SerialConfigError` |
| `V:<...>` (L407) | `exchange` V `.4f` (L785) | `Q:<...>` | `ERR:NOT_CONFIGURED`/`ERR:V_COUNT`/`ERR:V_INVALID`/`ERR:BUF_OVERFLOW` → `ArduinoProtocolError` |
| `END` (L489) | `close` (L568) | *(silent)* | — |
| *(unrecognised)* | — | — | `ERR:UNKNOWN` → **transient, retried** (1B L806) |

This table *closes the loop* with Doc 1B §5's exception taxonomy: `ERR:UNKNOWN` is
the firmware's catch-all for a corrupted prefix, which 1B treats as transient
(flush + retry); every other `ERR:` is deterministic and 1B raises on it.

---

## 4. Section A — Includes & constants (L163–196)

```c
#include <math.h>      /* isnan(), isinf(), fabsf() */
#include <stdlib.h>    /* strtof(), strtol(), atoi() */
#include <string.h>    /* strncmp(), memcpy(), strlen() */
#define U1_PU 0.96f
#define U2_PU 0.99f
#define U3_PU 1.01f
#define U4_PU 1.04f
#define Q_RATIO 0.25f /*was 0.48*/
#define BAUD_RATE 115200
#define MAX_DERS  105
#define BUF_SIZE  870
#define FLOAT_WIDTH 8
#define FLOAT_DEC   4
#define TMP_SIZE    20
#define LED_PIN     9
```

#### Line 178–182: Characteristic constants — the compile-time defaults

The same five values as 1A (`0.96/0.99/1.01/1.04`, `Q_RATIO = 0.25f`). **Design
rationale — why these exist even though CFG overwrites them.** They are *boot
defaults* so `compute_q` is well-defined before any `CFG:` arrives (bench testing
over a serial monitor). In the live HIL path the RPi's `CFG:` overwrites them
(§5, §8). The `f` suffix forces single-precision literals (the ATmega328P has no
double — `double == float == 32-bit`, §7). The `/*was 0.48*/` note records the
prior tuning value; the value in force is `0.25`, matching 1A (and the mirror
invariant requires it). **Note:** these are `#define` macros, not variables — they
seed the `cfg_*` globals at L215–219.

#### Line 188–196: Protocol & buffer constants — sized to the SRAM budget

`BAUD_RATE = 115200` (must match 1B L520). `MAX_DERS = 105` (the SRAM ceiling,
same as 1B's `ARDUINO_MAX_DERS`). `BUF_SIZE = 870` (the shared receive/response
buffer). **`FLOAT_DEC = 4`** — the decimal places `dtostrf` uses to format each Q
value (§9): **the Q leg of the quantisation finding.** `FLOAT_WIDTH = 8` (minimum
field width, space-padded then stripped), `TMP_SIZE = 20` (per-value scratch, sized
for any realistic MVAr). **Design rationale:** every buffer size is chosen against
the ATmega328P's 2 KB SRAM (§5); the removed `FLOAT_CHARS` comment (L194) records
that per-value width depends on magnitude, so `TMP_SIZE` (not `FLOAT_WIDTH`) is the
sizing basis.

---

## 5. Section B — Global state & the SRAM budget (L202–223)

```c
static float    p_installed[MAX_DERS];   /* rated capacity per DER (MW) */
static float    vm_buf[MAX_DERS];        /* voltages from V: — also CFG parse scratch */
static int      n_ders         = 0;
static bool     configured     = false;  /* true after P handshake */
static bool     configured_cfg = false;  /* true after CFG handshake */
static bool     skip_to_newline = false; /* BUF_OVERFLOW tail discard */
static float    cfg_q_ratio = Q_RATIO, cfg_u1 = U1_PU, ... cfg_u4 = U4_PU;
static char     buf[BUF_SIZE];           /* receive AND response buffer */
static uint16_t buf_idx = 0;
static uint32_t led_last_toggle_ms = 0;
```

#### Line 202–223: Global state — a masterclass in 2 KB SRAM budgeting

Every array is `static` (global), not a local — **design rationale:** a
`float[105]` on the stack is a 420-byte stack frame; on a 2 KB-SRAM MCU that risks
a stack/heap collision. Global static arrays are allocated once at a known address.
Three deliberate reuse tricks stretch the budget further:

1. **`vm_buf` doubles as the `CFG:` parse scratch** (first 6 slots, L204). Safe
   because the protocol guarantees `CFG` precedes the first `V:`, so `vm_buf` is
   "dead" (unused) when `CFG` parses into it. Zero extra SRAM for CFG parsing.
2. **`buf` is *both* the receive buffer and the response buffer** (L221, L454).
   Once a `V:` message is parsed into `vm_buf`, `buf` is free to be overwritten
   with the `Q:` response — avoiding a separate ~960-byte response buffer.
3. **`cfg_*` are 5 floats = 20 bytes** (L214), the runtime curve, initialised to
   the compile-time defaults so `compute_q` is always well-defined.

`configured` (after `P`) and `configured_cfg` (after `CFG`) are the two-stage
handshake flags that enforce `INIT → CFG → P` ordering (§8). `skip_to_newline` is
the overflow-recovery flag (§10). **This buffer-reuse engineering is why the
protocol ordering matters:** the memory optimisations are only safe *because*
`CFG` precedes `V` and `V` is fully parsed before the response is built.

---

## 6. Section C — compute_q: the mirror of 1A (L245–262)

```c
float compute_q(float vm_pu, float p_inst_mw) {
    float q_max = cfg_q_ratio * fabsf(p_inst_mw);
    if (vm_pu <= cfg_u1)      return q_max;
    else if (vm_pu < cfg_u2)  return q_max * (cfg_u2 - vm_pu) / (cfg_u2 - cfg_u1);
    else if (vm_pu <= cfg_u3) return 0.0f;
    else if (vm_pu < cfg_u4)  return -q_max * (vm_pu - cfg_u3) / (cfg_u4 - cfg_u3);
    else                      return -q_max;
}
```

#### Line 245–262: The Q(V) characteristic, on-device

**The exact mirror of 1A's `compute_setpoint`** (verified branch-by-branch in §3).
`q_max = cfg_q_ratio · |p|` (`fabsf` = float `abs`, guarding a negative dataset
value, exactly as 1A's `abs()`); then the five-segment piecewise curve using the
**CFG-supplied runtime params** `cfg_u1..cfg_u4` (not the compile-time `U1..U4`).
The comment L246–248 confirms the safety precondition: `ERR:CFG_INVALID` guarantees
`cfg_u1 < cfg_u2 < cfg_u3 < cfg_u4` at commit time, so both ramp denominators are
strictly positive — the device-side equivalent of 1A's monotonicity guard.

**The AVR-specific boundary rationale (L236–241).** The inclusive `<=` at the
saturation points (`cfg_u1`, `cfg_u3`) is not just style here — it is a
*performance* choice. A strict `<` at `U1` would force the ramp branch to run a
floating-point division even when `vm_pu == U1` (the result equals `q_max`
regardless — the curve is continuous, Doc 1A §7 continuity proof), and **the
ATmega328P has no hardware FPU**, so each division is a ~18-cycle software routine.
Inclusive `<=` routes exact-boundary voltages to the cheap constant branch. Same
*values* as 1A, same *reasoning* (avoid the ramp division at the boundary), with an
added hardware motivation.

**Units / worked example** (CFG params = canonical, `p = 0.5`): `q_max = 0.25·0.5 =
0.125`; `compute_q(1.025, 0.5) = −0.125·(1.025−1.01)/(1.04−1.01) = −0.0625 MVAr` —
identical to 1A's worked example, but computed in **float32**, which (with the 4dp
formatting, §9) is the source of the HIL/dry-run asymmetry.

---

## 7. Section D — parse_float_array (L285–299)

```c
int parse_float_array(char *data, float *out, int max_n) {
    int n = 0; char *ptr = data; char *end;
    while (n < max_n && *ptr != '\0') {
        out[n] = (float)strtod(ptr, &end);      // strtof on Uno R4
        if (end == ptr) break;                   // no token consumed → stop
        n++; ptr = end;
        if (*ptr == ',') ptr++;
    }
    return n;
}
```

#### Line 285–299: Comma-separated float parser, AVR-portable

Parses up to `max_n` comma-separated floats, returning the count actually parsed
(the caller checks it against the expected count). `strtod` advances `end` past the
token; `end == ptr` means nothing was consumed (a malformed token) → stop.

**Design rationale — `strtod`, not `strtof` (L282–283, L291).** `strtof` is
**absent from avr-libc** on the ATmega328P/Uno R3, so the R3 build uses `strtod`
(cast to float) — and since `double == float == 32-bit` on AVR, the two are
equivalent. The commented `strtof` line (L291) is the Uno R4 variant (the R4 has
real doubles and provides `strtof`). This is a real portability split between the
two boards. **Why not `sscanf`?** The docstring (L280–281) explains: `sscanf` on
AVR costs ~800 bytes of flash and gives no per-token error position; `strtod` in a
loop is leaner and pinpoints the failing token.

---

## 8. Section E — handle_message: the protocol handler (L315–497)

Dispatches a complete message by prefix. Each branch is the device-side responder
to a 1B message (§3). Walked message by message.

#### Line 318–338: INIT:<n> — announce the DER count

```c
    if (strncmp(msg, "INIT:", 5) == 0) {
        char *endptr; long n_long = strtol(msg + 5, &endptr, 10); int n = (int)n_long;
        if (endptr == msg + 5 || *endptr != '\0' || n <= 0 || n > MAX_DERS) {
            n_ders = 0; configured = false; Serial.println(F("ERR:INIT_RANGE")); return;
        }
        n_ders = n; configured = false; Serial.println(F("ACK:INIT")); return;
    }
```

Parses `n` with `strtol` + `endptr`. **Design rationale — `strtol`, not `atoi`
(L319–322):** `atoi("5xyz")` silently returns `5`, accepting a malformed message;
`strtol` with `endptr` lets the code *require* `endptr` to point at `'\0'`, so any
trailing non-digit is rejected. The range check (`0 < n ≤ MAX_DERS`) mirrors 1B's
board-capacity ceiling. `configured` is reset (a fresh INIT invalidates any prior
session). **`F(...)`** stores the string literal in flash, not SRAM — another
budget trick, used on every response. Replies `ACK:INIT` (1B expects this exact
string).

#### Line 341–381: CFG:<...> — receive the curve, validate, parse-then-commit

```c
        int n = parse_float_array(msg + 4, vm_buf, 6);         // ask for 6 to detect "too many"
        if (n != 5) { Serial.println(F("ERR:CFG_COUNT")); return; }
        for (i<5) if (isnan||isinf) { Serial.println(F("ERR:CFG_INVALID")); return; }
        if (vm_buf[0] <= 0 || vm_buf[0] > 1 || !(vm_buf[1]<vm_buf[2]<vm_buf[3]<vm_buf[4]))
            { Serial.println(F("ERR:CFG_INVALID")); return; }
        cfg_q_ratio = vm_buf[0]; cfg_u1..cfg_u4 = vm_buf[1..4]; configured_cfg = true;
        Serial.println(F("ACK:CFG"));
```

**The device-side twin of 1A's `set_qv_parameters` validation.** It parses into
`vm_buf` (the scratch-reuse trick, §5), asking for **6** floats (`max_n = 6`) so a
*sixth* value is caught as "too many" (`n != 5 → ERR:CFG_COUNT`) rather than
silently truncated. Then it enforces the **same rejection matrix as 1A**: all
finite, `0 < q_ratio ≤ 1`, strictly increasing breakpoints — exactly the checks 1A
runs on the host (Doc 1A §5). This is what makes 1B's "fail on the host where the
wire would fail" true: the firmware rejects precisely what the host rejects.
**Parse-then-commit (L348–350):** the `cfg_*` globals are written *only after* all
checks pass, so a malformed `CFG` cannot half-update the curve — the C mirror of
1A's atomic commit. **Note:** the firmware does **not** have 1A's
`PLAUSIBLE_PU_RANGE` typo check — that is deliberately host-only (Doc 1A §4, "catch
the typo on the host, where the message can be helpful").

#### Line 384–404: P:<...> — capacities, with ordering guards

```c
        if (n_ders <= 0) { Serial.println(F("ERR:P_BEFORE_INIT")); return; }
        if (!configured_cfg) { Serial.println(F("ERR:CFG_BEFORE_P")); return; }
        int n = parse_float_array(msg + 2, p_installed, n_ders);
        if (n != n_ders) { Serial.println(F("ERR:P_COUNT")); return; }
        configured = true; Serial.println(F("ACK:P"));
```

Enforces the **`INIT → CFG → P` ordering** with two guards: `P` before `INIT`
(`n_ders ≤ 0`) → `ERR:P_BEFORE_INIT`; `P` before `CFG` (`!configured_cfg`) →
`ERR:CFG_BEFORE_P`. **Design rationale (L390–392):** old RPi software that skips
`CFG` fails here with an *explicit, diagnosable* error rather than silently running
compile-time default parameters — the firmware refuses to run an unconfigured
curve. Parses exactly `n_ders` capacities into `p_installed`; `configured = true`
is the gate that lets `V:` proceed.

#### Line 407–433: V:<...> — the per-timestep request; parse and finiteness

```c
        if (!configured) { Serial.println(F("ERR:NOT_CONFIGURED")); return; }
        int n = parse_float_array(msg + 2, vm_buf, n_ders);
        if (n != n_ders) { Serial.println(F("ERR:V_COUNT")); return; }
        for (i<n_ders) if (isnan(vm_buf[i])||isinf(vm_buf[i])) { Serial.println(F("ERR:V_INVALID")); return; }
```

The hot path. Requires `configured` (full handshake done) → else `ERR:NOT_CONFIGURED`.
Parses exactly `n_ders` voltages into `vm_buf`; a count mismatch → `ERR:V_COUNT`.
Then the **finiteness check** rejects any `NaN`/`Inf` voltage → `ERR:V_INVALID`
(the deterministic error 1B raises `ArduinoProtocolError` on). **Design rationale
(L419–426):** a non-finite voltage would propagate through `compute_q` to a
non-finite Q, which would fail `float()` on the RPi and waste a retry; rejecting it
here is more informative and skips the `compute_q` call. This mirrors 1B's *own*
pre-transmit non-finite check (Doc 1B L779) — the check exists on **both** ends.

The Q computation and response are §9.

#### Line 489–496: END and the unknown-message fallback

```c
    if (strncmp(msg, "END", 3) == 0) { configured = false; n_ders = 0; return; }
    Serial.println(F("ERR:UNKNOWN"));
```

`END` (from 1B `close`, L568) clears the session state and sends **no response**
(1B does not wait for one). Any unrecognised message falls through to
**`ERR:UNKNOWN`** — the catch-all 1B treats as *transient* and retries (Doc 1B §6),
because an unrecognised prefix usually means a corrupted/phantom byte, not a
deterministic fault.

---

## 9. Section F — The Q-formatting quantisation leg (L459–484)

```c
        buf[pos++] = 'Q'; buf[pos++] = ':'; buf[pos] = '\0';
        for (int i = 0; i < n_ders; i++) {
            float q = compute_q(vm_buf[i], p_installed[i]);
            dtostrf(q, FLOAT_WIDTH, FLOAT_DEC, tmp);          // ← 4-decimal format
            char *t = tmp; while (*t == ' ') t++;              // strip leading pad spaces
            int len = strlen(t);
            if (pos + len + 2 >= BUF_SIZE) { Serial.println(F("ERR:BUF_OVERFLOW")); return; }
            memcpy(buf + pos, t, len); pos += len;
            if (i < n_ders - 1) buf[pos++] = ',';
        }
        buf[pos] = '\0'; Serial.println(buf);
```

#### Line 459–484: Build the Q response — the Q leg of the quantisation finding

**This loop is where the device-side quantisation happens.** For each DER,
`compute_q` returns a **float32** Q, and `dtostrf(q, FLOAT_WIDTH, FLOAT_DEC, tmp)`
formats it to **`FLOAT_DEC = 4` decimal places** of ASCII. That 4-decimal format is
the **Q leg** of the serial-protocol quantisation (Doc 1B §7): combined with the
RPi's 4-decimal **V leg** (1B L785) and the AVR's float32 recompute, it produces
the ~2.08×10⁻⁴ MVAr worst-case error ≈ 208× `SATURATION_TOL`. The dry-run path has
none of this (float64, no ASCII round trip).

**Code-to-physics / the format quantum.** `dtostrf` rounds to the nearest 10⁻⁴
MVAr; the worst-case rounding error is half that, 5×10⁻⁵ MVAr, which the Q(V) ramp
sensitivity (`dQ/dV`) does *not* amplify here (this is the Q value itself), but
which stacks with the V-leg error that *is* amplified by `dQ/dV`. **Worked
example:** `compute_q(1.025, 0.5) = −0.0625` → `dtostrf(−0.0625, 8, 4)` →
`"−0.0625"` (exact at 4 dp); a value like `−0.06253` → `"−0.0625"` (rounded, the
lost `3×10⁻⁵` is the quantum).

**Design rationale — buffer reuse and the overflow guard.** `buf` (the receive
buffer) is reused as the response buffer (§5) — safe because `vm_buf` already holds
the parsed voltages. `dtostrf` pads to `FLOAT_WIDTH = 8` with leading spaces, which
are **stripped** (L468–469) for compact output (`"0.4800"`, not `"  0.4800"`) that
keeps the response inside `BUF_SIZE`. The guard `pos + len + 2 >= BUF_SIZE`
(L472) catches the edge case where `MAX_DERS` is raised without raising `BUF_SIZE`,
returning `ERR:BUF_OVERFLOW` (safe, diagnosable) rather than overrunning the
buffer. `Serial.println` appends `\r\n`; the RPi's `readline` stops at `\n`, `.strip()`
removes the `\r` (Doc 1B).

---

## 10. Section G — setup & loop (L504–578)

#### Line 504–515: setup — serial, LED, the READY signal

```c
void setup() {
    Serial.begin(BAUD_RATE); pinMode(LED_PIN, OUTPUT); digitalWrite(LED_PIN, HIGH);
    Serial.println(F("READY"));
}
```

Starts the UART at `BAUD_RATE` (must match 1B), lights the LED, and prints
`"READY"`. **Design rationale — the READY handshake with 1B.** `"READY"` signals
that `setup()` finished and the loop is starting. The RPi **discards** it —
`ArduinoSerialInterface.open()` flushes the input buffer after the reset delay
(1B L563), and `_read_ack` skips non-`ACK`/`ERR` lines (1B L705). So `READY` is a
*development aid* (visible on the Serial Monitor), harmless on the live path. This
is the device end of the reset-delay dance in 1B §6 (`open` waits
`ARDUINO_RESET_DELAY_S`, then flushes exactly this boot output).

#### Line 517–567: loop — non-blocking accumulation and overflow recovery

```c
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n') {
            if (skip_to_newline) { skip_to_newline = false; buf_idx = 0; }
            else { buf[buf_idx] = '\0'; if (buf_idx > 0) handle_message(buf); buf_idx = 0; }
        } else if (c == '\r') { /* discard Windows CR */ }
        else if (skip_to_newline) { /* discard overflow tail */ }
        else if (buf_idx < BUF_SIZE - 1) { buf[buf_idx++] = c; }
        else { skip_to_newline = true; buf_idx = 0; Serial.println(F("ERR:BUF_OVERFLOW")); }
    }
```

**Non-blocking character accumulation.** One byte per `Serial.available()`
iteration into `buf`; on `\n`, dispatch the complete message and reset. **Design
rationale — why non-blocking (L522–526):** a blocking read (`readStringUntil`) would
stall `loop()` for the whole timeout with no data, freezing the MCU; non-blocking
returns immediately when the UART buffer is empty, so the LED heartbeat and future
work keep running. **`\r` discard (L528–530):** Windows terminals send `\r\n`;
dropping `\r` prevents dispatching a second empty message. **The overflow/tail
guard (`skip_to_newline`, L532–538):** if `buf` fills without a `\n` (runaway
message), it resets, sends `ERR:BUF_OVERFLOW`, and sets `skip_to_newline` — then
*discards every byte until the next `\n`*. Without this, the tail of the overflowed
message (bytes after the overflow) would be buffered and dispatched as a spurious
second message → phantom `ERR:UNKNOWN`. This is the device-side robustness that
lets 1B's truncation/`ERR:UNKNOWN` retry logic (Doc 1B §6) resynchronise cleanly.

#### Line 568–577: LED heartbeat

```c
    uint32_t now = millis();
    if (configured) {
        if (now - led_last_toggle_ms >= 500UL) { digitalWrite(LED_PIN, !digitalRead(LED_PIN)); led_last_toggle_ms = now; }
    } else { digitalWrite(LED_PIN, HIGH); led_last_toggle_ms = now; }
```

A non-blocking status indicator: **solid LED when idle/unconfigured, blinking at
1 Hz (500 ms toggle) once `configured`.** **Design rationale:** uses `millis()`
elapsed-time comparison, not `delay()`, so it never blocks the serial loop. It
gives an at-a-glance signal that the handshake completed and the board is servicing
`V:` requests — invaluable when debugging a HIL run at the bench.

---

## 11. Coverage checklist

Every named entity in `volt_var_arduino.ino`, each confirmed documented.

| Entity | Lines | Kind | Documented | Mirror / where-used |
|--------|-------|------|:----------:|:-------------------:|
| includes | 163–165 | preprocessor | ✅ | ✅ |
| `U1_PU`..`U4_PU`, `Q_RATIO` | 178–182 | `#define` | ✅ | ✅ (mirror 1A) |
| `BAUD_RATE`..`LED_PIN` | 188–196 | `#define` | ✅ | ✅ (`FLOAT_DEC` = Q leg) |
| `p_installed`, `vm_buf`, `n_ders`, `configured`, `configured_cfg`, `skip_to_newline` | 202–209 | globals | ✅ | ✅ |
| `cfg_q_ratio`, `cfg_u1`..`cfg_u4` | 215–219 | globals | ✅ | ✅ (CFG runtime curve) |
| `buf`, `buf_idx`, `led_last_toggle_ms` | 221–223 | globals | ✅ | ✅ |
| `compute_q` | 245–262 | function | ✅ | ✅ (mirror of 1A `compute_setpoint`) |
| `parse_float_array` | 285–299 | function | ✅ | ✅ (called by CFG/P/V handlers) |
| `handle_message` | 315–497 | function | ✅ | ✅ (mirror of 1B device end) |
| — INIT / CFG / P / V / END / unknown | 318–496 | branches | ✅ | ✅ |
| `setup` | 504–515 | function | ✅ | ✅ (READY ↔ 1B open) |
| `loop` | 517–578 | function | ✅ | ✅ |

**Also covered (not standalone entities):** the SRAM-budget buffer reuse
(`vm_buf` as CFG scratch, `buf` as response), the `strtol`-vs-`atoi` and
`strtod`-vs-`strtof` rationales, the parse-then-commit CFG validation, the `F(...)`
flash-string trick, the `dtostrf` 4-decimal Q leg, the READY handshake, the
non-blocking loop and `skip_to_newline` overflow recovery, and the LED heartbeat.

**Notes (not bugs):** `Q_RATIO 0.25f /*was 0.48*/` (L182) matches the in-force
value; `strtof` line commented for the Uno R4 variant (L291). The firmware
intentionally omits 1A's `PLAUSIBLE_PU_RANGE` typo check (host-only by design).

**Mirror invariant confirmed:** `compute_q` matches `QVCharacteristic.compute_setpoint`
(1A) branch-for-branch; the firmware's `ACK:`/`ERR:` set matches 1B's expected
responses and exception taxonomy exactly.

---

*End of Document 5 — completing the control-core + device set (1A, 1B, 1C, 5) and
the coordinator set (2, 3A, 3B, 4). Next per the queue: Document 6A — Scenario
Runners (result container + baseline: `scenario_result.py`, `scenario_1_baseline.py`).*
