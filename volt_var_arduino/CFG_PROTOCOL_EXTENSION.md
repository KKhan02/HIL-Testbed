# CFG Protocol Extension — Runtime Q(V) Characteristic Configuration

**Files changed:** `volt_var_arduino.ino`, `volt_var_controller.py` (only these two — no protected core files, no other modules).
**Dry-run path:** untouched. **Plugin systems:** unaffected (see §7).

---

## 1. What changed and why

Previously the five Q(V) characteristic parameters (`Q_RATIO`, `U1_PU`, `U2_PU`, `U3_PU`, `U4_PU`) were compile-time constants duplicated in `volt_var_controller.py` and `volt_var_arduino.ino`, protected only by the documented invariant "must be identical in both files" (Knowledge Base §9.1). Every parameter change (e.g. the Q_RATIO 0.48 → 0.25 revision) required a firmware re-flash and carried silent-mismatch risk: nothing at runtime verified that the two copies agreed.

The new `CFG:` protocol message pushes the parameters from the Raspberry Pi to the Arduino at session startup. `volt_var_controller.py` is now the **single source of truth for the parameter values**; the firmware `#define`s are boot defaults that are never used in a correctly-sequenced session, because the firmware **rejects `P:` until a valid `CFG:` has been received**.

Note the invariant is eliminated only for the parameter *values*. The curve *shape* (five-segment piecewise linear with deadband) is still implemented independently in `compute_q()` (C) and `QVCharacteristic.compute_setpoint()` (Python); structural changes still require editing both.

## 2. Protocol (new sequence)

```
RPi  -> Arduino :  INIT:<n>\n                          n = number of DERs
Arduino -> RPi  :  ACK:INIT\n
RPi  -> Arduino :  CFG:<q_ratio>,<u1>,<u2>,<u3>,<u4>\n   <-- NEW
Arduino -> RPi  :  ACK:CFG\n                             <-- NEW
RPi  -> Arduino :  P:<p1>,<p2>,...\n
Arduino -> RPi  :  ACK:P\n
--- per timestep: V: / Q: unchanged ---
--- session end: END unchanged ---
```

Value order is fixed: `q_ratio, u1, u2, u3, u4`, formatted at 4 decimal places (matching the `V:` message's precision convention). Example as actually sent with current constants:

```
CFG:0.2500,0.9600,0.9900,1.0100,1.0400
```

New error responses:

| Code | Meaning |
|---|---|
| `ERR:CFG_COUNT` | CFG body did not contain exactly 5 parseable floats (4 or fewer, or 6+) |
| `ERR:CFG_INVALID` | non-finite value, `q_ratio ∉ (0, 1]`, or breakpoints not strictly increasing `u1<u2<u3<u4` |
| `ERR:CFG_BEFORE_P` | `P:` received while `configured_cfg == false` |

## 3. Firmware changes (`volt_var_arduino.ino`)

**New globals** (Global state section):

```c
static bool  configured_cfg = false;
static float cfg_q_ratio = Q_RATIO;   // 0.25f
static float cfg_u1      = U1_PU;     // 0.96f
static float cfg_u2      = U2_PU;     // 0.99f
static float cfg_u3      = U3_PU;     // 1.01f
static float cfg_u4      = U4_PU;     // 1.04f
```

Initialising to the compile-time constants means `compute_q()` is always well-defined, even if the guards were ever bypassed during bench testing over a serial monitor.

**CFG handler** in `handle_message()`, placed between the `INIT:` and `P:` blocks. Its logic:

1. **Parse-then-commit, two phases.** The five floats are parsed into scratch memory first; the `cfg_*` globals are written only after *all* checks pass. A malformed CFG therefore can never leave the characteristic half-updated (e.g. new `u1` with old `u2`, which could invert a ramp).
2. **Scratch memory = `vm_buf[]` reuse, zero cost.** `vm_buf` is dead until the first `V:` message, which the protocol guarantees comes after CFG, so its first six slots serve as the parse buffer. No new stack frame, no new SRAM — the same pattern the sketch already uses when it reuses `buf[]` as the Q-response buffer.
3. **Count check via `max_n = 6`.** `parse_float_array(msg + 4, vm_buf, 6)` parses *up to six* values. A return of exactly 5 is required: fewer means truncated input; a return of 6 means the sender supplied too many values — both answer `ERR:CFG_COUNT`. Parsing with `max_n = 5` would have silently ignored a sixth value.
4. **Validity check** (`ERR:CFG_INVALID`): all five values finite; `0 < q_ratio ≤ 1`; `u1 < u2 < u3 < u4` strictly. Strict monotonicity guarantees both ramp denominators `(u2−u1)` and `(u4−u3)` in `compute_q()` are positive — a zero denominator would be a division by zero on a chip with software-float division and no exception mechanism.
5. On success: commit globals, `configured_cfg = true`, respond `ACK:CFG` (via `F()` — all four new string literals use `F()`, so they live in flash and cost **0 bytes SRAM**).

**P: guard.** Immediately after the existing `ERR:P_BEFORE_INIT` check:

```c
if (!configured_cfg) { Serial.println(F("ERR:CFG_BEFORE_P")); return; }
```

Ordering of the two guards mirrors the protocol ordering (INIT before CFG before P), so the error reported is always the *earliest* missing step.

**`compute_q()`** now reads `cfg_q_ratio, cfg_u1..cfg_u4` instead of the `#define`s. The branch structure, inclusive boundaries (`<=` at U1/U3 saturation points — see the pre-existing comment on avoiding needless software divisions), and sign convention are byte-for-byte the same logic as before.

**Deliberately unchanged:** `INIT:` does **not** reset `configured_cfg`, and `END` does not either. Rationale: the characteristic is session-independent physics, unlike `n_ders`/`p_installed` which are per-batch. This matters for `exchange_batched()` (>105 DERs): it calls `configure()` per batch, which re-sends CFG anyway, so behaviour is correct either way — but not resetting means a mid-run `INIT` (batch switch) can never strand the firmware in a P-rejecting state if a CFG line were ever corrupted. See §8 for the open question on `END`.

## 4. SRAM budget verification (ATmega328P, 2 KB)

Added static footprint: 5 × `float` (4 B) + 1 × `bool` (1 B) = **+21 bytes**. Nothing else: the CFG handler allocates no local arrays (vm_buf reuse), and all new string literals use `F()`.

| Item | Before | After |
|---|---|---|
| Declared globals subtotal | ~1716 B | ~1737 B |
| Hidden consumers (UART buffers, runtime) | ~202 B | ~202 B |
| **Total** | **1918 B** | **~1939 B** |
| Remaining for stack | 130 B | **109 B** |
| Estimated stack usage | ~80 B | ~80 B (unchanged — no new frames) |
| Free margin | ~50 B | **~29 B** |

This stays under the file's own documented ceiling ("globals under ~1968 bytes — leaves ~80 bytes for stack, absolute minimum"). The header comment has been updated with these numbers and now also names the cheapest reclaim path if the compiler-measured figure lands worse than estimated: `BUF_SIZE` 870 → 850 (still above the 842-byte typical worst-case Q response) recovers 20 bytes, and Option A (MAX_DERS reduction) remains documented as the bigger lever. **Action for you:** after compiling in the Arduino IDE, confirm the reported "Global variables use ~1939 bytes" figure and adjust the comment to the exact compiler value, as was done for the original 1918.

## 5. Python changes (`volt_var_controller.py`)

All inside `ArduinoSerialInterface.configure()` plus documentation updates (module docstring protocol section, architecture note, constants-block comment — the "edit both files" guidance is now obsolete for values and has been corrected).

The CFG step is inserted between the INIT ack and the P send:

```python
cfg_msg = f"CFG:{Q_RATIO:.4f},{U1_PU:.4f},{U2_PU:.4f},{U3_PU:.4f},{U4_PU:.4f}\n"
self._ser.write(cfg_msg.encode())
try:
    ack = self._read_ack("CFG")
except SerialConfigError as exc:
    raise SerialConfigError("CFG protocol step failed: ...") from exc
if ack != "ACK:CFG":
    raise SerialConfigError("CFG protocol step failed -- expected 'ACK:CFG' ...")
```

Design points:

- **Single source of truth.** The values are the module-level `Q_RATIO, U1_PU..U4_PU` constants — the same objects `QVCharacteristic` (dry-run), `scenario_4_volt_var.py`'s `_build_dynamics()`, and `SensitivityCoordinator` sizing already consume. Editing the constant once now changes dry-run, coordination sizing, *and* hardware behaviour consistently, with no re-flash.
- **Timeout/retry consistency.** The CFG ack is read through the existing `_read_ack()` helper under the same `ACK_TIMEOUT_S = 5 s` timeout that the surrounding `try/finally` already installs for INIT and P. `_read_ack()`'s 5-line skip loop provides the same boot-noise resilience and bounded-wait retry the INIT handshake has — no new mechanism, no divergent behaviour.
- **Backward compatibility / clear failure identification.** Old firmware answers `ERR:UNKNOWN` to CFG (its unknown-prefix fallback); a wedged board stays silent (5 timed-out reads). Both surface from `_read_ack()` as `SerialConfigError` and are re-raised as `SerialConfigError` with augmented context: the message (a) names the CFG protocol step explicitly, (b) includes the exact message sent and the firmware's response, and (c) states the likely cause (firmware predating CFG) and the fix (re-flash). The original exception is chained via `from exc`.

  **Exception taxonomy (deliberate):** `SerialConfigError` is used for *all* handshake-phase failures — INIT, CFG, and P — while `ArduinoProtocolError` remains reserved exclusively for the per-timestep `exchange()`/`exchange_batched()` V→Q loop. CFG is a handshake message sent from inside `configure()`, so it follows the handshake family. The failed step is identified by message text, not by type:

  ```python
  try:
      iface.configure(n, p)
  except SerialConfigError as e:
      # str(e) begins "INIT handshake ...", "CFG protocol step failed ...",
      # or "P handshake ..." — the CFG variant names the likely firmware
      # version mismatch and the re-flash remedy.
      ...
  ```

  This also means `run_timestep()`'s existing `except ArduinoProtocolError` / `except SerialTimeoutError` fallback-to-Q=0 handling is provably unaffected: no handshake exception can leak into the per-timestep catch clauses.

## 6. Why this design (logic summary)

The controlling constraint is the **2 KB SRAM budget with a pre-existing ~50-byte margin**. Every choice flows from spending the absolute minimum:

- Runtime parameters as 5 plain floats (+20 B) rather than any structured store.
- `vm_buf` reuse for parsing → zero new stack/SRAM, mirroring the sketch's established `buf[]`-reuse idiom.
- `F()` on all new literals → zero SRAM for strings (four new literals ≈ 60 bytes flash instead).
- Parse-then-commit with validation, because on hardware a bad characteristic doesn't throw — it silently mis-controls a physical grid model for an entire annual run. `ERR:CFG_INVALID` converts that into an immediate, diagnosable startup failure, consistent with the framework's existing philosophy (`ERR:V_INVALID`, the strtol endptr check on INIT, the Python-side NaN pre-checks).
- The `P:` gate makes protocol-version mismatch in the *other* direction (new firmware, old Python) equally loud: old Python fails fast with `ERR:CFG_BEFORE_P` at configure time instead of silently running whatever defaults are flashed.

Compatibility matrix:

| RPi software | Firmware | Outcome |
|---|---|---|
| new | new | Normal operation; RPi constants govern the curve |
| new | old | `SerialConfigError` at configure(), message names the CFG step |
| old | new | `SerialConfigError` at configure() (firmware answers `ERR:CFG_BEFORE_P` to `P:`, `_read_ack` surfaces it) |
| old | old | Unchanged legacy behaviour |

## 7. Interaction with the plugin subsystems

- **Controller plugins** (`custom_controller.py` / `plugin_runner.py`): unaffected. They instantiate `VoltVarController(..., dry_run=True)` purely as a clamping/capacity backstop; `VoltVarController.configure()` returns immediately on the `self._dry` guard, so `ArduinoSerialInterface.configure()` — and hence CFG — is never reached. Verified against the current `custom_controller.py` (line ~248).
- **Network plugins** (`network_plugin.py`): unaffected — they operate entirely on net/profiles construction and `plugin_meta`; no serial involvement. A plugin network run under Scenario 4 with hardware flows through the same `ctrl.configure()` → `iface.configure()` path and gets CFG automatically.
- **Scenario 4 hardware path** (`scenario_4_volt_var.py`): no changes needed. `ctrl.configure()` at line ~239 now transparently performs INIT → CFG → P. `_build_dynamics()` and the coordination Q_max sizing already read the Python `Q_RATIO`, so hardware and coordination now provably use the same value.
- **`exchange_batched()` (>105 DERs):** each per-batch `configure()` call re-sends CFG with identical values — idempotent, ~40 extra bytes on the wire per batch (~3.5 ms at 115200 baud), negligible against the documented ~390 ms two-batch cycle.

## 8. Verification performed

- `volt_var_controller.py`: `py_compile` clean; 6 mock-serial tests — message sequence and exact CFG payload (`CFG:0.2500,0.9600,0.9900,1.0100,1.0400`), old-firmware `ERR:UNKNOWN` → `SerialConfigError` naming the step, silent-firmware timeout → same, wrong-ACK → same, exception catchability, boot-noise skip on the CFG ack.
- `volt_var_arduino.ino`: compiled on host under `g++ -Wall -Wextra` with an Arduino shim (zero warnings); 8 behavioral tests — `ERR:CFG_BEFORE_P` gating, full INIT→CFG→P acceptance, global commit values, `compute_q()` tracking two different CFG parameter sets at saturation/ramp/deadband points, `ERR:CFG_COUNT` for 4 and 6 values with no partial commit, the full `ERR:CFG_INVALID` matrix (non-monotonic, q_ratio 0, q_ratio 1.5, NaN) with no partial commit, `ERR:UNKNOWN` preserved, and repeat INIT→CFG→P (the batched-exchange pattern).
- Host compilation does not verify AVR SRAM: **compile in the Arduino IDE and check the reported globals figure against §4 before flashing.**

## 9. Improvements included beyond the literal spec — confirm or I will strip them

Your spec said "parses five comma-separated floats ... sets configured_cfg = true, responds ACK:CFG". Three robustness additions are in the delivered code, each flagged here for your confirmation:

1. **`ERR:CFG_COUNT`** (count ≠ 5 detection, including a 6th-value check). Without it, `CFG:0.25,0.96` would commit two values and leave three stale — the exact half-updated-state hazard. Strongly recommend keeping.
2. **`ERR:CFG_INVALID`** (finiteness, `q_ratio ∈ (0,1]`, strict `u1<u2<u3<u4`). Prevents division-by-zero in `compute_q()` and silent physical mis-control from a corrupted line. Recommend keeping; if you consider `q_ratio > 1` legitimate for some future oversized-inverter study, say so and I'll relax that one bound (monotonicity must stay).
3. **Two-phase parse-then-commit via `vm_buf` scratch** (implementation detail enabling 1–2 at zero SRAM cost).

Resolved decisions (confirmed):

- **`END` untouched** — `configured_cfg` is not reset on END or INIT. The characteristic is session-independent physics, not per-batch state; the new Python re-sends CFG every `configure()` anyway.
- **No CFG readback echo** — consistent with the protocol's existing risk tolerance for `P:` (also unverified), and the current ~29-byte SRAM margin does not justify spending anything on a marginal benefit.
- **Exception taxonomy** — CFG failures raise `SerialConfigError`, matching the handshake-phase family (INIT/P), with `ArduinoProtocolError` reserved for the per-timestep exchange loop. The backward-compatibility requirement (catchable exception, message identifying the failed step) is met through the message text.

Still available on request: optional `cfg` keyword arguments on `configure()` (defaulting to the module constants) for future Q(V) parameter sweeps over hardware — purely additive.
