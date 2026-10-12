# TASK 4 — DOCUMENT 6C
## Scenario Runner — SVC
### `scenario_3_svc.py` — centralised static-var-compensator voltage regulation (642 lines, checkpointing)
### Complete line-by-line walkthrough

> **Source re-verified (project tree):** `scenario_3_svc.py` is now **644 lines**
> (+2 vs the upload this doc was written against); structure unchanged — internal
> line numbers below read ~+2. Content is current.
>
> **(prior) Source verified:** read against the **updated** `scenario_3_svc.py` (642 lines,
> with the checkpointing/resume additions) before writing. If the file is edited,
> re-verify line numbers.
>
> **Read alongside:** Master Index §5–§8; **6A-i/6A-ii** (`adapt_profiles`,
> `TimestepRecord` — note the SVC block `svc_q_mvar`/`svc_saturated` — and
> `ScenarioResult`); **6A-iii §4** (declarative-vs-manual); **6B-ii** (OLTC — the
> other manual actuator scenario, for contrast); **Document 1A** (the per-DER Q(V)
> curve, contrasted with this central droop).

---

## TABLE OF CONTENTS

1. [Overview — the centralised SVC](#1-overview)
2. [File logic flow — pseudocode](#2-file-logic-flow--pseudocode)
3. [Call-site map](#3-call-site-map)
4. [Section A — Constants (L111–114)](#4-section-a--constants-l111114)
5. [Section B — SVC bus selection (L121–290)](#5-section-b--svc-bus-selection-l121290)
6. [Section C — Sizing & the droop law (L293–352)](#6-section-c--sizing--the-droop-law-l293352)
7. [Section D — run_scenario_3: setup, resume, SVC creation (L363–455)](#7-section-d--run_scenario_3-setup-resume-svc-creation-l363455)
8. [Section E — The droop control loop (L457–594)](#8-section-e--the-droop-control-loop-l457594)
9. [Section F — Cleanup, result & summary (L596–642)](#9-section-f--cleanup-result--summary-l596642)
10. [Findings and flags](#10-findings-and-flags)
11. [Coverage checklist](#11-coverage-checklist)

---

## 1. Overview

Scenario 3 benchmarks a **Static Var Compensator (SVC)** — a *single, centralised*
reactive-power device at one MV bus, regulating voltage by a droop characteristic.
It is the third comparison scheme, and its defining contrast with the others is
**centralisation**:

| Scheme | Actuator | Location | Control law |
|--------|----------|----------|-------------|
| OLTC (6B) | transformer tap | substation | discrete deadband step |
| **SVC (this doc)** | **one reactive injection** | **one selected MV bus** | **continuous V-Q droop** |
| Volt-Var (6D) | per-DER reactive Q | every DER bus | distributed per-DER Q(V) |

Three ideas structure the file:

1. **Where to put it** (§5) — the SVC sits at *one* bus chosen once, by a 3-tier
   selection (stress analysis → first-violation scan → remote-bus fallback).
2. **How it acts** (§6) — a deadbanded proportional **V-Q droop**: reactive output
   proportional to the voltage error beyond a deadband, saturating at `±Q_MAX`.
3. **How it's modelled** (§7, §9) — as a *temporary* pandapower sgen (`type="SVC"`)
   created at setup and **removed in a `finally` block**, so the caller's network is
   returned clean even if the run throws.

**Stateless — a clean resume.** Unlike OLTC (tap position) and Volt-Var (PT1/ramp
state), the SVC droop is a *pure function of the current bus voltage* — it carries
no memory between timesteps. So the checkpoint resume needs **no state seeding**
(contrast 6B-ii's tap re-seed): it simply continues from `start_t`, re-deriving the
fixed `svc_bus`/`Q_MAX` for the result metadata. This is a genuine simplification,
not an omission.

---

## 2. File logic flow — pseudocode

```
CONSTANTS: voltage target (1.00), deadband half-width (±0.01 pu),
           full-action error (0.04 pu → Q_MAX), Q_MAX ratio (0.20 × Σ trafo MVA)

BUS SELECTION:
  _mv_kv(net)               → the dominant MV voltage level (modal vn_kv, 1–66 kV)
  _eligible_mv_buses(net)   → in-service MV buses, excluding HV/LV and the slack
  _select_svc_bus(net,…)    → 3-tier: (1) weakest bus under overvoltage stress,
                              (2) worst bus at first violation in a 7-day pre-scan,
                              (3) last-resort remote bus
SIZING & CONTROL:
  _compute_svc_params(net)  → Q_MAX = 0.20·Σ sn_mva ; k_q = Q_MAX/(0.04−0.01)
  _droop_q(vm, Q_MAX, k_q)  → deadbanded proportional droop → (q_cmd, saturated)

FUNCTION run_scenario_3(net, profiles, …, enable_checkpointing, live_csv_rewrite_fn):
    adapt profiles; announce start
    RESUME: resumed = get_resume_records("svc"); start_t = last.t+1
        if checkpoint covers all steps → re-derive svc_bus/Q_MAX, aggregate, return
    reset controllers/results; compute Q_MAX/k_q; select svc_bus; CREATE the SVC sgen
    TRY:
        FOR t in range(start_t, T):
            write load + DER profiles (DER q=0); SVC q=0
            pre-control PF (SVC idle)
              diverged → record (SVC held 0); continue
            read SVC-bus voltage → droop → (q_cmd, saturated)
            apply q_cmd; post-control PF only if q_cmd ≠ 0 (else reuse pre-PF)
            build record (svc_q_mvar, svc_saturated, energy balance); publish; append
            every 96 steps: rewrite live CSV; log progress
    FINALLY:
        remove the SVC sgen + reset results  (net returned clean, even on exception)
    aggregate → ScenarioResult(scenario_id="svc", svc_bus, svc_q_max); log SVC summary
```

---

## 3. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `run_scenario_3` | function | `benchmark_runner` `SCENARIO_REGISTRY[3]` (Doc 9). |
| `_mv_kv`, `_eligible_mv_buses` | helpers | `_select_svc_bus` chain. |
| `_select_svc_bus_stress`, `_select_svc_bus_first_violation` | helpers | `_select_svc_bus` (the two primary tiers). |
| `_select_svc_bus` | helper | `run_scenario_3` setup + the skip-if-complete branch. |
| `_compute_svc_params` | helper | `run_scenario_3` setup + skip-if-complete. |
| `_droop_q` | helper | `run_scenario_3` loop [3]. |
| `_empty_series` | helper | non-converged records. |
| `apply_overvoltage_stress` | optional import | `_select_svc_bus_stress` (guarded by `_HAS_STRESS`). |
| `adapt_profiles`, `TimestepRecord`, `ScenarioResult.from_records` | infra | 6A-i / 6A-ii. |
| `publish_fn.*`, `live_csv_rewrite_fn` | hooks | resume/streaming (Doc 9/10). |

---

## 4. Section A — Constants (L111–114)

```python
SVC_V_TARGET:       float = 1.00   # voltage setpoint (pu)
SVC_DEADBAND:       float = 0.01   # deadband half-width (pu) → active ±0.01
SVC_FULL_ACTION_DV: float = 0.04   # voltage error at which Q_MAX is reached
SVC_Q_MAX_RATIO:    float = 0.20   # Q_MAX = 0.20 × Σ net.trafo.sn_mva
```

#### Line 111–114: The four droop parameters

They fully specify the V-Q characteristic (§6). `SVC_V_TARGET = 1.00` (regulate to
nominal); `SVC_DEADBAND = 0.01` (no action within ±1% of target — avoids hunting
around small deviations); `SVC_FULL_ACTION_DV = 0.04` (the error at which the droop
demands full `Q_MAX` — so the proportional band runs from ±0.01 to ±0.04, a 0.03 pu
span); `SVC_Q_MAX_RATIO = 0.20` (the device is sized at 20% of total transformer
apparent power). **Design rationale — the three-band structure.** Like the OLTC
(6B-i §4), the SVC deadband (±0.01) sits *inside* the violation band (±0.05); the
SVC regulates preventively. But note it targets a *tighter* band than the OLTC
(±0.01 vs ±0.02) and acts *continuously* rather than in discrete steps.

---

## 5. Section B — SVC bus selection (L121–290)

The SVC is a single device, so *where* it goes matters enormously — a well-placed
SVC controls the weak part of the feeder; a badly-placed one wastes its capacity.

#### Line 121–159: _mv_kv & _eligible_mv_buses — the candidate set

```python
def _mv_kv(net) -> float:
    vn = net.bus["vn_kv"]; mv_mask = (vn > 1.0) & (vn < 66.0)
    mv_levels = vn[mv_mask] ... return float(mv_levels.mode().iloc[0])

def _eligible_mv_buses(net) -> pd.Index:
    slack_buses = set(net.ext_grid["bus"].values); target_kv = _mv_kv(net)
    mask = (net.bus["vn_kv"] == target_kv) & (net.bus["in_service"] == True) & (~net.bus.index.isin(slack_buses))
    return net.bus.index[mask]
```

`_mv_kv` finds the **dominant** MV voltage level as the *modal* `vn_kv` in the
1–66 kV window (falling back to <66 kV, then any bus). `_eligible_mv_buses` returns
in-service buses at that level, **excluding the slack bus** — the docstring is
explicit: injecting reactive power at the infinite busbar is physically meaningless
(the slack absorbs anything). **Design rationale:** restricting to one MV level and
excluding HV/LV/slack gives a clean candidate set of buses where an SVC could
actually help.

#### Line 162–195: _select_svc_bus_stress — the weakest bus under stress

```python
    if not _HAS_STRESS: return None
    probe_net = copy.deepcopy(net)
    try: apply_overvoltage_stress(probe_net); pp.runpp(probe_net, voltage_depend_loads=False)
    except Exception: return None
    vm = probe_net.res_bus.loc[mv_buses, "vm_pu"]; selected = int(vm.idxmin())
    return selected
```

**Tier 1 — physics-based placement.** On a deep copy, apply an overvoltage stress
scenario (`apply_overvoltage_stress`, an optional module guarded by `_HAS_STRESS`),
solve, and pick the MV bus with the **lowest** voltage — the electrically weakest /
highest-impedance bus, where reactive support does the most good. Returns `None` if
the stress module is absent or the solve diverges (→ fall through to Tier 2).
**Design rationale — deep copy:** the probe must not perturb the working net.

#### Line 198–243: _select_svc_bus_first_violation — first-violation pre-scan

```python
    scan_steps = min(len(ap.times), 7 * 24 * 6)   # ≤ 7 days
    probe_net = copy.deepcopy(net)
    for t in range(scan_steps):
        ... write profiles; runpp ...
        violated = vm[(vm > v_max + VOLTAGE_EPSILON) | (vm < v_min - VOLTAGE_EPSILON)]
        if not violated.empty:
            worst = int((violated - 1.0).abs().idxmax()); return worst
    return None
```

**Tier 2 — profile-driven placement.** A lightweight forward pass over the first
**7 days** of the actual profiles; at the *first* timestep with any MV violation,
place the SVC at the **worst** bus (largest absolute deviation from 1.0). Returns
`None` if no violation appears in the window (→ Tier 3). **Design rationale:** if the
stress module is unavailable, the profiles themselves reveal where trouble first
appears; capping at 7 days keeps the pre-scan cheap. Uses the same `±VOLTAGE_EPSILON`
threshold as everywhere else.

#### Line 246–290: _select_svc_bus — the 3-tier dispatcher

```python
    mv_buses = _eligible_mv_buses(net)
    if mv_buses.empty: raise ValueError("no eligible MV bus found ...")
    bus = _select_svc_bus_stress(net, mv_buses);            
    if bus is not None: return bus
    bus = _select_svc_bus_first_violation(net, ap, mv_buses, v_max, v_min)
    if bus is not None: return bus
    bus = int(mv_buses[-1]); logger.warning("using last-resort bus %d ...")   # Tier 3
    return bus
```

Orchestrates the three tiers in priority order: stress → first-violation →
**last-resort** (the last bus in the MV set, typically the most remote in
pandapower's ordering). Raises if there is no eligible MV bus at all. **Design
rationale — graceful degradation:** each tier is a fallback for the previous one's
unavailability, so the SVC always gets *a* defensible placement, with a warning when
it lands on the heuristic last resort. **Where used:** `run_scenario_3` setup (and,
on the skip-if-complete resume path, purely to populate `svc_bus` in the result).

---

## 6. Section C — Sizing & the droop law (L293–352)

#### Line 293–321: _compute_svc_params — Q_MAX and the droop gain

```python
    if net.trafo.empty: raise ValueError("net.trafo is empty — cannot derive Q_MAX ...")
    sn_total = float(net.trafo["sn_mva"].sum())
    q_max = SVC_Q_MAX_RATIO * sn_total
    k_q   = q_max / (SVC_FULL_ACTION_DV - SVC_DEADBAND)
    return q_max, k_q
```

**Sizing.** `Q_MAX = 0.20 · Σ(trafo sn_mva)` — the SVC's reactive capacity is 20% of
the substation transformer rating, a realistic central-compensator size. The **droop
gain** `k_q = Q_MAX / (FULL_ACTION_DV − DEADBAND) = Q_MAX / 0.03` is the slope that
takes the output from 0 (at the deadband edge, |error| = 0.01) to `Q_MAX` (at full
action, |error| = 0.04). **Units:** `Q_MAX` MVAr; `k_q` MVAr/pu. Raises if there is
no transformer (no capacity reference). **Worked example:** `Σ sn_mva = 25` → `Q_MAX
= 5.0` MVAr, `k_q = 5.0/0.03 = 166.7` MVAr/pu.

#### Line 324–352: _droop_q — the deadbanded proportional V-Q law

```python
    error = SVC_V_TARGET - vm_pu
    if abs(error) <= SVC_DEADBAND: return 0.0, False
    if error > SVC_DEADBAND: q_raw = k_q * (error - SVC_DEADBAND)
    else:                    q_raw = k_q * (error + SVC_DEADBAND)
    saturated = abs(q_raw) >= q_max
    q_cmd = float(np.clip(q_raw, -q_max, q_max))
    return q_cmd, saturated
```

**The control law, derived.** The voltage error is `error = V_target − V`. The
response is a **deadbanded proportional droop**:

```
          ┌ 0                          |error| ≤ 0.01   (deadband)
q_raw =   ┤ k_q·(error − 0.01)         error >  0.01    (undervoltage → inject +Q)
          └ k_q·(error + 0.01)         error < −0.01    (overvoltage  → absorb −Q)
q_cmd = clip(q_raw, −Q_MAX, +Q_MAX)
```

**Sign convention (code-to-physics):** `error > 0` means `V < V_target`
(undervoltage) → `q_raw > 0` = **inject** reactive power = raise voltage; `error < 0`
(overvoltage) → `q_raw < 0` = **absorb** = lower voltage. The deadband *offset*
(`error − 0.01`, not just `error`) makes the output **continuous** at the deadband
edge — Q starts from 0 exactly at `|error| = 0.01` (no jump), the same
continuity-at-the-breakpoint principle as the per-DER Q(V) curve (Doc 1A §7).
`saturated` flags when the *pre-clip* demand reached `Q_MAX` — recorded as
`svc_saturated` (6A-ii). **Units:** MVAr. **Worked examples** (`Q_MAX = 5`, `k_q =
166.7`): `vm = 1.005` → `|error| = 0.005 < 0.01` → `q = 0` (deadband); `vm = 0.96` →
`error = 0.04` → `q_raw = 166.7·0.03 = 5.0 = Q_MAX` → `q_cmd = +5.0`, **saturated**;
`vm = 1.02` → `error = −0.02` → `q_raw = 166.7·(−0.01) = −1.67` → `q_cmd = −1.67`
(absorb, not saturated).

**Design rationale — one central droop vs the distributed Q(V) curve.** This is a
*single* symmetric linear droop about a target, sized to the substation; contrast
Doc 1A's *per-DER* 5-segment VDE curve sized to each inverter. The SVC concentrates
reactive capability at one strong point; Volt-Var spreads it across every DER. That
contrast is exactly what the benchmark measures.

---

## 7. Section D — run_scenario_3: setup, resume, SVC creation (L363–455)

#### Line 363–421: Signature, start, and the resume block

```python
def run_scenario_3(net, profiles, network_id="unknown", v_min=V_MIN, v_max=V_MAX,
                   publish_fn=None, enable_checkpointing=True, live_csv_rewrite_fn=None):
    ap = adapt_profiles(net, profiles); _T = len(ap.times)
    if publish_fn is not None: publish_fn.on_scenario_start("svc", "SVC", _T)
    resumed_records = publish_fn.get_resume_records("svc") if (publish_fn and enable_checkpointing) else []
    start_t = (resumed_records[-1].t + 1) if resumed_records else 0
    if start_t >= len(ap.times) and resumed_records:
        q_max, k_q = _compute_svc_params(net); svc_bus = _select_svc_bus(net, ap, v_min, v_max)
        result = ScenarioResult.from_records(scenario_id="svc", records=resumed_records, ..., svc_bus=svc_bus, svc_q_max=q_max)
        return result
    time_steps = range(start_t, len(ap.times))
```

Same resume pattern as OLTC (6B-ii §4), with one SVC-specific detail: the
**skip-if-complete** branch still calls `_compute_svc_params` and `_select_svc_bus`
— **not** to run anything, but to populate `svc_bus`/`svc_q_max` in the returned
`ScenarioResult` (they are result metadata, and a fully-checkpointed run still needs
them). **No state seeding** is required (the SVC is stateless, §1). **Design note:**
this re-runs bus selection (a deep copy + power flows) on a fully-recovered run —
a small, bounded cost paid only to fill two metadata fields.

#### Line 423–455: Reset, size, select, and create the SVC sgen

```python
    net.controller.drop(...); pp.reset_results(net)
    q_max, k_q = _compute_svc_params(net); svc_bus = _select_svc_bus(net, ap, v_min, v_max)
    svc_idx = pp.create_sgen(net, bus=svc_bus, p_mw=0.0, q_mvar=0.0, name="SVC_Scenario3", type="SVC", in_service=True)
    records: list[TimestepRecord] = resumed_records.copy()
```

**The SVC is modelled as a static generator** with `p_mw = 0` (pure reactive device)
at the selected bus, `type="SVC"`. `pp.create_sgen` returns its index `svc_idx`,
used every timestep to write `q_mvar`. **Design rationale — a temporary sgen:**
pandapower has no native SVC element, so the SVC is a zero-P sgen whose `q_mvar` the
droop law drives; it is created here and **removed at the end** (§9) so the caller's
net is unchanged. `records` starts from the recovered set.

---

## 8. Section E — The droop control loop (L457–594)

Wrapped in `try:` (the matching `finally` removes the sgen, §9).

#### Line 461–508: [1]–[2] Profiles, SVC-idle pre-PF, non-convergence

```python
    for t in time_steps:
        ... write load + DER profiles; DER q_mvar = 0 ...
        net.sgen.at[svc_idx, "q_mvar"] = 0.0          # SVC idle before pre-PF
        try: pp.runpp(net, voltage_depend_loads=False); converged_pre = True
        except Exception: converged_pre = False
        if not converged_pre:
            records.append(TimestepRecord(..., q_applied_mvar=pd.Series({svc_idx: 0.0}), svc_q_mvar=0.0, svc_saturated=False, converged=False, ...)); continue
```

Writes the profiles (DER `q_mvar = 0` — like OLTC, the DERs provide no reactive
control; the SVC is the only reactive actuator) and **zeroes the SVC before the
pre-control solve**, so the pre-PF measures the *uncompensated* voltage the droop
then reacts to. On divergence, a record is written with the SVC held at 0 and the
step skipped. **Design rationale — zero-before-measure:** the droop must respond to
the voltage *without* the SVC's own previous injection, so `svc_idx.q_mvar` is reset
to 0 each step before the pre-PF (the SVC-scenario analogue of the coordinator's
`q=0` invariant, Doc 2 §7).

#### Line 510–531: [3]–[4] Droop decision and conditional post-PF

```python
        vm_svc = float(net.res_bus.at[svc_bus, "vm_pu"])
        q_cmd, saturated = _droop_q(vm_svc, q_max, k_q)
        net.sgen.at[svc_idx, "q_mvar"] = q_cmd
        if abs(q_cmd) > 0.0:
            try: pp.runpp(net, voltage_depend_loads=False); converged_post = True
            except Exception: converged_post = False; logger.warning("post-SVC PF diverged ...")
        else:
            converged_post = True     # SVC in deadband — reuse pre-control results
```

Reads the SVC-bus voltage, computes the droop command `(q_cmd, saturated)` (§6),
writes it, and **re-solves only if `q_cmd ≠ 0`**. **Design rationale — the deadband
skip (cf. OLTC `post_pf_reused`, 6B-ii):** when the SVC is inside its deadband
(`q_cmd = 0`), applying it changes nothing, so the pre-control power flow *is* the
settled state and the second `runpp` is skipped — a real saving on the many
in-deadband timesteps. Unlike OLTC, there is no rollback path: a diverged post-SVC
PF is recorded as non-converged (the SVC command is bounded by `±Q_MAX`, so a
diverging post-PF is rare and treated as a lost step, not an unstable actuator to
undo).

#### Line 532–594: [5] Build the record; periodic live-CSV/log

```python
        converged = converged_post
        if converged:
            vm = net.res_bus["vm_pu"].copy(); ll = ...; tl = ...
            ov_buses = ...; losses_mw_t = float(net.res_line["pl_mw"].sum() + net.res_trafo["pl_mw"].sum()); grid_import_mw_t = ...
        else: vm = ll = tl = _empty_series(); ... losses_mw_t = grid_import_mw_t = None
        rec = TimestepRecord(..., q_applied_mvar=pd.Series({svc_idx: q_cmd}), svc_q_mvar=q_cmd, svc_saturated=saturated, ...)
        if publish_fn is not None: publish_fn.on_timestep(rec)
        records.append(rec)
        if t % 96 == 0:
            if live_csv_rewrite_fn is not None: partial = ScenarioResult.from_records(scenario_id="svc", records=records, ...); live_csv_rewrite_fn(partial)
            logger.info("... t=%d/%d (%.1f%%) | vm_svc=%.4f | q_cmd=%.3f MVAr | violations=%d", ...)
```

Builds the record from the settled state, thresholding with the standard `±ε`
(6A-ii §5). **The SVC-specific fields:** `q_applied_mvar` is a **single-element
Series** `{svc_idx: q_cmd}` (one central device, unlike Volt-Var's per-DER Series);
`svc_q_mvar = q_cmd` and `svc_saturated = saturated` fill the `TimestepRecord` SVC
block (6A-ii §4). Energy balance is recorded for converged steps. The periodic block
rewrites the live CSV and logs progress + running violation count. `on_timestep`
streams and checkpoints the record.

---

## 9. Section F — Cleanup, result & summary (L596–642)

```python
    finally:
        net.sgen.drop(index=svc_idx, inplace=True); pp.reset_results(net)
        logger.debug("SVC sgen (idx=%d) removed from net.", svc_idx)
    ...
    result = ScenarioResult.from_records(scenario_id="svc", records=records, ..., svc_bus=svc_bus, svc_q_max=q_max)
    svc_active = sum(1 for r in records if r.svc_q_mvar and abs(r.svc_q_mvar) > 0.0)
    svc_saturated = sum(1 for r in records if r.svc_saturated)
    q_total_abs = sum(abs(r.svc_q_mvar) for r in records if r.svc_q_mvar is not None)
    logger.info("... Done. ... SVC active %d steps | saturated %d steps | |Q| total=%.2f ...", ...)
```

#### Line 596–642: The `finally` cleanup and the SVC summary

**The `finally` block is the key robustness feature.** It removes the SVC sgen and
resets results **unconditionally** — whether the loop completed normally, raised, or
was interrupted — so the caller always receives a clean network with no leftover
`SVC_Scenario3` element (the property `benchmark_runner` relies on, per its module
docstring). `from_records` then aggregates with `svc_bus`/`svc_q_max` as result
metadata, and the summary logs the SVC diagnostics: steps **active** (nonzero Q),
steps **saturated** (droop at `±Q_MAX`), and total absolute reactive throughput.
**Design rationale — cleanup in `finally`, not after the loop:** an exception mid-run
must not leave a phantom SVC on the caller's net; `finally` guarantees removal on
every exit path.

---

## 10. Findings and flags

**Flagged, not edited:**

1. **Resume progress-% overshoot (L590), cosmetic.** As in OLTC (6B-ii §7.1), the
   progress log divides the *absolute* `t` by the *post-resume* range length
   (`len(time_steps) = T − start_t`), so the percentage exceeds 100% after a resume.
   Simulation is unaffected. Same recommended fix (divide by `_T`).

2. **Skip-if-complete re-runs bus selection (L403–405).** A fully-checkpointed run
   still calls `_select_svc_bus` (deep copy + power flows) purely to populate
   `svc_bus`/`svc_q_max` metadata. Correct but slightly wasteful; could cache the
   selection in the checkpoint if it ever matters. Not a bug.

**Not bugs (documented for clarity):** the SVC is stateless, so resume needs no
seeding (a real simplification vs OLTC/Volt-Var); the `finally` sgen removal is
essential, not incidental; zeroing the SVC before the pre-PF is the correct
"measure-then-compensate" ordering; there is deliberately no post-PF rollback (the
`±Q_MAX`-bounded command makes divergence rare, and a lost step is recorded honestly).

---

## 11. Coverage checklist

Every named entity in `scenario_3_svc.py`, each confirmed documented.

| Entity | Lines | Kind | Documented | Where-used stated |
|--------|-------|------|:----------:|:-----------------:|
| `SVC_V_TARGET` / `SVC_DEADBAND` / `SVC_FULL_ACTION_DV` / `SVC_Q_MAX_RATIO` | 111–114 | constants | ✅ | ✅ |
| `_mv_kv` | 121–137 | helper | ✅ | ✅ |
| `_eligible_mv_buses` | 140–159 | helper | ✅ | ✅ |
| `_select_svc_bus_stress` | 162–195 | helper | ✅ | ✅ |
| `_select_svc_bus_first_violation` | 198–243 | helper | ✅ | ✅ |
| `_select_svc_bus` | 246–290 | helper | ✅ | ✅ |
| `_compute_svc_params` | 293–321 | helper | ✅ | ✅ |
| `_droop_q` | 324–352 | helper | ✅ | ✅ |
| `_empty_series` | 355–356 | helper | ✅ | ✅ |
| `run_scenario_3` | 363–642 | function | ✅ | ✅ |
| — signature (+ checkpointing) | 363–372 | — | ✅ | — |
| — resume + skip-if-complete | 391–421 | — | ✅ | — |
| — reset/size/select/create SVC | 423–455 | — | ✅ | — |
| — [1]–[2] profiles + pre-PF | 461–508 | — | ✅ | — |
| — [3]–[4] droop + conditional post-PF | 510–531 | — | ✅ | — |
| — [5] build record + periodic log | 532–594 | — | ✅ | — |
| — `finally` cleanup + result + summary | 596–642 | — | ✅ | — |

**Also covered (not standalone entities):** the centralised-vs-distributed contrast,
the 3-tier bus selection with graceful degradation, the deadbanded-proportional droop
derivation with the continuity-at-the-deadband-edge point, the stateless resume, the
zero-before-measure ordering, the deadband post-PF skip, and the `finally` cleanup
guarantee.

**Flags raised (not edited):** resume progress-% overshoot (L590, cosmetic);
skip-if-complete re-runs bus selection (minor).

---

*End of Document 6C. Next per the queue: Document 6D — Scenario Runner: Volt-Var
(HIL) (`scenario_4_volt_var.py`), the project's core scenario — the manual loop that
drives `VoltVarController`/`SensitivityCoordinator`/`DERDynamics` (Docs 1–4), the
4A/4B split, the Arduino HIL path, and the checkpoint resume with DER-state seeding.*
