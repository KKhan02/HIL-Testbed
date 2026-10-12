# TASK 4 — DOCUMENT 6B-ii
## Scenario Runner — OLTC, Part 2: The Control Loop
### `scenario_2_oltc.py` — `run_scenario_2` (L597–995, updated for checkpointing)

> **Source re-verified (project tree):** `scenario_2_oltc.py` is now **997 lines**
> (+2 vs the upload this doc was written against); the 6B-i setup helpers (L86–590)
> are unchanged, and the `run_scenario_2` structure is unchanged — internal line
> numbers below read ~+2. Content is current.
>
> **(prior) Source verified:** read against the **updated** `scenario_2_oltc.py` (995
> lines, with the checkpointing/resume additions) before writing. If the file is
> edited, re-verify line numbers.
>
> **Part 2 of 2 of Document 6B** (6B-i = the setup infrastructure: metadata,
> selection, ganging validation, tap-sign calibration — *unaffected* by the
> checkpointing update). This document covers the runner itself: the 7-step setup
> orchestration and the per-timestep **tap control state machine** whose outputs
> fill the `TimestepRecord` OLTC block (6A-ii).
>
> **Read alongside:** **6B-i** (the helpers this runner calls); **6A-ii** (the
> `tap_*` fields recorded here + `TimestepRecord`/`from_records`); **6A-iii §4**
> (the declarative-vs-manual contrast — OLTC is the first manual-loop scenario);
> **Document 10** (`publisher.PublishHandle`, the checkpoint source read on resume).

---

## TABLE OF CONTENTS

1. [Overview — the tap control state machine](#1-overview)
2. [File logic flow — pseudocode](#2-file-logic-flow--pseudocode)
3. [Call-site map](#3-call-site-map)
4. [Section A — Signature, resume & the 7-step setup (L597–737)](#4-section-a--signature-resume--the-7-step-setup-l597737)
5. [Section B — The tap control loop (L744–956)](#5-section-b--the-tap-control-loop-l744956)
6. [Section C — Result assembly & tap summary (L958–995)](#6-section-c--result-assembly--tap-summary-l958995)
7. [Findings and flags](#7-findings-and-flags)
8. [Coverage checklist](#8-coverage-checklist)

---

## 1. Overview

`run_scenario_2` is the OLTC benchmark runner — the first **manual-loop** scenario
(Scenario 1 was declarative; Doc 6A-iii §4). It has two phases:

1. **Setup** (L636–737): call the 6B-i helpers in the mandated 7-step order to
   select the OLTC group, complete/override/validate its tap metadata, pick the
   controlled buses, and calibrate the tap sign — then initialise the tap position.
2. **Control loop** (L744–956): for each timestep, apply the profiles, solve the
   pre-action power flow, and run a **tap control state machine** — compare the
   mean controlled-bus voltage against the OLTC deadband, move one tap toward
   nominal if outside it, and handle the edge cases (rail limit, post-tap
   divergence with rollback, deadband inaction). Every timestep's decision is
   recorded in the `TimestepRecord` OLTC fields.

**The state machine is the intellectual core**, and it maps one-to-one onto the
six `tap_*` record fields (6A-ii §4): `tap_attempted` (did the controller want to
move?), `tap_candidate` (to where?), `tap_changed` (did it actually move?),
`post_pf_reused` (did we skip the post-PF because nothing moved?), `tap_pos` (the
resulting position), and `tap_blocked_reason` (why a wanted move didn't happen:
`pre_pf_non_convergence` / `post_pf_non_convergence` / `tap_limit_reached`).

**OLTC uses no DER reactive control.** Every timestep writes `net.sgen.q_mvar = 0`
(L755) — the transformer tap is the *only* actuator. So every record has
`q_applied_mvar = None`; this is the pure-OLTC baseline the DER schemes (SVC,
Volt-Var) are compared against.

**Checkpointing (new).** The runner now resumes from `checkpoint/oltc.jsonl`:
`start_t` is the step after the last recovered record, the loop skips entirely if
the checkpoint already covers all steps, and — critically — the **tap position is
re-seeded from the last checkpoint record** (L719–724), without which a resumed
run would restart at neutral and lose the OLTC's accumulated state.

---

## 2. File logic flow — pseudocode

```
FUNCTION run_scenario_2(net, profiles, ..., enable_checkpointing, live_csv_rewrite_fn):
    adapt profiles; announce scenario start
    RESUME: resumed = publish_fn.get_resume_records("oltc"); start_t = last.t + 1
        if start_t covers all steps → aggregate resumed records and return (skip sim)
    reset controllers/results
    7-STEP SETUP (all 6B-i helpers):
        [1] select OLTC trafos  [2] print original metadata
        [3] complete missing metadata  [4] apply user override  [5] print final
        [6] validate ganging  [7] derive gang range + controlled buses + calibrate sign
    tap init: current_tap = last-checkpoint tap_pos if resuming, else tap_neutral
    records = resumed.copy()

    FOR t in range(start_t, T):
        [1] write load + DER profiles; DER q_mvar = 0 (no reactive control)
        [2] pre-action power flow at current tap
        [3] not converged → record (tap held, reason="pre_pf_non_convergence"); continue
        [4] vm_ctrl = mean voltage over controlled buses; decide against the deadband:
              > upper → candidate = clip(current + sign·1, gang range); attempted
              < lower → candidate = clip(current − sign·1, gang range); attempted
              else    → no action
        [5] if attempted and candidate ≠ current:
                move tap; post-PF;
                  converged → accept (tap_changed)
                  diverged  → rollback to previous tap + re-solve (reason="post_pf_non_convergence")
            elif attempted and candidate == current:  rail → reuse pre-PF (reason="tap_limit_reached")
            else:                                       deadband → reuse pre-PF (no move)
        [6] build TimestepRecord (tap_* fields, energy balance); publish; append
        every 96 steps: rewrite the live CSV; log progress + violation counts

    aggregate → ScenarioResult(scenario_id="oltc"); log tap-movement summary; return
```

---

## 3. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `run_scenario_2` | function | `benchmark_runner` `SCENARIO_REGISTRY[2]` (Doc 9) — dispatched as scenario 2. |
| 6B-i helpers (`_select_oltc_trafos`, `_print_tap_metadata`, `_complete_missing_tap_metadata`, `_apply_user_tap_metadata_override`, `_validate_tap_metadata`, `_select_controlled_buses`, `_calibrate_tap_sign`, `_empty_series`) | functions | Called here, in the 7-step setup / loop (§4, §5). |
| `adapt_profiles`, `TimestepRecord`, `ScenarioResult.from_records` | infra | 6A-i / 6A-ii. |
| `publish_fn.get_resume_records` / `.on_scenario_start` / `.on_timestep` / `.on_scenario_end` | `PublishHandle` methods | The resume/streaming hooks (Doc 10). |
| `live_csv_rewrite_fn` | injected callback | `benchmark_runner._rewrite_live_csv` (Doc 9), only when `live_csv_path` is set. |
| OLTC constants (`OLTC_CONTROL_LOWER/UPPER`, `MAX_TAP_STEPS_PER_ACTION`) | constants | The deadband test + tap-step computation (§5). |

---

## 4. Section A — Signature, resume & the 7-step setup (L597–737)

#### Line 597–641: Signature and scenario start

```python
def run_scenario_2(net, profiles, network_id="unknown", v_min=V_MIN, v_max=V_MAX,
                   auto_complete_tap_metadata=True, tap_metadata_override=None,
                   publish_fn=None, enable_checkpointing=True, live_csv_rewrite_fn=None):
    ...
    ap = adapt_profiles(net, profiles); _T = len(ap.times)
    if publish_fn is not None: publish_fn.on_scenario_start("oltc", "OLTC-only", _T)
```

The two OLTC-specific parameters carry into 6B-i: `auto_complete_tap_metadata`
(gate for step [3]) and `tap_metadata_override` (step [4] payload). **New:**
`enable_checkpointing` and `live_csv_rewrite_fn` (the checkpointing/live-CSV hooks,
forwarded by `benchmark_runner._build_kwargs`, Doc 9). `adapt_profiles` (6A-i)
normalises the profiles; `on_scenario_start` tells the dashboard a new scenario has
begun.

#### Line 643–663: The resume block

```python
    resumed_records: list[TimestepRecord] = []
    if publish_fn is not None and enable_checkpointing:
        resumed_records = publish_fn.get_resume_records("oltc")
    start_t = (resumed_records[-1].t + 1) if resumed_records else 0
    if start_t >= len(ap.times) and resumed_records:
        ... result = ScenarioResult.from_records(scenario_id="oltc", records=resumed_records, ...); return result
    if start_t > 0: logger.info("... Resuming from t=%d/%d.", ...)
    time_steps = range(start_t, len(ap.times))
```

**Crash-resume, three cases (Doc 6A-ii §1).** `get_resume_records("oltc")` rebuilds
any `TimestepRecord`s from `checkpoint/oltc.jsonl` (via `from_checkpoint_dict`, 6A-ii).
`start_t` is the step *after* the last recovered one. If the checkpoint already
covers every step, the runner **skips the simulation entirely** — aggregating the
recovered records into a `ScenarioResult` and returning. Otherwise the loop runs
`range(start_t, T)`, continuing where the crash left off. **Design rationale:** an
annual OLTC run is long; a crash at step 30,000 should not restart from zero.

#### Line 665–715: The 7-step setup (the 6B-i helpers in order)

```python
    net.controller.drop(...); pp.reset_results(net)
    trafo_idx = _select_oltc_trafos(net)                                  # [1]
    _print_tap_metadata(net, trafo_idx, "Original ...")                   # [2]
    if auto_complete_tap_metadata: _complete_missing_tap_metadata(...)    # [3]
    _apply_user_tap_metadata_override(net, trafo_idx, network_id, tap_metadata_override)  # [4]
    _print_tap_metadata(net, trafo_idx, "Final ...")                      # [5]
    _validate_tap_metadata(net, trafo_idx)                                # [6]
    tap_min_gang = int(...tap_min.max()); tap_max_gang = int(...tap_max.min())  # [7]
    tap_neutral = int(...tap_neutral.iloc[0]); ctrl_buses = _select_controlled_buses(...)
    sign = _calibrate_tap_sign(net, trafo_idx, ctrl_buses, tap_neutral, tap_min_gang, tap_max_gang)
```

The exact 7-step sequence 6B-i documents, executed in order. **Design rationale —
select before metadata, gang range as intersection, empirical sign** are all covered
in 6B-i (§7, §8, §10). The controllers are dropped first (a prior scenario on the
same net may have left `ConstControl`s). By the end, the runner holds: the trafo
group, the ganged range `[tap_min_gang, tap_max_gang]`, the neutral, the controlled
buses, and the calibrated `sign` (the direction that lowers voltage).

#### Line 717–737: Tap initialisation — seed from checkpoint on resume

```python
    if resumed_records and resumed_records[-1].tap_pos is not None:
        current_tap = resumed_records[-1].tap_pos
        logger.info("... Resuming with tap_pos=%d from checkpoint.", ...)
    else:
        current_tap = tap_neutral
    net.trafo.loc[trafo_idx, "tap_pos"] = current_tap
    records: list[TimestepRecord] = resumed_records.copy()
```

**The critical resume detail (Doc 6A-ii §1).** On a fresh run the tap starts at
`tap_neutral`; **on resume it is re-seeded from the last checkpoint record's
`tap_pos`** — because the OLTC's tap position is *stateful* (it only moves one step
per timestep, so the position at step *k* depends on the whole history). Without
this seeding, a resumed run would restart at neutral and produce a discontinuity in
the tap trajectory. `records` starts as a copy of the recovered records, so new
timesteps append to the recovered history.

---

## 5. Section B — The tap control loop (L744–956)

The per-timestep state machine. Each numbered step corresponds to the docstring's
sequence and maps to the `TimestepRecord` OLTC fields.

#### Line 744–755: [1] Write profiles — DER reactive power forced to zero

```python
    for t in time_steps:
        t0 = time.perf_counter(); timestamp = ap.times[t]
        if not ap.load_p.empty:
            net.load.loc[ap.load_idx, "p_mw"]   = ap.load_p.iloc[t].values
            net.load.loc[ap.load_idx, "q_mvar"] = ap.load_q.iloc[t].values
        if not ap.der_p.empty:
            net.sgen.loc[ap.der_p.columns, "p_mw"]   = ap.der_p.iloc[t].values
            net.sgen.loc[ap.der_p.columns, "q_mvar"] = 0.0
```

Writes the load P/Q and DER P profiles index-explicitly (6A-i). **`net.sgen.q_mvar
= 0.0` every step — the defining property of the OLTC-only scenario:** the DERs
inject *no* reactive power; the transformer tap is the sole voltage actuator. This
is why `q_applied_mvar = None` in every OLTC record (§5 [6]) and the whole point of
the comparison — OLTC vs the DER schemes that *do* use inverter reactive capability.

#### Line 757–788: [2]–[3] Pre-action power flow and non-convergence handling

```python
        try: pp.runpp(net, voltage_depend_loads=False); converged_pre = True
        except Exception: converged_pre = False
        if not converged_pre:
            logger.warning("... pre-PF diverged — tap held at %d.", ...)
            records.append(TimestepRecord(..., converged=False, tap_pos=current_tap,
                tap_changed=False, tap_attempted=False, tap_candidate=None,
                post_pf_reused=False, tap_blocked_reason="pre_pf_non_convergence", ...))
            continue
```

Solves the power flow at the *current* tap (`voltage_depend_loads=False`, the
SimBench mandatory flag). **On divergence**, the tap is held, an empty-result
record is written with `tap_blocked_reason="pre_pf_non_convergence"` — the first of
the three block reasons — and the step is skipped. **Design rationale:** a diverged
pre-PF gives no voltage reading to base a tap decision on, so the only safe action
is to hold and record why.

#### Line 790–812: [4] The deadband decision

```python
        vm_ctrl     = float(net.res_bus.loc[ctrl_buses, "vm_pu"].mean())
        vm_ctrl_min = float(...min()); vm_ctrl_max = float(...max())     # diagnostic only
        if vm_ctrl > OLTC_CONTROL_UPPER:
            tap_candidate = int(np.clip(current_tap + sign * MAX_TAP_STEPS_PER_ACTION, tap_min_gang, tap_max_gang)); tap_attempted = True
        elif vm_ctrl < OLTC_CONTROL_LOWER:
            tap_candidate = int(np.clip(current_tap - sign * MAX_TAP_STEPS_PER_ACTION, tap_min_gang, tap_max_gang)); tap_attempted = True
        else:
            tap_candidate = None; tap_attempted = False
```

**The control law.** `vm_ctrl` is the **mean** voltage across the controlled
secondary busbars (6B-i §9); `min`/`max` are logged for diagnostics but *not* used
in the decision. If `vm_ctrl` is above the deadband (`OLTC_CONTROL_UPPER = 1.02`),
the tap moves one step in the voltage-*lowering* direction (`current + sign·1`,
clipped to the ganged range); if below (`OLTC_CONTROL_LOWER = 0.98`), one step the
other way; inside the deadband, no action. **Design rationale — the sign and the
clip.** `sign` (from `_calibrate_tap_sign`, 6B-i §10) guarantees `+sign` lowers
voltage regardless of `tap_side`; the `np.clip` to `[tap_min_gang, tap_max_gang]`
enforces the ganged rail. **Recall the deadband (0.98–1.02) is tighter than the
violation band (0.95–1.05)** (6B-i §4) — the OLTC regulates preventively.

#### Line 814–880: [5] Apply the tap — the three outcomes

```python
        if tap_attempted and tap_candidate != current_tap:
            prev_tap = current_tap; net.trafo.loc[trafo_idx, "tap_pos"] = tap_candidate
            try: pp.runpp(net, voltage_depend_loads=False); converged_post = True
            except Exception: converged_post = False
            if converged_post:
                current_tap = tap_candidate; tap_changed = True; post_pf_reused = False; blocked_reason = None
            else:
                net.trafo.loc[trafo_idx, "tap_pos"] = prev_tap; current_tap = prev_tap
                tap_changed = False; post_pf_reused = False; blocked_reason = "post_pf_non_convergence"
                try: pp.runpp(net, voltage_depend_loads=False); converged_post = True   # restore valid res_bus
                except Exception: converged_post = False; logger.error("... rollback PF also diverged.")
        elif tap_attempted and tap_candidate == current_tap:
            converged_post = True; tap_changed = False; post_pf_reused = True; blocked_reason = "tap_limit_reached"
        else:
            converged_post = True; tap_changed = False; blocked_reason = None; post_pf_reused = True
```

The state machine's three branches — this is where all six `tap_*` fields are set:

- **Move accepted** (`attempted`, candidate ≠ current, post-PF converges): commit
  the new tap, `tap_changed=True`, `post_pf_reused=False`, no block reason.
- **Move rolled back** (post-PF *diverges*): **restore the previous tap and re-solve**
  to recover a valid `res_bus`, `tap_changed=False`, `blocked_reason=
  "post_pf_non_convergence"`. **Design rationale:** a tap move that breaks
  convergence is undone rather than left in a bad state — the second block reason.
- **Rail reached** (`attempted` but candidate *equals* current — the clip pinned it
  at the ganged limit): no move possible, `blocked_reason="tap_limit_reached"`,
  **reuse the pre-PF result** (`post_pf_reused=True`) since nothing changed — the
  third block reason.
- **Deadband** (not attempted): no move, `post_pf_reused=True`, no block reason.

**Design rationale — `post_pf_reused` as an efficiency flag.** When the tap does not
move (deadband or rail), the pre-action power flow *is* the settled state, so no
second `runpp` is run and the pre-PF result is reused. Only an accepted or
rolled-back move needs a post-PF. This roughly halves the power-flow count on the
many timesteps where the OLTC sits still.

#### Line 882–926: [6] Build the record

```python
        converged = converged_post
        if converged:
            vm = net.res_bus["vm_pu"].copy(); ll = ...; tl = ...
            ov_buses = vm.index[vm > v_max + VOLTAGE_EPSILON].tolist()  # + uv/ol/ot
            losses_mw_t = float(net.res_line["pl_mw"].sum() + net.res_trafo["pl_mw"].sum())
            grid_import_mw_t = float(net.res_ext_grid["p_mw"].sum()); der_gen_mw_t = ...; load_mw_t = ...
        else: vm = ll = tl = _empty_series(); ov_buses = ... = []
        rec = TimestepRecord(..., q_applied_mvar=None, ..., tap_pos=current_tap,
            tap_changed=tap_changed, tap_attempted=tap_attempted, tap_candidate=tap_candidate,
            post_pf_reused=post_pf_reused, tap_blocked_reason=blocked_reason,
            losses_mw=losses_mw_t if converged else None, ...)
        if publish_fn is not None: publish_fn.on_timestep(rec)
        records.append(rec)
```

Builds the record from the settled state, thresholding violations with the same
`±ε` as `violation_detector`/`make_record_from_report` (6A-ii §5). All six `tap_*`
fields are populated from the state machine; `q_applied_mvar`/`p_applied_mw`/
`p_target_mw` are `None` (OLTC applies no DER control and no ramp limiting). Energy
balance is recorded for converged steps. `on_timestep(rec)` streams the record to
the dashboard **and** (via `PublishHandle`, Doc 10) appends it to the checkpoint
stream — which is what a future resume reads back.

#### Line 928–956: Periodic live-CSV rewrite and progress log

```python
        if t % 96 == 0:
            if live_csv_rewrite_fn is not None:
                    partial = ScenarioResult.from_records(scenario_id="oltc", records=records, ...)
                    live_csv_rewrite_fn(partial)
            voltage_violation_steps = sum(...); any_violation_steps = sum(...)
            logger.info("... t=%d/%d (%.1f%%) | tap=%d | vm_ctrl=%.4f | ...", ...)
```

Every 96 steps (a day at 15-min resolution): if a `live_csv_rewrite_fn` was injected
(only when `benchmark_runner.live_csv_path` is set, Doc 9), build a **partial**
`ScenarioResult` from the records so far and rewrite the live comparison CSV; then
log progress and running violation counts. **Design rationale:** the partial-result
rewrite lets a dashboard show the in-progress scenario's metrics mid-run, merged
with completed scenarios (Doc 9 `_rewrite_live_csv`).

---

## 6. Section C — Result assembly & tap summary (L958–995)

```python
    result = ScenarioResult.from_records(scenario_id="oltc", network_id=network_id, records=records, elapsed_s=elapsed, dt_s=ap.dt_s)
    tap_moves = sum(1 for r in records if r.tap_changed)
    tap_blocks = sum(1 for r in records if r.tap_blocked_reason)
    tap_min_seen = min((r.tap_pos for r in records if r.tap_pos is not None), default=tap_neutral)
    tap_max_seen = max((r.tap_pos for r in records if r.tap_pos is not None), default=tap_neutral)
    logger.info("... Done. ... %d tap moves | %d blocked | tap range seen [%d, %d]", ...)
    if publish_fn is not None: publish_fn.on_scenario_end(result)
    return result
```

#### Line 958–995: Aggregate and summarise the tap activity

Aggregates all records (recovered + new) into a `ScenarioResult(scenario_id="oltc")`
via `from_records` (6A-ii), then logs an OLTC-specific summary: total accepted **tap
moves** (`tap_changed`), total **blocked** attempts (`tap_blocked_reason` set), and
the **tap range actually visited** (`tap_min_seen`/`tap_max_seen`, guarded with
`default=tap_neutral` for an all-`None` edge case — the correct guard pattern, cf.
the `min_vm` fix in 6A-ii). `on_scenario_end(result)` signals the dashboard the
scenario is complete. **Design rationale:** the tap-move/blocked counts are the
headline OLTC diagnostics — how often the regulator acted, and how often it wanted
to act but couldn't (rail/divergence).

---

## 7. Findings and flags

**Flagged, not edited (both cosmetic — introduced with the resume feature):**

1. **Progress percentage can exceed 100% on resume (L950–951).** The log computes
   `100.0 * t / max(len(time_steps), 1)`, but after resume `time_steps =
   range(start_t, T)` has length `T − start_t` while `t` is the *absolute* step
   index. So once resumed, `t / (T − start_t)` overshoots 100% (e.g. resuming at
   30,000 of 35,040 logs ~140% near the end). Cosmetic only — the simulation is
   correct. **Recommend:** divide by `_T` (the full length) or subtract `start_t`
   from `t` in the percentage.
2. **Over-indented live-CSV block (L929–935).** The `partial = ...` and
   `live_csv_rewrite_fn(partial)` are indented 20 spaces under the `if` at 16;
   valid Python, but inconsistent style (same pattern in scenarios 3/5).

**Not bugs (documented for clarity):** `net.sgen.q_mvar = 0` every step is the
defining OLTC property, not an omission; `post_pf_reused` reusing the pre-PF on a
no-move step is a deliberate power-flow-count optimisation; the tap-position
re-seed on resume is essential for state continuity.

---

## 8. Coverage checklist

Every element of `run_scenario_2`, each confirmed documented against the updated file.

| Element | Lines | Documented | Notes |
|--------|-------|:----------:|-------|
| Signature (+ new checkpointing params) | 597–608 | ✅ | `enable_checkpointing`, `live_csv_rewrite_fn` |
| Adapt + scenario start | 636–641 | ✅ | |
| Resume block (get_resume_records, start_t, skip-if-complete) | 643–663 | ✅ | |
| 7-step setup (6B-i helpers) | 665–715 | ✅ | delegates to 6B-i |
| Tap init (checkpoint-seeded) | 717–727 | ✅ | resume continuity |
| `records = resumed.copy()` | 739 | ✅ | |
| [1] Write profiles (DER q=0) | 744–755 | ✅ | OLTC-only property |
| [2]–[3] Pre-PF + non-convergence | 757–788 | ✅ | `pre_pf_non_convergence` |
| [4] Deadband decision | 790–812 | ✅ | mean vm_ctrl, sign, clip |
| [5] Apply tap (3 outcomes) | 814–880 | ✅ | accept / rollback / rail / deadband |
| [6] Build record | 882–926 | ✅ | all six `tap_*` fields |
| Live-CSV rewrite + progress log | 928–956 | ✅ | flagged % bug |
| Result assembly + tap summary | 958–995 | ✅ | |

**Also covered (not standalone entities):** the tap state machine ↔ `tap_*` field
mapping, the `post_pf_reused` efficiency optimisation, the rollback-on-divergence
robustness, the deadband-tighter-than-violation-band regulation, and the
checkpoint-seeded tap continuity.

**Flags raised (not edited):** resume progress-% overshoot (L950–951, cosmetic);
over-indented live-CSV block (L929–935, cosmetic).

---

*End of Document 6B-ii — completing Document 6B (OLTC: 6B-i setup + 6B-ii loop).
Next per the queue: Document 6C — Scenario Runner: SVC (`scenario_3_svc.py`), the
centralised static-var-compensator scheme, also now with checkpointing/resume.*
