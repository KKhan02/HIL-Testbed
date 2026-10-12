# TASK 4 — DOCUMENT 6D
## Scenario Runner — Volt-Var (HIL)
### `scenario_4_volt_var.py` — the project's core scenario (576 lines, checkpointing + DER-state seeding)
### Complete line-by-line walkthrough

> **Source re-verified (project tree):** `scenario_4_volt_var.py` is now **577 lines**
> (+1 vs the upload this doc was written against); structure unchanged — internal
> line numbers below read ~+1. Content is current.
>
> **(prior) Source verified:** read against the **updated** `scenario_4_volt_var.py` (576
> lines, with checkpointing/resume + DER-state seeding) before writing. If the file
> is edited, re-verify line numbers.
>
> **This is the core scenario** — the one implementing the paper's contribution. It
> *orchestrates* everything documented so far: it drives `VoltVarController` (Doc
> 1A–1C), the Arduino firmware (Doc 5) via the serial interface (1B), the
> `SensitivityCoordinator` (Docs 2/3), and `DERDynamics` (Doc 4) — through the
> single-timestep engine `run_coordinated_timestep` (Doc 3B) — and adds the third
> control tier (active-power curtailment).
>
> **Read alongside:** **Documents 1A–1C, 2, 3A–3B, 4, 5** (everything this scenario
> drives); **6A-i/6A-ii** (`adapt_profiles`, the Volt-Var `TimestepRecord` block,
> `ScenarioResult`); **6B-ii/6C** (the other manual-loop scenarios, for contrast).

---

## TABLE OF CONTENTS

1. [Overview — the three-tier core scenario](#1-overview)
2. [File logic flow — pseudocode](#2-file-logic-flow--pseudocode)
3. [Call-site map — what this scenario drives](#3-call-site-map--what-this-scenario-drives)
4. [Section A — Curtailment constants (L112–113)](#4-section-a--curtailment-constants-l112113)
5. [Section B — _build_dynamics (L118–148)](#5-section-b--_build_dynamics-l118148)
6. [Section C — Setup, resume & guards (L151–252)](#6-section-c--setup-resume--guards-l151252)
7. [Section D — _run_loop: construction & DER-state seeding (L261–321)](#7-section-d--_run_loop-construction--der-state-seeding-l261321)
8. [Section E — The timestep loop: Tiers 1+2 (L322–384)](#8-section-e--the-timestep-loop-tiers-12-l322384)
9. [Section F — Tier 3: active-power curtailment (L386–454)](#9-section-f--tier-3-active-power-curtailment-l386454)
10. [Section G — Record build & periodic log (L456–542)](#10-section-g--record-build--periodic-log-l456542)
11. [Section H — HIL dispatch & result (L544–576)](#11-section-h--hil-dispatch--result-l544576)
12. [Findings and flags](#12-findings-and-flags)
13. [Coverage checklist](#13-coverage-checklist)

---

## 1. Overview

Scenario 4 is the **rule-based Volt-Var HIL** benchmark — the project's primary
contribution. It is where every component documented so far comes together into a
**three-tier voltage-control hierarchy**, applied per timestep over a year:

1. **Tier 1 — local Q(V)** (Docs 1A/1B/1C/5): each DER sets reactive power from its
   own bus voltage via the VDE-AR-N 4110 curve — computed on the **Arduino** (HIL)
   or locally (dry-run).
2. **Tier 2 — coordination** (Docs 2/3): the `SensitivityCoordinator` adds a global
   reactive correction for any residual violations (4B only).
3. **Tier 3 — active-power curtailment** (this document, §9): when reactive power is
   exhausted (`curtailment_needed`), progressively reduce DER active power until the
   violation clears or P reaches zero.

**The division of labour.** Tiers 1+2 live inside `run_coordinated_timestep` (Doc
3B) — the single-timestep engine that does the pre-PF, the Arduino Q(V) exchange,
the coordination, the `DERDynamics` step, and the post-PF. This scenario's job is to
**orchestrate** that engine across all timesteps and **add Tier 3** around it. So
most of the per-step work is delegated; this file contributes the loop, the
curtailment, the record-building, the HIL dispatch, and the resume logic.

**4A vs 4B** — the single `coordination` flag selects the two published variants:
`False` = **4A** (local Q(V) only, `scenario_id="volt_var_local"`), `True` = **4B**
(local + coordinated, `scenario_id="volt_var_coord"`). They checkpoint separately.

**HIL vs dry-run** — the loop is a closure (`_run_loop`) called either with a live
`ArduinoSerialInterface` (opened as a context manager so the port always closes) or
`None` (dry-run, pure-Python Q(V)). This is the scenario-level dry-run/hardware split
that the quantisation finding turns on (Doc 1B §7).

---

## 2. File logic flow — pseudocode

```
CONSTANTS: curtailment step (10% of p_target per iter), max iters (10 → P=0)

_build_dynamics(ctrl, dt_s): construct DERDynamics with q_max = Q_RATIO·p_installed
                             (zero-sentinel guarded), MV presets (t95=10s, ramp 0.5%/s)

FUNCTION run_scenario_4(net, profiles, …, dry_run, coordination, enable_checkpointing, …):
    adapt profiles; announce start (4A or 4B id)
    RESUME: resumed = get_resume_records(volt_var_local|coord); start_t = last.t+1
        if checkpoint covers all steps → aggregate + return
    guard: ap.der_p must be non-empty (DER ordering source of truth)
    reset controllers/results; records = resumed.copy()

    NESTED _run_loop(arduino_iface):
        build VoltVarController(sgen_indices = ap.der_p.columns); configure()
        build SensitivityCoordinator, DERDynamics
        SEED DERDynamics: on resume from last record's q_applied/p_applied;
                          fresh → q=0, p=first profile row
        FOR t in range(start_t, T):
            select p_target (do NOT write it — the engine owns that write)
            write load profiles
            result = run_coordinated_timestep(…, coordination)   # Tiers 1+2 (Doc 3B)
            extract q/p applied, energy, coordination_active, q_saturated_count
            IF result.curtailment_needed:                        # Tier 3
                iteratively reduce P by 10%/iter, re-solve, stop when cleared or P=0
                update dynamics.p_prev ONCE (no intra-step dynamics.step)
            build TimestepRecord (Volt-Var block); publish; append
            every 96 steps: live CSV + progress log

    DISPATCH: dry_run → _run_loop(None); else → with ArduinoSerialInterface(port) as iface: _run_loop(iface)
    aggregate → ScenarioResult(volt_var_local|coord); log; return
```

---

## 3. Call-site map — what this scenario drives

| Entity | Kind | Called / drives |
|--------|------|-----------------|
| `run_scenario_4` | function | `benchmark_runner` `SCENARIO_REGISTRY[4]` (4A) **and** `[5]` (4B), via the `coordination` kwarg (Doc 9). |
| `_build_dynamics` | helper | `_run_loop` — constructs `DERDynamics` (Doc 4). |
| `VoltVarController` | class (Doc 1C) | Built in `_run_loop`; `.configure()` runs the handshake (Doc 1B). |
| `SensitivityCoordinator` | class (Docs 2/3) | Built in `_run_loop`. |
| `DERDynamics` | class (Doc 4) | Built + `.reset()`-seeded in `_run_loop`. |
| `run_coordinated_timestep` | function (Doc 3B) | The per-step engine — Tiers 1+2. |
| `ArduinoSerialInterface` | class (Doc 1B) | Context manager in the HIL dispatch (Doc 5 device). |
| `detect_violations` | fn (Doc 11) | Inside the curtailment loop. |
| `adapt_profiles`, `TimestepRecord`, `ScenarioResult.from_records` | infra (6A) | |
| `publish_fn.*`, `live_csv_rewrite_fn` | hooks | resume/streaming (Doc 9/10). |

---

## 4. Section A — Curtailment constants (L112–113)

```python
CURTAIL_STEP_FRAC: float = 0.10   # linear P reduction per iteration (fraction of p_target)
MAX_CURTAIL_ITERS: int   = 10     # 10 × 10% → floor at P=0 on final iteration
```

#### Line 112–113: The Tier 3 curtailment schedule

Curtailment reduces DER active power in **linear 10% steps** of the target, for at
most **10 iterations** — so iteration *k* sets `P = p_target − k·0.10·p_target`, and
iteration 10 reaches exactly `P = 0`. **Design rationale:** a simple, deterministic,
monotone schedule that is guaranteed to terminate (either the violation clears, or P
hits zero and the violation is declared structural). The 10%/10-iter choice trades
resolution against power-flow count — finer steps would clear violations with less
lost energy but cost more solves per curtailed timestep.

---

## 5. Section B — _build_dynamics (L118–148)

```python
def _build_dynamics(ctrl: VoltVarController, dt_s: float) -> DERDynamics:
    p_rated = ctrl.p_installed_mw
    q_max   = _vvc.Q_RATIO * p_rated                     # call-time Q_RATIO
    p_rated = np.where(p_rated > 0.0, p_rated, 1e-6)     # zero-sentinel guard
    q_max   = np.where(q_max   > 0.0, q_max,   1e-6)
    return DERDynamics(dt_s=dt_s, t95_q_s=MV_T95_Q_S, ramp_rate_p_frac_s=MV_RAMP_RATE_P_BASE,
                       q_max_mvar=q_max, p_rated_mw=p_rated)
```

#### Line 118–148: Construct DERDynamics with matched ceilings

Builds the `DERDynamics` (Doc 4) from the controller's DER array. **The critical
consistency point:** `q_max = _vvc.Q_RATIO · p_installed` uses the **call-time**
`Q_RATIO` (the `import … as _vvc` alias, Doc 1A/4), so the dynamics layer clips Q to
the *same* ceiling as the Q(V) curve and the coordinator — the single-source-of-truth
invariant. The **zero-sentinel guard** (`np.where(... > 0, ..., 1e-6)`) replaces any
non-positive `p_rated`/`q_max` with 1 µW/µVAr so `DERDynamics`'s strictly-positive
validation (Doc 4 §6) passes for an inert DER instead of raising. Uses the MV central
presets (`t95=10 s`, `ramp=0.5%/s`, Doc 4 §5). **Where used:** `_run_loop`.

---

## 6. Section C — Setup, resume & guards (L151–252)

#### Line 151–214: Signature, scenario-id, resume, skip-if-complete

```python
def run_scenario_4(net, profiles, network_id="unknown", port="/dev/ttyACM0",
                   dry_run=False, coordination=True, v_min=V_MIN, v_max=V_MAX,
                   publish_fn=None, enable_checkpointing=True, live_csv_rewrite_fn=None):
    ap = adapt_profiles(net, profiles); _T = len(ap.times)
    if publish_fn is not None: publish_fn.on_scenario_start("volt_var_coord" if coordination else "volt_var_local", ..., _T)
    scenario_id_for_resume = "volt_var_coord" if coordination else "volt_var_local"
    resumed_records = publish_fn.get_resume_records(scenario_id_for_resume) if (publish_fn and enable_checkpointing) else []
    start_t = (resumed_records[-1].t + 1) if resumed_records else 0
    if start_t >= _T_full and resumed_records:
        result = ScenarioResult.from_records(scenario_id=scenario_id_for_resume, records=resumed_records, ...); return result
```

The signature adds the HIL parameters (`port`, `dry_run`, `coordination`) to the
standard set. **The `coordination` flag threads everywhere:** it selects the
`scenario_id` (`volt_var_local`/`volt_var_coord`), which in turn is the
**checkpoint key** — so 4A and 4B resume from *separate* checkpoint files and never
cross-contaminate. The resume/skip-if-complete block is the standard pattern (6B-ii
§4).

#### Line 225–252: The DER-ordering guard and reset

```python
    der_idx = ap.der_p.columns
    if ap.der_p.empty:
        raise ValueError("Scenario 4 requires at least one profiled DER in ap.der_p ...")
    net.controller.drop(...); pp.reset_results(net)
    records: list[TimestepRecord] = resumed_records.copy()
    iface: Optional[ArduinoSerialInterface] = None
```

**The empty-`der_p` guard is load-bearing.** `ap.der_p.columns` is the *single source
of truth* for DER ordering; if it were empty, `VoltVarController` would fall back to
discovering all in-service sgens (Doc 1C), whose ordering may not match `ap.der_p` —
producing a `KeyError`/misalignment when the loop indexes profiles by
`ctrl.sgen_indices`. Failing loudly here prevents a silent DER-misalignment bug.

---

## 7. Section D — _run_loop: construction & DER-state seeding (L261–321)

#### Line 261–287: Build the controller, coordinator, and dynamics

```python
    def _run_loop(arduino_iface):
        nonlocal records
        ctrl = VoltVarController(net, interface=arduino_iface, sgen_indices=ap.der_p.columns, dry_run=dry_run)
        ctrl.configure()
        coordinator = SensitivityCoordinator(net, ctrl)
        dynamics    = _build_dynamics(ctrl, ap.dt_s)
        q_max_arr = np.where(_vvc.Q_RATIO * ctrl.p_installed_mw > 0.0, _vvc.Q_RATIO * ctrl.p_installed_mw, 1e-6)
```

**`_run_loop` is a closure** taking the (optional) Arduino interface — the pivot of
the HIL/dry-run dispatch (§11). It constructs the three engine components:
`VoltVarController` with **`sgen_indices = ap.der_p.columns`** (pinning DER ordering
to the profile), `.configure()` (the Arduino handshake, Doc 1B — a no-op in dry-run),
`SensitivityCoordinator`, and `DERDynamics`. `q_max_arr` is precomputed once (same
formula as `_build_dynamics`) for the per-step `q_saturated_count` check, avoiding
repeating the guarded arithmetic in the loop.

#### Line 289–312: DER-state seeding — the resume-continuity detail

```python
        if resumed_records:
            last = resumed_records[-1]
            q_init = last.q_applied_mvar.reindex(ctrl.sgen_indices).fillna(0.0).values if last.q_applied_mvar is not None else np.zeros(len(ctrl.sgen_indices))
            p_init = last.p_applied_mw.reindex(ctrl.sgen_indices).fillna(0.0).values if last.p_applied_mw is not None else ap.der_p.iloc[start_t].reindex(...).values
            dynamics.reset(q_init=q_init, p_init=p_init)
            logger.info("... DERDynamics state seeded from last checkpoint record.")
        else:
            p_init = ap.der_p.iloc[0].reindex(ctrl.sgen_indices).fillna(0.0).values
            dynamics.reset(q_init=0.0, p_init=p_init)
```

**The Scenario-4 resume-continuity detail (the analogue of OLTC's tap re-seed, 6B-ii
§4).** `DERDynamics` is *stateful* — its PT1 filter and ramp limiter depend on the
previous step's `q_prev`/`p_prev` (Doc 4). So on resume, it is **seeded from the last
checkpoint record's `q_applied`/`p_applied`** (reindexed to the controller's DER
order, NaN→0), so the PT1 and ramp continue seamlessly. Without this, a resumed run
would restart the dynamics from zero and inject a spurious transient at the resume
boundary. **Fresh run** (the `else`): `q_init = 0` (no prior Q injection) and
`p_init = first profile row` (so the t=0 ramp is checked against the true starting P,
not zero — Doc 4 §8). **Design rationale:** correct resume of a stateful physical
model requires restoring its state, not just its position.

---

## 8. Section E — The timestep loop: Tiers 1+2 (L322–384)

#### Line 314–354: Mandatory NR, profile write, and the engine call

```python
        runpp_kwargs = {"voltage_depend_loads": False, "algorithm": "nr"}
        for t in time_steps:
            p_target_row = ap.der_p.iloc[t].reindex(ctrl.sgen_indices).fillna(0.0).values   # [A] do NOT write it
            if not ap.load_p.empty: net.load.loc[ap.load_idx, "p_mw"] = ...; net.load.loc[..., "q_mvar"] = ...   # [B]
            result = run_coordinated_timestep(net=net, controller=ctrl, coordinator=coordinator, dynamics=dynamics,
                                              p_target=p_target_row, runpp_kwargs=runpp_kwargs, coordination=coordination)   # [C]
            if result.report_pre is not None and result.report_pre.any_violations: pre_violation_steps += 1
```

**`algorithm="nr"` is mandatory** — `run_coordinated_timestep`'s coordinator needs
the Newton–Raphson Jacobian in `net._ppc["internal"]["J"]`; `bfsw` would `KeyError`
(Docs 2/3B §15). **[A] deliberately does *not* write `net.sgen.p_mw`** — the comment
is explicit: `run_coordinated_timestep` **owns** the first `p_mw` write (its step [0],
Doc 3B) to keep the pre-PF `q=0` invariant consistent; writing it here would
double-apply. **[C]** delegates Tiers 1+2 to the engine (Doc 3B: write p → reset q=0
→ pre-PF → Arduino Q(V) → coordinate → dynamics.step → write → post-PF), returning a
`CoordinatorResult` (Doc 3A §B). `pre_violation_steps` counts uncontrolled violations.

#### Line 359–384: Extract the converged-state metrics

```python
            converged = result.post_pf_ok
            if converged and result.report_post is not None:
                q_ser = pd.Series(result.q_applied, index=ctrl.sgen_indices); p_ser = pd.Series(result.p_applied, ...)
                losses_mw_t = float(net.res_line["pl_mw"].sum() + net.res_trafo["pl_mw"].sum()); grid_import_mw_t = ...
                coordination_active_t = bool(coordination) and bool(np.any(np.abs(result.q_adjusted - result.q_initial) > 1e-6))
                q_saturated_count_t = int(np.sum(np.abs(result.q_applied) >= q_max_arr - 1e-6))
            else:
                q_ser = p_ser = None; losses_mw_t = grid_import_mw_t = None; coordination_active_t = q_saturated_count_t = None
```

Pulls the Volt-Var record fields from the `CoordinatorResult`. **`coordination_active_t`
is the coordination-rate signal (Doc 6A-ii §7):** `True` iff this is 4B *and* the
coordinator moved `q_initial` by >1e-6 MVAr on any DER — this is the per-step flag
`from_records` aggregates into `coordination_rate` (the paper's 9.1%/15.2% figures,
and the metric the quantisation finding inflates on the HIL path).
**`q_saturated_count_t`** counts DERs pinned at `±q_max` (using the same `q_max_arr`
ceiling `DERDynamics` clips to). Non-converged → all `None` (the `None`-means-N/A
convention, 6A-ii).

---

## 9. Section F — Tier 3: active-power curtailment (L386–454)

The scenario's own contribution beyond the engine — the last-resort tier.

```python
            curtail_exhausted_t = None
            active_report = result.report_post; active_p_ser = p_ser
            if result.curtailment_needed and converged:
                step_mw = CURTAIL_STEP_FRAC * p_target_row; curtail_exhausted_t = False; curtail_ok = True
                for curtail_iter in range(1, MAX_CURTAIL_ITERS + 1):
                    p_curtailed = np.maximum(p_target_row - curtail_iter * step_mw, 0.0)
                    net.sgen.loc[ctrl.sgen_indices, "p_mw"] = p_curtailed        # Q unchanged
                    try: pp.runpp(net, **runpp_kwargs); curtail_report = detect_violations(net)
                    except Exception: curtail_ok = False; curtail_exhausted_t = True; break
                    if not curtail_report.any_violations: break                  # cleared
                    if np.all(p_curtailed <= 0.0): curtail_exhausted_t = True; break   # structural
                else:
                    curtail_exhausted_t = True
                if curtail_ok:
                    dynamics.p_prev = p_curtailed.copy()                          # update ramp state ONCE
                    active_report = curtail_report; active_p_ser = pd.Series(p_curtailed, index=ctrl.sgen_indices)
                    losses_mw_t = ...; grid_import_mw_t = ...                     # refresh from post-curtailment PF
```

#### Line 386–454: Iterative active-power curtailment

**Tier 3 fires only when `result.curtailment_needed` is `True`** — i.e. when the
reactive tiers were exhausted (every mapped DER saturated, or a rank-deficient
coordination, per the post-PF check in Doc 3B). It then reduces DER active power in
10% steps (`p_curtailed = max(p_target − k·step_mw, 0)`), re-solving the power flow
each iteration, and stops when **either** the violation clears (`not any_violations`)
**or** P reaches zero (`np.all(p_curtailed <= 0)` → the violation is *structural*,
`curtail_exhausted=True`).

**Two design rationales worth emphasising:**

1. **Reactive-before-active is the whole hierarchy.** Reactive support (Tiers 1+2) is
   nearly free — it does not spill energy. Curtailment *loses* renewable generation,
   so it is the **last** resort, fired only when Q cannot fix the violation. This
   ordering is the physical justification for the three-tier design.
2. **No `dynamics.step()` inside the curtailment loop (L443–444).** The ramp state
   `p_prev` is updated **once**, at the end, to the final curtailed P — *not* stepped
   per iteration. The comment is explicit: calling `dynamics.step()` for each of up
   to 10 intra-timestep iterations would corrupt the ramp state (it would treat each
   10% reduction as a real timestep's ramp). Instead the loop finds the settled
   curtailed P, then sets `p_prev` to it directly, so the *next* timestep's ramp is
   checked against the true applied P. This is a subtle, correct handling of a
   stateful model inside an inner iteration.

On success, `active_report`/`active_p_ser` are replaced with the post-curtailment
state and the energy metrics refreshed. **Worked example:** `p_target = [0.5, 1.5]`,
`step_mw = [0.05, 0.15]`; iter 3 → `P = [0.35, 1.05]`; if the violation clears there,
`curtail_exhausted = False`; if it persists to iter 10 → `P = [0, 0]`,
`curtail_exhausted = True` (structural).

---

## 10. Section G — Record build & periodic log (L456–542)

```python
            if converged and active_report is not None:
                vm = net.res_bus["vm_pu"].copy(); ...
                ov_buses = vm.index[vm > v_max + VOLTAGE_EPSILON].tolist(); ...
            else: vm = ll = tl = pd.Series(dtype=float); ov_buses = ... = []
            rec = TimestepRecord(..., q_applied_mvar=q_ser, p_applied_mw=active_p_ser,
                curtailment_needed=result.curtailment_needed, curtail_exhausted=curtail_exhausted_t,
                p_target_mw=pd.Series(p_target_row, index=ctrl.sgen_indices), ...,
                coordination_active=coordination_active_t, q_saturated_count=q_saturated_count_t,
                hil_latency_ms=result.t_exchange_ms, t_total_ms=result.t_total_ms)
            if publish_fn is not None: publish_fn.on_timestep(rec)
            records.append(rec)
            if t % 96 == 0: ... live_csv_rewrite_fn(partial) ... logger.info(progress + pre/post/curtail counts)
```

#### Line 456–542: Build the full Volt-Var record; periodic streaming/log

Builds the `TimestepRecord` with **the complete Volt-Var block** (6A-ii §4):
`q_applied_mvar` (per-DER Series — unlike SVC's single value), `p_applied_mw`
(post-curtailment if Tier 3 ran, else post-dynamics), `p_target_mw` (raw, for
`curtailed_energy`), `curtailment_needed`/`curtail_exhausted`, `coordination_active`,
`q_saturated_count`, and — the HIL-specific — **`hil_latency_ms = result.t_exchange_ms`**
(the Arduino round-trip time, the basis of the timing analysis) and `t_total_ms`.
Violations are thresholded against `active_report`'s state with the standard `±ε`.
The periodic block streams the live CSV and logs progress with the pre/post violation
and curtailment counts. `on_timestep` streams **and** checkpoints each record.

---

## 11. Section H — HIL dispatch & result (L544–576)

```python
    if dry_run:
        logger.info("... dry_run=True — no serial port opened."); _run_loop(arduino_iface=None)
    else:
        with ArduinoSerialInterface(port=port) as iface:
            logger.info("... Arduino opened on %s.", port); _run_loop(arduino_iface=iface)
    result = ScenarioResult.from_records(scenario_id="volt_var_coord" if coordination else "volt_var_local", records=records, ...)
    logger.info("... Done. ... %d curtailment steps", ...)
    if publish_fn is not None: publish_fn.on_scenario_end(result)
    return result
```

#### Line 544–576: The dry-run/hardware split and final aggregation

**The dispatch is the scenario-level HIL switch.** In **dry-run**, `_run_loop(None)`
runs with no serial port — `VoltVarController` computes Q locally (Doc 1C), the
float64 path with no quantisation. In **hardware**, `ArduinoSerialInterface(port)` is
opened **as a context manager** (`with ... as iface`) so the port is *always* closed
even if the loop raises mid-run (Doc 1B), and `_run_loop(iface)` runs the HIL path —
the Arduino computes Q with the 4-decimal ASCII + float32 quantisation (Doc 5 §9).
**This one branch is the origin of the paper's central finding:** identical logic,
two Q-computation paths, differing only by the serial round-trip's quantisation
(Doc 1B §7). Finally, `from_records` aggregates (with the 4A/4B `scenario_id`), the
summary logs the curtailment-step count, and the result is returned.

---

## 12. Findings and flags

**Flagged, not edited:**

1. **Resume progress-% overshoot (L533), cosmetic.** Same as OLTC/SVC — the progress
   log divides absolute `t` by the post-resume range length, so it exceeds 100% after
   a resume. Simulation unaffected.

2. **`for…else` on the curtailment loop is defensively redundant (L428–429).** The
   `else: curtail_exhausted_t = True` runs only if the loop completes without `break`
   — but iteration 10 always yields `p_curtailed = max(p_target − p_target, 0) = 0`,
   which triggers the `np.all(p_curtailed <= 0)` break (L425–427) first. So the
   `else` is effectively unreachable belt-and-braces. Harmless; could be removed for
   clarity.

**Not bugs (documented for clarity):** the reactive-before-curtailment tier order is
the physical crux of the design; the single `dynamics.p_prev` update (no intra-step
`step()`) is the correct handling of the stateful ramp inside the curtailment
iteration; the DER-state seeding on resume is essential for continuity; the empty-`der_p`
guard prevents a real DER-misalignment bug; not writing `p_mw` in [A] preserves the
engine's `q=0` invariant.

---

## 13. Coverage checklist

Every element of `scenario_4_volt_var.py`, each confirmed documented.

| Element | Lines | Documented | Notes |
|--------|-------|:----------:|-------|
| `CURTAIL_STEP_FRAC` / `MAX_CURTAIL_ITERS` | 112–113 | ✅ | Tier 3 schedule |
| `_build_dynamics` | 118–148 | ✅ | matched Q ceiling, zero-sentinel |
| `run_scenario_4` signature (+HIL +checkpointing) | 151–163 | ✅ | `port`, `dry_run`, `coordination` |
| Adapt + start + resume + skip-if-complete | 185–214 | ✅ | 4A/4B separate checkpoints |
| Empty-`der_p` guard + reset | 225–252 | ✅ | DER-ordering source of truth |
| `_run_loop` construction (ctrl/coord/dynamics) | 261–287 | ✅ | drives Docs 1–4 |
| **DER-state seeding on resume** | 289–312 | ✅ | PT1/ramp continuity |
| Mandatory NR + [A]–[C] engine call | 314–357 | ✅ | Tiers 1+2 (Doc 3B) |
| [D] extract metrics (coordination_active, q_saturated) | 359–384 | ✅ | coordination-rate signal |
| **[E] Tier 3 curtailment** | 386–454 | ✅ | linear decay, single `p_prev` update |
| [F] record build | 456–494 | ✅ | full Volt-Var block + HIL latency |
| periodic live-CSV + progress log | 500–542 | ✅ | flagged % overshoot |
| HIL dispatch (dry-run/hardware) | 544–556 | ✅ | context-managed port |
| Result aggregation | 558–576 | ✅ | 4A/4B id |

**Also covered (not standalone entities):** the three-tier hierarchy and its
reactive-before-active justification, the delegation to `run_coordinated_timestep`,
the 4A/4B split threading through the `scenario_id`/checkpoint key, the dry-run vs
hardware quantisation origin, the DER-state resume seeding, and the no-intra-step-`step()`
curtailment ramp handling.

**Flags raised (not edited):** resume progress-% overshoot (L533, cosmetic);
defensively-redundant `for…else` on the curtailment loop (L428–429).

---

*End of Document 6D — the core scenario. Next per the queue: Document 6E — Scenario
Runner: OPF (`scenario_5_opf.py`), the optimal-power-flow benchmark (`runopp` on
CIGRE MV; SimBench MV structurally ill-conditioned), now genuinely streaming
`publish_fn`.*
