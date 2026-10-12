# TASK 4 — DOCUMENT 3B
## Sensitivity Coordinator — Walkthrough, Part 2: The Coordination Algorithm & Orchestration
### `sensitivity_coordinator.py` — `coordinate()` (L692–871) and `run_coordinated_timestep()` (L878–1106)

> **Source verified:** lines 692–1106 of the current `sensitivity_coordinator.py`
> were read directly before writing. If the file is edited, re-verify line
> numbers.
>
> **This is Part 2 of the coordinator walkthrough** (3A built the sensitivity
> matrix `X`; 3B uses it). It implements **Document 2 §7–§12** and the full
> single-step orchestration.
>
> **⚠ This document corrects an error in Document 2 §10.** Reading the live
> source shows the rank-deficiency branch does **not** pass `q_initial`
> through — the early `return` is commented out (L838), so `lstsq` proceeds
> regardless of rank. Document 2 §10 (which inherited the claim from the Task 1b
> summary) has been corrected accordingly. See §6 Step 8 and §8 below. This is
> exactly why the standing rule is *verify from source, not from summaries*.
>
> **Read alongside:** **Document 2** (theory), **Document 3A** (the machinery
> that produces `X`, `vm_pu_ppc`, `valid_target_mask`), **Document 4** (`DERDynamics`,
> called at step [5]), **Document 1C** (`VoltVarController` privates reused here).

---

## TABLE OF CONTENTS

1. [Overview](#1-overview)
2. [File logic flow — pseudocode (3B slice)](#2-file-logic-flow--pseudocode-3b-slice)
3. [Call-site map — where every entity is used](#3-call-site-map--where-every-entity-is-used)
4. [coordinate() — the 11-step algorithm (L692–871)](#4-coordinate--the-11-step-algorithm-l692871)
5. [run_coordinated_timestep() — the orchestration (L878–1106)](#5-run_coordinated_timestep--the-orchestration-l8781106)
6. [Findings and flags](#6-findings-and-flags)
7. [Coverage checklist](#7-coverage-checklist)

---

## 1. Overview

`coordinate()` turns the sensitivity matrix `X` (3A) into a reactive correction:
it predicts the post-`q_initial` voltages, finds residual violations, and solves
a minimum-norm least-squares problem for the coordinated adjustment
(Document 2 §7–§12). `run_coordinated_timestep()` is the production Scenario 4
loop — the richer replacement for `VoltVarController.run_timestep` (Doc 1C §6) —
that sequences all four "Items": pre-PF → local Q(V) (Item 2) → coordination
(Item 3) → dynamics (Item 4) → post-PF, and packages the result as a
`CoordinatorResult`.

The `coordination: bool` argument selects the two published scenarios: `False` =
**4A** (local Q(V) only, coordinator bypassed), `True` = **4B** (local +
coordinated — the paper's main contribution).

---

## 2. File logic flow — pseudocode (3B slice)

```
FUNCTION coordinate(q_initial):                        [Doc 2 §7–§12]
    1  validate q_initial length matches the DER count
    2  ASSERT net.sgen.q_mvar is still 0  (else raise — the double-count invariant)
    3  build the Jacobian blocks (3A)
    4  read PQ-ordered voltages + valid-target mask (3A)
    5  build the DER-column sensitivity matrix X (3A); if it failed → return q_initial (clipped)
    6  predict voltages after q_initial:  vm_predicted = vm + X·(q_initial/sn_mva)
    7  residual mask = valid AND finite AND outside the (dynamic) band
       none? → return q_initial (clipped); coordination did nothing this step
    8  S_viol = X[residual rows]; compute its rank
         rank-deficient? → log a throttled warning, set a (provisional) flag
                           [NOTE: the early return here is COMMENTED OUT — see §6]
    9  target = 1.0 − vm_predicted[residual];  dQ = lstsq(S_viol, target); pre-clip
   10  q_adjusted = clip(q_initial + dQ·sn_mva, ±q_max)
   11  provisional curtailment flag = (every mapped DER saturated at ±q_max)
    return q_adjusted

FUNCTION run_coordinated_timestep(net, controller, coordinator, dynamics, p_target, coordination):
    [0] write p_target to net.sgen.p_mw          (raw profile for the uncontrolled pre-PF)
    [1] reset net.sgen.q_mvar = 0                 (clean report_pre + coordinate() invariant)
    [2] pre-PF; build report_pre
        pre-PF failed/diverged? → return early (q=0, no post report)
    GATE: no violations pre-control? → hold Q=0, advance dynamics with q=0,
          reuse pre-PF as post, skip post-PF (don't let ramp zones disturb clean steps)
    [3] Item 2: read DER voltages → q_initial via dry-run (1A) or serial (1B);
        DO NOT write q to net (keep the q=0 invariant for coordinate())
    [4] Item 3: coordination? → q_adjusted = coordinate(q_initial); else → clip(q_initial)  (4A)
    [5] Item 4: (q_applied, p_applied) = dynamics.step(q_adjusted, p_target)  (PT1 + ramp)
    [6] write p_applied FIRST, then clamp q_applied (apparent-power cap reads p_applied),
        write q_clamped to net.sgen.q_mvar
    [7] post-PF; build report_post
    [8] authoritative curtailment_needed = post-PF has violations (if it converged)
    [9] build + return CoordinatorResult; log its summary
```

---

## 3. Call-site map — where every entity is used

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `coordinate` | method | `run_coordinated_timestep` L1040 (**the sole production caller**, only when `coordination=True`). Referenced in `der_dynamics`/`classify_s2_violations` docstrings. |
| `run_coordinated_timestep` | function | `scenario_4_volt_var.py` L297 (the Scenario 4 loop, both 4A and 4B). |
| `V_TARGET_PU` | constant | `coordinate` L841 (Step 9). |
| `MIN_VIOLATION_DV` | constant | `coordinate` L766–767 (Step 7). |
| `SATURATION_TOL` | constant | `coordinate` L865 (Step 11). |
| `_vd.V_MAX` / `_vd.V_MIN` | dynamic band | `coordinate` L766–767 (Step 7) — read from `violation_detector` at call time. |
| `self._q_max`, `self._sgen_idx`, `self._n_ders` | state | `coordinate` throughout; set in `__init__` (3A). |
| `self._rank_warning_count/_examples` | state | `coordinate` Step 8. |
| `self.curtailment_needed` | state | Set in `coordinate` Steps 5/7/8/11; **overwritten** by `run_coordinated_timestep` step [8] (L1082). |
| `controller._sgen_buses`, `controller._clamp_to_net_limits`, `controller._dry`, `controller._iface` | controller privates | `run_coordinated_timestep` (reuses Doc 1C internals — this is the "reimplemented loop"). |
| `dynamics.step` | method | `run_coordinated_timestep` L991, L1051 (Doc 4). |

---

## 4. coordinate() — the 11-step algorithm (L692–871)

The docstring numbers the algorithm 1–11; the walkthrough follows that numbering.
Every step maps to a Document 2 section.

#### Step 1 — Input validation (L713–718)

```python
        if len(q_initial) != self._n_ders:
            raise ValueError(f"q_initial length {len(q_initial)} != n_ders {self._n_ders}. Must match sgen_indices order: ...")
```

`q_initial` must have one entry per DER, in `sgen_indices` order. A length
mismatch means the caller aligned Q to the wrong DER set — caught immediately with
the correct ordering named. **Design rationale:** the coordinator indexes `X`
columns by DER position, so a wrong-length `q_initial` would silently misalign
every downstream operation; failing at entry is the only safe response.

#### Step 2 — Enforce the q_current = 0 invariant (L720–734)

```python
        q_current = self._net.sgen.loc[self._sgen_idx, "q_mvar"].values.astype(float)
        max_q_current = float(np.abs(q_current).max()) if len(q_current) > 0 else 0.0
        if max_q_current > 1e-9:
            raise RuntimeError("net.sgen.q_mvar is non-zero (max |q| = %.4f MVAr) before coordinate() ... reset ... to 0.0 and call runpp() ...")
```

**This is Document 2 §7 enforced in code.** The prediction `vm + X·dq` is only
correct if the pre-PF ran with `q_mvar = 0`, so that `dq = q_initial` (no
double-count). The method *reads back* the current `net.sgen.q_mvar` and raises a
`RuntimeError` if any exceeds `1e-9`. **Design rationale — assert, don't assume:**
a caller that reimplements the loop (as `run_coordinated_timestep` does) could
forget the reset; this turns a silent double-counted prediction into a loud,
diagnosable failure. `1e-9` (not exactly 0) tolerates float dust from a fresh
reset. **Worked example:** if a prior timestep left `q_mvar = −0.125` and the
reset was skipped, this raises rather than predicting a voltage shift that already
partly happened.

#### Steps 3–5 — Build blocks, read voltages, build X (L736–753)

```python
        blocks = self._build_jacobian_blocks()                                   # Step 3 (3A §D)
        vm_pu_ppc, ppc2pd, valid_target_mask = self._get_vm_pu_ppc(...)           # Step 4 (3A §F)
        X, der_col_positions = self._compute_der_sensitivity(blocks, ppc2pd)      # Step 5 (3A §G)
        if X is None:
            logger.warning("Sensitivity computation failed. Returning q_initial clipped to ±q_max.")
            self.curtailment_needed = False
            return np.clip(q_initial, -self._q_max, self._q_max)
```

Calls the three 3A machinery methods in order. **The `X is None` guard is the
safe-no-op contract (Doc 2 §11):** any factorisation failure inside
`_compute_der_sensitivity` (singular `J_PP`, ill-conditioned `J_red`, no DER on a
PQ bus) returns `(None, None)`, and here that means "the linear model can't be
trusted this timestep" → return `q_initial` clipped, coordination effectively
skipped. **Where used:** the three methods are covered in 3A §D/§F/§G.

#### Step 6 — Predict voltages after q_initial (L755–759)

```python
        dq_pu           = q_initial / self._net.sn_mva
        dV_from_initial = X @ dq_pu
        vm_pu_predicted = vm_pu_ppc + dV_from_initial
```

**Document 2 §8 in code.** With the invariant holding (`dq = q_initial`),
`dq_pu = q_initial / sn_mva` converts MVAr to per-unit on the system MVA base;
`X @ dq_pu` is the predicted voltage shift at every PQ bus; `vm_predicted` is the
linear estimate of the post-`q_initial` voltage — computed without a second power
flow. **Units:** `X` [pu/pu] · `dq_pu` [pu] = `dV` [pu]; added to `vm_pu_ppc`
[pu]. ✓ **Worked example** (Doc 2 §13 Case 2): `dV_from_initial[A] = 0.08·(−0.125)
= −0.010`; `vm_predicted[A] = 1.062 − 0.010 = 1.052` pu.

#### Step 7 — Identify residual violations (L761–784)

```python
        resid_mask = ( valid_target_mask
                     & np.isfinite(vm_pu_predicted)
                     & ( (vm_pu_predicted > _vd.V_MAX + MIN_VIOLATION_DV)
                       | (vm_pu_predicted < _vd.V_MIN - MIN_VIOLATION_DV) ) )
        if not resid_mask.any():
            self.curtailment_needed = False
            logger.debug("coordinate(): resid_mask empty — q_initial resolves violations. ...")
            return np.clip(q_initial, -self._q_max, self._q_max)
```

**Document 2 §8 in code.** A PQ row is a residual violation iff it is a valid
target (3A `valid_target_mask`), finite, and outside the band. **Three verified
details:** (1) `_vd.V_MAX`/`_vd.V_MIN` are read **dynamically** from
`violation_detector`, so a CLI-overridden band applies here too (the applied
V-band fix); (2) `MIN_VIOLATION_DV = 1e-3` ignores sub-milli-pu noise; (3) the
AND with `valid_target_mask` prevents the coordinator ever targeting a
non-physical ppc row. **The empty-`resid_mask` early return is the "Tier 1
sufficed" path** — `q_initial` is returned clipped, and 4B behaves like 4A this
step. **This is the exact locus of the quantisation finding (Doc 1B §7):** on the
dry-run path `resid_mask` is empty far more often than on the HIL path, because
4-decimal ASCII rounding nudges `vm_predicted` across the `V_MAX + MIN_VIOLATION_DV`
threshold.

#### Step 8 — Residual submatrix, rank diagnostic, and ⚠ the commented-out guard (L796–838)

```python
        S_viol  = X[resid_mask, :]
        n_resid = int(resid_mask.sum())
        rank = np.linalg.matrix_rank(S_viol)
        rank_limit = min(n_resid, self._n_ders)
        if rank < rank_limit:
            self._rank_warning_count += 1
            ... (capture first 5 examples; throttled warning on 1st and every 96th) ...
            # Rank-deficient: lstsq null-space components worsen voltages.
            # Pass q_initial through unchanged rather than corrupt it.
            self.curtailment_needed = True
        # return np.clip(q_initial, -self._q_max, self._q_max)   # ← THIS LINE
```

**Document 2 §10 in code — and here the code and its own comments diverge.** The
rank of `S_viol` is computed and compared to `rank_limit = min(n_resid, n_ders)`.
On a radial feeder this is routinely deficient (rank 3–4 despite 8+ DERs) because
DERs on the same branch produce collinear columns (Doc 2 §10). When deficient, the
code: increments the throttled warning counter, captures up to 5 diagnostic
examples, logs (first occurrence and every 96th), and sets
`self.curtailment_needed = True`.

**⚠ The critical detail:** the line that would *act* on rank deficiency —
`return np.clip(q_initial, ...)` — **is commented out (L838)**. So despite the
inline comment at L835–836 ("Pass q_initial through unchanged rather than corrupt
it"), execution **falls through to Step 9 and computes the `lstsq` correction
regardless of rank**. Two consequences:

1. **The comment is stale.** It describes the *old* early-return behaviour, which
   was removed. Code and comment disagree.
2. **The `curtailment_needed = True` at L837 is a dead assignment.** Because there
   is no early return, Step 11 (below) unconditionally overwrites
   `curtailment_needed`, and `run_coordinated_timestep` step [8] overwrites it
   *again* from the post-PF check. The Step-8 value never survives to affect
   anything.

**Is proceeding to `lstsq` actually correct?** Yes — and this is why the removal
was intentional (project memory). `np.linalg.lstsq(..., rcond=None)` computes the
**minimum-norm** solution via SVD, truncating the near-zero singular values that a
rank-deficient `S_viol` produces. The null-space components the comment feared are
*not* added — they are truncated. So the numerical behaviour is safe. **The bug is
documentary, not numerical:** the stale comment and the dead flag assignment
should be cleaned up (recommend: delete L835–838 or replace with a comment
explaining that `lstsq`'s SVD truncation handles rank deficiency). **Flagged, not
edited.** *(Document 2 §10 corrected to match this.)*

#### Steps 9–10 — Minimum-norm solve, convert, clip (L840–856)

```python
        dV_residual = V_TARGET_PU - vm_pu_predicted[resid_mask]
        dQ_corr_pu, _, _, _ = np.linalg.lstsq(S_viol, dV_residual, rcond=None)
        max_dQ_pu = 2.0 * self._q_max / self._net.sn_mva
        dQ_corr_pu = np.clip(dQ_corr_pu, -max_dQ_pu, max_dQ_pu)
        q_adjusted = np.clip(q_initial + dQ_corr_pu * self._net.sn_mva, -self._q_max, self._q_max)
```

**Document 2 §9 in code.** The target `dV_residual = V_TARGET_PU − vm_predicted`
aims the violated buses at **nominal (1.0 pu)**, giving margin (Doc 2 §9). `lstsq`
returns the minimum-norm `dQ_corr_pu` (the four return values are unpacked; only
the solution is kept). **Two clips:** the pre-clip `max_dQ_pu = 2·q_max/sn_mva`
bounds any single per-unit correction (a guard against a large excursion from a
near-singular `S_viol`) *before* scaling to MVAr; then `q_adjusted =
clip(q_initial + dQ_corr·sn_mva, ±q_max)` enforces each DER's physical ceiling.
**Units:** `dV_residual` [pu]; `lstsq` gives `dQ_corr_pu` [pu]; `·sn_mva` → MVAr.
**Worked example** (Doc 2 §13 Case 2): `dV_residual = 1.00 − 1.052 = −0.052`;
`lstsq([0.08,0.03], −0.052)` → `[−0.570, −0.214]` pu → `q_adjusted =
clip([−0.695, −0.214], ±0.125) = [−0.125, −0.125]` MVAr (both DERs to full
absorption; DER 2, previously idle, now contributes).

#### Step 11 — Provisional saturation flag (L858–871)

```python
        non_skipped = np.array([col is not None for col in der_col_positions], dtype=bool)
        if non_skipped.any():
            saturated = np.abs(q_adjusted[non_skipped]) >= self._q_max[non_skipped] - SATURATION_TOL
            self.curtailment_needed = bool(saturated.all())
        else:
            self.curtailment_needed = False
        return q_adjusted
```

**Document 2 §12 in code.** Over the non-excluded DERs (`der_col_positions is not
None`), a DER is "saturated" if `|q_adjusted| ≥ q_max − SATURATION_TOL`. If *all*
are saturated, reactive power is exhausted → provisional `curtailment_needed =
True`. **This assignment overwrites any value set in Steps 5/7/8** — confirming the
Step-8 rank flag is dead. It is **provisional** because `run_coordinated_timestep`
step [8] overrides it with the authoritative post-PF check. `SATURATION_TOL = 1e-6`
is the same tolerance the 208× quantisation figure is measured against (Doc 1B §7).
Returns `q_adjusted`.

---

## 5. run_coordinated_timestep() — the orchestration (L878–1106)

The production Scenario 4 loop. Signature (L878–886): `net`, `controller`,
`coordinator`, `dynamics`, `p_target`, `runpp_kwargs`, and **`coordination: bool`**
(the 4A/4B switch). Returns a `CoordinatorResult`.

#### Setup and enforced kwargs (L931–943)

```python
    kwargs = {"voltage_depend_loads": False}
    if runpp_kwargs:
        kwargs.update(runpp_kwargs); kwargs["voltage_depend_loads"] = False
    sgen_idx = controller.sgen_indices; p_inst = controller.p_installed_mw
    sgen_buses = controller._sgen_buses; empty_q = np.zeros(controller.n_ders)
```

`voltage_depend_loads=False` is force-set even if the caller passes `runpp_kwargs`
(mandatory for SimBench — Doc 1C). DER metadata and bus labels are pulled from the
controller (the shared source of truth), including the private `_sgen_buses`.

#### [0]–[1] Write raw P, reset Q to 0 (L945–954)

```python
    net.sgen.loc[sgen_idx, "p_mw"] = p_target          # [0] raw profile for pre-PF
    net.sgen.loc[sgen_idx, "q_mvar"] = 0.0             # [1] clean report_pre + q=0 invariant
```

**Design rationale — this function owns the `p_mw` write.** The docstring warns
callers *not* to write `net.sgen.p_mw` beforehand: `p_target` (raw profile) is
written for the uncontrolled pre-PF, and overwritten by `p_applied` at step [6].
The `q_mvar = 0` reset serves double duty: a clean uncontrolled `report_pre` *and*
the `coordinate()` q=0 invariant (Step 2). The comment "mirrors
`VoltVarController.run_timestep line 923`" is the parallelism from Doc 1C §6 (the
real line is now 1139 — the mirror comment's number is stale, flagged in Doc 1C).

#### [2] Pre-PF and the convergence early return (L956–983)

```python
    try: pp.runpp(net, **kwargs)
    except Exception as exc: pf_pre_ok = False; logger.error("Pre-PF runpp() raised: %s. Skipping timestep.", exc)
    report_pre = detect_violations(net)
    if not pf_pre_ok or not report_pre.converged:
        return CoordinatorResult(report_pre=report_pre, report_post=None, q_initial=empty_q, ..., curtailment_needed=False, post_pf_ok=False, ...)
```

Same double convergence condition as Doc 1C (raised *or* silently non-converged).
On failure, **`dynamics.step()` is deliberately NOT called** (the docstring L910–913:
state is not advanced for a failed timestep, so `p_prev`/`q_prev` stay valid for
the next step), and a Q=0 / `report_post=None` result is returned. **Return path 1
of 3.**

#### GATE — skip control on clean timesteps (L985–1005)

```python
    if not report_pre.any_violations:
        empty_q = np.zeros(controller.n_ders)
        _, p_applied = dynamics.step(q_target=empty_q, p_target=p_target)   # advance dynamics with q=0
        net.sgen.loc[sgen_idx, "p_mw"] = p_applied
        return CoordinatorResult(..., report_post=report_pre, ..., post_pf_ok=True)
```

**A performance and correctness gate.** If the uncontrolled network has no
violations, there is nothing to control: hold Q=0, reuse `report_pre` as the post
report (nothing changed), and **skip the post-PF entirely**. **Design rationale
(the comment L986–987):** running the Q(V) curve on a clean timestep would let the
ramp zones inject small reactive currents that *disturb* an already-healthy
network — so the gate suppresses that. Dynamics is still advanced with `q_target=0`
so `p_prev`/`q_prev` stay valid. **Return path 2 of 3.** *(This gate is also why
`resid_mask`-empty timesteps and clean timesteps are distinct: the gate catches
"no violation at all"; `coordinate()`'s Step 7 catches "local Q resolved it".)*

#### [3] Item 2 — get q_initial WITHOUT writing to net (L1007–1032)

```python
    vm_pu_at_ders = net.res_bus.loc[sgen_buses, "vm_pu"].values.astype(float)
    if controller._dry or controller._iface is None:
        q_initial = QVCharacteristic.compute_setpoints(pd.Series(vm_pu_at_ders, index=sgen_idx), pd.Series(p_inst, index=sgen_idx)).values
    else:
        try:
            q_initial, n_attempts = controller._iface.exchange_batched(vm_pu_at_ders, p_inst); n_retries = n_attempts - 1
        except (ArduinoProtocolError, SerialTimeoutError) as exc:
            logger.warning("Serial failure: %s. Falling back to local Q(V) ...")
            q_initial = QVCharacteristic.compute_setpoints(...).values; n_retries = controller._iface.max_retries
```

Reads DER-bus voltages and computes `q_initial` via the dry-run (1A) or hardware
(1B) path — **the same branch as `run_timestep` (Doc 1C), with one crucial
difference: `q_initial` is *not* written to `net.sgen.q_mvar`.** It must stay 0 so
`coordinate()`'s q=0 invariant holds at step [4]. On serial failure, both 1B
exceptions fall back to local Q(V) (not to Q=0 — a coordinated run degrades to the
local curve, which is still useful). **Where used:** `compute_setpoints` (1A §5),
`exchange_batched` (1B §6).

#### [4] Item 3 — coordinate, or the 4A bypass (L1034–1043)

```python
    if coordination:
        q_adjusted = coordinator.coordinate(q_initial)          # 4B
    else:
        q_adjusted = np.clip(q_initial, -coordinator._q_max, coordinator._q_max)   # 4A
```

**The 4A/4B split in one branch.** `coordination=True` → the full `coordinate()`
(§4); `coordination=False` → `q_adjusted` is just `q_initial` clipped to capacity,
with the coordinator entirely bypassed. Because `net.sgen.q_mvar` is still 0 (never
written since [1]), `coordinate()`'s Step 2 assertion passes. **Where used:**
`coordinate()` §4.

#### [5] Item 4 — DER dynamics (L1045–1054)

```python
    q_applied, p_applied = dynamics.step(q_target=q_adjusted, p_target=p_target)
```

`DERDynamics.step` (Doc 4) applies the PT1 filter to Q and ramp-limiting to P,
returning the *physically achievable* setpoints. `q_adjusted` is the target;
`q_applied` is what the DER can actually reach this step. **Where used:** covered
in Doc 4.

#### [6] Apply — the write-order dependency (L1056–1063)

```python
    net.sgen.loc[sgen_idx, "p_mw"] = p_applied                         # p FIRST
    q_clamped = controller._clamp_to_net_limits(q_applied)            # reads net.sgen.p_mw
    net.sgen.loc[sgen_idx, "q_mvar"] = q_clamped
```

**A subtle, load-bearing ordering.** `p_applied` is written to `net.sgen.p_mw`
**before** `_clamp_to_net_limits` runs, because the apparent-power cap
`|Q| ≤ √(sn² − p²)` (Doc 1C) reads `p_mw` *directly from `net.sgen`* — so it must
see `p_applied`, not the stale `p_target`. Writing p first ensures the cap uses the
correct, ramp-limited active power. Then the clamped Q is written. **Design
rationale:** getting this order wrong would compute the reactive headroom against
the wrong P and could over- or under-clamp Q. **Where used:**
`_clamp_to_net_limits` (Doc 1C §5).

#### [7]–[9] Post-PF, authoritative curtailment, result (L1065–1106)

```python
    try: pp.runpp(net, **kwargs)
    except Exception as exc: post_pf_ok = False; logger.warning("Post-PF runpp() raised: %s.", exc)
    report_post = detect_violations(net)
    if post_pf_ok:
        coordinator.curtailment_needed = report_post.any_violations      # [8] AUTHORITATIVE
    else:
        logger.warning("Post-PF failed. ... provisional value %s retained.", coordinator.curtailment_needed)
    result = CoordinatorResult(report_pre=report_pre, report_post=report_post, q_initial=q_initial, q_adjusted=q_adjusted, q_applied=q_clamped, p_target=p_target.copy(), p_applied=p_applied, curtailment_needed=coordinator.curtailment_needed, post_pf_ok=post_pf_ok, n_retries=n_retries, mode="coordinated" if coordination else "local", ...)
    logger.info("%s", result.summary())
```

The post-control power flow reflects physical reality (`p_applied`, `q_clamped`).
**Step [8] is where `curtailment_needed` becomes authoritative:** if the post-PF
converged, `curtailment_needed = report_post.any_violations` — overriding *every*
provisional value `coordinate()` set (Steps 8/11). If the post-PF failed, the
provisional value is retained with a warning. The `CoordinatorResult` records all
three Q stages and both P stages (3A §B), tags `mode` from `coordination`, and its
`summary()` is logged. **Return path 3 of 3.** **Where used:** `CoordinatorResult`
(3A §B); consumed by `scenario_4` to build `TimestepRecord`.

---

## 6. Findings and flags

**Flagged, not edited** (per the standing rule):

1. **Rank-deficiency early return is commented out (L838) while the inline comment
   (L835–836) still claims "pass `q_initial` through unchanged".** Code and comment
   disagree. The current behaviour — proceed to `lstsq` — is *numerically correct*
   (minimum-norm SVD truncates the null space), so the removal was intentional, but
   the comment is stale. **Recommend:** replace L835–838 with a comment stating that
   `lstsq`'s SVD truncation handles rank deficiency, and delete the dead assignment.

2. **`curtailment_needed = True` (L837) is a dead assignment.** With no early
   return, Step 11 (L862–869) unconditionally overwrites it, and
   `run_coordinated_timestep` step [8] (L1082) overwrites it again from the
   authoritative post-PF check. The Step-8 value never propagates. **Recommend:**
   remove it (the rank warning is the useful diagnostic; the flag is not).

3. **Document 2 §10 corrected.** Its earlier claim that the coordinator "returns
   `q_initial` unchanged" on rank deficiency (inherited from the Task 1b summary)
   is wrong for the current source. Corrected to: the rank check is now a
   *diagnostic* (throttled warning + captured examples); `lstsq` proceeds and its
   minimum-norm SVD solution handles the rank-deficient null space.

**Not a bug (documented for clarity):** the three-way `curtailment_needed`
overwrite chain (Step 8 → Step 11 → post-PF [8]) is by design — the post-PF check
is authoritative, the earlier settings provisional. Only the Step-8 rank flag is
genuinely dead; the Step-11 flag is a meaningful provisional value when the post-PF
fails to converge (L1086 retains it).

---

## 7. Coverage checklist

Every named entity in the 3B range, each confirmed documented.

| Entity | Lines | Kind | Documented | Where-used stated |
|--------|-------|------|:----------:|:-----------------:|
| `coordinate` | 692–871 | method | ✅ (11 steps) | ✅ |
| — Step 1 input validation | 713–718 | — | ✅ | — |
| — Step 2 q=0 invariant | 720–734 | — | ✅ | — |
| — Steps 3–5 blocks/voltages/X | 736–753 | — | ✅ | — |
| — Step 6 prediction | 755–759 | — | ✅ | — |
| — Step 7 residual mask | 761–784 | — | ✅ | — |
| — Step 8 rank diagnostic (+ commented-out return) | 796–838 | — | ✅ (flagged §6.1/§6.2) | — |
| — Steps 9–10 lstsq + clips | 840–856 | — | ✅ | — |
| — Step 11 saturation flag | 858–871 | — | ✅ | — |
| `run_coordinated_timestep` | 878–1106 | function | ✅ (steps [0]–[9]) | ✅ |
| — [0]–[1] write P / reset Q | 945–954 | — | ✅ | — |
| — [2] pre-PF + early return | 956–983 | — | ✅ | — |
| — GATE (clean timestep) | 985–1005 | — | ✅ | — |
| — [3] Item 2 q_initial | 1007–1032 | — | ✅ | — |
| — [4] Item 3 coordinate / 4A bypass | 1034–1043 | — | ✅ | — |
| — [5] Item 4 dynamics | 1045–1054 | — | ✅ | — |
| — [6] apply (write-order) | 1056–1063 | — | ✅ | — |
| — [7]–[9] post-PF / curtailment / result | 1065–1106 | — | ✅ | — |
| `V_TARGET_PU` / `MIN_VIOLATION_DV` / `SATURATION_TOL` | 121–124 | constants (used here) | ✅ | ✅ |

**Also covered (not standalone entities):** the q=0 invariant enforcement, the
`X is None` safe-no-op, the dynamic `_vd.V_MAX`/`_vd.V_MIN` read, the empty-`resid_mask`
"Tier 1 sufficed" path, the clean-timestep GATE, the `q_initial`-not-written
subtlety, the p-before-q write-order dependency, and the three-way
`curtailment_needed` overwrite chain.

**Flags raised (not edited):** commented-out rank-deficiency return (L838); stale
"pass through unchanged" comment (L835–836); dead `curtailment_needed = True`
(L837). **Document 2 §10 corrected** to match the verified behaviour.

---

*End of Document 3B — and of the `sensitivity_coordinator.py` set (3A + 3B). Next
per the queue: Document 4 — DER Dynamics (`der_dynamics.py`, the PT1 filter and
ramp limiting called at step [5] above).*
