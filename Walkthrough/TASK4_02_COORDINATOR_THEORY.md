# TASK 4 — DOCUMENT 2
## Sensitivity Coordinator — Theory Foundations
### The mathematics behind `sensitivity_coordinator.py`

> **This is a theory document, not a line-by-line walkthrough.** It derives the
> mathematics the coordinator implements — the Newton–Raphson Jacobian, the
> Schur complement, the dV/dQ sensitivity matrix, the double-count correction,
> the minimum-norm least-squares solve, and rank deficiency on radial feeders —
> so that **Document 3** (the line-by-line walkthrough of the file) can refer
> back to each result rather than re-deriving it. Where a symbol maps to a
> specific function, the function is named (see §14, the concept→implementation
> map), but the *code* is walked in Doc 3.
>
> **Two mandatory-section adaptations** (per Master Index §5): a theory document
> has no single line range to enumerate, so the "call-site map" is replaced by
> the §14 concept→implementation map, and the "coverage checklist" is deferred
> to Doc 3 (which walks every line). All derivations here are transcribed from
> the live function bodies (`_build_jacobian_blocks`, `_compute_der_sensitivity`,
> `coordinate`) and constants, verified against source — not from summaries.
>
> **Read alongside:** Master Index §7–§8; **Document 1A** (the local Q(V) law
> whose output `q_initial` is this coordinator's input); **Document 1C** (the
> loop; note the coordinator replaces `run_timestep` with
> `run_coordinated_timestep`). **Document 3** is the code walkthrough this
> theory underpins.

---

## TABLE OF CONTENTS

1. [Overview and scope](#1-overview-and-scope)
2. [The physical question and the two-tier architecture](#2-the-physical-question-and-the-two-tier-architecture)
3. [The linear approximation: the NR Jacobian as the operating-point slope](#3-the-linear-approximation)
4. [The power-flow equations and the Jacobian block structure](#4-the-power-flow-equations-and-the-jacobian-block-structure)
5. [The Schur complement — projecting out the angle sub-system](#5-the-schur-complement)
6. [The voltage sensitivity matrix X = dV/dQ](#6-the-voltage-sensitivity-matrix-x--dvdq)
7. [The double-count problem and the q_current = 0 invariant](#7-the-double-count-problem-and-the-q_current--0-invariant)
8. [Linear prediction and residual identification](#8-linear-prediction-and-residual-identification)
9. [The minimum-norm least-squares correction](#9-the-minimum-norm-least-squares-correction)
10. [Rank deficiency on radial feeders](#10-rank-deficiency-on-radial-feeders)
11. [Numerical safeguards](#11-numerical-safeguards)
12. [Saturation and curtailment signalling](#12-saturation-and-curtailment-signalling)
13. [Worked numerical example](#13-worked-numerical-example)
14. [Concept → implementation map (for Document 3)](#14-concept--implementation-map-for-document-3)
15. [Assumptions and limitations](#15-assumptions-and-limitations)

---

## 1. Overview and scope

The `SensitivityCoordinator` is the **second tier** of Scenario 4's Volt-Var
control. The first tier — the local Q(V) characteristic (Document 1A) — lets each
DER react to *its own* bus voltage with no communication. That is cheap and
distributed, but purely local: a DER cannot know that a *neighbour's* bus is
still in violation after everyone has done their local best. The coordinator adds
a global correction: using the network's own linearised voltage-to-reactive-power
sensitivity, it computes — in one shot, across all DERs — how much *additional*
reactive power each DER should contribute to pull any *residual* violated buses
back inside the band.

Crucially, it does this **without re-running the nonlinear power flow**. A single
converged `runpp()` has already produced the Newton–Raphson Jacobian; the
coordinator reuses that Jacobian as a first-order model of the network, solves a
small linear system, and applies the result. The entire justification rests on
four mathematical steps, derived below:

1. the NR Jacobian is the slope of the power-flow map at the operating point (§3);
2. its block structure separates active/reactive and angle/magnitude (§4);
3. a Schur complement collapses it to a pure dV/dQ sensitivity (§5–§6);
4. a minimum-norm least-squares solve turns "how far are we out?" into "how much
   Q should each DER add?" (§9), with care taken over double-counting (§7) and
   rank deficiency (§10).

---

## 2. The physical question and the two-tier architecture

**Tier 1 — local Q(V) (Document 1A, `q_initial`).** Each DER *i* sets
`q_initial[i]` from its own bus voltage via the VDE-AR-N 4110 Bild 8 curve. This
is what the Arduino computes on the HIL path. It resolves *most* violations most
of the time — indeed at `Q_RATIO = 0.30` it resolves *all* of them and the
coordinator never fires.

**Tier 2 — coordination (this document, `q_adjusted`).** After the pre-control
power flow, the coordinator asks: *given the voltages that would result once
`q_initial` is applied, which buses are still outside the band, and what
coordinated set of Q adjustments would fix them with the least total reactive
effort?* Its output `q_adjusted` is `q_initial` plus a correction `dQ_corr`.

**The pipeline** (from the module docstring, verified):

```
[Pre-PF]  net.sgen.q_mvar = 0 ; runpp()          → operating point + Jacobian
[Tier 1]  Arduino / dry-run   → q_initial[n_ders] (local VDE-AR-N 4110)
[Tier 2]  coordinate()        → q_adjusted[n_ders] (global residual correction)
[Apply]   net.sgen.q_mvar = q_adjusted
[Post-PF] runpp()                                 → authoritative result
```

The coordinator sits between Tier 1 and application. The `coordination: bool`
flag selects 4A (local only, coordinator bypassed) versus 4B (local + coordinated
— the paper's main contribution).

---

## 3. The linear approximation

The power flow is a nonlinear map from bus power injections to bus voltages.
Solving it *exactly* for a proposed set of Q injections would mean re-running
Newton–Raphson to convergence — many iterations, once per candidate. That is far
too expensive to do inside a per-timestep control loop over a year of 15-minute
steps.

Instead the coordinator linearises around the current operating point. Any smooth
map `f(x)` near a point `x₀` is approximated by its first-order Taylor term:

```
f(x₀ + Δx) ≈ f(x₀) + J·Δx ,   where   J = ∂f/∂x |_{x₀}
```

For the power flow, `x` is the vector of voltage angles and magnitudes and `f`
is the vector of power injections; the slope `J` is exactly the **Newton–Raphson
Jacobian**. And here is the key economy: pandapower *already computed and stored*
that Jacobian while solving the pre-control power flow. A converged `runpp()`
(with the `nr` or `iwamoto_nr` algorithm — **not** `bfsw`, which is
Jacobian-free) leaves it in `net._ppc["internal"]["J"]` as a sparse matrix. The
coordinator reads it rather than recomputing it: the linear model is free.

The approximation is first-order, so it is accurate only near the operating
point. Across the `0.95–1.05` pu band the error stays under ~5% (§6, §15), which
is well inside the margin the correction targets (it aims at 1.0 pu, not the band
edge — §9).

---

## 4. The power-flow equations and the Jacobian block structure

At each bus the injected active and reactive power are nonlinear functions of the
voltage angles `θ` and magnitudes `|V|` of all buses. Newton–Raphson linearises
the mismatch equations as:

```
┌      ┐   ┌                 ┐ ┌            ┐
│ ΔP   │ = │  J_PP    J_PQ   │ │ Δθ         │
│ ΔQ   │   │  J_QP    J_QQ   │ │ Δ(|V|/|V|) │
└      ┘   └                 ┘ └            ┘
```

with the four sub-blocks

```
J_PP = ∂P/∂θ          J_PQ = ∂P/∂(|V|/|V|)
J_QP = ∂Q/∂θ          J_QQ = ∂Q/∂(|V|/|V|)
```

**Dimensions and bus-type bookkeeping.** For a network with `n_pv_pq` non-slack
buses (PV + PQ) and `n_pq` PQ buses, `J` is
`(n_pv_pq + n_pq) × (n_pv_pq + n_pq)`:

```
              Δθ  (n_pv_pq cols)   Δ(|V|/|V|)  (n_pq cols)
            ┌────────────────────┬──────────────────────────┐
ΔP  (n_pv_pq rows) │   J_PP      │        J_PQ              │
ΔQ  (n_pq   rows)  │   J_QP      │        J_QQ              │
            └────────────────────┴──────────────────────────┘
```

The active-power mismatch equations exist for **all non-slack buses**; the
reactive mismatch equations only for **PQ buses** (a PV bus holds its voltage
magnitude, so its Q is whatever it must be — no `ΔQ` equation, no `Δ|V|` unknown).
This asymmetry is why `J_PP` is `n_pv_pq × n_pv_pq` while `J_QQ` is
`n_pq × n_pq`.

The coordinator slices these four blocks out of the stored sparse `J` **without
ever densifying the full matrix** (`.toarray()` is never called on `J` itself),
using the ppc bus-type array to build the PQ / PV+PQ masks and to record the
sorted ppc indices of the PQ buses — the row/column ordering that every
downstream array aligns to.

---

## 5. The Schur complement

We want only one relationship: **how bus voltage *magnitudes* at PQ buses respond
to *reactive* power injections**, i.e. `∂|V|/∂Q`. But the raw Jacobian couples
angles and magnitudes together — `J_PQ` and `J_QP` are nonzero. The angle
sub-system must be eliminated.

**Derivation.** We inject reactive power at fixed active power, so `ΔP = 0`. From
the top block row:

```
ΔP = J_PP·Δθ + J_PQ·Δv = 0            (Δv ≜ Δ(|V|/|V|))
  ⇒  Δθ = −J_PP⁻¹·J_PQ·Δv
```

Substitute into the bottom block row:

```
ΔQ = J_QP·Δθ + J_QQ·Δv
   = J_QP·(−J_PP⁻¹·J_PQ·Δv) + J_QQ·Δv
   = (J_QQ − J_QP·J_PP⁻¹·J_PQ)·Δv
   ≜ J_red·Δv
```

The matrix

```
J_red = J_QQ − J_QP·J_PP⁻¹·J_PQ          shape (n_pq × n_pq)
```

is the **Schur complement** of `J_PP` in `J`. It is the reduced Jacobian relating
*only* reactive injections to *only* PQ-bus voltage-magnitude changes, with the
angle coupling folded in exactly (no approximation in the elimination itself —
this is algebraic, not a further linearisation).

**Sparsity note (why the two factorisations differ).** `J_PP` is genuinely sparse
with no fill-in, so it is LU-factorised with a *sparse* solver (`splu`) and used
to form `J_PP⁻¹·J_PQ`. But the Schur complement *fills in* — `J_red` is dense — so
it is factorised with a *dense* solver (`lu_factor`). Choosing the right solver
for each is a real efficiency point, not a stylistic one.

---

## 6. The voltage sensitivity matrix X = dV/dQ

Inverting the reduced relation gives the object the coordinator actually uses:

```
Δv = J_red⁻¹·ΔQ            ⇒     X ≜ J_red⁻¹ = ∂|V|/∂Q
```

`X[i,j]` answers precisely: **"if DER *j* changes its reactive injection by 1 pu,
the voltage magnitude at PQ-bus *i* changes by `X[i,j]` pu."**

**Units and the |V| ≈ 1 approximation.** The pandapower NR Jacobian relates
`ΔP_pu / ΔQ_pu` to `Δθ` (radians) and `Δ(|V|/|V|)` (dimensionless). So `X` maps
`ΔQ_pu → Δ(|V|/|V|)`. Under the standard operating-point approximation `|V| ≈ 1.0`
pu, `Δ(|V|/|V|) ≈ Δ|V|_pu`, so `X` is read directly as **pu-volt per pu-power**
(dimensionless). The `|V| ≈ 1` step is where the <5% error across `0.95–1.05` pu
enters.

**Targeted DER-column solve (why we never invert `J_red` fully).** The coordinator
needs only the columns of `X` corresponding to buses that actually host a
controllable DER. So it builds a unit-column right-hand side `E_der` (a 1.0 in
each valid DER's PQ-row) and solves the *single* `J_red` factorisation against all
DER columns at once:

```
X_valid = J_red⁻¹·E_der          shape (n_pq × n_valid)
```

scattering the results into the full `X` of shape `(n_pq × n_ders)` with excluded
DERs as zero columns. **DER exclusion:** a DER whose bus is out-of-service, out of
range, or **not a PQ bus in this operating point** (e.g. it became a PV bus) is
dropped — its column position is `None`, a warning is logged, and its `q_initial`
passes through the coordinator unchanged. This keeps the sensitivity matrix
honest: only buses the linear model can actually speak about get a column.

---

## 7. The double-count problem and the q_current = 0 invariant

This is the subtlest point in the whole method, and it is enforced by a hard
runtime check.

**The linear model predicts *changes relative to the operating point where the
Jacobian was computed*.** That operating point is the pre-control power flow. The
coordinator must predict the voltage **after `q_initial` is applied**, so the
relevant reactive change is:

```
Δq = q_initial − q_at_pre_PF
```

If the pre-control power flow had been run with the DERs *already* injecting some
reactive power, then `q_at_pre_PF ≠ 0`, and using `Δq = q_initial` would
**double-count** that pre-existing injection — predicting a voltage shift that
partly already happened. The prediction would be wrong by exactly `X·q_at_pre_PF`.

**The invariant that removes the problem:** the pre-control power flow is run with
`net.sgen.q_mvar = 0` (the reset in `VoltVarController.run_timestep` L1139, Doc
1C, and mirrored in `run_coordinated_timestep`). Then `q_at_pre_PF = 0` and
`Δq = q_initial` cleanly. The coordinator *asserts* this at entry: it reads the
current `net.sgen.q_mvar`, and if any `|q| > 1e-9` it raises a `RuntimeError`
telling the caller to reset q to 0 and re-run the pre-PF first. The invariant is
not assumed — it is checked, so a mis-ordered caller fails loudly instead of
silently producing a double-counted prediction.

With the invariant holding, the per-unit reactive change is simply
`dq_pu = q_initial / sn_mva` (converting MVAr to per-unit on the system MVA base).

---

## 8. Linear prediction and residual identification

With `X` and `dq_pu` in hand, the predicted post-`q_initial` voltage at every PQ
bus is:

```
dV_from_initial = X · dq_pu
vm_predicted    = vm_pre + dV_from_initial
```

`vm_predicted` is what the voltages *would* be once Tier 1's local Q is applied —
computed linearly, without a second power flow. The coordinator then flags
**residual** violations: buses still outside the band *in the predicted state*.

```
resid_mask =  valid_target_mask
            & isfinite(vm_predicted)
            & ( (vm_predicted > V_MAX + MIN_VIOLATION_DV)
              | (vm_predicted < V_MIN − MIN_VIOLATION_DV) )
```

Three points, each verified against source:
- **`V_MAX` / `V_MIN` are read dynamically** from `violation_detector` (`_vd.V_MAX`,
  `_vd.V_MIN`), so a run that overrides the band (via the CLI wizard) uses the
  same band here as everywhere else — the coordinator does not carry its own
  hardcoded 0.95/1.05 (this is the applied fix; the old hardcoded `V_BAND_*` is
  gone).
- **`MIN_VIOLATION_DV = 1e-3`** ignores violations smaller than 1 milli-pu as
  numerical noise — the linear model is not trusted to sub-milli-pu resolution.
- **`valid_target_mask`** restricts attention to PQ buses the sensitivity model
  can actually speak about (finite, in-band-eligible).

If `resid_mask` is empty, Tier 1 already resolved everything: the coordinator
returns `q_initial` unchanged (clipped to ±q_max) and 4B behaves like 4A this
timestep. This is exactly the regime the quantisation finding turns on — on the
dry-run path `resid_mask` is empty far more often than on the HIL path, because
the 4-decimal ASCII rounding nudges `vm_predicted` across the threshold (Doc 1B
§7).

---

## 9. The minimum-norm least-squares correction

When residual violations exist, the coordinator solves for the reactive
correction that removes them. Restrict the sensitivity matrix to the residual
(violated) buses:

```
S_viol = X[resid_mask, :]           shape (n_resid × n_ders)
```

and set the target voltage change as the gap from the predicted voltage to the
desired voltage:

```
dV_residual = V_TARGET_PU − vm_predicted[resid_mask]
```

with **`V_TARGET_PU = 1.00`** — the correction aims the violated buses back
toward *nominal*, not merely to the band edge, giving margin so the result does
not sit exactly on the limit. We then want `dQ_corr_pu` such that

```
S_viol · dQ_corr_pu ≈ dV_residual
```

**Why least squares, not a direct solve.** `S_viol` is `(n_resid × n_ders)` and
typically has **more DERs than violated buses** — the system is *underdetermined*.
A square solve is impossible; and among the infinitely many exact solutions we
want a *specific* one. `np.linalg.lstsq(S_viol, dV_residual, rcond=None)` returns
the **minimum-norm** solution — the correction of smallest total magnitude — which
distributes the reactive effort across *all* DERs rather than saturating one. That
is both physically sensible (spread the burden) and numerically robust (the
smallest-norm answer is the well-defined one).

The result is converted back to MVAr and applied on top of `q_initial`, with two
clips:

```
dQ_corr_pu = clip(dQ_corr_pu, ±max_dQ_pu)                     # pre-clip guard
q_adjusted = clip(q_initial + dQ_corr_pu · sn_mva, ±q_max)    # hard capability cap
```

The pre-clip (`max_dQ_pu`, on the order of `2·q_max/sn_mva`) bounds any single
correction before it is scaled back to MVAr — a guard against a large least-squares
excursion from a near-singular residual system. The final clip enforces each DER's
physical `±q_max = Q_RATIO · P_installed` ceiling.

---

## 10. Rank deficiency on radial feeders

This is where the network's *topology* enters, and it is physics, not a bug.

**The observation.** On a rural radial feeder, `S_viol` is routinely
**rank-deficient**: `rank(S_viol) = 3–4` even when there are `8+` DERs and several
violated buses. `rank < min(n_resid, n_ders)`.

**Why.** DERs sitting on the *same radial branch* have **collinear sensitivity
columns**. Voltage on a radial feeder rises monotonically with distance from the
transformer, and a reactive injection anywhere upstream shifts every downstream
bus in a fixed proportion set by the series impedance to the branch point. So two
DERs on the same branch move the violated buses in *almost the same direction* —
their columns of `X` are (near-)linearly dependent. The number of *independent*
directions is set by the number of distinct branch paths to the violated buses,
which on a radial feeder is small (3–4), not by the DER count. (This is the
empirically-observed approximate low-rank structure of distribution-network
voltage-sensitivity matrices; it is a known property of radial topologies.)

**The hazard.** A rank-deficient `S_viol` has a non-trivial null space. `lstsq`'s
minimum-norm solution is safe on the *range*, but any component the caller adds in
the null space produces reactive adjustments that cancel at the violated buses
while still perturbing *other* buses — corrupting voltages elsewhere for no
benefit at the target.

**The guard (verified against source — see Document 3B §4 Step 8).** The
coordinator computes `rank = np.linalg.matrix_rank(S_viol)` and
`rank_limit = min(n_resid, n_ders)`. When `rank < rank_limit`, it logs a
**throttled diagnostic** warning (first occurrence and every 96th, capturing the
first five instances) and sets a provisional `curtailment_needed` flag. It then
**proceeds to the `lstsq` solve regardless of rank** — the historical early
return that would have passed `q_initial` through is present in the source only
as a *commented-out* line. This is deliberate: `np.linalg.lstsq(..., rcond=None)`
returns the **minimum-norm** solution via SVD, truncating the near-zero singular
values a rank-deficient `S_viol` produces, so the null-space components the old
early-return guarded against are *not* added — they are truncated. The rank check
is therefore now a **diagnostic**, not a gate.

> **Correction note.** Earlier drafts of this section (and the Task 1b summary
> they drew on) stated that the coordinator "returns `q_initial` unchanged" on
> rank deficiency. That is **not** what the current source does — the early
> return is commented out and `lstsq` proceeds. Document 3B §6 flags the stale
> inline comment (L835–836) and the dead `curtailment_needed = True` assignment
> (L837, overwritten by Step 11 and again by the post-PF check) that survive from
> the removed behaviour. The numerical result is safe; the leftover comment and
> assignment are documentation debt, flagged for cleanup.

**Interpretation.** Rank deficiency means some violated buses are only weakly
controllable from the available DERs — no reactive redistribution can fix them, so
the honest signal is `curtailment_needed = True` (active-power curtailment, Tier
3 in Scenario 4), not a Q correction that would do more harm than good.

---

## 11. Numerical safeguards

The method inverts matrices built from a network that can be near a singular
operating point, so several guards protect the result:

- **`J_PP` structural singularity** — `splu` raises; caught → the coordinator
  returns `(None, None)` and `q_initial` passes through (clipped).
- **`J_red` ill-conditioning** — after the dense LU of `J_red`, the reciprocal
  condition number is estimated *for free* from the U-diagonal:
  `rcond_est = min|diag(U)| / max|diag(U)|`. If `rcond_est < RCOND_THRESHOLD =
  1e-12`, the network is near voltage collapse and any inverse would be numerical
  garbage → return `(None, None)`, pass `q_initial` through. This costs nothing
  (the diagonal is already computed by the factorisation) and catches the
  dangerous case the sparse `J_PP` check can miss (a near-singular `J_PP` produces
  a garbage — but non-raising — fill-in block, which surfaces as an
  ill-conditioned `J_red`).
- **Non-finite predicted voltages** — excluded from `resid_mask` via the
  `isfinite` term, so a NaN from an edge case cannot flag a spurious violation.
- **Clipping** — the pre-clip on `dQ_corr_pu` and the hard `±q_max` clip on
  `q_adjusted` (§9) bound the output regardless of the linear system's
  conditioning.

The design philosophy throughout: **when the linear model cannot be trusted, do
nothing (pass `q_initial` through) rather than apply a correction computed from a
bad matrix.** Every failure path is a safe no-op, not a crash and not a guess.

---

## 12. Saturation and curtailment signalling

After computing `q_adjusted`, the coordinator sets a **provisional**
`curtailment_needed`:

```
saturated = |q_adjusted[non_skipped]| ≥ q_max − SATURATION_TOL
curtailment_needed = saturated.all()
```

with **`SATURATION_TOL = 1e-6`** MVAr — the float-comparison tolerance for "at the
ceiling". If *every* non-excluded DER is pinned at `±q_max`, reactive power is
exhausted and the only remaining lever is active-power curtailment, so
`curtailment_needed = True`. It is **provisional** because
`run_coordinated_timestep` overrides it with the *authoritative* post-control
power-flow violation check — the linear prediction proposes, the real power flow
disposes.

**`SATURATION_TOL` is the same tolerance that carries the quantisation finding**
(Doc 1B §7): the worst-case 4-decimal ASCII rounding error, `2.08×10⁻⁴` MVAr, is
~208× this `1e-6` tolerance, which is why the coordinator's saturation/coordination
gate reacts to hardware-path rounding that the dry-run path never produces.

---

## 13. Worked numerical example

A deliberately small case to make the whole pipeline concrete. **Two DERs**, system
base `sn_mva = 1.0` MVA, `Q_RATIO = 0.25`, both rated `P = 0.5` MW so
`q_max = 0.25·0.5 = 0.125` MVAr each. After the pre-control power flow, bus A
(hosting DER 1) reads `vm = 1.062` pu — an overvoltage (band top 1.05).

**Tier 1 (local Q(V), Doc 1A).** At `vm = 1.062 > U4 = 1.04`, DER 1 saturates to
full absorption: `q_initial[1] = −0.125` MVAr. DER 2 sits at a healthy bus
(`vm = 1.005`, in the deadband) → `q_initial[2] = 0`.

**Sensitivities (from `X`).** Suppose `X[A,1] = 0.20` pu/pu (a 1 pu reactive
change at DER 1 moves bus A by 0.20 pu) and `X[A,2] = 0.05` (DER 2 has weaker
influence on bus A).

**Case 1 — Tier 1 alone suffices.** With `dq_pu = q_initial / sn_mva = [−0.125, 0]`:

```
dV_from_initial[A] = 0.20·(−0.125) + 0.05·(0) = −0.025 pu
vm_predicted[A]    = 1.062 − 0.025 = 1.037 pu   ← inside the band [0.95, 1.05]
```

`resid_mask` is empty → the coordinator returns `q_initial` unchanged; 4B behaves
like 4A. (This is the `Q_RATIO = 0.48` "coordinator never fires" regime.)

**Case 2 — coordination needed.** Now suppose the sensitivities are weaker
(`X[A,1] = 0.08`, `X[A,2] = 0.03`), so Tier 1 alone leaves a residual:

```
dV_from_initial[A] = 0.08·(−0.125) + 0.03·(0) = −0.010 pu
vm_predicted[A]    = 1.062 − 0.010 = 1.052 pu   ← still > 1.05 (residual violation)
```

`resid_mask = {A}`, `n_resid = 1`, `n_ders = 2` (underdetermined). Target the bus
to nominal:

```
dV_residual = V_TARGET_PU − vm_predicted[A] = 1.00 − 1.052 = −0.052 pu
S_viol      = [X[A,1], X[A,2]] = [0.08, 0.03]      (1 × 2)
```

Minimum-norm `lstsq` solves `[0.08, 0.03]·dQ = −0.052` and, among the infinitely
many solutions, picks the smallest-norm one (proportional to the row):

```
dQ_corr_pu = S_violᵀ (S_viol S_violᵀ)⁻¹ dV_residual
           = [0.08, 0.03]ᵀ · (0.08² + 0.03²)⁻¹ · (−0.052)
           = [0.08, 0.03]ᵀ · (0.0073)⁻¹ · (−0.052)
           ≈ [0.08, 0.03]ᵀ · (−7.123)
           ≈ [−0.570, −0.214] pu
```

Convert to MVAr (`· sn_mva = 1.0`) and add to `q_initial`, then clip to ±q_max =
±0.125:

```
q_raw      = q_initial + dQ_corr = [−0.125 − 0.570, 0 − 0.214] = [−0.695, −0.214]
q_adjusted = clip(q_raw, ±0.125) = [−0.125, −0.125]
```

Both DERs are driven to full absorption — DER 2, previously idle, now contributes.
Both are saturated at −q_max, so the provisional `curtailment_needed = True` (the
reactive lever is exhausted; whether curtailment actually fires is decided by the
authoritative post-PF check). The minimum-norm solve spread the demanded
correction across *both* DERs rather than asking DER 1 (already saturated) to do
the impossible — which is exactly the coordination behaviour Tier 1 alone cannot
produce.

*(The unclipped `dQ` here exceeds capability precisely because the sensitivities
were made weak for illustration; the clip enforces physics, and the residual that
remains is what the curtailment tier handles.)*

---

## 14. Concept → implementation map (for Document 3)

Where each theoretical object lives in `sensitivity_coordinator.py`, so Doc 3's
line-by-line walkthrough can be read against this theory. (Line numbers are from
the current source; Doc 3 pins them exactly.)

| Theory (this doc) | Symbol | Function / element |
|-------------------|--------|--------------------|
| Jacobian retrieval + block slicing (§4) | `J_PP, J_PQ, J_QP, J_QQ`, masks | `_build_jacobian_blocks()` |
| Schur complement, sensitivity solve (§5–§6) | `J_red`, `X`, DER-column solve | `_compute_der_sensitivity()` |
| PQ-ordered voltages, NaN guard (§8) | `vm_pu_ppc`, `valid_target_mask` | `_get_vm_pu_ppc()` |
| q=0 invariant check (§7) | `q_current` guard | `coordinate()` Step 2 |
| Prediction + residual mask (§8) | `dq_pu`, `vm_predicted`, `resid_mask` | `coordinate()` Steps 4–7 |
| Rank diagnostic + guard (§10) | `rank`, `rank_limit` | `coordinate()` Step 8 |
| Minimum-norm solve + clips (§9) | `S_viol`, `dV_residual`, `lstsq`, `q_adjusted` | `coordinate()` Steps 9–10 |
| Saturation/curtailment signal (§12) | `SATURATION_TOL`, `curtailment_needed` | `coordinate()` Step 11 |
| Full single-step orchestration | — | `run_coordinated_timestep()` |
| Result container | `CoordinatorResult` | dataclass (parallels `VoltVarResult`, Doc 1C) |
| Constants | `V_TARGET_PU=1.00`, `MIN_VIOLATION_DV=1e-3`, `RCOND_THRESHOLD=1e-12`, `SATURATION_TOL=1e-6` | module level |
| Dynamic band | `_vd.V_MAX`, `_vd.V_MIN` | read from `violation_detector` |

---

## 15. Assumptions and limitations

- **A Jacobian must exist.** Only the `nr` / `iwamoto_nr` solvers populate
  `net._ppc["internal"]["J"]`. `bfsw` (backward-forward sweep) is Jacobian-free —
  the coordinator raises if the Jacobian block is absent, so Scenario 4 must use
  an NR solver. (`algorithm="nr"` is mandatory in `run_coordinated_timestep`.)
- **First-order accuracy.** The `|V| ≈ 1` linearisation holds to <5% across
  `0.95–1.05` pu; outside that band the prediction degrades, which is one reason
  the correction targets 1.0 pu (margin) and the post-PF check is authoritative.
- **Near-collapse operating points** are detected (`rcond_est < 1e-12`) and
  declined rather than trusted.
- **Weakly-controllable violations** (rank-deficient `S_viol`) are declined and
  signalled as curtailment, not forced with a null-space-contaminated correction.
- **OPF is a separate scenario.** This linear-sensitivity coordinator is Scenario
  4's mechanism; Scenario 5's `runopp()` is a different (full-OPF) approach and
  fails on SimBench MV for an unrelated reason (HV/MV transformer impedance
  contrast ill-conditioning the PYPOWER admittance matrix) — see Document 6E.
- **The invariant is a caller contract.** The `q_current = 0` requirement (§7) is
  enforced by a runtime check, but it is the *caller's* job to satisfy it (reset q,
  run the pre-PF); a caller that reimplements the loop must preserve it.

---

*End of Document 2 (Theory). Next: **Document 3** — the line-by-line walkthrough
of `sensitivity_coordinator.py`, which implements every result derived here, with
full six-part blocks, a file pseudocode, a call-site map, and a coverage
checklist over the whole file.*
