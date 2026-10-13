# Corrections to `TASK1B_MODULE_DOCUMENTATION.md`

**Date:** 2026-10-13
**Applies to:** `TASK1B_MODULE_DOCUMENTATION.md` (project knowledge; not tracked
in this repository, which is why these corrections are a separate file rather
than a diff).

Two kinds of entry below:

* **[PATCHED]** — the documentation described behaviour that three patches have
  since changed. The code is now correct and the doc is stale.
* **[WRONG]** — the documentation was already incorrect when written,
  independently of those patches. Two of these affect claims in the paper.

Each entry gives the current doc text and the replacement. The replacements are
written to be pasted into the corresponding section of the project doc.

The commits these corrections reflect:

```
Scenario 3: place the SVC at the substation busbar topologically
Hosting capacity: restore the Q(V) headroom and stop reporting an undefined HC as 0.00 MW
Publish dt_s, group fig09 by calendar day, and record rank deficiency
```

---

## 0. Document header — verification basis

**[PATCHED]** The header states:

> **Verification basis:** every claim below was read from the current function
> bodies (post-fix state, as the user confirmed all applied) …

That was true of the Task-2 fix state. It is three patches out of date. Replace
with a dated statement naming the commit the document reflects, so a future
reader can tell what "current" meant:

> **Verification basis:** every claim below was read from the function bodies,
> not from docstrings — several docstrings in this codebase were confirmed stale
> during the audit (Task-2 findings C.1–C.3, F.2–F.4, K.1). State reflected:
> the three post-audit patches of 2026-10-13 (SVC placement, hosting-capacity
> Q(V) headroom, `dt_s` publication + rank-deficiency reporting). Equations are
> transcribed from source and cross-checked against the project's VDE-AR-N 4110,
> SVC-ULTC and sensitivity-estimation reference PDFs.

---

## 1. `scenario_3_svc.py`

### 1a. Placement paragraph — **[PATCHED]**

Current text:

> **Placement.** `_select_svc_bus_stress()` applies an overvoltage stress
> scenario, runs a power flow, and picks the in-service MV bus with the lowest
> mean voltage (the worst spot to defend); fallback is the worst-voltage MV bus
> in a short violation pre-scan. HV slack and LV buses are excluded (injecting Q
> at an infinite busbar is meaningless).

`_select_svc_bus_stress()` and `_select_svc_bus_first_violation()` no longer
exist. The stated justification was also inverted: under overvoltage stress the
*lowest*-voltage MV bus is the bus most stiffly tied to the slack, not "the
worst spot to defend". Replacement:

> **Placement.** The device is centralised at the substation — one compensator at
> the point of common coupling regulating the whole MV group, as opposed to the
> distributed inverter control of Scenarios 4A/4B. Placement is therefore a
> topological fact about the network, read from `net.trafo`, and needs no power
> flow, no profile data and no stress case, so it is reproducible and
> independent of the run configuration. `_select_svc_bus(net)` tries three
> paths, all targeting the same location:
>
> 1. `_select_svc_bus_substation()` — the MV-side busbar of an in-service
>    step-down transformer (`net.trafo.lv_bus` where `vn_hv_kv > vn_lv_kv`),
>    ranked ext_grid-fed first, then descending `sn_mva`, then ascending bus
>    index.
> 2. `_select_svc_bus_nearest_slack()` — the eligible MV bus with the fewest
>    branches to the slack over the connectivity graph (`pandapower.topology`,
>    switches respected), for a network exposing no step-down transformer.
> 3. Last resort: the lowest-index eligible MV bus.
>
> HV slack and LV buses are excluded (injecting Q at an infinite busbar is
> meaningless). A voltage ranking is deliberately **not** used: it identifies the
> stiffest bus only while every feeder is generation-dominated, and on CIGRE MV,
> whose feeder 2 (buses 12–14) carries almost no DER, the minimum lands at the
> far end of that feeder instead. On `1-MV-rural--2-sw` the chain returns bus 2,
> `MV1.101 busbar1.1`, the LV side of `HV1-MV1.101-Trafo1`.

### 1b. `k_q` formula — **[WRONG]**

The Purpose section states:

```
    k_q   = Q_MAX / SVC_FULL_ACTION_DV                # full action reached at 0.04 pu error
```

and *What could break it* then says:

> **Doc inconsistency (minor):** the module-header control-law comment writes
> `k_q = Q_MAX / 0.03`, but the actual constant is `SVC_FULL_ACTION_DV = 0.04`
> → `k_q = Q_MAX/0.04`. Code is authoritative; the comment is stale (a Task-1b
> doc-correction item, harmless to behaviour).

This is backwards. `_compute_svc_params()` computes:

```python
k_q = q_max / (SVC_FULL_ACTION_DV - SVC_DEADBAND)      # = Q_MAX / 0.03
```

The module header and `Process Plots/S3 Plots/flow_s3.py` both state `/0.03`
correctly. The denominator is the *active* droop span — the error range between
the deadband edge and full action — which is why the deadband is subtracted.
Replacement for the Purpose line:

```
    Q_MAX = SVC_Q_MAX_RATIO · Σ net.trafo.sn_mva       # 0.20 × total trafo MVA
    k_q   = Q_MAX / (SVC_FULL_ACTION_DV − SVC_DEADBAND)  # = Q_MAX / 0.03
                                                      # full action at 0.04 pu error,
                                                      # measured from the deadband edge
```

and the *What could break it* bullet should be deleted outright — there is no
doc inconsistency to fix in the code.

### 1c. Worked numerical example — **[WRONG]**

The example assumes `Σ sn_mva = 40 MVA` ("two 20 MVA HV/MV units") giving
`Q_MAX = 8.0 MVAr` and `k_q = 200 MVAr/pu`. `1-MV-rural--2-sw` has two **25 MVA**
transformers, and the gain uses the 0.03 denominator. The run logs:

```
_compute_svc_params: sn_mva_sum=50.00 MVA | Q_MAX=10.000 MVAr | k_q=333.333 MVAr/pu
```

Every number downstream in that example changes. Corrected arithmetic:

```
    Q_MAX = 0.20 · 50 = 10.0 MVAr
    k_q   = 10.0 / (0.04 − 0.01) = 333.333 MVAr/pu
```

*Overvoltage timestep,* host bus at `vm_pu = 1.035`, `error = −0.035`:

```
    q_cmd = k_q · (error + deadband) = 333.333 · (−0.035 + 0.01) = −8.333 MVAr
    q_cmd = clip(−8.333, −10.0, +10.0) = −8.333 MVAr      (absorb)
```

*Deep overvoltage,* `vm = 1.06`, `error = −0.06`:
`q_cmd = 333.333 · (−0.05) = −16.667` → clipped to `−10.0 MVAr` (saturated).
*Deadband,* `vm = 1.006`: `|error| = 0.006 ≤ 0.01` → `q_cmd = 0`.
*Undervoltage,* `vm = 0.965`, `error = +0.035`:
`q_cmd = 333.333 · (0.035 − 0.01) = +8.333 MVAr` (inject).

For reference, the day-172 run on this network logs a first-step command of
`q_cmd = −5.631 MVAr` at `vm_svc = 1.0269`, which the formula reproduces:
`333.333 · (1.00 − 1.0269 + 0.01) = −5.633`.

### 1d. "Known result: catastrophic SVC" — **[WRONG, and it affects the paper]**

Current text:

> **Known result:** on `1-MV-rural--1-sw` the SVC is *catastrophic*
> (`min_vm = 0.827`) — a single central device cannot fix a distributed
> overvoltage problem, which is precisely the motivating contrast for the
> distributed Volt-Var scenarios.

and, in the Purpose section:

> That is exactly why a single central SVC cannot fix a *distributed*
> overvoltage, and why the sensitivity coordinator (which recruits DERs *at* the
> violated buses) succeeds where the SVC fails.

That result came from the **remote fallback placement**, not from centralised
compensation. When the stress import or selection failed, the old chain fell
through to `mv_buses[-1]` — the highest-index, i.e. most remote, MV bus — so the
"central SVC" being measured was in fact a single device at the feeder end,
which drives the far end into *undervoltage*. Re-measured on day 172 of
`1-MV-rural--1-sw`, both placements, same profile slice:

| placement | violation steps | max vm | min vm | Σ\|Q\| (MVAr·steps) |
|---|---|---|---|---|
| none (baseline) | 46 / 96 | 1.0565 | 1.0241 | — |
| **bus 2 — substation busbar (current)** | **0 / 96** | 1.0421 | 1.0114 | 548.17 |
| bus 98 — old remote fallback | 42 / 96 | 1.0394 | 0.9340 | 748.89 |

So on `--1-sw` the centralised SVC **resolves the day's violations entirely**
(46 → 0), while the remote device leaves 42 violation steps and pushes the
minimum voltage to 0.934 pu, spending 37 % more reactive energy to do it. The
0.827 figure is a property of remote placement.

The motivating contrast therefore has to be re-stated, and it is still available
— but from the *other* network. On `1-MV-rural--2-sw` (day 172), the centralised
SVC moves violations only 96 → 91 and `max_vm` 1.0814 → 1.0677, while local
Q(V) clears all 96. That is the honest version of the argument: a single central
device helps materially on a moderately stressed feeder (`--1-sw`, baseline
46/96) and barely at all on a heavily stressed one (`--2-sw`, baseline 96/96),
because its voltage influence falls off with electrical distance from the
substation while the violations do not. Suggested replacement:

> **Known result.** Centralised SVC effectiveness is strongly
> network-dependent. On `1-MV-rural--1-sw` (day 172, baseline 46/96 violation
> steps) the SVC at the substation busbar clears every violation (0/96,
> `min_vm = 1.0114`). On the more heavily loaded `1-MV-rural--2-sw` (baseline
> 96/96) the same device reduces violations only to 91/96 and `max_vm` to
> 1.0677, because its influence decays with electrical distance while the
> overvoltage is distributed along the feeders — whereas local Q(V) at the DERs
> clears all 96. That contrast, not a single catastrophic number, is the
> motivation for the distributed Volt-Var scenarios.
>
> A single remote device is worse than either: placed at the feeder end on
> `--1-sw` it leaves 42/96 violations and drives `min_vm` to 0.934 pu. The
> previously documented `min_vm = 0.827` result came from that configuration,
> reached through a since-removed placement fallback, and is not a property of
> centralised compensation.

**Before this goes in the paper**, re-run both networks over the annual profile
rather than one day — day 172 sits at roughly the 93rd percentile of annual net
injection on `--2-sw`, so neither number is a worst case.

### 1e. Inputs and Failure modes — **[PATCHED]**

* **Inputs** — delete "Optional `stress.apply_overvoltage_stress` (guarded
  import)". The module no longer imports `hiltest.stress`, and `copy` is no
  longer imported either.
* **Failure modes / safeguards** — replace "**Placement fallback** if stress
  import/selection fails" with "**Placement fallback chain** — transformer table
  → fewest hops to the slack → lowest-index eligible MV bus, all topological, so
  a failure of the first path lowers confidence in the placement without
  changing the configuration being benchmarked."

---

## 2. `hosting_capacity.py`

### 2a. Worst-case snapshot — **[PATCHED]**

The Purpose section describes the snapshot as "minimum load × maximum PV". That
is still the intent, but "maximum PV" now means **1.0 p.u. of rated active
power**, not `p_mw = sn_mva`. Add after the sweep description:

> **What "maximum PV" means.** `sn_mva` is the inverter's *apparent* rating, so
> setting `p_mw = sn_mva` forces cos φ = 1 and leaves
> `|Q| ≤ √(sn² − p²) = 0` — under which Case B has no reactive capability to
> deploy and is identical to Case A by construction, on every network. The
> snapshot therefore sets
>
> ```
> p_mw = sn_mva / √(1 + Q_RATIO²)          # _p_at_max_irradiance()
> ```
>
> leaving exactly `Q_RATIO × P` of headroom — the full capability the Q(V)
> characteristic asks for, available at rated active power as VDE-AR-N 4110
> §10.2.2 requires over `0.05 < P_mom/P_b,inst < 1`. The factor is derived from
> `volt_var_controller.Q_RATIO` rather than being a constant of its own, so the
> derating and the characteristic cannot drift apart: at `Q_RATIO = 0.25` it is
> 0.9701. On CIGRE MV this took the fleet's reactive headroom from 0.000 MVAr to
> `0.25 × P` and the HC gain from +0.000 MW to **+1.000 MW**.

### 2b. Sweep pseudocode — **[PATCHED]**

`_extract_binding()` returns three values now. In the Case A pseudocode replace

```
            binding_bus, binding_vm = extract_binding(report)
```

with

```
            binding_bus, binding_vm, binding_element = _extract_binding(report)
```

`binding_element` is a human-readable string naming what stopped the sweep
("overvoltage at bus 42", "line 7 at 104.2 % loading", "power flow did not
converge"). It is the only field that distinguishes a voltage-bound from a
thermally-bound hosting capacity, since `binding_bus`/`binding_vm_pu` are
`-1`/NaN for a thermal or angular stop.

### 2c. Validity of the starting point — **[PATCHED]**, new subsection

Nothing in the current document covers this. Add before the worked example:

> **The sweep's premise is now checked.** Hosting capacity is *additional*
> capacity, so it is only defined from a violation-free starting point. Where a
> network's pre-placed DER already breaches a limit under min-load ×
> max-irradiance, the first sweep step violates and `hc_mw` is left at
> `params["start"]` — previously published as a flat `0.00 MW`, which reads like
> a measurement rather than an undefined result, with a `+0.000 MW` gain that
> looks like Volt-Var achieving nothing. `_infeasible_snapshot()` now runs a
> power flow on the snapshot before the sweep and, if it already violates,
> returns an `HCResult` with `hc_mw = NaN`, `n_steps = 0` and a
> `binding_element` naming the pre-existing violation. `publisher` emits
> `gain_mw: null` (not NaN, which is invalid strict JSON) when either case is
> undefined, and the executor and benchmark runner print "n/a" with the reason.
>
> On `1-MV-rural--2-sw` this is the honest result: 45.6 MW of pre-placed DER at
> max irradiance against 2.8 MW of load puts bus 68 at 1.1216 pu with 50 buses
> over limit before the first step is added. End-of-feeder incremental HC was
> specified for networks with **no** pre-placed DER (Kerber, Dickert, CIGRE LV,
> Synthetic Voltage Control LV), where the snapshot is compliant by
> construction; the pre-placed-DER case was a later extension, and this is where
> it breaks.
>
> Conversely, on a network with no pre-placed DER no element is Q(V)-controlled
> at all — the swept HC sgens are deliberately excluded from control — so Case B
> reduces to Case A and the gain is zero *by construction* rather than by
> measurement; a warning now says so. A non-zero HC gain therefore requires a
> network that has pre-placed DER **and** a compliant snapshot. CIGRE MV
> (`with_der="pv_wind"`) satisfies both and is where the HC-gain figure should
> come from.

### 2d. Worked example — **[PATCHED]**

The 3.0 → 4.0 MW / +1.0 MW (+33 %) example remains a valid illustration of the
mechanism and can stay. Append one sentence:

> This illustrates the mechanism; for it to be reachable, the snapshot must be
> compliant at 0 MW added (see above). The measured gain on CIGRE MV is
> +1.000 MW, from `hc_mw(A) = 0.000` to `hc_mw(B) = 1.000`.

### 2e. Outputs — **[PATCHED]**

The `HCResult` field list is missing `binding_element`. Replace the field list
with: `hc_mw`, `violated_at_mw`, `binding_bus`, `binding_vm_pu`,
`binding_element`, `n_steps`, `hc_limit_reached`, `endoffeeder_bus`,
`dist_voltage_kv`, `params`, `qv_converged`, `qv_iters_max`, `sweep_curve`.

### 2f. Q(V) iteration count — **[PATCHED]**, confirmation only

"Q(V) is a contractive mapping under normal conditions, converging in 3–5
iterations" was unverifiable before the headroom fix (with zero headroom the
loop converged trivially at one iteration). It is now measured at **3**
iterations on CIGRE MV, so the claim stands as written.

---

## 3. `sensitivity_coordinator.py`

### 3a. Rank-deficiency paragraph — **[PATCHED]**

Current text, in §6:

> `coordinate()` computes `rank(S_viol)`; when `rank < min(n_resid, n_ders)` it
> sets `curtailment_needed = True` and **returns `q_initial` unchanged** (via
> `lstsq` proceeding but the rank guard preventing null-space corruption; the
> old rank-deficiency early-return was removed so `lstsq`'s SVD truncation
> handles the null space). A throttled warning fires (first time, then every
> 96th).

Wrong in three ways: it does not return `q_initial` unchanged (the parenthetical
contradicts the sentence it qualifies); it no longer sets `curtailment_needed`;
and the comparison no longer uses `n_ders`. Replacement:

> `coordinate()` computes `rank(S_viol)` and compares it against
> `min(n_resid, n_eff_ders)`, where `n_eff_ders` counts only DERs that actually
> have a column — an excluded DER carries an all-zero column, so the rank can
> never exceed the number of included DERs, and comparing against `self._n_ders`
> would report deficiency purely because a DER was excluded. Two distinct
> conditions are tested, both meaning "no Q allocation can drive every violated
> bus to target":
>
> * **under-determined** — `n_eff_ders < n_resid`: fewer controllable DERs than
>   violated buses.
> * **rank-deficient** — enough DERs, but their voltage sensitivities are
>   collinear over the violated buses. On radial feeders DERs on the same branch
>   produce collinear columns, so this is the common case and is physics, not a
>   bug.
>
> Either sets `self.rank_deficient = True`, reset at the top of every
> `coordinate()` call so a stale value cannot be reported against a later
> timestep. `lstsq` proceeds either way — its SVD truncation handles the null
> space — so `q_initial` is *not* returned unchanged. A throttled warning fires
> (first time, then every 96th).
>
> The flag is deliberately separate from `curtailment_needed`. The earlier
> implementation set `curtailment_needed = True` here, but Step 11 overwrote it
> unconditionally from the saturation test and `run_coordinated_timestep()` step
> [8] overwrote it again from `report_post.any_violations`, so the signal never
> reached a record. `rank_deficient` now travels:
> `SensitivityCoordinator.rank_deficient` → `CoordinatorResult.rank_deficient` →
> `TimestepRecord.coordination_rank_deficient` →
> `ScenarioResult.rank_deficient_steps` / `rank_deficient_rate`, exposed in
> `summary_dict()` and round-tripped through the checkpoint. It distinguishes
> "cannot reach these buses" from "ran out of Q" (`q_saturated_count`).
>
> Measured on `1-MV-rural--2-sw` under a stressed snapshot: `rank = 26`,
> `n_resid = 27`, `n_controllable_ders = 102`, `under_determined = False` — the
> solve can independently target 26 of 27 violated buses and compromises on the
> last one.

### 3b. DER exclusions — **[PATCHED]**

In §4 and in *Failure modes*, the exclusion list is "out-of-service
(`ppc_bus == −1`), out of range, or not a PQ bus in this operating point". Add
zero reactive capability:

> A DER with `q_max ≤ 0` (zero or non-finite `Q_RATIO × p_installed`) is also
> excluded. `__init__` already warned that such DERs "will be excluded from
> coordination", but nothing acted on it: the DER kept a column in `X` and
> absorbed part of the least-squares correction, which the `±q_max` clip then
> zeroed, so the DERs that *could* act under-corrected.
> `_compute_der_sensitivity()` now excludes them, making the warning true. If
> *every* DER is excluded the existing `n_valid == 0` branch returns
> `(None, None)` and `coordinate()` falls back to clipping `q_initial`.

### 3c. Outputs — **[PATCHED]**

The `CoordinatorResult` field list needs `rank_deficient: bool` added. Also note
that the three construction sites (failed pre-PF, clean-timestep gate, final
result) set it to `False`, `False` and `coordinator.rank_deficient`
respectively, since coordination does not run in the first two.

### 3d. Failure modes — **[PATCHED]**

Delete:

> **Rank-deficiency guard:** collinear radial columns → `q_initial` returned
> unchanged rather than a null-space-corrupted correction; throttled warning.

It describes behaviour that was already absent when the document was written.
Replace with:

> **Rank/under-determination diagnostic:** collinear or insufficient DER
> sensitivities set `rank_deficient` and are recorded per timestep; `lstsq`
> still returns the minimum-norm correction, so the timestep is not skipped.

### 3e. The `Q_RATIO = 0.25` justification — **[WRONG]**

In the `volt_var_controller.py` Purpose section:

> `Q_RATIO = 0.25` … the project's knowledge base documents that at `0.30` the
> coordinator is never needed (local Q(V) over-controls) and at `0.48` the
> coordinator never fires, so `0.25` is the value that makes the coordination
> layer observably active on violated days.

Not supported by measurement at the stated value. On day 172 of
`1-MV-rural--2-sw` at `Q_RATIO = 0.25`, every logged coordinated step reports
`max|dQ_coord| = 0.0000 MVAr`: the pre-PF has 26 overvoltage buses, local Q(V)
alone clears all of them, `resid_mask` is empty at Step 7, and `coordinate()`
returns `q_initial` untouched. Scenarios 4 and 5 are consequently identical to
six decimal places (`vdi = 148.509994`, `total_losses_mwh = 12.671010`,
`Q_tot = 931.41` in both). That is correct behaviour, not a defect — there is
nothing left to coordinate — but it means 0.25 has not been shown to make
coordination active on this network.

Until the annual run settles it, state what is known:

> `Q_RATIO = 0.25` corresponds roughly to cos φ ≈ 0.97 and is the value used for
> the published benchmark results. The project's knowledge base records that at
> `0.30` local Q(V) over-controls and at `0.48` the coordinator never fires.
> Whether `0.25` makes the coordination layer active is network- and
> day-dependent: on day 172 of `1-MV-rural--2-sw` local Q(V) clears all 26
> overvoltage buses on its own, so the coordinator correctly applies no
> correction and Scenario 5 reproduces Scenario 4 exactly. Day 172 peaks at
> 12.38 MW net injection against an annual peak of 27.21 MW (day 207), with
> 8.2 % of annual timesteps more stressed than that day's peak, so the annual
> run is what determines whether coordination activates at this `Q_RATIO`.

---

## 4. `scenario_result.py`

### 4a. `dt_s` — **[PATCHED]**

*What could break it* currently says:

> `dt_s` must match the profile resolution or energy/duration metrics scale
> wrongly (default 900 s = 15-min SimBench).

The default is gone — `from_records(dt_s)` is a required argument, because a
hardcoded 15-minute default on the function computing every energy integral
over-reports by 90× at `dt = 10 s`. Replacement:

> `dt_s` is a **required** argument to `from_records()` and must match the
> profile resolution; it scales every energy and duration integral. It is now
> also stored on the result (`ScenarioResult.dt_s`, default NaN when unset) and
> published in `summary_dict()`, so a run states the resolution its MWh/MVArh
> figures were integrated at — without it, two runs at different resolutions are
> indistinguishable in the output, and downstream figure code has no way to
> derive steps per day.

### 4b. New fields — **[PATCHED]**

Add to the `TimestepRecord` field list: `coordination_rank_deficient`
(`Optional[bool]`, None outside Scenario 4's coordinated mode). Add to the
`ScenarioResult` list: `dt_s`, `rank_deficient_steps`, `rank_deficient_rate`.

### 4c. Metric name — **[WRONG]**

The computed-quantities block writes `violation_hours`. The field is
`violation_duration_h`.

---

## 5. `violation_detector.py` — **[PATCHED]**

*Failure modes* says:

> **`converged=False` short-circuit** returns empty-but-schema-correct frames
> (no crash on a diverged input).

Still true, with an addition worth recording: the short-circuit now passes the
resolved limits into the report. The `v_min_used` / `v_max_used` /
`unbalance_max_used` fields are bare dataclass defaults evaluated at
class-definition time, so a non-converged report previously printed the
import-time band in its `summary()` rather than the override in force for the
run. Both `detect_violations()` and `detect_violations_3ph()` now pass `v_min`,
`v_max` (and `unbalance_max`) explicitly on that path.

---

## 6. `plot_results.py` — **[PATCHED]**

The integrity paragraph states:

> **Integrity property (Task-2 Category E, confirmed clean):** every figure
> builds its series by iterating the loaded JSON — no hardcoded data arrays, no
> synthesised fallback series.

The claim about fabricated data holds, but the audit missed a hardcoded
*resolution* assumption in `fig09_violation_heatmap`:

> One exception was found after that audit and has been fixed:
> `fig09_violation_heatmap()` hardcoded `step = 144` ("144 × 10-min steps per
> day") while labelling the x-axis "Day of simulation period". At SimBench's
> 15-min cadence each column therefore aggregated 36 hours, and an annual run
> produced 243 columns labelled days 0–242 for a 365-day period, silently
> dropping the remainder. It now groups by calendar date from the published
> per-row `timestamp`, falling back to `86400 / summary.dt_s` and then to a
> single column — resolution-independent, and the same approach
> `_extreme_day_slice()` already used correctly. Verified to produce 3 columns
> for a 3-day series at 15-min, 10-min and 10-second cadence.

---

## 7. `publisher.py` — **[WRONG]**

The live-mode paragraph says:

> `update_every_k` cadence: at 15-min resolution, `k=1` = every 15 min
> (~35k frames/yr), `k=6` = hourly, `k=96` = daily

At 15-min resolution `k=6` is **90 minutes**; `k=4` is hourly. The "k=6 =
hourly" figure is correct only at 10-minute resolution, which is what
`publisher.py`'s own docstring says (and is self-consistent there). `k=96` =
daily is correct at 15-min. Replacement:

> `update_every_k` cadence at SimBench's 15-min resolution: `k=1` = every 15 min
> (~35k frames/yr), `k=4` = hourly, `k=6` = every 90 min, `k=96` = daily. The
> module docstring's "`k=6` → one frame per hour" assumes 10-min data. The
> executor sets it from `plan.stream_every_k`.

Also worth adding to the Outputs section: `scenarios/<id>.json` now carries
`summary.dt_s`, since `build_scenario_payload()` forwards every
`summary_dict()` key — this is the authoritative resolution for any consumer.

---

## Summary of affected claims

| § | Doc section | Kind | Affects the paper? |
|---|---|---|---|
| 1a | scenario_3_svc — Placement | patched | no |
| 1b | scenario_3_svc — `k_q` formula | wrong | if quoted |
| 1c | scenario_3_svc — worked example | wrong | if quoted |
| **1d** | **scenario_3_svc — "catastrophic SVC"** | **wrong** | **yes — motivating contrast** |
| 1e | scenario_3_svc — Inputs / Failure modes | patched | no |
| 2a–2f | hosting_capacity | patched | yes — HC gain figure |
| 3a–3d | sensitivity_coordinator | patched | no |
| **3e** | **`Q_RATIO = 0.25` justification** | **wrong** | **yes — 4A vs 4B claim** |
| 4a–4c | scenario_result | patched / wrong | no |
| 5 | violation_detector | patched | no |
| 6 | plot_results — fig09 | patched | yes — figure is in the set |
| 7 | publisher — cadence | wrong | no |
