# TASK 4 — DOCUMENT 12A
## Custom Controller & Plugin Subsystem, Part 1: The Generic Controller Loop
### `custom_controller.py` — `run_custom_controller_scenario` (498 lines)
### Complete line-by-line walkthrough

> **Source verified:** read against the current `/mnt/project/custom_controller.py`
> (**498 lines**, the project tree). The earlier duplicate-`live_csv_rewrite_fn`
> `SyntaxError` is **fixed** — the file compiles (single parameter, L182).
>
> **Document 12 covers the extension subsystem** — how a user plugs in their *own*
> Q(V) controller or network without touching the core. It splits into:
> - **12A (this document): `custom_controller.py`** — the generic per-timestep HIL
>   loop that drives a user-supplied controller function.
> - **12B: `plugin_runner.py`** — loading a controller/network plugin from a YAML +
>   `.py` and registering it as a benchmark scenario.
> - **12C: `network_plugin.py`** — the custom-network subsystem (loading, validating,
>   adapting a user network).
> - **12D:** the small loaders (`my_network_loader.py`, `network_export.py`) and the
>   worked examples (`droop_controller.py`, `deadband_controller.py`, the example
>   networks, and their YAMLs).
>
> **Read alongside:** **1A–1C** (`VoltVarController`, reused here), **6D** (Scenario 4
> — the built-in sibling this path mirrors), **6A-ii** (`make_record_from_report`,
> `TimestepRecord`, `ScenarioResult`), **Doc 11** (`detect_violations`).

---

## TABLE OF CONTENTS

1. [Overview — the custom controller path](#1-overview)
2. [File logic flow — pseudocode](#2-file-logic-flow--pseudocode)
3. [Call-site map](#3-call-site-map)
4. [Section A — The plugin contract (L1–127)](#4-section-a--the-plugin-contract-l1127)
5. [Section B — _validate_controller_output (L129–166)](#5-section-b--_validate_controller_output-l129166)
6. [Section C — Setup, VoltVarController reuse, resume (L169–297)](#6-section-c--setup-voltvarcontroller-reuse-resume-l169297)
7. [Section D — The two-runpp loop: [A]–[2] (L299–349)](#7-section-d--the-two-runpp-loop-a2-l299349)
8. [Section E — The clean-timestep gate (L351–382)](#8-section-e--the-clean-timestep-gate-l351382)
9. [Section F — Controller call, clamp, post-PF, record (L384–448)](#9-section-f--controller-call-clamp-post-pf-record-l384448)
10. [Section G — Aggregate, result & _periodic_log (L450–498)](#10-section-g--aggregate-result--_periodic_log-l450498)
11. [Findings and flags](#11-findings-and-flags)
12. [Coverage checklist](#12-coverage-checklist)

---

## 1. Overview

`custom_controller.py` lets a user run a benchmark scenario with **their own reactive
controller** — a plain Python function `controller_fn(vm_pu, p_mw) -> q_mvar` — through
the same measurement pipeline the built-in scenarios use. It is the runtime behind the
`--controller` CLI plugin path (Doc 12B).

**The design principle is faithful reuse.** Rather than re-implement DER handling, the
loop constructs a `VoltVarController` in **dry-run mode** (Doc 1C) purely to borrow its
*exact* semantics — sgen ordering, installed-capacity resolution (`sn_mva` with `p_mw`
fallback), DER-bus mapping, and the net-limit Q clamp. So a custom scenario's numbers are
directly comparable with Scenario 4's: the *only* difference is which function computes Q.

**It is a static sibling of Scenario 4.** The loop is the same two-runpp structure as
`run_coordinated_timestep` (Doc 3B) — write `p_target`, reset `Q=0`, pre-PF, compute Q,
apply, post-PF — but **without the coordinator, `DERDynamics`, or curtailment**. The
controller's Q is applied *directly* (no PT1 filtering, no ramp, no Tier-3 curtailment).
So the custom path is a clean way to benchmark *any* local Q(V) rule; `curtailment_needed`
is always `False` (there is no curtailment stage — residual violations simply count in
`n_violation_steps`). It is **stateless**, so resume needs no seeding (like SVC/OPF).

**Two safety layers** make user plugins robust: `_validate_controller_output` (§5) coerces
and checks the return value (named errors at the first offending timestep), and the
optional `clamp_to_net_limits` (default on) passes the output through the inverter
capability clamp so a physically impossible plugin output cannot crash the power flow.

---

## 2. File logic flow — pseudocode

```
_validate_controller_output(q_raw, n_ders, scenario_id):
    coerce to 1-D finite float ndarray of length n_ders; raise named errors otherwise

FUNCTION run_custom_controller_scenario(net, profiles, controller_fn, scenario_id, label,
                                        …, gate_clean_timesteps, clamp_to_net_limits,
                                        enable_checkpointing, live_csv_rewrite_fn):
    adapt profiles; guard der_p non-empty; announce start
    ctrl = VoltVarController(dry_run=True); ctrl.configure()   # borrow DER semantics
    reset controllers/results
    RESUME (defensive: hasattr(publish_fn,"get_resume_records")); skip-if-complete
    FOR t in range(start_t, T):
        [A] p_target (sgen order); [B] write loads
        [0] write p_target, [1] reset Q=0; [2] pre-PF; detect_violations
        pre-PF failed / diverged → non-converged record; continue
        gate_clean_timesteps and no violations → hold Q=0, reuse pre-PF, record, continue
        [3] q = controller_fn(vm at DER buses, p_installed); validate
        [3b] if clamp_to_net_limits → ctrl._clamp_to_net_limits(q)
        [4] apply Q (p unchanged — no dynamics/ramp); [5] post-PF; detect_violations
        [F] build record (curtailment_needed=False by design); publish; append
        every 96 steps: live CSV + _periodic_log
    aggregate → ScenarioResult(scenario_id); log; return
```

---

## 3. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `run_custom_controller_scenario` | function | `plugin_runner._plugin_runner` (Doc 12B) — the registered scenario runner for a `--controller` plugin. Imported at `plugin_runner` top-level (L80). |
| `_validate_controller_output` | helper | The loop [3] (every timestep). |
| `_periodic_log` | helper | The loop (daily progress). |
| `VoltVarController` (dry-run) | class (1C) | Built for DER semantics + the net-limit clamp. |
| `detect_violations` | fn (Doc 11) | Pre- and post-PF. |
| `make_record_from_report`, `ScenarioResult.from_records` | infra (6A-ii) | Record building + aggregation. |
| `publish_fn.*`, `live_csv_rewrite_fn` | hooks | resume/streaming (Doc 10). |

---

## 4. Section A — The plugin contract (L1–127)

The module docstring specifies the contract a user's controller must satisfy:

```
def controller_fn(vm_pu: np.ndarray, p_mw: np.ndarray) -> np.ndarray
```

- **`vm_pu`** — the per-DER **bus voltages** (pu) from the *pre-control* power flow, in
  `sgen_indices` order.
- **`p_mw`** — the per-DER **installed capacity** `p_installed_mw` (constant across
  timesteps), *not* the current available power — so the controller sees "how big is this
  inverter", and can scale its response to rating.
- **returns** — one `q_mvar` per DER, in the **same order**.

**Design rationale — a pure, stateless function.** The contract is deliberately minimal:
the user writes a mathematical Q(V) rule with no framework knowledge, no net object, no
serial handling. Any extra parameters (droop slope, deadband) must be pre-bound with
`functools.partial` before the call, so the signature stays fixed. The docstring's worked
example (a simple droop) shows the whole pattern end-to-end. **Note the `p_mw`-is-installed-
capacity subtlety:** a controller cannot read the current PV output through this argument —
it sees voltage (varying) and rating (fixed). That is intentional (a Q(V) rule needs
voltage and a scale), but is a contract detail a plugin author must know.

---

## 5. Section B — _validate_controller_output (L129–166)

```python
    try: q_arr = np.asarray(q_raw, dtype=float)
    except Exception as exc: raise TypeError(f"Custom controller '{scenario_id}': return value could not be converted ...")
    if q_arr.ndim != 1: raise ValueError(f"... expected a 1-D array ... got ndim={q_arr.ndim} ...")
    if len(q_arr) != n_ders: raise ValueError(f"... expected {n_ders} q_mvar values ... got {len(q_arr)}.")
    if not np.all(np.isfinite(q_arr)): raise ValueError(f"... non-finite q_mvar at positions {bad[:10]} ...")
    return q_arr
```

#### Line 129–166: Coerce and validate the plugin's return

Turns the plugin's return value into a clean 1-D finite float array of exactly `n_ders`
entries, raising a **named** `TypeError`/`ValueError` if it is unconvertible, wrong-shape,
wrong-length, or non-finite. **Design rationale (docstring):** a plugin shape/NaN bug must
be diagnosed **at the first offending timestep**, naming the plugin and the exact fault —
otherwise it would surface far downstream as a silent index misalignment or an
unexplained `runpp` divergence, which is near-impossible to trace back to user code. The
error messages name positions (`bad[:10]`) so the author can find the offending DER.
**Where used:** the loop's [3], every timestep.

---

## 6. Section C — Setup, VoltVarController reuse, resume (L169–297)

#### Line 169–183: Signature (SyntaxError fixed)

```python
def run_custom_controller_scenario(net, profiles, controller_fn, scenario_id, label,
        network_id="unknown", v_min=V_MIN, v_max=V_MAX, publish_fn=None,
        gate_clean_timesteps=False, clamp_to_net_limits=True,
        enable_checkpointing=True, live_csv_rewrite_fn=None):
```

The signature now has a **single** `live_csv_rewrite_fn` (L182) — the earlier duplicate
that caused a `SyntaxError` (blocking the whole plugin path at import) is removed; the file
compiles. `gate_clean_timesteps` and `clamp_to_net_limits` are the two behaviour toggles
(§8, §9); the checkpointing pair mirrors the built-in runners.

#### Line 224–256: Adapt, and borrow VoltVarController's DER semantics

```python
    ap = adapt_profiles(net, profiles)
    if ap.der_p.empty: raise ValueError("... requires at least one profiled DER ...")
    ctrl = VoltVarController(net, interface=None, sgen_indices=ap.der_p.columns, dry_run=True)
    ctrl.configure()
    sgen_idx = ctrl.sgen_indices; sgen_buses = ctrl._sgen_buses; p_inst = ctrl.p_installed_mw; n_ders = ctrl.n_ders
```

**The key reuse.** A `VoltVarController` is built in **dry-run mode** (no serial port;
`configure()` is a no-op) *solely* to obtain the built-in scenarios' exact DER handling —
`sgen_indices` (ordering), `_sgen_buses` (voltage read locations), `p_installed_mw` (the
`sn_mva`-with-`p_mw`-fallback installed capacity), `n_ders`, and later its
`_clamp_to_net_limits`. **Design rationale:** this guarantees the custom path's DER
semantics are *identical* to Scenario 4's, so results are directly comparable — the custom
scenario is a faithful sibling, not a re-implementation that could subtly diverge.

#### Line 267–297: Reset, defensive resume, skip-if-complete

```python
    net.controller.drop(...); pp.reset_results(net)
    if publish_fn is not None and enable_checkpointing and hasattr(publish_fn, "get_resume_records"):
        resumed_records = publish_fn.get_resume_records(scenario_id)
    start_t = (resumed_records[-1].t + 1) if resumed_records else 0
    if start_t >= _T and resumed_records: ... return ScenarioResult.from_records(scenario_id, ...)
    records = resumed_records.copy()
```

Standard resume, with one refinement: the **`hasattr(publish_fn, "get_resume_records")`
guard** (L273) — more defensive than scenarios 2–5, which call the method directly. This
tolerates a `publish_fn` that is a bare callable rather than a full `PublishHandle`.
**Stateless**, so no state seeding is needed. The `scenario_id` (caller-supplied) is the
checkpoint key.

---

## 7. Section D — The two-runpp loop: [A]–[2] (L299–349)

```python
    for t in range(start_t, _T):
        p_target_row = ap.der_p.iloc[t].reindex(sgen_idx).fillna(0.0).values      # [A]
        if not ap.load_p.empty: net.load.loc[ap.load_idx, "p_mw"/"q_mvar"] = ...  # [B]
        net.sgen.loc[sgen_idx, "p_mw"] = p_target_row                             # [0]
        net.sgen.loc[sgen_idx, "q_mvar"] = 0.0                                    # [1] pre-PF invariant
        try: pp.runpp(net, **runpp_kwargs)                                        # [2] pre-PF
        except Exception: pre_pf_ok = False; logger.warning("... pre-PF runpp() raised ...")
        report_pre = detect_violations(net, v_min, v_max)
        if not pre_pf_ok or not report_pre.converged:
            rec = make_record_from_report(t, ts, net, converged=False, ...); ... records.append(rec); continue
```

The per-timestep opening mirrors Scenario 4 / `run_coordinated_timestep` exactly:
`p_target` in DER order **[A]**, load profiles **[B]**, then the **pre-PF invariant** —
write `p_target` **[0]** and **reset `Q=0` [1]** so the pre-control solve measures the
*uncompensated* voltage the controller reacts to (Doc 3B §7). Pre-PF **[2]** with the
SimBench `voltage_depend_loads=False`. A failed or non-converged pre-PF records a
non-converged `TimestepRecord` (with the profile-derived energy fields still set) and skips
to the next step. `runpp_kwargs` here is voltage-depend-loads only — note the custom path
uses pandapower's default algorithm (it has no coordinator needing the NR Jacobian, unlike
Scenario 4).

---

## 8. Section E — The clean-timestep gate (L351–382)

```python
    if gate_clean_timesteps and not report_pre.any_violations:
        q_arr = np.zeros(n_ders)
        rec = make_record_from_report(t, ts, net, converged=True, q_applied=0, p_applied=p_target, curtailment_needed=False)
        rec.losses_mw = ...; rec.grid_import_mw = ...; ...
        records.append(rec); ... _periodic_log(...); continue
```

#### Line 351–382: Optional skip of clean timesteps

When `gate_clean_timesteps=True` and the pre-PF shows **no violations**, the loop holds
`Q=0`, treats the **pre-PF state as the settled state**, and skips both the controller call
and the post-PF. **Design rationale:** this mirrors `run_coordinated_timestep`'s clean-gate
optimisation (Doc 3B) — on a network that is in-bounds most of the year, evaluating the
controller and re-solving every clean step is wasted work; the gate roughly halves the
power-flow count. It is **opt-in** (default `False`) because some controllers are meant to
act pre-emptively even without a violation, and gating would suppress that. The gate's
record is fully populated (energy balance from the pre-PF) so gated steps are not
second-class in the aggregate.

---

## 9. Section F — Controller call, clamp, post-PF, record (L384–448)

```python
        vm_pu_at_ders = net.res_bus.loc[sgen_buses, "vm_pu"].values                 # [3]
        q_arr = _validate_controller_output(controller_fn(vm_pu_at_ders, p_inst), n_ders, scenario_id)
        if clamp_to_net_limits: q_arr = ctrl._clamp_to_net_limits(q_arr)            # [3b]
        net.sgen.loc[sgen_idx, "q_mvar"] = q_arr                                    # [4] p unchanged — no dynamics/ramp
        try: pp.runpp(net, **runpp_kwargs)                                          # [5] post-PF
        except Exception: post_pf_ok = False; logger.warning("... post-PF runpp() raised ...")
        report_post = detect_violations(net, v_min, v_max); converged = post_pf_ok and report_post.converged
        rec = make_record_from_report(t, ts, net, converged, q_applied=q_arr, p_applied=p_target, curtailment_needed=False)
```

#### Line 384–448: The controller call and the record

**[3]** reads the DER-bus voltages from the *pre-PF* result and calls the user's function
with `(vm_pu_at_ders, p_installed)`, then validates the output (§5). **[3b]** the optional
`_clamp_to_net_limits` applies the inverter capability limits (`min/max_q_mvar` + the
apparent-power cap `|Q| ≤ √(sn² − p²)`, Doc 1C) — a **safety backstop** so a plugin
returning an impossible Q cannot break the post-PF. **[4]** writes Q; **crucially `p_mw` is
unchanged from [0]** — there is *no* dynamics or ramp layer (the comment says so), so the
controller's Q takes effect instantly and fully. **[5]** post-PF; `converged` requires both
the solve and `report_post.converged`. The record sets `q_applied`/`p_applied`/`p_target`
and **`curtailment_needed=False` by design** — the custom path has no curtailment stage, so
residual violations are captured in `n_violation_steps` only (documented at L412–414). The
`live_csv_rewrite_fn` block (L440–446) is **correctly indented** (unlike scenarios 2/3/5).

---

## 10. Section G — Aggregate, result & _periodic_log (L450–498)

```python
    result = ScenarioResult.from_records(scenario_id=scenario_id, ..., records=records, dt_s=ap.dt_s)
    logger.info("... Done. ... %d/%d converged | %d violation steps | Q_total=%.2f MVAr", ...)
    if publish_fn is not None: publish_fn.on_scenario_end(result)
    return result

def _periodic_log(scenario_id, network_id, t, T, records):
    if t % 96 != 0: return
    ... logger.info("... t=%d/%d (%.1f %%) | violation_steps=%d | Q_cum=%.2f MVAr", ..., 100.0 * t / max(T, 1), ...)
```

#### Line 450–498: Aggregate and the daily progress log

`from_records` aggregates (with the coverage-integrity check, 6A-ii §7) under the
caller's `scenario_id`, and the summary logs converged fraction, violation steps, and total
reactive throughput. **`_periodic_log` computes progress correctly** — `100·t/max(T,1)`
with `T = _T` (the full length) — so, like OPF (6E) and unlike scenarios 2/3/4, it does
**not** overshoot on resume. **Design rationale:** the custom path was written after the
built-in runners and quietly adopts the correct progress-% pattern.

---

## 11. Findings and flags

**Positive notes (this module is cleaner than the built-in runners on three points):**

1. **SyntaxError fixed** — the duplicate `live_csv_rewrite_fn` parameter is gone; the file
   compiles, unblocking the whole `--controller` path.
2. **Correct live-CSV indentation** (L373–379, L440–446) — properly nested, unlike the
   20-space over-indent in scenarios 2/3/5.
3. **Correct progress-%** (`100·t/max(T,1)`, full length) — no resume overshoot, unlike
   scenarios 2/3/4.
4. **Defensive resume** — `hasattr(publish_fn, "get_resume_records")` guards against a
   non-`PublishHandle` `publish_fn`.

**Not bugs (documented for clarity):** `curtailment_needed=False` is by design (no
curtailment stage; residual violations counted in `n_violation_steps`); the static Q
application (no dynamics/ramp) is the intended simplification distinguishing the custom
path from Scenario 4; the `VoltVarController` dry-run reuse is the mechanism guaranteeing
comparable DER semantics.

**No bugs found.**

---

## 12. Coverage checklist

| Entity | Lines | Kind | Documented | Where-used stated |
|--------|-------|------|:----------:|:-----------------:|
| module docstring / plugin contract | 1–127 | doc | ✅ | ✅ |
| `_validate_controller_output` | 129–166 | helper | ✅ | ✅ (loop [3]) |
| `run_custom_controller_scenario` | 169–471 | function | ✅ | ✅ (plugin_runner) |
| — signature (SyntaxError fixed) | 169–183 | — | ✅ | — |
| — adapt + VoltVarController reuse | 224–256 | — | ✅ | — |
| — reset + defensive resume + skip-if-complete | 267–297 | — | ✅ | — |
| — [A]–[2] loop opening + non-converged | 299–349 | — | ✅ | — |
| — clean-timestep gate | 351–382 | — | ✅ | — |
| — [3]–[5] controller call/clamp/post-PF/record | 384–448 | — | ✅ | — |
| — aggregate + result | 450–471 | — | ✅ | — |
| `_periodic_log` | 474–498 | helper | ✅ | ✅ (loop) |

**Also covered (not standalone entities):** the plugin contract (incl. the
`p_mw`-is-installed-capacity subtlety), the dry-run `VoltVarController` reuse for identical
DER semantics, the pre-PF `Q=0` invariant, the static (no-dynamics) Q application, the two
safety layers (validation + clamp), the opt-in clean gate, and the stateless resume.

**Flags raised:** none (bugs). **Positives:** SyntaxError fixed; correct indentation;
correct progress-%; defensive resume.

---

*End of Document 12A. Next: Document 12B — `plugin_runner.py` (loading a controller/network
plugin from YAML + `.py`, validating it, and registering it as a benchmark scenario that
calls `run_custom_controller_scenario`).*
