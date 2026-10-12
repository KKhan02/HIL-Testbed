# TASK 4 — DOCUMENT 6A-iii
## Scenario Runner — Baseline
### `scenario_1_baseline.py` — the no-control reference (`_timed_runpp`, `run_scenario_1`)
### Complete line-by-line walkthrough

> **Source re-verified:** re-pinned against the current `/mnt/project/scenario_1_baseline.py`
> (**353 lines** — was 346). The +7 lines are two new signature parameters,
> `enable_checkpointing` and `live_csv_rewrite_fn`, **accepted but deliberately not
> implemented** (see §6 and §11) — added only for `benchmark_runner`'s unconditional
> kwargs injection. `_RUNPP_TIMING` is still never cleared (the §11 flag stands).
> Everything else is unchanged; internal line numbers below L96 read ~+7.
>
> **(prior note) Source verified:** `scenario_1_baseline.py` was read in full before
> writing. If the file is edited, re-verify line numbers.
>
> **Part 3 of 3 of Document 6A** (6A-i = profile adapter; 6A-ii = the result types
> this runner produces). Baseline is the simplest consumer of that infrastructure —
> and the *only* scenario that uses pandapower's built-in time-series engine rather
> than a manual loop.
>
> **⚠ Contains a flagged bug** — `_RUNPP_TIMING` is never cleared despite its
> docstring claiming otherwise, so per-timestep timings accumulate across runs in
> one process. See §6 and §12.
>
> **Read alongside:** **6A-i/6A-ii** (the infrastructure it uses); Master Index
> §5–§8; **Document 9** (`benchmark_runner`, which calls this); the manual-loop
> runners **6B/6C/6D** (the architectural contrast this document draws out).

---

## TABLE OF CONTENTS

1. [Overview — the reference scenario](#1-overview)
2. [File logic flow — pseudocode](#2-file-logic-flow--pseudocode)
3. [Call-site map](#3-call-site-map)
4. [Declarative vs manual: why baseline is different](#4-declarative-vs-manual-why-baseline-is-different)
5. [Section A — _RUNPP_TIMING & _timed_runpp (L76–86)](#5-section-a--_runpp_timing--_timed_runpp-l7686)
6. [Section B — run_scenario_1: setup & integer reindex (L112–147)](#6-section-b--run_scenario_1-setup--integer-reindex-l112147)
7. [Section C — [1]–[2] reset & ConstControl (L149–190)](#7-section-c--12-reset--constcontrol-l149190)
8. [Section D — [3]–[4] OutputWriter & run_timeseries (L192–221)](#8-section-d--34-outputwriter--run_timeseries-l192221)
9. [Section E — [5] post-processing (L223–327)](#9-section-e--5-post-processing-l223327)
10. [Section F — from_records & return (L329–346)](#10-section-f--from_records--return-l329346)
11. [Findings and flags](#11-findings-and-flags)
12. [Coverage checklist](#12-coverage-checklist)

---

## 1. Overview

Scenario 1 is the **no-control reference** — the baseline every controlled scenario
is measured against. It runs a full annual time series with **no reactive control,
no Arduino, no coordinator, no manual timestep loop**: just the profiles applied to
the network and the resulting voltages logged. Its `ScenarioResult` has every
control field `None` (no `q_applied`, no `p_applied`, no curtailment,
`coordination_active = None`), so the comparison layer sees it as "what the network
does when left alone".

**The architectural distinction (developed in §4).** Baseline is the *only* scenario
that uses pandapower's declarative time-series engine — `ConstControl` data sources
+ `OutputWriter` + `run_timeseries()` — because it needs no per-timestep control
hook. Scenarios 3 and 4 must interpose control *between* the pre- and post-control
power flows each step, which the declarative engine cannot express, so they use
manual loops. Baseline gets to be simple precisely because it does nothing.

Two module-level pieces support it: `_RUNPP_TIMING` (a timing accumulator) and
`_timed_runpp` (a `runpp` wrapper passed as the `run=` hook), which capture
per-timestep solve times the declarative engine would otherwise hide.

---

## 2. File logic flow — pseudocode

```
MODULE STATE: _RUNPP_TIMING — a dict of per-timestep runpp durations

FUNCTION _timed_runpp(net, **kwargs):    [the run= hook for run_timeseries]
    time a pp.runpp() call; add the ms to _RUNPP_TIMING[current OutputWriter step]

FUNCTION run_scenario_1(net, profiles, network_id, v_min, v_max, publish_fn):
    adapt profiles → AdaptedProfiles
    reindex load_p / load_q / der_p from timestamps to integer steps 0..T-1
    validate no integer step is missing
    [1] drop any existing controllers; reset result tables
    [2] attach ConstControl for der p_mw, load p_mw, load q_mvar (by element index)
    [3] attach an OutputWriter logging vm_pu, line/trafo loading, line/trafo losses, grid P
    [4] run_timeseries(voltage_depend_loads=False, continue_on_divergence=True, run=_timed_runpp)
    [5] post-process the OutputWriter logs into a TimestepRecord list:
          for each step: converged? (log row exists and not all-NaN)
            no  → record with empty results, control fields None, profile energy still set
            yes → threshold violations (±ε), read losses/grid from logs, build record
    aggregate with ScenarioResult.from_records(scenario_id="baseline", dt_s=ap.dt_s)
    return the result
```

---

## 3. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `_RUNPP_TIMING` | module dict | Written by `_timed_runpp`; read in the post-processing loop (`t_total_ms`). **Never cleared** — see §11. |
| `_timed_runpp` | function | Passed as the `run=` hook to `run_timeseries` (L219). Not called directly. |
| `run_scenario_1` | function | The Scenario-1 entry point — called by `benchmark_runner` (Doc 9) as one of the five runners. |

---

## 4. Declarative vs manual: why baseline is different

This is the single most instructive thing about this file, so it is worth stating
before the line walk.

| | Scenario 1 (baseline) | Scenarios 3/4 (SVC / Volt-Var) |
|--|----------------------|--------------------------------|
| Engine | pandapower `run_timeseries()` | hand-written `for t in range(T)` loop |
| Profiles applied by | `ConstControl` data sources | explicit `net.sgen.loc[...] = ...` |
| Results captured by | `OutputWriter` logs | per-step `TimestepRecord` from live tables |
| Per-step control | **none** | coordinator/SVC injected between pre- and post-PF |
| Convergence known via | NaN rows in the log (`continue_on_divergence`) | direct `runpp` result |

**Why baseline can be declarative.** `run_timeseries` runs a closed pre-defined
sequence: apply profile → `runpp` → log. There is no place to insert "read voltages,
compute Q, re-solve" — and baseline needs none. The controlled scenarios need exactly
that insertion, so they cannot use `run_timeseries` and write the loop by hand
(Documents 6C/6D). Understanding this contrast explains why the five runners look so
different despite producing the same `ScenarioResult`.

---

## 5. Section A — _RUNPP_TIMING & _timed_runpp (L76–86)

```python
_RUNPP_TIMING: dict = {}

def _timed_runpp(net, **kwargs):
    t0 = time.perf_counter()
    pp.runpp(net, **kwargs)
    dt_ms = (time.perf_counter() - t0) * 1e3
    ts_step = net["output_writer"].iat[0, 0].time_step
    _RUNPP_TIMING[ts_step] = _RUNPP_TIMING.get(ts_step, 0.0) + dt_ms
```

#### Line 76–86: The per-timestep timing hook

`run_timeseries` accepts a `run=` callable that it uses instead of the default
`pp.runpp`. `_timed_runpp` is that callable: it times the solve and stores the
milliseconds keyed by the **OutputWriter's current `time_step`** (read from
`net["output_writer"].iat[0,0].time_step`). **Design rationale — why a hook, not a
wrapper around `run_timeseries`.** The declarative engine hides per-step timing;
the only way to capture it is to intercept each `runpp` call, which the `run=` hook
provides. The `+= dt_ms` accumulation handles a step that solves more than once
(e.g. a `voltage_depend_loads` retry) — the total for that step is the sum.

**⚠ Flagged bug — `_RUNPP_TIMING` is never cleared.** The comment at L73–76 says
*"Cleared at the start of each call; populated by `_timed_runpp`; read back during
post-processing"* — but **`run_scenario_1` never clears it.** Because the dict is
module-global and `_timed_runpp` *accumulates* (`get(step, 0.0) + dt_ms`), a
**second** `run_scenario_1` call in the same process adds its timings to the first
run's for every shared `time_step` key, inflating `t_total_ms`. Dormant for a single
run per process; wrong for repeated runs (a parametric sweep, a test suite). **Recommend:**
add `_RUNPP_TIMING.clear()` at the top of `run_scenario_1` (which the docstring
already claims happens). **Flagged, not edited.**

---

## 6. Section B — run_scenario_1: setup & integer reindex (L119–154)

#### Checkpointing params — accepted but deliberately not implemented (L95–96)

```python
        enable_checkpointing: bool = True,   # accepted, not implemented — see note below
        live_csv_rewrite_fn         = None,  # accepted, not implemented — see note below
```

Unlike the other runners, baseline's signature **accepts `enable_checkpointing` and
`live_csv_rewrite_fn` but does not act on them**, and its docstring says so explicitly.
**Design rationale:** `benchmark_runner._build_kwargs` injects these kwargs into *every*
runner unconditionally (Doc 9), so baseline must accept them for signature
compatibility — but `run_timeseries()` is a **single opaque call** covering the whole
year with **no per-timestep hook**, so neither crash-resume nor a live progress CSV is
possible here without abandoning `run_timeseries()` entirely (the very declarative
engine that makes baseline simple, §4). So the parameters are accepted and ignored, by
design — the honest choice given the engine, and a direct consequence of the
declarative-vs-manual split. (The manual-loop scenarios 2–4, by contrast, fully
implement both.)

```python
    ap: AdaptedProfiles = adapt_profiles(net, profiles)
    _T = len(ap.times)
    time_steps = list(range(len(ap.times)))
    def _integer_indexed(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy(); out.index = time_steps; return out
    der_p_ts = _integer_indexed(ap.der_p); load_p_ts = _integer_indexed(ap.load_p); load_q_ts = _integer_indexed(ap.load_q)
    for name, df in {"load_p": load_p_ts, "load_q": load_q_ts, "der_p": der_p_ts}.items():
        missing = pd.Index(time_steps).difference(df.index)
        if len(missing): raise ValueError(f"{name} is missing {len(missing)} requested integer time steps. First missing: {missing[0]}")
```

#### Line 112–147: Adapt, reindex to integer steps, validate

Calls `adapt_profiles` (6A-i) once, then **reindexes the three profile frames from
their `DatetimeIndex` to integer steps `0..T-1`** via `_integer_indexed`. **Design
rationale — why the integer reindex (docstring L34–35, and the 6A-i `slice_profiles`
note).** `ConstControl`'s `DFData` data source is addressed by **integer positional
timestep**, not calendar label; `run_timeseries(time_steps=range(T))` iterates
integers. So the profiles must carry a 0-based integer index for the data source to
line up. This is also what makes sliced profiles (6A-i `slice_profiles`) work here —
the calendar labels are discarded and rebuilt as `0..len-1`. The validation loop
then asserts no requested integer step is missing from any frame, failing early with
the first gap named. **Worked example:** a 35,040-step annual series →
`time_steps = [0..35039]`; each profile frame reindexed to that.

---

## 7. Section C — [1]–[2] reset & ConstControl (L149–190)

```python
    net.controller.drop(net.controller.index, inplace=True)   # [1]
    pp.reset_results(net)
    if not ap.der_p.empty:                                     # [2]
        ctrl.ConstControl(net, element="sgen", element_index=ap.der_p.columns.tolist(),
                           variable="p_mw", data_source=DFData(der_p_ts), profile_name=ap.der_p.columns.tolist())
    if not ap.load_p.empty:
        ctrl.ConstControl(net, element="load", element_index=..., variable="p_mw", ...)
        ctrl.ConstControl(net, element="load", element_index=..., variable="q_mvar", data_source=DFData(load_q_ts), ...)
```

#### Line 149–190: Clear controllers, attach the profile data sources

**[1]** drops any controllers left on the net from a prior run and resets result
tables — a clean slate, since `run_timeseries` would otherwise re-fire stale
controllers. **[2]** attaches one `ConstControl` per (element, variable): DER
`p_mw`, load `p_mw`, and load `q_mvar`. **Design rationale — `element_index` and
`profile_name` are the *element indices*, not positions (docstring L34).**
`ConstControl` maps each profile column to the network element with that index; using
positional integers would apply DER 0's profile to whatever sgen sits at position 0,
which in SimBench's sparse indexing is generally the wrong element. The `if not
empty` guards skip attaching a controller for an absent element type (a network with
no DERs). **Note:** load `q_mvar` is driven by `ap.load_q` (the Q/P-ratio-derived
reactive profile from 6A-i), so the baseline preserves each load's power factor
across time.

---

## 8. Section D — [3]–[4] OutputWriter & run_timeseries (L192–221)

```python
    ow = OutputWriter(net, time_steps, output_path=None, output_file_type=None, log_variables=list())  # [3]
    ow.log_variable("res_bus", "vm_pu"); ow.log_variable("res_line", "loading_percent"); ow.log_variable("res_trafo", "loading_percent")
    ow.log_variable("res_line", "pl_mw"); ow.log_variable("res_trafo", "pl_mw"); ow.log_variable("res_ext_grid", "p_mw")
    ts.run_timeseries(net, time_steps=time_steps, continue_on_divergence=True,   # [4]
                      verbose=False, voltage_depend_loads=False, run=_timed_runpp)
```

#### Line 192–221: Configure logging and run the annual series

**[3]** creates an in-memory `OutputWriter` (no disk output — `output_path=None`)
logging exactly the six tables post-processing needs: bus voltage, line/trafo
loading, line/trafo losses (`pl_mw`), and slack-bus power (`res_ext_grid.p_mw`).
**Design rationale:** logging only these keeps memory low over 35k steps; the
empty `log_variables=list()` start avoids pandapower's default (larger) log set.

**[4]** runs the series. **Three mandatory kwargs, each justified:**
- **`voltage_depend_loads=False`** — the SimBench mandatory flag (singular matrix
  without it on pandapower 3.2+). Passed as a direct kwarg that `run_timeseries`
  forwards to the internal `runpp` (docstring L206–208, confirmed against pandapower
  3.4.0 source).
- **`continue_on_divergence=True`** — one non-converging step must not abort the
  whole annual run; diverged steps become NaN rows in the log (detected in §9).
- **`run=_timed_runpp`** — the timing hook (§5).

---

## 9. Section E — [5] post-processing (L223–327)

```python
    vm_log = ow.output.get("res_bus.vm_pu", pd.DataFrame()); ll_log = ...; tl_log = ...; pl_line_log = ...; pl_trafo_log = ...; p_grid_log = ...
    records = []
    for i, t in enumerate(time_steps):
        converged = (not vm_log.empty and t in vm_log.index and not vm_log.loc[t].isna().all())
        if not converged:
            der_gen_mw_t = float(ap.der_p.iloc[i].sum()) if not ap.der_p.empty else 0.0
            load_mw_t = float(ap.load_p.iloc[i].sum()) if not ap.load_p.empty else 0.0
            records.append(TimestepRecord(t=i, ..., converged=False, der_gen_mw=der_gen_mw_t, load_mw=load_mw_t, t_total_ms=_RUNPP_TIMING.get(t, None)))
            continue
        vm = vm_log.loc[t].dropna(); ll = ...; tl = ...
        ov_buses = vm.index[vm > v_max + VOLTAGE_EPSILON].tolist()   # + uv / ol / ot
        losses_mw_t = float(pl_line_log.loc[t].sum() + pl_trafo_log.loc[t].sum()) if (...) else None
        grid_import_mw_t = float(p_grid_log.loc[t].sum()) if ... else None
        rec = TimestepRecord(t=i, ..., converged=True, losses_mw=losses_mw_t, grid_import_mw=grid_import_mw_t, ...)
        records.append(rec)
```

#### Line 223–327: Turn the logs into TimestepRecords

Reads the six log DataFrames, then builds one `TimestepRecord` per step.

**Convergence detection is indirect (L239–243).** Because `continue_on_divergence`
leaves diverged steps as NaN in the log, "converged" = *the vm log has a row for
this step and it is not all-NaN*. This is the declarative-engine equivalent of the
manual scenarios' direct `runpp` convergence flag. Non-converged steps still record
**profile-derived** `der_gen_mw`/`load_mw` (available without a PF), with everything
else empty/`None`.

**Violation thresholding (L283–286) uses the same `±ε` as `make_record_from_report`
and `violation_detector`** — so baseline's violation lists are consistent with every
other scenario's, which is essential for a fair comparison. **Energy balance** for a
converged step: losses = `Σ line pl_mw + Σ trafo pl_mw`; `grid_import_mw` = slack-bus
`p_mw` sum (signed). `t_total_ms` reads the timing hook's accumulator.

**Design rationale — build the record inline, not via `make_record_from_report`.**
The factory (6A-ii) reads from *live* `net.res_*` tables (one settled state); baseline
reads from the *whole-series* `OutputWriter` logs after the run, indexed by step. The
factory does not fit that access pattern, so records are built inline. **This is why
`make_record_from_report` is imported (L60) but unused here** — a dead import (§11).
All control fields (`q_applied_mvar`, `p_applied_mw`, `p_target_mw`,
`curtailment_needed`) are `None`/`False`: baseline applies no control. The optional
`publish_fn.on_timestep(rec)` streams each record to the live dashboard (Mateo's
layer).

---

## 10. Section F — from_records & return (L329–346)

```python
    result = ScenarioResult.from_records(scenario_id="baseline", network_id=network_id, records=records, elapsed_s=elapsed, dt_s=ap.dt_s)
    logger.info("[Scenario 1 | %s] Done. %.1f s | %d/%d converged | %d violation steps", ...)
    if publish_fn is not None: publish_fn.on_scenario_end(result)
    return result
```

#### Line 329–346: Aggregate and return

Hands the record list to `ScenarioResult.from_records` (6A-ii) with `dt_s = ap.dt_s`
(so `violation_duration_h` and the energy metrics use the correct timestep), tagged
`scenario_id="baseline"`. Because every record has `q_applied_mvar = None`, the
result's reactive/curtailment/coordination metrics all come back `None` — the
comparison layer reads baseline as pure "uncontrolled network" numbers (VDI,
violation duration, losses, grid exchange). **Worked example:** on
`1-MV-rural--2-sw` the baseline typically shows the largest `n_violation_steps` and
VDI of the five scenarios — the reference the controlled scenarios reduce.

---

## 11. Findings and flags

**Flagged, not edited:**

1. **`_RUNPP_TIMING` never cleared (L76 + `run_scenario_1`).** The docstring (L73–76)
   states it is "cleared at the start of each call", but no clear exists, and
   `_timed_runpp` accumulates. A second `run_scenario_1` in the same process inflates
   `t_total_ms` for every shared step. **Recommend: `_RUNPP_TIMING.clear()` at the top
   of `run_scenario_1`.**
2. **`make_record_from_report` imported but unused (L60).** Baseline builds
   `TimestepRecord` inline from the `OutputWriter` logs because the factory reads live
   `net.res_*` tables, which don't fit the post-run log access pattern. Dead import —
   remove, or (better) note why the factory is not used here.

**Not a bug (documented for clarity):** the indirect convergence detection (NaN log
rows) is the correct idiom for `continue_on_divergence=True`, not a workaround.

---

## 12. Coverage checklist

Every named entity in `scenario_1_baseline.py`, each confirmed documented.

| Entity | Lines | Kind | Documented | Where-used stated |
|--------|-------|------|:----------:|:-----------------:|
| `_RUNPP_TIMING` | 76 | module dict | ✅ (never-cleared bug flagged) | ✅ |
| `_timed_runpp` | 78–86 | function | ✅ | ✅ (run= hook) |
| `run_scenario_1` | 88–346 | function | ✅ | ✅ (benchmark_runner) |
| — setup + integer reindex + validation | 112–147 | — | ✅ | — |
| — `_integer_indexed` (nested) | 120–123 | function | ✅ | ✅ |
| — [1] reset controllers/results | 149–153 | — | ✅ | — |
| — [2] ConstControl (der p, load p, load q) | 155–190 | — | ✅ | — |
| — [3] OutputWriter | 192–202 | — | ✅ | — |
| — [4] run_timeseries | 204–221 | — | ✅ | — |
| — [5] post-processing loop | 223–327 | — | ✅ | — |
| — from_records + return | 329–346 | — | ✅ | — |

**Also covered (not standalone entities):** the declarative-vs-manual architectural
contrast (§4), the integer-reindex rationale (`ConstControl` positional addressing),
the element-index (not positional) `profile_name` mapping, the three mandatory
`run_timeseries` kwargs, the indirect NaN-based convergence detection, and the
±ε violation thresholding consistent with `violation_detector`.

**Flags raised (not edited):** `_RUNPP_TIMING` never cleared (docstring says it is);
`make_record_from_report` imported but unused.

---

*End of Document 6A-iii — completing Document 6A (the shared scenario infrastructure
6A-i/6A-ii + the baseline runner 6A-iii). Next per the queue: Document 6B — Scenario
Runner: OLTC (`scenario_2_oltc.py`), the first manual-loop scenario, with the tap
state machine the `TimestepRecord` OLTC block (6A-ii) records.*
