# TASK 4 — DOCUMENT 6A-ii  (v3)
## Scenario Result Infrastructure, Part 2: The Result Types
### `scenario_result.py` — `TimestepRecord` (L532–815), `make_record_from_report` (L822–889), `ScenarioResult` (L891–1348)

> **Source re-verified (v3):** re-pinned against the current `/mnt/project/scenario_result.py`
> (**1,348 lines** — the project tree, now newer than the earlier upload v2 used). The file
> gained (i) the applied cross-version fix + `_ckpt_version` stamp in the checkpoint
> methods, and (ii) a **new timestep-coverage integrity check** in `from_records`
> (§7). Current anchors:
>
> | Entity | Line |
> |---|---|
> | `TimestepRecord` | 532 |
> | `to_checkpoint_dict` (`_ckpt_version` at 763) | 718 |
> | `from_checkpoint_dict` (version check 778; `d.get` fields 801–805) | 767 |
> | `_empty_series` | 817 |
> | `make_record_from_report` | 822 |
> | `ScenarioResult` | 891 |
> | `from_records` (coverage check 1096–1135; `min_vm` 1147; VDI 1169) | 1072 |
> | `summary_dict` | 1309 |
>
> **✅ Resolved since v2:** the `min_vm` double-compute (now single, guarded, L1147);
> the `from_checkpoint_dict` cross-version `d[...]` → `d.get(...)` fix **and** a
> `_ckpt_version` schema stamp (both my suggested fixes, now applied); and the
> publisher NaN→null encoder (Doc 10). The three §9 flags from v2 are now closed.
>
> **Part 2 of 3 of Document 6A** (6A-i = the profile adapter — *confirmed
> unaffected* by this update; 6A-iii = `scenario_1_baseline.py`).
>
> **Read alongside:** **6A-i**; Master Index §5–§8; the runners **6A-iii/6B/6C/6D/6E**;
> **Document 10** (`publisher.py`, whose `PublishHandle` now writes the checkpoint
> stream that `from_checkpoint_dict` reads back).

---

## TABLE OF CONTENTS

1. [Overview](#1-overview)
2. [File logic flow — pseudocode](#2-file-logic-flow--pseudocode)
3. [Call-site map](#3-call-site-map)
4. [Section A — TimestepRecord (L531–808)](#4-section-a--timesteprecord-l531808)
   - [4.5 Checkpoint serialisation (L717–808)](#45-checkpoint-serialisation-l717808)
5. [Section B — _empty_series & make_record_from_report (L811–877)](#5-section-b--_empty_series--make_record_from_report-l811877)
6. [Section C — ScenarioResult fields (L885–1064)](#6-section-c--scenarioresult-fields-l8851064)
7. [Section D — from_records (L1066–1278)](#7-section-d--from_records-l10661278)
8. [Section E — summary_dict (L1280–1319)](#8-section-e--summary_dict-l12801319)
9. [Findings and flags](#9-findings-and-flags)
10. [Coverage checklist](#10-coverage-checklist)

---

## 1. Overview

These are the types every scenario runner produces. The design contract is
**uniformity**: whichever of the five runners produced it, a `ScenarioResult`
exposes the same scalar metrics, so the comparison/plotting/publishing layer
(Documents 9–10) treats all scenarios identically. Two levels:

- **`TimestepRecord`** — one per simulated timestep, holding the raw per-element
  state plus pre-derived violation lists plus `Optional` per-scenario fields, and
  (as of the checkpointing update) **its own lossless (de)serialisation** for
  crash-resume.
- **`ScenarioResult`** — the aggregate over the whole series, with ~30 derived
  metrics computed once via `from_records()`.

**The `None`-means-not-applicable convention** runs through both: a field is
`None` when the scenario does not produce that output. This lets one record/result
type serve all five scenarios without subclassing — the metric computations skip
`None` fields.

**New in the checkpointing update.** `TimestepRecord` now round-trips through
`to_checkpoint_dict`/`from_checkpoint_dict` (§4.5). The runners write full records
to `checkpoint/<id>.jsonl` (via `PublishHandle`, Doc 10) at the `update_every_k`
cadence; on restart they call `publish_fn.get_resume_records(...)`, which rebuilds
`TimestepRecord`s from that file, and continue from the last completed step. This
is why the runners now take `enable_checkpointing`/`live_csv_rewrite_fn` and seed
their per-DER state (tap position, PT1/ramp state) from the last recovered record.

**The paper's headline metrics live here:** `vdi`, `violation_duration_h`,
`coordination_rate`, and `reactive_energy_mvarh` are all computed in `from_records`
(§7). This document derives each precisely.

---

## 2. File logic flow — pseudocode

```
DATACLASS TimestepRecord:
    always present: t, timestamp, vm_pu, line/trafo loading, the four violation
                    lists, converged, energy-balance scalars
    optional (None when N/A): q/p applied, p_target, curtailment flags;
                    OLTC tap block; SVC command block; Volt-Var coordination block
    checkpoint I/O:
        to_checkpoint_dict()   → plain dict (Series→dict, timestamp→isoformat)
        from_checkpoint_dict() → rebuild a TimestepRecord (dict→Series, iso→Timestamp)

FUNCTION make_record_from_report(t, ts, net, converged, ...):
    not converged? → record with empty result Series + empty violation lists
    converged?     → read vm/line/trafo from net result tables; threshold each
                     against V_MIN/V_MAX/LOADING limits (± epsilon) → violation lists

DATACLASS ScenarioResult:
    identifiers + the record list + ~30 derived scalar metrics (default 0/NaN/None)

CLASSMETHOD from_records(...):
    filter to converged; count timesteps/violations; extremes (guarded — except min_vm);
    VDI = Σ|vm−1| ; reactive/curtailed energy ; energy balance ; control effort (S4)

METHOD summary_dict(): flatten every scalar metric into a dict (for DataFrame rows)
```

---

## 3. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `TimestepRecord` | dataclass | Constructed in **all five runners** (`scenario_1..5`) + the factory below. |
| `TimestepRecord.to_checkpoint_dict` | method | `PublishHandle` checkpoint writer (`publisher.py`, Doc 10) — serialises each record to `checkpoint/<id>.jsonl`. |
| `TimestepRecord.from_checkpoint_dict` | classmethod | `PublishHandle.get_resume_records` (`publisher.py`) — rebuilds records on resume; consumed by every runner's resume block (`scenario_2..5`). |
| `_empty_series` | function | `make_record_from_report` (non-converged branch). |
| `make_record_from_report` | function | The record **factory** — `custom_controller.py`; imported by `scenario_1_baseline`. (Scenarios 2/3/4/5 build `TimestepRecord` inline.) |
| `ScenarioResult` | dataclass | Returned by `from_records`; consumed by the comparison/publisher layer. |
| `ScenarioResult.from_records` | classmethod | **The universal aggregator** — every runner + `custom_controller`; also the runners' resume/live-CSV partial-result builders. |
| `ScenarioResult.summary_dict` | method | `publisher.py`, `benchmark_runner.py` (comparison table), `scenario_1/2`. |

---

## 4. Section A — TimestepRecord (L531–808)

The record groups its fields into an always-present core and four `Optional`
per-scenario blocks, followed (as of the update) by the two checkpoint methods.
Every field is enumerated in §10; here they are covered by group.

#### Line 646–658: The always-present core

```python
    t: int; timestamp: pd.Timestamp
    vm_pu: pd.Series; line_loading: pd.Series; trafo_loading: pd.Series
    over_voltage_buses: List[int]; ...; overloaded_trafos: List[int]
    q_applied_mvar: Optional[pd.Series]; p_applied_mw: Optional[pd.Series]
    curtailment_needed: bool; converged: bool
```

The state every scenario produces. **Design rationale — two key choices.** (1) The
result Series are **indexed by pandapower element index, not positional integer**,
so they join back to `net.bus`/`net.line`/`net.trafo` for plotting; empty typed
Series signal non-convergence. (2) The **four violation lists are pre-derived** —
computed once by the factory (§5) or inline by each runner — so the comparison
layer counts with `len(...)` and never re-thresholds. `q_applied_mvar`/`p_applied_mw`
are `Optional` (`None` for Baseline).

#### Line 659–669: p_target and the OLTC (Scenario 2) block

`p_target_mw` (L659, raw profile P before ramp limiting) is a proper dataclass field
so it survives export/copy and `from_records` can compute `curtailed_energy_mwh`.
The six `tap_*` fields (L661–669) are **OLTC-only** and encode a precise tap-action
state machine (Doc 6B): `tap_attempted` vs `tap_changed` distinguishes "no action"
from "action blocked"; `tap_blocked_reason` names *why* a tap was refused (pre-PF
divergence, post-PF divergence, ganged-range limit).

#### Line 671–715: SVC (S3), energy-balance, and Volt-Var (S4) blocks

- **SVC block** (L671–673): `svc_q_mvar`, `svc_saturated` — the reactive command
  and droop saturation (Scenario 3 only).
- **Energy-balance scalars** (L675–695): `losses_mw`, `grid_import_mw` (signed:
  + import / − export), `der_gen_mw`, `load_mw` — populated by **all** runners;
  `der_gen_mw`/`load_mw` come from the *profile*, so they exist even on
  non-converged steps.
- **Volt-Var block** (L697–715): `coordination_active` (coordinator moved
  `q_initial` by >1e-6 MVAr), `q_saturated_count`, `curtail_exhausted`,
  `hil_latency_ms`, `t_total_ms` — Scenario 4 only.

**Design rationale — one wide record, not five subclasses.** All per-scenario fields
default to `None`, so a single dataclass serves every runner and every metric guards
on `is not None`. This is why the comparison layer is scenario-agnostic — and why the
checkpoint methods (§4.5) can serialise every scenario's record with one code path.

### 4.5 Checkpoint serialisation (L717–808)

Added in the checkpointing update: lossless round-trip to and from a plain dict, so
a `TimestepRecord` can be written to a JSONL checkpoint file and rebuilt on resume.

#### Line 717–762: to_checkpoint_dict — lossless serialisation

```python
    def to_checkpoint_dict(self) -> dict:
        def _s(x): return None if x is None else x.to_dict()
        return { "t": self.t, "timestamp": self.timestamp.isoformat(),
                 "vm_pu": _s(self.vm_pu), ..., "t_total_ms": self.t_total_ms }
```

Serialises **every** field to JSON-friendly types: Series via `_s` (`.to_dict()` →
`{element_index: value}`, or `None`), the timestamp via `.isoformat()`, and the
scalars/lists/bools as-is. **Design rationale — "lossless, distinct from the live
frame".** The docstring is explicit: this is *not* the compact dashboard live frame
(`build_live_frame`, Doc 10), which drops fields the voltage-heatmap animation does
not need. Checkpointing needs *every* field so a resumed run is byte-for-byte a
continuation — including the per-scenario blocks (`tap_pos`, `svc_q_mvar`,
`coordination_active`) that seed the resumed runner's state. The fields are grouped
with comments by scenario, mirroring the dataclass. **Where used:** the
`PublishHandle` checkpoint writer (Doc 10).

#### Line 764–808: from_checkpoint_dict — reconstruction

```python
    @classmethod
    def from_checkpoint_dict(cls, d: dict) -> "TimestepRecord":
        def _d(x):
            if x is None: return None
            return pd.Series({int(k): v for k, v in x.items()}, dtype=float)
        def _series_or_empty(x):
            s = _d(x); return s if s is not None else pd.Series(dtype=float)
        return cls(t=d["t"], timestamp=pd.Timestamp(d["timestamp"]),
                   vm_pu=_series_or_empty(d["vm_pu"]), ..., tap_pos=d.get("tap_pos"), ...)
```

Rebuilds a `TimestepRecord` from `to_checkpoint_dict` output. Two nested helpers:
`_d` reconstructs an optional Series (returns `None` for a `None` field);
`_series_or_empty` reconstructs an always-present result Series (empty Series if
`None`). **Two design details worth calling out:**

- **`{int(k): v for k, v in x.items()}` — the JSON string-key fix.** After JSONL
  serialisation, a Series' integer element-index keys become **strings** (`"12"`);
  `int(k)` restores them to the integer element indices the rest of the pipeline
  expects. Correct and necessary.
- **`d[...]` vs `d.get(...)` asymmetry.** Always-present fields and the energy-balance
  scalars use `d["..."]` (a `KeyError` if absent); the per-scenario optionals use
  `d.get("...")` (defaulting to `None`). **Design rationale:** the optionals may be
  absent from a checkpoint written by a scenario that doesn't produce them, so
  `.get` is correct there; the required fields are always written by
  `to_checkpoint_dict`, so `d[...]` is a valid within-version assumption. *(Task 2
  note: `losses_mw`/`grid_import_mw`/`der_gen_mw`/`load_mw` use `d[...]` — a checkpoint
  written by an **older** version lacking those keys would `KeyError`. Within-version
  it is fine; flagged in §9 as a minor cross-version robustness point.)*

**Round-trip correctness.** For the fields it covers the round-trip is lossless,
**provided the checkpoint writer serialises `NaN` safely** (a non-converged step's
`vm_pu` contains `NaN`). Standard `json` emits invalid `NaN`; the write path must use
publisher's NaN→null `_Encoder`, and on read `null → None → NaN` via
`pd.Series(dtype=float)`. §9 flags this as a point to verify in the writer.
**Where used:** `PublishHandle.get_resume_records` → the runners' resume blocks.

---

## 5. Section B — _empty_series & make_record_from_report (L811–877)

```python
def _empty_series(dtype=float) -> pd.Series:
    return pd.Series(dtype=dtype)
```

#### Line 811–813: The typed empty-Series helper

Returns a properly-typed empty Series for the non-converged branch — the explicit
`dtype=float` keeps non-converged records type-consistent with converged ones and
avoids the untyped-Series deprecation. **Where used:** the factory's non-converged
branch.

#### Line 816–877: The record factory — read net tables, threshold once

```python
def make_record_from_report(t, timestamp, net, converged, v_min=V_MIN, v_max=V_MAX, ...):
    if not converged: return TimestepRecord(..., vm_pu=_empty_series(), ..., converged=False)
    vm = net.res_bus["vm_pu"].copy(); ll = ...; tl = ...
    ov_buses = vm.index[vm > v_max + VOLTAGE_EPSILON].tolist()  # + uv / ol / ot
    return TimestepRecord(..., over_voltage_buses=ov_buses, ..., converged=True)
```

Builds a `TimestepRecord` from the *current* `net` result tables. **Design
rationale — the ± epsilon in every threshold.** `vm > v_max + VOLTAGE_EPSILON` uses
the same `VOLTAGE_EPSILON`/`LOADING_EPSILON` as `violation_detector` (Doc 11), so a
bus exactly on the limit is not flagged by floating-point dust — the record's
pre-derived lists match what `detect_violations` would produce. `.copy()` detaches
the Series from the live `net.res_*` tables. `v_min`/`v_max` default to
`violation_detector.V_MIN/V_MAX` but are overridable (CLI band flows through).
**Where used:** baseline + custom_controller. **Worked example:** a converged step
with bus 12 at 1.058 pu → `ov_buses = [12]`.

---

## 6. Section C — ScenarioResult fields (L885–1064)

The aggregate's ~30 fields fall into groups; each is derived in `from_records` (§7)
and every one is listed in §10. Identifiers and the record list are caller-set; **all
derived fields default to `0`/`NaN`/`None`** (L1037–1064) so a `ScenarioResult` is
valid even before `from_records` fills them.

The derived metrics, by group: **counts** (`n_timesteps`, `n_converged`,
`n_violation_steps`, the four `total_*_steps`); **duration** (`violation_duration_h`);
**voltage quality** (`max_vm_pu`, `min_vm_pu`, `max_line_loading_pct`,
`max_trafo_loading_pct`, `vdi`); **reactive effort** (`q_total_mvar_abs`,
`reactive_energy_mvarh`); **curtailment** (`curtailment_steps`,
`curtailed_energy_mwh`, `curtail_exhausted_steps`); **SVC** (`svc_bus`, `svc_q_max`);
**energy balance** (`total_losses_mwh`, `grid_import_mwh`, `grid_export_mwh`,
`der_gen_mwh`, `load_demand_mwh`); **control effort** (`coordination_steps`,
`coordination_rate`, `q_saturation_rate`).

**Design rationale — store scalars, not the record list, for metrics.** Metrics are
computed once and stored as plain numbers so the comparison layer never iterates the
records. The record list *is* retained (`records` field) for post-hoc plotting/export
(and is what the checkpoint stream persists), but no derived metric reads it after
construction.

---

## 7. Section D — from_records (L1072–1308)

The classmethod that turns a record list into all derived metrics. Walked by metric
group; the definitions are the ones the paper reports. **The checkpointing update
added a timestep-coverage integrity guard at the top** (below); the metric logic
below it is otherwise unchanged.

#### Line 1096–1135: ⭐ Timestep-coverage integrity check (new)

```python
    if records:
        observed_ts = sorted(r.t for r in records)
        expected_ts = list(range(observed_ts[0], observed_ts[-1] + 1))
        if observed_ts != expected_ts:
            missing = sorted(set(expected_ts) - set(observed_ts))
            raise ValueError(f"... records has {len(missing)} gap(s) in timestep coverage ... "
                             "a checkpoint file was written with a sparse cadence ... resumed run "
                             "silently lost coverage ... Do not trust this result — re-run from scratch.")
        dup_ts = [t for t, c in Counter(observed_ts).items() if c > 1]
        if dup_ts:
            raise ValueError(f"... records has duplicate timestep(s): {sorted(set(dup_ts))[:10]} ... "
                             "resumed_records and newly-simulated records overlapped — check start_t.")
```

**A resume-safety guard added with the checkpointing feature.** Before computing any
metric, `from_records` verifies the record set is a *contiguous, gap-free, duplicate-free*
run of timesteps: `observed_ts` must equal the full integer range from the first to the
last `t`. **Two failure modes it catches, both resume-specific:**
- **Gaps** — a checkpoint written at the sparse `update_every_k` cadence (instead of
  every timestep) would, on resume, silently lose coverage of the pre-resume segment;
  the aggregate metrics (VDI, violation counts) would then be computed over an
  incomplete year and be quietly wrong. The guard raises with the missing timesteps
  named and the instruction to re-run.
- **Duplicates** — if `resumed_records` and the newly-simulated records overlap (a
  `start_t` miscomputation), a timestep would be counted twice. The guard raises
  naming the duplicates.

**Design rationale:** the metrics are only meaningful over a complete, non-overlapping
year; a resumed run that silently lost or double-counted steps would produce a
plausible-but-wrong `ScenarioResult`. Failing loudly here — "do not trust this result"
— is far safer than emitting a corrupted benchmark number. `Counter` (new import) does
the duplicate detection. This directly hardens the checkpoint/resume path documented in
§4.5 and across the runners.

#### Line 1137–1145: Counts and violation-step sums

```python
    converged = [r for r in records if r.converged]
    n_violation_steps = sum(1 for r in converged if r.over_voltage_buses or ... )
    violation_duration_h = n_violation_steps * dt_s / 3600.0
    total_ov = sum(len(r.over_voltage_buses) for r in converged)   # + uv / ol / ot
```

`n_violation_steps` counts converged steps with *any* violation; the four `total_*`
sum the per-step violation counts. **`violation_duration_h = n_violation_steps ×
dt_s / 3600`** converts step counts to hours — comparable across networks with
different `dt_s` (480 steps at 15-min = 120 h/yr).

#### Line 1147–1155: Voltage/loading extremes (min_vm double-compute now fixed)

```python
    if converged:
        max_vm = max((r.vm_pu.max() for r in converged if not r.vm_pu.empty), default=float("nan"))
        min_vm = min((r.vm_pu.min() for r in converged if not r.vm_pu.empty), default=float("nan"))
        max_ll = max((r.line_loading.max() for r in converged if not r.line_loading.empty), default=float("nan"))
        max_tl = max((r.trafo_loading.max() for r in converged if not r.trafo_loading.empty), default=float("nan"))
    else:
        max_vm = min_vm = max_ll = max_tl = float("nan")
```

Network-wide extremes across converged timesteps. **✅ The `min_vm` double-compute
bug is fixed:** `min_vm` is now computed **once**, with the same
`default=float("nan")` empty-sequence guard as `max_vm`/`max_ll`/`max_tl`, so an
all-empty-`vm_pu` converged set yields `NaN` rather than raising `ValueError`. (The
previously-flagged redundant second computation has been removed; this is why the
line numbers below shift −1 from the v2 header estimate.)

#### Line 1133–1155: VDI and reactive energy

```python
    vm_records = [r for r in converged if not r.vm_pu.empty]
    vdi = float(sum((r.vm_pu - 1.0).abs().sum() for r in vm_records)) if vm_records else float("nan")
    q_records = [r for r in converged if r.q_applied_mvar is not None]
    q_total = sum(r.q_applied_mvar.abs().sum() for r in q_records) if q_records else None
    reactive_energy_mvarh = float(q_total) * (dt_s / 3600.0) if q_total is not None else None
```

**VDI (Voltage Deviation Index):** `Σ_t Σ_bus |vm − 1.0|` over all converged
timesteps. **Design rationale — not normalised:** VDI is additive and directly
comparable across scenarios on the same network; it captures continuous
voltage-quality degradation that binary violation counts miss (a network at 1.049
pu everywhere has zero violations but a large VDI). **Reactive energy:**
`Σ|q_applied| × dt/3600` → MVArh/yr — `None` for Baseline.

#### Line 1157–1175: Curtailment metrics

```python
    curtailment_steps = sum(1 for r in records if r.curtailment_needed)
    for r in p_records: curtailed_mw = (r.p_target_mw - r.p_applied_mw).clip(lower=0.0); mwh += curtailed_mw.sum() * (dt_s/3600.0)
    curtail_exhausted_steps = sum(1 for r in records if r.curtail_exhausted is True)
```

**Curtailed energy:** `Σ_t Σ_i max(0, p_target_i − p_applied_i) × dt/3600` — the
active energy not delivered because of ramp-limiting/curtailment; `.clip(lower=0.0)`
keeps only reductions. `None` for Baseline and OPF.

#### Line 1177–1233: Energy balance and control effort

```python
    grid_import_mwh = float(sum(max(0.0, r.grid_import_mw) for r in grid_recs)) * (dt_s/3600.0)
    grid_export_mwh = float(sum(abs(min(0.0, r.grid_import_mw)) for r in grid_recs)) * (dt_s/3600.0)
    coordination_rate_val = coordination_steps_val / len(converged) if coord_recs and converged else None
    q_saturation_rate_val = sum(1 for r in sat_recs if r.q_saturated_count > 0) / len(converged) if ... else None
```

**Grid import/export** splits the *signed* `grid_import_mw` into two non-negative
totals (`max(0,·)` imports, `|min(0,·)|` exports). **Control effort (Scenario 4):**
`coordination_rate = coordination_steps / n_converged` — the paper's 9.1%/15.2%
figures, the metric the quantisation finding inflates on the HIL path;
`q_saturation_rate` = fraction of steps with ≥1 DER at its reactive ceiling. All
`None` outside Scenario 4.

#### Line 1235–1278: Assemble and log

Constructs the `ScenarioResult` with every computed metric and logs a one-line
per-scenario summary.

---

## 8. Section E — summary_dict (L1280–1319)

```python
    def summary_dict(self) -> dict:
        return { "scenario_id": ..., "vdi": self.vdi, "coordination_rate": ..., "elapsed_s": ... }
```

#### Line 1280–1319: Flatten every scalar metric into a dict

Returns a flat `{name: value}` dict of every scalar derived metric (excluding the
`records` list). **Design rationale:** the comparison layer builds
`pd.DataFrame([r.summary_dict() for r in results])`, and `benchmark_runner` maps it
explicitly (not dynamically) so a new metric doesn't silently appear/disappear from
the comparison table. **Where used:** publisher, benchmark_runner, scenario_1/2.

---

## 9. Findings and flags

**All v2 flags now resolved (verified against the 1,348-line project source):**

1. **`min_vm` double-compute — FIXED.** The redundant, unguarded second computation
   of `min_vm` has been removed; it is now computed once with the
   `default=float("nan")` guard (L1147), matching `max_vm`/`max_ll`/`max_tl`.

2. **Checkpoint NaN → invalid JSON — FIXED (in publisher).** `publisher._append_jsonl`
   *and* `_dump` now apply `_nan_to_none` + `allow_nan=False`, so non-finite floats
   serialise as `null` and the JSONL is strict-valid (Doc 10). The round-trip stays
   lossless (`null → None → NaN` on read).

3. **`from_checkpoint_dict` cross-version `d[...]` KeyError — FIXED.** The
   `None`-defaulted fields (`p_target_mw`, `losses_mw`, `grid_import_mw`, `der_gen_mw`,
   `load_mw`) now use `d.get(...)` (L801–805); required fields keep `d[...]`. A
   **`_ckpt_version` schema stamp** was also added — `to_checkpoint_dict` writes
   `"_ckpt_version": 1` (L763) and `from_checkpoint_dict` warns on a mismatch (L778).
   (Both were my suggested fixes, now applied.)

**New hardening (documented in §7):** a **timestep-coverage integrity check** in
`from_records` (L1096–1135) rejects gapped or duplicated record sets — a resume-safety
guard against sparse-checkpoint coverage loss and resumed/new-record overlap.

**Not bugs (documented for clarity):** the wide `Optional`-heavy `TimestepRecord` is
intentional; the `int(k)` JSON string-key coercion is necessary and correct; the
`d[...]`/`d.get` split for *required vs optional* fields is the right distinction.

---

## 10. Coverage checklist

Every named entity in the 6A-ii range, each confirmed documented against the updated
file.

| Entity | Lines | Kind | Documented | Where-used stated |
|--------|-------|------|:----------:|:-----------------:|
| `TimestepRecord` | 531–808 | dataclass | ✅ | ✅ |
| — core fields (13) | 646–658 | fields | ✅ (grouped) | ✅ |
| — `p_target_mw` + OLTC block (7) | 659–669 | fields | ✅ | ✅ |
| — SVC block (2) | 671–673 | fields | ✅ | ✅ |
| — energy-balance scalars (4) | 675–695 | fields | ✅ | ✅ |
| — Volt-Var block (5) | 697–715 | fields | ✅ | ✅ |
| **`to_checkpoint_dict`** | **717–762** | **method (new)** | ✅ | ✅ |
| **`from_checkpoint_dict`** | **764–808** | **classmethod (new)** | ✅ | ✅ |
| `_empty_series` | 811–813 | function | ✅ | ✅ |
| `make_record_from_report` | 816–877 | function | ✅ | ✅ |
| `ScenarioResult` | 885–1064 | dataclass | ✅ | ✅ |
| — identifiers + records + elapsed | ~1024–1036 | fields | ✅ | ✅ |
| — ~30 derived metric fields | 1037–1064 | fields | ✅ (grouped §6) | ✅ |
| `ScenarioResult.from_records` | 1066–1277 | classmethod | ✅ (min_vm bug fixed) | ✅ |
| `ScenarioResult.summary_dict` | 1279–1318 | method | ✅ | ✅ |

**Also covered (not standalone entities):** the pre-derived violation lists and ±ε
thresholding, the signed-`grid_import_mw` split, the additive-unnormalised VDI, the
`curtailed_energy` `clip(lower=0)`, the `None`-means-not-applicable convention, and
(new) the checkpoint round-trip's JSON string-key coercion and NaN-safety
dependency.

**Status:** `min_vm` double-compute **fixed** (§9.1). Two flags remain with fixes
proposed (not applied): checkpoint NaN-encoder (§9.2), `from_checkpoint_dict`
cross-version `d[...]` (§9.3).

**Deferred:** `scenario_1_baseline.py` → **Document 6A-iii**.

---

*End of Document 6A-ii (v2, reissued for the checkpointing update). The runner-side
resume/checkpoint logic is documented in each runner's walkthrough (6A-iii/6B-ii/6C/
6D/6E); the checkpoint file format and writer in Document 10 (`publisher.py`).*
