# TASK 4 — DOCUMENT 6B-i
## Scenario Runner — OLTC, Part 1: Setup Infrastructure
### `scenario_2_oltc.py` — constants (L86–112) and the setup helpers (L119–590)

> **Source verified:** the cited lines of the current `scenario_2_oltc.py` were
> read directly before writing. If the file is edited, re-verify line numbers.
>
> **6B splits into two parts** (flagged deviation — `scenario_2_oltc.py` is
> **956 lines**, not the "medium" originally sized). Split by responsibility:
> - **6B-i (this document): OLTC setup infrastructure** — the tap-metadata
>   completion/override machinery, the topology-based transformer selection, the
>   ganging validation, and the network-probing tap-sign calibration. These
>   *prepare* the OLTC before any timestep runs.
> - **6B-ii: `run_scenario_2`** — the 7-step orchestration and the per-timestep
>   tap control loop (the state machine whose data the `TimestepRecord` OLTC block
>   records, 6A-ii).
>
> **Read alongside:** Master Index §5–§8; **6A-ii** (the `tap_*` `TimestepRecord`
> fields this scenario fills); **Document 6A-iii §4** (the manual-loop vs
> declarative contrast — OLTC is the first manual-loop scenario).

---

## TABLE OF CONTENTS

1. [Overview — OLTC control and the setup pipeline](#1-overview)
2. [File logic flow — pseudocode (setup)](#2-file-logic-flow--pseudocode-setup)
3. [Call-site map](#3-call-site-map)
4. [Section A — Constants (L86–112)](#4-section-a--constants-l86112)
5. [Section B — _is_missing_tap_value & _print_tap_metadata (L119–150)](#5-section-b--_is_missing_tap_value--_print_tap_metadata-l119150)
6. [Section C — completion & override (L153–219)](#6-section-c--completion--override-l153219)
7. [Section D — _select_oltc_trafos (L222–295)](#7-section-d--_select_oltc_trafos-l222295)
8. [Section E — _validate_tap_metadata (L298–408)](#8-section-e--_validate_tap_metadata-l298408)
9. [Section F — _select_controlled_buses (L411–475)](#9-section-f--_select_controlled_buses-l411475)
10. [Section G — _calibrate_tap_sign (L478–586)](#10-section-g--_calibrate_tap_sign-l478586)
11. [Section H — _empty_series (L589–590)](#11-section-h--_empty_series-l589590)
12. [Coverage checklist](#12-coverage-checklist)

---

## 1. Overview

Scenario 2 models an **On-Load Tap Changer (OLTC)** — the classic distribution
voltage-control device. The HV/MV substation transformer has a tap changer that
adjusts its turns ratio in discrete steps; raising or lowering the tap shifts the
whole downstream MV voltage profile up or down. The OLTC is the *incumbent*
technology the DER-based schemes (SVC, Volt-Var) are compared against.

This file is unusually setup-heavy because a robust OLTC benchmark must work across
networks with inconsistent, incomplete, or absent tap metadata (SimBench, CIGRE,
Kerber, custom). Before a single timestep runs, six setup helpers must:

1. **select** which transformer(s) form the OLTC group (topology-based);
2. **complete** any missing tap metadata from safe defaults;
3. **override** with explicit user values if supplied;
4. **validate** that the group can be *ganged* (controlled as one);
5. **select** the secondary buses whose voltage the OLTC regulates;
6. **calibrate** the tap-sign by probing the network.

Two domain concepts recur:

- **Ganging** — when several parallel transformers share one substation, they are
  driven as a single synchronised OLTC group with one tap command. Ganging requires
  the transformers to share tap metadata (neutral, side, step, type), and the
  group's usable range is the *intersection* `[max(tap_min), min(tap_max)]`.
- **Tap-sign** — whether *raising* the tap raises or lowers the controlled voltage
  depends on `tap_side` (hv vs lv) and the winding convention. Getting it wrong
  would make the controller drive voltage the wrong way. §10 determines it
  *empirically* rather than assuming.

---

## 2. File logic flow — pseudocode (setup)

```
CONSTANTS: OLTC regulation deadband (0.98–1.02, tighter than the 0.95–1.05
           violation band); one tap step per action; a min voltage response to
           confirm the tap is electrically live; default tap-metadata fill values

_is_missing_tap_value(v):   True if v is NaN or the string "none"/"nan"/"<na>"
_print_tap_metadata(...):   audit-log the tap columns (before & after edits)

_complete_missing_tap_metadata(net, trafos, defaults):
    for each tap field: fill ONLY where missing; preserve valid network values; log fills
_apply_user_tap_metadata_override(net, trafos, override):
    write only the user-supplied fields (overrides both network + default values)

_select_oltc_trafos(net) → trafo indices:      [topology only, no tap metadata yet]
    Tier 1: HV/MV trafos (vn_hv ≥ 66 kV)
    Tier 2: trafo whose HV bus is the slack bus (LV-only networks)
    Tier 3: highest-HV trafo (last resort, warn)

_validate_tap_metadata(net, trafos):           [can this group be ganged?]
    required fields present & non-NaN; shared tap_neutral / tap_side / step / type;
    ganged range [max(tap_min), min(tap_max)] non-empty; neutral inside each range;
    type == "Ratio"; no dependency table

_select_controlled_buses(net, trafos) → LV bus list:
    the secondary busbars to measure (mean voltage); all at one voltage level;
    split-busbar allowed (warn, still one synchronised group)

_calibrate_tap_sign(net, trafos, ctrl_buses, neutral, min, max) → ±1:
    on a DEEP COPY: solve at neutral, solve at neutral±1, observe mean Δvoltage;
    return the sign such that (current_tap + sign·step) LOWERS voltage;
    raise if the range is too narrow to probe, or the response is negligible
```

---

## 3. Call-site map

Every entity here is an internal setup helper called by `run_scenario_2`
(Document 6B-ii); none is called from outside the module.

| Entity | Kind | Called from (all in `run_scenario_2`, Doc 6B-ii) |
|--------|------|--------------------------------------------------|
| `OLTC_CONTROL_LOWER/UPPER` | constants | the control loop's deadband test |
| `MAX_TAP_STEPS_PER_ACTION` | constant | the tap-step computation + `_calibrate_tap_sign` |
| `TAP_RESPONSE_EPSILON` | constant | `_calibrate_tap_sign` (L557) |
| `DEFAULT_TAP_METADATA_FILL` | dict | passed to `_complete_missing_tap_metadata` (step [3]) |
| `_TAP_PRINT_COLS` | list | `_print_tap_metadata` |
| `_is_missing_tap_value` | function | `_complete_missing_tap_metadata` (L175) |
| `_print_tap_metadata` | function | steps [2] and [5] (before/after edits) |
| `_complete_missing_tap_metadata` | function | step [3] |
| `_apply_user_tap_metadata_override` | function | step [4] |
| `_select_oltc_trafos` | function | step [1] |
| `_validate_tap_metadata` | function | step [6] |
| `_select_controlled_buses` | function | step [7] |
| `_calibrate_tap_sign` | function | step [7] |
| `_empty_series` | function | the loop's non-converged records |

---

## 4. Section A — Constants (L86–112)

```python
OLTC_CONTROL_LOWER:       float = 0.98   # OLTC deadband lower bound (pu)
OLTC_CONTROL_UPPER:       float = 1.02   # OLTC deadband upper bound (pu)
MAX_TAP_STEPS_PER_ACTION: int   = 1
TAP_RESPONSE_EPSILON:     float = 1e-5
DEFAULT_TAP_METADATA_FILL: dict = { "tap_changer_type": "Ratio", "tap_side": "lv", "tap_neutral": 0, "tap_min": -9, "tap_max": 9, "tap_step_percent": 1.5, ... }
```

#### Line 86–89: The OLTC regulation constants

**The key design insight — the OLTC deadband is *tighter* than the violation band.**
`OLTC_CONTROL_LOWER/UPPER = 0.98/1.02` (±2%) sits *inside* the `V_MIN/V_MAX =
0.95/1.05` (±5%) violation band. **Design rationale:** the OLTC is a *preventive*
regulator — it acts to hold voltage near nominal (±2%) so violations (±5%) rarely
occur, rather than waiting for a violation. This is fundamentally different from the
DER schemes, whose Q(V) deadband (0.99–1.01, Doc 1A) is different again. The three
bands (OLTC regulation, Q(V) deadband, violation limits) are distinct and must not
be conflated. **`MAX_TAP_STEPS_PER_ACTION = 1`** models a realistic OLTC that moves
one tap per control interval (not jumping many steps at once). **`TAP_RESPONSE_EPSILON
= 1e-5`** is the minimum voltage change that confirms a tap is *electrically active*
(§10).

#### Line 94–112: Default fill values and the audit column list

`DEFAULT_TAP_METADATA_FILL` supplies conservative benchmark tap parameters (±9 taps,
1.5%/step, LV-side, Ratio type) **only for networks that lack them** — existing
valid values are never overwritten (§6). `_TAP_PRINT_COLS` is the ordered column
list for the before/after audit logging. **Design rationale:** hardcoding defaults
here (not per-network) keeps the benchmark reproducible; a network with real tap
metadata uses its own, and the audit log (§5) shows exactly what was filled.

---

## 5. Section B — _is_missing_tap_value & _print_tap_metadata (L119–150)

```python
def _is_missing_tap_value(value) -> bool:
    try:
        if pd.isna(value): return True
    except (TypeError, ValueError): pass
    return str(value).strip().lower() in {"", "none", "nan", "<na>"}
```

#### Line 119–132: Robust missing-value detection

Returns `True` if a tap field is genuinely absent *or* a placeholder string.
**Design rationale — why `pd.isna` alone is insufficient (docstring L123–125).**
pandapower stores tap metadata in mixed-type columns where an "absent" value can be
the literal *string* `"None"` or `"nan"` (not a real `NaN`). `pd.isna("None")` is
`False`, so a string-fallback check (`in {"", "none", "nan", "<na>"}`, case-folded)
is needed to catch them. The `try/except` guards `pd.isna` against types it can't
evaluate. Without this, string placeholders would pass as "valid" and break
validation downstream.

#### Line 135–150: Audit logging of tap metadata

`_print_tap_metadata` logs the tap columns for the selected trafos, using only
columns that exist (so it never crashes on a network missing optional columns).
**Design rationale:** called **twice** in `run_scenario_2` — once before and once
after completion/override — so the log shows exactly what changed. Reproducibility
and debuggability: a reviewer can see the OLTC's effective configuration.

---

## 6. Section C — completion & override (L153–219)

```python
def _complete_missing_tap_metadata(net, trafo_idx, network_id, defaults):
    for col, default_value in defaults.items():
        if col not in net.trafo.columns: net.trafo[col] = pd.NA
        for idx in trafo_idx:
            if _is_missing_tap_value(net.trafo.at[idx, col]):
                net.trafo.at[idx, col] = default_value; filled...[idx] = default_value
    if filled: logger.warning("... Completed missing OLTC tap metadata ... %s", ...)
```

#### Line 153–190: Fill missing tap metadata, preserving valid values

Iterates each default field × each selected trafo, filling **only** where
`_is_missing_tap_value` is `True`. Creates the column (`pd.NA`) if absent entirely.
**Every fill is logged at WARNING** with column/index/value. **Design rationale —
non-destructive completion:** a network with real tap data keeps it; only genuine
gaps are filled, and the warning makes every synthetic value auditable (so a
reviewer knows which parameters came from the network vs the benchmark defaults).

#### Line 193–219: Explicit user override

`_apply_user_tap_metadata_override` writes only the fields present in the user's
`override` dict, overwriting **both** network and default-completion values. A
`None`/empty dict is a no-op. **Design rationale — the precedence chain:** network
value → default completion → user override, each layer able to supersede the last,
with the user override as the final word. Logged at WARNING so an override is never
silent.

---

## 7. Section D — _select_oltc_trafos (L222–295)

```python
    candidates = net.trafo[net.trafo["in_service"] == True].copy()
    # Tier 1: HV/MV
    hv_mv = candidates[candidates["vn_hv_kv"] >= 66]
    if not hv_mv.empty: return hv_mv.index
    # Tier 2: slack-connected
    slack_conn = candidates[candidates["hv_bus"].isin(set(net.ext_grid["bus"].values))]
    if not slack_conn.empty: return slack_conn.index
    # Tier 3: highest HV voltage
    fallback = candidates[candidates["vn_hv_kv"] == candidates["vn_hv_kv"].max()]
    logger.warning("... Tier 3 fallback ..."); return fallback.index
```

#### Line 222–295: Three-tier topology-based transformer selection

Selects the OLTC transformer group by **topology alone** — tap metadata is *not*
required yet (that is completed afterwards). Three tiers, tried in order:

- **Tier 1 — HV/MV (`vn_hv_kv ≥ 66`).** Catches SimBench MV and CIGRE MV (both 110
  kV HV side). The normal case.
- **Tier 2 — slack-connected.** For LV-only networks (Kerber, Dickert, CIGRE LV)
  where the head transformer's HV side is 10–20 kV (below 66), the OLTC is the trafo
  whose HV bus *is* the ext_grid slack bus.
- **Tier 3 — highest-HV (last resort).** Any remaining topology; warns to verify.

**Design rationale — why selection precedes metadata completion (docstring
L226–230).** Separating *which* transformer (topology) from *does it have tap data*
(metadata) lets the completion step fill metadata for networks that have a valid
transformer but absent `tap_changer_type` etc. If selection required complete
metadata, those networks could not be benchmarked at all. **Where used:**
`run_scenario_2` step [1]. **Worked example:** `1-MV-rural--2-sw` → Tier 1 selects
its 110/20 kV HV/MV transformer(s).

---

## 8. Section E — _validate_tap_metadata (L298–408)

```python
    for col in required_cols: if ...isna().any(): raise ValueError(...)
    if len(net.trafo.loc[trafo_idx, "tap_neutral"].unique()) > 1: raise ...   # shared neutral
    if len(net.trafo.loc[trafo_idx, "tap_side"].unique()) > 1: raise ...      # shared side
    tap_min_gang = int(...tap_min.max()); tap_max_gang = int(...tap_max.min())
    if tap_min_gang > tap_max_gang: raise ...                                  # feasible range
    ... tap_neutral inside each range; tap_step_percent numeric/nonzero/shared; type=="ratio"; no dependency table ...
```

#### Line 298–408: Ganging feasibility validation

Confirms the selected group can be driven as **one synchronised OLTC**. The checks
enforce the ganging requirements (Doc 6B overview):

- **Required fields present & non-NaN** — a redundant guard after completion, kept
  explicit.
- **Shared `tap_neutral` and `tap_side`** — ganging needs one reference position and
  one direction convention across all trafos.
- **Feasible ganged range** — `tap_min_gang = max(tap_min)`, `tap_max_gang =
  min(tap_max)` (the *intersection* of the trafos' ranges); if `min > max` the group
  has no common movable range → error.
- **`tap_neutral` inside each trafo's own range**, **`tap_step_percent` numeric /
  nonzero / shared**, **`tap_changer_type == "Ratio"`** (amplitude-only, no phase
  shift), **no `tap_dependency_table`** (no LTC lookup table).

**Design rationale — why the ganged range is the intersection.** A tap command sent
to the group must be valid for *every* trafo, so the usable range is bounded by the
most restrictive trafo on each side. Ganging two transformers with ranges `[-9,+9]`
and `[-7,+9]` gives `[max(-9,-7), min(+9,+9)] = [-7,+9]`. **Where used:**
`run_scenario_2` step [6].

---

## 9. Section F — _select_controlled_buses (L411–475)

```python
    lv_buses = sorted(int(b) for b in net.trafo.loc[trafo_idx, "lv_bus"].unique())
    vn_levels = net.bus.loc[lv_buses, "vn_kv"].round(6).unique()
    if len(vn_levels) != 1: raise ValueError("... different voltage levels ...")
    if len(hv_buses) > 1: logger.warning("... multiple HV busbar sections ... split-busbar ...")
    return lv_buses
```

#### Line 411–475: Select the regulated secondary buses

Returns the sorted LV (secondary) bus indices whose **mean voltage** the OLTC
regulates. **Design rationale — mean across split busbars.** A substation may have
multiple secondary busbar sections; the OLTC regulates their *mean* voltage as one
signal (the group is synchronised). Two guards: all secondary buses must be at **one
voltage level** (else they can't form a single measurement signal → error); multiple
**HV** busbar sections are *allowed* (split-busbar substations like
`1-MV-rural--2-sw`) and only warn. **Where used:** `run_scenario_2` step [7]. **Worked
example:** `--2-sw` split-busbar → two HV sections (warn), secondary buses at one MV
level (ok) → mean voltage across them is the control signal.

---

## 10. Section G — _calibrate_tap_sign (L478–586)

The most substantive helper: it determines the tap-sign **empirically** rather than
assuming it.

```python
    probe_net = copy.deepcopy(net)
    if tap_neutral + 1 <= tap_max_gang: probe_pos, direction = tap_neutral + 1, +1
    elif tap_neutral - 1 >= tap_min_gang: probe_pos, direction = tap_neutral - 1, -1
    else: raise ValueError("... tap range too narrow to probe ...")
    probe_net.trafo.loc[trafo_idx, "tap_pos"] = tap_neutral; pp.runpp(probe_net, voltage_depend_loads=False)
    v_neutral = float(probe_net.res_bus.loc[ctrl_buses, "vm_pu"].mean())
    probe_net.trafo.loc[trafo_idx, "tap_pos"] = probe_pos; pp.runpp(probe_net, ...)
    v_probe = float(probe_net.res_bus.loc[ctrl_buses, "vm_pu"].mean())
    delta_v = v_probe - v_neutral
    if abs(delta_v) < TAP_RESPONSE_EPSILON: raise RuntimeError("... negligible voltage response ...")
    if direction == +1: sign = +1 if delta_v < 0.0 else -1
    else:               sign = -1 if delta_v < 0.0 else +1
    return sign
```

#### Line 478–586: Determine the tap-sign by network probing

**The problem.** Whether *raising* the tap position raises or lowers the controlled
(secondary) voltage depends on `tap_side` (hv vs lv) and the winding convention — it
is **not** universally `+`. A controller that assumes the wrong sign would drive
voltage the wrong way, worsening every violation. So the sign is *measured*.

**The method (probing).** On a **deep copy** of the network (so the working net's
results are never touched): solve at `tap_neutral`, record the mean controlled-bus
voltage `v_neutral`; move one tap in a feasible direction (`+1` if room above
neutral, else `−1`), solve again, record `v_probe`; the response is `delta_v = v_probe
− v_neutral`. The returned **sign** is defined so that `new_tap = current + sign ·
MAX_TAP_STEPS_PER_ACTION` always moves the mean controlled voltage **down** (the
action taken on overvoltage):
- probed `+1` and voltage went down (`delta_v < 0`) → sign `+1`;
- probed `+1` and voltage went up → sign `−1`; and mirror for a `−1` probe.

**Two guards, each with a clear failure:**
- **Range too narrow** (neutral is the only position) → `ValueError` (cannot probe).
- **Negligible response** (`|delta_v| < TAP_RESPONSE_EPSILON = 1e-5`) → `RuntimeError`
  *with the full tap metadata printed*. **Design rationale (L553–556):** metadata can
  pass validation yet produce no voltage change — a `tap_side` mismatch or a disabled
  regulator. Failing explicitly (with the metadata) turns a silent no-op controller
  into a diagnosable error.

**Design rationale — deep copy, two power flows, once per run.** Calibration costs
two extra `runpp` calls at setup, but it is done *once* (not per timestep) and makes
the controller topology-agnostic — correct on any network regardless of its winding
convention. Using `copy.deepcopy` guarantees the probe leaves no stale results on the
working net. **Where used:** `run_scenario_2` step [7]. **Worked example:** at neutral
`v_ctrl = 1.030`; at neutral+1 `v_ctrl = 1.015` → `delta_v = −0.015 < 0`, direction
`+1` → **sign `+1`**: to correct an overvoltage, move the tap *up* by one step.

---

## 11. Section H — _empty_series (L589–590)

```python
def _empty_series() -> pd.Series:
    return pd.Series(dtype=float)
```

#### Line 589–590: Typed empty Series for non-converged records

The scenario-2-local twin of `scenario_result._empty_series` (6A-ii): a typed empty
Series for the control loop's non-converged `TimestepRecord`s. **Design rationale:**
kept local so the runner does not depend on the result module's private helper.
**Where used:** `run_scenario_2`'s non-converged branch (Doc 6B-ii).

---

## 12. Coverage checklist

Every named entity in the 6B-i range (L86–590), each confirmed documented.

| Entity | Lines | Kind | Documented | Where-used stated |
|--------|-------|------|:----------:|:-----------------:|
| `OLTC_CONTROL_LOWER/UPPER` | 86–87 | constants | ✅ | ✅ |
| `MAX_TAP_STEPS_PER_ACTION` | 88 | constant | ✅ | ✅ |
| `TAP_RESPONSE_EPSILON` | 89 | constant | ✅ | ✅ |
| `DEFAULT_TAP_METADATA_FILL` | 94–104 | dict | ✅ | ✅ |
| `_TAP_PRINT_COLS` | 107–112 | list | ✅ | ✅ |
| `_is_missing_tap_value` | 119–132 | function | ✅ | ✅ |
| `_print_tap_metadata` | 135–150 | function | ✅ | ✅ |
| `_complete_missing_tap_metadata` | 153–190 | function | ✅ | ✅ |
| `_apply_user_tap_metadata_override` | 193–219 | function | ✅ | ✅ |
| `_select_oltc_trafos` | 222–295 | function | ✅ | ✅ |
| `_validate_tap_metadata` | 298–408 | function | ✅ | ✅ |
| `_select_controlled_buses` | 411–475 | function | ✅ | ✅ |
| `_calibrate_tap_sign` | 478–586 | function | ✅ | ✅ |
| `_empty_series` | 589–590 | function | ✅ | ✅ |

**Also covered (not standalone entities):** the OLTC-deadband-inside-violation-band
insight, the ganging concept and intersection range, the string-placeholder
missing-value detection, the non-destructive completion + override precedence chain,
the topology-before-metadata selection ordering, and the empirical probing rationale
for the tap sign.

**No bugs found in this slice.** No uncalled public entities (all are internal
helpers invoked by `run_scenario_2`).

**Deferred to Document 6B-ii:** `run_scenario_2` (L597–956) — the 7-step setup
orchestration and the per-timestep tap control loop (deadband test, rail limits,
post-tap rollback, the `tap_*` record fields).

---

*End of Document 6B-i. Next: Document 6B-ii — `run_scenario_2`, the OLTC control
loop that uses this infrastructure: the deadband test, one-tap-per-action movement,
ganged-range rail handling, post-tap power-flow rollback, and the `tap_blocked_reason`
state machine recorded in each `TimestepRecord`.*
