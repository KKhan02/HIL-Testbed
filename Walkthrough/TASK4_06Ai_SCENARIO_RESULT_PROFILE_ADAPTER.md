# TASK 4 — DOCUMENT 6A-i
## Scenario Result Infrastructure, Part 1: The Profile Adapter
### `scenario_result.py` — `AdaptedProfiles` (L68–113), `adapt_profiles()` (L116–250), `oversize_inverters()` (L257–374), `slice_profiles()` (L381–524)

> **Source re-verified (project tree):** `scenario_result.py` is now **1,348 lines**;
> the profile-adapter region anchors shifted ≤+2 — `AdaptedProfiles` L70,
> `adapt_profiles` L117, `oversize_inverters` L258, `slice_profiles` L382. The +29
> file growth is all in the `TimestepRecord`/`ScenarioResult` region (Doc 6A-ii), so
> **6A-i's content is unaffected**; internal line numbers below read ~+1–2.
>
> **(prior) Source verified:** the cited lines of the current `scenario_result.py` were
> read directly before writing. If the file is edited, re-verify line numbers.
>
> **6A splits into three parts** (flagged deviation — see below). `scenario_result.py`
> alone is **1,226 lines** of shared infrastructure used by *all* scenario runners,
> far larger than the "medium" the Master Index originally sized 6A at. It splits by
> responsibility:
> - **6A-i (this document): the profile adapter** — turns `profile_builder`'s output
>   dict into the normalised `AdaptedProfiles` every runner consumes, plus two study
>   utilities.
> - **6A-ii: the result types** — `TimestepRecord` (per-timestep) and `ScenarioResult`
>   (aggregate + derived metrics).
> - **6A-iii: `scenario_1_baseline.py`** — the baseline runner (the simplest consumer
>   of this infrastructure).
>
> **Read alongside:** Master Index §5–§8; **Document 7** (`profile_builder.py`, which
> *produces* the dict this adapter *consumes* — physics there, network-adaptation
> here); the scenario runners **6A-iii/6B/6C/6D** (which all call `adapt_profiles`).

---

## TABLE OF CONTENTS

1. [Overview](#1-overview)
2. [File logic flow — pseudocode (adapter region)](#2-file-logic-flow--pseudocode-adapter-region)
3. [Call-site map](#3-call-site-map)
4. [Section A — AdaptedProfiles (L68–113)](#4-section-a--adaptedprofiles-l68113)
5. [Section B — adapt_profiles (L116–250)](#5-section-b--adapt_profiles-l116250)
6. [Section C — oversize_inverters (L257–374)](#6-section-c--oversize_inverters-l257374)
7. [Section D — slice_profiles (L381–524)](#7-section-d--slice_profiles-l381524)
8. [Coverage checklist](#8-coverage-checklist)

---

## 1. Overview

`scenario_result.py` is the **shared output infrastructure** for all five scenario
runners: every runner returns a `ScenarioResult` built from `TimestepRecord`s, so
the comparison/metrics/plotting layer treats all scenarios uniformly without
knowing which runner produced them. This document (6A-i) covers the *input* side of
that infrastructure — the **profile adapter** that normalises `profile_builder`'s
output into the shape every runner expects.

**The adapter's job.** `profile_builder.build_profiles()` (Document 7) returns a
loosely-typed dict (`{"load": …, "pv": …, "wind": …, "times": …}`). Before any
runner's timestep loop, `adapt_profiles(net, profiles)` converts that into an
`AdaptedProfiles` object with consistent field names and network-aligned content:

```
profiles["load"]  →  load_p (MW)  +  load_q (MVAr, derived from net Q/P ratios)
profiles["pv"]    ┐
profiles["wind"]  ┴→ der_p (MW, PV+wind merged, sorted by sgen index)
profiles["times"] →  times  +  dt_s (inferred)
```

Every runner does `ap = adapt_profiles(net, profiles)` once, then iterates
`range(len(ap.times))`. Two study utilities live here too: `oversize_inverters`
(scale inverter apparent power for capacity studies) and `slice_profiles` (run on a
sub-period — a month, week, or day — without touching runner code).

---

## 2. File logic flow — pseudocode (adapter region)

```
DATACLASS AdaptedProfiles:
    a normalised, network-aligned view of the profiles —
      load_p (MW), load_q (MVAr), der_p (MW, PV+wind merged),
      the three index sets (pv/wind/load), the time axis, and dt_s

FUNCTION adapt_profiles(net, profiles):
    require keys load/pv/wind/times (else KeyError naming the gap)
    copy load_p; reindex to the time axis, filling gaps with 0 (not NaN)
    derive load_q: for each load, ratio = net q_mvar / p_mw (0 where p ≤ 0);
                   load_q = load_p × ratio  (preserves the network power factor)
    merge PV + wind into der_p: concat, sort by sgen index, align to times, fill 0
    infer dt_s from the time axis frequency (fallback: difference two timestamps),
        floored at 1 s
    return AdaptedProfiles(...)

FUNCTION oversize_inverters(net, factor):        [study utility]
    require factor ≥ 1; find PV/wind in-service sgens
    sanitise sn_mva (NaN/inf → p_mw, floor 1e-6); multiply by factor  (raises Q_max)

FUNCTION slice_profiles(profiles, period, index):[study utility]
    validate period ∈ {month, week, day} and index in range
    mask the time axis to the requested calendar window
    return a shallow copy with load/pv/wind/times sliced (original untouched)
```

---

## 3. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `AdaptedProfiles` | dataclass | Returned by `adapt_profiles`; type-hinted and consumed by **every runner** — `scenario_1_baseline` L114, `scenario_2_oltc` L635, `scenario_3_svc` L390, `scenario_4_volt_var` L184, `custom_controller` L222. |
| `adapt_profiles` | function | **The universal pre-loop call** — all four scenario runners + `custom_controller` (same line refs). Imported in each runner's header. |
| `oversize_inverters` | function | **No call site in the runners** — a study utility called from CLI/notebooks before `run_benchmark` (its own docstring is the usage). Verdict: **intentional** (public study utility). |
| `slice_profiles` | function | **No call site in the runners** — a study utility for sub-period runs (month/week/day), called from CLI/notebooks. Verdict: **intentional** (public study utility). |

---

## 4. Section A — AdaptedProfiles (L68–113)

```python
@dataclass
class AdaptedProfiles:
    load_p:   pd.DataFrame   # (T × N_loads) MW, ≥ 0
    load_q:   pd.DataFrame   # (T × N_loads) MVAr, from net Q/P ratio
    der_p:    pd.DataFrame   # (T × N_ders) MW, PV+wind, sorted ascending
    pv_idx:   pd.Index       # sgen indices of PV units
    wind_idx: pd.Index       # sgen indices of wind units
    load_idx: pd.Index       # load element indices
    times:    pd.DatetimeIndex
    dt_s:     float          # timestep seconds, for DERDynamics
```

#### Line 68–113: The normalised profile container

Eight fields giving every runner a *uniform, network-aligned* view of the time
series. **Design rationale — why a typed container, not the raw dict.** The
`profile_builder` dict has loose keys and unaligned frames; passing it around would
force every runner to re-derive `load_q`, re-merge PV/wind, and re-infer `dt_s`.
`AdaptedProfiles` does that once, names the results, and fixes their shapes and
column semantics (columns = element indices, rows = time). **Two field details that
matter downstream:** `der_p` columns are **sorted ascending** to match
`VoltVarController.sgen_indices` (Doc 1C) so profile columns align with the
controller's DER ordering; `dt_s` is what constructs `DERDynamics` (Doc 4) — the
1-s-vs-900-s distinction that makes the PT1 visible or invisible.

---

## 5. Section B — adapt_profiles (L116–250)

#### Line 154–171: Validate keys, align load to the time axis

```python
    required = ("load", "pv", "wind", "times")
    for k in required:
        if k not in profiles: raise KeyError(f"... missing key '{k}'. Keys present: {list(profiles.keys())}")
    times = profiles["times"]; load_p = profiles["load"].copy(); pv_df = profiles["pv"].copy(); wind_df = profiles["wind"].copy()
    load_p = load_p.reindex(times).fillna(0.0)
```

Requires the four keys (naming what's missing), copies the frames (so the caller's
dict is never mutated), and **reindexes `load_p` to `times`, filling gaps with 0.0,
not NaN**. **Design rationale (L167–171):** `profile_builder` usually pre-aligns,
but ERA5/custom profiles can have mismatched indices; filling missing rows with 0
means "no load" — filling with NaN would silently corrupt `load_q` (NaN × ratio =
NaN) and then the power flow. Cheap insurance.

#### Line 176–193: Derive load_q from the network Q/P ratio

```python
    load_idx = load_p.columns
    net_p = net.load.loc[load_idx, "p_mw"].values.astype(float)
    net_q = net.load.loc[load_idx, "q_mvar"].values.astype(float)
    ratio = np.divide(net_q, net_p, out=np.zeros_like(net_q, dtype=float), where=net_p > 0.0)
    load_q = load_p.multiply(ratio, axis="columns")
```

**Derivation.** Reactive load is not in the profile (the profile is active power);
it is reconstructed from each load's **network power factor**. For load *i*,
`ratio_i = q_mvar_i / p_mw_i` (from `net.load` at call time), and
`load_q[:, i] = load_p[:, i] · ratio_i`. This scales the reactive load in lockstep
with the active profile, preserving the network's power-factor assumptions across
all T timesteps.

**Design rationale — `np.divide` with `out` and `where`, not `np.where`.** Loads
with `p_mw ≤ 0` must get `ratio = 0` (no division). The idiomatic
`np.where(net_p>0, net_q/net_p, 0)` would still **evaluate `net_q/net_p` on the zero
rows** (NumPy computes both branches before selecting), emitting divide-by-zero
warnings. `np.divide(..., out=zeros, where=net_p>0)` computes the quotient *only*
where `net_p > 0`, leaving the pre-zeroed `out` elsewhere — no warning, no wasted
work. A small but correct idiom. **Units:** `ratio` dimensionless (MVAr/MW);
`load_q` MVAr. **Worked example:** a load with `net_p=1.0, net_q=0.484` → `ratio=0.484`
(cos φ ≈ 0.9); at a timestep where `load_p=0.6`, `load_q=0.6·0.484=0.29 MVAr`.

#### Line 198–213: Merge PV and wind into der_p

```python
    pv_idx = pv_df.columns; wind_idx = wind_df.columns
    if pv_df.empty and wind_df.empty:
        logger.warning("... both pv and wind ... empty ..."); der_p = pd.DataFrame(index=times, dtype=float)
    else:
        der_p = pd.concat([pv_df, wind_df], axis=1).sort_index(axis=1).reindex(times).fillna(0.0)
```

Concatenates the PV and wind frames column-wise, **sorts columns ascending** (sgen
index order), aligns to `times`, and fills gaps with 0. **Design rationale —
`pd.concat` handles disjoint DER sets safely:** a network with only PV or only wind
has one empty frame; `concat` unions the columns, and `fillna(0.0)` supplies 0 for
any DER absent from one side. The ascending sort is the same DER-ordering contract
as `VoltVarController` (Doc 1C) — so `der_p` column *j* is the same DER as
`sgen_indices[j]`. **Worked example:** PV sgens `{2,5}` + wind sgens `{3}` →
`der_p` columns `[2,3,5]` sorted, each a time series of MW.

#### Line 218–250: Infer dt_s and construct the result

```python
    if len(times) >= 2:
        try:
            if times.freq is not None: dt_s = ptf.to_offset(times.freq).nanos / 1e9
            else: dt_s = (times[1] - times[0]).total_seconds()
        except Exception: dt_s = (times[1] - times[0]).total_seconds()
    else:
        dt_s = 900.0; logger.warning("... fewer than 2 entries; dt_s defaulting ...")
    dt_s = max(dt_s, 1.0)
    return AdaptedProfiles(load_p=load_p, load_q=load_q, der_p=der_p, pv_idx=..., dt_s=dt_s)
```

**dt_s inference — three tiers (Doc 4 depends on this).** Preferred: convert the
index's declared frequency to seconds (`to_offset(freq).nanos / 1e9`). Fallback:
difference the first two timestamps (`total_seconds()`) when `freq` is `None`
(ERA5/custom indices often have no freq). Degenerate: `< 2` timesteps → default to
`900.0` (SimBench 15-min) with a warning. **The `max(dt_s, 1.0)` floor** guards
against a zero/negative `dt` from a malformed index that would make
`DERDynamics`'s `α = 1 − e^(−dt/τ)` and `δP_max` nonsensical (Doc 4). **Worked
example:** a 15-min SimBench index → `dt_s = 900`; a 1-s HIL index → `dt_s = 1`,
which is what makes the PT1 dynamics visible (Doc 4 §4).

---

## 6. Section C — oversize_inverters (L257–374)

```python
def oversize_inverters(net, factor: float = 1.1) -> None:
    if factor < 1.0: raise ValueError("... factor must be >= 1.0 ...")
    if net.sgen.empty or "sn_mva" not in net.sgen.columns: logger.warning(...); return
    type_col = net.sgen.get("type", ...).fillna("")
    is_pv = type_col.str.lower().str.contains("pv|solar"); is_wind = type_col.str.lower().str.contains("wind|wp")
    der_mask = (is_pv | is_wind) & (net.sgen["in_service"] == True)
    sn = net.sgen["sn_mva"].replace([inf,-inf], nan).fillna(net.sgen["p_mw"]).clip(lower=1e-6)
    net.sgen["sn_mva"] = sn * factor
```

#### Line 257–374: Scale inverter apparent power for capacity studies

A study utility that multiplies DER `sn_mva` by `factor` (≥ 1), modelling oversized
inverters. Since every Q-capable component derives `Q_max` from `sn_mva`
(`VoltVarController`, `SensitivityCoordinator`, `DERDynamics` all use
`Q_RATIO · sn_mva`), a single pre-run call raises the reactive ceiling everywhere.

**Design rationale.** `factor < 1` raises (undersizing is done via profile scaling,
not here). The **DER mask** (`pv|solar|wind|wp` type strings, in-service) excludes
SimBench's load-as-negative-sgen entries — mirroring the type logic in
`profile_builder` (Doc 7) — so only real inverters are scaled. The **sanitise
step** (`inf → NaN → p_mw`, `clip(1e-6)`) ensures a NaN/inf `sn_mva` doesn't
silently propagate through the multiply. **Where used:** no runner call site —
called from CLI/notebooks before `run_benchmark` (verdict: intentional study
utility, §3).

> **⚠ Flagged stale docstring.** The docstring (L267, L270) computes its examples
> with `Q_RATIO = 0.48` (`Q_max = 0.48 × 1.1 × P_rated`). The in-force value is
> `Q_RATIO = 0.25` (Doc 1A). The *code* is unaffected (it scales `sn_mva`, not
> `Q_RATIO`); only the illustrative arithmetic in the docstring is stale. Same
> family as the Doc 1A/4/5 `Q_RATIO` flags. **Flagged, not edited.**

---

## 7. Section D — slice_profiles (L381–524)

```python
def slice_profiles(profiles: dict, period: str, index: int = 1) -> dict:
    if period not in {"month","week","day"}: raise ValueError(...)
    lo, hi = {"month":(1,12),"week":(1,53),"day":(1,366)}[period]
    if not (lo <= index <= hi): raise ValueError(...)
    times = profiles["times"]
    if   period == "month": mask = times.month == index
    elif period == "week":  mask = times.isocalendar().week.values == index
    else:                   mask = times.day_of_year == index
    if len(times[mask]) == 0: raise ValueError("... empty slice ...")
    out = profiles.copy(); out["times"] = times[mask]
    for key in ("load","pv","wind"):
        if key in profiles and profiles[key] is not None: out[key] = profiles[key].loc[out["times"]].copy()
    ... out["extreme_days"] = find_extreme_days(out) ...
    return out
```

#### Line 381–524: Slice the annual series to a sub-period without touching runners

Returns a *new* profiles dict restricted to one calendar month, ISO week, or
day-of-year, so a benchmark can run on (say) January or the summer-solstice day
without any change inside the runners. Validates `period` and `index` range, builds
a boolean mask on the time axis, raises on an **empty slice** (e.g. week 53 in a
non-leap year — with the data's actual date span in the message), then shallow-copies
the dict and slices `load`/`pv`/`wind`/`times`.

**Design rationale — the positional-not-label subtlety (docstring L429–439).** The
sliced frames keep their `DatetimeIndex` (positions are *not* reset). This is
correct because `adapt_profiles` and the runners iterate by **integer position**
(`range(len(ap.times))`, `iloc[t]`), which aligns with the sliced frame's positional
axis regardless of the calendar labels. Scenario 1's integer-reindex step (6A-iii)
rebuilds a 0-based index anyway. **The original dict is never mutated** (`copy()` +
per-key `.loc[...].copy()`), so the annual profiles remain usable after slicing.
`extreme_days` is recomputed for the slice (falling back to the original on error).
**Where used:** no runner call site — CLI/notebook study utility (verdict:
intentional, §3). **Worked example:** `slice_profiles(profiles, "month", 1)` →
January only (~2,976 fifteen-minute steps); `slice_profiles(profiles, "day", 172)`
→ the summer-solstice day (~96 steps).

---

## 8. Coverage checklist

Every named entity in the 6A-i range (L64–524), each confirmed documented.

| Entity | Lines | Kind | Documented | Where-used stated |
|--------|-------|------|:----------:|:-----------------:|
| `AdaptedProfiles` | 68–113 | dataclass | ✅ | ✅ (all runners) |
| `AdaptedProfiles` fields (8) | 106–113 | fields | ✅ | ✅ |
| `adapt_profiles` | 116–250 | function | ✅ | ✅ (all runners + custom_controller) |
| `oversize_inverters` | 257–374 | function | ✅ (stale `Q_RATIO=0.48` docstring flagged) | ✅ (no runner caller — intentional) |
| `slice_profiles` | 381–524 | function | ✅ | ✅ (no runner caller — intentional) |

**Also covered (not standalone entities):** the fill-0-not-NaN reindex rationale,
the `np.divide(out=, where=)` idiom vs `np.where`, the PV/wind `concat` disjoint-set
handling, the three-tier `dt_s` inference with the `max(.,1.0)` floor, the DER
type-mask in `oversize_inverters`, and the positional-not-label slice subtlety.

**Flags raised (not edited):** `oversize_inverters` docstring `Q_RATIO = 0.48`
(L267, L270) is stale (operative value 0.25; code unaffected).

**Deferred to later parts of 6A:** `TimestepRecord` (L530–790) and `ScenarioResult`
(L791–1226) → **Document 6A-ii**; `scenario_1_baseline.py` → **Document 6A-iii**.

---

*End of Document 6A-i. Next: Document 6A-ii — `TimestepRecord` and `ScenarioResult`
(the per-timestep record and the aggregate result with its derived metrics and
`from_records`/`summary_dict`).*
