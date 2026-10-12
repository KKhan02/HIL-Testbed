# TASK 4 — DOCUMENT 6E
## Scenario Runner — OPF
### `scenario_5_opf.py` — the AC optimal-power-flow "theoretical bound" (646 lines, checkpointing)
### Complete line-by-line walkthrough

> **Source verified:** read against the current `/mnt/project/scenario_5_opf.py`
> (**646 lines**, the project tree). If the file is edited, re-verify line numbers.
>
> **Read alongside:** Master Index §5–§8; **6A-i/6A-ii** (`adapt_profiles`, the
> `TimestepRecord` fields, `ScenarioResult`); **6D** (Volt-Var — the rule-based
> scheme OPF upper-bounds); **Document 1A** (`Q_RATIO` — OPF reuses the same reactive
> envelope). **Registry:** OPF is scenario **6** (`scenario_id="opf"`, "theoretical
> bound", `supports_lv=False`).

---

## TABLE OF CONTENTS

1. [Overview — OPF as the theoretical bound](#1-overview)
2. [File logic flow — pseudocode](#2-file-logic-flow--pseudocode)
3. [Call-site map](#3-call-site-map)
4. [Section A — Constants (L77–80)](#4-section-a--constants-l7780)
5. [Section B — Setup helpers (L83–223)](#5-section-b--setup-helpers-l83223)
6. [Section C — _compute_sn_rated (L226–249)](#6-section-c--_compute_sn_rated-l226249)
7. [Section D — _write_timestep_opf_state (L252–312)](#7-section-d--_write_timestep_opf_state-l252312)
8. [Section E — _print_opf_debug (L315–366)](#8-section-e--_print_opf_debug-l315366)
9. [Section F — run_scenario_5: reindex, setup, resume (L368–490)](#9-section-f--run_scenario_5-reindex-setup-resume-l368490)
10. [Section G — The OPF loop (L492–621)](#10-section-g--the-opf-loop-l492621)
11. [Section H — Aggregate & result (L623–646)](#11-section-h--aggregate--result-l623646)
12. [Findings and flags](#12-findings-and-flags)
13. [Coverage checklist](#13-coverage-checklist)

---

## 1. Overview

Scenario 5 (registry entry 6) is the **AC optimal-power-flow benchmark** — the
*theoretical upper bound* the rule-based schemes are measured against. Where OLTC,
SVC, and Volt-Var each apply a *heuristic* (a tap rule, a droop, a Q(V) curve), OPF
solves a **global optimisation** each timestep: it finds the reactive dispatch (and,
if necessary, active curtailment) that keeps voltage and thermal limits satisfied at
**minimum cost** — which no rule-based scheme can beat. Its VDI, violation count, and
curtailed energy are therefore the *best achievable* numbers, the yardstick for the
others.

**The objective makes it a bound.** The OPF minimises `Σ cp1·P` with:
- **DER cost `cp1 = −1.0`** — a *negative* cost *rewards* DER active power, so the
  optimiser **maximises DER output** (equivalently, **minimises curtailment**).
- **ext_grid cost `cp1 = +0.001`** — a mild penalty on grid exchange.

So the OPF uses reactive power first (free) and curtails active power only as much as
the voltage/thermal constraints force — optimally. That is exactly the Volt-Var
hierarchy's intent (reactive-before-active, Doc 6D), but solved to global optimality
rather than by rules.

**A fair comparison — the same reactive envelope.** The OPF does not get *unlimited*
reactive power; its per-DER Q bound is `q_lim = min(q_vde, q_circle)` (§7) — the
smaller of the VDE-AR-N `Q_RATIO·sn_rated` limit (the same 0.25 the rule-based schemes
use, Doc 1A) and the inverter apparent-power circle `√(sn² − p²)`. So OPF wields the
**identical** reactive capability as Volt-Var; the benchmark isolates *optimal vs
heuristic use of the same hardware*.

**The known limitation.** `runopp()` uses the PYPOWER OPF backend, which is sensitive
to network conditioning. **SimBench MV fails** `runopp()` — the large HV/MV transformer
impedance ratio ill-conditions the admittance matrix — so OPF runs on **CIGRE MV only**
(the module docstring documents this; `supports_lv=False` also excludes LV). Non-convergence
is expected on some networks/timesteps and handled gracefully (§10).

**Stateless — clean resume.** Like SVC (6C) and unlike OLTC/Volt-Var, each timestep's
OPF is independent (no tap/PT1/ramp memory), so the checkpoint resume needs **no state
seeding** — it simply continues from `start_t`.

---

## 2. File logic flow — pseudocode

```
CONSTANTS: ext_grid cost (+0.001), DER cost (−1.0 → maximise DER / minimise curtailment),
           active-power epsilon, disabled-thermal-limit sentinel (1e6)

SETUP HELPERS:
  _clear_poly_cost / _clear_controllers  — strip stale OPF costs / Scenario-1 controllers
  _series_peak_sum                        — max per-timestep Σ|·| of a profile frame
  _prepare_ext_grid_for_opf               — make the slack a controllable source with feeder-scale P/Q bounds
  _apply_network_constraints              — write bus v-limits + line/trafo thermal limits (or disable)
  _setup_opf                              — static OPF columns; DERs start non-controllable; runs pp.diagnostic
  _compute_sn_rated                       — per-DER apparent-power rating (sn_mva → |p_mw| → profile peak)
  _write_timestep_opf_state               — per-step DER flexibility + costs (THE core; see §7)
  _print_opf_debug                        — one-step opf_task() dump for debugging

FUNCTION run_scenario_5(net, profiles, …, publish_fn, enable_checkpointing, live_csv_rewrite_fn):
    create_continuous_bus_index(net, start=0)      ← MANDATORY before adapt_profiles (PYPOWER needs contiguous buses)
    adapt profiles; guard der_p non-empty
    announce start; RESUME (get_resume_records("opf"); start_t); skip-if-complete
    _setup_opf(...); set_user_pf_options(voltage_depend_loads=False); sn_rated = _compute_sn_rated(...)
    FOR t in range(start_t, T):
        _write_timestep_opf_state(t)  → (p_bound, active_der_idx)   # per-step bounds + costs
        try: pp.runopp(init=..., OPF_SOLVER="cyipopt")  ; converged=True
        except OPFNotConverged: converged=False (throttled warnings)
        converged → read vm/loading/res_sgen p&q; else all None
        build TimestepRecord (q/p applied, p_target=p_bound); publish; append
        every 96 steps: live CSV + progress log
    aggregate → ScenarioResult(scenario_id="opf"); log curtailed MWh; return
```

---

## 3. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `run_scenario_5` | function | `benchmark_runner` `SCENARIO_REGISTRY[6]` (Doc 9), `supports_opf_verbose=True`, `supports_lv=False`. |
| `_setup_opf` (+ its four sub-helpers) | helpers | `run_scenario_5` setup. |
| `_compute_sn_rated` | helper | `run_scenario_5` setup. |
| `_write_timestep_opf_state` | helper | The per-timestep loop. |
| `_clear_poly_cost` | helper | `_setup_opf` + `_write_timestep_opf_state` (per step). |
| `_print_opf_debug` | helper | The loop, gated by `debug_opf_task`. |
| `create_continuous_bus_index` | pandapower | L418 (mandatory pre-adapt). |
| `pp.runopp` (cyipopt) | pandapower | L507 — the OPF solve. |
| `adapt_profiles`, `TimestepRecord`, `ScenarioResult.from_records` | infra | 6A. |
| `Q_RATIO` | const (Doc 1A) | `_write_timestep_opf_state` (the VDE Q bound). |
| `publish_fn.*`, `live_csv_rewrite_fn` | hooks | resume/streaming — **now genuinely used** (the old "deliberately ignored" comment is gone). |

---

## 4. Section A — Constants (L77–80)

```python
_EXT_GRID_COST         = 0.001    # mild penalty on grid exchange
_DER_COST              = -1.0     # negative → OPF maximises DER active power (minimises curtailment)
_ACTIVE_POWER_EPS      = 1e-9     # DER availability threshold (below = inactive this step)
_DISABLED_THERMAL_LIMIT = 1e6     # sentinel loading % that never binds
```

#### Line 77–80: The OPF cost and sentinel constants

`_DER_COST = −1.0` is the crux of the whole scenario (§1): a negative marginal cost on
DER active power means the optimiser is *rewarded* for dispatching more DER, so it
minimises curtailment. `_EXT_GRID_COST = +0.001` gently discourages grid exchange
without dominating the objective. `_ACTIVE_POWER_EPS = 1e-9` is the floor below which a
DER has no available power this timestep (and so is excluded from the OPF's controllable
set and cost). `_DISABLED_THERMAL_LIMIT = 1e6` is a loading-% so high it never binds —
used to *disable* a thermal-constraint class when the caller passes `None` (§5). **Design
rationale:** the sign and magnitude of these costs *define* the optimisation problem; the
comment on `_DER_COST` records the intent so the sign is never "fixed" by mistake.

---

## 5. Section B — Setup helpers (L83–223)

#### Line 83–96: _clear_poly_cost & _clear_controllers — clean slate

Strip stale `poly_cost` rows (from a prior OPF run) and any `ConstControl` left by
Scenario 1. **Design rationale:** `create_poly_cost` *appends*, so a stale cost row would
double the objective for that element; and a leftover time-series controller would fight
the OPF's dispatch. Both are cleared before setup (and `poly_cost` again every timestep,
§7). *(The "drop `poly_cost` rows before switching cost types" rule from the project KB.)*

#### Line 99–103: _series_peak_sum — the feeder-scale helper

Returns the maximum over time of the per-timestep sum of absolute values of a profile
frame (0 for empty). **Where used:** sizing the ext_grid bounds (below) to the feeder's
peak load/generation.

#### Line 106–147: _prepare_ext_grid_for_opf — the controllable slack

```python
    net.ext_grid["controllable"] = True; net.ext_grid[...].astype(bool)
    p_margin = max(50.0, 3.0 * (p_load_peak + p_der_peak))
    q_margin = max(50.0, 3.0 * (q_load_peak + q_der_proxy))
    net.ext_grid.loc[:, "min_p_mw"] = -p_margin ; "max_p_mw" = p_margin ; ± q_margin
```

Makes the slack bus a **controllable source** with generous, feeder-scaled P/Q bounds
(3× the peak load+DER, floor 50 MVA). **Two design rationales:** (1) SimBench leaves
`ext_grid.controllable` as `NaN`, but PYPOWER OPF does boolean logic on it — so it must
be a real `bool` column, or the solve errors. (2) The bounds must be *loose enough never
to bind* (the slack should freely balance the feeder) but *finite* (PYPOWER needs bounded
variables); `3× peak` with a 50 MVA floor achieves both. A `q_der_proxy` from `Σ sn_mva`
sizes the reactive margin when DER apparent power is the relevant scale.

#### Line 150–173: _apply_network_constraints — voltage & thermal limits

Writes `min_vm_pu`/`max_vm_pu` to every bus and `max_loading_percent` to lines/trafos —
these are the **OPF constraints** the solver must satisfy. A `None` thermal limit is
written as `_DISABLED_THERMAL_LIMIT = 1e6` (never binds), letting a caller run
voltage-only OPF. **Design rationale:** OPF constraints live in the network tables (not
solver args), so they are set once here and read by `runopp`.

#### Line 176–223: _setup_opf — static preparation + diagnostic

```python
    _clear_poly_cost(net)
    net.sgen["controllable"] = False; net.sgen.loc[der_idx, ...] = 0.0   # DERs start inert
    _prepare_ext_grid_for_opf(net, ap); _apply_network_constraints(net, ...)
    pp.diagnostic(net, warnings_only=False)
```

Prepares the *static* OPF columns: all DERs start **non-controllable with zeroed
bounds** (their real bounds are rebuilt per timestep, §7, because available power changes
with the profile), the ext_grid is made controllable, and the network constraints are
written. **`pp.diagnostic(net, warnings_only=False)`** runs pandapower's structural check
before the loop — the "diagnostic-first" principle (catches an ill-formed OPF setup up
front rather than as a cryptic `runopp` failure). **Design rationale — static vs
per-timestep:** only the *invariant* setup (ext_grid, voltage limits, DER columns exist)
is done here; anything that varies with the profile (DER P/Q bounds, costs) is deferred to
`_write_timestep_opf_state`.

---

## 6. Section C — _compute_sn_rated (L226–249)

```python
    sn_vals = net.sgen.loc[der_idx, "sn_mva"]; p_fb = |p_mw|; p_peak = ap.der_p.max(axis=0)
    sn_rated = sn_vals.where(sn_vals.notna() & (sn_vals > 0), p_fb)
    sn_rated = sn_rated.where(sn_rated.notna() & (sn_rated > 0), p_peak)
    if (sn_rated <= 0).any(): raise ValueError("... cannot infer positive sn_rated ...")
```

#### Line 226–249: Per-DER apparent-power rating, three-tier fallback

Determines each DER's inverter apparent-power rating with a priority chain: **(1)**
`net.sgen.sn_mva` if positive, else **(2)** the absolute initial `p_mw`, else **(3)** the
peak of the available power profile. Raises (naming the offending sgens) if none yields a
positive rating. **Design rationale:** `sn_rated` sets both the VDE Q limit and the
apparent-power circle (§7), so it must be positive and defensible for every DER; the
fallback chain makes the scenario robust across networks with missing `sn_mva`. **Computed
once** (ratings are static), unlike the P/Q *bounds* which vary per step. **Where used:**
`_write_timestep_opf_state`.

---

## 7. Section D — _write_timestep_opf_state (L252–312)

The core per-timestep function: it turns the profile row into the OPF's DER flexibility
and costs. This is where the "fair envelope" and "minimise curtailment" ideas become
concrete.

#### Line 269–281: The reactive envelope — min(VDE, circle)

```python
    profile_row = ap.der_p.iloc[t].reindex(der_idx).fillna(0.0)
    p_bound_ser = profile_row.clip(lower=0.0)                        # available active power
    active_der_idx = p_bound_ser.index[p_bound_ser > _ACTIVE_POWER_EPS]
    q_vde    = Q_RATIO * sn_rated                                    # VDE-AR-N 4110 limit
    q_circle = np.sqrt(np.maximum(0.0, sn_rated**2 - p_bound_ser**2))# inverter capability circle
    q_lim_ser = np.minimum(q_vde, q_circle)                         # the binding one
    q_lim_ser[p_bound_ser <= eps] = 0.0
```

**The reactive limit is the *smaller* of two physical constraints (§1):**
- **`q_vde = Q_RATIO · sn_rated`** — the VDE-AR-N 4110 reactive allowance (the same
  `Q_RATIO = 0.25` the rule-based schemes use, Doc 1A) — so OPF gets no more reactive
  headroom than Volt-Var.
- **`q_circle = √(sn² − p²)`** — the inverter's apparent-power capability circle: at high
  active output there is little apparent-power headroom left for reactive.

Taking `min(q_vde, q_circle)` respects **both** — the OPF cannot demand reactive power a
real inverter could not supply. **Units:** MVAr. **Worked example** (`sn = 1.0`): at
`p_bound = 0.6` → `q_vde = 0.25`, `q_circle = √(1−0.36) = 0.8` → `q_lim = 0.25` (VDE
binds); at `p_bound = 0.98` → `q_circle = √(1−0.96) = 0.2` → `q_lim = 0.2` (the circle
binds — near full output, reactive is squeezed). This is exactly the DER-A/DER-B envelope
Volt-Var uses, so the comparison is apples-to-apples.

#### Line 283–298: Reset-then-activate the DER flexibility

```python
    net.sgen.loc[der_idx, "controllable"] = False; ...= 0.0          # reset ALL profiled DERs
    if len(active_der_idx) > 0:
        net.sgen.loc[active_der_idx, "controllable"] = True
        net.sgen.loc[active_der_idx, "max_p_mw"] = p_bound_ser[active]  # can dispatch up to available
        net.sgen.loc[active_der_idx, "p_mw"]     = p_bound_ser[active]  # start at max
        net.sgen.loc[active_der_idx, "min_q_mvar"] = -q_lim[active]; "max_q_mvar" = +q_lim[active]
```

**Reset-then-activate** (the docstring's explicit rationale): every profiled DER is first
zeroed and made non-controllable, so a DER that is *inactive this timestep* (no available
power) cannot carry stale `p_mw`/`q_mvar` bounds into the OPF. Then only the **active**
DERs (available power > eps) are made controllable with `max_p_mw = p_bound` (the OPF may
dispatch anywhere in `[0, p_bound]` — `min_p_mw` stays 0, so **curtailment is available
but costed against**) and `±q_lim` reactive range. **Design rationale:** the OPF's active
variable is bounded above by *available* power and below by 0 (curtailment); the `−1` cost
makes it choose the top of that range unless a constraint forces curtailment — precisely
"minimum curtailment".

#### Line 300–312: Loads, and rebuild costs for active DERs only

```python
    if not ap.load_p.empty: net.load.loc[ap.load_idx, "p_mw"/"q_mvar"] = ap.load_p/q.iloc[t]
    _clear_poly_cost(net)
    for eg_idx in net.ext_grid.index: pp.create_poly_cost(net, eg_idx, "ext_grid", cp1_eur_per_mw=_EXT_GRID_COST)
    for idx in active_der_idx:        pp.create_poly_cost(net, idx, "sgen", cp1_eur_per_mw=_DER_COST)
```

Writes the timestep's loads, then **rebuilds the cost table from scratch each step**:
`poly_cost` is cleared and re-created with the ext_grid penalty and the `−1` DER reward
**only for active DERs**. **Design rationale (docstring):** if the `−1` curtailment-reward
cost were left on an inactive (zero-profile) DER, the OPF objective would carry a stale
term; rebuilding per step keeps the objective exactly matched to this timestep's active
set. Returns `(p_bound_ser, active_der_idx)` for the record and the loop's diagnostics.

---

## 8. Section E — _print_opf_debug (L315–366)

Prints a one-step diagnostic dump — `pp.opf_task(net)` (the OPF problem statement),
ext_grid OPF fields, and DER bounds — for a single timestep. **Design rationale:** OPF
non-convergence is hard to debug blind; this dump (gated by `debug_opf_task`, and by
default only for `t == 0` via `debug_first_only`) shows the exact problem handed to the
solver. Uses `print` (not `logger`) because it is an interactive debug aid for short runs.
**Where used:** the loop, behind the `debug_opf_task` flag.

---

## 9. Section F — run_scenario_5: reindex, setup, resume (L368–490)

#### Line 413–428: Validate, the mandatory bus reindex, adapt

```python
    if opf_init not in {"flat", "pf"}: raise ValueError(...)
    create_continuous_bus_index(net, start=0)                # ← MANDATORY, before adapt_profiles
    ap = adapt_profiles(net, profiles)
    if ap.der_p.empty: raise ValueError("Scenario 5 requires at least one profiled DER ...")
```

**`create_continuous_bus_index(net, start=0)` is mandatory and must precede
`adapt_profiles` (a project KB invariant).** PYPOWER's OPF backend requires **contiguous,
0-based bus numbering**; SimBench's sparse bus labels would break the internal index maps.
Reindexing renumbers every bus (and all references) to `0..n−1`. It must come *before*
`adapt_profiles` so the adapter's `load_idx`/`der_p.columns` align to the reindexed net.
The empty-`der_p` guard mirrors Scenario 4 (no DERs → nothing to optimise).

#### Line 430–490: Announce, resume, static setup

```python
    if publish_fn is not None: publish_fn.on_scenario_start("opf", "OPF (theoretical bound)", _T)
    resumed_records = publish_fn.get_resume_records("opf") if (publish_fn and enable_checkpointing) else []
    start_t = (resumed_records[-1].t + 1) if resumed_records else 0
    ... _setup_opf(...); pp.set_user_pf_options(net, voltage_depend_loads=False); sn_rated = _compute_sn_rated(...)
    if start_t >= _T and resumed_records: ... return ScenarioResult.from_records("opf", ...)   # skip-if-complete
```

The standard resume pattern (6C §7) — **stateless, so no seeding** (each OPF is
independent). `on_scenario_start` and `get_resume_records` confirm `publish_fn` is now a
first-class citizen here (the old "deliberately ignored" workaround is gone; OPF genuinely
streams and checkpoints). `_setup_opf` does the static preparation (§5), `set_user_pf_options`
carries the SimBench `voltage_depend_loads=False` flag into `runopp`, and `sn_rated` is
computed once. Skip-if-complete returns the aggregated recovered records.

---

## 10. Section G — The OPF loop (L492–621)

```python
    for t in time_steps:
        p_bound_ser, active_der_idx = _write_timestep_opf_state(net, ap, der_idx, sn_rated, t)   # §7
        try:
            pp.runopp(net, verbose=verbose_opf, init=opf_init, OPF_SOLVER="cyipopt"); converged = True
        except pandapower.optimal_powerflow.OPFNotConverged as exc:
            converged = False
            if n_failed_so_far < max_warning_timesteps: logger.warning(... "OPF did not converge" ...)
            elif not suppressed_warning_logged: logger.warning("... suppressed."); suppressed_warning_logged = True
        if converged:
            vm = net.res_bus["vm_pu"]; ll = ...; tl = ...
            p_result = net.res_sgen["p_mw"].reindex(der_idx); q_result = net.res_sgen["q_mvar"]...; p_target = p_bound_ser
            losses_mw_t = ...; grid_import_mw_t = ...
        else: vm=ll=tl=empty; p_result=q_result=p_target=None; losses=grid=None
        rec = TimestepRecord(..., q_applied_mvar=q_result, p_applied_mw=p_result, p_target_mw=p_target,
                             curtailment_needed=bool(ov_buses or ...), converged=converged, ...)
```

#### Line 492–621: Solve the OPF, extract, record

Each timestep: build the flexibility/costs (§7), then **`pp.runopp` with the `cyipopt`
solver** and the chosen `init` (`"flat"` or warm-start `"pf"`). **Non-convergence is
expected and handled** — `OPFNotConverged` is caught, `converged=False` recorded, and the
warnings are **throttled** (`max_warning_timesteps = 3` individual messages, then a single
"suppressed" notice) so a network that fails most timesteps (e.g. SimBench MV) does not
flood the log. On success, the record captures the OPF-optimal dispatch: **`p_applied =
res_sgen.p_mw`** (what the OPF chose, possibly curtailed), **`p_target = p_bound`** (what
was available) — so `from_records` computes `curtailed_energy = Σ(p_target − p_applied)`,
the **theoretical minimum curtailment**. `q_applied = res_sgen.q_mvar` is the optimal
reactive dispatch. `curtailment_needed` flags any residual violation in the OPF solution
(should be rare on a converged voltage-constrained OPF; possible if thermal limits were
disabled). Non-converged steps record all-`None` (the `None`-means-N/A convention, 6A-ii).

**The progress log (L604–621) is correct on resume.** Note `100.0 * t / max(_T_full, 1)`
uses the **full** length `_T_full` with the absolute `t` — so unlike scenarios 2/3/4 (which
divide by the post-resume range length and overshoot 100%), **OPF's percentage is right**.
This is the reference pattern the others should adopt (§12).

---

## 11. Section H — Aggregate & result (L623–646)

```python
    result = ScenarioResult.from_records(scenario_id="opf", ..., records=records, dt_s=ap.dt_s)
    logger.info("... Done. %.1f s | %d/%d converged | %d violation steps | curtailed=%.3f MWh", ...)
    if publish_fn is not None: publish_fn.on_scenario_end(result)
    return result
```

#### Line 623–646: Aggregate and summarise

`from_records` aggregates (with the new coverage-integrity check, 6A-ii §7). The summary
logs the OPF-specific headline: **converged fraction** (how often the solver succeeded —
low on SimBench MV, high on CIGRE MV) and **curtailed MWh** (the theoretical-minimum
curtailment — the bound the rule-based schemes' curtailment is compared against).
`on_scenario_end` signals completion. **Design rationale:** the converged fraction is
itself a result — it quantifies where AC OPF is even *tractable*, which is part of the
paper's framing (rule-based schemes run everywhere; OPF does not).

---

## 12. Findings and flags

**Positive note (contrast with the other runners):**

1. **OPF computes its progress-% correctly (L610).** `100·t/_T_full` (full length,
   absolute `t`) does not overshoot on resume — unlike scenarios 2/3/4 (6B-ii/6C/6D §7),
   which divide by the post-resume range length. OPF is the **reference pattern**; the
   others should be aligned to it.

**Flagged, not edited (cosmetic):**

2. **Over-indented live-CSV block (L596–603).** The `partial = ...` is indented 20 spaces
   under the `if` at 16 — valid, inconsistent style (same as scenarios 2/3). The inline
   comment "correct value, already in scope, no placeholder" (L601) marks a previously-fixed
   `dt_s` placeholder; no action needed.

**Not bugs (documented for clarity):** OPF non-convergence is an expected outcome (SimBench
MV), handled with throttled warnings and honest `converged=False` records, not a failure to
fix; the stateless resume (no seeding) is correct; the per-timestep cost/flexibility rebuild
is necessary (available power varies); `create_continuous_bus_index` before `adapt_profiles`
is mandatory for PYPOWER.

---

## 13. Coverage checklist

Every named entity in `scenario_5_opf.py`, each confirmed documented.

| Entity | Lines | Kind | Documented | Notes |
|--------|-------|------|:----------:|-------|
| `_EXT_GRID_COST` / `_DER_COST` / `_ACTIVE_POWER_EPS` / `_DISABLED_THERMAL_LIMIT` | 77–80 | constants | ✅ | objective-defining |
| `_clear_poly_cost` | 83–88 | helper | ✅ | |
| `_clear_controllers` | 91–96 | helper | ✅ | |
| `_series_peak_sum` | 99–103 | helper | ✅ | |
| `_prepare_ext_grid_for_opf` | 106–147 | helper | ✅ | controllable slack |
| `_apply_network_constraints` | 150–173 | helper | ✅ | v + thermal limits |
| `_setup_opf` | 176–223 | helper | ✅ | static prep + diagnostic |
| `_compute_sn_rated` | 226–249 | helper | ✅ | 3-tier rating |
| `_write_timestep_opf_state` | 252–312 | helper | ✅ | **the core (§7)** |
| `_print_opf_debug` | 315–366 | helper | ✅ | debug dump |
| `run_scenario_5` | 368–646 | function | ✅ | |
| — reindex + adapt + guard | 413–428 | — | ✅ | mandatory continuous bus index |
| — announce + resume + static setup | 430–490 | — | ✅ | stateless, publish_fn now used |
| — OPF loop (runopp + extract + record) | 492–621 | — | ✅ | cyipopt, throttled non-convergence |
| — aggregate + summary | 623–646 | — | ✅ | curtailed MWh = theoretical min |

**Also covered (not standalone entities):** the OPF objective as a theoretical bound, the
`min(q_vde, q_circle)` fair-envelope derivation, the reset-then-activate flexibility, the
per-step cost rebuild, the mandatory continuous-bus reindex, the SimBench-MV limitation,
the stateless resume, and the correct progress-% (contrast with the other runners).

**Flags raised (not edited):** over-indented live-CSV block (L596–603, cosmetic).
**Positive:** OPF's progress-% is the correct reference pattern.

---

*End of Document 6E — completing the scenario-runner series (6A baseline + result infra,
6B OLTC, 6C SVC, 6D Volt-Var, 6E OPF). Next per the queue: Document 12 — the Custom
Controller & Plugin Subsystem (`custom_controller.py`, `plugin_runner.py`,
`network_plugin.py`, `my_network_loader.py`, the example controllers/networks), now that
`custom_controller.py` compiles.*
