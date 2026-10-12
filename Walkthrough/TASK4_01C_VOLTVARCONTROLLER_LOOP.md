# TASK 4 — DOCUMENT 1C
## VoltVarController HIL Loop — `volt_var_controller.py`
### `VoltVarResult` (L411–487) and `VoltVarController` (L897–1229)
### Complete line-by-line walkthrough

> **Source verified:** lines 411–487 and 897–1229 of the current
> `volt_var_controller.py` were read directly before writing. If the file is
> edited, re-verify line numbers.
>
> **Read alongside:** Master Index §5–§8; **Document 1A** (the dry-run Q law this
> loop calls) and **Document 1B** (the serial interface this loop calls on the
> hardware path, and whose three exceptions it catches). The coordinated
> production variant of this loop lives in **Document 3**
> (`run_coordinated_timestep` in `sensitivity_coordinator.py`).
>
> **Flowchart:** `flow_run_timestep.*` (Task 1a).

---

## TABLE OF CONTENTS

1. [Overview — the loop that ties 1A and 1B together](#1-overview)
2. [File logic flow — pseudocode (this slice)](#2-file-logic-flow--pseudocode-this-slice)
3. [Call-site map — where every entity is used](#3-call-site-map--where-every-entity-is-used)
4. [Section A — VoltVarResult (L411–487)](#4-section-a--voltvarresult-l411487)
5. [Section B — VoltVarController (L897–1229)](#5-section-b--voltvarcontroller-l8971229)
   - [Class docstring & __init__ (L901–963)](#class-docstring--__init__-l901963)
   - [_resolve_p_installed (L969–982)](#_resolve_p_installed-l969982)
   - [_get_vm_pu (L984–1004)](#_get_vm_pu-l9841004)
   - [_clamp_to_net_limits (L1006–1042)](#_clamp_to_net_limits-l10061042)
   - [Public properties (L1048–1058)](#public-properties-l10481058)
   - [configure (L1060–1087)](#configure-l10601087)
   - [run_timestep (L1089–1229)](#run_timestep-l10891229)
6. [run_timestep vs run_coordinated_timestep](#6-run_timestep-vs-run_coordinated_timestep)
7. [Failure modes](#7-failure-modes)
8. [Coverage checklist](#8-coverage-checklist)

---

## 1. Overview

Document 1A is the *law*, 1B is the *wire*; this document is the *loop* that runs
each timestep and chooses between them. `VoltVarController` owns one pandapower
network and (optionally) one `ArduinoSerialInterface`, and its `run_timestep()`
executes the canonical HIL sequence:

```
runpp (pre) → read DER voltages → get Q (dry-run 1A OR hardware 1B)
            → clamp Q to network limits → apply → runpp (post) → VoltVarResult
```

`VoltVarResult` is the structured record of one such step: the before/after
violation reports, the Q applied, timing, and retry count.

**A key architectural fact, stated up front.** `run_timestep()` is the
*reference* single-step loop — it is what the module's usage examples show
(L97, L103) and what the tests drive. It is **not** on the production Scenario 4
path. Scenario 4 (both the local 4A and the coordinated 4B variants) calls
`run_coordinated_timestep()` in `sensitivity_coordinator.py` (Document 3)
instead, because it must interpose the `SensitivityCoordinator` and
`DERDynamics` *between* the pre-control power flow and the Q application — a
richer sequence than `run_timestep`'s fixed one. `run_coordinated_timestep`
returns its own `CoordinatorResult`, not a `VoltVarResult`. The two loops are
deliberately parallel and must be kept in sync (the coordinator's step even
carries a comment saying it "mirrors `VoltVarController.run_timestep`"). This is
why several `VoltVarResult` convenience properties have no caller in the current
tree — the coordinator reimplemented equivalents on `CoordinatorResult`. All of
this is documented honestly in §3 and §6, with verdicts.

**How this depends on 1A/1B.** In dry-run mode `run_timestep` calls
`QVCharacteristic.compute_setpoints` (1A) at L1179; on the hardware path it calls
`self._iface.exchange_batched` (1B) at L1185 and catches `ArduinoProtocolError`
/ `SerialTimeoutError` (1B) at L1187/L1195, falling back to Q=0.

---

## 2. File logic flow — pseudocode (this slice)

```
DATACLASS VoltVarResult — the outcome of one timestep:
    holds:  pre-control violation report; post-control report (None if pre-PF failed);
            the Q series applied; exchange time; total time; serial retry count
    readouts:
        converged_pre / converged_post   — did each power flow solve?
        violations_resolved              — were all violations cleared?
                                            (forced False if post-PF diverged)
        voltage_violations_reduced       — did the violating-bus count drop?
                                            (forced False if post-PF diverged)
        summary()                        — one-line log string

CLASS VoltVarController — one per network, drives the loop:

    __init__(net, interface, sgen_indices, p_installed, dry_run):
        remember net + interface; dry-run if interface is None
        resolve which sgens to control  (default: all in-service, ascending)
        resolve each DER's installed capacity (rated sn_mva, else p_mw)
        assert the two lengths agree; warn on any zero-capacity DER
        snapshot each DER's bus index for fast voltage lookup

    _resolve_p_installed():
        prefer the fixed rated apparent power sn_mva; fall back to p_mw where sn is bad
        (p_mw changes every timestep with irradiance — the capacity must be the fixed one)

    _get_vm_pu():
        read vm_pu at each DER's bus from the last power-flow result
        any bus missing from the result → raise loudly (never silently NaN)

    _clamp_to_net_limits(q):
        cap Q by explicit per-DER max_q_mvar / min_q_mvar columns where set
        cap |Q| by spare apparent power  sqrt(sn² − p²)
        return a NEW array (never mutate the caller's)

    configure():
        dry-run → nothing to send
        no DERs → skip; more DERs than one board → defer to the batched path
        else → push INIT + CFG + P to the Arduino (1B)

    run_timestep():                                  [reference single-step loop]
        no DERs?           → run PF, report, return (empty Q)
        reset controlled Q to 0     (so the pre-PF reflects the UNcontrolled network)
        run pre-control PF
        PF failed OR didn't converge? → return early with Q=0, no post report
        read DER-bus voltages
        dry-run?  → compute Q via 1A (compute_setpoints)
        hardware? → exchange over serial via 1B; on ArduinoProtocolError /
                    SerialTimeoutError → Q=0 for this timestep, keep going
        clamp Q to network limits; write to net.sgen
        run post-control PF
        return VoltVarResult(pre, post, Q, timings, retries)
```

---

## 3. Call-site map — where every entity is used

Verified by grep. Uncalled entities carry a verdict per the Master Index §5
convention.

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `VoltVarResult` (class) | dataclass | Returned by `run_timestep` (three paths: L1126, L1165, L1221). Its fields are read in the module's usage examples; the production path uses the coordinator's `CoordinatorResult` instead. |
| `VoltVarResult.converged_pre` | property | **No caller in tree.** Verdict: **intentional** — public convenience accessor. |
| `VoltVarResult.converged_post` | property | Internal only — by `violations_resolved` (L461) and `voltage_violations_reduced` (L474). |
| `VoltVarResult.violations_resolved` | property | **No caller in tree.** Historically consumed by the test suite's `_print_volt_var_summary` (test_suite.py, not in this snapshot). Verdict: **wire-in** to any per-run benchmark summary; intentional until then. |
| `VoltVarResult.voltage_violations_reduced` | property | **No caller in tree.** Same as above. Verdict: **wire-in**; intentional until then. |
| `VoltVarResult.summary` | method | **No caller in tree** (`sensitivity_coordinator.py` L1106's `result.summary()` is `CoordinatorResult.summary`, a different class). Verdict: **intentional** — logging convenience. |
| `VoltVarController` (class) | class | **Instantiated** in `scenario_4_volt_var.py` L238, `sensitivity_coordinator.py` L54/L67, `hosting_capacity.py` L638, `custom_controller.py` L242. Imported by those four modules. Frequently built in `dry_run=True` purely as the authoritative source of DER metadata. |
| `__init__` | method | Via the instantiations above. |
| `_resolve_p_installed` | method | `__init__` (L944). Its logic is mirrored (referenced) in `scenario_4_volt_var.py` L127 and `custom_controller.py` L59. |
| `_get_vm_pu` | method | `run_timestep` (L1173). |
| `_clamp_to_net_limits` | method | `run_timestep` (L1206). Referenced as the canonical clamp by `custom_controller.py` L22/L210 and `hosting_capacity.py` L606 docstrings (which reimplement equivalent clamping). |
| `n_ders` | property | Internal (`configure`, `run_timestep`) **and** external: `hosting_capacity` L644/L671/L707, `sensitivity_coordinator` L257/L943/L989, `custom_controller` L253. |
| `sgen_indices` | property | External, heavily: `scenario_4_volt_var.py` (many — the source of truth for DER ordering), `sensitivity_coordinator.py` L256/L940, `hosting_capacity.py` L415/L673, `custom_controller.py` L250. |
| `p_installed_mw` | property | External: `scenario_4_volt_var.py` L134/L253, `sensitivity_coordinator.py` L258/L941, `hosting_capacity.py` L416, `custom_controller.py` L252. |
| `configure` | method | `hosting_capacity.py` L639, `scenario_4_volt_var.py` L244, `sensitivity_coordinator.py` L55/L68, `custom_controller.py` L248. (Delegates to `ArduinoSerialInterface.configure`, 1B.) |
| `run_timestep` | method | **No production caller.** Referenced in the module usage examples (L97, L103) and by the coordinator (L14: "cannot be used directly"; L951: "mirrors … line 923"). Verdict: **intentional** — the reference/standalone single-step loop and test entry point; production Scenario 4 uses `run_coordinated_timestep` (§6). |

**Two stale references flagged (not edited):**
- The coordinator's comment "mirrors `VoltVarController.run_timestep` line 923"
  (`sensitivity_coordinator.py` L951) — the q-reset it mirrors is now at **L1139**,
  not 923.
- The `_clamp_to_net_limits` "Layer 2" comment (L1026) is mislabelled — see §5.

---

## 4. Section A — VoltVarResult (L411–487)

```python
# Lines 415-441
@dataclass
class VoltVarResult:
    """Structured output from VoltVarController.run_timestep()."""
    report_pre    : ViolationReport
    report_post   : Optional[ViolationReport]
    q_setpoints   : pd.Series
    t_exchange_ms : float = 0.0
    t_total_ms    : float = 0.0
    n_retries     : int   = 0
```

#### Line 415–441: The result dataclass and its fields

**Fields:** `report_pre` (`ViolationReport`, the pre-control power-flow's
violations); `report_post` (`Optional[ViolationReport]` — **`None` when the
pre-PF did not converge**, so there is no meaningful post state); `q_setpoints`
(`pd.Series`, the Q actually applied, indexed by sgen index); `t_exchange_ms`
(serial round-trip time, 0.0 in dry-run); `t_total_ms` (wall-clock for the
whole timestep); `n_retries` (serial retries this step, 0 in dry-run).

**Design rationale — `@dataclass` with defaults on the timing/retry fields.** The
three required fields (`report_pre`, `report_post`, `q_setpoints`) must always be
supplied; the timing/retry fields default so the early-return paths (no DERs,
pre-PF failed) can construct a valid result without inventing timing numbers.
`report_post` being `Optional` is the type-level encoding of "the post state may
not exist" — every consumer must handle `None`, which the convenience properties
below do.

#### Line 443–449: converged_pre / converged_post

```python
    @property
    def converged_pre(self) -> bool:
        return self.report_pre.converged
    @property
    def converged_post(self) -> bool:
        return self.report_post is not None and self.report_post.converged
```

Two thin accessors. `converged_pre` forwards the pre-report's flag.
`converged_post` **guards the `None`**: a missing post-report reads as "not
converged" rather than raising `AttributeError`. **Where used:** `converged_post`
by the two properties below; `converged_pre` has no caller (verdict: intentional
convenience, §3).

#### Line 451–478: violations_resolved / voltage_violations_reduced — the diverged-solver guard

```python
    @property
    def violations_resolved(self) -> bool:
        if not self.converged_post:
            return False
        return self.report_pre.any_violations and not self.report_post.any_violations

    @property
    def voltage_violations_reduced(self) -> bool:
        if not self.converged_post:
            return False
        n_pre  = self.report_pre.n_over_voltage  + self.report_pre.n_under_voltage
        n_post = self.report_post.n_over_voltage + self.report_post.n_under_voltage
        return n_post < n_pre
```

**Design rationale — why the `if not self.converged_post: return False` guard is
load-bearing.** A diverged post-control power flow produces an *empty*
`ViolationReport` (all DataFrames empty → `any_violations` is `False`,
`n_over_voltage`/`n_under_voltage` are 0). Without the guard:

- `violations_resolved` would compute `pre.any_violations and not
  post.any_violations` = `True and not False` = **`True`** — reporting "all
  violations resolved!" when in fact the solver *crashed*. A silent, dangerous
  false positive that would mask solver failures in benchmark logs.
- `voltage_violations_reduced` would compute `n_post (0) < n_pre (>0)` = **`True`**
  — same false "improvement" from a crash.

The guard turns both into an honest `False` when the post-PF did not converge, so
a solver crash can never masquerade as a control success. This is the same
integrity principle as `VoltVarResult.report_post = None` on pre-PF failure and
the coordinator's own `post_pf_ok` gate.

**Worked example.** Pre: 5 over-voltage buses. Post converged with 1: `n_post(1)
< n_pre(5)` → `voltage_violations_reduced = True`, `violations_resolved = False`
(1 ≠ 0). Post *diverged*: both return `False`, not a spurious success.

**Where used:** no caller in tree (verdict: wire-in to a benchmark summary, §3).

#### Line 480–487: summary()

```python
    def summary(self) -> str:
        pre  = self.report_pre.summary()
        post = self.report_post.summary() if self.report_post else "no post-PF"
        return (f"VoltVarResult | pre: {pre} | post: {post} | "
                f"exchange: {self.t_exchange_ms:.1f}ms | retries: {self.n_retries}")
```

One-line log string; substitutes `"no post-PF"` for a `None` post-report.
**Where used:** no caller in tree (the coordinator logs its own
`CoordinatorResult.summary`); verdict: intentional logging convenience (§3).

---

## 5. Section B — VoltVarController (L897–1229)

### Class docstring & __init__ (L901–963)

```python
# Lines 921-963 (abridged)
    def __init__(self, net, interface=None, sgen_indices=None,
                 p_installed_mw=None, dry_run=False):
        self._net   = net
        self._iface = interface
        self._dry   = dry_run or (interface is None)
        if sgen_indices is None:
            self._sgen_idx = (net.sgen.index[net.sgen["in_service"] == True]
                              .sort_values().copy())
        else:
            self._sgen_idx = pd.Index(sgen_indices)
        if p_installed_mw is None:
            self._p_installed = self._resolve_p_installed()
        else:
            self._p_installed = np.asarray(p_installed_mw, dtype=float)
        if len(self._p_installed) != len(self._sgen_idx):
            raise ValueError(...)
        zero_mask = self._p_installed <= 0.0
        if zero_mask.any():
            warnings.warn(...)
        self._sgen_buses = net.sgen.loc[self._sgen_idx, "bus"].values.copy()
```

#### Line 921–931: Store net/interface; derive dry-run

**Parameters:** `net` (pandapower network, **modified in place** by
`run_timestep`); `interface` (`ArduinoSerialInterface` or `None`);
`sgen_indices` (iterable, default all in-service sgens); `p_installed_mw`
(array, default resolved from the net); `dry_run` (bool).

**Design rationale — `self._dry = dry_run or (interface is None)`.** Dry-run is
forced whenever there is no interface, so `dry_run=False` with `interface=None`
cannot produce a controller that tries to talk to a non-existent port. The two
ways to get a dry-run controller (`dry_run=True`, or `interface=None`) collapse
to one internal flag.

#### Line 933–940: Resolve controlled sgen indices

Default: every in-service sgen, **sorted ascending**, `.copy()`-ed. **Design
rationale — sorted + copied.** Sorted ascending fixes a deterministic DER
ordering that every downstream module relies on (the serial protocol is
position-based, 1B; `scenario_4` reindexes profiles to `ctrl.sgen_indices`).
`.copy()` detaches the index from the live DataFrame so later net mutations
cannot silently reorder the controller's DER set. This ordering is the
"source of truth for DER ordering" the scenario code references.

#### Line 942–952: Resolve installed capacities and length-check

If not supplied, `_resolve_p_installed()` (below) derives them; else coerce to a
float array. Then **assert the two lengths agree** — a mismatch between the DER
index and the capacity array would misalign Q to DER and is caught here as a
`ValueError`, before any timestep runs.

#### Line 954–963: Zero-capacity warning; snapshot bus indices

`zero_mask = self._p_installed <= 0.0` — any DER with non-positive capacity gets
`Q_max = 0` (no control authority); the controller **warns** (naming the sgen
indices) rather than failing, because a zero-capacity DER is legal, just inert.
Then `self._sgen_buses = net.sgen.loc[self._sgen_idx, "bus"].values.copy()`
snapshots each DER's bus index — copied so a later net edit cannot move it — for
the fast per-timestep voltage lookup in `_get_vm_pu`.

---

### _resolve_p_installed (L969–982)

```python
    def _resolve_p_installed(self) -> np.ndarray:
        sgens  = self._net.sgen.loc[self._sgen_idx]
        sn     = sgens["sn_mva"].values.astype(float)
        p_mw   = sgens["p_mw"].values.astype(float)
        use_sn = np.isfinite(sn) & (sn > 0.0)
        return np.where(use_sn, sn, p_mw).astype(float)
```

#### Line 969–982: Rated apparent power, with p_mw fallback

**Derivation — why `sn_mva`, not `p_mw`.** `Q_max = Q_RATIO · P_installed`, and
`P_installed` must be the DER's *fixed rated capability*, which is the rated
apparent power `sn_mva`. `p_mw` is the *instantaneous* active output — it varies
every timestep with solar irradiance. Sizing `Q_max` off `p_mw` would shrink the
DER's reactive authority at exactly the low-irradiance moments it might still be
needed, and would make `Q_max` time-varying (wrong: the inverter's reactive
capability is a hardware constant). So use `sn_mva` where it is valid.

**The `use_sn` mask.** `np.isfinite(sn) & (sn > 0.0)` selects rows with a usable
rated value; `np.where(use_sn, sn, p_mw)` falls back to `p_mw` for DERs whose
`sn_mva` is `NaN`, zero, or negative (a dataset gap). **Worked example:** DER-A
with `sn_mva = 0.5` → uses 0.5; a DER with `sn_mva = NaN`, `p_mw = 0.4` → falls
back to 0.4. **Where used:** `__init__` L944.

---

### _get_vm_pu (L984–1004)

```python
    def _get_vm_pu(self) -> np.ndarray:
        res_bus_set = set(self._net.res_bus.index)
        missing = [int(b) for b in self._sgen_buses if b not in res_bus_set]
        if missing:
            raise RuntimeError(f"[VoltVarController] Buses {missing} absent from "
                               f"net.res_bus after runpp(). ...")
        return self._net.res_bus.loc[self._sgen_buses, "vm_pu"].values.astype(float)
```

#### Line 984–1004: Read DER-bus voltages, guarding missing buses

Reads `vm_pu` at each DER's connection bus from the *last power flow's*
`res_bus`. **Design rationale — the explicit missing-bus check.** If a DER's bus
is absent from `res_bus` (e.g. it became out-of-service, or a bus was dropped),
a raw `.loc` would raise a `KeyError` (newer pandas) or silently return `NaN`
(older pandas) — the latter would propagate a `NaN` voltage into the Q
computation. The check builds an **O(1) set** of `res_bus` indices, lists any
missing buses, and raises a `RuntimeError` naming them, converting a silent-NaN
hazard into a loud, diagnosable failure. **Where used:** `run_timestep` L1173.

---

### _clamp_to_net_limits (L1006–1042)

```python
        q_out = q_arr.copy()
        sgens = self._net.sgen
        # Layer 1: explicit min/max_q_mvar columns (where finite)
        if "max_q_mvar" in sgens.columns:
            max_q = sgens.loc[self._sgen_idx, "max_q_mvar"].values.astype(float)
            fin   = np.isfinite(max_q)
            if fin.any():
                q_out = np.where(fin, np.minimum(q_out, max_q), q_out)
        # Layer 2: apparent power limit |Q| <= sqrt(sn_mva^2 - p_mw^2)
        if "min_q_mvar" in sgens.columns:
            min_q = sgens.loc[self._sgen_idx, "min_q_mvar"].values.astype(float)
            fin   = np.isfinite(min_q)
            if fin.any():
                q_out = np.where(fin, np.maximum(q_out, min_q), q_out)
        # Apparent power cap
        sn    = sgens.loc[self._sgen_idx, "sn_mva"].values.astype(float)
        p_now = sgens.loc[self._sgen_idx, "p_mw"].values.astype(float)
        fin   = np.isfinite(sn) & (sn > 0.0) & np.isfinite(p_now)
        if fin.any():
            p_clamp = np.minimum(np.abs(p_now), sn)
            q_lim   = np.sqrt(np.maximum(0.0, sn ** 2 - p_clamp ** 2))
            q_out   = np.where(fin, np.clip(q_out, -q_lim, q_lim), q_out)
        return q_out
```

#### Line 1017–1018: Copy, then bind the sgen table

`q_out = q_arr.copy()` — **design rationale:** the method returns a *new* array
and never mutates the caller's `q_arr`. The caller (`run_timestep`) still holds
the raw Q from 1A/1B; clamping into a copy keeps the pre-clamp values available
and avoids a spooky-action-at-a-distance bug where clamping alters an array the
caller assumed was untouched.

#### Line 1020–1031: Explicit per-DER Q limits (both belong to "Layer 1")

Where the sgen table has `max_q_mvar` / `min_q_mvar` columns, clamp Q into
`[min_q, max_q]` **per DER, where finite** — `np.where(fin, …, q_out)` leaves
DERs whose limit is `NaN`/`Inf` (not set) untouched. `np.minimum` applies the
upper limit, `np.maximum` the lower.

> **⚠ Flagged comment mislabel (not edited).** The comment at L1026 reads
> "Layer 2: apparent power limit …", but the block it heads (L1027–1031) applies
> the **`min_q_mvar`** limit, which the method docstring correctly assigns to
> **Layer 1** ("explicit min/max_q_mvar"). The *actual* Layer 2 (apparent-power
> cap) is the block below (L1033–1040). The comment is misplaced — the `max_q`
> and `min_q` blocks are both Layer 1. Recommend relabelling the L1026 comment to
> "Layer 1 (cont.): min_q_mvar lower limit".

#### Line 1033–1040: The apparent-power cap (the real Layer 2)

**Derivation from first principles.** An inverter's apparent power obeys
`S² = P² + Q²`, and `S` cannot exceed the rating `sn_mva`. So the reactive
headroom left after supplying `P` is:

```
|Q| ≤ √(S_rated² − P²) = √(sn_mva² − p_mw²)
```

This is the safety backstop: `Q_RATIO · P_installed` (1A) can exceed what the
inverter can actually deliver *if* `sn_mva` was set below `p_installed_mw`, so
this cap enforces the real hardware envelope regardless of what the
characteristic asked for. `p_clamp = min(|p_now|, sn)` guards the square root
against a (nonphysical) `p_mw > sn_mva` that would make `sn² − p² < 0`;
`np.maximum(0.0, …)` is a second guard so the `sqrt` never sees a negative.
`np.clip(q_out, −q_lim, q_lim)` applies the symmetric cap.

**Units check.** `√(MVA² − MW²)` → MVAr (all in the same MVA base). ✓

**Worked example** (DER-A, `sn_mva = 0.5`):
- `p_mw = 0.40`: `q_lim = √(0.5² − 0.4²) = √(0.25 − 0.16) = √0.09 = 0.30 MVAr`.
  A Q request of 0.125 MVAr (1A at half-ramp) is within ±0.30 → unclamped.
- `p_mw = 0.48`: `q_lim = √(0.25 − 0.2304) = √0.0196 = 0.14 MVAr`. A request of
  0.125 is still within ±0.14 → unclamped, but a saturated request of 0.125…
  near the ceiling shows how little headroom remains at high output.
- `p_mw = 0.50` (full output): `q_lim = √(0.25 − 0.25) = 0` → Q forced to 0, the
  inverter has no spare apparent power. Physically correct.

**Where used:** `run_timestep` L1206; logic mirrored by `custom_controller` and
`hosting_capacity`.

---

### Public properties (L1048–1058)

```python
    @property
    def n_ders(self) -> int:            return len(self._sgen_idx)
    @property
    def sgen_indices(self) -> pd.Index: return self._sgen_idx
    @property
    def p_installed_mw(self) -> np.ndarray: return self._p_installed.copy()
```

#### Line 1048–1058: The read-only DER-metadata interface

The three-property public face of the controller. **`p_installed_mw` returns a
`.copy()`** so external callers cannot mutate the controller's internal capacity
array — the same defensive-copy discipline as `_clamp_to_net_limits`.
`sgen_indices` returns the live `pd.Index` (immutable by nature, so no copy
needed). **Design rationale — why these are the external interface.** Other
modules build a `VoltVarController` (often `dry_run=True`) purely to get an
authoritative, consistently-ordered DER set and their fixed capacities, then
drive their own loops from `ctrl.sgen_indices` / `ctrl.p_installed_mw` /
`ctrl.n_ders`. **Where used:** heavily external — see §3.

---

### configure (L1060–1087)

```python
    def configure(self) -> None:
        if self._dry:
            return
        if self._iface is None:
            raise SerialConfigError("interface is None but dry_run=False.")
        if self.n_ders == 0:
            warnings.warn("... No DERs in network; skipping Arduino INIT.", ...)
            return
        if self.n_ders > ARDUINO_MAX_DERS:
            warnings.warn("... exceeds ARDUINO_MAX_DERS ... using exchange_batched().", ...)
            return
        self._iface.configure(self.n_ders, self._p_installed)
```

#### Line 1060–1087: Session setup — guarded delegation to the interface

**Dry-run → no-op** (L1065): nothing to send. Then three guards before delegating
to `ArduinoSerialInterface.configure` (1B):

- **`interface is None` but not dry-run** → `SerialConfigError` (a contradictory
  construction that the `_dry` derivation should prevent, but is asserted
  defensively).
- **`n_ders == 0`** → warn and skip: no DERs to initialise.
- **`n_ders > ARDUINO_MAX_DERS`** → warn and skip the *upfront* INIT, because
  `exchange_batched` (1B) will `configure()` per batch on demand — a single
  upfront INIT for more DERs than one board holds would be wrong.

Only in the normal case (`0 < n_ders ≤ 105`) does it send the one-shot
INIT+CFG+P. **Where used:** called via `ctrl.configure()` from scenario_4,
hosting_capacity, coordinator, custom_controller (§3).

---

### run_timestep (L1089–1229)

The reference single-step HIL loop. Five phases: guard → pre-PF → get-Q →
clamp+apply → post-PF, producing a `VoltVarResult` at one of three return points.

#### Line 1119–1131: Timing start and the no-DER fast path

```python
        t0 = time.perf_counter()
        if self.n_ders == 0:
            empty_q = pd.Series([], dtype=float, name="q_mvar")
            pp.runpp(self._net, **{"voltage_depend_loads": False, **(runpp_kwargs or {})})
            report = detect_violations(self._net)
            return VoltVarResult(report_pre=report, report_post=report,
                                 q_setpoints=empty_q, t_total_ms=(...))
```

With no controlled DERs, there is nothing to control: run one power flow, report
once, and return that report as *both* pre and post (they are identical — no Q
was applied). `voltage_depend_loads=False` is enforced (the mandatory flag for
all SimBench `runpp` calls). **Return path 1 of 3.**

#### Line 1133–1139: Reset controlled Q to zero — the uncontrolled-baseline guarantee

```python
        # Reset q_mvar to 0 before the pre-control power flow. ...
        self._net.sgen.loc[self._sgen_idx, "q_mvar"] = 0.0
```

**Design rationale — why this reset is essential (and the most subtle line in
the loop).** Without it, timestep *T*'s pre-control power flow would inherit the
Q setpoints applied in timestep *T−1*, so `report_pre` would reflect a
*partially controlled* network, not the genuinely uncontrolled one.
`report_pre` must be the true uncontrolled baseline so that Scenario 4 can be
benchmarked against the Scenario 1 baseline (where `q_mvar = 0` always). This is
the line the coordinator's `run_coordinated_timestep` mirrors (its stale comment
says "line 923"; the real line is **1139**).

#### Line 1141–1157: Pre-control power flow with convergence capture

```python
        kwargs = {"voltage_depend_loads": False}
        if runpp_kwargs:
            kwargs.update(runpp_kwargs)
            kwargs["voltage_depend_loads"] = False
        pf_pre_ok = True
        try:
            pp.runpp(self._net, **kwargs)
        except Exception as exc:
            pf_pre_ok = False
            warnings.warn(f"[VoltVarController] Pre-control runpp() raised: {exc}", ...)
        report_pre = detect_violations(self._net)
```

`voltage_depend_loads=False` is re-forced even if the caller passed
`runpp_kwargs` (it must never be overridden for SimBench). A raised `runpp` sets
`pf_pre_ok = False` and warns, but `detect_violations` is called regardless —
because it independently inspects `net.res_bus` and must run to produce a report.

#### Line 1159–1170: Convergence gate — early return on pre-PF failure

```python
        if not pf_pre_ok or not report_pre.converged:
            empty_q = pd.Series(np.zeros(self.n_ders), index=self._sgen_idx, name="q_mvar")
            return VoltVarResult(report_pre=report_pre, report_post=None,
                                 q_setpoints=empty_q, t_total_ms=(...))
```

**Design rationale — the double condition `not pf_pre_ok or not
report_pre.converged`.** Catches both a *raised* `runpp` (`pf_pre_ok=False`) and
a *silently non-converged* one (`report_pre.converged=False`, e.g. a stale
`net.converged=True` left from a prior timestep). Either way, proceeding to Q
exchange on an unconverged network would compute Q from garbage voltages, so the
loop returns early with **Q=0 and `report_post=None`** — the `None` is what makes
`VoltVarResult.converged_post`/`violations_resolved` honest (§4). **Return path 2
of 3.**

#### Line 1172–1203: Get Q — the dry-run (1A) / hardware (1B) branch

```python
        vm_pu     = self._get_vm_pu()
        n_retries = 0
        if self._dry:
            vm_s = pd.Series(vm_pu,             index=self._sgen_idx)
            p_s  = pd.Series(self._p_installed, index=self._sgen_idx)
            q_arr         = QVCharacteristic.compute_setpoints(vm_s, p_s).values
            t_exchange_ms = 0.0
        else:
            t_exch = time.perf_counter()
            try:
                q_arr, n_attempts = self._iface.exchange_batched(vm_pu, self._p_installed)
                n_retries = n_attempts - 1
            except ArduinoProtocolError as exc:
                warnings.warn(...); q_arr = np.zeros(self.n_ders); n_retries = self._iface.max_retries
            except SerialTimeoutError as exc:
                warnings.warn(...); q_arr = np.zeros(self.n_ders); n_retries = self._iface.max_retries
            t_exchange_ms = (time.perf_counter() - t_exch) * 1e3
```

Reads DER voltages via `_get_vm_pu`, then branches:
- **Dry-run** → build index-aligned Series and call `QVCharacteristic.
  compute_setpoints` (1A). `t_exchange_ms = 0.0` (no wire). This is the path
  whose Q values are *not* quantised — the reference against which the serial
  path's 4-decimal quantisation is measured (1B §7).
- **Hardware** → `exchange_batched` (1B), timed. **Both 1B exceptions are
  caught** and converted to `q_arr = 0` with `n_retries = max_retries` and a
  warning — the loop degrades to "uncontrolled this timestep" rather than
  crashing. The `ArduinoProtocolError`/`SerialTimeoutError` distinction is
  preserved in the warning text for diagnostics (per 1B's taxonomy).

**Design rationale — why catch here, not in 1B.** 1B *raises* precise errors; the
loop decides the *policy* (fall back to Q=0 and continue). Separating mechanism
(1B) from policy (here) means a different caller could choose to abort instead.

#### Line 1205–1219: Clamp, apply, post-control power flow

```python
        q_arr       = self._clamp_to_net_limits(q_arr)
        q_setpoints = pd.Series(q_arr, index=self._sgen_idx, name="q_mvar")
        self._net.sgen.loc[self._sgen_idx, "q_mvar"] = q_arr
        try:
            pp.runpp(self._net, **kwargs)
        except Exception as exc:
            warnings.warn(f"[VoltVarController] Post-control runpp() raised: {exc}", ...)
        report_post = detect_violations(self._net)
```

Clamp the raw Q to network limits (`_clamp_to_net_limits`), wrap as a labelled
Series, **write it to `net.sgen.q_mvar` by sgen label** (so each Q lands on the
right DER), then run the post-control power flow (same enforced kwargs). A raised
post-PF warns but still calls `detect_violations` for `report_post`.

#### Line 1221–1228: Assemble and return the result

```python
        return VoltVarResult(report_pre=report_pre, report_post=report_post,
                             q_setpoints=q_setpoints, t_exchange_ms=t_exchange_ms,
                             t_total_ms=(time.perf_counter()-t0)*1e3, n_retries=n_retries)
```

**Return path 3 of 3** — the normal path, with both reports, the applied Q, the
exchange and total timings, and the retry count. `t_total_ms` spans from `t0`
(L1119) to here, covering both power flows and the exchange.

---

## 6. run_timestep vs run_coordinated_timestep

Why `run_timestep` has no production caller, stated plainly so it is not
mistaken for dead code:

- **`run_timestep` (here)** is the self-contained reference loop: pre-PF → local
  or serial Q → clamp → apply → post-PF. It is what the module's usage examples
  (L89–103) demonstrate and what unit tests drive. It handles the *uncoordinated*
  case perfectly.
- **`run_coordinated_timestep` (Document 3, `sensitivity_coordinator.py`)** is the
  production Scenario 4 loop. It needs to insert two extra stages the fixed
  `run_timestep` sequence has no room for: the `SensitivityCoordinator`
  (Jacobian-based Q adjustment, between the pre-PF and the apply) and
  `DERDynamics` (PT1 + ramp). It also produces a richer `CoordinatorResult`
  (with `q_initial`, `q_adjusted`, `p_applied`, `curtailment_needed`, etc.).
- **They mirror each other deliberately** — the q-reset (L1139 here), the
  convergence gate, the Q=0 fallback on serial failure. The coordinator's comment
  "mirrors `VoltVarController.run_timestep` line 923" points at exactly this
  parallelism (with a stale line number, flagged §3).

**Consequence for the convenience API.** Because production uses
`CoordinatorResult`, the equivalent `VoltVarResult` properties
(`violations_resolved`, `voltage_violations_reduced`, `summary`, `converged_pre`)
are reimplemented there and are uncalled here — the verdicts in §3 (intentional /
wire-in) reflect that they remain valid standalone/test API.

---

## 7. Failure modes

| Condition | Where | Handling |
|-----------|-------|----------|
| Capacity/index length mismatch | `__init__` L948 | `ValueError` at construction |
| Zero-capacity DER | `__init__` L954 | Warn, continue (Q_max=0) |
| `sn_mva` NaN/≤0 | `_resolve_p_installed` L981 | Fall back to `p_mw` |
| DER bus missing from `res_bus` | `_get_vm_pu` L999 | `RuntimeError` (no silent NaN) |
| `p_mw > sn_mva` (nonphysical) | `_clamp_to_net_limits` L1038–1039 | `p_clamp` + `max(0,…)` guard the sqrt |
| Pre-PF raises | `run_timestep` L1150 | `pf_pre_ok=False`, warn |
| Pre-PF non-converged (incl. stale flag) | `run_timestep` L1159 | Early return, Q=0, `report_post=None` |
| Arduino protocol error / timeout | `run_timestep` L1187/L1195 | Q=0 this step, warn, continue |
| Post-PF raises | `run_timestep` L1213 | Warn, still build `report_post` |
| Diverged post-PF read as "resolved" | `VoltVarResult` L461/L474 | Guard forces `False` |

---

## 8. Coverage checklist

Every named entity in L411–487 and L897–1229, each confirmed documented.

| Entity | Lines | Kind | Documented | Where-used stated |
|--------|-------|------|:----------:|:-----------------:|
| `VoltVarResult` | 415–487 | dataclass | ✅ | ✅ |
| `VoltVarResult.report_pre/report_post/q_setpoints/t_exchange_ms/t_total_ms/n_retries` | 436–441 | fields | ✅ | ✅ |
| `VoltVarResult.converged_pre` | 443–445 | property | ✅ | ✅ (no caller — intentional) |
| `VoltVarResult.converged_post` | 447–449 | property | ✅ | ✅ (internal) |
| `VoltVarResult.violations_resolved` | 451–463 | property | ✅ | ✅ (no caller — wire-in) |
| `VoltVarResult.voltage_violations_reduced` | 465–478 | property | ✅ | ✅ (no caller — wire-in) |
| `VoltVarResult.summary` | 480–487 | method | ✅ | ✅ (no caller — intentional) |
| `VoltVarController` | 897–1229 | class | ✅ | ✅ |
| `__init__` | 921–963 | method | ✅ | ✅ |
| `_resolve_p_installed` | 969–982 | method | ✅ | ✅ |
| `_get_vm_pu` | 984–1004 | method | ✅ | ✅ |
| `_clamp_to_net_limits` | 1006–1042 | method | ✅ (mislabelled comment flagged) | ✅ |
| `n_ders` | 1048–1050 | property | ✅ | ✅ |
| `sgen_indices` | 1052–1054 | property | ✅ | ✅ |
| `p_installed_mw` | 1056–1058 | property | ✅ | ✅ |
| `configure` | 1060–1087 | method | ✅ | ✅ |
| `run_timestep` | 1089–1229 | method | ✅ | ✅ (no production caller — intentional, §6) |

**Also covered (not standalone entities):** the `_dry` derivation (L931), the
sorted-and-copied sgen index (L935–938), the enforced `voltage_depend_loads=False`
(L1124, L1141–1144), the q-reset uncontrolled-baseline guarantee (L1139), the
convergence gate's double condition (L1159), the three `VoltVarResult` return
paths (L1126, L1165, L1221), and the defensive `.copy()` in `p_installed_mw` and
`_clamp_to_net_limits`.

**Uncalled public entities (verdicts in §3):** `run_timestep` (intentional —
reference loop/test entry point), `VoltVarResult.{converged_pre, summary}`
(intentional), `VoltVarResult.{violations_resolved, voltage_violations_reduced}`
(wire-in to a benchmark summary).

---

*End of Document 1C — and of the `volt_var_controller.py` set (1A + 1B + 1C).
Next per the queue: Document 2 — Sensitivity Coordinator (Theory). Recommended
reading detour first: Document 5 (Arduino firmware), the mirror of 1A/1B on the
device side.*
