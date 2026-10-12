# TASK 4 — DOCUMENT 3A
## Sensitivity Coordinator — Walkthrough, Part 1: Building the Sensitivity Matrix
### `sensitivity_coordinator.py` — imports/constants (L84–124), `CoordinatorResult` (L131–221), `SensitivityCoordinator.__init__` + machinery (L249–686)

> **Source verified:** the cited lines of the current `sensitivity_coordinator.py`
> were read directly before writing. If the file is edited, re-verify line
> numbers.
>
> **This is Part 1 of the coordinator walkthrough.** `sensitivity_coordinator.py`
> is 1,106 lines; at the required density a single walkthrough would far exceed
> the sensible size of one document, so — flagged per the Master Index deviation
> rule, exactly as Document 1 split into 1A/1B/1C — Document 3 splits into:
> - **3A (this document): building the sensitivity matrix `X`** — the constants,
>   the result container, `__init__`, and the four machinery methods
>   (`_build_jacobian_blocks`, `_bus_lookup_maps`, `_get_vm_pu_ppc`,
>   `_compute_der_sensitivity`). Implements **Document 2 §4–§6**.
> - **3B (next): using `X`** — `coordinate()`'s 11-step algorithm (Doc 2 §7–§12)
>   and `run_coordinated_timestep()`. Implements **Document 2 §7–§12**.
>
> **Read alongside:** **Document 2** (the theory — every derivation referenced
> here is proven there); Master Index §5–§8.

---

## TABLE OF CONTENTS

1. [Overview](#1-overview)
2. [File logic flow — pseudocode (3A slice)](#2-file-logic-flow--pseudocode-3a-slice)
3. [Call-site map — where every entity is used](#3-call-site-map--where-every-entity-is-used)
4. [Section A — Imports & constants (L84–124)](#4-section-a--imports--constants-l84124)
5. [Section B — CoordinatorResult (L131–221)](#5-section-b--coordinatorresult-l131221)
6. [Section C — SensitivityCoordinator.__init__ (L249–278)](#6-section-c--sensitivitycoordinator__init__-l249278)
7. [Section D — _build_jacobian_blocks (L284–354)](#7-section-d--_build_jacobian_blocks-l284354)
8. [Section E — _bus_lookup_maps (L356–389)](#8-section-e--_bus_lookup_maps-l356389)
9. [Section F — _get_vm_pu_ppc (L449–508) + the deprecated string (L394–447)](#9-section-f--_get_vm_pu_ppc-l449508--the-deprecated-string-l394447)
10. [Section G — _compute_der_sensitivity (L513–686)](#10-section-g--_compute_der_sensitivity-l513686)
11. [Coverage checklist](#11-coverage-checklist)

---

## 1. Overview

This half of `sensitivity_coordinator.py` builds the object at the heart of
Document 2: the **dV/dQ sensitivity matrix `X`**. Given a converged
Newton–Raphson power flow, it extracts the stored Jacobian, reduces it by a
Schur complement to a pure voltage-vs-reactive-power sensitivity, and solves for
just the columns that correspond to controllable DERs. Nothing here decides a
control action — that is 3B's `coordinate()`. This half is pure linear algebra
plus the bookkeeping needed to line up three different bus-indexing schemes
(pandapower labels, ppc internal indices, and J_QQ column positions).

Everything in 3A maps onto Document 2:

| 3A code | Document 2 theory |
|---------|-------------------|
| `_build_jacobian_blocks` | §4 (Jacobian block structure) |
| `_bus_lookup_maps`, `_get_vm_pu_ppc` | §6 (bus bookkeeping for the DER-column solve) |
| `_compute_der_sensitivity` | §5 (Schur complement), §6 (sensitivity matrix, targeted solve), §11 (numerical safeguards) |

The three indexing schemes are the source of most of the code's bulk, so it is
worth stating them up front:
- **pandapower bus labels** — the DataFrame index of `net.bus`; in SimBench these
  are *sparse* (large, non-contiguous integers), not `0..n`.
- **ppc bus indices** — PYPOWER's internal contiguous ordering inside
  `net._ppc["bus"]`; this is the ordering the Jacobian is built in.
- **J_QQ column positions** — the position of each PQ bus within the reactive
  sub-block, i.e. its row/column in the reduced system.

`_bus_lookup_maps` and the `pq_pos_lookup` dict exist solely to translate between
these three, so that DER *j* (a pandapower label) ends up mapped to the correct
column of `X`.

---

## 2. File logic flow — pseudocode (3A slice)

```
CONSTANTS:
    PYPOWER bus-type codes (PQ / PV / REF) and the VM (voltage-magnitude) column
    control constants: target voltage, violation noise floor, rcond floor, saturation tol

DATACLASS CoordinatorResult:
    the full record of one coordinated timestep —
      pre/post violation reports; the four Q stages (initial → adjusted → applied);
      the two P stages (target → applied); curtailment + convergence flags; timing; mode
    readouts:
      dq_correction  = q_adjusted − q_initial   (what coordination added)
      dq_dynamics    = q_applied  − q_adjusted   (what the PT1 filter smoothed)
      dp_ramp        = p_applied  − p_target     (what ramp-limiting held back)
      violations_resolved, summary()

CLASS SensitivityCoordinator:

    __init__(net, controller):
        borrow DER metadata from the controller (indices, capacities, count)
        q_max[i] = Q_RATIO × capacity[i]      (read Q_RATIO now, so overrides apply)
        warn on any zero-capacity DER; snapshot each DER's bus label

    _build_jacobian_blocks():                          [Doc 2 §4]
        require a converged NR power flow — else raise (bfsw stores no Jacobian)
        read the stored sparse Jacobian J and the ppc bus-type array
        build PQ and PV+PQ masks; count n_pq / n_pv_pq; record the PQ ppc indices
        slice the four sub-blocks J_PP / J_PQ / J_QP / J_QQ  WITHOUT densifying J

    _bus_lookup_maps():
        build forward (pd-label → ppc-index) and reverse (ppc-index → pd-label) maps
        iterate by actual bus LABEL, because SimBench labels are sparse

    _get_vm_pu_ppc():
        read solved voltage magnitudes straight from the ppc bus matrix VM column
          (NOT from res_bus — some internal ppc buses have no res_bus counterpart)
        mark which PQ rows are valid physical targets (finite AND have a pd bus)

    _compute_der_sensitivity():                        [Doc 2 §5–§6, §11]
        map each DER's bus → its J_QQ column; exclude unmappable DERs (log why)
        sparse-LU factorise J_PP; form the fill-in block J_PP⁻¹·J_PQ
        form the dense Schur complement  J_red = J_QQ − J_QP·(J_PP⁻¹·J_PQ)
        dense-LU factorise J_red; estimate rcond from the U-diagonal for free
        near-singular (rcond < floor)? → give up, return None (safe no-op)
        build a unit-column RHS for the valid DERs; ONE solve serves all columns
        scatter the solved columns into the full X (excluded DERs stay zero columns)
```

---

## 3. Call-site map — where every entity is used

Verified by grep. Uncalled entities carry a verdict (Master Index §5).

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `_BUS_TYPE_COL` (L113) | constant | `_build_jacobian_blocks` L327 |
| `_BUS_PQ` (L114) | constant | `_build_jacobian_blocks` L330–331 |
| `_BUS_PV` (L115) | constant | `_build_jacobian_blocks` L331 |
| `_BUS_REF` (L116) | constant | **No use in tree.** Defined for completeness of the PYPOWER bus-type set; the masks only need PQ/PV. Verdict: **intentional** (documentation constant) — or delete if strict. |
| `_BUS_VM_COL` (L117) | constant | `_get_vm_pu_ppc` L479 |
| `V_TARGET_PU` (L121) | constant | `coordinate()` L841 (Doc 3B) |
| `MIN_VIOLATION_DV` (L122) | constant | `coordinate()` L766–767 (Doc 3B) |
| `RCOND_THRESHOLD` (L123) | constant | `_compute_der_sensitivity` L649 |
| `SATURATION_TOL` (L124) | constant | `coordinate()` L865 (Doc 3B) |
| `CoordinatorResult` | dataclass | Returned by `run_coordinated_timestep` (L971, L994, L1089 — Doc 3B); consumed by `scenario_4_volt_var.py` (builds `TimestepRecord`), `scenario_result.py` L750. |
| `CoordinatorResult.dq_correction` | property | `summary()` L200; `coordination_active` gate in `scenario_4` uses `q_adjusted − q_initial` directly. |
| `CoordinatorResult.dq_dynamics` | property | `summary()` L202. |
| `CoordinatorResult.dp_ramp` | property | `summary()` L203. |
| `CoordinatorResult.violations_resolved` | property | **No caller in tree.** Verdict: **wire-in** to a benchmark summary (mirrors `VoltVarResult.violations_resolved`, Doc 1C); intentional until then. |
| `CoordinatorResult.summary` | method | `run_coordinated_timestep` L1106 (`logger.info`). |
| `SensitivityCoordinator` | class | Instantiated in `scenario_4_volt_var.py` L246. Imported scenario_4 L83. |
| `__init__` | method | Via instantiation. |
| `_build_jacobian_blocks` | method | `coordinate()` L737 (Doc 3B). |
| `_bus_lookup_maps` | method | `_get_vm_pu_ppc` L491; `_compute_der_sensitivity` L552. |
| `_get_vm_pu_ppc` | method | `coordinate()` L740 (Doc 3B). |
| `_compute_der_sensitivity` | method | `coordinate()` L746 (Doc 3B). |

---

## 4. Section A — Imports & constants (L84–124)

#### Line 90–105: The imports that encode design decisions

```python
from scipy.linalg import lu_factor, lu_solve      # dense solver (for J_red)
from scipy.sparse.linalg import splu               # sparse solver (for J_PP)
from violation_detector import ViolationReport, detect_violations
import violation_detector as _vd                   # for dynamic V_MIN/V_MAX
import volt_var_controller as _vvc                 # for call-time Q_RATIO
```

Two imports carry design intent worth calling out:
- **Both a sparse and a dense scipy solver** are imported — `splu` for the
  genuinely-sparse `J_PP`, `lu_factor`/`lu_solve` for the fill-in-dense `J_red`
  (Doc 2 §5). The import list foreshadows the two-solver strategy.
- **`import volt_var_controller as _vvc`** (not `from … import Q_RATIO`) — the
  comment at L102–104 is explicit: `Q_RATIO` is read as `_vvc.Q_RATIO` **at use
  time**, so a runtime override via `set_qv_parameters` (the CLI executor path,
  Doc 1A) sizes the coordinator's `q_max` with the *same* value the Arduino
  receives over `CFG:`. A `from … import Q_RATIO` would bind the value at import
  and silently ignore later overrides — the single-source-of-truth invariant
  would break at the coordinator. Similarly `import violation_detector as _vd`
  enables the dynamic V-band read (Doc 2 §8).

#### Line 113–117: PYPOWER bus-type constants

```python
_BUS_TYPE_COL: int = 1    # column index of BUS_TYPE in ppc["bus"]
_BUS_PQ:       int = 1    # load bus
_BUS_PV:       int = 2    # generator bus
_BUS_REF:      int = 3    # slack / reference bus
_BUS_VM_COL:   int = 7    # VM voltage magnitude in p.u.
```

The magic column/code numbers of the PYPOWER `bus` matrix, named so the slicing
code reads meaningfully. `net._ppc["bus"]` is a raw NumPy array with fixed column
semantics; `[:, 1]` is the bus-type code (1=PQ, 2=PV, 3=REF), `[:, 7]` is the
solved voltage magnitude. **Design rationale:** naming these avoids scattering
bare `1` and `7` literals through the slicing logic, where they would be
inscrutable and error-prone. **`_BUS_REF` is defined but unused** — the masks
only need PQ and PV (the slack bus is neither controlled nor a PQ target); it is
kept for completeness of the bus-type set. **Where used:** `_BUS_TYPE_COL`,
`_BUS_PQ`, `_BUS_PV` in `_build_jacobian_blocks`; `_BUS_VM_COL` in
`_get_vm_pu_ppc`; `_BUS_REF` nowhere (§3).

#### Line 121–124: Control constants

```python
V_TARGET_PU:      float = 1.00
MIN_VIOLATION_DV: float = 1e-3
RCOND_THRESHOLD:  float = 1e-12
SATURATION_TOL:   float = 1e-6
```

The four tunables, each derived and justified in Document 2: `V_TARGET_PU=1.00`
(aim residual buses at nominal, not the band edge — Doc 2 §9); `MIN_VIOLATION_DV
=1e-3` (ignore sub-milli-pu numerical noise — §8); `RCOND_THRESHOLD=1e-12`
(near-collapse cutoff — §11); `SATURATION_TOL=1e-6` (float "at the ceiling"
tolerance, and the yardstick for the 208× quantisation figure — §12, Doc 1B §7).
Only `RCOND_THRESHOLD` is consumed in 3A (L649); the other three are used in
`coordinate()` (Doc 3B).

---

## 5. Section B — CoordinatorResult (L131–221)

```python
# Lines 157-169
    report_pre:         ViolationReport
    report_post:        Optional[ViolationReport]
    q_initial:          np.ndarray   # Item 2 output (Arduino / dry-run) [MVAr]
    q_adjusted:         np.ndarray   # Item 3 output — coordinated target, NOT applied [MVAr]
    q_applied:          np.ndarray   # after PT1 dynamics — written to net.sgen [MVAr]
    p_target:           np.ndarray   # raw profile value (or curtailed) [MW]
    p_applied:          np.ndarray   # after ramp limiting — written to net.sgen [MW]
    curtailment_needed: bool
    post_pf_ok:         bool
    n_retries:          int   = 0
    t_total_ms:         float = 0.0
    mode:               str = "coordinated"
    t_exchange_ms:      float = 0.0
```

#### Line 131–169: The result dataclass — three Q stages, two P stages

**The richer sibling of `VoltVarResult` (Doc 1C).** Where `VoltVarResult` records
one Q, `CoordinatorResult` records the **full Scenario 4 pipeline**, and the field
order tells the story of a timestep:

```
q_initial ──(coordinate)──▶ q_adjusted ──(PT1 dynamics)──▶ q_applied ──▶ net.sgen.q_mvar
p_target  ─────────────────────────────(ramp limiting)──▶ p_applied ──▶ net.sgen.p_mw
```

- **`q_initial`** — Tier 1 local Q(V) (Doc 1A output; Arduino on HIL).
- **`q_adjusted`** — after coordination (Doc 3B `coordinate()`); *not* applied.
- **`q_applied`** — after `DERDynamics` PT1 filtering and the `±q_max` clip; the
  value actually written to the network (Doc 4).
- **`p_target` → `p_applied`** — the active-power profile after ramp limiting.
- **`mode`** — `"local"` (4A) or `"coordinated"` (4B), which `summary()` branches
  on. **`report_post` is `Optional`** (`None` if the post-PF failed), the same
  diverged-solver honesty as Doc 1C.

**Design rationale — why keep all three Q stages, not just the applied one.** The
paper's analysis needs to *decompose* the applied Q into its contributions: how
much came from the local curve, how much from coordination, how much the PT1
smoothing changed. Storing every stage makes `dq_correction` and `dq_dynamics`
(below) exact differences rather than reconstructions.

#### Line 171–186: dq_correction / dq_dynamics / dp_ramp — the stage deltas

```python
    @property
    def dq_correction(self): return self.q_adjusted - self.q_initial   # coordination added
    @property
    def dq_dynamics(self):   return self.q_applied  - self.q_adjusted   # PT1 smoothed
    @property
    def dp_ramp(self):       return self.p_applied  - self.p_target     # ramp held back
```

Three difference properties, each isolating one pipeline stage's effect.
**Physical reading:** `dq_correction` is exactly the coordinator's contribution
(nonzero only when coordination fired — the basis for the `coordination_active`
gate); `dq_dynamics` is the PT1 filter's smoothing gap (≈0 at 15-min resolution
where the filter coefficient α→1, visible at the 1-s HIL loop, per the L179
docstring); `dp_ramp` is the ramp limiter's gap (binding only during steep
ramps). **Units:** MVAr, MVAr, MW. **Where used:** all three by `summary()`
(L200–203); `dq_correction`'s logic is also what `scenario_4`'s
`coordination_active` gate computes inline.

#### Line 188–193: violations_resolved — the diverged-solver guard (again)

```python
    @property
    def violations_resolved(self) -> bool:
        if not self.post_pf_ok or self.report_post is None:
            return False
        return self.report_pre.any_violations and not self.report_post.any_violations
```

Identical integrity pattern to `VoltVarResult.violations_resolved` (Doc 1C §4):
the `post_pf_ok`/`None` guard forces `False` when the post-PF diverged, so a
solver crash cannot read as "all resolved." **Where used:** no caller in tree
(verdict: wire-in to a benchmark summary, §3).

#### Line 195–221: summary() — the mode-branched log line

Builds a one-line log string, branching on `mode`: the `"local"` (4A) form omits
the coordination/dynamics/ramp deltas (they are meaningless without a
coordinator), while the `"coordinated"` (4B) form reports `max|dQ_coord|`,
`max|dQ_dyn|`, `max|dP_ramp|`, and the retry count. **Design rationale:** one
result type serves both 4A and 4B; `mode` selects the right readout rather than
having two dataclasses. **Where used:** `run_coordinated_timestep` L1106.

---

## 6. Section C — SensitivityCoordinator.__init__ (L249–278)

```python
        self._net         = net
        self._ctrl        = volt_var_controller
        self._sgen_idx    = volt_var_controller.sgen_indices       # pd.Index, ascending
        self._n_ders      = volt_var_controller.n_ders
        self._p_installed = volt_var_controller.p_installed_mw     # ndarray (n_ders,)
        self._rank_warning_count = 0
        self._rank_warning_examples = []
        self.curtailment_needed: bool = False
        self._q_max: np.ndarray = _vvc.Q_RATIO * self._p_installed  # (n_ders,) [MVAr]
        zero_mask = self._q_max <= 0.0
        if zero_mask.any():
            logger.warning("q_max=0 for sgen indices %s. ... excluded ...", ...)
        self._sgen_buses = self._net.sgen.loc[self._sgen_idx, "bus"].values.copy()
```

#### Line 249–278: Borrow DER metadata from the controller; derive q_max

**Design rationale — the coordinator does not re-derive the DER set.** It takes
`sgen_indices`, `n_ders`, and `p_installed_mw` *from the `VoltVarController`*
(Doc 1C's public properties). This guarantees the coordinator's DER ordering is
**identical** to the controller's — the same ascending sgen index, the same
capacities. If it independently re-discovered DERs, a mismatch could pair Q with
the wrong DER (the exact hazard Doc 1A's index-equality guard protects against,
here avoided by sharing one source of truth).

**`self._q_max = _vvc.Q_RATIO * self._p_installed`** — computed **at construction
using the call-time module `Q_RATIO`** (the reason for the `import … as _vvc`
alias, §4). This must equal Tier 1's `q_max` so that the coordinator clips to the
same ceiling the Arduino uses. `zero_mask` warns on zero-capacity DERs (they will
be excluded from coordination). `self._sgen_buses` snapshots each DER's bus label
(`.copy()`, so later net edits cannot move it) for the DER→column mapping in
`_compute_der_sensitivity`. `_rank_warning_count`/`_rank_warning_examples`
initialise the throttled rank-deficiency diagnostics (Doc 3B). **Where used:**
instantiated in `scenario_4` L246.

---

## 7. Section D — _build_jacobian_blocks (L284–354)

Implements **Document 2 §4**: retrieve the stored NR Jacobian and slice its four
blocks.

#### Line 313–325: Require a Newton–Raphson power-flow result

```python
        if not hasattr(self._net, "_ppc") or self._net._ppc is None:
            raise RuntimeError("net._ppc not found. Call runpp() before SensitivityCoordinator.")
        internal = self._net._ppc.get("internal", {})
        if "J" not in internal:
            raise RuntimeError("net._ppc['internal']['J'] not found. ... Use algorithm='nr' or 'iwamoto_nr'; do not use 'bfsw'.")
        J = internal["J"]
```

Two guards, each a precise error. **Design rationale (Doc 2 §3, §15):** the whole
method is free *because* pandapower already computed and stored `J` — but only the
Newton–Raphson family stores it. `bfsw` (backward-forward sweep) is Jacobian-free,
so `net._ppc["internal"]["J"]` is absent, and the second guard raises with the
exact remedy ("use `nr`/`iwamoto_nr`"). This is why `run_coordinated_timestep`
forces `algorithm="nr"` (Doc 3B). Failing here is loud and diagnosable rather than
a mysterious `KeyError` deeper in the pipeline.

#### Line 327–336: Bus-type masks and PQ ordering

```python
        bus_types   = self._net._ppc["bus"][:, _BUS_TYPE_COL]
        n_ppc_buses = len(bus_types)
        pq_mask    = (bus_types == _BUS_PQ)
        pv_pq_mask = (bus_types == _BUS_PQ) | (bus_types == _BUS_PV)
        n_pv_pq = int(pv_pq_mask.sum())
        n_pq    = int(pq_mask.sum())
        pq_bus_indices_ppc = np.where(pq_mask)[0]   # sorted ascending
```

Reads the bus-type column and builds the two masks that define the Jacobian's
block dimensions (Doc 2 §4): `pv_pq_mask` (all non-slack buses → the ΔP equations
and Δθ unknowns, count `n_pv_pq`) and `pq_mask` (PQ buses → the ΔQ equations and
Δ|V| unknowns, count `n_pq`). `pq_bus_indices_ppc` records the **sorted ppc
indices of the PQ buses** — this is the canonical row/column ordering every
downstream array (X rows, `vm_pu_ppc`, the DER-column lookup) aligns to.
**Units:** all indices/counts, dimensionless. **Worked example:** a network with
one slack, two PV, and 40 PQ buses → `n_pv_pq = 42`, `n_pq = 40`,
`pq_bus_indices_ppc` is a length-40 sorted array.

#### Line 339–354: Slice the four sub-blocks without densifying J

```python
        J_PP = J[:n_pv_pq,  :n_pv_pq]
        J_PQ = J[:n_pv_pq,  n_pv_pq:]
        J_QP = J[n_pv_pq:,  :n_pv_pq]
        J_QQ = J[n_pv_pq:,  n_pv_pq:]
        return { "J_PP": J_PP, ..., "pq_bus_indices_ppc": pq_bus_indices_ppc }
```

The four blocks of Doc 2 §4, sliced directly from the sparse `J`. **Design
rationale — sparsity preserved.** These are *sparse* slices; `.toarray()` is
never called on the full `J` (which for a large network would be a costly dense
allocation). Densification is deferred to exactly where the Schur complement
forces it (`_compute_der_sensitivity`, §10). The method returns the four blocks
plus all the bookkeeping (`pq_mask`, counts, `pq_bus_indices_ppc`) in one dict,
so the caller computes it once and threads it to the sub-functions. **Where
used:** `coordinate()` L737 (Doc 3B).

---

## 8. Section E — _bus_lookup_maps (L356–389)

```python
        pd2ppc: np.ndarray = self._net._pd2ppc_lookups["bus"]
        ppc2pd: dict = {}
        for pd_bus_idx in self._net.bus.index:
            pd_bus_int = int(pd_bus_idx)
            if pd_bus_int < 0 or pd_bus_int >= len(pd2ppc):
                continue
            ppc_bus = int(pd2ppc[pd_bus_int])
            if ppc_bus < 0:
                continue
            ppc2pd[ppc_bus] = pd_bus_idx
        return pd2ppc, ppc2pd
```

#### Line 356–389: Forward and reverse bus-index maps — the sparse-label subtlety

Returns `pd2ppc` (pandapower label → ppc index, an array pandapower provides) and
builds `ppc2pd` (the reverse, a dict). **Design rationale — why iterate by
*label*, not position (the docstring's warning).** In SimBench, `net.bus.index`
labels are *sparse* — they can be large, non-contiguous integers, much bigger than
`len(net.bus)`. `pd2ppc` is indexed by the **label**, so it must be read as
`pd2ppc[label]`, never `pd2ppc[position]`. Enumerating by compact position would
read the wrong entries. The loop therefore iterates `net.bus.index` (the actual
labels), skips out-of-range labels and out-of-service buses (`ppc_bus < 0`), and
records only valid `ppc → pd` pairs. **Worked example:** a SimBench bus labelled
`1043` maps via `pd2ppc[1043] → ppc 27`; `ppc2pd[27] = 1043`. **Where used:**
`_get_vm_pu_ppc` L491, `_compute_der_sensitivity` L552.

---

## 9. Section F — _get_vm_pu_ppc (L449–508) + the deprecated string (L394–447)

#### Line 394–447: ⚠ Deprecated function preserved as a string literal — dead code

```python
    '''DEPRECIATED FUNCTION FOR TESTING PPC BUS NOT FOUND ERROR
    def _get_vm_pu_ppc(self, pq_mask, pq_bus_indices_ppc):
        ...
        vm_pu_pd = self._net.res_bus["vm_pu"]
        ... (reads voltages via res_bus, raises if a PQ ppc bus has no pd bus) ...
    '''
```

**This entire earlier implementation is inside a triple-quoted string** (L394
opens `'''DEPRECIATED…`, L447 closes `'''`). It is not executed — it is dead code
retained as a comment. It reads voltages through `net.res_bus` and **raises** if a
PQ ppc bus has no pandapower counterpart. That raising behaviour is exactly why it
was replaced: some internal/auxiliary ppc buses legitimately have no `res_bus`
row, so the old version crashed on valid networks. **Flagged, not edited.**
Recommendation: delete the dead string (it bloats the file and a future reader may
mistake it for live code); the reason it was kept — "for testing PPC bus not
found error" — is better served by a unit test than a commented-out function.

#### Line 449–479: The live _get_vm_pu_ppc — read voltages from the ppc matrix

```python
        ppc_bus = self._net._ppc["bus"]
        if ppc_bus is None or len(ppc_bus) == 0:
            raise RuntimeError("net._ppc['bus'] is empty. Run runpp() first.")
        vm_all_ppc = ppc_bus[:, _BUS_VM_COL].astype(float)
        vm_pu_ppc = vm_all_ppc[pq_bus_indices_ppc]
```

**Design rationale — read from `ppc["bus"]`, not `res_bus` (the fix over the
deprecated version).** The docstring is explicit: some internal ppc buses appear
in `net._ppc["bus"]` and in the Jacobian but have no clean `res_bus` counterpart.
The ppc bus matrix already holds the *solved* voltage magnitude in the `VM` column
(`_BUS_VM_COL = 7`), so the method reads it directly and indexes the PQ rows by
`pq_bus_indices_ppc` — guaranteeing perfect alignment with the Jacobian's PQ
ordering (both come from the same ppc array). **Units:** pu. **Where used:**
`coordinate()` L740 (Doc 3B).

#### Line 481–486: ⚠ Commented-out NaN-raise block

```python
        #if np.isnan(vm_all_ppc).any():
        #   bad_rows = np.where(np.isnan(vm_all_ppc))[0].tolist()
        #  raise RuntimeError(f"NaN in net._ppc['bus'] VM column ... Likely runpp() divergence.")
```

**A disabled guard, flagged.** An earlier design raised on *any* NaN in the VM
column. It is commented out because — per the "Update#1" docstring note — internal
/auxiliary ppc rows can legitimately carry NaN VM, so raising would reject valid
networks. The NaN handling moved to `valid_target_mask` (below), which *excludes*
NaN rows as control targets rather than crashing. Flagged so a future reader knows
the omission of the NaN raise is deliberate, not an oversight.

#### Line 490–508: valid_target_mask — which PQ rows are usable control targets

```python
        _pd2ppc, ppc2pd = self._bus_lookup_maps()
        valid_target_mask = np.array(
            [ np.isfinite(vm_pu_ppc[i]) and int(ppc_idx) in ppc2pd
              for i, ppc_idx in enumerate(pq_bus_indices_ppc) ],
            dtype=bool,
        )
        n_invalid = int((~valid_target_mask).sum())
        if n_invalid:
            logger.info("Ignoring %d internal/non-observable PQ PPC rows as voltage targets.", n_invalid)
        return np.array(vm_pu_ppc, dtype=float), ppc2pd, valid_target_mask
```

**`valid_target_mask`** marks each PQ row that is *both* finite (real solved
voltage) *and* has a pandapower bus counterpart (`int(ppc_idx) in ppc2pd`) — i.e.
a row that represents a real, observable bus the coordinator may legitimately
target. **Design rationale:** this replaces the deleted NaN-raise with a *soft*
exclusion — internal ppc rows are quietly dropped from the target set (logged at
info level) rather than crashing the run. Downstream (`coordinate()`, Doc 3B),
`resid_mask` is AND-ed with `valid_target_mask` so the coordinator never tries to
correct a voltage at a non-physical row. **Where used:** returned to `coordinate()`
L740 (Doc 3B).

---

## 10. Section G — _compute_der_sensitivity (L513–686)

The heart of 3A: the full **Schur-complement pipeline** of Document 2 §5–§6, plus
the numerical safeguards of §11. Returns `(X, der_col_positions)` on success,
`(None, None)` on any factorisation failure (a safe no-op — the caller passes
`q_initial` through).

#### Line 552–558: The ppc→J_QQ-column lookup

```python
        pd2ppc, _ppc2pd = self._bus_lookup_maps()
        pq_pos_lookup: dict = {
            int(ppc_idx): pos for pos, ppc_idx in enumerate(pq_bus_indices_ppc)
        }
```

Builds an **O(1)** dict from ppc bus index to its *position* in the J_QQ column
ordering. This is the third indexing scheme (§1): to place DER *j* in the right
column of `X`, we need to know which J_QQ column its bus is. **Design rationale:**
a dict lookup keeps the per-DER mapping loop O(1) rather than an O(n_pq) search per
DER.

#### Line 563–602: Map each DER's bus to its column, excluding the unmappable

```python
        for j, pd_bus in enumerate(self._sgen_buses):
            ...
            if pd_bus_int < 0 or pd_bus_int >= len(pd2ppc):      # out of pd2ppc range
                der_col_positions.append(None); continue
            ppc_bus = int(pd2ppc[pd_bus_int])
            if ppc_bus == -1:                                     # out-of-service
                der_col_positions.append(None); continue
            if ppc_bus < 0 or ppc_bus >= n_ppc_buses:             # out of ppc range
                der_col_positions.append(None); continue
            col = pq_pos_lookup.get(ppc_bus, None)                # not a PQ bus?
            if col is None:
                logger.warning("sgen %s: bus ... is not a PQ bus ... q_initial passes through.", ...)
            der_col_positions.append(col)
```

Four exclusion gates per DER, each logged (Doc 2 §6, "DER exclusion"): out of
`pd2ppc` range, out-of-service (`ppc_bus == -1`), out of ppc range, or **not a PQ
bus in this operating point** (a DER whose bus became PV, e.g. voltage-regulating).
An excluded DER records `None` and its `q_initial` passes through the coordinator
unchanged. **Design rationale:** the sensitivity model can only speak about PQ
buses; forcing a column for a non-PQ DER would be meaningless, so exclusion keeps
`X` honest. **Worked example:** a DER on a PQ bus at ppc index 27 → `pq_pos_lookup
[27] = 12` → column 12; a DER on an out-of-service bus → `None`.

#### Line 612–628: Sparse LU of J_PP, fill-in block, dense Schur complement

```python
        try:
            lu_PP = splu(J_PP.tocsc())
        except Exception as exc:
            logger.warning("splu(J_PP) failed: %s. J_PP may be singular. Returning q_initial unchanged ...")
            return None, None
        J_PP_inv_J_PQ = lu_PP.solve(J_PQ.toarray())          # (n_pv_pq, n_pq), dense
        J_red = J_QQ.toarray() - J_QP.toarray() @ J_PP_inv_J_PQ   # (n_pq, n_pq), dense
```

**This is Document 2 §5 in code.** `splu(J_PP.tocsc())` sparse-LU-factorises the
genuinely-sparse `J_PP` (converted to CSC, the format `splu` wants). A structural
singularity raises → caught → `(None, None)` (safe no-op). Then the fill-in block
`J_PP⁻¹·J_PQ` is formed by solving against the dense `J_PQ.toarray()` (a sparse
solve with a dense RHS yields a dense result), and the **Schur complement**
`J_red = J_QQ − J_QP·(J_PP⁻¹·J_PQ)` is assembled — dense, because the Schur
complement fills in. **Code-to-math mapping:** `J_red` (code) = `J_red` (Doc 2
§5) exactly. **Design rationale — sparse then dense:** `J_PP` stays sparse
through `splu`; densification (`.toarray()`) happens only on `J_PQ`/`J_QP`/`J_QQ`
at the point the maths forces it, matching the two-solver strategy.

#### Line 633–656: Dense LU of J_red and the free rcond check

```python
        try:
            lu_red, piv_red = lu_factor(J_red)
        except Exception as exc:
            logger.warning("lu_factor(J_red) failed ... Returning q_initial unchanged."); return None, None
        U_diag    = np.abs(np.diag(lu_red))
        rcond_est = (U_diag.min() / U_diag.max()) if U_diag.max() > 0.0 else 0.0
        if rcond_est < RCOND_THRESHOLD:
            logger.warning("J_red ill-conditioned (rcond_est=%.2e < %.2e). Network may be near voltage collapse ...")
            return None, None
```

**Document 2 §11 in code.** Dense-LU-factorise `J_red` with `lu_factor`. Then the
**reciprocal condition estimate at zero extra cost**: `lu_factor` packs L and U
into `lu_red`, so `np.diag(lu_red)` is the diagonal of U, and
`rcond_est = min|diag(U)| / max|diag(U)|` approximates the reciprocal condition
number without any extra factorisation. If `rcond_est < RCOND_THRESHOLD (1e-12)`,
the network is near voltage collapse and the inverse would be garbage → `(None,
None)`. **Design rationale:** this catches the case the sparse `J_PP` check
misses — a *near*-singular `J_PP` produces a garbage-but-non-raising fill-in
block, which surfaces here as an ill-conditioned `J_red` (the comment at L606–608
makes this explicit).

#### Line 661–686: Targeted DER-column solve and assembly

```python
        valid_pairs = [(j, col) for j, col in enumerate(der_col_positions) if col is not None]
        n_valid = len(valid_pairs)
        if n_valid == 0:
            logger.warning("No DERs mapped to PQ buses. Coordination skipped entirely."); return None, None
        E_der = np.zeros((n_pq, n_valid), dtype=float)
        for k, (j, col) in enumerate(valid_pairs):
            E_der[col, k] = 1.0
        X_valid = lu_solve((lu_red, piv_red), E_der)         # (n_pq, n_valid)
        X = np.zeros((n_pq, self._n_ders), dtype=float)
        for k, (j, col) in enumerate(valid_pairs):
            X[:, j] = X_valid[:, k]
        return X, der_col_positions
```

**Document 2 §6 (targeted column solve) in code.** Rather than invert `J_red`
fully, build a **unit-column RHS** `E_der` (a 1.0 in each valid DER's J_QQ row) of
shape `(n_pq × n_valid)`, and solve the **single** `J_red` factorisation against
all DER columns at once with `lu_solve` — reusing `(lu_red, piv_red)`. This yields
`X_valid = J_red⁻¹·E_der`, exactly the DER columns of the sensitivity matrix. They
are scattered into the full `X` of shape `(n_pq × n_ders)`, with **excluded DERs
left as zero columns** (so `X @ dq` naturally ignores them). If no DER maps to a
PQ bus, `(None, None)` — nothing to coordinate. **Units:** `X` is dimensionless
pu-volt/pu-power (Doc 2 §6). **Worked example:** 8 DERs, 2 excluded → `n_valid=6`,
`E_der` is `(n_pq × 6)`, one `lu_solve` produces all 6 columns, scattered into an
`(n_pq × 8)` `X` with 2 zero columns. **Design rationale — one factorisation, many
RHS:** the expensive step (factorising `J_red`) is done once; each DER is just
another cheap right-hand side. **Where used:** `coordinate()` L746 (Doc 3B).

---

## 11. Coverage checklist

Every named entity in the 3A range, each confirmed documented.

| Entity | Lines | Kind | Documented | Where-used stated |
|--------|-------|------|:----------:|:-----------------:|
| `_BUS_TYPE_COL` | 113 | constant | ✅ | ✅ |
| `_BUS_PQ` | 114 | constant | ✅ | ✅ |
| `_BUS_PV` | 115 | constant | ✅ | ✅ |
| `_BUS_REF` | 116 | constant | ✅ | ✅ (unused — intentional) |
| `_BUS_VM_COL` | 117 | constant | ✅ | ✅ |
| `V_TARGET_PU` / `MIN_VIOLATION_DV` / `RCOND_THRESHOLD` / `SATURATION_TOL` | 121–124 | constants | ✅ | ✅ (RCOND here; rest in 3B) |
| `CoordinatorResult` | 131–221 | dataclass | ✅ | ✅ |
| `CoordinatorResult` fields (13) | 157–169 | fields | ✅ | ✅ |
| `dq_correction` / `dq_dynamics` / `dp_ramp` | 171–186 | properties | ✅ | ✅ |
| `violations_resolved` | 188–193 | property | ✅ | ✅ (no caller — wire-in) |
| `summary` | 195–221 | method | ✅ | ✅ |
| `SensitivityCoordinator` | 228–871 | class | ✅ (3A: L228–686; 3B: L692–871) | ✅ |
| `__init__` | 249–278 | method | ✅ | ✅ |
| `_build_jacobian_blocks` | 284–354 | method | ✅ | ✅ |
| `_bus_lookup_maps` | 356–389 | method | ✅ | ✅ |
| `_get_vm_pu_ppc` (live) | 449–508 | method | ✅ | ✅ |
| deprecated `_get_vm_pu_ppc` string | 394–447 | dead code | ✅ (flagged for removal) | — |
| `_compute_der_sensitivity` | 513–686 | method | ✅ | ✅ |

**Also covered (not standalone entities):** the two-solver import strategy
(L90–91), the call-time `Q_RATIO` alias rationale (L102–105), the sparse-slice
(no full densify) discipline (L339–342), the commented-out NaN-raise (L481–486),
and the `valid_target_mask` soft-exclusion (L493–499).

**Deferred to Document 3B:** `coordinate()` (L692–871) and
`run_coordinated_timestep()` (L878–1106) — the algorithm that *uses* `X`, plus the
control constants `V_TARGET_PU`/`MIN_VIOLATION_DV`/`SATURATION_TOL` consumed there.

**Flags raised (not edited):** deprecated `_get_vm_pu_ppc` dead string (L394–447);
commented-out NaN-raise (L481–486); `_BUS_REF` defined-but-unused (L116).

---

*End of Document 3A. Next: Document 3B — `coordinate()` (the 11-step algorithm,
Doc 2 §7–§12) and `run_coordinated_timestep()` (the full single-step
orchestration), which consume the `X` built here.*
