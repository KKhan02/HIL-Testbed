# TASK 4 — DOCUMENT 10A
## Output Subsystem, Part 1: The Publisher
### `publisher.py` — JSON/JSONL payloads, live streaming & crash-resume checkpointing (1153 lines)
### Complete line-by-line walkthrough

> **Source verified:** read against the current `/mnt/project/publisher.py` (1153
> lines, the project tree). Compiles. **Both NaN-safety fixes I suggested are
> confirmed applied** across `_dump`, `_jsonl_line`, and `_append_jsonl` (§6).
>
> **Document 10 covers the output subsystem** and splits into:
> - **10A (this document): `publisher.py`** — the machine-readable output layer:
>   JSON payloads for the Streamlit dashboard, the live JSONL stream, and the
>   checkpoint JSONL stream that powers crash-resume.
> - **10B: `plot_results.py`** — the matplotlib report figures.
> - **10C: `network_plotter.py`** — the network topology plots.
>
> **Read alongside:** **6A-ii §4.5** (`to_checkpoint_dict`/`from_checkpoint_dict` —
> the record (de)serialisation this stream carries), **Document 9**
> (`benchmark_runner` — wires the `PublishHandle` into `run_benchmark`), the
> `HIL_RPi_Setup_Log` (the SD-card `mmc_rescan` hang this module's held-open handle
> avoids).

---

## TABLE OF CONTENTS

1. [Overview — the publish layer](#1-overview)
2. [File logic flow — pseudocode](#2-file-logic-flow--pseudocode)
3. [Call-site map](#3-call-site-map)
4. [Section A — Encoder & NaN-safe writers (L238–305)](#4-section-a--encoder--nan-safe-writers-l238305)
5. [Section B — build_topology (L312–453)](#5-section-b--build_topology-l312453)
6. [Section C — build_profiles_payload (L460–584)](#6-section-c--build_profiles_payload-l460584)
7. [Section D — build_scenario_payload (L590–725)](#7-section-d--build_scenario_payload-l590725)
8. [Section E — build_hc_payload & build_live_frame (L732–854)](#8-section-e--build_hc_payload--build_live_frame-l732854)
9. [Section F — PublishHandle (L861–1053)](#9-section-f--publishhandle-l8611053)
10. [Section G — The three-way publish split (L1059–1153)](#10-section-g--the-three-way-publish-split-l10591153)
11. [Findings and flags](#11-findings-and-flags)
12. [Coverage checklist](#12-coverage-checklist)

---

## 1. Overview

`publisher.py` is the **machine-readable output layer** — it turns the network,
profiles, and scenario results into JSON/JSONL files that (a) the Streamlit dashboard
reads and (b) a crashed run reads to resume. It never draws anything (that's 10B/10C);
it *serialises*.

Four ideas define it:

1. **Strict-JSON, NaN-safe writers** (§4) — three writers (`_dump` for pretty/compact
   files, `_append_jsonl`/`_jsonl_line` for streams), all routed through `_nan_to_none`
   + `allow_nan=False` so non-finite floats become `null` and the output is valid JSON
   for any consumer. This is where the fixes from the checkpointing work landed.
2. **Payload builders** (§5–§8) — pure functions turning a net/profiles/`ScenarioResult`
   into dicts: `build_topology`, `build_profiles_payload`, `build_scenario_payload`
   (the full per-timestep record), `build_hc_payload`, and the compact
   `build_live_frame`.
3. **The two-stream `PublishHandle`** (§9) — per scenario, a **live** JSONL (compact
   dashboard frames, truncated fresh each run, gaps tolerated) and a **checkpoint**
   JSONL (full `to_checkpoint_dict` records, never truncated, every timestep, the
   resume source). The checkpoint file is **opened once and held open** — the fix for
   the RPi SD-card `mmc_rescan` hang caused by per-timestep open/close over 100k+ steps.
4. **The three-way write split** (§10) — `publish_topology_and_profiles` (before the
   scenarios), `publish_scenario_result` (per scenario, immediately), and
   `publish_hc_and_comparison` (after all) — replacing the old monolithic `publish_result`
   for crash-resilience and to avoid one huge end-of-run write burst.

**Compact vs lossless** — a recurring distinction: `build_live_frame` is *compact* (only
the fields the animation needs), while the checkpoint stream carries the *lossless*
`to_checkpoint_dict` (every field, so a resumed run is a faithful continuation). They are
different serialisations for different purposes (6A-ii §4.5).

---

## 2. File logic flow — pseudocode

```
WRITERS: _Encoder (numpy/pandas → native); _nan_to_none (non-finite float → None);
         _dump (whole-file JSON, allow_nan=False); _append_jsonl/_jsonl_line (stream)

PAYLOAD BUILDERS (pure net/result → dict):
  build_topology(net)           — buses/lines/trafos/sgens/loads + feeder_dist + voltage_limits
  build_profiles_payload(prof)  — load/pv/wind, 10-min + hourly resample
  build_scenario_payload(result)— every TimestepRecord field as a JSON row + summary_dict
  build_hc_payload(hc_results)  — baseline vs volt_var HC + gain
  build_live_frame(rec, ...)    — COMPACT single-timestep frame for the animation

@dataclass PublishHandle(output_dir, update_every_k, enable_checkpointing):
  on_scenario_start: truncate live/<id>.jsonl; OPEN-and-HOLD checkpoint/<id>.jsonl; load .elapsed
  get_resume_records(id): rebuild TimestepRecords from checkpoint (+ .completed fallback); drop corrupt tail
  cumulative_elapsed_s(): elapsed across all attempts (survives crash)
  on_timestep(rec): checkpoint EVERY step (write+flush); live frame every update_every_k
  on_scenario_end: close checkpoint handle; write .elapsed; append "scenario_complete"

THREE-WAY SPLIT:
  publish_topology_and_profiles (before scenarios) | publish_scenario_result (per scenario, compact)
  | publish_hc_and_comparison (after all)
```

---

## 3. Call-site map

| Entity | Kind | Called / read from |
|--------|------|--------------------|
| `PublishHandle` | dataclass | Constructed by `benchmark_runner`/CLI and wired into `run_benchmark` as `publish_fn`; its methods are called by **every scenario runner** (6A-iii–6E, 12A) and `get_resume_records` by their resume blocks. |
| `publish_topology_and_profiles` | function | `run_benchmark` — once, before the scenario loop. |
| `publish_scenario_result` | function | `run_benchmark` — per scenario, right after it finishes. |
| `publish_hc_and_comparison` | function | `run_benchmark` — after the scenario loop + HC. |
| `build_*` | functions | The above publishers + `PublishHandle.on_timestep`. |
| `load_topology`/`load_profiles`/`load_scenario` | functions | The Streamlit dashboard (external consumer). |
| `_dump`/`_append_jsonl`/`_jsonl_line`/`_nan_to_none`/`_Encoder` | writers | All payload writes. |

---

## 4. Section A — Encoder & NaN-safe writers (L238–305)

```python
class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, (np.floating, float)):
            if math.isnan(obj) or math.isinf(obj): return None
            return float(obj)
        ... np.bool_ → bool; np.ndarray/Series/Index → list; Timestamp → iso; DataFrame → records ...

def _nan_to_none(obj):   # recursive: non-finite float → None, into dicts/lists
def _dump(obj, path, indent=2):        json.dump(_nan_to_none(obj), cls=_Encoder, allow_nan=False, ...)
def _jsonl_line(frame):     return json.dumps(_nan_to_none(frame), cls=_Encoder, allow_nan=False, ...) + "\n"
def _append_jsonl(frame, path):        fh.write(json.dumps(_nan_to_none(frame), cls=_Encoder, allow_nan=False, ...))
```

#### Line 238–305: The JSON layer — and the applied NaN fixes

`_Encoder` converts the numpy/pandas types pandapower emits (`np.integer/floating/bool_`,
`ndarray`, `Timestamp`, `Series`, `Index`, `DataFrame`) into JSON-native types. **The
NaN-safety fixes are here and complete:** `_nan_to_none` recursively replaces non-finite
floats with `None`, and it is applied — with `allow_nan=False` — in **all three writers**
(`_dump` L279, `_jsonl_line` L297, `_append_jsonl` L304). **Design rationale (the fix's own
comment, L284–286):** `_Encoder.default()`'s `NaN→None` branch **never fires for native
Python floats** (json serialises them itself), so a pre-pass sanitiser is required to keep
the output strict-valid JSON; `allow_nan=False` then makes any *missed* non-finite raise
loudly rather than silently emit the invalid `NaN` token. This closes the issue flagged in
6A-ii for both the checkpoint stream and the whole-file summaries. *(The `_Encoder` NaN branch
now only fires for `np.floating` NaN and is harmless belt-and-braces.)*

---

## 5. Section B — build_topology (L312–453)

```python
    for idx, row in net.bus.iterrows():
        if has_geo: x, y = _parse_geo(row.get("geo"))      # GeoJSON Point string
        elif has_x and has_y: x, y = row["x"], row["y"]     # pandapower 3.x columns
        else: x, y = None, None
    ... lines, trafos (tap_* fields), sgens (nameplate p_mw, sn_mva, type), loads ...
    dist_series = pptop.calc_distance_to_bus(deepcopy(net), ext_grid.bus[0])   # feeder_dist
    from violation_detector import V_MIN, V_MAX; "voltage_limits": {V_MIN, V_MAX}
```

#### Line 312–453: Static network geometry payload

Builds the once-per-run `topology.json`: bus coordinates (from a **GeoJSON `geo` string**
*or* the pandapower-3.x `x`/`y` columns, `null` when absent — Kerber/Dickert without
`create_generic_coordinates`), line/trafo/sgen/load metadata (sgen `p_mw` is the
**nameplate** rating, not a profile value), and **`feeder_dist`** — topological distance from
each bus to the slack via `calc_distance_to_bus` on a **deep copy** (so the working net is
untouched), used for the voltage-vs-feeder-length plots. The voltage band is read from
`violation_detector` (Doc 11) so the dashboard highlights the same limits the detector uses.
**Design rationale:** everything static about the network lives in one payload the dashboard
loads once and pairs with the per-timestep frames.

---

## 6. Section C — build_profiles_payload (L460–584)

```python
    times = profiles["times"]
    agg_load = load_df.sum(axis=1); agg_pv = pv_df.sum(axis=1); agg_wind = wind_df.sum(axis=1)
    payload = {
        "times_10min":  [t.isoformat() for t in times],
        "load_10min"/"pv_10min"/"wind_10min": _to_list(agg_*),           # full resolution
        "times_hourly": ...,  "load_hourly"/"pv_hourly"/"wind_hourly": _resample(agg_*, "1H"),
        "extreme_days": profiles["extreme_days"],                        # {max_load, min_load, max_pv, max_wind}
    }
```

#### Line 460–584: The dual-resolution profile payload

Serialises the aggregate load/PV/wind series (summed across all elements — the feeder totals
the dashboard charts) at **two resolutions**: the full **10-minute** series and an
**hourly-resampled** series (`_resample`), plus the four **extreme-day markers**. **Units:**
MW; timestamps ISO-8601. **Derivation of the two-resolution choice:** a full annual 10-minute
series is 35,136 points per quantity — three of them is ~105k points, heavy to chart as an
overview but essential for a zoomed day. Emitting **both** lets the dashboard draw the
annual overview from the ~8,760-point hourly series (light) and switch to the 10-minute series
only when the user zooms. **Worked example:** for `1-MV-rural--2-sw` the payload carries
`load_hourly` (8,784 points, 2016 leap year), `load_10min` (35,136 points), and
`extreme_days = {"max_load": "2016-01-20", "max_pv": "2016-07-05", …}` — the same markers
`plot_results` fig08 and `network_plotter`'s day-zoom use, so dashboard and figures reference
the same days. **Design rationale:** pre-resampling server-side (once, at publish) beats
resampling client-side (every chart render).

---

## 7. Section D — build_scenario_payload (L590–725)

```python
    for rec in result.records:
        row = {"t", "timestamp", "converged",
               "max_vm_pu"/"min_vm_pu"/"max_line_loading_pct"/... (scalars),
               "vm_pu_by_bus"/"line_loading_pct"/"trafo_loading_pct" (_series_to_dict),
               violation lists + violation_flag,
               tap_* (S2), svc_* (S3), q/p_by_sgen + coordination/curtail/q_sat + hil_latency/t_total (S4),
               energy balance}
    summary = {k: _s(v) for k,v in result.summary_dict().items()}
    return {"scenario_id", "network_id", "elapsed_s", "summary", "timeseries": records}
```

#### Line 590–725: The full per-scenario timeseries payload

Serialises **every** `TimestepRecord` field (6A-ii) into a flat list of JSON rows, plus the
per-element dicts (`vm_pu_by_bus` etc., keyed by string element index), the violation lists,
every scenario-specific block (OLTC `tap_*`, SVC `svc_*`, Volt-Var `q/p_by_sgen` +
`coordination_active` + `hil_latency_ms`/`t_total_ms`), and the energy balance — then appends
the scalar `summary_dict`. `_s` and `_series_to_dict` are the NaN-safe scalar/series coercers.
**Design rationale — this is the complete record**, written by `publish_scenario_result`
compactly (`indent=None`, §10) because an annual scenario is ~100k rows and pretty-printing
would double the write burst for no reader benefit. Contrast the **compact** live frame (§8).

---

## 8. Section E — build_hc_payload & build_live_frame (L732–854)

#### Line 732–782: build_hc_payload

```python
    for label, hc in [("baseline", hc_baseline), ("volt_var", hc_volt_var)]:
        payload[label] = {"hc_mw", "binding_bus", "binding_vm_pu", "eof_bus", "sweep": [...]}
    payload["gain_mw"] = hc_volt_var.hc_mw - hc_baseline.hc_mw
```

Serialises the hosting-capacity comparison — for each of baseline and volt_var: the HC in MW,
the **binding bus** and its voltage (where the sweep first violates), the end-of-feeder bus,
and the sweep curve — then the headline **`gain_mw = volt_var − baseline`**. **Code-to-physics:**
`gain_mw` is the extra DER capacity Volt-Var control unlocks before the first voltage violation.
**Worked example:** on a Synthetic-LV suburb where baseline HC = 0.42 MW and volt_var HC =
0.55 MW, `gain_mw = 0.13` MW — the dashboard's headline number. **Units:** MW; voltages pu.
The docstring notes per-sweep-step voltages aren't stored, so it reconstructs the sweep
endpoints from `HC_PARAMS`. **Design rationale:** the gain is *the* HC result — surfacing it as
a single field (not something the reader must compute) matches the "structured, not raw"
philosophy.

#### Line 789–854: build_live_frame

```python
    frame = {"t", "timestamp", "progress": (t + 1) / total_steps,
             "vm_pu_by_bus", "line_loading_pct", "trafo_loading_pct",         # per-element (for the map)
             "over_voltage_buses"/"under_voltage_buses"/"overloaded_lines",   # for red highlighting
             "tap_pos", "svc_q_mvar", "curtailment_needed", "coordination_active",   # control scalars
             "max_vm_pu"/"min_vm_pu"/"max_line_loading_pct"}                  # network extremes
```

Builds the **compact** single-timestep frame for the animated dashboard map — only the fields
the animation renders: per-bus/line/trafo values (to colour the map), the violation lists (to
highlight in red), a handful of control scalars, the network extremes, and a **`progress`**
fraction. **The compact-vs-lossless contrast, made concrete (6A-ii §4.5):** a
`TimestepRecord.to_checkpoint_dict()` carries *every* field (energy balance, per-DER q/p
arrays, hil-latency, t_total) so a resumed run is faithful; `build_live_frame` **drops** all of
those — the animation doesn't draw them — keeping each live frame small. **Worked example:** at
`t = 15,000` of 35,136, `progress = 0.4269`; the frame carries `vm_pu_by_bus` (one value per
bus, for the colour map) and `over_voltage_buses = [12, 47]` (highlight those two red), but
**not** the per-DER reactive arrays the checkpoint keeps. **Design rationale:** two
serialisations for two consumers — a *coarse visual sample* at `update_every_k` cadence for the
eye, and a *lossless, every-timestep* record for resume.

---

## 9. Section F — PublishHandle (L861–1053)

The stateful callback wired into `run_benchmark` — the checkpointing core.

#### Line 861–896: Fields & the two-stream contract

`output_dir`, `update_every_k` (live-frame cadence, default 6), `enable_checkpointing`, plus
internal per-scenario state (paths, the held-open `_checkpoint_fh`, elapsed tracking). The
class docstring states the **two-stream contract** precisely: `live/<id>.jsonl` (compact,
truncated every run, display-only, never used for resume) and `checkpoint/<id>.jsonl` (full
records, **never truncated by this class**, the resume source). **A fresh run on a reused
`output_dir` requires deleting the checkpoint explicitly** — the class will not silently discard
it (so a resume is never lost by accident).

#### Line 898–947: on_scenario_start — truncate live, hold-open checkpoint

Resets the live file (fresh dashboard each run), records context, loads the `.elapsed` sidecar,
and **opens the checkpoint file once (`open(..., "a")`), holding the handle for the whole
scenario.** **Design rationale (the docstring, verbatim intent):** reopening the checkpoint
100,000+ times over a run generates real open/close syscall + ext4 metadata-journal overhead,
**confirmed as a likely contributor to a sustained `mmc_rescan` hang on the RPi** — the
held-open handle is the fix. (This is the SD-card-crash root cause documented in the RPi log.)

#### Line 948–1005: get_resume_records & cumulative_elapsed_s

`get_resume_records` rebuilds `TimestepRecord`s from `checkpoint/<id>.jsonl` via
`from_checkpoint_dict` (6A-ii), **falling back to a `.completed` archive** (written by
`benchmark_runner` once the final `scenarios/<id>.json` is confirmed — so a corrupted final JSON
doesn't force a full re-simulation), and **silently drops a partial/corrupt trailing line** from
a crash (`JSONDecodeError`/`KeyError` → skip). `cumulative_elapsed_s` returns wall-time across
**all** attempts (via the `.elapsed` sidecar + this attempt's delta) so `elapsed_s` survives a
crash-and-resume cycle rather than resetting. **Design rationale:** correct resume must recover
both the *records* and the *elapsed clock*, and must tolerate the half-written last line a crash
inevitably leaves.

#### Line 1007–1053: on_timestep & on_scenario_end — two cadences

**`on_timestep`** does two things at **different cadences** (the key design point, L1008–1011):
it writes the checkpoint **every timestep, unconditionally** (write + `flush` on the held-open
handle — the resume source must be gap-free), but emits the live frame + `.elapsed` update only
**every `update_every_k`** steps (a coarse visual sample that tolerates gaps). **`on_scenario_end`**
closes the held-open handle, writes the final `.elapsed`, and appends a terminal
`scenario_complete` event to the live stream. **Design rationale — gap-free checkpoint, coarse
live:** conflating the two cadences would either bloat the live file or leave gaps in the resume
source; separating them gets both right.

---

## 10. Section G — The three-way publish split (L1059–1153)

```python
def publish_topology_and_profiles(net, profiles, output_dir, network_id):   # BEFORE scenarios
    _dump(build_topology(net) + network_id, out/"topology.json"); _dump(build_profiles_payload(...), out/"profiles.json")
def publish_scenario_result(scenario_result, output_dir):                    # per scenario, immediately
    _dump(build_scenario_payload(scenario_result), out/"scenarios"/f"{sid}.json", indent=None)   # compact
def publish_hc_and_comparison(result, output_dir):                           # AFTER all
    if hc_results: _dump(build_hc_payload(...), out/"hc.json")
    if comparison_df: _dump(comparison_df.to_dict("records"), out/"comparison.json")
```

#### Line 1059–1153: Three independent write points

The old monolithic `publish_result` is split into three write points by **when their inputs
become available**: topology/profiles **before** any scenario runs, each scenario's timeseries
**immediately** as it finishes, and hc/comparison **after** the full set is ready. **Two design
rationales (docstrings):** **crash-resilience** — a scenario crash no longer loses topology and
the already-finished scenarios' files (they're already on disk); and **no end-of-run write
burst** — writing each ~100k-row scenario file as it completes (compact, `indent=None`) spreads
the I/O instead of dumping everything at once at the end of a long run. `publish_scenario_result`
returns `None` for a skipped scenario (e.g. LV-excluded SVC).

---

## 11. Findings and flags

**No bugs found. The checkpointing-work fixes are confirmed fully applied:**

1. **NaN → null in every writer** — `_nan_to_none` + `allow_nan=False` in `_dump`,
   `_jsonl_line`, and `_append_jsonl` (both the fixes I suggested; §4). Output is strict-valid
   JSON everywhere.
2. **Held-open checkpoint handle** — the RPi `mmc_rescan`-hang fix (§9), documented in the code.

**Not bugs (documented for clarity):** the `_Encoder` NaN branch being dead for native floats is
now harmless (the sanitiser runs first); the two cadences in `on_timestep` are deliberate
(gap-free checkpoint vs coarse live); the checkpoint-not-truncated + explicit-delete-for-fresh-run
is a safety property; the three-way split is intentional crash-resilience.

---

## 12. Coverage checklist

| Entity | Lines | Kind | Documented | Notes |
|--------|-------|------|:----------:|-------|
| load helpers (`load_topology`/`load_profiles`/`load_scenario`) | 93–101 | fns | ✅ | dashboard consumers |
| `_Encoder` | 238–271 | class | ✅ | numpy/pandas → native |
| `_dump` / `_nan_to_none` / `_jsonl_line` / `_append_jsonl` | 274–305 | writers | ✅ | **NaN fixes applied** |
| `build_topology` (+ `_parse_geo`) | 312–453 | fn | ✅ | geometry + feeder_dist |
| `build_profiles_payload` | 460–584 | fn | ✅ | 10-min + hourly |
| `build_scenario_payload` | 590–725 | fn | ✅ | full per-timestep record |
| `build_hc_payload` | 732–782 | fn | ✅ | HC comparison + gain |
| `build_live_frame` | 789–854 | fn | ✅ | compact frame |
| `PublishHandle` (`__post_init__`/`on_scenario_start`/`get_resume_records`/`cumulative_elapsed_s`/`on_timestep`/`on_scenario_end`) | 861–1053 | dataclass | ✅ | **checkpointing core** |
| `publish_topology_and_profiles` | 1059–1093 | fn | ✅ | before scenarios |
| `publish_scenario_result` | 1096–1122 | fn | ✅ | per scenario, compact |
| `publish_hc_and_comparison` | 1125–1153 | fn | ✅ | after all |

**Also covered (not standalone entities):** the strict-JSON NaN-safety mechanism, the
two-stream (live vs checkpoint) design, the held-open-handle SD-card fix, the two-cadence
`on_timestep`, the compact-vs-lossless distinction, the crash-surviving elapsed clock, the
`.completed` archive fallback, and the three-way write split.

**Flags raised:** none (no bugs). Fixes confirmed applied.

---

*End of Document 10A. Next: Document 10B — `plot_results.py` (1315 lines): the matplotlib
report figures generated from the published scenario results.*
