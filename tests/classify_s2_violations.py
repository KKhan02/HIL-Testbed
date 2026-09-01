"""
classify_s2_violations.py

Purpose
-------
For every timestep where coordination_active=True but a real violation
still remains after coordination (the 18,379-timestep population from
check_coordination_ground_truth.py's RPi output), classify WHICH of the
following candidate factors was present at that timestep:

  F1  Q-exhaustion       : q_saturated_count is high relative to n_ders
                            (DERs are pinned at their own +-q_max ceiling)
  F2  Coordinator pass-through (rank-deficiency proxy)
                            : q_adjusted ~= q_initial despite coordination_active
                            and a residual violation -- consistent with the
                            rank-deficient S_viol early-return path in
                            sensitivity_coordinator.coordinate(), which is
                            never logged to the JSON directly (only to
                            logger.warning), so this is inferred, not read.
  F3  Band-mismatch artefact : the violation is only a "violation" under
                            sensitivity_coordinator's hardcoded 0.95/1.05
                            V_BAND, but would NOT be a violation under the
                            RunPlan's actual overridden v_min/v_max (e.g.
                            S2's 0.94/1.04). Confirms how much of the
                            residual-violation count is an artefact of the
                            confirmed V_BAND_LOWER/V_BAND_UPPER bug rather
                            than a real physical violation under the run's
                            own stated limits.
  F4  Curtailment-exhausted, still violating
                            : curtail_exhausted=True -- P was cut to the
                            MAX_CURTAIL_ITERS floor (or P=0) and the
                            violation still did not clear. Strong signal
                            the residual is structural (thermal loading or
                            a bus no available DER/curtailment can reach),
                            not a Q or P headroom problem at all.
  F5  Thermal (not voltage) violation present alongside the voltage one
                            : overloaded_lines / overloaded_trafos non-empty
                            at the same timestep -- distinguishes "voltage
                            violation coexists with a thermal violation"
                            from "voltage violation in isolation."

A timestep can match multiple factors simultaneously (they are not mutually
exclusive) -- the script reports the full co-occurrence breakdown, not just
a single label per timestep, since the whole point of this pass is to see
which combinations dominate rather than force a single explanation.

Data sources (all read-only; no files modified)
------------------------------------------------
Only publisher/ output is used -- topology.json (static DER metadata) and
scenarios/volt_var_coord.json (per-timestep TimestepRecord fields:
coordination_active, curtailment_needed, curtail_exhausted,
q_saturated_count, q_mvar_by_sgen, vm_pu_by_bus, over_voltage_buses,
under_voltage_buses, overloaded_lines, overloaded_trafos).

Usage
-----
Edit the two placeholder paths below (RPI_PUBLISHER_DIR /
LAPTOP_PUBLISHER_DIR) to point at the run's publisher/ folder -- the
directory that directly contains topology.json, profiles.json,
scenarios/, comparison.json. Do not point at a parent folder; the script
does its own navigation from the publisher/ root down into scenarios/.

    python classify_s2_violations.py

Optional CLI overrides (skip editing the script):

    python classify_s2_violations.py --rpi-dir <path> --laptop-dir <path>

By default the RPi side is treated as the primary population to classify
(matching check_coordination_ground_truth.py's convention), with the
Laptop side loaded alongside for cross-reference only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Optional

# ===========================================================================
# ---- EDIT THESE TWO PATHS -------------------------------------------------
# Point each at the run's publisher/ directory -- the folder that directly
# contains topology.json, scenarios/, comparison.json (NOT a parent folder,
# NOT the scenarios/ subfolder itself).
# ===========================================================================
RPI_PUBLISHER_DIR    = r"D:\My Files\Personal Projects\HIL-Testbed\outputs (RPi)\S2-HIL-Arduino\c53b876e-9d46-4e7f-8686-76f3be9aefb6\publisher"
LAPTOP_PUBLISHER_DIR = r"D:\My Files\Personal Projects\HIL-Testbed\outputs\publisher\1-MV-rural--2-sw_S2_Laptop_Dry"

SCENARIO_ID = "volt_var_coord"

COORD_MIN_VIOLATION_DV = 1e-3

# Coordinator band + the run's actual v_min/v_max are read from the run's
# own run_plan.json (written by executor._save_run_plan_copy), so they can
# never drift out of sync with the run being analysed. Fallback constants
# are used only if no run_plan.json is found next to the publisher dir.
_FALLBACK_COORD_V_BAND = (0.95, 1.05)
_FALLBACK_RUNPLAN_BAND = (0.94, 1.04)

def _load_run_bands(publisher_dir: Path) -> tuple[tuple[float, float],
                                                  tuple[float, float]]:
    """Return ((coord_lower, coord_upper), (runplan_v_min, runplan_v_max)).

    The coordinator band is the framework planning band (violation_detector
    V_MIN/V_MAX defaults, 0.95/1.05). The RunPlan band is this run's actual
    parameters.v_min/v_max. Both are read from run_plan.json when present;
    absent -> documented fallbacks with a printed warning."""
    # run_plan.json is saved one level up from publisher/ (in the run root).
    candidates = [
        publisher_dir / "run_plan.json",
        publisher_dir.parent / "run_plan.json",
        publisher_dir.parent.parent / "run_plan.json",
    ]
    for c in candidates:
        if c.is_file():
            with open(c, encoding="utf-8") as f:
                plan = json.load(f)
            params = plan.get("parameters") or {}
            v_min = float(params.get("v_min", _FALLBACK_RUNPLAN_BAND[0]))
            v_max = float(params.get("v_max", _FALLBACK_RUNPLAN_BAND[1]))
            print(f"[classify_s2] Read run band from {c}: "
                  f"v_min={v_min}, v_max={v_max}")
            # Coordinator uses the framework planning band, independent of
            # the run's tightened limits — kept as the documented default.
            return _FALLBACK_COORD_V_BAND, (v_min, v_max)
    print("[classify_s2] WARNING: no run_plan.json found near "
          f"{publisher_dir} — using fallback bands "
          f"{_FALLBACK_COORD_V_BAND} / {_FALLBACK_RUNPLAN_BAND}. "
          "Verify these match the run being analysed.")
    return _FALLBACK_COORD_V_BAND, _FALLBACK_RUNPLAN_BAND
    
# q_saturated_count threshold for F1 ("Q-exhaustion"): fraction of profiled
# DERs (n_ders, read from the data itself, not hardcoded) at their own
# +-q_max limit. 0.30 is a starting heuristic -- inspect the printed
# distribution and adjust if the natural break in the data sits elsewhere.
F1_SATURATION_FRACTION_THRESHOLD = 0.30

# F2 (coordinator pass-through / rank-deficiency proxy): q_adjusted is
# considered "unchanged" from q_initial if every DER's |q_adjusted -
# q_initial| is below this tolerance. Matches SATURATION_TOL's order of
# magnitude from sensitivity_coordinator.py, loosened slightly since we are
# comparing persisted q_applied_mvar (post-PT1-dynamics) rather than the
# raw pre-dynamics q_adjusted, which is NOT persisted separately -- see
# note in the loader below.
F2_PASSTHROUGH_TOL_MVAR = 1e-4


# ===========================================================================
# Loading
# ===========================================================================

def load_scenario(publisher_dir: Path, scenario_id: str) -> list[dict]:
    """
    Navigate from the publisher/ root down to scenarios/<scenario_id>.json
    and return the list of per-timestep records.

    Handles both the {"timeseries": [...]} wrapper shape and a bare
    top-level list, since build_scenario_payload's exact top-level shape
    has varied across project sessions per prior diagnostic scripts.
    """
    path = publisher_dir / "scenarios" / f"{scenario_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Scenario file not found: {path}\n"
            f"Check that {publisher_dir} is the publisher/ directory itself "
            f"(the folder that directly contains topology.json and scenarios/), "
            f"not a parent or child of it."
        )
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "timeseries" in payload:
        return payload["timeseries"]
    if isinstance(payload, list):
        return payload
    raise ValueError(
        f"Unrecognised scenario JSON shape at {path}: top-level type "
        f"{type(payload)}, keys="
        f"{list(payload.keys()) if isinstance(payload, dict) else 'n/a'}"
    )


def load_topology(publisher_dir: Path) -> dict:
    """Load topology.json from the publisher/ root. Used for n_ders / DER metadata."""
    path = publisher_dir / "topology.json"
    if not path.exists():
        raise FileNotFoundError(f"topology.json not found at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# Classification
# ===========================================================================

def closest_margin_to_band(vm_pu_by_bus: dict, v_lower: float, v_upper: float) -> float:
    """
    Smallest signed distance from [v_lower, v_upper] across all buses.
    Positive => inside the band. Negative => a real violation under THIS band.
    Mirrors check_coordination_ground_truth.py's closest_margin_to_band(),
    parametrised so it can be evaluated against two different bands (the
    coordinator's fixed band vs. the RunPlan's actual overridden band).
    """
    if not vm_pu_by_bus:
        return float("nan")
    margins = []
    for v in vm_pu_by_bus.values():
        if v is None:
            continue
        margins.append(min(v - v_lower, v_upper - v))
    return min(margins) if margins else float("nan")


def classify_timestep(rec: dict, n_ders: int) -> dict:
    """
    Evaluate F1-F5 for one timestep record. Returns a dict of booleans
    plus the underlying values used, so the raw numbers are auditable
    alongside the classification.
    """
    vm_pu_by_bus = rec.get("vm_pu_by_bus") or {}

    margin_coord_band = closest_margin_to_band(
        vm_pu_by_bus, COORD_V_BAND_LOWER, COORD_V_BAND_UPPER
    )
    margin_runplan_band = closest_margin_to_band(
        vm_pu_by_bus, RUNPLAN_V_MIN, RUNPLAN_V_MAX
    )

    # F1 -- Q-exhaustion
    q_sat_count = rec.get("q_saturated_count")
    f1_q_exhaustion = (
        q_sat_count is not None
        and n_ders > 0
        and (q_sat_count / n_ders) >= F1_SATURATION_FRACTION_THRESHOLD
    )

    # F2 -- coordinator pass-through proxy.
    # NOTE: q_adjusted (Item 3's pre-dynamics output) is NOT persisted to
    # the scenario JSON separately from q_applied_mvar (post-PT1-dynamics,
    # per scenario_result.py's TimestepRecord fields). This is therefore an
    # approximation: at 15-min resolution the PT1 filter's alpha is very
    # close to 1 (near-instant settling), so q_applied_mvar ~= q_adjusted
    # in practice, but this is NOT the same guarantee the project's own
    # diagnose_qinitial_clip.py makes about q_initial (which also isn't
    # persisted). Flagged here rather than silently assumed.
    q_applied = rec.get("q_mvar_by_sgen") or {}
    # No q_initial persisted at all -- this factor cannot be computed
    # directly from published output. Left as None / not computed, with
    # coordination_active and a real violation as the only weaker proxy
    # available (i.e. "coordination ran, but didn't help" without being
    # able to say whether that's because q_adjusted == q_initial).
    f2_note = (
        "NOT COMPUTABLE from publisher output -- q_initial is never "
        "persisted (confirmed: diagnose_qinitial_clip.py's own docstring "
        "states this). True rank-deficiency detection requires re-running "
        "coordinate() with debug logging against the raw run, not the "
        "published JSON. This field is left as None throughout."
    )

    # F3 -- band-mismatch artefact: real violation under the coordinator's
    # fixed band, but NOT a violation under the RunPlan's actual overridden
    # band. This isolates how much of the "real violation" count is a
    # consequence of the confirmed V_BAND_LOWER/V_BAND_UPPER bug.
    f3_band_mismatch_artefact = (
        margin_coord_band == margin_coord_band  # not NaN
        and margin_runplan_band == margin_runplan_band  # not NaN
        and margin_coord_band < 0
        and margin_runplan_band >= 0
    )

    # F4 -- curtailment exhausted, still violating
    f4_curtail_exhausted = rec.get("curtail_exhausted") is True

    # F5 -- thermal violation coexists
    overloaded_lines = rec.get("overloaded_lines") or []
    overloaded_trafos = rec.get("overloaded_trafos") or []
    f5_thermal_coexists = len(overloaded_lines) > 0 or len(overloaded_trafos) > 0

    return {
        "t": rec.get("t"),
        "timestamp": rec.get("timestamp"),
        "margin_coord_band": margin_coord_band,
        "margin_runplan_band": margin_runplan_band,
        "q_saturated_count": q_sat_count,
        "n_ders": n_ders,
        "F1_q_exhaustion": f1_q_exhaustion,
        "F2_coordinator_passthrough": None,  # see f2_note
        "F2_note": f2_note,
        "F3_band_mismatch_artefact": f3_band_mismatch_artefact,
        "F4_curtail_exhausted": f4_curtail_exhausted,
        "F5_thermal_coexists": f5_thermal_coexists,
        "n_overloaded_lines": len(overloaded_lines),
        "n_overloaded_trafos": len(overloaded_trafos),
    }


def analyse(env_name: str, records: list[dict], n_ders: int, emit=print) -> list[dict]:
    """
    Reproduce check_coordination_ground_truth.py's own filter first
    (coordination_active=True, margin < 0 under the COORDINATOR's band --
    i.e. the exact 18,379-style population), then classify each one.
    """
    emit("\n" + "=" * 78)
    emit(f"{env_name} -- classification of real-violation timesteps")
    emit("=" * 78)

    active_recs = [r for r in records if r.get("coordination_active") is True]
    emit(f"Timesteps with coordination_active=True: {len(active_recs)} of {len(records)}")

    real_violation_recs = []
    for r in active_recs:
        vm_pu_by_bus = r.get("vm_pu_by_bus") or {}
        m = closest_margin_to_band(vm_pu_by_bus, COORD_V_BAND_LOWER, COORD_V_BAND_UPPER)
        if m == m and m < 0:  # not NaN and genuinely negative
            real_violation_recs.append(r)

    emit(f"Of those, real violations under the coordinator's own 0.95/1.05 band: "
         f"{len(real_violation_recs)}")

    if not real_violation_recs:
        emit("No real-violation timesteps to classify.")
        return []

    classified = [classify_timestep(r, n_ders) for r in real_violation_recs]

    n = len(classified)
    f1_count = sum(1 for c in classified if c["F1_q_exhaustion"])
    f3_count = sum(1 for c in classified if c["F3_band_mismatch_artefact"])
    f4_count = sum(1 for c in classified if c["F4_curtail_exhausted"])
    f5_count = sum(1 for c in classified if c["F5_thermal_coexists"])

    emit(f"\n--- Individual factor prevalence (out of {n} real-violation timesteps) ---")
    emit(f"  F1 Q-exhaustion (>= {F1_SATURATION_FRACTION_THRESHOLD:.0%} of DERs saturated) : "
         f"{f1_count} ({100*f1_count/n:.1f}%)")
    emit(f"  F2 Coordinator pass-through (rank-deficiency proxy)     : NOT COMPUTABLE "
         f"(q_initial not persisted -- see F2_note in per-record output)")
    emit(f"  F3 Band-mismatch artefact (not a violation under RunPlan's "
         f"{RUNPLAN_V_MIN}/{RUNPLAN_V_MAX} band) : {f3_count} ({100*f3_count/n:.1f}%)")
    emit(f"  F4 Curtailment exhausted, still violating               : "
         f"{f4_count} ({100*f4_count/n:.1f}%)")
    emit(f"  F5 Thermal violation coexists (line/trafo overload)     : "
         f"{f5_count} ({100*f5_count/n:.1f}%)")

    emit(f"\n--- Co-occurrence: F1 x F4 x F5 (F3 excluded -- orthogonal to F1/F4/F5, "
         f"reported separately since it flags a MEASUREMENT artefact, not a physical "
         f"cause) ---")
    combos: dict[tuple, int] = {}
    for c in classified:
        key = (c["F1_q_exhaustion"], c["F4_curtail_exhausted"], c["F5_thermal_coexists"])
        combos[key] = combos.get(key, 0) + 1
    for key, count in sorted(combos.items(), key=lambda kv: -kv[1]):
        f1, f4, f5 = key
        label = (
            f"F1={'Y' if f1 else 'N'} F4={'Y' if f4 else 'N'} F5={'Y' if f5 else 'N'}"
        )
        emit(f"  {label} : {count} ({100*count/n:.1f}%)")

    emit(f"\n--- Margin distribution under each band (pu, worst bus per timestep) ---")
    coord_margins = [c["margin_coord_band"] for c in classified if c["margin_coord_band"] == c["margin_coord_band"]]
    runplan_margins = [c["margin_runplan_band"] for c in classified if c["margin_runplan_band"] == c["margin_runplan_band"]]
    if coord_margins:
        emit(f"  Coordinator band (0.95/1.05)   : mean={mean(coord_margins):.5f} "
             f"median={median(coord_margins):.5f} min={min(coord_margins):.5f} "
             f"max={max(coord_margins):.5f}")
    if runplan_margins:
        emit(f"  RunPlan band ({RUNPLAN_V_MIN}/{RUNPLAN_V_MAX})    : mean={mean(runplan_margins):.5f} "
             f"median={median(runplan_margins):.5f} min={min(runplan_margins):.5f} "
             f"max={max(runplan_margins):.5f}")

    emit(f"\n--- q_saturated_count distribution at real-violation timesteps ---")
    sat_counts = [c["q_saturated_count"] for c in classified if c["q_saturated_count"] is not None]
    if sat_counts:
        emit(f"  n={len(sat_counts)} mean={mean(sat_counts):.1f} median={median(sat_counts):.1f} "
             f"min={min(sat_counts)} max={max(sat_counts)} (of {n_ders} DERs)")

    return classified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpi-dir", type=str, default=RPI_PUBLISHER_DIR,
                         help="Path to the RPi run's publisher/ directory")
    parser.add_argument("--laptop-dir", type=str, default=LAPTOP_PUBLISHER_DIR,
                         help="Path to the Laptop run's publisher/ directory")
    parser.add_argument("--out-json", type=str, default=None,
                         help="Optional path to dump full per-timestep classification as JSON")
    args = parser.parse_args()

    rpi_dir = Path(args.rpi_dir)
    laptop_dir = Path(args.laptop_dir)

    all_results = {}

    for env_name, pub_dir in (("RPi", rpi_dir), ("Laptop", laptop_dir)):
        if "PLACEHOLDER" in str(pub_dir):
            print(f"\n[SKIPPED] {env_name}: path is still a placeholder "
                  f"({pub_dir}). Edit RPI_PUBLISHER_DIR / LAPTOP_PUBLISHER_DIR "
                  f"at the top of this script, or pass --rpi-dir / --laptop-dir.")
            continue

        topo = load_topology(pub_dir)
        n_ders = len(topo.get("sgens", []))
        # Prefer the count of DERs that actually appear in the profiled
        # timeseries (q_mvar_by_sgen keys), since topology.json's sgens
        # list is a superset (confirmed: 102 static vs 98 profiled on this
        # project's primary network) -- fall back to len(sgens) only if no
        # records are available yet to sample from.
        records = load_scenario(pub_dir, SCENARIO_ID)
        profiled_der_ids = set()
        for r in records[:50]:
            profiled_der_ids.update((r.get("q_mvar_by_sgen") or {}).keys())
        if profiled_der_ids:
            n_ders = len(profiled_der_ids)

        print(f"\n[{env_name}] publisher dir : {pub_dir}")
        print(f"[{env_name}] topology.json sgens (static)      : {len(topo.get('sgens', []))}")
        print(f"[{env_name}] profiled DERs (from timeseries)   : {n_ders}")

        classified = analyse(env_name, records, n_ders)
        all_results[env_name] = classified

    if args.out_json and all_results:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nFull per-timestep classification written to {args.out_json}")


if __name__ == "__main__":
    main()
