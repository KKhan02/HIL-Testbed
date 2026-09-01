"""
publisher.py
============
Serialises HIL benchmark outputs into structured JSON payloads for the
Streamlit dashboard (or any future Flask/Django consumer).

This module is deliberately transport-agnostic.  It produces plain Python
dicts and lists — the Streamlit app decides how to render them (Plotly,
Altair, st.dataframe, etc.).  No Streamlit, Flask, or SSE code appears here.

Architecture
------------
The publisher operates in two modes:

1.  POST-RUN  (static export)
    publish_result(result, net, profiles, output_dir) → writes a set of JSON
    files to disk.  The Streamlit app loads them with st.cache_data.
    No callback wiring needed — works with the existing run_benchmark() call
    as-is.

2.  LIVE (streaming, optional)
    Wire a PublishHandle into the scenario runners.  The handle's
    on_timestep() method is called after every k-th timestep, appending a
    compact JSON frame to a newline-delimited JSONL file which the Streamlit
    app tails with st.empty() + polling.

    update_every_k controls the cadence (default=6 → one frame per hour at
    10-min resolution).  Set to 1 for every timestep, 144 for once per day.

File layout (output_dir/)
--------------------------
topology.json
    Static network geometry: buses (with x/y coords), lines, trafos, sgens,
    loads, feeder distances from slack.  Written once per run.

profiles.json
    Full annual (or selected-period) load + DER time series at full 10-min
    resolution, plus hourly totals for fast charting.  No pre-sliced extreme
    days — the Streamlit app applies the user's time-window filter and finds
    extreme days within that window.

hc.json
    Hosting capacity results: baseline and volt_var HCResult fields plus
    sweep parameters so the app can reconstruct the HC curve plot.

scenarios/<scenario_id>.json
    Full per-timestep time series for each completed scenario: vm_pu per bus,
    line/trafo loading, tap position, Q setpoints, curtailment flags, energy
    balance.  Written once per scenario after the runner finishes.

live/<scenario_id>.jsonl
    Newline-delimited compact JSON frames appended in real-time during a live
    run.  Each line is one timestep event.  Streamlit polls this file.

comparison.json
    Flat list of one dict per scenario — same schema as the benchmark CSV.

Dashboard figure mapping
------------------------
Figure                      Source file             Key(s) to use
---------------------------------------------------------------------
Network map (generation)    topology.json           sgens[*].{bus,sn_mva,type}
                                                    buses[*].{x,y,vn_kv}
Network map (line loading)  topology.json +         lines[*].{from_bus,to_bus}
                            scenarios/*.json        timeseries[t].line_loading_pct
Installed capacity chart    topology.json           sgens grouped by type, sum sn_mva
Voltage heatmap (animated)  live/<id>.jsonl         vm_pu_by_bus, line_loading_pct,
                            OR scenarios/*.json     over_voltage_buses (each frame/row)
Voltage vs feeder distance  topology.json +         feeder_dist[bus_idx]
  (SimBench Fig 10)         scenarios/*.json        timeseries[t].vm_pu_by_bus
Time series panels          scenarios/*.json        timeseries[*].{max_vm_pu,
  (pandapower Fig 7/8)                              min_vm_pu, max_line_loading_pct,
                                                    tap_pos, losses_mw, grid_import_mw}
Annual load + DER profile   profiles.json           times_10min, load_total_mw,
                                                    pv_total_mw, wind_total_mw
Extreme days (4 days)       profiles.json           Streamlit slices times_10min
                                                    to the selected window, then
                                                    finds idxmax/idxmin locally
Violation heatmap           scenarios/*.json        timeseries[*].violation_flag
  (timestep × scenario)                             (one column per scenario)
Q(V) operating scatter      scenarios/volt_var*.json timeseries[t].{vm_pu_by_bus,
                                                    q_mvar_by_sgen} joined on bus
HC sweep curve              hc.json                 baseline/volt_var.{hc_mw,
                                                    violated_at_mw, sweep_step_mw}
Coordination scatter        scenarios/volt_var_coord timeseries[t].coordination_active
                            .json                   (filter rows where True)

Concrete usage (Streamlit side)
--------------------------------
    import json, pandas as pd, streamlit as st

    @st.cache_data
    def load_topology(run_dir):
        return json.load(open(f"{run_dir}/topology.json"))

    @st.cache_data
    def load_profiles(run_dir):
        return json.load(open(f"{run_dir}/profiles.json"))

    @st.cache_data
    def load_scenario(run_dir, scenario_id):
        return json.load(open(f"{run_dir}/scenarios/{scenario_id}.json"))

    topo  = load_topology("publisher_output")
    prof  = load_profiles("publisher_output")
    sc4b  = load_scenario("publisher_output", "volt_var_coord")

    # Time series DataFrame from one scenario
    df = pd.DataFrame(sc4b["timeseries"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Buses as a DataFrame for map plotting
    buses_df = pd.DataFrame(topo["buses"])   # cols: index, name, vn_kv, x, y

    # Annual load profile (hourly)
    load_h = pd.Series(prof["load_total_mw"], index=pd.to_datetime(prof["times_hourly"]))

    # Live polling (in a Streamlit loop)
    import time
    placeholder = st.empty()
    with open("publisher_output/live/volt_var_coord.jsonl") as fh:
        while True:
            line = fh.readline()
            if line:
                frame = json.loads(line)
                if frame["event"] == "scenario_complete":
                    break
                # update network map with frame["vm_pu_by_bus"] etc.
                placeholder.json(frame)
            else:
                time.sleep(0.5)

JSON structure quick reference
-------------------------------
topology.json
    {
      "network_id": "1-MV-rural--2-sw",
      "buses":    [{index, name, vn_kv, x, y}, ...],
      "lines":    [{index, name, from_bus, to_bus, max_i_ka, length_km}, ...],
      "trafos":   [{index, name, hv_bus, lv_bus, sn_mva, tap_min, tap_max,
                    tap_neutral, tap_pos}, ...],
      "sgens":    [{index, name, bus, type, p_mw, q_mvar, sn_mva,
                    in_service}, ...],
      "loads":    [{index, name, bus, p_mw, q_mvar, in_service}, ...],
      "feeder_dist": {"7": 2.34, "8": 3.01, ...},    ← bus_idx_str → km
      "voltage_limits": {"v_min": 0.95, "v_max": 1.05}
    }

profiles.json
    {
      "network_id": "...",
      "times_10min":       ["2024-01-01T00:00:00+01:00", ...],   ← full resolution
      "times_hourly":      ["2024-01-01T00:00:00+01:00", ...],   ← hourly totals
      "load_total_mw":     [float, ...],   ← hourly, len = len(times_hourly)
      "pv_total_mw":       [float, ...],
      "wind_total_mw":     [float, ...],
      "net_injection_mw":  [float, ...],
      "load_total_10min":  [float, ...],   ← full resolution, len = len(times_10min)
      "pv_total_10min":    [float, ...],
      "wind_total_10min":  [float, ...],
      "net_injection_10min": [float, ...],
      "load_by_bus":  {"0": [float,...], "1": [...], ...},   ← omitted if >200 loads
      "pv_by_sgen":   {"4": [float,...], ...},
      "wind_by_sgen": {"8": [float,...], ...}
    }

scenarios/<id>.json
    {
      "scenario_id": "volt_var_coord",
      "network_id":  "1-MV-rural--2-sw",
      "elapsed_s":   142.3,
      "summary":     {n_violation_steps, vdi, curtailment_steps, ...},
      "timeseries": [
        {
          "t": 0,
          "timestamp": "2024-01-01T00:10:00+01:00",
          "converged": true,
          "max_vm_pu": 1.012,  "min_vm_pu": 0.998,
          "max_line_loading_pct": 34.2,  "max_trafo_loading_pct": 18.5,
          "vm_pu_by_bus":     {"5": 1.012, "6": 1.008, ...},
          "line_loading_pct": {"0": 34.2,  "1": 12.1,  ...},
          "trafo_loading_pct":{"0": 18.5},
          "over_voltage_buses": [],  "under_voltage_buses": [],
          "overloaded_lines":   [],  "overloaded_trafos":   [],
          "violation_flag": false,
          "tap_pos": null,  "tap_changed": null,
          "svc_q_mvar": null,  "svc_saturated": null,
          "q_mvar_by_sgen":  {"4": -0.12, "7": -0.08, ...},
          "p_mw_by_sgen":    {"4": 0.85,  "7": 0.62,  ...},
          "curtailment_needed": false,  "curtail_exhausted": false,
          "coordination_active": false,  "q_saturated_count": 0,
          "losses_mw": 0.042,  "grid_import_mw": 1.23,
          "der_gen_mw": 2.41,  "load_mw": 3.12
        },
        ...
      ]
    }

hc.json
    {
      "baseline": {case, hc_mw, violated_at_mw, binding_bus, binding_vm_pu,
                   n_steps, hc_limit_reached, endoffeeder_bus, dist_voltage_kv,
                   sweep_start_mw, sweep_step_mw, sweep_max_mw},
      "volt_var": {same fields, plus qv_converged, qv_iters_max},
      "gain_mw": 0.010
    }

live/<id>.jsonl   (one JSON object per line)
    {"event":"timestep", "scenario_id":"volt_var_coord", "t":0,
     "timestamp":"...", "progress":0.0, "converged":true,
     "vm_pu_by_bus":{...}, "line_loading_pct":{...}, "trafo_loading_pct":{...},
     "over_voltage_buses":[], "violation_flag":false,
     "tap_pos":null, "svc_q_mvar":null, "curtailment_needed":false,
     "max_vm_pu":1.012, "min_vm_pu":0.998, "max_line_loading":34.2}
    {"event":"scenario_complete", "scenario_id":"volt_var_coord", "elapsed_s":142.3}
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
import time
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON encoder — handles numpy scalars, NaN, Timestamps
# ---------------------------------------------------------------------------

class _Encoder(json.JSONEncoder):
    """
    Extend JSONEncoder to handle types common in pandapower/pandas output.

    Conversions
    -----------
    np.integer          → int
    np.floating / float → float (NaN/Inf → null)
    np.bool_            → bool
    np.ndarray          → list
    pd.Timestamp        → ISO 8601 string
    pd.Series / Index   → list
    pd.DataFrame        → records list
    """
    def default(self, obj):                             # noqa: C901
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.Index):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        return super().default(obj)


def _dump(obj: Any, path: Path, indent: int | None = 2) -> None:
    """Serialise obj to path as JSON using _Encoder (non-finite floats → null)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seps = (",", ":") if indent is None else (",", ": ")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_nan_to_none(obj), fh, cls=_Encoder, allow_nan=False,
                   indent=indent, separators=seps)
    logger.debug("publisher: wrote %s", path)

def _nan_to_none(obj):
    """Recursively replace non-finite floats with None for strict-JSON output.
    Needed because json serialises native float NaN/Inf as the invalid tokens
    NaN/Infinity — _Encoder.default() never sees native floats."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    return obj

def _jsonl_line(frame: dict) -> str:
    """Serialize one JSON object as a single JSONL line (caller writes it)."""
    return json.dumps(_nan_to_none(frame), cls=_Encoder,
                       allow_nan=False, separators=(",", ":")) + "\n"

def _append_jsonl(frame: dict, path: Path) -> None:
    """Append a single JSON object as one line to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_nan_to_none(frame), cls=_Encoder,
                            allow_nan=False, separators=(",", ":")) + "\n")


# ===========================================================================
# 1.  TOPOLOGY PAYLOAD
# ===========================================================================

def build_topology(net) -> dict:
    """
    Extract static network geometry and element metadata from a pandapower net.

    Returns a dict with keys:

    buses : list of {index, name, vn_kv, x, y}
        Coordinates are taken from net.bus["x"]/["y"] (pandapower 3.x).
        If coordinates are absent (Kerber/Dickert without create_generic_coordinates),
        x and y will be null — the Streamlit app must handle this gracefully.

    lines : list of {index, name, from_bus, to_bus, max_i_ka, length_km}

    trafos : list of {index, name, hv_bus, lv_bus, sn_mva, tap_min, tap_max,
                       tap_neutral, tap_pos}
        tap fields default to 0 / null when absent.

    sgens : list of {index, name, bus, type, p_mw, q_mvar, sn_mva, in_service}
        p_mw is the nameplate / rated active power (from net.sgen, not a
        profile value).  Used for Fig 2 left (generation distribution map)
        and Fig 3 (installed capacity chart).

    loads : list of {index, name, bus, p_mw, q_mvar, in_service}

    feeder_dist : dict {bus_index → distance_km}
        Topological distance from each bus to the slack (ext_grid) bus,
        computed via pandapower.topology.calc_distance_to_bus().
        Used for the voltage-vs-feeder-length plot (SimBench Fig 10).
        Null for buses that are not connected (isolated areas).

    voltage_limits : {v_min, v_max}
        The planning band used in violation detection.

    network_id : str   (passed through from the caller)
    """
    buses = []
    import json as _json

    def _parse_geo(geo_val):
        """Return (x, y) from a GeoJSON Point string, or (None, None)."""
        if geo_val is None or (isinstance(geo_val, float) and pd.isna(geo_val)):
            return None, None
        try:
            coords = _json.loads(geo_val)["coordinates"]  # [lon, lat] or [x, y]
            return float(coords[0]), float(coords[1])
        except Exception:
            return None, None

    has_geo = "geo" in net.bus.columns
    has_x   = "x"   in net.bus.columns
    has_y   = "y"   in net.bus.columns

    buses = []
    for idx, row in net.bus.iterrows():
        if has_geo:
            x, y = _parse_geo(row.get("geo"))
        elif has_x and has_y:
            x = float(row["x"]) if not pd.isna(row["x"]) else None
            y = float(row["y"]) if not pd.isna(row["y"]) else None
        else:
            x, y = None, None
        buses.append({
            "index": int(idx),
            "name":  str(row.get("name", "")),
            "vn_kv": float(row["vn_kv"]),
            "x":     x,
            "y":     y,
        })

    lines = []
    for idx, row in net.line.iterrows():
        lines.append({
            "index":     int(idx),
            "name":      str(row.get("name", "")),
            "from_bus":  int(row["from_bus"]),
            "to_bus":    int(row["to_bus"]),
            "max_i_ka":  float(row.get("max_i_ka", float("nan"))),
            "length_km": float(row.get("length_km", float("nan"))),
        })

    trafos = []
    for idx, row in net.trafo.iterrows():
        trafos.append({
            "index":       int(idx),
            "name":        str(row.get("name", "")),
            "hv_bus":      int(row["hv_bus"]),
            "lv_bus":      int(row["lv_bus"]),
            "sn_mva":      float(row.get("sn_mva", float("nan"))),
            "tap_min":     (int(row["tap_min"])     if "tap_min"     in net.trafo.columns and not pd.isna(row["tap_min"])     else None),
            "tap_max":     (int(row["tap_max"])     if "tap_max"     in net.trafo.columns and not pd.isna(row["tap_max"])     else None),
            "tap_neutral": (int(row["tap_neutral"]) if "tap_neutral" in net.trafo.columns and not pd.isna(row["tap_neutral"]) else None),
            "tap_pos": (int(row["tap_pos"]) if "tap_pos" in net.trafo.columns and not pd.isna(row["tap_pos"]) else 0),
        })

    sgens = []
    for idx, row in net.sgen.iterrows():
        sgens.append({
            "index":      int(idx),
            "name":       str(row.get("name", "")),
            "bus":        int(row["bus"]),
            "type":       str(row.get("type", "")),
            "p_mw":       float(row["p_mw"]),
            "q_mvar":     float(row.get("q_mvar", 0.0)),
            "sn_mva":     float(row["sn_mva"]) if "sn_mva" in net.sgen.columns else None,
            "in_service": bool(row["in_service"]),
        })

    loads = []
    for idx, row in net.load.iterrows():
        loads.append({
            "index":      int(idx),
            "name":       str(row.get("name", "")),
            "bus":        int(row["bus"]),
            "p_mw":       float(row["p_mw"]),
            "q_mvar":     float(row.get("q_mvar", 0.0)),
            "in_service": bool(row["in_service"]),
        })

    # Feeder distances from slack
    feeder_dist: dict[str, Any] = {}
    try:
        import pandapower.topology as pptop
        import copy
        _net = copy.deepcopy(net)
        dist_series = pptop.calc_distance_to_bus(_net, _net.ext_grid["bus"].iloc[0])
        for bus_idx, dist in dist_series.items():
            feeder_dist[str(int(bus_idx))] = (
                None if (math.isnan(dist) or math.isinf(dist)) else float(dist)
            )
    except Exception as exc:
        logger.warning("publisher: feeder distance calculation failed: %s", exc)

    from violation_detector import V_MIN, V_MAX
    return {
        "buses":          buses,
        "lines":          lines,
        "trafos":         trafos,
        "sgens":          sgens,
        "loads":          loads,
        "feeder_dist":    feeder_dist,
        "voltage_limits": {"v_min": V_MIN, "v_max": V_MAX},
    }


# ===========================================================================
# 2.  PROFILES PAYLOAD
# ===========================================================================

def build_profiles_payload(profiles: dict, network_id: str = "") -> dict:
    """
    Serialise profile_builder output for the dashboard.

    Parameters
    ----------
    profiles : dict returned by build_annual_profiles().
               Expected keys: "load", "pv", "wind", "times".

    Returns a dict with the following keys.  All list lengths match their
    named time axis (either times_10min or times_hourly).

    Time axes
    ---------
    times_10min : list[str]
        ISO timestamps at full 10-min resolution.  Use as x-axis for any
        chart that needs sub-hourly detail (extreme day plots, violation
        overlays).

    times_hourly : list[str]
        ISO timestamps at hourly resolution.  Use as x-axis for the annual
        overview charts — 8,784 points vs 52,698.

    Hourly totals (len = len(times_hourly))
    ----------------------------------------
    load_total_mw, pv_total_mw, wind_total_mw, net_injection_mw

    Full 10-min totals (len = len(times_10min))
    --------------------------------------------
    load_total_10min, pv_total_10min, wind_total_10min, net_injection_10min

    These allow the Streamlit app to:
    - Show the annual overview at hourly resolution (fast to render).
    - Let the user select a date range (e.g. June) and replot at 10-min
      resolution without a round-trip to the server.
    - Find the 4 extreme days within the user's selected window by
      scanning the 10-min totals (idxmax/idxmin on a sliced Series).

    Per-element series (full 10-min resolution)
    -------------------------------------------
    load_by_bus   : dict {load_index_str → list[float]}
        Per-load MW at full resolution.  Omitted (empty dict) when > 200
        loads — use the total instead.

    pv_by_sgen    : dict {sgen_index_str → list[float]}
    wind_by_sgen  : dict {sgen_index_str → list[float]}

    NOTE on extreme days
    --------------------
    Extreme days are NOT pre-computed here.  The Streamlit app should:
        1.  Convert times_10min to a pd.DatetimeIndex.
        2.  Apply the user's time-range filter (e.g. st.date_input).
        3.  Build daily sums of load_total_10min / net_injection_10min.
        4.  Call idxmax() / idxmin() within the filtered window.
        5.  Slice times_10min ± the selected day to get the 144-point
            (or 48-point for short days) extreme day series.
    This ensures "peak load day in June" rather than "peak load day in August
    being shown in a June-filtered view".
    """
    times: pd.DatetimeIndex = profiles["times"]
    load_df: pd.DataFrame   = profiles["load"]
    pv_df:   pd.DataFrame   = profiles["pv"]
    wind_df: pd.DataFrame   = profiles["wind"]

    # --- Network-wide totals ---
    load_total = load_df.sum(axis=1)
    pv_total   = pv_df.sum(axis=1)
    wind_total = wind_df.sum(axis=1)
    net_inj    = pv_total + wind_total - load_total

    # --- Hourly downsample ---
    hourly_kws = {"rule": "h", "closed": "left", "label": "left"}
    times_h    = load_total.resample(**hourly_kws).mean().index

    def _to_list(s: pd.Series) -> list:
        return [
            None if (math.isnan(v) or math.isinf(v)) else float(v)
            for v in s.values
        ]

    def _resample(s: pd.Series) -> list:
        return _to_list(s.resample(**hourly_kws).mean())

    # --- Per-element series (memory guard: omit per-bus if too many loads) ---
    N_LOADS_FULL = 200
    if len(load_df.columns) <= N_LOADS_FULL:
        load_by_bus = {
            str(col): _to_list(load_df[col])
            for col in load_df.columns
        }
    else:
        load_by_bus = {}

    pv_by_sgen = {
        str(col): _to_list(pv_df[col])
        for col in pv_df.columns
    }
    wind_by_sgen = {
        str(col): _to_list(wind_df[col])
        for col in wind_df.columns
    }

    return {
        "network_id":          network_id,
        # Full 10-min time axis
        "times_10min":         [ts.isoformat() for ts in times],
        # Hourly time axis
        "times_hourly":        [ts.isoformat() for ts in times_h],
        # Hourly totals
        "load_total_mw":       _resample(load_total),
        "pv_total_mw":         _resample(pv_total),
        "wind_total_mw":       _resample(wind_total),
        "net_injection_mw":    _resample(net_inj),
        # Full 10-min totals
        "load_total_10min":    _to_list(load_total),
        "pv_total_10min":      _to_list(pv_total),
        "wind_total_10min":    _to_list(wind_total),
        "net_injection_10min": _to_list(net_inj),
        # Per-element full resolution
        "load_by_bus":         load_by_bus,
        "pv_by_sgen":          pv_by_sgen,
        "wind_by_sgen":        wind_by_sgen,
        "extreme_days": profiles.get("extreme_days", {}),
    }


# ===========================================================================
# 3.  SCENARIO TIMESERIES PAYLOAD
# ===========================================================================

def build_scenario_payload(result) -> dict:
    """
    Serialise a ScenarioResult into a per-scenario JSON payload.

    Covers SimBench Fig 4/9/10 and pandapower Fig 7/8.

    Each timestep is one row in a flat list of dicts (orient="records").
    Null values replace NaN/inf for JSON compatibility.

    Per-timestep fields serialised
    --------------------------------
    t, timestamp
    converged
    max_vm_pu, min_vm_pu               — network-wide voltage extremes
    max_line_loading_pct               — network-wide thermal maximum
    max_trafo_loading_pct
    vm_pu_by_bus     dict {bus_idx → vm_pu}
    line_loading_pct dict {line_idx → loading_pct}
    trafo_loading_pct dict {trafo_idx → loading_pct}
    over_voltage_buses  list[int]
    under_voltage_buses list[int]
    overloaded_lines    list[int]
    overloaded_trafos   list[int]
    violation_flag      bool  — any violation this timestep
    tap_pos             int | null   — Scenario 2
    tap_changed         bool | null
    svc_q_mvar          float | null — Scenario 3
    svc_saturated       bool | null
    q_mvar_by_sgen      dict {sgen_idx → q_mvar} | null  — Scenario 4
    p_mw_by_sgen        dict {sgen_idx → p_mw}   | null
    curtailment_needed  bool | null
    curtail_exhausted   bool | null
    coordination_active bool | null
    q_saturated_count   int | null
    losses_mw           float | null
    grid_import_mw      float | null
    der_gen_mw          float | null
    load_mw             float | null

    Summary fields (from ScenarioResult scalars)
    --------------------------------------------
    summary : dict — all scalar metrics from result.summary_dict()
    """
    def _s(v):
        """Scalar → JSON-safe value."""
        if v is None:
            return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    def _series_to_dict(s: Optional[pd.Series]) -> Optional[dict]:
        if s is None or (isinstance(s, pd.Series) and s.empty):
            return None
        return {
            str(int(k)): _s(v)
            for k, v in s.items()
        }

    records = []
    for rec in result.records:
        vm   = rec.vm_pu
        ll   = rec.line_loading
        tl   = rec.trafo_loading

        row = {
            "t":                   int(rec.t),
            "timestamp":           rec.timestamp.isoformat() if pd.notna(rec.timestamp) else None,
            "converged":           bool(rec.converged),

            # Network-wide extremes (fast scalars for time-series charts)
            "max_vm_pu":           _s(float(vm.max()))  if not vm.empty else None,
            "min_vm_pu":           _s(float(vm.min()))  if not vm.empty else None,
            "max_line_loading_pct":_s(float(ll.max()))  if not ll.empty else None,
            "max_trafo_loading_pct":_s(float(tl.max())) if not tl.empty else None,

            # Full per-element dicts (for network map animation)
            "vm_pu_by_bus":        _series_to_dict(vm),
            "line_loading_pct":    _series_to_dict(ll),
            "trafo_loading_pct":   _series_to_dict(tl),

            # Violation lists
            "over_voltage_buses":  [int(b) for b in rec.over_voltage_buses],
            "under_voltage_buses": [int(b) for b in rec.under_voltage_buses],
            "overloaded_lines":    [int(l) for l in rec.overloaded_lines],
            "overloaded_trafos":   [int(t) for t in rec.overloaded_trafos],
            "violation_flag":      bool(
                rec.over_voltage_buses or rec.under_voltage_buses
                or rec.overloaded_lines or rec.overloaded_trafos
            ),

            # Scenario 2 — OLTC
            "tap_pos":             _s(rec.tap_pos),
            "tap_changed":         _s(rec.tap_changed),
            "tap_attempted":       _s(rec.tap_attempted),
            "tap_blocked_reason":  rec.tap_blocked_reason,

            # Scenario 3 — SVC
            "svc_q_mvar":         _s(rec.svc_q_mvar),
            "svc_saturated":      _s(rec.svc_saturated),

            # Scenario 4 — Volt-Var
            "q_mvar_by_sgen":     _series_to_dict(rec.q_applied_mvar),
            "p_mw_by_sgen":       _series_to_dict(rec.p_applied_mw),
            "curtailment_needed": _s(rec.curtailment_needed),
            "curtail_exhausted":  _s(rec.curtail_exhausted),
            "coordination_active":_s(rec.coordination_active),
            "q_saturated_count":  _s(rec.q_saturated_count),
            "hil_latency_ms":     _s(rec.hil_latency_ms), 
            "t_total_ms":         _s(rec.t_total_ms),

            # Energy balance (all scenarios)
            "losses_mw":          _s(rec.losses_mw),
            "grid_import_mw":     _s(rec.grid_import_mw),
            "der_gen_mw":         _s(rec.der_gen_mw),
            "load_mw":            _s(rec.load_mw),
        }
        records.append(row)

    # Clean scalar summary (NaN-safe)
    raw_summary = result.summary_dict()
    summary = {k: _s(v) for k, v in raw_summary.items()}

    return {
        "scenario_id":  result.scenario_id,
        "network_id":   result.network_id,
        "elapsed_s":    _s(result.elapsed_s),
        "summary":      summary,
        "timeseries":   records,
    }


# ===========================================================================
# 4.  HOSTING CAPACITY PAYLOAD
# ===========================================================================

def build_hc_payload(hc_results: list) -> dict:
    """
    Serialise a list of HCResult objects (baseline + volt_var) into a dict.

    Returns
    -------
    dict with keys:
        baseline : HCResult summary fields + sweep curve points
        volt_var : same
        gain_mw  : volt_var.hc_mw − baseline.hc_mw

    The sweep curve is reconstructed from HC_PARAMS and the n_steps field
    so the dashboard can plot MW added vs binding vm_pu at end-of-feeder.
    Since individual step voltages are not stored in HCResult, the curve
    has only the terminal points.  The full curve would require changes to
    hosting_capacity.py to emit per-step voltages.
    """
    def _hc_dict(r) -> dict:
        return {
            "case":             r.case,
            "hc_mw":            r.hc_mw,
            "violated_at_mw":   None if (r.violated_at_mw is None or
                                          (isinstance(r.violated_at_mw, float)
                                           and math.isnan(r.violated_at_mw)))
                                      else float(r.violated_at_mw),
            "binding_bus":      int(r.binding_bus) if r.binding_bus != -1 else None,
            "binding_vm_pu":    None if (isinstance(r.binding_vm_pu, float)
                                         and math.isnan(r.binding_vm_pu))
                                      else float(r.binding_vm_pu),
            "n_steps":          int(r.n_steps),
            "hc_limit_reached": bool(r.hc_limit_reached),
            "endoffeeder_bus":  int(r.endoffeeder_bus),
            "dist_voltage_kv":  float(r.dist_voltage_kv),
            "qv_converged":     r.qv_converged,
            "qv_iters_max":     r.qv_iters_max,
            # Sweep reconstruction (start, end, step from params)
            "sweep_start_mw":   float(r.params.get("start", 0.0)),
            "sweep_step_mw":    float(r.params.get("step",  0.5)),
            "sweep_max_mw":     float(r.params.get("max",  20.0)),
            "sweep_curve":      r.sweep_curve,
        }

    if not hc_results or len(hc_results) < 2:
        return {"error": "HC results unavailable"}

    hc_b, hc_v = hc_results[0], hc_results[1]
    return {
        "baseline": _hc_dict(hc_b),
        "volt_var": _hc_dict(hc_v),
        "gain_mw":  round(hc_v.hc_mw - hc_b.hc_mw, 4),
    }


# ===========================================================================
# 5.  LIVE FRAME (single timestep, compact)
# ===========================================================================

def build_live_frame(
        scenario_id: str,
        scenario_label: str,
        rec,            # TimestepRecord
        t_total: int,
) -> dict:
    """
    Build a compact live frame for a single timestep.

    Emitted to the JSONL live file during a running simulation.  The
    Streamlit app polls this file and updates the animated network map
    (SimBench Fig 9 equivalent) at the configured cadence.

    Deliberately compact — only the fields needed for live animation:
        - vm_pu per bus (for colour-coding nodes)
        - line_loading_pct per line (for colour-coding edges)
        - trafo_loading_pct per trafo
        - violation lists (for red highlighting)
        - control action scalars (tap, svc_q, curtailment flag)
        - progress fraction (t / t_total) for the progress bar

    The Streamlit app pairs this with topology.json (loaded once) to
    reconstruct the full network map.
    """
    def _s(v):
        if v is None:
            return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    vm  = rec.vm_pu
    ll  = rec.line_loading
    tl  = rec.trafo_loading

    return {
        "event":            "timestep",
        "scenario_id":      scenario_id,
        "scenario_label":   scenario_label,
        "t":                int(rec.t),
        "timestamp":        rec.timestamp.isoformat() if pd.notna(rec.timestamp) else None,
        "progress":         round(rec.t / max(t_total, 1), 4),
        "converged":        bool(rec.converged),
        "vm_pu_by_bus":     {str(int(k)): _s(float(v)) for k, v in vm.items()} if not vm.empty else {},
        "line_loading_pct": {str(int(k)): _s(float(v)) for k, v in ll.items()} if not ll.empty else {},
        "trafo_loading_pct":{str(int(k)): _s(float(v)) for k, v in tl.items()} if not tl.empty else {},
        "over_voltage_buses":  [int(b) for b in rec.over_voltage_buses],
        "under_voltage_buses": [int(b) for b in rec.under_voltage_buses],
        "overloaded_lines":    [int(l) for l in rec.overloaded_lines],
        "overloaded_trafos":   [int(t) for t in rec.overloaded_trafos],
        "violation_flag":      bool(
            rec.over_voltage_buses or rec.under_voltage_buses
            or rec.overloaded_lines or rec.overloaded_trafos
        ),
        # Control scalars (scenario-specific; null when not applicable)
        "tap_pos":            _s(rec.tap_pos),
        "tap_changed":        _s(rec.tap_changed),
        "svc_q_mvar":         _s(rec.svc_q_mvar),
        "svc_saturated":      _s(rec.svc_saturated),
        "curtailment_needed": _s(rec.curtailment_needed),
        "curtail_exhausted":  _s(rec.curtail_exhausted),
        "coordination_active":_s(rec.coordination_active),
        "max_vm_pu":          _s(float(vm.max())) if not vm.empty else None,
        "min_vm_pu":          _s(float(vm.min())) if not vm.empty else None,
        "max_line_loading":   _s(float(ll.max())) if not ll.empty else None,
    }


# ===========================================================================
# 6.  PUBLISH HANDLE (live streaming)
# ===========================================================================

@dataclass
class PublishHandle:
    """
    Stateful callback object wired into run_benchmark() for live publishing
    AND crash-resume checkpointing.

    Two independent JSONL streams per scenario:
      live/<id>.jsonl        — compact dashboard frames, TRUNCATED every run.
                                Display-only; never used for resume.
      checkpoint/<id>.jsonl  — full TimestepRecord serialization, NEVER
                                truncated by this class. Read by
                                get_resume_records() to recover state after
                                a crash. Written only when enable_checkpointing
                                is True.

    A caller that wants a genuinely fresh run on a reused output_dir must
    delete checkpoint/<id>.jsonl explicitly (or pass a new output_dir) —
    this class will not silently discard it.
    """
    output_dir:            Path | str = field(default="publisher_output")
    update_every_k:        int        = field(default=6)
    enable_checkpointing:  bool       = field(default=True)

    # Internal state — set by on_scenario_start()
    _scenario_id:     str = field(default="", init=False, repr=False)
    _scenario_label:  str = field(default="", init=False, repr=False)
    _t_total:         int = field(default=1,  init=False, repr=False)
    _live_path:       Optional[Path] = field(default=None, init=False, repr=False)
    _checkpoint_path: Optional[Path] = field(default=None, init=False, repr=False)
    _checkpoint_fh:   Optional[Any]  = field(default=None, init=False, repr=False)
    _attempt_start_perf:    Optional[float] = field(default=None, init=False, repr=False)
    _elapsed_before_attempt: float          = field(default=0.0, init=False, repr=False)
    _elapsed_path: Optional[Path] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)

    def on_scenario_start(
            self,
            scenario_id:    str,
            scenario_label: str,
            t_total:        int,
    ) -> None:
        """
        Call once before the timestep loop of each scenario runner.

        Resets the live JSONL file for this scenario and stores context.
        Opens the checkpoint file once for the whole scenario (see
        on_timestep()/on_scenario_end()) instead of reopening it every
        timestep — reopening 100,000+ times over a run generates real
        open/close syscall and ext4 metadata-journal overhead, confirmed as
        a likely contributor to a sustained mmc_rescan hang on the RPi.
        """
        if self._checkpoint_fh is not None:
            self._checkpoint_fh.close()
            self._checkpoint_fh = None

        self._scenario_id    = scenario_id
        self._scenario_label = scenario_label
        self._t_total        = t_total
        self._live_path       = self.output_dir / "live" / f"{scenario_id}.jsonl"
        self._live_path.parent.mkdir(parents=True, exist_ok=True)
        self._live_path.write_text("", encoding="utf-8")  # dashboard file: always fresh
        self._attempt_start_perf = time.perf_counter()

        if self.enable_checkpointing:
            self._checkpoint_path = self.output_dir / "checkpoint" / f"{scenario_id}.jsonl"
            self._elapsed_path = self._checkpoint_path.with_name(self._checkpoint_path.name + ".elapsed")
            try:
                self._elapsed_before_attempt = float(self._elapsed_path.read_text().strip())
            except (FileNotFoundError, ValueError):
                self._elapsed_before_attempt = 0.0
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            # NOT truncated -- this is the resume source. Opened once here,
            # held open for the whole scenario.
            self._checkpoint_fh = open(self._checkpoint_path, "a", encoding="utf-8")
        else:
            self._checkpoint_path = None
            self._checkpoint_fh = None
            self._elapsed_path = None
            self._elapsed_before_attempt = 0.0

        logger.info(
            "publisher: live=%s checkpoint=%s (every %d steps, checkpointing=%s)",
            self._live_path, self._checkpoint_path, self.update_every_k,
            self.enable_checkpointing,
        )
    def get_resume_records(self, scenario_id: str) -> list:
        """
        Reconstruct TimestepRecord objects from an existing checkpoint file.

        Returns [] if checkpointing is disabled, or no checkpoint exists —
        live or archived — with any valid lines (i.e. this is a fresh run).

        Checks the live checkpoint path first
        (checkpoint/<scenario_id>.jsonl), then falls back to an archived,
        completed checkpoint (checkpoint/<scenario_id>.jsonl.completed) —
        written by benchmark_runner.py's scenario loop once that scenario's
        final scenarios/<id>.json is confirmed written.

        This fallback only matters in one narrow case: the final JSON later
        gets corrupted or goes missing, the layer-1 "already complete"
        check in benchmark_runner.py fails its read and falls through to
        calling the runner again, and — without this fallback — the runner
        would silently re-simulate the whole scenario from scratch even
        though a complete, valid checkpoint is sitting right there under
        its archived name.
        """
        if not self.enable_checkpointing:
            return []
        from scenario_result import TimestepRecord
        path = self.output_dir / "checkpoint" / f"{scenario_id}.jsonl"
        if not path.exists():
            archived = path.parent / (path.name + ".completed")
            if archived.exists():
                logger.info(
                    "get_resume_records(%r): live checkpoint not found, "
                    "using archived checkpoint %s instead.",
                    scenario_id, archived,
                )
                path = archived
            else:
                return []
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(TimestepRecord.from_checkpoint_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError):
                    continue  # drop a partial/corrupt trailing line from a crash
        return out
    
    def cumulative_elapsed_s(self) -> float:
        """
        Total wall-clock time on this scenario_id across ALL attempts, including
        any that crashed before completing — not just this process's own
        runtime. Use this instead of a local time.perf_counter() delta so
        elapsed_s survives a crash-and-resume cycle.
        """
        if self._attempt_start_perf is None:
            return 0.0
        return self._elapsed_before_attempt + (time.perf_counter() - self._attempt_start_perf)

    def on_timestep(self, rec) -> None:
        # Checkpoint: EVERY timestep, unconditionally -- this is the resume
        # source and must be a complete, gap-free record of the run. The
        # update_every_k cadence below applies ONLY to the dashboard live
        # frame, which is a coarse visual sample and can tolerate gaps.
        if self._checkpoint_fh is not None:
            self._checkpoint_fh.write(_jsonl_line(rec.to_checkpoint_dict()))
            self._checkpoint_fh.flush()

        if rec.t % self.update_every_k != 0:
            return

        if self._live_path is None:
            logger.warning("publisher.on_timestep: on_scenario_start() not called yet")
            return

        if rec.t % self.update_every_k == 0 and self._elapsed_path is not None:
            self._elapsed_path.write_text(f"{self.cumulative_elapsed_s():.3f}")
        frame = build_live_frame(self._scenario_id, self._scenario_label, rec, self._t_total)
        _append_jsonl(frame, self._live_path)


    def on_scenario_end(self, result) -> None:
        """
        Call once after a scenario runner completes.

        Closes the checkpoint file handle opened in on_scenario_start(), if
        one is open, then appends a terminal "scenario_complete" event.
        """
        if self._checkpoint_fh is not None:
            self._checkpoint_fh.close()
            self._checkpoint_fh = None

        if self._elapsed_path is not None:
            self._elapsed_path.write_text(f"{self.cumulative_elapsed_s():.3f}")

        if self._live_path is None:
            return
        _append_jsonl(
            {
                "event":       "scenario_complete",
                "scenario_id": self._scenario_id,
                "elapsed_s":   result.elapsed_s if result else None,
            },
            self._live_path,
        )


# ===========================================================================
# 7.  TOP-LEVEL PUBLISH FUNCTION
# ===========================================================================

def publish_topology_and_profiles(
        net,
        profiles:    dict,
        output_dir:  str | Path = "publisher_output",
        network_id:  str = "",
) -> dict[str, Path]:
    """
    Write topology.json and profiles.json — the two files that don't depend
    on any scenario result. Call this once, right after the network is
    loaded/built, BEFORE any scenario runs. This is the first of three
    independent write points that together replace the old monolithic
    publish_result() — splitting the writes both for crash-resilience (a
    scenario crash no longer loses topology/profiles too) and to avoid a
    single large write burst at the very end of a long run.

    Returns
    -------
    dict mapping "topology" -> topology.json path, "profiles" -> profiles.json path.
    """
    out = Path(output_dir)
    written: dict[str, Path] = {}

    topo_path = out / "topology.json"
    topo      = build_topology(net)
    topo["network_id"] = network_id
    _dump(topo, topo_path)
    written["topology"] = topo_path

    prof_path = out / "profiles.json"
    prof      = build_profiles_payload(profiles, network_id=network_id)
    _dump(prof, prof_path)
    written["profiles"] = prof_path

    logger.info("publisher: wrote %d system-level files to %s", len(written), out)
    return written


def publish_scenario_result(
        scenario_result,
        output_dir:  str | Path = "publisher_output",
) -> Optional[Path]:
    """
    Write ONE scenario's timeseries JSON, immediately after that scenario
    finishes. Call this from inside run_benchmark()'s scenario loop, right
    after results[n] = result — not batched with the others at the end.

    Written compact (indent=None), not pretty-printed: for a full annual
    scenario this file has ~100k+ timeseries rows, and indentation was
    roughly doubling the size of the single write burst that happens right
    as a scenario completes, for no informational benefit — nothing reads
    this file that needs the formatting.

    Returns the written Path, or None if scenario_result is None (e.g. a
    skipped LV scenario).
    """
    if scenario_result is None:
        return None
    out  = Path(output_dir)
    (out / "scenarios").mkdir(parents=True, exist_ok=True)
    sid  = scenario_result.scenario_id
    path = out / "scenarios" / f"{sid}.json"
    _dump(build_scenario_payload(scenario_result), path, indent=None)
    logger.info("publisher: wrote scenario/%s to %s", sid, path)
    return path


def publish_hc_and_comparison(
        result,
        output_dir:  str | Path = "publisher_output",
) -> dict[str, Path]:
    """
    Write hc.json (if present) and comparison.json — the two files that
    depend on the FULL set of scenario results being available. Call this
    once, after run_benchmark()'s scenario loop AND the HC step have both
    finished (i.e. right where publish_result() used to be called, but now
    only handling these two remaining pieces).

    Returns
    -------
    dict mapping "hc" -> hc.json path (if present), "comparison" -> comparison.json path (if present).
    """
    out = Path(output_dir)
    written: dict[str, Path] = {}

    if result.hc_results:
        hc_path = out / "hc.json"
        _dump(build_hc_payload(result.hc_results), hc_path)
        written["hc"] = hc_path

    if result.comparison_df is not None and not result.comparison_df.empty:
        cmp_path = out / "comparison.json"
        _dump(result.comparison_df.to_dict(orient="records"), cmp_path)
        written["comparison"] = cmp_path

    logger.info("publisher: wrote %d final-summary files to %s", len(written), out)
    return written