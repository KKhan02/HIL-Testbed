"""
network_plugin.py
=================
YAML-configured network plugins for the HIL benchmark framework.

Loads a pandapower network from a user-supplied file (or loader function),
builds time-series profiles with one of three strategies, and returns a
``(net, profiles)`` pair ready to pass straight to
``benchmark_runner.run_benchmark()``.  Pure addition — network_catalogue.py,
benchmark_runner.py, and profile_builder.py are untouched.  This mirrors the
controller plugin pattern in plugin_runner.py: a self-contained YAML + data
folder that students can drop next to the project without editing core files.

YAML format
-----------
::

    name: my_lv_feeder                      # required, unique id (used as network_id)
    label: "My Custom LV Feeder"            # optional, display name (defaults to name)
    source: json                            # required: json | pickle | function
    path: networks/my_feeder.json           # required for json/pickle sources
    # module: my_network_loader.py          # required for function source (a .py FILE PATH)
    # function: get_network                 # required for function source
    profiles:
      strategy: simbench_native             # simbench_native | dwd_pvlib | flat
      year: 2016                            # reference year for the time axis
      # data_dir: data/dwd                  # optional override for dwd_pvlib
    voltage_limits:                         # optional, forwarded to the benchmark
      v_min: 0.95
      v_max: 1.05
    notes: "Real LV feeder from Stadtwerke Oldenburg, anonymised"

All relative paths (``path``, ``module``, ``profiles.data_dir``) are resolved
relative to the YAML file's directory, so a plugin folder (YAML + network
file side by side) is self-contained and portable between laptop and RPi —
same convention as plugin_runner.py.

Source types
------------
json
    Load via ``pandapower.from_json(path)``.  RECOMMENDED.  pandapower JSON
    export (``pp.to_json(net, "networks/my_net.json")``) preserves ALL
    element tables — bus, line, trafo, sgen, load, switch, std_types, AND
    extra attributes such as SimBench's ``net.profiles`` dict — making it
    the recommended serialisation format for exchanging networks.
pickle
    Load via ``pandapower.from_pickle(path)``.  Always goes through
    pandapower's own serialisation function — never the raw ``pickle``
    module — so pandapower can handle version compatibility itself.
    Prefer JSON: pickles are Python/pandas-version fragile and opaque.
function
    Import a Python module from a file path and call a zero-argument
    function that returns a pandapowerNet.  Same importlib file-location
    pattern as the controller plugin (plugin_runner.py).

Profile strategies
------------------
simbench_native
    Calls ``sb.get_absolute_values()`` on the loaded net.  Requires the
    SimBench ``net.profiles`` metadata dict to be present (it survives
    ``pp.to_json`` round-trips).  If the metadata is absent the strategy
    WARNS and falls back to dwd_pvlib.  Post-processing mirrors
    profile_builder's SimBench path exactly: 15-min time axis rebuilt for
    the configured year, pv_mask including 'lv_res', night-time PV zeroing,
    lower-clipping of load/PV/wind.
dwd_pvlib
    Calls ``profile_builder.build_annual_profiles()`` on the fallback (DWD)
    path: DWD station 691 Bremen CSVs + pvlib Erbs decomposition + NOCT
    correction + oemof.demand BDEW 2025 SLPs.  Default data directory is
    ``<project_root>/data/dwd`` (same as run_benchmark_script.py).
flat
    Constant profiles at rated capacity (``net.load.p_mw`` for loads,
    ``net.sgen.p_mw`` for PV/wind) over a full year at 15-min resolution.
    Useful for worst-case testing without any weather data: every timestep
    is the simultaneous-peak snapshot.

Usage
-----
::

    from network_plugin import load_network_from_yaml, validate_network_plugin
    from benchmark_runner import BenchmarkConfig, run_benchmark

    net, profiles = load_network_from_yaml("example_networks/custom_lv_flat.yaml")
    for w in validate_network_plugin(net, profiles):
        print("WARNING:", w)

    result = run_benchmark(net, profiles,
                           network_id=profiles["plugin_meta"]["name"],
                           config=BenchmarkConfig(dry_run=True))

Returned profiles dict
----------------------
Same schema as profile_builder.build_annual_profiles() — consumed unchanged
by adapt_profiles(), slice_profiles(), publisher, and the plotting stack:

    "load"         pd.DataFrame  index=times  cols=load indices   [MW]
    "pv"           pd.DataFrame  index=times  cols=sgen indices   [MW]
    "wind"         pd.DataFrame  index=times  cols=sgen indices   [MW]
    "times"        pd.DatetimeIndex
    "extreme_days" dict (max_der / min_der / max_load / min_load)
    "net_type"     str
    "plugin_meta"  dict — ADDITIVE key carrying the parsed YAML metadata
                   (name, label, source, strategy, v_min, v_max, notes).
                   slice_profiles() passes non-indexed values through
                   unchanged and adapt_profiles() reads only the four
                   required keys, so this extra key is invisible to all
                   existing consumers.
"""

from __future__ import annotations

import calendar
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from profile_builder import (
    SIMBENCH_IDENTIFIERS,
    build_annual_profiles,
    detect_network_type,
    find_extreme_days,
)

logger = logging.getLogger(__name__)

# Project root convention — identical to run_benchmark_script.py:
# this file lives at the project root, _ROOT is its parent's parent so the
# shared data/ and outputs/ trees resolve the same way in both entry points.
_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DWD_DIR = _ROOT / "data" / "dwd"

_VALID_SOURCES    = ("json", "pickle", "function")
_VALID_STRATEGIES = ("simbench_native", "dwd_pvlib", "flat", "custom")

# Recognised DER type strings for validate_network_plugin().  Matched
# case-insensitively so both the SimBench convention ("PV", "WKA", "lv_res")
# and the CIGRE/manual convention ("pv", "wind", "WP") pass.
_RECOGNISED_SGEN_TYPES = frozenset(
    t.lower() for t in ("PV", "WKA", "lv_res", "pv", "wind", "wp", "solar")
)

# Masks consistent with profile_builder's SimBench path.
_PV_TYPE_PATTERN   = "pv|solar|lv_res"
_WIND_TYPE_PATTERN = "wind|wp|wka"

V_MV_LOW,  V_MV_HIGH = 1.0, 36.0     # MV band [kV]
V_LV_HIGH            = 1.0           # LV: below this [kV]


# ===========================================================================
# 1.  YAML parsing and validation
# ===========================================================================

def _load_yaml_config(yaml_path: Path) -> dict:
    """Read the network YAML and normalise it into a validated config dict."""
    try:
        import yaml
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError(
            "PyYAML is required for network plugins. "
            "Install it with:  pip install pyyaml   (inside hil_env on the RPi)."
        ) from exc

    if not yaml_path.is_file():
        raise FileNotFoundError(f"Network YAML not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Network YAML {yaml_path} must contain a top-level mapping, "
            f"got {type(raw).__name__}."
        )

    # ---- name / label -----------------------------------------------------
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"'name' is required and must be a non-empty string "
                         f"({yaml_path}).")
    name  = name.strip()
    label = str(raw.get("label", name)).strip() or name

    # ---- source -----------------------------------------------------------
    source = str(raw.get("source", "")).strip().lower()
    if source not in _VALID_SOURCES:
        raise ValueError(
            f"'source' must be one of {list(_VALID_SOURCES)}, got "
            f"{raw.get('source')!r} ({yaml_path})."
        )

    cfg: dict = {
        "name":   name,
        "label":  label,
        "source": source,
        "notes":  str(raw.get("notes", "")),
        "yaml_dir": yaml_path.parent,
    }

    # ---- path (json / pickle) --------------------------------------------
    if source in ("json", "pickle"):
        path_field = raw.get("path")
        if not isinstance(path_field, str) or not path_field.strip():
            raise ValueError(
                f"'path' is required for source '{source}' ({yaml_path})."
            )
        net_path = Path(path_field.strip()).expanduser()
        if not net_path.is_absolute():
            net_path = yaml_path.parent / net_path
        net_path = net_path.resolve()
        if not net_path.is_file():
            raise FileNotFoundError(
                f"Network file not found: {net_path} "
                f"(resolved from 'path: {path_field}' relative to "
                f"{yaml_path.parent})."
            )
        cfg["path"] = net_path

    # ---- module / function (function source) ------------------------------
    if source == "function":
        # 'module' is the canonical key (matches plugin_runner.py); 'path'
        # is accepted as an alias so the YAML schema stays uniform.
        module_field = raw.get("module", raw.get("path"))
        if not isinstance(module_field, str) or not module_field.strip():
            raise ValueError(
                f"'module' (a .py file path) is required for source "
                f"'function' ({yaml_path})."
            )
        module_path = Path(module_field.strip()).expanduser()
        if not module_path.is_absolute():
            module_path = yaml_path.parent / module_path
        module_path = module_path.resolve()
        if module_path.suffix != ".py":
            raise ValueError(f"'module' must point to a .py file, got: "
                             f"{module_path}")
        if not module_path.is_file():
            raise FileNotFoundError(
                f"Loader module not found: {module_path} (resolved relative "
                f"to {yaml_path.parent})."
            )
        function = raw.get("function")
        if not isinstance(function, str) or not function.isidentifier():
            raise ValueError(
                f"'function' must be a valid Python identifier for source "
                f"'function', got {function!r} ({yaml_path})."
            )
        cfg["module_path"] = module_path
        cfg["function"]    = function

    # ---- profiles ----------------------------------------------------------
    prof_raw = raw.get("profiles", {}) or {}
    if not isinstance(prof_raw, dict):
        raise ValueError(f"'profiles' must be a mapping ({yaml_path}).")
    strategy = str(prof_raw.get("strategy", "")).strip().lower()
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"profiles.strategy must be one of {list(_VALID_STRATEGIES)}, "
            f"got {prof_raw.get('strategy')!r} ({yaml_path})."
        )
    if strategy == "custom" and not prof_raw.get("file_map"):
        raise ValueError(
            f"profiles.strategy='custom' requires profiles.file_map "
            f"(and normally profiles.data_dir) — got neither ({yaml_path})."
        )
    year = prof_raw.get("year", 2016)
    if not isinstance(year, int) or not (1990 <= year <= 2100):
        raise ValueError(f"profiles.year must be an integer year, got "
                         f"{year!r} ({yaml_path}).")

    data_dir = prof_raw.get("data_dir")
    if data_dir is not None:
        data_dir = Path(str(data_dir)).expanduser()
        if not data_dir.is_absolute():
            data_dir = yaml_path.parent / data_dir
        data_dir = data_dir.resolve()
    cfg["strategy"] = strategy
    cfg["year"]     = year
    cfg["data_dir"] = data_dir            # None -> project default data/dwd
    cfg["file_map"] = prof_raw.get("file_map")
    cfg["col_map"]  = prof_raw.get("col_map")

    # ---- voltage limits (optional) -----------------------------------------
    vlim = raw.get("voltage_limits", {}) or {}
    if not isinstance(vlim, dict):
        raise ValueError(f"'voltage_limits' must be a mapping ({yaml_path}).")
    v_min = float(vlim.get("v_min", 0.95))
    v_max = float(vlim.get("v_max", 1.05))
    if not (0.5 <= v_min < v_max <= 1.5):
        raise ValueError(
            f"voltage_limits invalid: require 0.5 <= v_min < v_max <= 1.5, "
            f"got v_min={v_min}, v_max={v_max} ({yaml_path})."
        )
    cfg["v_min"], cfg["v_max"] = v_min, v_max
    return cfg


# ===========================================================================
# 2.  Network loading (three sources)
# ===========================================================================

def _import_loader_fn(module_path: Path, function_name: str):
    """
    Import ``function_name`` from the Python file at ``module_path`` using
    importlib file-location machinery — the same pattern as
    plugin_runner._import_controller_fn().  The module is registered in
    sys.modules under a mangled unique name so two loader files sharing a
    stem in different folders do not clash.
    """
    mod_name = (f"_hil_network_{module_path.stem}_"
                f"{abs(hash(str(module_path))) & 0xFFFFFF:06x}")

    spec = importlib.util.spec_from_file_location(mod_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build an import spec for {module_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise

    if not hasattr(module, function_name):
        raise AttributeError(
            f"Network loader module {module_path} has no attribute "
            f"'{function_name}'. Available callables: "
            f"{[n for n in dir(module) if not n.startswith('_') and callable(getattr(module, n))]}."
        )
    fn = getattr(module, function_name)
    if not callable(fn):
        raise TypeError(f"'{function_name}' in {module_path} is not callable "
                        f"({type(fn).__name__}).")
    return fn


def _load_net(cfg: dict):
    """Load the pandapower network according to cfg['source']."""
    import pandapower as pp

    source = cfg["source"]
    if source == "json":
        logger.info("network_plugin: loading pandapower JSON  %s", cfg["path"])
        net = pp.from_json(str(cfg["path"]))
    elif source == "pickle":
        # Always through pandapower's own serialisation function — never the
        # raw pickle module — so pandapower handles its own compatibility.
        logger.info("network_plugin: loading pandapower pickle %s", cfg["path"])
        net = pp.from_pickle(str(cfg["path"]))
    else:  # function
        logger.info("network_plugin: importing %s::%s",
                    cfg["module_path"], cfg["function"])
        fn  = _import_loader_fn(cfg["module_path"], cfg["function"])
        net = fn()

    if not isinstance(net, pp.pandapowerNet):
        raise TypeError(
            f"Network source '{source}' did not produce a pandapowerNet "
            f"(got {type(net).__name__}). For source 'function' the loader "
            f"must be a zero-argument function returning a pandapowerNet."
        )
    return net


# ===========================================================================
# 3.  Profile strategies
# ===========================================================================

def _has_simbench_metadata(net) -> bool:
    """True when the SimBench ``net.profiles`` dict is present and non-empty."""
    prof = net.get("profiles", None) if hasattr(net, "get") else None
    return isinstance(prof, dict) and len(prof) > 0 and any(
        isinstance(v, pd.DataFrame) and not v.empty for v in prof.values()
    )


def _annual_15min_index(year: int) -> pd.DatetimeIndex:
    """Full calendar year at 15-min resolution, Europe/Berlin (leap-aware)."""
    n_days = 366 if calendar.isleap(year) else 365
    return pd.date_range(start=f"{year}-01-01", periods=n_days * 96,
                         freq="15min", tz="Europe/Berlin")


def _build_profiles_simbench_native(net, cfg: dict) -> dict:
    """
    Native SimBench profiles via sb.get_absolute_values() on the loaded net.

    Mirrors profile_builder's SimBench path exactly (pv_mask including
    'lv_res', night-time PV zeroing, lower clipping, DatetimeIndex
    reconstruction) but operates on the ALREADY-LOADED net instead of
    re-downloading via a simbench_code — that is what makes a JSON-exported
    SimBench network portable.
    """
    import simbench as sb

    logger.info("network_plugin: building profiles via sb.get_absolute_values()")
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)

    times = profiles[("load", "p_mw")].index
    # SimBench returns an integer step index (0..N-1), not timestamps.
    # Reconstruct the DatetimeIndex for the configured reference year
    # (SimBench's internal reference year is 2016 — a leap year, 35136 steps).
    if not isinstance(times, pd.DatetimeIndex):
        times = pd.date_range(start=f"{cfg['year']}-01-01", periods=len(times),
                              freq="15min", tz="Europe/Berlin")

    load_df = profiles[("load", "p_mw")].copy()
    load_df.index = times
    load_df = load_df.clip(lower=0.0)

    sgen_prof = profiles[("sgen", "p_mw")].copy()
    sgen_prof.index = times

    pv_mask   = net.sgen["type"].str.lower().str.contains(_PV_TYPE_PATTERN,   na=False)
    wind_mask = net.sgen["type"].str.lower().str.contains(_WIND_TYPE_PATTERN, na=False)
    pv_idx,  wind_idx = net.sgen[pv_mask].index, net.sgen[wind_mask].index

    pv_df = sgen_prof[[i for i in pv_idx if i in sgen_prof.columns]].copy()
    # Zero out physically impossible night-time PV generation (SimBench
    # commercial/semiurb networks contain small non-zero values at night).
    night_mask = (pv_df.index.hour >= 22) | (pv_df.index.hour <= 4)
    pv_df.loc[night_mask] = 0.0

    wind_df = sgen_prof[[i for i in wind_idx if i in sgen_prof.columns]].copy()
    pv_df, wind_df = pv_df.clip(lower=0.0), wind_df.clip(lower=0.0)

    result = {"load": load_df, "pv": pv_df, "wind": wind_df,
              "times": times, "net_type": "simbench"}
    result["extreme_days"] = find_extreme_days(result)
    logger.info("network_plugin: simbench_native done — %d timesteps | "
                "%d loads | %d PV | %d wind", len(times),
                load_df.shape[1], pv_df.shape[1], wind_df.shape[1])
    return result


def _dwd_safe_name(name: str) -> str:
    """
    Return a net_name that routes build_annual_profiles() onto its DWD
    (cigre/fallback) path even when the user's plugin name happens to
    contain a SimBench identifier (e.g. 'simbench_rural_export').
    profile_builder routes purely on the name string, so a colliding name
    would raise 'simbench_code must be provided'.
    """
    if detect_network_type(name) != "simbench":
        return name
    candidate = name.lower()
    for ident in SIMBENCH_IDENTIFIERS:
        candidate = candidate.replace(ident, ident.replace("-", "_")
                                                   .replace("simbench", "sb"))
    if detect_network_type(candidate) == "simbench":       # belt and braces
        candidate = "plugin_custom_net"
    logger.info("network_plugin: net_name '%s' collides with SimBench "
                "detection — using '%s' for the DWD profile path.",
                name, candidate)
    return candidate


def _build_profiles_dwd(net, cfg: dict) -> dict:
    """DWD station 691 Bremen + pvlib + BDEW path via build_annual_profiles()."""
    data_dir = cfg["data_dir"] or _DEFAULT_DWD_DIR
    logger.info("network_plugin: building profiles via DWD/pvlib path "
                "(data_dir=%s)", data_dir)
    return build_annual_profiles(
        net,
        net_name      = _dwd_safe_name(cfg["name"]),
        data_dir      = str(data_dir),
        simbench_code = None,
        file_map      = cfg.get("file_map"),
        col_map       = cfg.get("col_map"),
    )


def _build_profiles_flat(net, cfg: dict) -> dict:
    """
    Constant profiles at rated capacity — worst-case simultaneity snapshot
    repeated over a full year at 15-min resolution.  Loads run at
    net.load.p_mw, PV and wind sgens at net.sgen.p_mw, on every timestep.
    No weather data required.
    """
    times = _annual_15min_index(cfg["year"])
    logger.info("network_plugin: building FLAT profiles (%d timesteps, "
                "year %d)", len(times), cfg["year"])

    load_df = pd.DataFrame(
        np.tile(net.load["p_mw"].values.astype(float), (len(times), 1)),
        index=times, columns=net.load.index,
    ).clip(lower=0.0)

    pv_mask   = net.sgen["type"].str.lower().str.contains(_PV_TYPE_PATTERN,   na=False)
    wind_mask = net.sgen["type"].str.lower().str.contains(_WIND_TYPE_PATTERN, na=False)
    pv_idx,  wind_idx = net.sgen[pv_mask].index, net.sgen[wind_mask].index

    def _const_frame(idx) -> pd.DataFrame:
        if len(idx) == 0:
            return pd.DataFrame(index=times, dtype=float)
        rated = net.sgen.loc[idx, "p_mw"].values.astype(float)
        return pd.DataFrame(np.tile(rated, (len(times), 1)),
                            index=times, columns=idx).clip(lower=0.0)

    pv_df, wind_df = _const_frame(pv_idx), _const_frame(wind_idx)

    result = {"load": load_df, "pv": pv_df, "wind": wind_df,
              "times": times, "net_type": "flat"}
    result["extreme_days"] = find_extreme_days(result)
    logger.info("network_plugin: flat done — %d loads | %d PV | %d wind",
                load_df.shape[1], pv_df.shape[1], wind_df.shape[1])
    return result


def _build_profiles_for_strategy(net, cfg: dict) -> Tuple[dict, str]:
    """
    Dispatch to the configured profile strategy for the given net,
    applying the simbench_native -> dwd_pvlib fallback when the net
    carries no SimBench metadata.  Returns (profiles, strategy_used).

    Shared by load_network_from_yaml() and make_profile_factory() so the
    HC-stressed re-benchmark rebuilds profiles with EXACTLY the same
    strategy and fallback semantics as the outer run.
    """
    strategy = cfg["strategy"]
    if strategy == "simbench_native" and not _has_simbench_metadata(net):
        logger.warning(
            "network_plugin: profiles.strategy=simbench_native requested but "
            "the loaded net has no SimBench 'net.profiles' metadata "
            "(was it exported from a SimBench net with pp.to_json?). "
            "Falling back to dwd_pvlib."
        )
        strategy = "dwd_pvlib"

    if strategy == "simbench_native":
        profiles = _build_profiles_simbench_native(net, cfg)
    elif strategy in ("dwd_pvlib", "custom"):
        profiles = _build_profiles_dwd(net, cfg)
    else:
        profiles = _build_profiles_flat(net, cfg)
    return profiles, strategy


# ===========================================================================
# 4.  Public API — load_network_from_yaml
# ===========================================================================

def load_network_from_yaml(yaml_path) -> Tuple[object, dict]:
    """
    Load a network plugin YAML and return ``(net, profiles)`` ready for
    ``run_benchmark()``.

    Parameters
    ----------
    yaml_path : str | Path
        Path to the network YAML config (format in the module docstring).

    Returns
    -------
    (net, profiles)
        net      : pandapowerNet loaded from the configured source.
        profiles : dict with the profile_builder schema ("load", "pv",
                   "wind", "times", "extreme_days", "net_type") plus an
                   additive "plugin_meta" dict carrying name, label, source,
                   strategy, v_min, v_max, and notes from the YAML.

    Notes
    -----
    pandapower JSON export (``pp.to_json(net, "networks/my_net.json")``)
    preserves all element tables (including std_types and extra attributes
    such as SimBench's ``net.profiles``) and is the RECOMMENDED
    serialisation format.  Pickle loading always goes through
    ``pandapower.from_pickle()`` — never the raw pickle module.

    simbench_native fallback
        If ``profiles.strategy: simbench_native`` is requested but the
        loaded net carries no SimBench ``net.profiles`` metadata (e.g. the
        JSON was exported from a non-SimBench net, or the metadata was
        stripped), a warning is logged and the strategy falls back to
        dwd_pvlib.
    """
    yaml_path = Path(yaml_path).expanduser().resolve()
    cfg = _load_yaml_config(yaml_path)
    logger.info("network_plugin: loading '%s' (%s) — source=%s, strategy=%s",
                cfg["name"], cfg["label"], cfg["source"], cfg["strategy"])

    net = _load_net(cfg)
    profiles, strategy = _build_profiles_for_strategy(net, cfg)

    profiles["plugin_meta"] = {
        "name":      cfg["name"],
        "label":     cfg["label"],
        "source":    cfg["source"],
        "strategy":  strategy,           # the strategy actually used
        "requested_strategy": cfg["strategy"],
        "year":      cfg["year"],
        "v_min":     cfg["v_min"],
        "v_max":     cfg["v_max"],
        "notes":     cfg["notes"],
        "yaml_path": str(yaml_path),
    }
    return net, profiles


# ===========================================================================
# 5.  Public API — validate_network_plugin
# ===========================================================================

def validate_network_plugin(net, profiles: dict) -> list[str]:
    """
    Check a loaded (net, profiles) pair for benchmark-framework
    compatibility.  Returns a list of human-readable warning strings —
    empty when all checks pass.  Warnings, not exceptions: some networks
    (e.g. DER-free Kerber feeders for HC-only analysis) legitimately fail
    individual checks, and the caller decides whether to proceed.
    """
    warnings: list[str] = []

    # ---- 1. sgen table non-empty -------------------------------------------
    if len(net.sgen) == 0:
        warnings.append(
            "net.sgen is empty — Scenario 4 (Volt-Var) has no DERs to "
            "control. Only Scenarios 1-3 are meaningful; skip 4/5 via "
            "BenchmarkConfig(scenarios=[1, 2, 3])."
        )
    else:
        # ---- 2. at least one recognised DER type ---------------------------
        types = net.sgen["type"].astype(str).str.strip().str.lower()
        if not types.isin(_RECOGNISED_SGEN_TYPES).any():
            warnings.append(
                "No sgen has a recognised type in "
                "['PV', 'WKA', 'lv_res', 'pv', 'wind'] — profile_builder's "
                "pv/wind masks will match nothing and der_p will be empty. "
                f"Types found: {sorted(net.sgen['type'].astype(str).unique())}. "
                "Set net.sgen['type'] before export."
            )

    # ---- 3. voltage level in MV (1.0-36.0 kV) or LV (< 1.0 kV) --------------
    vn = net.bus["vn_kv"].astype(float)
    in_mv = ((vn >= V_MV_LOW) & (vn <= V_MV_HIGH)).any()
    in_lv = (vn < V_LV_HIGH).any()
    if not (in_mv or in_lv):
        warnings.append(
            f"net.bus.vn_kv contains no bus in the MV range "
            f"({V_MV_LOW}-{V_MV_HIGH} kV) or LV range (< {V_LV_HIGH} kV) — "
            f"levels present: {sorted(vn.unique())}. The benchmark targets "
            f"distribution networks; HV-only networks are out of scope."
        )

    # ---- 4. at least one transformer -----------------------------------------
    n_trafo   = len(net.trafo)
    n_trafo3w = len(net.get("trafo3w", [])) if hasattr(net, "get") else 0
    if n_trafo == 0 and n_trafo3w == 0:
        warnings.append(
            "Network has no transformer (net.trafo and net.trafo3w are "
            "empty) — Scenario 2 (OLTC) cannot run and the network has no "
            "HV/MV or MV/LV coupling point."
        )
    elif n_trafo == 0:
        warnings.append(
            "net.trafo is empty (only three-winding trafo3w present) — "
            "Scenario 2 (OLTC) operates on net.trafo.tap_pos and will find "
            "no two-winding transformer to control."
        )

    # ---- 5. voltage_depend_loads=False compatibility --------------------------
    for col in ("const_i_percent", "const_z_percent"):
        if col in net.load.columns:
            vals = pd.to_numeric(net.load[col], errors="coerce").fillna(0.0)
            n_bad = int((vals != 0.0).sum())
            if n_bad:
                warnings.append(
                    f"{n_bad} load(s) have non-zero net.load.{col} — these "
                    f"model voltage-dependent (ZIP) loads, but ALL runpp() "
                    f"calls in this framework use voltage_depend_loads=False "
                    f"(mandatory pandapower 3.2.0+ workaround). The ZIP "
                    f"shares will be silently ignored; results assume "
                    f"constant-power loads."
                )

    # ---- 6. (additive) profile/net index alignment ------------------------------
    if isinstance(profiles, dict):
        load_cols = getattr(profiles.get("load", None), "columns", pd.Index([]))
        orphan_loads = [c for c in load_cols if c not in net.load.index]
        if orphan_loads:
            warnings.append(
                f"profiles['load'] contains {len(orphan_loads)} column(s) "
                f"not present in net.load.index (e.g. {orphan_loads[:5]}) — "
                f"adapt_profiles() will raise a KeyError."
            )
        n_der_cols = sum(
            getattr(profiles.get(k, None), "shape", (0, 0))[1]
            for k in ("pv", "wind")
        )
        if len(net.sgen) > 0 and n_der_cols == 0:
            warnings.append(
                "profiles contain zero PV and zero wind columns despite "
                "net.sgen being non-empty — check net.sgen['type'] values "
                "against the pv/wind masks (pv|solar|lv_res, wind|wp)."
            )
        for key in ("load", "pv", "wind"):
            df = profiles.get(key, None)
            if isinstance(df, pd.DataFrame) and not df.empty and \
                    bool(df.isna().any().any()):
                warnings.append(
                    f"profiles['{key}'] contains NaN values — runpp() will "
                    f"fail or silently mis-solve on affected timesteps."
                )

    return warnings


# ===========================================================================
# 6.  Public API — make_profile_factory
# ===========================================================================

def make_profile_factory(yaml_path):
    """
    Return a ``callable(net) -> profiles dict`` that rebuilds profiles for
    ANY network using the strategy configured in the plugin YAML.

    Purpose: BenchmarkConfig.profile_factory for the HC-stressed
    re-benchmark (``run_hc_scenarios=True``).  Hosting-capacity analysis
    adds new sgens to a copy of the network; the re-benchmark then needs
    fresh profiles that COVER those new sgens.  The built-in script wiring
    hardcodes build_annual_profiles() (the DWD path) for this — wrong for a
    flat- or simbench_native-strategy plugin.  This factory closes that gap:
    the stressed net gets profiles built with exactly the same strategy,
    year, data_dir, and fallback semantics as the outer run.

    Strategy behaviour on the HC-stressed net
    -----------------------------------------
    flat            : new HC sgens get constant rated-capacity columns
                      automatically (the frame is rebuilt from net.sgen).
    dwd_pvlib       : new HC sgens get pvlib/BDEW columns automatically.
    simbench_native : the stressed net is a deepcopy of the plugin net, so
                      the SimBench metadata survives — but HC sgens have no
                      native profile columns (they did not exist in the
                      SimBench dataset).  This matches the built-in
                      SimBench HC behaviour; the standard fallback check
                      still applies if metadata is somehow absent.

    Usage
    -----
    ::

        from network_plugin import load_network_from_yaml, make_profile_factory

        net, profiles = load_network_from_yaml("my_net.yaml")
        config = BenchmarkConfig(
            run_hc           = True,
            run_hc_scenarios = True,
            profile_factory  = make_profile_factory("my_net.yaml"),
        )
    """
    yaml_path = Path(yaml_path).expanduser().resolve()
    cfg = _load_yaml_config(yaml_path)      # parse once, at factory creation

    def _factory(net_hc) -> dict:
        profiles, used = _build_profiles_for_strategy(net_hc, cfg)
        profiles["plugin_meta"] = {
            "name":               cfg["name"] + "_hc_stressed",
            "label":              cfg["label"],
            "source":             cfg["source"],
            "strategy":           used,
            "requested_strategy": cfg["strategy"],
            "year":               cfg["year"],
            "v_min":              cfg["v_min"],
            "v_max":              cfg["v_max"],
            "notes":              cfg["notes"],
            "yaml_path":          str(yaml_path),
        }
        return profiles

    return _factory
