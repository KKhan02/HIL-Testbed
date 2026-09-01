from dataclasses import dataclass, asdict, field
import uuid

@dataclass
class ParameterConfig:
    """
    Stores all numerical simulation constraint and scaling parameters for a study run.

    Voltage limits follow EN 50160 / VDE-AR-N 4100 planning limits (0.95–1.05 pu).
    Thermal limits are expressed as a percentage of rated current (100% = rated).
    Angle limit follows VDE-AR-N 4110 operational guidance (30°).
    Unbalance limit follows IEC 62749 (2.0%).

    Scaling convention for der_scaling and load_scaling
    ----------------------------------------------------
    Both fields use a dict mapping bus index → scaling factor.
    The key None acts as a sentinel meaning "apply to the entire network."

        {None: 1.5}           → scale all DERs / loads by 1.5
        {12: 0.8, 15: 0.8}   → scale only buses 12 and 15
        {None: 1.5, 12: 0.8} → scale all by 1.5, override bus 12 to 0.8

    The wizard asks the user whether to target the entire network, a single bus,
    or a set of buses, then builds the dict accordingly. This class stores the result.
    """
    v_min: float = 0.95 # Lower voltage planning limit in per-unit. Buses below this are in violation.
    v_max: float = 1.05 # Upper voltage planning limit in per-unit. Buses above this are in violation.
    line_max_loading: float = 100 # Maximum thermal loading for lines as a percentage of rated current.
    trafo_max_loading: float = 100 # Maximum thermal loading for transformers as a percentage of rated current.
    va_diff_max_degree: float = 30 # Maximum permitted voltage angle difference across any single line, in degrees.
    unbalance_max_percent: float = 2.0 # Maximum voltage unbalance between phases, as a percentage.
    der_scaling: dict = field(default_factory=lambda: {None: 1.0}) # DER (sgen) active power scaling per bus. Key None = entire network.
    load_scaling: dict = field(default_factory=lambda: {None: 1.0}) # Load active power scaling per bus. Key None = entire network.
    timestep_resolution: int = 15 # Time resolution of the simulation in minutes. SimBench native profiles are 15-min

    # Optional per-run Q(V) characteristic overrides (VDE-AR-N 4110 Bild 8).
    # None = use the framework defaults in volt_var_controller.py
    # (Q_RATIO=0.25, U1..U4 = 0.96/0.99/1.01/1.04).
    # When set, the executor applies them via
    # volt_var_controller.set_qv_parameters(), which updates the dry-run
    # curve, the coordinator q_max sizing, AND the CFG: message pushed to
    # the Arduino at session startup — one value, three consumers.
    q_ratio: float = None # Q_max / P_installed ratio, must satisfy 0 < q <= 1
    u1_pu:   float = None # lower saturation breakpoint [pu]
    u2_pu:   float = None # deadband lower edge [pu]
    u3_pu:   float = None # deadband upper edge [pu]
    u4_pu:   float = None # upper saturation breakpoint [pu]

    def to_dict(self) -> dict:
        """
        Serialises this ParameterConfig to a plain dictionary for JSON output.

        Called by RunPlan.to_dict() via dataclasses.asdict(), which recursively
        converts all nested dataclasses. This method exists for direct use when
        only ParameterConfig needs to be serialised in isolation.

        Returns
        -------
        dict
            All fields as key-value pairs. None values are preserved so that
            from_dict() can reconstruct the exact object on reload.
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ParameterConfig":
        """
        Reconstructs a ParameterConfig from a plain dictionary.

        Used when loading a saved RunPlan preset from disk. Every field uses
        .get() with its original default so that older preset files missing
        a newly added field do not raise KeyErrors on load.

        The scaling fields fall back to {None: 1.0} (no scaling applied) rather
        than None, because None would cause a TypeError in the executor when it
        attempts to iterate over the scaling dict.

        Parameters
        ----------
        data : dict
            Raw dictionary, typically the "parameters" sub-dict parsed from JSON.

        Returns
        -------
        ParameterConfig
            Fully reconstructed instance. Missing fields receive their defaults.
        """

        return cls(
            v_min = data.get("v_min",0.95),
            v_max = data.get("v_max",1.05),
            line_max_loading = data.get("line_max_loading",100),
            trafo_max_loading = data.get("trafo_max_loading",100),
            va_diff_max_degree = data.get("va_diff_max_degree", 30),
            unbalance_max_percent = data.get("unbalance_max_percent",2.0),
            # Plain dict literals used here — field(default_factory=...) is only
            # valid at class definition time, not inside method calls.
            der_scaling  = {(None if k == "null" else k): v for k, v in data.get("der_scaling",  {None: 1.0}).items()},
            load_scaling = {(None if k == "null" else k): v for k, v in data.get("load_scaling", {None: 1.0}).items()},
            timestep_resolution = data.get("timestep_resolution",15),
            q_ratio = data.get("q_ratio", None),
            u1_pu   = data.get("u1_pu", None),
            u2_pu   = data.get("u2_pu", None),
            u3_pu   = data.get("u3_pu", None),
            u4_pu   = data.get("u4_pu", None),
        )


@dataclass
class NetworkConfig:
    """
    Specifies how to locate and load the pandapower network for a study run.

    This is not a description of the network itself — it records the user's
    source selection made during the CLI wizard. The executor reads this config
    and calls the appropriate loading function at runtime.

    Three mutually exclusive source types are supported:

        "preset"        : A named network from the built-in catalogue.
                          The executor calls PRESET_CATALOGUE[preset_name]().
                          Covers SimBench, CIGRE, Kerber, Dickert, Synthetic LV.

        "simbench_code" : A raw SimBench code assembled by the 4-question wizard
                          (voltage level → grid type → DER level → switching).
                          The executor calls sb.get_simbench_net(simbench_code).
                          SimBench codes never appear in PRESET_CATALOGUE.

        "custom"        : A user-provided Python file containing a function that
                          returns a pandapowerNet object. The executor imports
                          custom_path and calls custom_function_name().

    A fourth source type connects the research-lab plugin system:

        "plugin"        : A YAML-configured network plugin (network_plugin.py).
                          The executor calls load_network_from_yaml(plugin_path),
                          which returns BOTH the network and its profiles —
                          for this source type DatasetConfig is ignored (the
                          YAML's profiles.strategy owns profile building).
    """
    source_type: str = None # Which loading strategy to use i.e. preset, simbench_code, custom
    preset_name: str = None # Name of the built-in preset network. Used only when source_type == "preset"
    simbench_code: str = None # Raw SimBench network code entered by user (source_type == "simbench_code")
    custom_path: str = None # Absolute or relative path to the user-provided pandapower Net python file
    custom_function_name: str = "get_network" # Name of the function inside custom_path to call.
    preset_family: str = None # Human-readable network family name for display & UI labelling
    simbench_selections: dict = None # Raw wizard selections used to assemble the Simbench code.
    plugin_path: str = None # Path to a network-plugin YAML. Used only when source_type == "plugin"

    # Optional pre-run network modifications, applied by the executor in this
    # order BEFORE profile building (so injected DERs receive profile columns):
    #   1. der_placements   2. switches_to_flip
    # der_placements: list of {"bus": int, "p_mw": float, "sn_mva": float}
    # dicts — each creates one PV sgen (type="PV") at the named bus, the same
    # pattern documented in run_benchmark_script.py's DER-injection block.
    # switches_to_flip: list of net.switch indices whose closed state is
    # toggled (open->closed or closed->open) — mirrors the script's
    # switch-manipulation block (e.g. sectionalizers or tie switches),
    # but as a toggle rather than an unconditional open.
    der_placements: list = None
    switches_to_flip: list = None

    def to_dict(self) -> dict:
        """
        Serialises this NetworkConfig to a plain dictionary for JSON output.

        Used when saving a RunPlan preset to disk, either directly or via
        RunPlan.to_dict() which calls dataclasses.asdict() recursively.

        Returns
        -------
        dict
            All fields as key-value pairs. None values are preserved.
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "NetworkConfig":
        """
        Reconstructs a NetworkConfig from a plain dictionary.

        Used when loading a saved RunPlan preset from disk. Every field uses
        .get() with its original default so that older preset files missing
        a newly added field do not raise KeyErrors on load.

        Parameters
        ----------
        data : dict
            Raw dictionary, typically the "network" sub-dict parsed from JSON.

        Returns
        -------
        NetworkConfig
            Fully reconstructed instance. Missing fields receive their defaults.
        """
        return cls(
            source_type = data.get("source_type",None),
            preset_name = data.get("preset_name",None),
            simbench_code = data.get("simbench_code",None),
            custom_path = data.get("custom_path",None),
            custom_function_name = data.get("custom_function_name", "get_network"),
            preset_family = data.get("preset_family",None),
            simbench_selections = data.get("simbench_selections",None),
            plugin_path = data.get("plugin_path", None),
            der_placements = data.get("der_placements", None),
            switches_to_flip = data.get("switches_to_flip", data.get("switches_to_open", None)),
        )
    

@dataclass
class DatasetConfig:
    '''
    Specifies how to build or locate the time-series profiles for a study run.
    This is not a description of the dataset itself — it is the user's selection
    of profile source made using the CLI wizard.

    Three mutually exclusive source types are supported:

        "simbench_native" : Use the profiles embedded in the SimBench network
                            directly via sb.get_absolute_values(). Only valid
                            when NetworkConfig.source_type == "simbench_code".
                            No additional fields are required.

        "dwd"             : Build profiles from locally stored DWD CDC CSV files
                            using profile_builder.build_annual_profiles().
                            Requires: data_dir, station_id, year.

        "custom"          : User-supplied pre-built profile CSV (can be ERA-5
                            or any other profile). Needs custom file_map
                            and col_map for profile_builder
                            Requires: custom_path, file_map, col_map.

    Field relevance by source_type
    --------------------------------
        source_type       : all
        data_dir          : "dwd", "custom"
        station_id        : "dwd" only
        year              : "dwd" only
        file_map          : "custom" only
        col_map           : "custom" only
        custom_path       : "custom" only
    '''
    # Which profile-building strategy to use.
    # One of: "simbench_native", "dwd", "custom"
    source_type: str = None 
    
    # Root directory containing DWD or custom data files.
    # Expected subdirectory layout for DWD:
    #   <data_dir>/PV/       — RAD-G files  (J/cm²)
    #   <data_dir>/Wind/     — F files       (m/s)
    #   <data_dir>/Temperature/ — T2M files  (°C)
    data_dir: str = None 
    
    # DWD station identifier used to match filenames via glob.
    # Example: "691" matches "data_OBS_DEU_PT10M_691*.csv"
    # Used only when source_type == "dwd".
    station_id: str = None 

    # Calendar year of the dataset. Used to reconstruct the DatetimeIndex
    # as pd.Timestamp(f"{year}-01-01").
    # SimBench native profiles use 2016 (leap year) internally — this field
    # is ignored for source_type == "simbench_native".
    year: int = None 

    # Custom file map: maps profile_builder's internal keys to actual filenames.
    # Keys must include "solar", "wind", "temp".
    # Example: {"solar": "era5_solar.csv", "wind": "era5_wind.csv",
    #           "temp": "era5_temp.csv"}
    # Used only when source_type == "custom".
    file_map: dict = None

    # ERA5 column map: maps profile_builder's internal column keys to actual
    # column names present in the ERA5 CSV files.
    # Example: {"timestamp": "time", "solar": "ssrd", "wind": "si10",
    #           "temp": "t2m", "sep": ","}
    # Used only when source_type == "era5".
    col_map: dict = None

    # Absolute or relative path to a pre-built profile CSV file.
    # The executor loads this directly — profile_builder is not called.
    # Used only when source_type == "custom".
    custom_path: str = None 
    
    def to_dict(self) -> dict:
        """
        Serialises this DatasetConfig to a plain dictionary for JSON output.

        Used when saving a RunPlan preset to disk, either directly or via
        RunPlan.to_dict() which calls dataclasses.asdict() recursively.

        Returns
        -------
        dict
            All fields as key-value pairs. None values are preserved so that
            from_dict() can reconstruct the exact object on reload.
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "DatasetConfig":
        """
        Reconstructs a DatasetConfig from a plain dictionary.

        Used when loading a saved RunPlan preset from disk. Every field uses
        .get() with its original default so that older preset files missing
        a newly added field do not raise KeyErrors on load.

        Parameters
        ----------
        data : dict
            Raw dictionary, typically the "dataset" sub-dict parsed from JSON.

        Returns
        -------
        DatasetConfig
            Fully reconstructed instance. Missing fields receive their defaults.
        """
        return cls(
            source_type = data.get("source_type", None),
            data_dir    = data.get("data_dir", None),
            station_id  = data.get("station_id", None),
            year        = data.get("year", None),
            file_map    = data.get("file_map", None),
            col_map     = data.get("col_map", None),
            custom_path = data.get("custom_path", None),
        )
    
@dataclass
class RunPlan:

    """
    The single serialisable unit of configuration for one complete study run.

    RunPlan is the top-level container that ties together all configuration
    decisions made during the CLI wizard. It nests the three focused config
    classes and adds run-control, streaming, and identity fields.

    Saving a RunPlan to disk as JSON and reloading it via from_dict() is
    sufficient to reproduce any run identically — no code modifications needed.

    Design principle — separation of concerns
    ------------------------------------------
    RunPlan is a pure data container. It does not validate its own fields,
    resolve incompatible combinations, or apply defaults based on other fields.
    That responsibility belongs to wizard.py (interactive validation) and
    executor.py (runtime guards). This keeps RunPlan simple, testable, and
    reusable across different entry points.

    Nested config classes
    ----------------------
        parameters : ParameterConfig — voltage limits, thermal limits, scaling
        network    : NetworkConfig   — which network to load and how
        dataset    : DatasetConfig   — which profiles to use and how to build them

    Streaming fields
    -----------------
    The executor publishes timestep results to the Flask dashboard via SSE.
    stream_every_k controls the cadence of that live output without 
    affecting the simulation itself.
    """
    parameters: ParameterConfig = None # Simulation constraint and scaling parameters.
    network: NetworkConfig = None # Network source configuration.
    dataset: DatasetConfig = None # Dataset / profile source configuration.
    study: str = None # Which study protocol to run (e.g. scenario_comparision, hosting_capacity etc.)

    focus_buses: list = None # Subset of bus indices to highlight in live dashboard output and result logs. None = all buses

    # ---- Hardware (Scenario 4 HIL) ----
    # hardware=False (default) → dry_run=True everywhere: Scenario 4 computes
    # Q via the pure-Python QVCharacteristic, no Arduino needed.
    # hardware=True → BenchmarkConfig.dry_run=False and port is REQUIRED;
    # the executor verifies the port exists before the run starts.
    hardware: bool = False
    port: str = None # Serial port for the Arduino, e.g. /dev/ttyACM0 (RPi) or COM3

    # Optional controller-plugin YAML (plugin_runner.py). When set, the
    # executor routes the run through register_and_run() so the custom
    # controller appears as an extra row alongside the selected scenarios.
    controller_plugin_path: str = None

    # ---- Time window (FLAGGED IMPROVEMENT — accept or strip) ----
    # None/"full" = full annual run. "day"/"week"/"month" + time_index
    # slice the built profiles via scenario_result.slice_profiles() before
    # the benchmark — the same fast-iteration mechanism
    # run_benchmark_script.py uses. Slicing happens AFTER profile building,
    # so extreme-day metadata and profile construction are unaffected.
    time_period: str = None
    time_index: int = None

    # hosting_capacity study only: when True, after the HC sweep the
    # benchmark re-runs the scenario set on the HC-stressed network
    # (run_hc_scenarios=True with an executor-built profile_factory).
    hc_stressed: bool = False
    
    # Publish a timestep result to the Flask dashboard every k steps.
    # k=1 is ideal but can saturate the SSE connection on long runs.
    # At 15-min resolution, k=4 updates the dashboard every simulated hour
    stream_every_k: int = 4

    # Universally unique identifier for this run.
    # Auto-generated via uuid4() on instantiation — no two runs share an ID.
    # Used to name the output directory (runs/<run_id>/) and to match
    # SSE events to the correct Flask endpoint.
    # default_factory ensures each instance gets a fresh UUID, not a shared one.
    run_id: str = field(default_factory=lambda:str(uuid.uuid4()))
    output_dir: str = "runs" # Root directory under which run output is written.

    def to_dict(self) -> dict:
        """
        Serialises the entire RunPlan — including all nested configs — to a
        plain dictionary suitable for JSON output.

        dataclasses.asdict() recursively converts ParameterConfig, NetworkConfig,
        and DatasetConfig to plain dicts, so the result is fully JSON-serialisable
        with no further processing required.

        Returns
        -------
        dict
            Complete run configuration as a nested plain dictionary.
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "RunPlan":
        """
        Reconstructs a complete RunPlan from a plain dictionary.

        Each nested config is reconstructed via its own from_dict() method.
        The `or {}` pattern on each nested field guards against both a missing
        key (data.get() returns None) and an explicit null in the JSON — either
        way, from_dict() receives an empty dict and fills all fields with defaults.

        Parameters
        ----------
        data : dict
            Raw dictionary parsed from a saved JSON preset file.

        Returns
        -------
        RunPlan
            Fully reconstructed instance. Fields absent from the JSON receive
            their original defaults. Nested configs are returned as typed objects,
            not raw dicts.
        """
        return cls(
            # Reconstruct each nested config via its own from_dict().
            # `or {}` ensures a missing or null key never reaches from_dict() as None.
            parameters = ParameterConfig.from_dict(data.get("parameters") or {}),
            network    = NetworkConfig.from_dict(data.get("network") or {}),
            dataset  = DatasetConfig.from_dict(data.get("dataset") or {}),
            # Flat run-control fields — fall back to class-level defaults if absent.
            study     = data.get("study", None),
            focus_buses        = data.get("focus_buses", None),
            hardware           = data.get("hardware", False),
            port               = data.get("port", None),
            controller_plugin_path = data.get("controller_plugin_path", None),
            hc_stressed        = data.get("hc_stressed", False),
            time_period        = data.get("time_period", None),
            time_index         = data.get("time_index", None),
            stream_every_k    = data.get("stream_every_k", 4),
            # If run_id is absent (e.g. a preset saved before this field existed),
            # generate a fresh UUID rather than leaving the field as None.
            run_id     = data.get("run_id", str(uuid.uuid4())),
            output_dir     = data.get("output_dir", "runs"),
        )
