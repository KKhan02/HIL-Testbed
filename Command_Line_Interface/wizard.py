from rich.prompt import IntPrompt, FloatPrompt, Prompt, Confirm
from .run_plan import NetworkConfig, DatasetConfig, ParameterConfig, RunPlan
from .network_catalogue import get_preset_families, get_presets_for_family
from .executor import _preset_loaders, _import_custom_network_fn
import ast
from volt_var_controller import PLAUSIBLE_PU_RANGE
from violation_detector import THERMAL_LOADING_PLAUSIBLE_MAX


from ._console import console


class BackRequested(Exception):
    """
    Raised by any wizard prompt when the user asks to go back.

    Caught only by run_wizard()'s step loop, which decrements the step
    index. Menus expose this as option 0 ("go back"); free-text prompts
    accept "<" as the back token (chosen because it is not a plausible
    path, station ID, number, or dict literal).
    """


BACK_TOKEN = "<"

def _ask(
        prompt: str,
        choices: list[tuple[str,str]],
        default: int | None = None,
        allow_back: bool = False,
) -> str:
    '''
    Show a numbered menu and return the selected machine-readable value.

    Parameters
    ----------
    prompt:
        Question shown above the menu.

    choices:
        List of (display_label, return_value) tuples.

    default:
        Optional zero-based default index into choices.
        Example: default=0 means option 1 is the default.

    Returns
    -------
    str
        The machine-readable return value from the selected choice.
    '''

    if not choices:
        raise ValueError("_ask() requires at least one choice")
    
    if default is not None and not 0 <= default < len(choices):
        raise ValueError(
            f"default index {default} is out of range for {len(choices)} choices."
        )
    
    console.print()
    console.print(f"[stage]{prompt}[/stage]")

    for index, (label,_) in enumerate(choices, start=1):
        console.print(f"[muted]{index}.[/muted] {label}")
    if allow_back:
        console.print("[muted]0.[/muted] ← Go back to the previous step")

    display_default = default + 1 if default is not None else None

    while True:
        selected_number = IntPrompt.ask(
            "Enter choice number",
            default=display_default,
            show_default=display_default is not None,
            console=console,
        )

        if allow_back and selected_number == 0:
            raise BackRequested()
        if 1 <= selected_number <= len(choices):
            selected_index = selected_number - 1
            return choices[selected_index][1]
        low = 0 if allow_back else 1
        console.print(
            f"[error] Invalid choice.[/error] "
            f"Please enter a number between {low} and {len(choices)}."
        )

def _label_for_value(choices: list[tuple[str,str]], value:str) -> str:
    '''Return the display label for a selected machine-readable value'''
    return next(label for label, return_value in choices if return_value == value) 


def _ask_value(
        prompt: str,
        default=None,
        cast=str,
        allow_back: bool = True,
):
    '''
    Back-aware replacement for Prompt/IntPrompt/FloatPrompt.ask.

    Reads a raw string, treats BACK_TOKEN ("<") as a request to return to
    the previous wizard step, and otherwise casts to the requested type,
    re-asking on a cast failure. default=None with an empty answer re-asks
    (a value is required); a non-None default is returned on Enter.
    '''
    show_default = default is not None
    while True:
        raw = Prompt.ask(
            f"{prompt} [muted]('{BACK_TOKEN}' = back)[/muted]",
            default=str(default) if show_default else None,
            show_default=show_default,
            console=console,
        )
        raw = (raw or "").strip()
        if allow_back and raw == BACK_TOKEN:
            raise BackRequested()
        if raw == "":
            if show_default:
                raw = str(default)
            else:
                console.print("[error]A value is required.[/error]")
                continue
        try:
            return cast(raw)
        except (ValueError, TypeError):
            console.print(
                f"[error]Invalid value.[/error] Expected {cast.__name__}."
            )

def _ask_study() -> str:
    '''
    Ask which study runner should be used for this CLI run.

    Returns the machine-readable study name stored in Runplan.study
    '''
    return _ask(
        "Select study type",
        [
            (
                "Scenario comparison — run the 5 benchmark scenarios sequentially and compare results",
                "scenario_comparison",
            ),
            (
                "Voltage variation — run timestep-by-timestep voltage control with remediation logic",
                "voltage_variation",
            ),
            (
                "Hosting capacity — sweep PV penetration and find the first violation point",
                "hosting_capacity",
            ),
            (
                "OPF benchmark — run AC optimal power flow as a theoretical performance bound",
                "opf_benchmark",
            ),
        ],
        default=0,
        allow_back=True,
    )

def _ask_network() -> NetworkConfig:
    '''
    Ask how the pandapower network should be selected.
    Returns a fully populated NetworkConfig.
    '''

    source_type = _ask(
        "Select Network Source",
        [
            (
                "Curated preset catalogue — choose from the SimBench (Main), CIGRE, Kerber, Dickert, or Synthetic LV networks",
                "preset",  
            ),
            (
                "Assemble any SimBench network by code — build a code such as 1-LV-rural--0-sw from menu choices",
                "simbench_code",
            ),
            (
                "Custom Python network file — load your own pandapower network from a Python file",
                "custom",
            ),
            (
                "Network plugin (YAML) — load a self-contained network + profile plugin folder (network_plugin.py)",
                "plugin",
            ),
        ],
        default=0,
        allow_back=True,
    )

    if source_type == "preset":

        families = get_preset_families()
        family_choices = [
            (family_name, family_name)
            for family_name in families
        ]

        selected_family = _ask(
            "Select preset network family",
            family_choices,
            default=0,
            allow_back=True,
        )

        preset_entries = get_presets_for_family(selected_family)
        preset_choices = [
            (entry["label"], entry["preset_name"])
            for entry in preset_entries
        ]

        selected_preset_name = _ask(
            f"Select preset network from {selected_family}",
            preset_choices,
            default=0,
            allow_back=True,
        )

        return NetworkConfig(
            source_type="preset",
            preset_name=selected_preset_name,
            preset_family=selected_family,
        )

    elif source_type == "simbench_code":
        voltage_level_choices = [
            ("MV, medium-voltage network", "MV"),
            ("LV, low-voltage network", "LV"),
            ("MVLV, medium and low-voltage mixed network", "MVLV"),
        ]

        mv_grid_type_choices = [
            ("Rural MV grid", "rural"),
            ("Semi-urban MV grid", "semiurb"),
            ("Urban MV grid", "urban"),
            ("Commercial MV grid", "comm"),
        ]

        lv_grid_type_choices = [
            ("Rural LV grid 1", "rural1"),
            ("Rural LV grid 2", "rural2"),
            ("Rural LV grid 3", "rural3"),
            ("Semi-urban LV grid 4", "semiurb4"),
            ("Semi-urban LV grid 5", "semiurb5"),
            ("Urban LV grid 6", "urban6"),
        ]

        mvlv_subnet_choices_by_grid_type = {
            "rural": [
                ("All connected rural LV grids", "all"),
                ("Rural LV subnet 1.108", "1.108"),
                ("Rural LV subnet 2.107", "2.107"),
                ("Rural LV subnet 4.101", "4.101"),
            ],
            "semiurb": [
                ("All connected semi-urban LV grids", "all"),
                ("Semi-urban LV subnet 3.202", "3.202"),
                ("Semi-urban LV subnet 4.201", "4.201"),
                ("Semi-urban LV subnet 5.220", "5.220"),
            ],
            "urban": [
                ("All connected urban LV grids", "all"),
                ("Urban LV subnet 5.303", "5.303"),
                ("Urban LV subnet 6.305", "6.305"),
                ("Urban LV subnet 6.309", "6.309"),
            ],
            "comm": [
                ("All connected commercial LV grids", "all"),
                ("Commercial LV subnet 3.403", "3.403"),
                ("Commercial LV subnet 4.416", "4.416"),
                ("Commercial LV subnet 5.401", "5.401"),
            ],
        }

        scenario_level_choices = [
            ("Baseline scenario without DER or load scaling, Scenario 0", "0"),
            ("Future scenario with moderate load and DER growth, Scenario 1", "1"),
            ("Future scenario with high DER and load growth, Scenario 2", "2"),
        ]

        switching_suffix_choices = [
            ("Switched network, includes switch representation", "sw"),
            ("Unswitched network, no switch representation", "no_sw"),
        ]

        voltage_level = _ask(
            "Select SimBench voltage level",
            voltage_level_choices,
            default=0,
            allow_back=True,
        )

        if voltage_level == "LV":
            grid_type = _ask(
                "Select SimBench LV grid type",
                lv_grid_type_choices,
                default=0,
                allow_back=True,
            )
            lv_subnet = None

        else:
            grid_type = _ask(
                "Select SimBench MV grid type",
                mv_grid_type_choices,
                default=0,
                allow_back=True,
            )

            if voltage_level == "MVLV":
                mvlv_subnet_choices = mvlv_subnet_choices_by_grid_type[grid_type]
                lv_subnet = _ask(
                    "Select connected LV subnet for the MVLV network",
                    mvlv_subnet_choices,
                    default=0,
                    allow_back=True,
                )
            else:
                lv_subnet = None

        scenario_level = _ask(
            "Select SimBench scenario level",
            scenario_level_choices,
            default=2,
            allow_back=True,
        )

        switching_suffix = _ask(
            "Select SimBench switch representation",
            switching_suffix_choices,
            default=0,
            allow_back=True,
        )

        if voltage_level == "MVLV":
            simbench_code = (
                f"1-MVLV-{grid_type}-{lv_subnet}-{scenario_level}-{switching_suffix}"
            )
        else:
            simbench_code = (
                f"1-{voltage_level}-{grid_type}--{scenario_level}-{switching_suffix}"
            )

        simbench_selections = {
            "Voltage Level": _label_for_value(voltage_level_choices, voltage_level),
            "Grid Type": (
                _label_for_value(lv_grid_type_choices, grid_type)
                if voltage_level == "LV"
                else _label_for_value(mv_grid_type_choices, grid_type)
            ),
            "Scenario Level": _label_for_value(scenario_level_choices, scenario_level),
            "Switch Representation": _label_for_value(
                switching_suffix_choices,
                switching_suffix,
            ),
        }

        if voltage_level == "MVLV":
            simbench_selections["Connected LV Subnet"] = _label_for_value(
                mvlv_subnet_choices_by_grid_type[grid_type],
                lv_subnet,
            )

        return NetworkConfig(
            source_type="simbench_code",
            simbench_code=simbench_code,
            simbench_selections=simbench_selections,
        )
    elif source_type == "custom":

        custom_path = _ask_value("Enter path to custom Python network file")
        custom_function_name = _ask_value(
            "Enter network factory function name", default="get_network",
        )

        return NetworkConfig(
            source_type="custom",
            custom_path=custom_path,
            custom_function_name=custom_function_name,
        )

    elif source_type == "plugin":
        console.print(
            "[muted]A network plugin is a YAML file describing the network "
            "source (json/pickle/function) and its profile strategy "
            "(simbench_native / dwd_pvlib / flat). Relative paths inside the "
            "YAML resolve against the YAML's own folder. NOTE: the plugin "
            "owns profile building, so the dataset step will be skipped.[/muted]"
        )
        plugin_path = _ask_value("Enter path to the network plugin YAML")

        return NetworkConfig(
            source_type="plugin",
            plugin_path=plugin_path,
        )

    else:
        raise RuntimeError(f"Unexpected network source type: {source_type}")


def _ask_optional_dict(prompt: str) -> dict | None:
    '''
    Ask for an optional dictionary override.
    
    Pressing Enter returns None. A non-empty answer must have a valid
    Python dict literal, for example: {"solar": "solar.csv", "wind": "wind.csv"}.
    '''

    while True:
        console.print()
        console.print(f"[stage]{prompt}[/stage]")
        console.print("[muted]Press Enter for no scaling, or enter a dict, e.g.[/muted]")
        console.print("[cyan]{None: 1.0}[/cyan] or [cyan]{12: 1.5, 15: 0.8}[/cyan]")

        raw_value = Prompt.ask(
            prompt,
            default="",
            show_default=False,
            console=console,
        ).strip()

        if raw_value == BACK_TOKEN:
            raise BackRequested()
        if not raw_value:
            return None
        
        try:
            parsed_value = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            console.print(
                "[error]Invalid dictionary format.[/error] "
                "Use a Python dict literal, for example: "
                '{"solar": "solar.csv", "wind": "wind.csv"}'
            )
            continue

        if not isinstance(parsed_value, dict):
            console.print("[error]Invalid input.[/error] Please enter a dictionary.")
            continue
        
        return parsed_value
    
def _ask_dataset() -> DatasetConfig:
    '''
    Ask which dataset source should be used for this run.

    Returns a fully populated DatasetConfig for the selected source type.
    '''

    source_type = _ask(
        "Select dataset source",
        [
            (
                "SimBench native profiles — use time-series profiles bundled with the selected SimBench network",
                "simbench_native",
            ),
            (
                "DWD Climate Data Centre files — build profiles from the standard project DWD folder structure",
                "dwd",
            ),
            (
                "Custom dataset file — use a user-provided profile file with optional file and column mappings",
                "custom",
            ),
        ],
        default=1,
        allow_back=True,
    )

    if source_type == "simbench_native":
        return DatasetConfig(
            source_type="simbench_native",
        )
    elif source_type == "dwd":
        data_dir   = _ask_value("Enter DWD data directory", default="data/dwd")
        station_id = _ask_value("Enter DWD Station ID", default="691")
        year       = _ask_value("Enter dataset year", default=2024, cast=int)
        return DatasetConfig(
            source_type="dwd",
            data_dir=data_dir,
            station_id=station_id,
            year=year,
        )
    elif source_type == "custom":
        custom_path = _ask_value(
            "Enter path to the custom dataset directory (passed to "
            "profile_builder as data_dir)",
        )

        file_map = _ask_optional_dict(
            "Optional file_map override — maps profile type to filename.\n"
            "Press Enter to skip, or enter a dict, e.g.\n"
            '{"RAD-G": "solar_691.csv", "F": "wind_691.csv", "T2M": "temp_691.csv"}'
        )

        col_map = _ask_optional_dict(
            "Optional col_map override — maps profile type to column name within the file.\n"
            "Press Enter to skip, or enter a dict, e.g.\n"
            '{"timestamp": "Zeitstempel", "solar": "Wert", "wind": "Wert", "temp": "Wert", "sep": ","}'
        )

        return DatasetConfig(
            source_type="custom",
            custom_path=custom_path,
            file_map=file_map,
            col_map=col_map,
        )
    
    else:
        raise RuntimeError(f"Unexpected dataset source type: {source_type}")

def _ask_scaling_dict(prompt: str) -> dict:
    '''
    Ask for an optional scaling dictionary.

    Pressing Enter returns the default no-scaling convention: {None: 1.0}.    
    '''

    scaling = _ask_optional_dict(prompt)
    return scaling if scaling is not None else {None: 1.0}

def _ask_parameters() -> ParameterConfig:
    '''
    Ask for numerical simulation limits, scaling, and timestep resolution.

    Returns a fully populated ParameterConfig.
    '''

    while True:
        v_min = _ask_value("Enter minimum voltage limit in pu", default=0.95, cast=float)
        v_max = _ask_value("Enter maximum voltage limit in pu", default=1.05, cast=float)
        if not (0.5 <= v_min < v_max <= 1.5):
            console.print(
                "[error]Voltage band invalid: require 0.5 <= v_min < v_max <= 1.5 "
                f"pu, got v_min={v_min}, v_max={v_max}.[/error]"
            )
            continue
        break

    while True:
        line_max_loading = _ask_value("Enter maximum line loading in percent", default=100, cast=int)
        if not (0.0 < line_max_loading <= THERMAL_LOADING_PLAUSIBLE_MAX):
            console.print(f"[error]Line loading must satisfy 0 < value <= {THERMAL_LOADING_PLAUSIBLE_MAX:.0f}%.[/error]")
            continue
        break

    while True:
        trafo_max_loading = _ask_value("Enter maximum transformer loading in percent", default=100, cast=int)
        if not (0.0 < trafo_max_loading <= THERMAL_LOADING_PLAUSIBLE_MAX):
            console.print(f"[error]Transformer loading must satisfy 0 < value <= {THERMAL_LOADING_PLAUSIBLE_MAX:.0f}%.[/error]")
            continue
        break

    while True:
        va_diff_max_degree = _ask_value("Enter maximum voltage angle difference in degrees", default=30, cast=int)
        if not (0.0 < va_diff_max_degree <= 180.0):
            console.print("[error]Angle difference must satisfy 0 < value <= 180 degrees.[/error]")
            continue
        break

    while True:
        unbalance_max_percent = _ask_value("Enter maximum voltage unbalance in percent", default=2.0, cast=float)
        if not (0.0 < unbalance_max_percent <= 100.0):
            console.print("[error]Unbalance must satisfy 0 < value <= 100 %.[/error]")
            continue
        break

    timestep_resolution = _ask(
        "Select timestep resolution",
        [
            ("15 minutes, SimBench native profile resolution", "15"),
            ("10 minutes, DWD weather data resolution", "10"),
            ("Other — enter a custom value", "custom"),
        ],
        default=0,
        allow_back=True,
    )
    if timestep_resolution == "custom":
        while True:
            timestep_resolution = _ask_value(
                "Enter custom timestep resolution in minutes (fractional values "
                "allowed, e.g. 5, 1, 0.5 for 30 seconds)",
                default=5, cast=float,
            )
            if timestep_resolution <= 0.0:
                console.print(
                    "[error]Timestep resolution must be greater than 0 "
                    "minutes.[/error]"
                )
                continue
            break
    else:
        timestep_resolution = float(timestep_resolution)
    
    der_scaling = _ask_scaling_dict(
        "Optional DER scaling override — maps bus index to scaling factor."
    )
    load_scaling = _ask_scaling_dict(
        "Optional load scaling override — maps bus index to scaling factor."
    )

    # ---- Optional per-run Q(V) characteristic (VDE-AR-N 4110 Bild 8) ----
    # Defaults (Enter on the first question) keep the framework constants:
    # Q_RATIO=0.25, U1..U4 = 0.96/0.99/1.01/1.04. Overrides are applied by
    # the executor through volt_var_controller.set_qv_parameters(), which
    # updates the dry-run curve, the coordinator sizing, AND the CFG:
    # message pushed to the Arduino — one value, all consumers.
    q_ratio = u1 = u2 = u3 = u4 = None
    override_qv = Confirm.ask(
        "Override the Q(V) characteristic parameters for this run?",
        default=False,
        console=console,
    )
    if override_qv:
        while True:
            q_ratio = _ask_value("Q ratio (Q_max / P_installed, 0 < q ≤ 1)", default=0.25, cast=float)
            u1 = _ask_value("U1 — lower saturation breakpoint [pu]", default=0.96, cast=float)
            u2 = _ask_value("U2 — deadband lower edge [pu]",          default=0.99, cast=float)
            u3 = _ask_value("U3 — deadband upper edge [pu]",          default=1.01, cast=float)
            u4 = _ask_value("U4 — upper saturation breakpoint [pu]",  default=1.04, cast=float)
            # Same validity matrix the Arduino firmware enforces (ERR:CFG_INVALID).
            if not (0.0 < q_ratio <= 1.0):
                console.print("[error]q_ratio must satisfy 0 < q ≤ 1.[/error]")
                continue
            bad = [(name, v) for name, v in
                   (("U1", u1), ("U2", u2), ("U3", u3), ("U4", u4))
                   if not (PLAUSIBLE_PU_RANGE[0] <= v <= PLAUSIBLE_PU_RANGE[1])]
            if bad:
                for name, v in bad:
                    console.print(
                        f"[error]{name}={v} is outside the plausible range "
                        f"[{PLAUSIBLE_PU_RANGE}] pu — likely a typo (e.g. 6.0 instead of "
                        f"1.06).[/error]"
                    )
                continue
            if not (u1 < u2 < u3 < u4):
                console.print("[error]Breakpoints must be strictly increasing: U1 < U2 < U3 < U4.[/error]")
                continue
            break

    return ParameterConfig(
        v_min=v_min,
        v_max=v_max,
        line_max_loading=line_max_loading,
        trafo_max_loading=trafo_max_loading,
        va_diff_max_degree=va_diff_max_degree,
        unbalance_max_percent=unbalance_max_percent,
        der_scaling=der_scaling,
        load_scaling=load_scaling,
        timestep_resolution=timestep_resolution,
        q_ratio=q_ratio,
        u1_pu=u1, u2_pu=u2, u3_pu=u3, u4_pu=u4,
    )

def _ask_hc_stressed() -> bool:
    """hosting_capacity study only: also re-benchmark the HC-stressed net?"""
    return _ask(
        "After the hosting-capacity sweep, also re-run the benchmark "
        "scenarios on the HC-stressed network (network loaded to its "
        "hosting-capacity limit)?",
        [
            ("No — hosting-capacity sweep only (baseline + Volt-Var HC)", "no"),
            ("Yes — HC sweep, then full scenario re-benchmark on the stressed network (much longer run)", "yes"),
        ],
        default=0,
        allow_back=True,
    ) == "yes"


def _ask_hardware() -> tuple[bool, str | None]:
    """
    Ask whether Scenario 4 should run on the Arduino (HIL) or dry-run.

    dry-run (default) computes Q via the pure-Python QVCharacteristic; the
    hardware path requires a serial port, which the executor verifies
    exists BEFORE the run starts.
    """
    mode = _ask(
        "Run the Volt-Var scenarios on Arduino hardware (HIL) or in "
        "software (dry run)?",
        [
            ("Dry run — pure-Python Q(V), no Arduino needed (default)", "dry"),
            ("Hardware — Arduino in the loop over serial (requires a connected board)", "hw"),
        ],
        default=0,
        allow_back=True,
    )
    if mode == "dry":
        return False, None
    port = _ask_value(
        "Serial port for the Arduino", default="/dev/ttyACM0",
    )
    return True, port


def _ask_controller_plugin() -> str | None:
    """
    Optional controller plugin (plugin_runner.py): a YAML naming a Python
    controller function — and, for hardware plugins, the firmware sketch —
    run ALONGSIDE the selected scenarios as an extra comparison row.
    """
    attach = Confirm.ask(
        "Attach a custom controller plugin (YAML) to this run?",
        default=False,
        console=console,
    )
    if not attach:
        return None
    return _ask_value("Enter path to the controller plugin YAML")

def _load_preview_network(network_config: NetworkConfig):
    """
    Load a bare net (no profiles) purely to preview net.switch for the
    switch-flip prompt in _ask_network_modifications(). Mirrors
    executor.py's own network-loading branches for all four source types
    so switch indices shown here match what the real run will use.
    Returns None if the network can't be loaded at this point (e.g.
    custom file doesn't exist yet, plugin YAML malformed).
    """
    st = network_config.source_type

    if st == "preset":
        loaders = _preset_loaders()
        if network_config.preset_name not in loaders:
            return None
        return loaders[network_config.preset_name]()

    if st == "simbench_code":
        import simbench as sb
        return sb.get_simbench_net(network_config.simbench_code)

    if st == "custom":
        fn = _import_custom_network_fn(
            network_config.custom_path, network_config.custom_function_name,
        )
        return fn()

    if st == "plugin":
        from network_plugin import load_network_from_yaml
        net, _profiles = load_network_from_yaml(network_config.plugin_path)
        return net

    return None

def _ask_network_modifications(network_config: NetworkConfig) -> tuple[list | None, list | None]:
    """
    Optional pre-run network modifications, mirroring the documented manual
    blocks in run_benchmark_script.py: DER injection at named buses and
    opening of switches (e.g. sectionalizers). Applied by the executor
    BEFORE profile building so injected DERs receive profile columns.
    """
    der_placements = None
    switches_to_open = None

    if Confirm.ask(
        "Inject additional DERs at specific buses before the run?",
        default=False,
        console=console,
    ):
        der_placements = []
        while True:
            while True:
                bus = _ask_value("Bus index for the new PV sgen", cast=int)
                if bus < 0:
                    console.print("[error]Bus index must be non-negative.[/error]")
                    continue
                break
            while True:
                p_mw = _ask_value("Installed active power P [MW]", default=0.5, cast=float)
                if p_mw <= 0:
                    console.print("[error]P must be greater than 0 MW.[/error]")
                    continue
                break
            while True:
                sn_mva = _ask_value("Inverter rating S [MVA]",
                                    default=round(p_mw * 1.1, 4), cast=float)
                if sn_mva < p_mw:
                    console.print("[error]Inverter rating S must be >= P.[/error]")
                    continue
                break
            der_placements.append({"bus": bus, "p_mw": p_mw, "sn_mva": sn_mva})
            if not Confirm.ask("Add another DER?", default=False, console=console):
                break

    switches_to_flip = None
    if Confirm.ask(
        "Change the state of specific switches before the run?",
        default=False,
        console=console,
    ):
        preview_net = None
        try:
            preview_net = _load_preview_network(network_config)
        except Exception as exc:
            console.print(f"[error]Could not load network to preview switches: {exc}[/error]")

        if preview_net is None:
            console.print("[error]No network data available to preview switches — skipping switch changes.[/error]")
        else:
            console.print(preview_net.switch[["name", "bus", "element", "et", "closed"]].to_string())
            raw = _ask_value(
                "Switch indices to flip (open<->closed), comma-separated (e.g. 5 or 5,12)",
            )
            try:
                parsed = [int(x) for x in str(raw).split(",") if str(x).strip()]
                missing = [s for s in parsed if s not in preview_net.switch.index]
                if missing:
                    console.print(
                        f"[error]Switch indices {missing} do not exist in this "
                        f"network (valid: {list(preview_net.switch.index)}) — "
                        f"skipping switch changes.[/error]"
                    )
                else:
                    switches_to_flip = parsed
            except ValueError:
                console.print("[error]Could not parse switch indices — skipping switch changes.[/error]")

    return der_placements, switches_to_flip


def _ask_time_window() -> tuple[str | None, int | None]:
    """
    FLAGGED IMPROVEMENT (accept or strip): choose the simulated window.
    Full annual is the research default; day/week/month slices reuse
    scenario_result.slice_profiles() for fast student iteration —
    identical mechanism to run_benchmark_script.py's slice block.
    """
    period = _ask(
        "Select the simulated time window",
        [
            ("Full annual run — the research default (35k+ timesteps, hours of runtime)", "full"),
            ("One day — fast iteration (~100-150 timesteps)", "day"),
            ("One week", "week"),
            ("One month", "month"),
        ],
        default=0,
        allow_back=True,
    )
    if period == "full":
        return None, None
    ranges = {"day": (1, 366), "week": (1, 53), "month": (1, 12)}
    lo, hi = ranges[period]
    defaults = {"day": 172, "week": 25, "month": 6}
    while True:
        index = _ask_value(
            f"Which {period} of the year ({lo}-{hi})",
            default=defaults[period], cast=int,
        )
        if lo <= index <= hi:
            return period, index
        console.print(f"[error]Enter a value between {lo} and {hi}.[/error]")


def run_wizard() -> RunPlan:
    """
    Run the interactive CLI wizard and return a complete RunPlan.

    This is the only public function in wizard.py. __main__.py calls this
    function, then passes the returned RunPlan to the next CLI layer.

    Navigation: the wizard is an ordered list of steps walked by index.
    Every menu offers "0 — go back" and every free-text prompt accepts
    "<" as a back token; either raises BackRequested, which this loop
    catches to re-enter the PREVIOUS step (re-entering a step re-asks its
    questions with the standard defaults). Back on the first step simply
    re-asks it.
    """
    state: dict = {}

    def step_study(s):
        s["study"] = _ask_study()
        s["hc_stressed"] = (
            _ask_hc_stressed() if s["study"] == "hosting_capacity" else False
        )

    def step_network(s):
        s["network"] = _ask_network()

    def step_network_mods(s):
        der, sw = _ask_network_modifications(s["network"])
        s["network"].der_placements = der
        s["network"].switches_to_flip = sw

    def step_time_window(s):
        s["time_period"], s["time_index"] = _ask_time_window()

    def step_dataset(s):
        if s["network"].source_type == "plugin":
            console.print(
                "[muted]Dataset step skipped — the network plugin's YAML "
                "profiles.strategy owns profile building for this run.[/muted]"
            )
            s["dataset"] = DatasetConfig(source_type="plugin")
            return
        s["dataset"] = _ask_dataset()

    def step_parameters(s):
        s["parameters"] = _ask_parameters()

    def step_hardware(s):
        s["hardware"], s["port"] = _ask_hardware()

    def step_stream_every_k(s):
        while True:
            k = _ask_value("Publish a live frame every N timesteps",
                           default=4, cast=int)
            if k < 1:
                console.print("[error]N must be >= 1.[/error]")
                continue
            break
        s["stream_every_k"] = k

    def step_controller_plugin(s):
        s["controller_plugin_path"] = _ask_controller_plugin()

    steps = [
        ("Study",               step_study),
        ("Network",             step_network),
        ("Network modifications", step_network_mods),
        ("Dataset",             step_dataset),
        ("Time window",          step_time_window),
        ("Parameters",          step_parameters),
        ("Hardware",            step_hardware),
        ("Streaming",           step_stream_every_k),
        ("Controller plugin",   step_controller_plugin),
    ]

    index = 0
    while index < len(steps):
        name, fn = steps[index]
        console.print()
        console.rule(
            f"[stage]Step {index + 1}/{len(steps)} — {name}[/stage]",
            style="rule",
        )
        try:
            fn(state)
        except BackRequested:
            if index == 0:
                console.print("[muted]Already at the first step.[/muted]")
            index = max(index - 1, 0)
            continue
        index += 1

    return RunPlan(
        study=state["study"],
        network=state["network"],
        dataset=state["dataset"],
        parameters=state["parameters"],
        hardware=state["hardware"],
        port=state["port"],
        stream_every_k=state["stream_every_k"],
        controller_plugin_path=state["controller_plugin_path"],
        hc_stressed=state["hc_stressed"],
        time_period=state["time_period"],
        time_index=state["time_index"],
    )