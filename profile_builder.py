'''
Builds annual 10-min time-series profiles for load, PV, and wind for any pandapower network

For Simbench networks, sb.get_absolute_values() provides native 15-min annual profiles

For CIGRE/Dickert/Kerber/Other, the annual profiles are build from DWD local files (PV + wind) + oemof.demand BDEW 2025 SLPs(load)

DWD files should be present in the data_dir (locally stored) under their respective folders (Temperature, PV, Wind)

Column structure (all DWD files):
Produkt_Code | SDO_ID | Zeitstempel | Wert | Qualitaet_Byte | Qualitaet_Niveau
Wert = the measured value in native units

Unit conversion applied:
RAD-G : J/cm²  → W/m²  via  Wert × 10,000 / 600
F     : m/s    (no conversion needed)
T2M   : °C     (no conversion needed)

PV conversion: pvlib POA (Erbs decomposition) + NOCT temperature correction (need DWD temperature data)

Wind conversion: piecewise cubic with effective Cp factor. Uses mean wind speed

Load fallback:   BDEW 2025 updated SLPs via oemof.demand >= 0.2.2 (H25 / G25 / L25). Per-load type assignment from 
net.load.name / net.load.type fields. Holidays via workalendar (optional — graceful fallback to empty list).

Note on BDEW 1999 vs 2025
--------------------------
The conventional BDEW 1999 profiles (H0/G0/L0) were based on measurements from over 20 years ago and showed increasing 
deviations from current load behaviour (Gabrielski & Haeger, TU Dortmund, 2024). BDEW acknowledged this and released 
updated profiles in March 2025 (H25/G25/L25) based on 2018-2023 smart meter data from 62 DSOs.
These updated profiles are used here as the methodologically current standard.
 
Dependencies
------------
pip install pandapower simbench pvlib 'oemof-demand>=0.2.2' workalendar

'''

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ===========================================================================
# Station / site constants  (DWD station 691 — Bremen)
# ===========================================================================
LATITUDE          = 53.0451   #Degrees N
LONGITUDE         = 8.7981    #Degrees E
TIMEZONE          = "Europe/Berlin"

# Default column names following the DWD CDC convention.
# Override via col_map parameter in build_annual_profiles()
# when using a different datasource.

DWD_COL_MAP = {
    "timestamp":    "Zeitstempel",  # datetime column
    "solar":        "Wert",         # solar irradiance column
    "wind":         "Wert",         # wind speed column
    "temp":         "Wert",         # temperature column
    "sep":          ","             # column seperator in CSV files. DWD uses std comma seperator
}


# ===========================================================================
# PV model constants
# ===========================================================================

G_STC             = 1000.0    # W/m2 STC irradiance
ETA_REF           = 0.20      # reference cell efficiency at STC (modern Si)
BETA_TEMP         = 0.0045    # temperature coefficient /degC
T_REF             = 25.0      # reference temperature   /degC
NOCT              = 45.0      # nominal operating cell temperature degC
ETA_INVERTER      = 0.97      # inverter efficiency
SURFACE_TILT      = 30.0      # panel tilt from horizontal (degrees)
SURFACE_AZIMUTH   = 180.0     # south-facing (degrees from north)
GHI_MIN_THRESHOLD = 10.0      # Clamp PV output to zero below this (W/m2)

# ===========================================================================
# Wind turbine constants
# ===========================================================================

V_CUT_IN    = 3.0   #m/s
V_RATED     = 12.0  #m/s
V_CUT_OUT   = 25.0  #m/s

# ===========================================================================
# BDEW 2025 load type mapping
# Maps pandapower load name/type keywords to oemof.demand 2025 profile class.
# Out of the 5 new profiles; only H25, G25 and L25 are relevant 
# These are consumer categories, each describing an electricity user 
# with a distinct daily and seasonal consumption pattern
# If name or type column in net.load is populated meaningfully with sectors,
# these load profiles are assigned to that bus (scaled to its MW rating)
# ===========================================================================

LOAD_TYPE_MAP = {
    "residential":  "h25",
    "household":    "h25",
    "h0":           "h25",
    "h25":          "h25",
    "commercial":   "g25",
    "office":       "g25",
    "business":     "g25",
    "industrial":   "g25",
    "retail":       "g25",
    "bakery":       "g25",
    "weekend":      "g25",
    "g0":           "g25",
    "agriculture":  "l25",
    "agricultural": "l25",
    "l0":           "l25",
    "l25":          "l25"
}

DEFAULT_LOAD_TYPE = "h25" #fallback if there are no defined consumer categories

# ===========================================================================
# 1.  Network type detection
# ===========================================================================

# Keywords of all Simbench Voltage-Level and topology identifiers (including single-level
# and coupled networks) across all area types (rural,urban,semiurban,comm,mixed) are listed

SIMBENCH_IDENTIFIERS = {
    # Explicit SimBench label
    "simbench",
    # Single voltage level
    "1-mv-rural", "1-mv-urban", "1-mv-semiurb", "1-mv-comm",
    "1-lv-rural",  "1-lv-urban",  "1-lv-semiurb",
    "1-hv-mixed",  "1-hv-urban",
    "1-ehv-mixed",
    # Coupled voltage levels
    "1-mvlv-rural",   "1-mvlv-urban",
    "1-mvlv-semiurb", "1-mvlv-comm",
    "1-hvmv-mixed",   "1-hvmv-urban",
    "1-ehvhv-mixed",
    "1-ehvhvmvlv-mixed",
    "1-complete_data-mixed",
}

def detect_network_type(net_name:str) -> str:
    '''
    Returns 'simbench', 'cigre' or 'fallback'

    Parameters:
    net_name : str
        Human readable network identifier e.g. '1-MV-rural--2-sw', 'cigre_mv' etc.
        Simbench detection uses an explicit identifier set covering all known SimBench network codes
        across all voltages and are types, ensuring correct routing for any current or future Simbench
        network
    '''
    n = net_name.lower()
    if any(identifier in n for identifier in SIMBENCH_IDENTIFIERS):
        return "simbench"
    elif "cigre" in n:
        return "cigre"
    else:
        return "fallback"

# ===========================================================================
# 2.  DWD data loading  (reads local CSV files)
# ===========================================================================

def find_dwd_file(data_dir:str, 
                  parameter_code:str,
                  file_map: dict = None) -> str:
    '''
    Locates a weather data CSV file for the given parameter.

    Resolution Order:
        If file_map is provided and contains parameter_code as a key, use that filename directly. 
        This will hopefully support any datasource naming convention (ERA5, DWD, metmast etc.)
        Otherwise, fall back to DWD CDC glob pattern i.e. data_OBS_DEU_PT10M_{parameter_code}*.csv  

    Parameters:
        data_dir        : Folder containing the weather CSV files
        parameter_code  : logical parameter name, e.g. 'RAD-G' for solar, 'F' for wind, 'T2M' for temp
        file_map        : optional dictionary mapping parameter codes to filenames.
                          e.g. {
                                'RAD-G' : 'metmast_solar_2024.csv',
                                'F':      'metmast_wind_2024.csv',
                                'T2M':    'metmast_temperature_2024.csv'  
                          }
                          If None, DWD CDC naming convention is assumed


    Note: if switching default station (691 Bremen) or datasource, update LATITUDE and LONGITUDE constants
    at the top of this file to match the new measurment location. pvlib solar position calculations 
    depend on them.
    '''
    import glob
    
    # Path 1 - explicit file map provided
    if file_map is not None:
        if parameter_code not in file_map:
            raise KeyError(
                f"parameter_code '{parameter_code}' not found in file_map.\n"
                f"file_map keys: {list(file_map.keys())}"
            )
        fpath = os.path.join(data_dir,file_map[parameter_code])
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"File specified in file_map not found: {fpath}"
            )
        return fpath
    
    # Path 2 - DWD CDC default glob pattern
    pattern = os.path.join(data_dir,f"data_OBS_DEU_PT10M_{parameter_code}*.csv")

    # glob.glob(pattern) searches the filesystem for all files matching the pattern 
    # and returns a list of matching file paths. The * wildcard means "anything can be here"

    matches = glob.glob(pattern)
    if matches:
        return sorted(matches)[0] # If two files match, always get the same file irrespective of OS file ordering  
    raise FileNotFoundError(
        f"DWD file not found for parameter '{parameter_code}' in '{data_dir}'.\n"
        f"Expected a file matching: "
        f"data_OBS_DEU_PT10M_{parameter_code}*.csv\n"
        f"For a non-DWD datasource, pass a file_map dict to "
        f"build_annual_profiles().\n"
        f"DWD datasets: https://cdc.dwd.de/portal/202209231028/searchview"
    )

def read_weather_csv(fpath:str,
                     timestamp_col: str,
                     value_col: str,
                     sep: str = ",") -> pd.DataFrame:
    '''
    Shared reader for all weather CSV files

    Tries ISO 8601 timestamp format first (DWD default: 2024-01-01T00:00:00),
    falls back to pd.to_datetime() for any other standard format. Returns a 
    DataFrame with a timezone-aware DatetimeIndex and a single 'value' column.

    Parameters:
        fpath           : full path to the CSV file
        timestamp_col   : name of the datetime column in the CSV
        value_col       : name of the measurement value column in the CSV
        sep             : seperator for measurement values in the CSV. Default DWD seperator is ','
    '''

    df = pd.read_csv(fpath,sep=sep,index_col=False,skipinitialspace=True)
    df.columns = df.columns.str.strip()

    # Validate columns exist before proceeding
    # Read as: "For each column name c in the list [timestamp_col, value_col], include it in the result 
    # list if it is NOT present in df.columns"
    missing = [c for c in [timestamp_col,value_col] if c not in df.columns] 
    if missing:
        raise KeyError(
            f"Expected columns {missing} not found in {fpath}.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Update col_map in build_annual_profiles() to match"
            f"your datasource column names."
        )
    
    # Parse timestamp, try ISO first as the DWD default, fall back to pandas
    try:
        df["timestamp"] = pd.to_datetime(
            df[timestamp_col].astype(str), format= "%Y-%m-%dT%H:%M:%S"
        )
    except ValueError:
        df["timestamp"] = pd.to_datetime(df[timestamp_col])

    # promotes timestamp column to become the row index and sorts rows chronologically
    df = df.set_index("timestamp").sort_index() 
    
    # DWD timestamps are in UTC which are converted to Europe/Berlin to correctly display weather patterns
    df.index = df.index.tz_localize("UTC").tz_convert(TIMEZONE) 

    # converts the value column to numbers and replaces all non-numeric data like - or n/a to NaN
    df["value"] = pd.to_numeric(df[value_col], errors="coerce")
    return df[["value"]] 

def load_dwd_solar(data_dir: str,
                   file_map: dict = None,
                   col_map: dict = None) -> pd.DataFrame:
    '''
    Loads GHI from station 691 or custom source.
    
    Unit conversion (DWD only):
        GHI [W/m2] = Wert [J/cm2]  x 10,000 / 600 (Irradiance = (Energy/Area)/Time)
    For non-DWD sources the value column is assumed to already be in W/m2  
    
    Returns:
    pd.Dataframe column: ghi_wm2, timezone-aware index
    '''
    cm      = col_map or DWD_COL_MAP # Dict with key-value pairs mapping logical names to actual column names in CSV
    fpath   = find_dwd_file(data_dir, "RAD-G", file_map) 
    df      = read_weather_csv(fpath, cm["timestamp"], cm["solar"], sep=cm.get("sep",","))

    # Unit conversion only for DWD (J/cm2 -> W/m2)
    # Non-DWD sources expected to provide W/m2 directly
    # if no custom file map was provided OR if RAD-G is not in the custom file map, then we default to DWD file
    if file_map is None or "RAD-G" not in file_map:
        df["value"] = df ["value"] * 10_000 / 600
    
    df["ghi_wm2"] = df["value"].clip(lower=0)
    return df[["ghi_wm2"]].dropna() 

def load_dwd_wind(data_dir: str,
                   file_map: dict = None,
                   col_map: dict = None) -> pd.DataFrame:
    '''
    Loads mean wind speed at 10m from station 691 or custom source. 
    Value column assumed to be in m/s for all sources

    Returns:
    pd.Dataframe column: wind_speed_ms, timezone-aware index
    '''
    cm      = col_map or DWD_COL_MAP
    fpath   = find_dwd_file(data_dir, "F", file_map)
    df      = read_weather_csv(fpath, cm["timestamp"], cm["wind"], sep=cm.get("sep",","))
    df["wind_speed_ms"] = df["value"].clip(lower=0)
    return df[["wind_speed_ms"]].dropna()

def load_dwd_temperature(data_dir: str,
                        file_map: dict = None,
                        col_map: dict = None) -> pd.DataFrame:
    '''
    Loads air temperature at 2m above ground from  from station 691 or custom source. 
    Value column assumed to be in degC for all sources

    2m height is the WMO standard (see WMO 2023 Section 5.4.5 Vertical positioning) and matches the 
    reference height used in NOCT datasheets, ensuring physical consistency in cell temperature
    correction.

    Returns:
    pd.Dataframe column: temp_air_c, timezone-aware index
    '''
    cm      = col_map or DWD_COL_MAP
    fpath   = find_dwd_file(data_dir, "T2M", file_map)
    df      = read_weather_csv(fpath, cm["timestamp"], cm["temp"], sep=cm.get("sep",","))
    df["temp_air_c"] = df["value"]
    return df[["temp_air_c"]].dropna()


# ===========================================================================
# 3.  PV power conversion  (pvlib POA + NOCT temperature correction)
# ===========================================================================

def compute_pv_profile(solar_df:pd.DataFrame,
                       p_rated_mw:float,
                       temp_series:pd.Series = None) -> pd.Series:
    '''
    Converts GHI time series to PV AC output for one unit.

    Conversion Chain:
    GHI [W/m2]
        -> Erbs decomposition -> DNI + DHI
        -> pvlib get_total_irradiance -> POA [W/m2] (30 deg tilt, south)
        -> NOCT cell temperature (measured 2m air temp or 15 degC fallback)
        -> temp-corrected efficiency: eta = eta_ref * [1 - beta*(T-T_ref)]
        -> DC = AC Power [MW]
    
    Parameters:
        solar_df    : DataFrame with column ghi_mw2 (timezone-aware index)
        p_rated_mw  : nameplate PV capacity [MW]
        temp_series : optional series of 2m air temperature [degC],
                      compatible index with solar_df. Falls back to 15
                      degC Bremen annual mean if None.

    Returns:
    pd.Series AC power [MW], same index as solar_df 
    '''
    try:
        import pvlib
    except ImportError:
        raise ImportError("pvlib not installed. Run: pip install pvlib")
    
    pv_ac = pd.Series(0.0, index=solar_df.index)
    location = pvlib.location.Location(
        latitude=LATITUDE, longitude=LONGITUDE, tz=TIMEZONE
    )
    times = solar_df.index
    solar_pos = location.get_solarposition(times)
    solar_pos = solar_pos[~solar_pos.index.duplicated(keep="first")]

    # Erbs decomposition: GHI -> DNI + DHI
    erbs = pvlib.irradiance.erbs(
        ghi=solar_df["ghi_wm2"],
        zenith=solar_pos["apparent_zenith"],
        datetime_or_doy=times
    )

    # Plane-of-Array irradiance (south-facing, 30 degree tilt)
    poa_components = pvlib.irradiance.get_total_irradiance(
        surface_tilt=SURFACE_TILT,
        surface_azimuth=SURFACE_AZIMUTH,
        dni=erbs["dni"],
        ghi=solar_df["ghi_wm2"],
        dhi=erbs["dhi"],
        solar_zenith=solar_pos["apparent_zenith"],
        solar_azimuth=solar_pos["azimuth"]
    )

    poa_irr = poa_components["poa_global"].fillna(0).clip(lower=0)

    # Low-irradiance cut-off (inverter minimum input threshold)
    # keep any POA value that is at or above the threshold, replace everything below with 0
    poa_irr = poa_irr.where(poa_irr >= GHI_MIN_THRESHOLD, other=0.0)

    # Cell temperature via NOCT model
    # T_cell = T_ambient + (NOCT - 20) * (POA/800)
    if temp_series is not None:
        t_amb = temp_series.reindex(times,method="nearest").fillna(15.0)
    else:
        t_amb = 15.0 #Fallback: Bremen annual mean
    t_cell = t_amb + (NOCT - 20.0) * (poa_irr / 800)

    # Temperature-corrected efficiency relative to STC
    eta_ratio = 1.0 - BETA_TEMP * (t_cell - T_REF)

    # DC -> AC (clip to rated capacity)
    p_dc = p_rated_mw * (poa_irr / G_STC) * eta_ratio
    p_ac = (p_dc * ETA_INVERTER).clip(lower=0.0,upper=p_rated_mw)
    pv_ac = p_ac.rename(None)
    return pv_ac.reindex(solar_df.index).fillna(0.0)

# ===========================================================================
# 4.  Wind power conversion  (piecewise cubic approximation)
# ===========================================================================

def compute_wind_profile(wind_df: pd.DataFrame,
                         p_rated_mw: float) -> pd.Series:
    '''
    Converts 10-min mean wind speed to wind power output

    Power curve:
        P = 0                                           v < v_cut_in
        P = p = p_rated_mw * 
            (v³ - V_CUT_IN³) / (V_RATED³ - V_CUT_IN³)   v_cut_in <= v < v_rated (12 m/s)
        P = P_rated                                     v_rated <= v < v_cut_out (25 m/s)
        P = 0                                           v >= v_cut_out

    Parameters:
        wid_df      : DataFrame with column wind_speed_ms 
        p_rated_mw  : name plate widn turbine capacity [MW]

    Returns:
    pd.Series AC Power [MW], same index as wind_df
    '''

    v = wind_df["wind_speed_ms"].values
    p = np.zeros(len(v))

    mask_partial = (v>V_CUT_IN) & (v<V_RATED) # bool array
    mask_rated = (v>=V_RATED) & (v<V_CUT_OUT)

    denom = V_RATED**3 - V_CUT_IN**3
    p[mask_partial] = p_rated_mw * ((v[mask_partial]**3 - V_CUT_IN**3) / denom) # formula applied only to where mask=True
    p[mask_rated] = p_rated_mw

    return pd.Series(p, index=wind_df.index).clip(lower=0.0,upper=p_rated_mw)

# ===========================================================================
# 5.  BDEW 2025 load profile generation
# ===========================================================================

# When no semantic load type metadata is available, statistically representative
# LV mixed-use composition will be used based on typical residential LV feeder
# composition i.e. predominantly residential but inclusive of small commercial 
# premises (bakeries, offices, small shops)

LV_MIXED_USE_SHARES = {
    "h25":  0.82,    # residential households
    "g25":  0.15,    # small commercial / office / retail
    "l25":  0.03     # agricultural (relevant for rural Kerber variants)
}

def assign_bdew_type(name:str,
                     load_type: str) -> str:
    '''
    Maps pandapower load name/type to a BDEW 2025 profile key.
    Returns None if no keyword match found - triggers mixed-use assignment
    '''
    combined = (str(name) + " " + str(load_type)).lower()
    for keyword, bdew_key in LOAD_TYPE_MAP.items():
        if keyword in combined:
            return bdew_key
    return None

def assign_mixed_use_types(net) -> dict:
    '''
    Calculates how many loads get each type (82% × n_loads for H25, 15% for G25, 3% for L25), 
    builds a list with that many entries of each type, shuffles it randomly (but reproducibly 
    due to the seed), then zips that shuffled list with net.load.index to produce a dictionary 
    mapping each load index to a profile type when no semantic type metadata is present in
    net.load.
    Assignment is deterministic (seeded) so results are reproducible across runs. So every time 
    the code is run, the same load buses get assigned H25, G25, L25 in the same pattern. Without 
    a seed the assignment would change every run, making the simulation results non-reproducible. Loads
    with existing metadata from assign_bdew_type are not overridden

    Returns:
    dict load index -> BDEW 2025 profile key
    '''
    import math
    rng = np.random.default_rng(seed=42) # Starting number for the random algorithm
    n_loads = len(net.load)
    types = list(LV_MIXED_USE_SHARES.keys()) # ["h25", "g25", "l25"]
    shares = list(LV_MIXED_USE_SHARES.values()) # [0.82, 0.15, 0.03]

    # Computes exact integer counts preserving total
    counts = [math.floor(s*n_loads) for s in shares]
    remainder = n_loads - sum(counts)

    # Assign remainder to residential as flooring looses fractions
    counts[0] += remainder

    #Builds a flat list: ["h25","h25",...×18, "g25","g25"] — 18 copies of h25 
    # followed by 2 copies of g25. Not shuffled yet
    assignment = []
    for bdew_type, count in zip(types,counts):
        assignment.extend([bdew_type] * count)
    # Build shuffled assignment array
    rng.shuffle(assignment)

    return {idx: bdew_type
            for idx, bdew_type in zip(net.load.index, assignment)}

def compute_load_profiles_bdew(net,
                               times: pd.DatetimeIndex) -> pd.DataFrame:
    '''
    Generates per-load annual BDEW 2025 profiles scaled to net.load.p_mw
    Uses H25 (residential), G25 (all commercial/industrial) and L25 (agricultural)
    from oemof.demand, based on March 2025 BDEW release.

    Parameters:
    net     : Pandapower network
    times   : DateTimeIndex at target resolution, timezone-aware (Europe/Berlin)

    Returns:
    pd.DataFrame index = times, columns = load element indices, values in MW 
    '''

    try:
        from oemof.demand import bdew
    except ImportError:
        raise ImportError(
            "oemof.demand not installed"
            "Run: pip install oemof.demand"
        )
    
    year = times[0].year

    # Holidays: oemof.demand expects list of datetime.date objects
    try:
        from workalendar.europe import Germany
        holidays = [date for date,_ in Germany().holidays(year)]
    except ImportError:
        holidays = []

    # Map each load to its 2025 BDEW type. First attempt keyword-based
    # assignment from name/type metadata. Loads with no recognisable
    # metadata (return None) fall into the mixed-use statistical assignment
    load_bdew_type = {}
    metadata_hits = 0
    for idx, row in net.load.iterrows():
        result = assign_bdew_type(
            row.get("name",""), row.get("type","")
        )  
        if result is not None:
            load_bdew_type[idx] = result
            metadata_hits += 1
    
    if metadata_hits < len(net.load):
        # Some or all loads have no metadata - apply mixed-use composition
        mixed = assign_mixed_use_types(net)
        for idx, bdew_type in mixed.items():
            if idx not in load_bdew_type:
                load_bdew_type[idx] = bdew_type
        n_mixed = len(net.load) - metadata_hits
        print(f"[Profile Builder] Mixed-use assignment applied to "
              f"{n_mixed} loads without metadata "
              f"(H25: {LV_MIXED_USE_SHARES['h25']*100:.0f}%, "
              f"G25: {LV_MIXED_USE_SHARES['g25']*100:.0f}%, "
              f"L25: {LV_MIXED_USE_SHARES['l25']*100:.0f}%).")  

    # Build profile class map - fallback to H25 if G25/L25 unavailable

    '''getattr(object, name, default) fetches an attribute from an object by name as a string
    getattr(bdew, "H25", None) is equivalent to writing bdew.H25 but with a fallback of None 
    if that class does not exist. profile_class_map[key] = cls then stores the class itself 
    (not an instance) in the dictionary under that key for later use'''

    profile_class_map = {}
    for key, attr in [("h25","H25"),("g25","G25"),("l25","L25")]:
        cls = getattr(bdew,attr,None)
        if cls is None:
            print(f"[profile_builder] WARNING: oemof.demand has no {attr}, "
                  f"falling back to H25 for '{key}' loads.")
            cls = bdew.H25
        profile_class_map[key] = cls
    
    # Generate one normalised profile per unique type
    raw_profiles = {}
    for profile_type in set(load_bdew_type.values()):
        cls = profile_class_map.get(profile_type, bdew.H25)
        raw = cls(times,holidays)
        if not isinstance(raw,pd.Series):
            raw = pd.Series(raw, index=times)
        else:
            raw = raw.reindex(times,method="nearest")
        raw_profiles[profile_type] = raw
    
    # Scale each load to rated p_mw via peak normalization
    '''
    Creates an empty DataFrame with times as the row index. Then for each load index idx, 
    it fetches the corresponding raw profile shape, finds its peak value, divides by that 
    peak to get a normalised 0-to-1 shape, fetches the rated power from net.load, and 
    multiplies to get actual MW values. Each load gets its own column in the result DataFrame.
    '''

    result = pd.DataFrame(index=times,dtype=float)
    for idx, bdew_type in load_bdew_type.items():
        col     = raw_profiles[bdew_type]
        peak    = col.max()
        norm    = col / peak if peak > 0 else col
        p_rated = float(net.load.at[idx,"p_mw"])
        result[idx] = norm * p_rated
    result = result.clip(lower=0.0).fillna(0.0)
    return result

# ===========================================================================
# 6.  Extreme day identification
# ===========================================================================

def find_extreme_days(profiles:dict) -> dict:
    '''
    Auto identifies the four most informative days for zoom-in plots

    Returns dict with keys max_der, min_der, max_load, min_load as YYYY-MM-DD
    strings. Returns None for any day that cannot be determined e.g. no wind or
    PV units in the network
    '''

    def safe_sum(df):
        if df is None or df.empty:
            return pd.Series(0,index=profiles["times"])
        return df.sum(axis=1)
    
    
    total_pv = safe_sum(profiles.get("pv")) # sum of all 102 PV generators
    total_wind = safe_sum(profiles.get("wind")) # sum of all wind generators
    total_load = safe_sum(profiles.get("load")) # sum of all loads
    total_der = total_pv + total_wind

    #Calculates average daily mean and finds the date with highest or lowest daily mean
    def peak_day(series:pd.Series, find_max:bool): 
        if series.empty or series.max() == 0:
            return None
        daily = series.resample("D").mean()
        ts = daily.idxmax() if find_max else daily.idxmin()
        return ts.strftime("%Y-%m-%d")
    
    return {
        "max_der":  peak_day(total_der, find_max=True),
        "min_der": peak_day(total_der,find_max=False),
        "max_load": peak_day(total_load,find_max=True),
        "min_load": peak_day(total_load,find_max=False)
    }

# ===========================================================================
# 7.  Main entry point
# ===========================================================================

def build_annual_profiles(
        net,
        net_name: str,
        data_dir:   str = "data/dwd",
        simbench_code:  str = None,
        file_map:   dict=None,
        col_map:    dict=None
) -> dict:
    '''
    Builds annual 10-min profiles for all loads, PV sgens, and wind sgens.

    Parameters:
        net             : pandapower network object (already loaded)
        net_name        : string identifier e.g. '1-MV-rural--2-sw', 'cigre_mv'
        data_dir        : folder containing local DWD CSV files (used only for 
                          non-SimBench networks)
        simbench_code   : SimBench grid code, required for SimBench networks,
                          e.g. '1-MV-rural--2--sw'
        file_map        : optional dict mapping parameter codes to filenames
                          e.g. {
                         'RAD-G': 'metmast_solar_2024.csv',
                         'F':     'metmast_wind_2024.csv',
                         'T2M':   'metmast_temp_2024.csv',
                          }
                          If None, DWD CDC naming convention is assumed.
        col_map         : Dict mapping the column names to parameters. If not specified 
                          DWD CDC default column naming will be followed.
    
    Returns:
    dict with keys:
        'load'           pd.DataFrame   index=timestamps    cols=load indices [MW]
        'pv'             pd.DataFrame   index=timestamps    cols=sgen indices [MW]
        'wind'           pd.DataFrame   index=timestamps    cols=sgen indices [MW]
        'times'          pd.DatetimeIndex
        'extreme_days'   dict   (max_der,min_der,max_load,min_load)
        'net_type'       str    ('simbench','cigre', or 'fallback')
    '''

    import simbench as sb
    
    net_type = detect_network_type(net_name)
    print(f"[profile builder] Network: '{net_name}' -> type: '{net_type}'")

    # -----------------------------------------------------------------------
    # SimBench path
    # -----------------------------------------------------------------------
    if net_type == "simbench":
        if simbench_code is None:
            raise ValueError(
                "simbench_code must be provided for SimBench Networks, "
                "e.g. simbench_code='1-MV-rural--2-sw'"   
            )
        print(f"[profile_builder] Loading SimBench profiles: {simbench_code}")
        raw_net = sb.get_simbench_net(simbench_code)
        profiles = sb.get_absolute_values(
            raw_net, profiles_instead_of_study_cases=True
        )

        times = profiles[("load","p_mw")].index

        # SimBench returns integer step index (0..35135) not timestamps.
        # 35136 steps = 366 days × 96 intervals = leap year 2016 at 15-min resolution.
        # Reconstruct DatetimeIndex using SimBench's internal reference year.
        if not isinstance(times, pd.DatetimeIndex):
            times = pd.date_range(
                start="2016-01-01",
                periods=len(times),
                freq="15min",
                tz="Europe/Berlin"
        )
        load_df = profiles[("load","p_mw")].copy()
        load_df.index = times
        load_df = load_df.clip(lower=0.0)
        sgen_prof = profiles[("sgen","p_mw")].copy()
        sgen_prof.index = times

        pv_mask = net.sgen["type"].str.lower().str.contains("pv|solar", na=False)
        wind_mask = net.sgen["type"].str.lower().str.contains("wind", na=False)

        pv_idx = net.sgen[pv_mask].index
        wind_idx = net.sgen[wind_mask].index

        pv_df = sgen_prof[
            [i for i in pv_idx if i in sgen_prof.columns]
        ].copy()

        # Zero out physically impossible night-time PV generation.
        # SimBench commercial/semiurb networks contain small non-zero
        # PV values at night due to profile scaling.
        night_mask = (pv_df.index.hour >= 22) | (pv_df.index.hour <= 4)
        pv_df.loc[night_mask] = 0.0

        wind_df = sgen_prof[
            [i for i in wind_idx if i in sgen_prof.columns]
        ].copy()

        # Clip physical bounds — SimBench profiles can contain small
        # negative values due to numerical precision in profile scaling
        pv_df   = pv_df.clip(lower=0.0)
        wind_df = wind_df.clip(lower=0.0)

        result = {
            "load": load_df, "pv": pv_df, "wind": wind_df,
            "times":times, "net_type":net_type
        }
        result["extreme_days"] = find_extreme_days(result)
        print(f"[profile_builder] Done: {len(times)} timesteps | "
              f"{load_df.shape[1]} loads |"
              f"{pv_df.shape[1]} PV | {wind_df.shape[1]} wind")
        
        return result
    
    # -----------------------------------------------------------------------
    # CIGRE / Fallback path
    # -----------------------------------------------------------------------
    print(f"[profile_builder] Loading DWD data from '{data_dir}' ...")
    solar_df = load_dwd_solar(data_dir, file_map, col_map)
    wind_raw = load_dwd_wind(data_dir, file_map, col_map)
    temp_df = load_dwd_temperature(data_dir, file_map, col_map)

    solar_df = solar_df[~solar_df.index.duplicated(keep="first")]
    wind_raw = wind_raw[~wind_raw.index.duplicated(keep="first")]
    temp_df  = temp_df[~temp_df.index.duplicated(keep="first")]

    # Align all three DWD series to a common time index so only those data is present
    # where all three measurement are available. Afterwards using loc only those rows
    # are kept whose index appears in times guaranteeing identical lengths and indices
    times = (solar_df.index.intersection(wind_raw.index).intersection(temp_df.index))
    times = times[~times.duplicated()]

    solar_df = solar_df.loc[times]
    wind_raw = wind_raw.loc[times]
    temp_df = temp_df.loc[times]

    print(f"[profile_builder] DWD aligned: {times[0]} -> {times[-1]}"
          f"({len(times)}) timesteps")
    
    # Load Profiles
    print("[profile_builder] Generating BDEW 2025 load profiles ...")
    load_df = compute_load_profiles_bdew(net,times)
    print(f"[profile_builder] {load_df.shape[1]} load profiles generated.")

    # PV Profiles
    pv_mask = net.sgen["type"].str.lower().str.contains("pv|solar",na=False)
    pv_idx = net.sgen[pv_mask].index
    pv_df = pd.DataFrame(index=times, dtype=float)
    if len(pv_idx) > 0:
        print(f"[profile_builder] Computing PV profiles"
              f"for {len(pv_idx)} unit(s) ...")
        for idx in pv_idx:
            p_rated = float(net.sgen.at[idx,"p_mw"])
            pv_series = compute_pv_profile(
                solar_df, p_rated, temp_series=temp_df["temp_air_c"]
            )
            pv_series = pv_series[~pv_series.index.duplicated(keep="first")]
            pv_df[idx] = pv_series.reindex(times).fillna(0.0)
        
            # Zero out night-time PV (physically impossible generation)
            night_mask = (pv_df.index.hour >= 22) | (pv_df.index.hour <= 4)
            pv_df.loc[night_mask] = 0.0
    else:
        print("[profile_builder] No PV sgens found in network.")

    # Wind Profiles
    wind_mask = net.sgen["type"].str.lower().str.contains("wind",na=False)
    wind_idx = net.sgen[wind_mask].index
    wind_df = pd.DataFrame(index=times, dtype=float)
    if len(wind_idx) > 0:
        print(f"[profile_builder] Computing Wind profiles"
              f"for {len(wind_idx)} unit(s) ...")
        for idx in wind_idx:
            p_rated = float(net.sgen.at[idx,"p_mw"])
            wind_df[idx] = compute_wind_profile(wind_raw,p_rated).values
    else:
        print("[profile_builder] No Wind sgens found in network.")

    result = {
        "load": load_df, "pv":pv_df, "wind":wind_df,
        "times": times, "net_type": net_type
    }

    result["extreme_days"] = find_extreme_days(result)
    print(f"[profile_builder] Done: {len(times)} timesteps | "
        f"{load_df.shape[1]} loads |"
        f"{pv_df.shape[1]} PV | {wind_df.shape[1]} wind")
        
    return result

