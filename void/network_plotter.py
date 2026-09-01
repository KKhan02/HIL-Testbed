''' 
Visualization Layer for the HIL testbed

plot_topology(net, net_name, schematic_dir,save_path):
    Standalone topology figure. Behaviour is network type aware:

    - Simbench:         two panels always produced (no external PNG needed):
                        Left  - geographical plot using embedded geodata
                        Right - structured tree/schematic layout using generic 
                                coordinates on a copy, showing clear topology
    - CIGRE / Kerber / Dickert / Synthetic LV
                        : schematic PNG (left) + pandapower simple_plot (right)
                          if a matching PNG is found in schematic_dir; otherwise fall 
                          back to single structured simple_plot. PNG files should be 
                          placed in data/schematics/ and named to match the net_name
                          passed to build_annual_profiles(), 
                          e.g.: cigre_mv_with_der.png, cigre_lv.png, 
                          kerber_landnetz_kabel_1.png etc

plot_profiles(net_name, profiles, save_path):
    Standalone annual profile figure.
    Three subplots - aggregate load, PV, wind - each showing a light 10-min
    fill and a daily mean line for readbility
    Extreme day markers annotated on all three panels.

plot_network_and_profiles (net, net_name, profiles, schematic_dir, save_path):
    Convenience wrapper: calls plot_topology() and then plot_profiles()
    Kept for backward compatibility

plot_day(profiles, date_str,net_name,save_path):
    Zooms into a single calendar day at full 10-min resolution. 
    Pass any YYYY-MM-DD string or one of profiles["extreme_days"] values

plot_extreme_days(profiles, net_name, save_dir):
    Calls plot_day() for all four auto_detected extreme days

Usage example
-------------
    import simbench as sb
    from profile_builder import build_annual_profiles
    from network_plotter  import plot_network_and_profiles, plot_day
 
    net  = sb.get_simbench_net("1-MV-rural--2-sw")
    prof = build_annual_profiles(net, "1-MV-rural--2-sw",
                                 simbench_code="1-MV-rural--2-sw")
 
    plot_topology(net, "1-MV-rural--2-sw")
    plot_profiles("1-MV-rural--2-sw", prof)
    plot_day(prof, prof["extreme_days"]["max_der"], "1-MV-rural--2-sw")
 
Note on SimBench study cases
-----------------------------
SimBench provides two data modes via sb.get_absolute_values():
  profiles_instead_of_study_cases=True  — full annual 35,136-step time series
                                          (what we use: required for HIL loop,
                                           scenario comparison, hosting capacity)
  profiles_instead_of_study_cases=False — four predefined snapshot operating
                                          points (max/min load × max/min gen).
                                          Useful for quick worst-case checks
                                          only; incompatible with HIL architecture.
''' 

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import pandapower as pp
import pandapower.plotting as pplot
from pandapower.plotting import simple_plot

warnings.filterwarnings("ignore")

# ===========================================================================
# Image loading  (PNG / JPG / WebP — format-agnostic via Pillow)
# ===========================================================================

def load_image(fpath:str) -> np.ndarray:
    '''
    Loads any image format supported by Pillow (JPG, PNG, WebP, etc.) and 
    returns it as a RGB numpy array suitable for imshow()

    matplotlib.image.imread() only handles JPG/PNG natively and silently
    fails on WebP.

    Raises FileNotFoundError if fpath does not exist.
    Raises ImportError if Pillow is somehow not installed
    '''
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for image loading. "
            "Run: pip install Pillow"
        )
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Image not found: {fpath}")
    with Image.open(fpath) as img:
        img.load()
    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        mask = img.split()[-1] if img.mode == "RGBA" else None
        background.paste(img, mask=mask)
        img = background
    else:
        img = img.convert("RGB")
    return np.asarray(img)

# ---------------------------------------------------------------------------
# Colour palette  (consistent across all figures)
# ---------------------------------------------------------------------------

C_LOAD  = "#2166ac" # blue
C_PV    = "#f4a582" # orange-yellow
C_WIND  = "#4dac26" # green
C_FILL  = 0.18        # alpha for area fill

# Default directory for schematic PNG Files
DEFAULT_SCHEMATIC_DIR = "data/schematics"

# Supported image extensions, tried in preference order
IMAGE_EXTENSIONS = (".webp", ".png", ".jpg", ".jpeg")

# Schematic filename stems for non-SimBench networks.
# Keys are substrings matched case-insensitively against net_name.
# Values are filename stems WITHOUT extension — the loader tries each
# extension in IMAGE_EXTENSIONS automatically
# SimBench is intentionally absent — geographic + structured panels are
# generated from the network object directly; no PNG/WebP exists or is needed.
SCHEMATIC_MAP = {
    # ------------------------------------------------------------------
    # CIGRE — only in-scope variants (pv_wind MV + LV)
    # HV is transmission level (out of scope).
    # with_der="all" variant is not used in this project.
    # ------------------------------------------------------------------
    "cigre_mv_with_der":              "cigre_network_mv_der",
    "cigre_lv":                       "cigre_network_lv",
 
    # ------------------------------------------------------------------
    # Kerber standard
    # kabel_1 and kabel_2 share one topology diagram (differ only in
    # cable cross-sections). Same for freileitung_1 and _2.
    # vorstadtnetz_kabel_1 and _2 have separate named schematics (a / b).
    # ------------------------------------------------------------------
    "kerber_landnetz_kabel":          "kerber_landnetz_kabel",
    "kerber_landnetz_freileitung":    "kerber_landnetz_freileitung",
    "kerber_vorstadtnetz_kabel_1":    "kerber_vorstadtnetz_a",
    "kerber_vorstadtnetz_kabel_2":    "kerber_vorstadtnetz_b",
    "kerber_dorfnetz":                "kerber_dorfnetz_1",
 
    # ------------------------------------------------------------------
    # Kerber extreme
    # Most-specific keys listed; longest-first sort at lookup prevents
    # prefix collisions (e.g. _trafo variants checked before base).
    # ------------------------------------------------------------------
    # Landnetz — kabel (incl. trafo variant) → typ_1
    "kb_extrem_landnetz_kabel":       "kerber_extrem_landnetz_typ_1",
    # Landnetz — freileitung, two key spellings to match both net_names:
    #   kb_extrem_landnetz_freileitung   → pandapower fn name
    #   kb_extrem_landnetz_frltg_trafo   → test suite name for trafo variant
    "kb_extrem_landnetz_freileitung": "kerber_extrem_landnetz_typ_2",
    "kb_extrem_landnetz_frltg":       "kerber_extrem_landnetz_typ_2",
    # Dorfnetz — trafo variant must be listed before base (longer key → matched first)
    "kb_extrem_dorfnetz_trafo":       "kerber_extrem_dorfnetz_typ_2",
    "kb_extrem_dorfnetz":             "kerber_extrem_dorfnetz_typ_1",
    # Vorstadtnetz — four distinct schematics (a/b/c/d × typ_1/typ_2)
    "kb_extrem_vorstadtnetz_trafo_1": "kerber_extrem_vorstadt_b_typ_1",
    "kb_extrem_vorstadtnetz_trafo_2": "kerber_extrem_vorstadt_d_typ_2",
    "kb_extrem_vorstadtnetz_1":       "kerber_extrem_vorstadt_a_typ_1",
    "kb_extrem_vorstadtnetz_2":       "kerber_extrem_vorstadt_c_typ_2",
 
    # ------------------------------------------------------------------
    # Synthetic Voltage Control LV — one schematic per class
    # ------------------------------------------------------------------
    "synthetic_lv_rural_1":           "rural_1_en",
    "synthetic_lv_rural_2":           "rural_2_en",
    "synthetic_lv_village_1":         "village_1_en",
    "synthetic_lv_village_2":         "village_2_en",
    "synthetic_lv_suburb_1":          "suburb_1_en",
 
    # ------------------------------------------------------------------
    # Dickert LV — one representative schematic for all 18 combinations
    # (mid-range cable feeder shown; topology structure is the same
    #  across all feeder length / customer type / line type variants)
    # ------------------------------------------------------------------
    "dickert":  "example_of_dickert_lv-feeders_for_mid-range_cable",
}

# Networks whose net_name matches any of these substrings are treated as SimBench
SIMBENCH_KEYWORDS = (
    "1-mv-", "1-lv-", "1-mvlv-", "1-hv-", "1-ehv-",
    "simbench",
)

# ===========================================================================
# Internal helpers
# ===========================================================================

def is_simbench(net_name:str) -> bool:
    '''True if net_name looks like a SimBench network code. '''
    n = net_name.lower()
    return any(kw in n for kw in SIMBENCH_KEYWORDS)

def find_schematic(net_name: str, schematic_dir: str):
    '''
    Returns the path to the schematic image for the given network, or None.
    
    Search strategy:
        1. Match net_name against SCHEMATIC_MAP keys (longest key first to avoid
        prefix collisions). For each match, try every extension in IMAGE_EXTENSIONS

        2. Filesystem fallback: any file in schematic_dir whose stem share a token 
        longer than 3 characters with net_name, regardless of extension

        Returns None immediately for SimBench networks
    '''

    if is_simbench(net_name):
        return None
    
    if not os.path.isdir(schematic_dir):
        return None
    
    n = net_name.lower()

    # Sort by key length descending so more-specific keys are tried first
    for key in sorted(SCHEMATIC_MAP,key=len,reverse=True):
        if key in n:
            stem = SCHEMATIC_MAP[key]
            for ext in IMAGE_EXTENSIONS:
                candidate = os.path.join(schematic_dir, stem + ext)
                if os.path.isfile(candidate):
                    return candidate
    
    # Fallback: any supported image whose stem shared a long token with net_name
    supported = set(IMAGE_EXTENSIONS)
    for fname in os.listdir(schematic_dir):
        root, ext = os.path.splitext(fname)
        if ext.lower() not in supported:
            continue
        if any(
            part in root.lower()
            for part in n.replace("_","-").split("-")
            if len(part) > 3
        ):
            return os.path.join(schematic_dir, fname)
    
    return None
def needs_generic_coords(net) -> bool:
    '''True if the network has no meaningful geodata'''
    if "x" not in net.bus.columns or "y" not in net.bus.columns:
        return True
    valid = net.bus[["x", "y"]].dropna()
    return len(valid) < 2

def auto_bus_size(net) -> float:
    '''
    Returns a bus marker size scaled to the network bounding box so that
    the markers look proportional regardless of network extent
    Logic:
    1. Get the bounding box: find the x-width and y-height of all buses in data coordinates
    2. Take the larger dimension: so the scaling is governed by whichever axis the network 
        is more spread out along
    3. 1.2% of that as the marker diameter: purely empirical; keeps markers visible but not 
        overlapping on both dense and sparse networks
    4. Clip to [0.01, 500.0]: The floor stops markers disappearing on very compact networks; 
        the ceiling stops SimBench markers (which would otherwise compute to ~600) from filling 
        the entire axes
    '''
    if "x" not in net.bus.columns or "y" not in net.bus.columns:
        return 0.08

    gd = net.bus[["x", "y"]].dropna()
    if len(gd) < 2:
        return 0.08

    x_range = float(gd["x"].max() - gd["x"].min())
    y_range = float(gd["y"].max() - gd["y"].min())
    extent  = max(x_range, y_range)

    if extent == 0:
        return 0.08

    raw = float(np.clip(extent * 0.012, 0.01, 500.0))

    # If computed size is suspiciously small, coordinates are likely in
    # normalised [0,1] range from create_generic_coordinates — scale up
    if raw < 0.5:
        raw = raw * 80.0

    return raw

def aggregate(df: pd.DataFrame) -> pd.Series:
    '''Sum across all colums for each row; returns empty Series if df is None/empty'''
    if df is None or df.empty:
        return pd.Series(dtype=float)
    return df.select_dtypes(include="number").sum(axis=1)

def topology_legend(ax):
    '''Adds a compact topology legend to an axes.
    simple_plot draws the network but doesn't add a legend automatically
    This fucntion manually creates one by using dummy coordinates and 
    markers with the same color as what is plotted on the simple_plot
    '''
    legend_elements = [
        Line2D([0], [0], color="#1f77b4", lw=1.5,
               label="Lines / cables"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="black", markersize=7,    label="Buses"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#ffdd00", markeredgewidth=0.8, markersize=8,  label="Sgens (DER)"),
        Line2D([0], [0], marker="v", color="w",
               markerfacecolor="black", markersize=7,    label="Loads"),
    ]
    ax.legend(handles = legend_elements, fontsize=8,
              loc="lower right", framealpha=0.85)
    
def profile_axis(ax, series: pd.Series, label: str, color: str,
                 unit: str = "MW", daily_stat="mean"):
    '''
    Plots an annual aggregate profile on ax. Shows a light 10-min fill for
    texture and a daily mean line for readability
    '''
    #If the Series is empty (no units of that type in the network), it writes 
    # a grey "No X units in network" preventing matplotlib crashing
    if series.empty:
        ax.text(0.5, 0.5, f"No {label} units in network",
                ha = "center", va = "center", transform=ax.transAxes,
                fontsize = 11, color = "grey")
        ax.set_ylabel(f"{label} [{unit}]", fontsize = 10)
        return
    
    daily_mean = series.resample("D").mean().interpolate(method="time", limit=14)
    daily_min  = series.resample("D").min().interpolate(method="time", limit=14)
    daily_max  = series.resample("D").max().interpolate(method="time", limit=14)

    # Choose central line based on stat argument
    if daily_stat == "max":
        daily_line = daily_max
        line_label = f"{label} (daily max)"
    else:
        daily_line = daily_mean
        line_label = f"{label} (daily mean)"

    ax.fill_between(daily_mean.index, daily_min.values, daily_max.values,
                    alpha=0.30, color=color, linewidth=0,
                    label="Daily min–max range")
    ax.plot(daily_line.index, daily_line.values, color=color, linewidth=2.0,
            label=line_label)
    ax.set_ylabel(f"{label} [{unit}]", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)


def annotate_extreme_days(ax, series: pd.Series, extreme_days: dict,
                          day_keys: list, color:str):
    '''
    Draws vertical dotted lines and text labels on ax for the extreme days
    listed in day_keys (subset of extreme_days dict)
    '''
    if series.empty:
        return
    for key, label, va in day_keys:
        day_str = extreme_days.get(key)
        if day_str is None:
            continue
        day_ts = pd.Timestamp(day_str, tz=series.index.tz)
        #Timezone safe date comparision: normalize both side to midnight
        mask = series.index.normalize() == day_ts.normalize()
        #if the date isn't in the series at all (e.g. the profiles dict covers 
        # a different year), skip
        if not mask.any():
            continue
        day_val = series[mask].mean()
        ax.axvline(day_ts, color=color, lw=0.9, linestyle=":", alpha=0.75)
        ax.text(day_ts, day_val, f"  {label}", fontsize=7, color=color,
                va=va, rotation=90, clip_on=True)
        
# ===========================================================================
# Public function 1:  topology figure
# ===========================================================================

def plot_topology(
        net,
        net_name: str,
        schematic_dir: str = DEFAULT_SCHEMATIC_DIR,
        save_path: str = None,
        show=True,
        show_bus_ids = True,
):
    '''
    Standalone network topology figure.  Layout is network-type aware.
 
    SimBench networks:
    Left  — geographic plot using embedded EPSG:31467 geodata.
             Shows real feeder routing and spatial relationships
             (comparable to Figure 5 in the SimBench paper).
    Right — structured tree/schematic layout.
             A deep copy of the network is made so the original geodata
             is not disturbed; create_generic_coordinates() is applied
             to the copy to produce a clean topological diagram
             (comparable to Figure 7 in the SimBench paper).
 
    CIGRE / Kerber / Dickert / Synthetic LV:
    If a matching PNG is found in schematic_dir:
        Left  — reference schematic image
        Right — pandapower simple_plot (structured layout)
    If no PNG is found:
        Single full-width structured simple_plot with a footer note.
 
    Parameters
    ----------
    net           : pandapower network object (not modified)
    net_name      : human-readable identifier (title + schematic lookup)
    schematic_dir : folder for schematic PNG files (non-SimBench only).
                    Default: 'data/schematics/'.  PNG filenames should match
                    the net_name used in build_annual_profiles(), e.g.
                    'cigre_mv_with_der.png', 'kerber_landnetz_kabel_1.png'.
    save_path     : optional PNG save path
    '''
    import copy
    is_sb   = is_simbench(net_name)
    schematic_path = None if is_sb else find_schematic(net_name,schematic_dir)
    show_schematic = (not is_sb) and (schematic_path is not None)
    
    # -----------------------------------------------------------------------
    # Figure layout: always two panels
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16,8))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.06)
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])
    # -----------------------------------------------------------------------
    # Left panel
    # -----------------------------------------------------------------------
    if is_sb:
        # Use original geo data
        geo_bus_sz = auto_bus_size(net)
        simple_plot(net,
                    ax=ax_left,
                    plot_sgens=True,
                    plot_loads=True,
                    bus_size=geo_bus_sz,
                    line_width=1.2,
                    sgen_size=geo_bus_sz * 10.0,
                    load_size=geo_bus_sz * 6.0,
                    bus_color="#0066cc",
                    line_color="#0066cc",
                    trafo_color="#0066cc",
                    ext_grid_color="#ffcc00",
                    show_plot=False)
        if show_bus_ids and "x" in net_struct.bus.columns and "y" in net_struct.bus.columns:
            for bus_idx, row in net_struct.bus.iterrows():
                x, y = row.get("x"), row.get("y")
                if pd.notna(x) and pd.notna(y):
                    ax_right.annotate(
                        str(bus_idx),
                        xy=(x, y),
                        fontsize=5,
                        ha="center",
                        va="bottom",
                        color="#cc0000",
                        xytext=(0, 4),
                        textcoords="offset points",
                    )
        ax_left.set_title("Geographic Layout (EPSG:31467)", fontsize=11, pad=6)
        # Reduce sgen patch linewidth post-draw
        # simple_plot draws sgens as the last patch collections added
        for collection in ax_left.collections:
            if hasattr(collection, 'get_linewidth'):
                lw = collection.get_linewidth()
                # Sgen collections typically have thicker default linewidth
                if len(lw) > 0 and np.mean(lw) > 1.0:
                    collection.set_linewidth(0.85)                    
                    collection.set_edgecolor("black")
                    collection.set_facecolor("#ffdd00")
            
    elif show_schematic:
        # Reference schematic PNG
        ax_left.axis("off")
        img = load_image(schematic_path)
        ax_left.imshow(img, aspect="equal",interpolation="lanczos")
        ax_left.set_title("Reference schematic", fontsize=11, pad=6)

    else:
        # No PNG available
        ax_left.axis("off")
        ax_left.text(0.5, 0.5,
                     f"No Schematic PNG found.\n"
                     f"Place file in '{schematic_dir}/'.\n"
                     f"See SCHEMATIC_MAP for naming convention.",
                     ha="center", va="center", fontsize=9,
                     color="#888888", transform=ax_left.transAxes,
                     wrap=True)
        ax_right.set_title("Structured topology (schematic layout)", fontsize=11, pad=6)
        if show_bus_ids and "x" in net_struct.bus.columns and "y" in net_struct.bus.columns:
            for bus_idx, row in net_struct.bus.iterrows():
                x, y = row.get("x"), row.get("y")
                if pd.notna(x) and pd.notna(y):
                    ax_right.annotate(
                        str(bus_idx),
                        xy=(x, y),
                        fontsize=5,
                        ha="center",
                        va="bottom",
                        color="#cc0000",
                        xytext=(0, 4),
                        textcoords="offset points",
                    )
    
    # -----------------------------------------------------------------------
    # Right panel — structured / schematic layout
    # -----------------------------------------------------------------------

    if is_sb:
        # Deep copy: create_generic_coordinates must not overwrite real geodata
        net_struct = copy.deepcopy(net)
        pplot.create_generic_coordinates(net_struct, overwrite=True)
        struct_bus_sz = auto_bus_size(net_struct)
        simple_plot(net_struct,
                    ax=ax_right,
                    plot_sgens=True,
                    plot_loads=True,
                    bus_size=struct_bus_sz,
                    line_width=1.2,
                    sgen_size=struct_bus_sz * 10.0,
                    load_size=struct_bus_sz * 6.0,
                    bus_color="#0066cc",
                    line_color="#0066cc",
                    trafo_color="#0066cc",
                    ext_grid_color="#ffcc00",
                    show_plot=False
        )
        if show_bus_ids and "x" in net_struct.bus.columns and "y" in net_struct.bus.columns:
            for bus_idx, row in net_struct.bus.iterrows():
                x, y = row.get("x"), row.get("y")
                if pd.notna(x) and pd.notna(y):
                    ax_right.annotate(
                        str(bus_idx),
                        xy=(x, y),
                        fontsize=5,
                        ha="center",
                        va="bottom",
                        color="#cc0000",
                        xytext=(0, 4),
                        textcoords="offset points",
                    )
        if "x" in net_struct.bus.columns and "y" in net_struct.bus.columns:
            xs = net_struct.bus["x"].dropna()
            ys = net_struct.bus["y"].dropna()
            if len(xs) > 1 and len(ys) > 1:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                x_span = x_max - x_min
                y_span = y_max - y_min
                span = max(x_span, y_span)

                # Only scale when the network is "tiny"
                TINY_SPAN = 50.0  # adjust as needed (units from create_generic_coordinates)
                if 0 < span < TINY_SPAN:
                    # Dynamic padding: smaller span -> bigger padding (capped)
                    # padding_pct ranges from 0.30 to 0.80
                    padding_pct = min(0.80, max(0.30, (TINY_SPAN / span) * 0.10))
                    x_pad = x_span * padding_pct or 1.0
                    y_pad = y_span * padding_pct or 1.0
                    ax_right.set_xlim(x_min - x_pad, x_max + x_pad)
                    ax_right.set_ylim(y_min - y_pad, y_max + y_pad)

        ax_right.set_aspect("equal", adjustable="box")
        ax_right.margins(0.05)
        ax_right.set_title("Structured topology (schematic layout)", fontsize=11, pad=6)
        if show_bus_ids and "x" in net_struct.bus.columns and "y" in net_struct.bus.columns:
            for bus_idx, row in net_struct.bus.iterrows():
                x, y = row.get("x"), row.get("y")
                if pd.notna(x) and pd.notna(y):
                    ax_right.annotate(
                        str(bus_idx),
                        xy=(x, y),
                        fontsize=5,
                        ha="center",
                        va="bottom",
                        color="#cc0000",
                        xytext=(0, 4),
                        textcoords="offset points",
                    )
        # Reduce sgen patch linewidth post-draw
        # simple_plot draws sgens as the last patch collections added
        for collection in ax_right.collections:
            if hasattr(collection, 'get_linewidth'):
                lw = collection.get_linewidth()
                # Sgen collections typically have thicker default linewidth
                if len(lw) > 0 and np.mean(lw) > 1.0:
                    collection.set_linewidth(0.85)
                    collection.set_edgecolor("black")
                    collection.set_facecolor("#ffdd00")
    else:
        # Non SimBench: generate generic coords on a copy for structured view
        net_struct = copy.deepcopy(net)
        if needs_generic_coords(net_struct):
            pplot.create_generic_coordinates(net_struct, overwrite=True) 
        struct_bus_sz = auto_bus_size(net_struct)

        simple_plot(net_struct,
                    ax=ax_right,
                    plot_sgens=True,
                    plot_loads=True,
                    bus_size=struct_bus_sz,
                    line_width=1.2,
                    sgen_size=struct_bus_sz * 15.0,
                    load_size=struct_bus_sz * 10.0,
                    bus_color="#0066cc",
                    line_color="#0066cc",
                    trafo_color="#0066cc",
                    ext_grid_color="#ffcc00",
                    show_plot=False
        )
        ax_right.set_title("Network topology (pandapower)", fontsize=11, pad=6)
        if show_bus_ids and "x" in net_struct.bus.columns and "y" in net_struct.bus.columns:
            for bus_idx, row in net_struct.bus.iterrows():
                x, y = row.get("x"), row.get("y")
                if pd.notna(x) and pd.notna(y):
                    ax_right.annotate(
                        str(bus_idx),
                        xy=(x, y),
                        fontsize=5,
                        ha="center",
                        va="bottom",
                        color="#cc0000",
                        xytext=(0, 4),
                        textcoords="offset points",
                    )
        # After simple_plot(net_struct, ax=ax_right, ...)
        if "x" in net_struct.bus.columns and "y" in net_struct.bus.columns:
            xs = net_struct.bus["x"].dropna()
            ys = net_struct.bus["y"].dropna()
            if len(xs) > 1 and len(ys) > 1:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                x_span = x_max - x_min
                y_span = y_max - y_min
                span = max(x_span, y_span)

                # Only scale when the network is "tiny"
                TINY_SPAN = 50.0  # adjust as needed (units from create_generic_coordinates)
                if 0 < span < TINY_SPAN:
                    # Dynamic padding: smaller span -> bigger padding (capped)
                    # padding_pct ranges from 0.30 to 0.80
                    padding_pct = min(0.80, max(0.30, (TINY_SPAN / span) * 0.10))
                    x_pad = x_span * padding_pct or 1.0
                    y_pad = y_span * padding_pct or 1.0
                    ax_right.set_xlim(x_min - x_pad, x_max + x_pad)
                    ax_right.set_ylim(y_min - y_pad, y_max + y_pad)

        ax_right.set_aspect("equal", adjustable="box")
        ax_right.margins(0.05)
        # Reduce sgen patch linewidth post-draw
        # simple_plot draws sgens as the last patch collections added
        for collection in ax_right.collections:
            if hasattr(collection, 'get_linewidth'):
                lw = collection.get_linewidth()
                # Sgen collections typically have thicker default linewidth
                if len(lw) > 0 and np.mean(lw) > 1.0:
                    collection.set_linewidth(0.85)                    
                    collection.set_edgecolor("black")
                    collection.set_facecolor("#ffdd00")
    
    topology_legend(ax_right)

    # -----------------------------------------------------------------------
    # Network stats footer and title
    # -----------------------------------------------------------------------
    n_bus = len(net.bus)
    n_line = len(net.line) + len(net.trafo)
    n_sgen = len(net.sgen)
    n_load = len(net.load)
    stats = (f"{n_bus} buses | {n_line} lines/trafos | "
             f"{n_sgen} sgens | {n_load} loads")
    fig.text(0.5, 0.01, stats, ha="center", fontsize=8, color="#555555")
    fig.suptitle(f"HIL Testbed - {net_name}\nNetwork topology",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.06)
    if save_path:
        fig.savefig(save_path, dpi=600,bbox_inches="tight")
        print(f"[network_plotter] Topology figure saved -> {save_path}")
    if show:
        plt.show(block=True)
#        plt.waitforbuttonpress()
        plt.close(fig)
    else:
        plt.close(fig)
    return fig

    # ===========================================================================
    # Public function 2:  annual profiles figure
    # ===========================================================================

def plot_profiles(
            net_name: str,
            profiles: dict,
            save_path: str = None,
            show = True,
    ):
        '''
        Standalone annual profile figure
        Three subplots are generated (load/PV/Wind), each showing:
        - Light 10-min area fill (texture)
        - Daily-mean line (readability)
        - Extreme-Day markers (annotated on the relevant panel)

        Parameters:
        net_name: used for figure title only
        profiles: dict returned by annual_build_profiles()
        save_path: optional PNG save path
        '''
        load_s = aggregate(profiles.get("load"))
        pv_s = aggregate(profiles.get("pv"))
        wind_s = aggregate(profiles.get("wind"))
        ed = profiles.get("extreme_days", {})

        fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
        fig.subplots_adjust(hspace=0.12, top=0.92, bottom=0.08)

        profile_axis(axes[0], load_s, "Load",             C_LOAD, daily_stat="mean")
        profile_axis(axes[1], pv_s,   "PV generation",    C_PV,   daily_stat="max")
        profile_axis(axes[2], wind_s, "Wind generation",  C_WIND, daily_stat="mean")

        # Extreme day annotations
        # Load Panel: peak load day + min load day
        annotate_extreme_days(axes[0], load_s, ed,
                              [("max_load", "Peak load", "bottom"),
                               ("min_load", "Min load", "top")],
                               color=C_LOAD)
        # PV panel: max DER day (most generation, most likely to show overvoltage)
        annotate_extreme_days(axes[1], pv_s, ed,
                              [("max_der", "Max DER", "bottom"),
                               ("min_der", "Min DER", "top")],
                               color=C_PV)
        
        # Wind panel: max DER day + min DER day
        annotate_extreme_days(axes[2], wind_s, ed, 
                              [("max_der", "Max DER", "bottom"),
                               ("min_der", "Min DER", "top")],
                               color=C_WIND)
        year_str = ""
        if len(profiles.get("times", [])) > 0:
            year_str = f" | {profiles['times'][0].year}"
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        axes[-1].tick_params(axis="x", labelsize=9)
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        fig.suptitle(f"HIL Testbed - {net_name}{year_str}\n"
                     f"Annual DER & load profiles",
                     fontsize=13, fontweight="bold", y = 0.98)
        if save_path:
            fig.savefig(save_path, dpi=600, bbox_inches="tight")
            print(f"[network_plotter] Profiles figure saved -> {save_path}")

        if show:
            plt.show(block=False)
            plt.waitforbuttonpress()
            plt.close(fig)
        else:
            plt.close(fig)        
        return fig

# ===========================================================================
# Public function 3:  convenience wrapper (backward compatible)
# ===========================================================================
 
def plot_network_and_profiles(
    net,
    net_name: str,
    profiles: dict,
    schematic_dir: str = DEFAULT_SCHEMATIC_DIR,
    save_path: str = None,
    show=True,
):
    """
    Convenience wrapper: calls plot_topology() then plot_profiles().
    Produces two separate figures (avoids cramping topology and profiles
    into a single window).
 
    save_path, if given, is used as a base: topology is saved as
    <save_path>_topology.png and profiles as <save_path>_profiles.png.
    """
    topo_save    = None
    profile_save = None
    if save_path:
        base, ext    = os.path.splitext(save_path)
        ext          = ext or ".png"
        topo_save    = f"{base}_topology{ext}"
        profile_save = f"{base}_profiles{ext}"
 
    fig_topo = plot_topology(net, net_name,
                             schematic_dir=schematic_dir,
                             save_path=topo_save,show=show)
    fig_prof = plot_profiles(net_name, profiles,
                             save_path=profile_save,show=show)
    return fig_topo, fig_prof
 
 
# ===========================================================================
# Public function 4:  single-day zoom plot
# ===========================================================================
 
def plot_day(
    profiles: dict,
    date_str: str,
    net_name: str = "",
    day_label: str = "",
    save_path: str = None,
    show = True,
):
    """
    Zooms into a single calendar day showing load, PV, and wind at full
    10-min resolution on a shared time axis.
 
    Parameters
    ----------
    profiles  : dict returned by build_annual_profiles()
    date_str  : YYYY-MM-DD string, e.g. profiles["extreme_days"]["max_der"]
    net_name  : used only for the figure title
    save_path : optional PNG save path
    """
    if date_str is None:
        print(f"[network_plotter] plot_day skipped — no date for '{day_label}' "
            f"(network has no relevant DER).")
        return None
    
    if "times" not in profiles or len(profiles["times"]) == 0:
        print("[network_plotter] plot_day skipped — profiles has no times.")
        return None
    tz        = profiles["times"].tz
    day_start = pd.Timestamp(date_str, tz=tz)
    day_end   = day_start + pd.Timedelta(days=1)
 
    def _slice(df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(dtype=float)
        s = df.sum(axis=1)
        # Use half-open interval [day_start, day_end) to include all 144
        # 10-min intervals without duplicating midnight
        return s[(s.index >= day_start) & (s.index < day_end)]
 
    load_day = _slice(profiles.get("load"))
    pv_day   = _slice(profiles.get("pv"))
    wind_day = _slice(profiles.get("wind"))
 
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.subplots_adjust(hspace=0.10, top=0.90, bottom=0.09)
 
    datasets = [
        (axes[0], load_day, "Load",            C_LOAD,  "MW"),
        (axes[1], pv_day,   "PV generation",   C_PV,    "MW"),
        (axes[2], wind_day, "Wind generation",  C_WIND,  "MW"),
    ]
 
    for ax, series, label, color, unit in datasets:
        if series.empty:
            ax.text(0.5, 0.5, f"No {label} data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="grey")
            ax.set_ylabel(f"{label}\n[{unit}]", fontsize=10)
        else:
            ax.fill_between(series.index, series, alpha=0.22, color=color,
                            linewidth=0)
            ax.plot(series.index, series, color=color, linewidth=1.8,
                    label=label)
            ax.set_ylabel(f"{label}\n[{unit}]", fontsize=10)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3, linestyle="--")
 
            # Peak annotation
            if series.max() > 0:
                peak_t = series.idxmax()
                peak_v = series.max()
                ax.annotate(
                    f"Peak: {peak_v:.3f} MW",
                    xy=(peak_t, peak_v),
                    xytext=(10, -16), textcoords="offset points",
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8)
                )
 
            ax.legend(fontsize=9, loc="upper right")
 
    # Shared x-axis: hours
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].tick_params(axis="x", labelsize=9)
    axes[-1].set_xlabel("Time of day (10-min resolution)", fontsize=10)
    
    label_str = f"  [{day_label}]" if day_label else ""
    fig.suptitle(
        f"HIL Testbed — {net_name}  |  Day zoom: {date_str}{label_str}",
        fontsize=13, fontweight="bold", y=0.97
    )
 
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[network_plotter] Day figure saved → {save_path}")
 
    if show:
        plt.show(block=False)
        plt.waitforbuttonpress()
        plt.close(fig)
    else:
        plt.close(fig)
    return fig
 
 
# ===========================================================================
# Public function 5:  plot all four extreme days
# ===========================================================================
 
def plot_extreme_days(
    profiles: dict,
    net_name: str = "",
    save_dir: str = None,
    show=True,
):
    """
    Calls plot_day() for all four auto-detected extreme days.
 
    Parameters
    ----------
    profiles : dict from build_annual_profiles()
    net_name : used in figure titles
    save_dir : if given, saves each figure as a PNG in this directory
    """
    ed = profiles.get("extreme_days", {})
    labels = {
        "max_der":  "Max DER generation day",
        "min_der":  "Min DER generation day",
        "max_load": "Peak load day",
        "min_load": "Min load day",
    }
    for key, description in labels.items():
        day_str = ed.get(key)
        if day_str is None:
            print(f"[network_plotter] No extreme day found for '{key}', skipping.")
            continue
        print(f"[network_plotter] Plotting {description}: {day_str}")
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            safe_name = net_name.replace(" ", "_").replace("/", "-")
            save_path = os.path.join(
                save_dir, f"{safe_name}_{key}_{day_str}.png"
            )
        plot_day(profiles, day_str, net_name=net_name,day_label=description, save_path=save_path, show=show)
 

