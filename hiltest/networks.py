"""
hiltest/networks.py
====================
Representative network catalogue — pure data specs plus deferred-loader
factory functions.

Design
------
The spec lists (REPRESENTATIVE_NETWORK_SPECS, REPRESENTATIVE_NETWORK_PLOTTER_SPECS)
are pure Python tuples with no heavy imports. They can be read safely at any
import time including during --help.

The loader functions (get_representative_networks, get_representative_networks_plotter)
import pandapower and simbench inside their bodies so the import only happens
when a section actually runs, not at CLI parse time.

Section files call these functions inside their run_*_tests() body, after
the section's own imports of pn/sb have already happened.

Why C&OHL is not in the representative Dickert
-----------------------------------------------
The representative list picks one Dickert variant that exercises the code
path without covering every line combination. C&OHL appears in the full
ALL_DICKERT_CASES catalogue in catalogues.py and is tested in the all-network
sweep. The representative list uses cable/single/short/good as the most
basic, reproducible Dickert variant.

Tuple schemas
-------------
REPRESENTATIVE_NETWORK_SPECS
    (test_name, sb_code_or_None, loader_tag, label)
    test_name  : key for --only filtering and TestCase naming
    sb_code    : SimBench code string or None
    loader_tag : identifies the pandapower factory (see make_loader())
    label      : human-readable label used in summaries and profile builder

REPRESENTATIVE_NETWORK_PLOTTER_SPECS
    (test_name, sb_code_or_None, loader_tag, net_name, simbench_profile_code)
    net_name           : label passed to build_annual_profiles()
    simbench_profile_code : passed as simbench_code= kwarg, or None
"""

# ---------------------------------------------------------------------------
# Raw spec data — no pandapower/simbench imports here
# ---------------------------------------------------------------------------

# Primary representative list: 9 networks, one per family
# (test_name, sb_code, loader_tag, label)
# NOTE: Dickert label uses the full variant name for summary clarity.
REPRESENTATIVE_NETWORK_SPECS = [
    ("sb_mv_rural",    "1-MV-rural--2-sw",      "simbench",
     "1-MV-rural--2-sw"),
    ("sb_lv_rural",    "1-LV-rural1--0-sw",     "simbench",
     "1-LV-rural1--0-sw"),
    ("sb_mvlv_rural",  "1-MVLV-rural-all-0-sw", "simbench",
     "1-MVLV-rural-all-0-sw"),
    ("cigre_mv",       None, "cigre_mv",
     "cigre_mv"),
    ("cigre_lv",       None, "cigre_lv",
     "cigre_lv"),
    ("kerber_std",     None, "create_kerber_landnetz_kabel_1",
     "kerber_landnetz_kabel_1"),
    ("kerber_extreme", None, "kb_extrem_landnetz_kabel",
     "kb_extrem_landnetz_kabel"),
    ("synthetic_lv",   None, "synthetic_lv_rural_1",
     "synthetic_lv_rural_1"),
    ("dickert",        None, "dickert_short_cable_single_good",
     "dickert_short_cable_single_good"),  # full name for summary clarity
]

# Plotter representative list — 5-tuple with extra simbench_profile_code
# (test_name, sb_code, loader_tag, net_name, simbench_profile_code)
REPRESENTATIVE_NETWORK_PLOTTER_SPECS = [
    ("sb_mv_rural",     "1-MV-rural--2-sw",      "simbench",
     "1-MV-rural--2-sw",        "1-MV-rural--2-sw"),
    ("sb_lv_rural",     "1-LV-rural1--0-sw",     "simbench",
     "1-LV-rural1--0-sw",       "1-LV-rural1--0-sw"),
    ("sb_mvlv_rural",   "1-MVLV-rural-all-0-sw", "simbench",
     "1-MVLV-rural-all-0-sw",   "1-MVLV-rural-all-0-sw"),
    ("cigre_mv",        None, "cigre_mv",
     "cigre_mv_with_der",        None),
    ("cigre_lv",        None, "cigre_lv",
     "cigre_lv",                 None),
    ("kerber_standard", None, "create_kerber_landnetz_kabel_1",
     "kerber_landnetz_kabel_1",  None),
    ("kerber_extreme",  None, "kb_extrem_landnetz_kabel",
     "kb_extrem_landnetz_kabel", None),
    ("synthetic_lv",    None, "synthetic_lv_rural_1",
     "synthetic_lv_rural_1",     None),
    ("dickert",         None, "dickert_short_cable_single_good",
     "dickert_short_cable_single_good", None),
]


# ---------------------------------------------------------------------------
# Deferred loader factory
# ---------------------------------------------------------------------------

def make_loader(loader_tag: str, sb_code: str | None):
    """
    Return a zero-argument callable that loads the named network.

    pandapower and simbench are imported inside this function so the caller
    controls when the import cost is paid.

    loader_tag conventions
    ----------------------
    "simbench"               → sb.get_simbench_net(sb_code)
    "cigre_mv"               → pn.create_cigre_network_mv(with_der="pv_wind")
    "cigre_lv"               → pn.create_cigre_network_lv()
    "kerber_*"/"kb_extrem_*" → getattr(pn, loader_tag)()
    "synthetic_lv_<class>"   → pn.create_synthetic_voltage_control_lv_network(<class>)
    "dickert_<fr>_<lt>_<cu>_<ca>"
                             → pn.create_dickert_lv_network(fr, lt, cu, ca)
                               Only underscore-safe combinations are present
                               in the representative list. C&OHL variants are
                               in the full sweep and handled by explicit tuples
                               in catalogues.py, not by this tag parser.
    """
    import pandapower.networks as pn

    if loader_tag == "simbench":
        import simbench as sb
        code = sb_code
        return lambda: sb.get_simbench_net(code)

    if loader_tag == "cigre_mv":
        return lambda: pn.create_cigre_network_mv(with_der="pv_wind")

    if loader_tag == "cigre_lv":
        return lambda: pn.create_cigre_network_lv()

    if loader_tag.startswith("synthetic_lv_"):
        nc = loader_tag[len("synthetic_lv_"):]
        return lambda c=nc: pn.create_synthetic_voltage_control_lv_network(c)

    if loader_tag.startswith("dickert_"):
        # tag format: dickert_<feeders>_<linetype>_<customer>_<case>
        # Only underscore-safe combinations appear here (no C&OHL).
        parts = loader_tag.split("_")
        fr, lt, cu, ca = parts[1], parts[2], parts[3], parts[4]
        return lambda f=fr, l=lt, c=cu, s=ca: \
            pn.create_dickert_lv_network(f, l, c, s)

    # Kerber standard and extreme: pandapower function name IS the tag
    fn = getattr(pn, loader_tag, None)
    if fn is not None:
        return fn
    
    fn = getattr(pn, f"create_{loader_tag}", None)
    if fn is not None:
        return fn

    raise ValueError(f"make_loader: unknown loader_tag {loader_tag!r}")


# ---------------------------------------------------------------------------
# Deferred list builders — called inside section run functions, never at
# module import time. Return the canonical (test_name, loader_fn, label)
# tuples that section loops expect.
# ---------------------------------------------------------------------------

def get_representative_networks() -> list[tuple]:
    """
    Build and return the primary representative network list.
    Returns: list of (test_name, loader_fn, label)

    Import cost: pandapower.networks + simbench on first call.
    Call this inside run_*_tests() bodies, not at module level.
    """
    return [
        (test_name, make_loader(tag, sb_code), label)
        for test_name, sb_code, tag, label in REPRESENTATIVE_NETWORK_SPECS
    ]


def get_representative_networks_plotter() -> list[tuple]:
    """
    Build and return the plotter representative network list.
    Returns: list of (test_name, loader_fn, net_name, simbench_profile_code)

    Import cost: pandapower.networks + simbench on first call.
    Call this inside run_network_plotter_tests(), not at module level.
    """
    return [
        (test_name, make_loader(tag, sb_code), net_name, profile_code)
        for test_name, sb_code, tag, net_name, profile_code
        in REPRESENTATIVE_NETWORK_PLOTTER_SPECS
    ]
