"""
hiltest/catalogues.py
======================
Network family catalogues — all pure Python, no pandapower/simbench imports.

Moved here from constants.py (which retains only scalar configuration values).

Used by:
    sections/profile_builder.py
    sections/violation.py
    sections/volt_var.py
    sections/coordinator.py
"""

# ---------------------------------------------------------------------------
# In-scope SimBench codes  (156 total)
# ---------------------------------------------------------------------------
IN_SCOPE_SIMBENCH_CODES = [

    # MV+LV coupled — rural  (24 codes)
    "1-MVLV-rural-all-0-sw",     "1-MVLV-rural-all-0-no_sw",
    "1-MVLV-rural-all-1-sw",     "1-MVLV-rural-all-1-no_sw",
    "1-MVLV-rural-all-2-sw",     "1-MVLV-rural-all-2-no_sw",
    "1-MVLV-rural-1.108-0-sw",   "1-MVLV-rural-1.108-0-no_sw",
    "1-MVLV-rural-1.108-1-sw",   "1-MVLV-rural-1.108-1-no_sw",
    "1-MVLV-rural-1.108-2-sw",   "1-MVLV-rural-1.108-2-no_sw",
    "1-MVLV-rural-2.107-0-sw",   "1-MVLV-rural-2.107-0-no_sw",
    "1-MVLV-rural-2.107-1-sw",   "1-MVLV-rural-2.107-1-no_sw",
    "1-MVLV-rural-2.107-2-sw",   "1-MVLV-rural-2.107-2-no_sw",
    "1-MVLV-rural-4.101-0-sw",   "1-MVLV-rural-4.101-0-no_sw",
    "1-MVLV-rural-4.101-1-sw",   "1-MVLV-rural-4.101-1-no_sw",
    "1-MVLV-rural-4.101-2-sw",   "1-MVLV-rural-4.101-2-no_sw",

    # MV+LV coupled — semiurban  (24 codes)
    "1-MVLV-semiurb-all-0-sw",   "1-MVLV-semiurb-all-0-no_sw",
    "1-MVLV-semiurb-all-1-sw",   "1-MVLV-semiurb-all-1-no_sw",
    "1-MVLV-semiurb-all-2-sw",   "1-MVLV-semiurb-all-2-no_sw",
    "1-MVLV-semiurb-3.202-0-sw", "1-MVLV-semiurb-3.202-0-no_sw",
    "1-MVLV-semiurb-3.202-1-sw", "1-MVLV-semiurb-3.202-1-no_sw",
    "1-MVLV-semiurb-3.202-2-sw", "1-MVLV-semiurb-3.202-2-no_sw",
    "1-MVLV-semiurb-4.201-0-sw", "1-MVLV-semiurb-4.201-0-no_sw",
    "1-MVLV-semiurb-4.201-1-sw", "1-MVLV-semiurb-4.201-1-no_sw",
    "1-MVLV-semiurb-4.201-2-sw", "1-MVLV-semiurb-4.201-2-no_sw",
    "1-MVLV-semiurb-5.220-0-sw", "1-MVLV-semiurb-5.220-0-no_sw",
    "1-MVLV-semiurb-5.220-1-sw", "1-MVLV-semiurb-5.220-1-no_sw",
    "1-MVLV-semiurb-5.220-2-sw", "1-MVLV-semiurb-5.220-2-no_sw",

    # MV+LV coupled — urban  (20 codes)
    "1-MVLV-urban-all-0-sw",     "1-MVLV-urban-all-0-no_sw",
    "1-MVLV-urban-all-1-sw",     "1-MVLV-urban-all-1-no_sw",
    "1-MVLV-urban-all-2-sw",     "1-MVLV-urban-all-2-no_sw",
    "1-MVLV-urban-5.303-0-sw",   "1-MVLV-urban-5.303-0-no_sw",
    "1-MVLV-urban-5.303-1-sw",   "1-MVLV-urban-5.303-1-no_sw",
    "1-MVLV-urban-5.303-2-sw",   "1-MVLV-urban-5.303-2-no_sw",
    "1-MVLV-urban-6.305-0-sw",   "1-MVLV-urban-6.305-0-no_sw",
    "1-MVLV-urban-6.305-1-sw",   "1-MVLV-urban-6.305-1-no_sw",
    "1-MVLV-urban-6.305-2-sw",   "1-MVLV-urban-6.305-2-no_sw",
    "1-MVLV-urban-6.309-0-sw",   "1-MVLV-urban-6.309-0-no_sw",
    "1-MVLV-urban-6.309-1-sw",   "1-MVLV-urban-6.309-1-no_sw",
    "1-MVLV-urban-6.309-2-sw",   "1-MVLV-urban-6.309-2-no_sw",

    # MV+LV coupled — commercial  (24 codes)
    "1-MVLV-comm-all-0-sw",      "1-MVLV-comm-all-0-no_sw",
    "1-MVLV-comm-all-1-sw",      "1-MVLV-comm-all-1-no_sw",
    "1-MVLV-comm-all-2-sw",      "1-MVLV-comm-all-2-no_sw",
    "1-MVLV-comm-3.403-0-sw",    "1-MVLV-comm-3.403-0-no_sw",
    "1-MVLV-comm-3.403-1-sw",    "1-MVLV-comm-3.403-1-no_sw",
    "1-MVLV-comm-3.403-2-sw",    "1-MVLV-comm-3.403-2-no_sw",
    "1-MVLV-comm-4.416-0-sw",    "1-MVLV-comm-4.416-0-no_sw",
    "1-MVLV-comm-4.416-1-sw",    "1-MVLV-comm-4.416-1-no_sw",
    "1-MVLV-comm-4.416-2-sw",    "1-MVLV-comm-4.416-2-no_sw",
    "1-MVLV-comm-5.401-0-sw",    "1-MVLV-comm-5.401-0-no_sw",
    "1-MVLV-comm-5.401-1-sw",    "1-MVLV-comm-5.401-1-no_sw",
    "1-MVLV-comm-5.401-2-sw",    "1-MVLV-comm-5.401-2-no_sw",

    # MV single level  (24 codes)
    "1-MV-rural--0-sw",          "1-MV-rural--0-no_sw",
    "1-MV-rural--1-sw",          "1-MV-rural--1-no_sw",
    "1-MV-rural--2-sw",          "1-MV-rural--2-no_sw",
    "1-MV-semiurb--0-sw",        "1-MV-semiurb--0-no_sw",
    "1-MV-semiurb--1-sw",        "1-MV-semiurb--1-no_sw",
    "1-MV-semiurb--2-sw",        "1-MV-semiurb--2-no_sw",
    "1-MV-urban--0-sw",          "1-MV-urban--0-no_sw",
    "1-MV-urban--1-sw",          "1-MV-urban--1-no_sw",
    "1-MV-urban--2-sw",          "1-MV-urban--2-no_sw",
    "1-MV-comm--0-sw",           "1-MV-comm--0-no_sw",
    "1-MV-comm--1-sw",           "1-MV-comm--1-no_sw",
    "1-MV-comm--2-sw",           "1-MV-comm--2-no_sw",

    # LV single level  (36 codes)
    "1-LV-rural1--0-sw",         "1-LV-rural1--0-no_sw",
    "1-LV-rural1--1-sw",         "1-LV-rural1--1-no_sw",
    "1-LV-rural1--2-sw",         "1-LV-rural1--2-no_sw",
    "1-LV-rural2--0-sw",         "1-LV-rural2--0-no_sw",
    "1-LV-rural2--1-sw",         "1-LV-rural2--1-no_sw",
    "1-LV-rural2--2-sw",         "1-LV-rural2--2-no_sw",
    "1-LV-rural3--0-sw",         "1-LV-rural3--0-no_sw",
    "1-LV-rural3--1-sw",         "1-LV-rural3--1-no_sw",
    "1-LV-rural3--2-sw",         "1-LV-rural3--2-no_sw",
    "1-LV-semiurb4--0-sw",       "1-LV-semiurb4--0-no_sw",
    "1-LV-semiurb4--1-sw",       "1-LV-semiurb4--1-no_sw",
    "1-LV-semiurb4--2-sw",       "1-LV-semiurb4--2-no_sw",
    "1-LV-semiurb5--0-sw",       "1-LV-semiurb5--0-no_sw",
    "1-LV-semiurb5--1-sw",       "1-LV-semiurb5--1-no_sw",
    "1-LV-semiurb5--2-sw",       "1-LV-semiurb5--2-no_sw",
    "1-LV-urban6--0-sw",         "1-LV-urban6--0-no_sw",
    "1-LV-urban6--1-sw",         "1-LV-urban6--1-no_sw",
    "1-LV-urban6--2-sw",         "1-LV-urban6--2-no_sw",
]

# ---------------------------------------------------------------------------
# Dickert LV — 18 combinations
# (name, feeders_range, linetype, customer, case)
# ---------------------------------------------------------------------------
ALL_DICKERT_CASES = [
    ("dickert_short_cable_single_good",       "short", "cable",  "single",   "good"),
    ("dickert_short_cable_single_average",    "short", "cable",  "single",   "average"),
    ("dickert_short_cable_single_bad",        "short", "cable",  "single",   "bad"),
    ("dickert_short_cable_multiple_good",     "short", "cable",  "multiple", "good"),
    ("dickert_short_cable_multiple_average",  "short", "cable",  "multiple", "average"),
    ("dickert_short_cable_multiple_bad",      "short", "cable",  "multiple", "bad"),
    ("dickert_middle_cable_multiple_good",    "middle","cable",  "multiple", "good"),
    ("dickert_middle_cable_multiple_average", "middle","cable",  "multiple", "average"),
    ("dickert_middle_cable_multiple_bad",     "middle","cable",  "multiple", "bad"),
    ("dickert_middle_cohl_multiple_good",     "middle","C&OHL",  "multiple", "good"),
    ("dickert_middle_cohl_multiple_average",  "middle","C&OHL",  "multiple", "average"),
    ("dickert_middle_cohl_multiple_bad",      "middle","C&OHL",  "multiple", "bad"),
    ("dickert_long_cable_multiple_good",      "long",  "cable",  "multiple", "good"),
    ("dickert_long_cable_multiple_average",   "long",  "cable",  "multiple", "average"),
    ("dickert_long_cable_multiple_bad",       "long",  "cable",  "multiple", "bad"),
    ("dickert_long_cohl_multiple_good",       "long",  "C&OHL",  "multiple", "good"),
    ("dickert_long_cohl_multiple_average",    "long",  "C&OHL",  "multiple", "average"),
    ("dickert_long_cohl_multiple_bad",        "long",  "C&OHL",  "multiple", "bad"),
]

# ---------------------------------------------------------------------------
# Synthetic Voltage Control LV — 5 classes
# ---------------------------------------------------------------------------
ALL_SYNTHETIC_LV_CASES = [
    "rural_1", "rural_2", "village_1", "village_2", "suburb_1",
]

# ---------------------------------------------------------------------------
# Kerber — 17 variants  (name, pandapower_function_name_string)
# ---------------------------------------------------------------------------
ALL_KERBER_CASES = [
    ("kerber_landnetz_kabel_1",        "create_kerber_landnetz_kabel_1"),
    ("kerber_landnetz_kabel_2",        "create_kerber_landnetz_kabel_2"),
    ("kerber_landnetz_freileitung_1",  "create_kerber_landnetz_freileitung_1"),
    ("kerber_landnetz_freileitung_2",  "create_kerber_landnetz_freileitung_2"),
    ("kerber_vorstadtnetz_kabel_1",    "create_kerber_vorstadtnetz_kabel_1"),
    ("kerber_vorstadtnetz_kabel_2",    "create_kerber_vorstadtnetz_kabel_2"),
    ("kerber_dorfnetz",                "create_kerber_dorfnetz"),
    ("kb_extrem_landnetz_kabel",       "kb_extrem_landnetz_kabel"),
    ("kb_extrem_landnetz_freileitung", "kb_extrem_landnetz_freileitung"),
    ("kb_extrem_landnetz_kabel_trafo", "kb_extrem_landnetz_kabel_trafo"),
    ("kb_extrem_landnetz_frltg_trafo", "kb_extrem_landnetz_freileitung_trafo"),
    ("kb_extrem_dorfnetz",             "kb_extrem_dorfnetz"),
    ("kb_extrem_dorfnetz_trafo",       "kb_extrem_dorfnetz_trafo"),
    ("kb_extrem_vorstadtnetz_1",       "kb_extrem_vorstadtnetz_1"),
    ("kb_extrem_vorstadtnetz_2",       "kb_extrem_vorstadtnetz_2"),
    ("kb_extrem_vorstadtnetz_trafo_1", "kb_extrem_vorstadtnetz_trafo_1"),
    ("kb_extrem_vorstadtnetz_trafo_2", "kb_extrem_vorstadtnetz_trafo_2"),
]
