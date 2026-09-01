'''
Preset network catalogue for the HIL CLI wizard

This module stores menu data only. It does not load pandapower networks. 
The executor is responsible for turning preset_name into an actual network.
'''

from __future__ import annotations

PresetEntry = dict[str,str]
PresetCatalogue = dict[str, list[PresetEntry]]

_PRESET_CATALOGUE: PresetCatalogue = {
    "Simbench": [
        {
            "label": "Simbench 1-MV-rural--2-sw, primary MV HIL network, 99 buses, 102 DERs",
            "preset_name": "1-MV-rural--2-sw",
            "preset_family": "SimBench",
        },
    ],

    "CIGRE": [
        {
            "label": "CIGRE MV without DER, 15-bus medium-voltage benchmark",
            "preset_name": "cigre_mv_no_der",
            "preset_family": "CIGRE",
        },
        {
            "label": "CIGRE MV with PV and wind DER, 15-bus MV benchmark with 9 DERs",
            "preset_name": "cigre_mv_pv_wind",
            "preset_family": "CIGRE",
        },
        {
            "label": "CIGRE LV, low-voltage distribution benchmark",
            "preset_name": "cigre_lv",
            "preset_family": "CIGRE",
        },
    ],

    "Kerber": [
        {
            "label": "Kerber Landnetz Kabel 1, standard rural LV cable network",
            "preset_name": "kerber_landnetz_kabel_1",
            "preset_family": "Kerber",
        },
        {
            "label": "Kerber Landnetz Kabel 2, standard rural LV cable network",
            "preset_name": "kerber_landnetz_kabel_2",
            "preset_family": "Kerber",
        },
        {
            "label": "Kerber Landnetz Freileitung 1, standard rural LV overhead-line network",
            "preset_name": "kerber_landnetz_freileitung_1",
            "preset_family": "Kerber",
        },
        {
            "label": "Kerber Landnetz Freileitung 2, standard rural LV overhead-line network",
            "preset_name": "kerber_landnetz_freileitung_2",
            "preset_family": "Kerber",
        },
        {
            "label": "Kerber Vorstadtnetz Kabel 1, standard suburban LV cable network",
            "preset_name": "kerber_vorstadtnetz_kabel_1",
            "preset_family": "Kerber",
        },
        {
            "label": "Kerber Vorstadtnetz Kabel 2, standard suburban LV cable network",
            "preset_name": "kerber_vorstadtnetz_kabel_2",
            "preset_family": "Kerber",
        },
        {
            "label": "Kerber Dorfnetz, standard village LV network",
            "preset_name": "kerber_dorfnetz",
            "preset_family": "Kerber",
        },
        {
            "label": "Kerber extreme Landnetz Kabel",
            "preset_name": "kb_extrem_landnetz_kabel",
            "preset_family": "Kerber",
        },
        {
            "label": "Kerber extreme Landnetz Freileitung",
            "preset_name": "kb_extrem_landnetz_freileitung",
            "preset_family": "Kerber",
        },
        {
            "label": "Kerber extreme Landnetz Kabel with transformer stress",
            "preset_name": "kb_extrem_landnetz_kabel_trafo",
            "preset_family": "Kerber",
        },
        {
            "label": "Kerber extreme Landnetz Freileitung with transformer stress",
            "preset_name": "kb_extrem_landnetz_frltg_trafo",
            "preset_family": "Kerber",
        },
        {
            "label": "Kerber extreme Dorfnetz",
            "preset_name": "kb_extrem_dorfnetz",
            "preset_family": "Kerber",
        },
        {
            "label": "Kerber extreme Dorfnetz with transformer stress",
            "preset_name": "kb_extrem_dorfnetz_trafo",
            "preset_family": "Kerber",
        },
        {
            "label": "Kerber extreme Vorstadtnetz 1",
            "preset_name": "kb_extrem_vorstadtnetz_1",
            "preset_family": "Kerber",
        },
        {
            "label": "Kerber extreme Vorstadtnetz 2",
            "preset_name": "kb_extrem_vorstadtnetz_2",
            "preset_family": "Kerber",
        },
        {
            "label": "Kerber extreme Vorstadtnetz transformer case 1",
            "preset_name": "kb_extrem_vorstadtnetz_trafo_1",
            "preset_family": "Kerber",
        },
        {
            "label": "Kerber extreme Vorstadtnetz transformer case 2",
            "preset_name": "kb_extrem_vorstadtnetz_trafo_2",
            "preset_family": "Kerber",
        },
    ],
    
    "Dickert": [
        {
            "label": "Dickert short cable, single feeder, good case",
            "preset_name": "dickert_short_cable_single_good",
            "preset_family": "Dickert",
        },
        {
            "label": "Dickert short cable, single feeder, average case",
            "preset_name": "dickert_short_cable_single_average",
            "preset_family": "Dickert",
        },
        {
            "label": "Dickert short cable, single feeder, bad case",
            "preset_name": "dickert_short_cable_single_bad",
            "preset_family": "Dickert",
        },
        {
            "label": "Dickert short cable, multiple feeders, good case",
            "preset_name": "dickert_short_cable_multiple_good",
            "preset_family": "Dickert",
        },
        {
            "label": "Dickert short cable, multiple feeders, average case",
            "preset_name": "dickert_short_cable_multiple_average",
            "preset_family": "Dickert",
        },
        {
            "label": "Dickert short cable, multiple feeders, bad case",
            "preset_name": "dickert_short_cable_multiple_bad",
            "preset_family": "Dickert",
        },
        {
            "label": "Dickert middle cable, multiple feeders, good case",
            "preset_name": "dickert_middle_cable_multiple_good",
            "preset_family": "Dickert",
        },
        {
            "label": "Dickert middle cable, multiple feeders, average case",
            "preset_name": "dickert_middle_cable_multiple_average",
            "preset_family": "Dickert",
        },
        {
            "label": "Dickert middle cable, multiple feeders, bad case",
            "preset_name": "dickert_middle_cable_multiple_bad",
            "preset_family": "Dickert",
        },
        {
            "label": "Dickert middle cable and overhead-line mix, multiple feeders, good case",
            "preset_name": "dickert_middle_cohl_multiple_good",
            "preset_family": "Dickert",
        },
        {
            "label": "Dickert middle cable and overhead-line mix, multiple feeders, average case",
            "preset_name": "dickert_middle_cohl_multiple_average",
            "preset_family": "Dickert",
        },
        {
            "label": "Dickert middle cable and overhead-line mix, multiple feeders, bad case",
            "preset_name": "dickert_middle_cohl_multiple_bad",
            "preset_family": "Dickert",
        },
        {
            "label": "Dickert long cable, multiple feeders, good case",
            "preset_name": "dickert_long_cable_multiple_good",
            "preset_family": "Dickert",
        },
        {
            "label": "Dickert long cable, multiple feeders, average case",
            "preset_name": "dickert_long_cable_multiple_average",
            "preset_family": "Dickert",
        },
        {
            "label": "Dickert long cable, multiple feeders, bad case",
            "preset_name": "dickert_long_cable_multiple_bad",
            "preset_family": "Dickert",
        },
        {
            "label": "Dickert long cable and overhead-line mix, multiple feeders, good case",
            "preset_name": "dickert_long_cohl_multiple_good",
            "preset_family": "Dickert",
        },
        {
            "label": "Dickert long cable and overhead-line mix, multiple feeders, average case",
            "preset_name": "dickert_long_cohl_multiple_average",
            "preset_family": "Dickert",
        },
        {
            "label": "Dickert long cable and overhead-line mix, multiple feeders, bad case",
            "preset_name": "dickert_long_cohl_multiple_bad",
            "preset_family": "Dickert",
        },
    ],

    "Synthetic LV": [
        {
            "label": "Synthetic voltage-control LV rural 1",
            "preset_name": "synthetic_lv_rural_1",
            "preset_family": "Synthetic LV",
        },
        {
            "label": "Synthetic voltage-control LV rural 2",
            "preset_name": "synthetic_lv_rural_2",
            "preset_family": "Synthetic LV",
        },
        {
            "label": "Synthetic voltage-control LV village 1",
            "preset_name": "synthetic_lv_village_1",
            "preset_family": "Synthetic LV",
        },
        {
            "label": "Synthetic voltage-control LV village 2",
            "preset_name": "synthetic_lv_village_2",
            "preset_family": "Synthetic LV",
        },
        {
            "label": "Synthetic voltage-control LV suburb 1",
            "preset_name": "synthetic_lv_suburb_1",
            "preset_family": "Synthetic LV",
        },
    ],
}

def get_preset_families() -> list[str]:
    '''Return preset family names in menu order'''
    return list(_PRESET_CATALOGUE.keys())

def get_presets_for_family(family:str) -> list[PresetEntry]:
    '''Return all preset entries for one preset family'''
    return _PRESET_CATALOGUE[family]


