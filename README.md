# HIL Testbed — Distribution Grid Voltage Control

Hardware-in-the-Loop (HIL) testbed for distribution grid voltage control algorithms, developed as part of the Master of Sustainable Renewable Energy Systems programme at Universität Oldenburg.

The platform uses **pandapower** as the simulation engine with **SimBench** and **CIGRE** network models, running on Raspberry Pi 5 + Arduino Uno R3 hardware. It demonstrates and benchmarks five voltage control scenarios against each other, with hosting capacity analysis to quantify the benefit of active voltage regulation.

---

## Project Structure

```
HIL-Project/
│
├── data/
│   ├── dwd/                        # DWD CDC 10-min observation files (station 691, Bremen)
│   │   ├── data_OBS_DEU_PT10M_RAD-G_691.csv   # Global horizontal irradiance [J/cm²]
│   │   ├── data_OBS_DEU_PT10M_F_691.csv        # Wind speed at 10m [m/s]
│   │   └── data_OBS_DEU_PT10M_T2M.csv          # Air temperature at 2m [°C]
│   └── era5/                       # ERA5 reanalysis data (alternative weather source)
│       ├── era5_solar.csv          # GHI [W/m²]
│       ├── era5_wind.csv           # Wind speed [m/s]
│       └── era5_temp.csv           # Air temperature [°C]
│
├── profile_builder.py              # Annual time-series profile generator
├── era5_to_csv.py                  # Splits the single ERA5 CSV download into three separate files
├── test_suite.py                   # Master test framework (199 test cases)
└── README.md
```

---

## Modules

### `profile_builder.py`
Builds annual 10-minute time-series profiles (load, PV, wind) for any pandapower network.

**SimBench networks** — uses `sb.get_absolute_values()` for native 15-minute annual profiles (35,136 timesteps, 2016 reference year).

**CIGRE / Kerber / Dickert / Synthetic LV / fallback networks** — builds profiles from local weather files:
- **PV**: pvlib Erbs decomposition (GHI → DNI + DHI) + plane-of-array irradiance (30° tilt, south-facing) + NOCT cell temperature correction
- **Wind**: piecewise cubic power curve (cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s)
- **Load**: BDEW 2025 updated SLPs (H25/G25/L25) via `oemof.demand ≥ 0.2.2`, scaled to `net.load.p_mw`. Mixed-use statistical assignment (H25 82% / G25 15% / L25 3%) applied when no load type metadata is present.

**Supported weather sources**: DWD CDC (default), ERA5 reanalysis, or any custom CSV via `file_map` / `col_map` parameters.

```python
from profile_builder import build_annual_profiles
import simbench as sb

net = sb.get_simbench_net("1-MV-rural--2-sw")
profiles = build_annual_profiles(
    net,
    net_name="1-MV-rural--2-sw",
    simbench_code="1-MV-rural--2-sw"
)
# profiles["load"]  → pd.DataFrame  [MW], cols = load indices
# profiles["pv"]    → pd.DataFrame  [MW], cols = sgen indices
# profiles["wind"]  → pd.DataFrame  [MW], cols = sgen indices
# profiles["times"] → pd.DatetimeIndex
# profiles["extreme_days"] → dict (max_der, min_der, max_load, min_load)
```

### `era5_to_csv.py`
Splits the single ERA5 CSV download into three separate files (solar, wind, temperature) with correct column names and unit conversions expected by profile_builder.py

Unit conversions applied:
- `ssrd` (J/m²/hr accumulated) → W/m² via division by 3600
- `u10` + `v10` wind components → resultant speed [m/s]
- `t2m` (K) → °C

### `test_suite.py`
Extensible stress test framework covering all in-scope networks. Structured for adding new modules as they are implemented.

**Current coverage — 199 test cases:**

| Group | Count | Notes |
|---|---|---|
| SimBench (MV+LV coupled, MV single, LV single) | 156 | All in-scope codes |
| CIGRE MV (with DER) | 1 | `with_der="pv_wind"` |
| CIGRE LV | 1 | Load-only |
| Kerber (7 standard + 10 extreme) | 17 | Load-only, DER added at hosting capacity stage |
| Synthetic Voltage Control LV | 5 | All classes |
| Dickert LV | 18 | All valid feeder/customer/linetype combinations |
| ERA5 datasource (CIGRE MV) | 1 | Custom col_map + file_map |

**Out of scope** (EHV/HV/HVMV networks — transmission level, >9 GB RAM per network):
`complete_data`, `EHVHVMVLV`, `EHVHV`, `HVMV`, `HV single` — 90 codes excluded.

```bash
# Run all 199 cases
python test_suite.py

# Run only specific network groups (substring match on case name)
python test_suite.py --only cigre kerber kb dickert synthetic era5

# Run a single section
python test_suite.py --section profile_builder

# Full tracebacks on failure
python test_suite.py --verbose
```

---

## Network Coverage

| Network | Type | Voltage | DER pre-placed | Primary use |
|---|---|---|---|---|
| SimBench `1-MV-rural--2-sw` | SimBench | MV | ✓ PV + Wind | Primary MV testbed |
| All 156 in-scope SimBench codes | SimBench | MV/LV/coupled | ✓ | Full validation |
| CIGRE MV (`with_der="pv_wind"`) | CIGRE | MV | ✓ 8 PV + 1 Wind | Secondary MV validation |
| CIGRE LV | CIGRE | LV | ✗ | LV baseline |
| Kerber (7 standard + 10 extreme) | Kerber | LV | ✗ | German LV stress testing |
| Synthetic Voltage Control LV | Synthetic | LV | ✗ | LV voltage control studies |
| Dickert LV (18 combinations) | Dickert | LV | ✗ | LV benchmark feeders |

For networks without pre-placed DER, PV is added deterministically at end-of-feeder buses (highest impedance from slack) during hosting capacity analysis.

---

## Five Comparison Scenarios

| # | Scenario | Description |
|---|---|---|
| 1 | Baseline | No voltage regulation, PV scaled ×2 to force violations |
| 2 | OLTC-only | On-load tap changer, no reactive power control |
| 3 | SVC | Static VAr compensator at MV busbar |
| 4 | **Rule-based Volt-VAr HIL** | IEEE 1547-2018 Q(V) curve, Arduino in the loop |
| 5 | OPF benchmark | `runopp()` optimal power flow, theoretical upper bound |

---

## Hosting Capacity Analysis

Runs across scenarios to quantify the benefit of voltage regulation:

- **Baseline HC**: increment PV scaling factor until first planning constraint violation (V > 1.05 pu or line loading > 100%). No control active.
- **HC with Volt-VAr**: same incremental sweep with Tier 1 algorithm active. Quantifies hosting capacity gain from reactive power control.

---

## Hardware

| Component | Specification |
|---|---|
| Simulation node | Raspberry Pi 5 8GB |
| Controller node | Arduino Uno R3 (dept-provided, USB) |
| Storage | Samsung EVO Plus 32GB microSDXC |

HIL communication uses TCP over USB serial. The Arduino receives bus voltage magnitudes from the Pi, computes Q setpoints via the VDE-AR-N 4110 piecewise linear Q(V) curve, and returns them to the Pi for the second `runpp()` call within each timestep.

---

## Dependencies

```bash
pip install pandapower simbench pvlib "oemof-demand>=0.2.2" workalendar
```

| Package | Version | Purpose |
|---|---|---|
| pandapower | 3.4.0 | Power flow engine |
| simbench | 1.6.1 | SimBench network loader and profiles |
| pvlib | latest | Solar irradiance and PV modelling |
| oemof-demand | ≥ 0.2.2 | BDEW 2025 SLPs (H25/G25/L25) |
| workalendar | latest | German public holidays for BDEW profiles |

**Critical**: always pass `voltage_depend_loads=False` to `runpp()` with SimBench networks (pandapower ≥ 3.2.0 changed voltage-dependent load implementation, causing singular matrix errors).

---

## Standards

- **VDE-AR-N 4110** — Medium voltage grid connection (Q(V) curve, cos φ(P) curve)
- **VDE-AR-N 4105** — Low voltage grid connection
- **IEEE 1547-2018** — Interconnection and interoperability of DER

---

## Weather Data

DWD observation data is downloaded from the [DWD Climate Data Centre](https://cdc.dwd.de/portal/) (CDC). Default station: **Bremen (ID 691)**, 53.05°N 8.80°E.

ERA5 reanalysis data is available via the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/). Convert downloaded NetCDF files using `era5_to_csv.py` before use.

**DWD file naming convention expected:**
```
data_OBS_DEU_PT10M_RAD-G_691.csv   ← solar
data_OBS_DEU_PT10M_F_691.csv        ← wind
data_OBS_DEU_PT10M_T2M.csv          ← temperature
```

For custom datasources, pass `file_map` and `col_map` to `build_annual_profiles()`.

---

## Acknowledgements

Supervisors: Andreas Günther,  Adrian Jimenez — Universität Oldenburg.

Weather data: DWD Climate Data Centre, ERA5 (Copernicus/ECMWF).

Network models: SimBench (Fraunhofer IEE / TU Dortmund), CIGRE Technical Brochure 575, Kerber (TU Munich), Dickert (TU Dresden), pandapower test networks (University of Kassel).
