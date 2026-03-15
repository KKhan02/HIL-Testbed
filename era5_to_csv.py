"""
ERA5 CSV → profile_builder-compatible CSV converter
====================================================
Converts the ERA5 single-location CSV downloaded from Copernicus CDS
into three clean CSVs that profile_builder.py can read via file_map + col_map.

ERA5 unit conversions applied:
    Wind    : sqrt(u10² + v10²)     [m/s]   — scalar speed from components
    Solar   : ssrd [J/m²] / 3600   [W/m²]  — hourly accumulation → mean power
    Temp    : t2m [K] - 273.15     [°C]    — Kelvin → Celsius

Input file (from Copernicus CDS ERA5 single-levels timeseries):
    Columns: valid_time | u10 | v10 | t2m | ssrd | latitude | longitude

Output files (written to same folder as input):
    era5_wind.csv   — columns: timestamp, WS_ms
    era5_solar.csv  — columns: timestamp, GHI_Wm2
    era5_temp.csv   — columns: timestamp, AT_degC

Usage in profile_builder.py:
    profiles = build_annual_profiles(
        net, net_name,
        data_dir="data/era5",
        file_map={
            "RAD-G": "era5_solar.csv",
            "F":     "era5_wind.csv",
            "T2M":   "era5_temp.csv",
        },
        col_map={
            "timestamp": "timestamp",
            "solar":     "GHI_Wm2",
            "wind":      "WS_ms",
            "temp":      "AT_degC",
            "sep":       ",",
        }
    )
"""

import os
import numpy as np
import pandas as pd

# ===========================================================================
# Configuration — update ERA5_FILE path if needed
# ===========================================================================
ERA5_FILE  = "data/era5/reanalysis-era5-single-levels-timeseries-sfcaulxhb6s.csv"
OUTPUT_DIR = "data/era5"
TIMEZONE   = "Europe/Berlin"


def convert(era5_file: str = ERA5_FILE, output_dir: str = OUTPUT_DIR):
    """
    Reads the ERA5 CSV, applies unit conversions, writes three output CSVs.
    """
    print(f"[era5_to_csv] Reading: {era5_file}")
    df = pd.read_csv(era5_file)

    print(f"[era5_to_csv] Columns found : {list(df.columns)}")
    print(f"[era5_to_csv] Rows          : {len(df)}")
    print(f"[era5_to_csv] Period        : "
          f"{df['valid_time'].iloc[0]} → {df['valid_time'].iloc[-1]}")

    # ------------------------------------------------------------------
    # Timestamp — parse UTC and convert to Europe/Berlin
    # ------------------------------------------------------------------
    ts = pd.to_datetime(df["valid_time"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    ts = ts.dt.tz_convert(TIMEZONE)

    # ------------------------------------------------------------------
    # Wind speed  [m/s]   sqrt(u10² + v10²)
    # ------------------------------------------------------------------
    wind_speed = np.sqrt(df["u10"]**2 + df["v10"]**2)
    print(f"[era5_to_csv] Wind  : {wind_speed.min():.1f} – "
          f"{wind_speed.max():.1f} m/s")

    # ------------------------------------------------------------------
    # Solar GHI  [W/m²]   ssrd [J/m²] / 3600
    # ssrd is energy accumulated over the preceding hour.
    # Dividing by 3600 s gives mean irradiance power for that hour.
    # ------------------------------------------------------------------
    ghi = (df["ssrd"] / 3600.0).clip(lower=0.0)
    print(f"[era5_to_csv] Solar : {ghi.min():.1f} – "
          f"{ghi.max():.1f} W/m²")

    # ------------------------------------------------------------------
    # Temperature  [°C]   t2m [K] - 273.15
    # ------------------------------------------------------------------
    temp_c = df["t2m"] - 273.15
    print(f"[era5_to_csv] Temp  : {temp_c.min():.1f} – "
          f"{temp_c.max():.1f} °C")

    # ------------------------------------------------------------------
    # Write output CSVs
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    def write(series, filename, col_name):
        out = pd.DataFrame({
            "timestamp": ts.dt.strftime("%Y-%m-%dT%H:%M:%S"),
            col_name:    series.values
        })
        fpath = os.path.join(output_dir, filename)
        out.to_csv(fpath, index=False)
        print(f"[era5_to_csv] Written : {fpath}  ({len(out)} rows)")

    write(wind_speed, "era5_wind.csv",  "WS_ms")
    write(ghi,        "era5_solar.csv", "GHI_Wm2")
    write(temp_c,     "era5_temp.csv",  "AT_degC")

    print(f"\n[era5_to_csv] All done. CSVs ready in '{output_dir}'.")


if __name__ == "__main__":
    convert()
