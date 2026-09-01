import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
from scipy.io import savemat
from profile_builder import build_annual_profiles, load_dwd_wind

# 1. Load network
net = pn.create_cigre_network_mv(with_der="pv_wind")
pp.runpp(net, voltage_depend_loads=False)

print(net.sgen[["bus", "name", "p_mw", "sn_mva", "type"]])
'''net.sgen.loc[net.sgen["type"] == "PV", "p_mw"]    = 1.0
net.sgen.loc[net.sgen["type"] == "PV", "sn_mva"]  = 1.0

# 2. Build annual profiles — needs DWD files in data/dwd
profiles = build_annual_profiles(
    net=net,
    net_name="cigre_mv",
    data_dir="data/dwd"
)

# 3. Identify peak generation day
extreme_day = profiles["extreme_days"]["max_der"]
print(f"Peak generation day: {extreme_day}")

# 4. Extract that day's slice
times = profiles["times"]
target_date = pd.Timestamp(extreme_day).date()
day_mask = np.array([t.date() == target_date for t in times])

load_day = profiles["load"].loc[times[day_mask]]  # (144, 18) [MW]
pv_day   = profiles["pv"].loc[times[day_mask]]    # (144, 8)  [MW]
wind_day = profiles["wind"].loc[times[day_mask]]  # (144, 1)  [MW]

# Wind speed signal for WindGFLMpp.slx (raw m/s, not converted to power)
wind_raw       = load_dwd_wind("data/dwd")
wind_raw       = wind_raw[~wind_raw.index.duplicated(keep="first")]
target_year    = pd.Timestamp(extreme_day).year
wind_raw       = wind_raw[wind_raw.index.year == target_year]
wind_speed_day = wind_raw.loc[wind_raw.index[day_mask]].values   # (144, 1)

# 5. Time vector in seconds from start of day
n_steps = len(load_day)
t_s = np.arange(n_steps) * 600.0  # 0, 600, ..., 85800

# 6. Find peak generation timestep → fault trigger time
total_der = pv_day.sum(axis=1) + wind_day.sum(axis=1)
peak_idx  = int(total_der.values.argmax())
peak_t_s  = t_s[peak_idx]
print(f"Peak at step {peak_idx}, t = {peak_t_s:.0f} s  →  fault trigger time")

# 7. Q profiles for loads — constant Q/P ratio from pandapower
qp_ratio = (net.load["q_mvar"] / net.load["p_mw"]).fillna(0.0)
q_load_day = pd.DataFrame(index=load_day.index, columns=load_day.columns, dtype=float)
for col in load_day.columns:
    q_load_day[col] = load_day[col] * qp_ratio.loc[col]

# 8. Bus index mapping arrays
load_bus_arr  = np.array([net.load.at[c, "bus"] for c in load_day.columns])
pv_bus_arr    = np.array([net.sgen.at[c, "bus"] for c in pv_day.columns])
wind_bus_arr  = np.array([net.sgen.at[c, "bus"] for c in wind_day.columns])

# 9. Export
savemat("cigre_mv_day_profiles.mat", {
    "t_s":          t_s,
    "P_load_W":     load_day.values   * 1e6,   # (144, 18) W
    "Q_load_VAR":   q_load_day.values * 1e6,   # (144, 18) VAR
    "load_bus":     load_bus_arr,               # (18,)
    "P_pv_W":       pv_day.values     * 1e6,   # (144, 8)  W
    "P_wind_W":     wind_day.values   * 1e6,   # (144, 1)  W
    "der_bus_pv":   pv_bus_arr,                 # (8,)
    "der_bus_wind": wind_bus_arr,               # (1,)
    "peak_t_s":     np.array([peak_t_s]),       # scalar — use as fault trigger in Simulink
    "peak_step":    np.array([peak_idx]),
    "vm_pu_peak":     net.res_bus["vm_pu"].values,
    "va_deg_peak":    net.res_bus["va_degree"].values,
    "wind_speed_ms": wind_speed_day,   # (144, 1) m/s — for WindGFLMpp.slx
})

print(f"Saved: cigre_mv_day_profiles.mat")
print(f"Set fault trigger in Simulink to t = {peak_t_s:.0f} s")


import scipy.io as sio

# Load the .mat to get the peak step index
mat = sio.loadmat('cigre_mv_day_profiles.mat')
peak_step = int(mat['peak_step'][0][0])  # step 76

# Set network to peak generation operating point
load_day  = profiles["load"]
pv_day    = profiles["pv"]
wind_day  = profiles["wind"]

# Apply load values at peak step
for i, idx in enumerate(load_day.columns):
    net.load.at[idx, "p_mw"] = load_day.iloc[peak_step, i]
    net.load.at[idx, "q_mvar"] = net.load.at[idx, "p_mw"] * (net.load.at[idx, "q_mvar"] / net.load.at[idx, "p_mw"]) if net.load.at[idx, "p_mw"] != 0 else 0

# Apply DER values at peak step
for i, idx in enumerate(pv_day.columns):
    net.sgen.at[idx, "p_mw"] = pv_day.iloc[peak_step, i]

for i, idx in enumerate(wind_day.columns):
    net.sgen.at[idx, "p_mw"] = wind_day.iloc[peak_step, i]

# Run power flow at peak operating point
pp.runpp(net, voltage_depend_loads=False)

# Extract bus voltages
print("\n=== Bus voltages at peak generation (t=45600s) ===")
print(net.res_bus[["vm_pu", "va_degree"]])


# Set network to midnight operating point (step 0)
for i, idx in enumerate(load_day.columns):
    net.load.at[idx, "p_mw"] = load_day.iloc[0, i]
for i, idx in enumerate(pv_day.columns):
    net.sgen.at[idx, "p_mw"] = 0.0   # no PV at midnight
for i, idx in enumerate(wind_day.columns):
    net.sgen.at[idx, "p_mw"] = wind_day.iloc[0, i]

pp.runpp(net, voltage_depend_loads=False)
print("\n=== Bus voltages at midnight (t=0) ===")
print(net.res_bus[["vm_pu", "va_degree"]])


'''