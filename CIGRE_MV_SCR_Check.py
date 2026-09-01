"""
SCR at the CIGRE MV PV buses — IEC 60909 three-phase short-circuit, case="max".

Purpose: get the authoritative S_sc / SCR at each of the 8 PV buses from
pandapower's actual network topology, to confirm (or correct) the hand-reduced
estimate at Bus 11 — Z_th = 4.871 + j8.927 Ohm, SCR ~ 39.3 at S_n=1.0 MVA,
computed by treating the len_3_8 "trunk branch" as a normally-closed parallel
loop with the 3-4-5-6-7-8 chain.

Run this in the environment where pandapower 3.4.0 is already installed
(not available in the sandbox used to derive the hand calc).
"""
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandapower.shortcircuit as sc

net = pn.create_cigre_network_mv(with_der="pv_wind")

# Match the PV rescale already used in Resilient_Ops_project.py, so SCR
# reflects the actual DER rating used in the SP16 model, not pandapower's default.
net.sgen.loc[net.sgen["type"] == "PV", "p_mw"] = 1.0
net.sgen.loc[net.sgen["type"] == "PV", "sn_mva"] = 1.0

# Sanity check — confirm TB575 source strength (5000 MVA, R/X=0.1) is set on
# ext_grid. The built-in network should already carry this; fall back if not.
print("ext_grid short-circuit parameters:")
print(net.ext_grid[["bus", "s_sc_max_mva", "rx_max", "s_sc_min_mva", "rx_min"]])
if net.ext_grid["s_sc_max_mva"].isna().any():
    print("WARNING: s_sc_max_mva not set on ext_grid — applying TB575 value (5000 MVA, R/X=0.1)")
    net.ext_grid["s_sc_max_mva"] = 5000.0
    net.ext_grid["rx_max"] = 0.1

# SCR is defined as the network's own short-circuit strength at the terminal,
# excluding the contribution of the inverter under study itself — otherwise
# we'd need a fault-current ratio (net.sgen.k) per DER, which is an inverter
# protection-setting assumption we don't actually need for this metric, and
# which create_cigre_network_mv() doesn't populate by default (hence the
# ValueError). Taking all sgens out of service isolates pure network strength
# (source + transformers + lines), matching the hand-reduced Thevenin calc.
net.sgen["in_service"] = False

# Three-phase bolted fault, maximum fault level case (c_max), all buses at once
sc.calc_sc(net, case="max", fault="3ph", ip=False)

# If this errors with a KeyError on 'ikss_ka', run:
#   print(net.res_bus_sc.columns)
# and adjust — column names can shift slightly across pandapower versions.
V_ll_kV = net.bus["vn_kv"]
Ikss_kA = net.res_bus_sc["ikss_ka"]
S_sc_mva = np.sqrt(3) * V_ll_kV * Ikss_kA

results = []
for idx, row in net.sgen[net.sgen["type"] == "PV"].iterrows():
    bus = row["bus"]
    bus_name = net.bus.at[bus, "name"]
    s_sc = S_sc_mva.at[bus]
    scr = s_sc / row["sn_mva"]
    results.append((bus_name, bus, row["sn_mva"], s_sc, scr))

results.sort(key=lambda r: r[4])  # ascending SCR = weakest first

print(f"\n{'Bus':<10}{'idx':<5}{'S_n (MVA)':<11}{'S_sc (MVA)':<13}{'SCR':<8}")
for name, idx, sn, s_sc, scr in results:
    print(f"{name:<10}{idx:<5}{sn:<11.2f}{s_sc:<13.2f}{scr:<8.1f}")

print(
    "\nCompare the Bus 11 row above against the hand-reduced estimate: "
    "Z_th=4.871+j8.927 Ohm, S_sc=39.34 MVA, SCR=39.3 (at S_n=1.0 MVA)."
)