#!/usr/bin/env python3
"""
flow_profile.py — profile_builder.build_annual_profiles. Same template.

  1. flow_profile_simbench  — SimBench native profile path
  2. flow_profile_fallback  — CIGRE/fallback DWD weather-to-power pipeline

Verified against profile_builder.py. Facts: one entry, two paths chosen by
detect_network_type(net_name). SimBench: sb.get_absolute_values ->
35136-step (2016 leap year) 15-min reconstruction. Fallback: DWD station 691
Bremen CSVs -> unit conversion (RAD-G J/cm^2 -> W/m^2 x10000/600) -> single
complete-year trim -> BDEW-2025 SLP loads (oemof.demand) + pvlib PV
(Erbs -> POA 30 deg S -> NOCT -> AC clip) + piecewise-cubic wind.
Output schema identical on both paths.
"""
from graphviz import Digraph

F = {"m": "#E7F0FA", "s": "#EDF5E4", "sb": "#F1EFE9",
     "lib": "#EAE7F6", "out": "#F3F8EC", "gate": "#FCF5D6",
     "pv": "#FBEFF4", "wd": "#E6F5EF", "ld": "#EEEDFE"}
INK = "#1B1B1A"


def E(g, a, b, t="", **k):
    g.edge(a, b, xlabel=t, color=INK, penwidth="2.0", arrowsize="0.75", **k)

def D(g, a, b, t="", **k):
    g.edge(a, b, xlabel=t, color="#1F6FB2", style="dashed", fontcolor="#1F6FB2",
           penwidth="1.25", arrowsize="0.65", **k)


def base(title, ranksep, nodesep="0.24"):
    g = Digraph(format="svg")
    g.attr(rankdir="TB", splines="ortho", newrank="true", compound="true",
           forcelabels="true", pad="0.25", nodesep=nodesep, ranksep=ranksep,
           fontname="Helvetica-bold", labelloc="t", fontsize="16", label=title)
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica",
           fontsize="10", margin="0.13,0.075", color="#8A887F", height="0.3")
    g.attr("edge", fontname="Helvetica", fontsize="8.5")
    return g


def lane(g, cid, label, fill, nodes):
    with g.subgraph(name=f"cluster_{cid}") as c:
        c.attr(label=label, style="rounded", bgcolor=fill + "55", color="#8A887F",
               fontname="Helvetica-bold", fontsize="11", margin="8")
        for nid, lbl, nf in nodes:
            c.node(nid, lbl, fillcolor=nf)


def legend(g, anchor):
    with g.subgraph(name="cluster_legend") as c:
        c.attr(label="Legend", style="rounded", color="#8A887F", labeljust="l",
               fontname="Helvetica-bold", fontsize="10", bgcolor="#FFFFFF", margin="6")
        for n in ("le1", "le2", "ld1", "ld2"):
            c.node(n, "", shape="point", width="0.02", color="#FFFFFF")
        E(c, "le1", "le2", "execution")
        D(c, "ld1", "ld2", "data (array/df)")
        c.edge("le2", "ld1", style="invis")
    g.edge(anchor, "le1", style="invis")


def build_simbench():
    g = base("profile_builder : SimBench native path", ranksep="0.34")
    lane(g, "m", "build_annual_profiles  ·  profile_builder.py", F["m"], [
        ("b1", "build_annual_profiles(net, net_name, simbench_code)", F["m"]),
        ("b2", "detect_network_type(net_name) -> 'simbench'", F["gate"]),
        ("b3", "require simbench_code (else ValueError)", F["m"]),
        ("p1", "reconstruct DatetimeIndex\\lstep 0..35135 -> 2016-01-01, 15-min,\\lEurope/Berlin (366 x 96 = leap year)", F["m"]),
        ("p2", "load_df = profiles[(load, p_mw)]\\lindex=times ; clip >= 0", F["m"]),
        ("p3", "PV / wind masks on net.sgen.type\\l(pv|solar|lv_res  /  wind|wp)", F["m"]),
        ("p4", "pv_df night-zeroing (h>=22 or <=4);\\lclip pv/wind >= 0", F["m"]),
        ("o1", "assemble result {load, pv, wind, times, net_type}", F["out"]),
        ("o2", "extreme_days = find_extreme_days(result)", F["out"]),
        ("o3", "return profiles dict", F["m"]),
    ])
    lane(g, "sb", "SimBench library", F["sb"], [
        ("s1", "sb.get_simbench_net(code) -> raw_net", F["sb"]),
        ("s2", "sb.get_absolute_values\\lprofiles_instead_of_study_cases=True", F["sb"]),
    ])
    E(g, "b1", "b2"); E(g, "b2", "b3"); E(g, "b3", "p1"); E(g, "p1", "p2")
    E(g, "p2", "p3"); E(g, "p3", "p4"); E(g, "p4", "o1"); E(g, "o1", "o2"); E(g, "o2", "o3")
    E(g, "b3", "s1", constraint="false")
    E(g, "s1", "s2", constraint="false")
    D(g, "s2", "p1", "(load/sgen, p_mw)\\lstep-indexed frames", constraint="false")
    legend(g, "o3")
    return g


def build_fallback():
    g = base("profile_builder : CIGRE / fallback DWD weather-to-power pipeline",
             ranksep="0.32")
    lane(g, "m", "build_annual_profiles  ·  fallback path", F["m"], [
        ("b1", "detect_network_type -> 'cigre' / 'fallback'", F["gate"]),
        ("d1", "load solar/wind/temp CSVs\\lstation 691 default  OR  custom dir + file_map/col_map", F["m"]),
        ("d2", "solar unit handling (conditional)\\lfile_map names RAD-G -> custom, already W/m^2 (skip)\\lelse DWD J/cm^2 -> W/m^2 = value x 10000 / interval_seconds\\l(interval = median timestamp gap; 600 s fallback)\\lwind m/s, temp degC assumed for all sources", F["m"]),
        ("d3", "align times = solar & wind & temp\\l(intersection, dedup)", F["m"]),
        ("d4", "trim to one complete calendar year\\l(most recent with all 12 months)", F["m"]),
        ("disp", "build per-element profiles", F["gate"]),
        ("o1", "assemble result {load, pv, wind, times,\\lnet_type, extreme_days}", F["out"]),
        ("o2", "return profiles dict", F["m"]),
    ])
    lane(g, "conv", "Weather-to-power converters", F["s"], [
        ("ld", "compute_load_profiles_bdew\\lBDEW 2025 SLP H25/G25/L25 (oemof.demand);\\lper-load type; mixed 82/15/3 (seeded)", F["ld"]),
        ("pv", "compute_pv_profile (per PV sgen)\\lErbs decomp GHI->DHI/DNI -> POA (30 deg S)\\l-> NOCT cell temp -> eta -> DC -> AC clip", F["pv"]),
        ("wd", "compute_wind_profile\\lpiecewise cubic power curve\\l(cut-in 3, rated 12, cut-out 25 m/s), Cp", F["wd"]),
    ])
    lane(g, "io", "DWD loaders, libraries & custom-source hooks", F["lib"], [
        ("l1", "load_dwd_solar / _wind / _temperature", F["lib"]),
        ("l2", "pvlib (Erbs, transposition, temperature)", F["lib"]),
        ("l3", "oemof.demand (BDEW 2025) ; workalendar", F["lib"]),
        ("l4", "custom already-W/m^2 source, 3 routes:\\lscript file_map/col_map ; CLI wizard 'custom' dataset ;\\lnetwork YAML profiles.data_dir/file_map/col_map", "#F3F8EC"),
    ])
    E(g, "b1", "d1"); E(g, "d1", "d2"); E(g, "d2", "d3"); E(g, "d3", "d4")
    E(g, "d4", "disp")
    E(g, "disp", "ld", "loads"); E(g, "disp", "pv", "PV"); E(g, "disp", "wd", "wind")
    E(g, "ld", "o1"); E(g, "pv", "o1"); E(g, "wd", "o1"); E(g, "o1", "o2")
    E(g, "d1", "l1", constraint="false")
    E(g, "pv", "l2", constraint="false")
    E(g, "ld", "l3", constraint="false")
    D(g, "l1", "d1", "raw series", constraint="false")
    D(g, "d4", "ld", "times", constraint="false")
    D(g, "d4", "pv", "solar+temp", constraint="false")
    D(g, "d4", "wd", "wind speed", constraint="false")
    legend(g, "o2")
    return g


for name, gb in (("flow_profile_simbench", build_simbench),
                 ("flow_profile_fallback", build_fallback)):
    g = gb()
    g.format = "svg"; g.render(f"/home/claude/{name}", cleanup=True)
    g.format = "pdf"; g.render(f"/home/claude/{name}", cleanup=True)
    print("wrote", name)
