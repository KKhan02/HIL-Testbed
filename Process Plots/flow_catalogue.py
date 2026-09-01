#!/usr/bin/env python3
"""
flow_catalogue.py — preset catalogue & construction path. Same template.

  flow_catalogue — network_catalogue (menu data only) -> wizard selection ->
                   executor._preset_loaders (preset_name -> constructor) -> net

Verified from source (function bodies, not the memory summary):
- network_catalogue.py is MENU DATA ONLY: _PRESET_CATALOGUE (families ->
  [{label, preset_name, preset_family}]) + get_preset_families() +
  get_presets_for_family(). It does NOT construct networks.
- The 44 preset BUILDERS live in executor._preset_loaders(): preset_name ->
  pandapower.networks / simbench constructor callable.
- 44 = 1 SimBench + 3 CIGRE + 17 Kerber (7 std + 10 kb_extrem_) + 18 Dickert
  + 5 Synthetic LV. CIGRE MV uses with_der='pv_wind'; Dickert encodes
  (feeders_range, linetype, customer, case) with 'average' and 'C&OHL'.
"""
from graphviz import Digraph

F = {"cat": "#F1EFE9", "wiz": "#E7F0FA", "exe": "#EDE7F6",
     "bld": "#EDF5E4", "out": "#F3F8EC", "note": "#FCEBEA"}
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
        D(c, "ld1", "ld2", "data (menu / config)")
        c.edge("le2", "ld1", style="invis")
    g.edge(anchor, "le1", style="invis")


def build():
    g = base("Preset catalogue & construction : network_catalogue -> wizard -> "
             "executor._preset_loaders", ranksep="0.34")
    lane(g, "cat", "network_catalogue.py  (MENU DATA ONLY - no construction)", F["cat"], [
        ("pc", "_PRESET_CATALOGUE : family -> [ {label, preset_name,\\lpreset_family} ]  (44 entries)\\lfamilies: SimBench, CIGRE, Kerber, Dickert, Synthetic LV", F["cat"]),
        ("gf", "get_preset_families() -> family names (menu order)", F["cat"]),
        ("gp", "get_presets_for_family(family) -> preset entries", F["cat"]),
    ])
    lane(g, "wiz", "wizard._ask_network  (preset branch)", F["wiz"], [
        ("w1", "pick family (get_preset_families)\\lthen pick preset (get_presets_for_family)", F["wiz"]),
        ("w2", "NetworkConfig(source_type='preset',\\lpreset_name, preset_family)", F["wiz"]),
    ])
    lane(g, "exe", "executor._preset_loaders  (preset_name -> constructor)", F["exe"], [
        ("el", "_preset_loaders() dict\\lpreset_name -> zero-arg constructor callable\\l(names verified vs pandapower 3.4.0)", F["exe"]),
    ])
    lane(g, "bld", "pandapower.networks / simbench builders", F["bld"], [
        ("b1", "SimBench (1):\\lsb.get_simbench_net('1-MV-rural--2-sw')", F["bld"]),
        ("b2", "CIGRE (3):\\lpn.create_cigre_network_mv(with_der=False | 'pv_wind');\\lpn.create_cigre_network_lv()", F["bld"]),
        ("b3", "Kerber (17):\\lpn.create_kerber_* (7 std) ;\\lpn.kb_extrem_* (10 extreme)", F["bld"]),
        ("b4", "Dickert (18):\\lpn.create_dickert_lv_network(feeders_range,\\llinetype, customer, case)  ['average', 'C&OHL']", F["bld"]),
        ("b5", "Synthetic LV (5):\\lpn.create_synthetic_voltage_control_lv_network(name)", F["bld"]),
    ])
    lane(g, "out", "Result", F["out"], [
        ("net", "pandapowerNet\\l(-> oversize_inverters, profiles, run_benchmark)", F["out"]),
    ])
    lane(g, "note", "Verified note", F["note"], [
        ("nc", "network_catalogue.py builds NO networks - menu strings only.\\lConstruction lives in executor._preset_loaders().\\l44 = 1 + 3 + 17 + 18 + 5.", F["note"]),
    ])
    # menu data -> accessors
    D(g, "pc", "gf"); D(g, "pc", "gp")
    # accessors -> wizard
    E(g, "gf", "w1", constraint="false"); D(g, "gp", "w1", "entries", constraint="false")
    E(g, "w1", "w2")
    # wizard config -> executor loader
    E(g, "w2", "el", "preset_name")
    # loader -> builders
    E(g, "el", "b1"); E(g, "el", "b2"); E(g, "el", "b3"); E(g, "el", "b4"); E(g, "el", "b5")
    # builders -> net
    D(g, "b1", "net"); D(g, "b2", "net"); D(g, "b3", "net")
    D(g, "b4", "net"); D(g, "b5", "net")
    g.edge("net", "nc", style="invis")
    legend(g, "net")
    return g


g = build()
g.format = "svg"; g.render("/home/claude/flow_catalogue", cleanup=True)
g.format = "pdf"; g.render("/home/claude/flow_catalogue", cleanup=True)
print("wrote flow_catalogue")
