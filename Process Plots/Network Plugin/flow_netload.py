#!/usr/bin/env python3
"""
flow_netload.py — network_plugin.py loader. Same template as flow_s4.

  1. flow_netload_load  — load_network_from_yaml: parse/validate, source
                          dispatch (json/pickle/function), plugin_meta, validate
  2. flow_netload_strat — _build_profiles_for_strategy: strategy dispatch
                          (simbench_native / dwd_pvlib / flat) + make_profile_factory

Verified against network_plugin.py function bodies (not docstrings):
- _load_yaml_config validates source in _VALID_SOURCES and strategy in
  _VALID_STRATEGIES=("simbench_native","dwd_pvlib","flat"); 'custom' is REJECTED
  (ValueError). Custom weather files go through dwd_pvlib + data_dir/file_map/col_map.
- source dispatch: json->pp.from_json, pickle->pp.from_pickle, function->importlib
  file-location zero-arg fn; result type-checked vs pp.pandapowerNet.
- strategy dispatch is if/elif/else: simbench_native (fallback to dwd_pvlib when
  no net.profiles metadata) / dwd_pvlib / else->flat.
"""
from graphviz import Digraph

F = {"m": "#E7F0FA", "s": "#EDF5E4", "src": "#F1EFE9",
     "val": "#EAE7F6", "gate": "#FCF5D6", "out": "#F3F8EC", "note": "#FCEBEA"}
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


def build_load():
    g = base("network_plugin : load_network_from_yaml (load & validate)", ranksep="0.34")
    lane(g, "m", "load_network_from_yaml  ·  network_plugin.py", F["m"], [
        ("n1", "load_network_from_yaml(yaml_path)", F["m"]),
        ("n2", "_load_yaml_config: parse + validate\\lname/label; source in {json,pickle,function};\\lpaths resolve YAML-relative;\\lstrategy in {simbench_native,dwd_pvlib,flat};\\lyear 1990-2100; 0.5<=v_min<v_max<=1.5", F["m"]),
        ("n3", "_load_net(cfg): source dispatch", F["gate"]),
        ("n4", "_build_profiles_for_strategy (Diagram 2)", F["m"]),
        ("n5", "attach plugin_meta\\lname,label,source, strategy vs requested_strategy,\\lyear, v_min/v_max, notes, yaml_path", F["m"]),
        ("n6", "return (net, profiles)", F["m"]),
        ("v1", "validate_network_plugin(net, profiles)\\l-> warning strings (NOT exceptions)", F["m"]),
    ])
    lane(g, "src", "Network source dispatch", F["src"], [
        ("sj", "json -> pandapower.from_json(path)\\l(recommended; keeps net.profiles)", F["src"]),
        ("sp", "pickle -> pandapower.from_pickle(path)", F["src"]),
        ("sf", "function -> importlib file-location\\lzero-arg fn -> pandapowerNet", F["src"]),
        ("tc", "type-check result vs pp.pandapowerNet", F["src"]),
    ])
    lane(g, "val", "validate_network_plugin checks", F["val"], [
        ("val", "profile/net column alignment; empty-DER;\\lZIP-load (const_z); orphan profile cols;\\ltransformer / trafo3w awareness", F["val"]),
    ])
    E(g, "n1", "n2"); E(g, "n2", "n3"); E(g, "n3", "n4"); E(g, "n4", "n5")
    E(g, "n5", "n6"); E(g, "n6", "v1")
    E(g, "n3", "sj", "json", constraint="false")
    E(g, "n3", "sp", "pickle", style="dashed", constraint="false")
    E(g, "n3", "sf", "function", style="dashed", constraint="false")
    E(g, "sj", "tc"); E(g, "sp", "tc"); E(g, "sf", "tc")
    E(g, "v1", "val", constraint="false")
    D(g, "tc", "n4", "net", constraint="false")
    legend(g, "v1")
    return g


def build_strat():
    g = base("network_plugin : _build_profiles_for_strategy (strategy dispatch)",
             ranksep="0.32")
    lane(g, "m", "_build_profiles_for_strategy  ·  network_plugin.py", F["m"], [
        ("s0", "_build_profiles_for_strategy(net, cfg)", F["m"]),
        ("s1", "strategy == simbench_native\\lAND no net.profiles metadata?\\l-> fall back to dwd_pvlib\\l(record requested vs used)", F["gate"]),
        ("s2", "dispatch on strategy (if / elif / else)", F["gate"]),
        ("s3", "return (profiles, strategy_used)", F["m"]),
    ])
    lane(g, "st", "Three strategies (only valid keywords)", F["s"], [
        ("sn", "simbench_native -> _build_profiles_simbench_native\\lsb.get_absolute_values(net); rebuild 15-min index (year);\\lpv_mask incl lv_res; night-zero; clip", "#EDF5E4"),
        ("dp", "dwd_pvlib -> _build_profiles_dwd\\lbuild_annual_profiles(data_dir or default,\\lfile_map, col_map)  <- custom weather hooks HERE", "#E6F5EF"),
        ("fl", "flat (else) -> _build_profiles_flat\\lconstant rated: net.load.p_mw / net.sgen.p_mw,\\lfull year 15-min", "#EEEDFE"),
    ])
    lane(g, "hc", "HC re-benchmark factory", F["out"], [
        ("mf", "make_profile_factory(yaml)\\lreuses _build_profiles_for_strategy on stressed net\\l(same strategy + fallback; meta name+'_hc_stressed')", F["out"]),
    ])
    lane(g, "note", "Verified note", F["note"], [
        ("nc", "strategy: custom is NOT valid on this path\\l_load_yaml_config raises ValueError.\\lFor custom weather CSVs use dwd_pvlib + file_map/col_map.\\l(wizard/executor 'custom' is a separate code path)", F["note"]),
    ])
    E(g, "s0", "s1"); E(g, "s1", "s2"); E(g, "s2", "s3")
    E(g, "s2", "sn", "simbench_native"); E(g, "s2", "dp", "dwd_pvlib", style="dashed")
    E(g, "s2", "fl", "else -> flat", style="dashed")
    E(g, "s1", "dp", "fallback", style="dashed", constraint="false")
    D(g, "sn", "s3", "profiles", constraint="false")
    D(g, "dp", "s3", "profiles", constraint="false")
    D(g, "fl", "s3", "profiles", constraint="false")
    E(g, "s0", "mf", "HC path reuse", style="dashed", constraint="false")
    g.edge("s3", "nc", style="invis")
    legend(g, "s3")
    return g


for name, gb in (("flow_netload_load", build_load), ("flow_netload_strat", build_strat)):
    g = gb()
    g.format = "svg"; g.render(f"/home/claude/{name}", cleanup=True)
    g.format = "pdf"; g.render(f"/home/claude/{name}", cleanup=True)
    print("wrote", name)
