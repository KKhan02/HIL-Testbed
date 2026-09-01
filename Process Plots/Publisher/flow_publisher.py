#!/usr/bin/env python3
"""
flow_publisher.py — publisher.py. Same template as flow_s4.

  1. flow_publisher_static  — publish_result: post-run JSON export
  2. flow_publisher_live    — PublishHandle: live JSONL streaming callbacks

Verified from publisher.py function bodies (not docstrings):
- publish_result writes topology.json, profiles.json, hc.json (if hc_results),
  scenarios/<sid>.json (per non-None ScenarioResult), comparison.json; returns
  {logical name -> Path}. Builders: build_topology / build_profiles_payload /
  build_hc_payload / build_scenario_payload.
- PublishHandle: on_scenario_start truncates live/<sid>.jsonl + stores context;
  on_timestep emits build_live_frame every update_every_k steps (rec.t % k == 0);
  on_scenario_end appends {"event":"scenario_complete"}. Runners call all three;
  BenchmarkConfig.publish_fn holds the handle object.
"""
from graphviz import Digraph

F = {"m": "#E7F0FA", "b": "#EDF5E4", "out": "#F1EFE9",
     "run": "#EDE7F6", "gate": "#FCF5D6", "note": "#F3F8EC"}
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
        D(c, "ld1", "ld2", "data (dict/df)")
        c.edge("le2", "ld1", style="invis")
    g.edge(anchor, "le1", style="invis")


def build_static():
    g = base("publisher : publish_result (post-run static export)", ranksep="0.32")
    lane(g, "m", "publish_result  ·  publisher.py", F["m"], [
        ("p1", "publish_result(result, net, profiles,\\loutput_dir, network_id)", F["m"]),
        ("p2", "[1] build_topology(net) + network_id -> topology.json", F["m"]),
        ("p3", "[2] build_profiles_payload(profiles) -> profiles.json", F["m"]),
        ("p4", "[3] if result.hc_results:\\lbuild_hc_payload -> hc.json", F["gate"]),
        ("p5", "[4] per non-None ScenarioResult:\\lbuild_scenario_payload -> scenarios/<sid>.json", F["m"]),
        ("p6", "[5] comparison_df.to_dict(records) -> comparison.json", F["m"]),
        ("p7", "return written : {logical name -> Path}", F["m"]),
    ])
    lane(g, "b", "Payload builders", F["b"], [
        ("bt", "build_topology\\lbuses (x/y), lines, trafos, sgens, loads,\\lfeeder distances from slack", F["b"]),
        ("bp", "build_profiles_payload\\lload + DER @10-min + hourly totals\\l(no pre-sliced extreme days)", F["b"]),
        ("bh", "build_hc_payload\\lbaseline + volt_var HCResult + sweep params", F["b"]),
        ("bs", "build_scenario_payload\\lper-timestep: vm_pu/bus, line/trafo loading,\\ltap, Q setpoints, curtailment, energy", F["b"]),
    ])
    lane(g, "out", "output_dir/ files", F["out"], [
        ("fo", "topology.json ; profiles.json ; hc.json ;\\lscenarios/<sid>.json (per scenario) ;\\lcomparison.json", F["out"]),
    ])
    E(g, "p1", "p2"); E(g, "p2", "p3"); E(g, "p3", "p4"); E(g, "p4", "p5")
    E(g, "p5", "p6"); E(g, "p6", "p7")
    E(g, "p2", "bt", constraint="false")
    E(g, "p3", "bp", constraint="false")
    E(g, "p4", "bh", constraint="false")
    E(g, "p5", "bs", constraint="false")
    D(g, "bt", "fo", constraint="false")
    D(g, "bs", "fo", constraint="false")
    D(g, "p7", "fo", "written paths", constraint="false")
    legend(g, "p7")
    return g


def build_live():
    g = base("publisher : PublishHandle (live JSONL streaming)", ranksep="0.34")
    lane(g, "run", "Scenario runner  (via BenchmarkConfig.publish_fn = handle)", F["run"], [
        ("r1", "before loop:\\lpublish_fn.on_scenario_start(sid, label, t_total)", F["run"]),
        ("r2", "each timestep:\\lpublish_fn.on_timestep(rec)", F["run"]),
        ("r3", "after loop:\\lpublish_fn.on_scenario_end(result)", F["run"]),
    ])
    lane(g, "m", "PublishHandle  ·  publisher.py", F["m"], [
        ("h1", "on_scenario_start\\lstore context; _live_path = live/<sid>.jsonl;\\lmkdir; truncate previous file", F["m"]),
        ("h2", "on_timestep\\lif rec.t % update_every_k == 0:\\lbuild_live_frame -> _append_jsonl", F["gate"]),
        ("h3", "on_scenario_end\\lappend {'event':'scenario_complete', elapsed_s}", F["m"]),
    ])
    lane(g, "out", "output_dir/live/", F["out"], [
        ("j", "live/<sid>.jsonl\\lnewline-delimited compact frames\\l(Streamlit tails + polls)", F["out"]),
    ])
    lane(g, "note", "Cadence", F["note"], [
        ("nc", "update_every_k (default 6 = hourly @ 10-min)\\lk=1 every step; k=144 daily.\\lterminal event lets the app switch to static payload", F["note"]),
    ])
    E(g, "r1", "r2"); E(g, "r2", "r2", "loop"); E(g, "r2", "r3")
    E(g, "r1", "h1", constraint="false")
    E(g, "r2", "h2", constraint="false")
    E(g, "r3", "h3", constraint="false")
    D(g, "h1", "j", "create/truncate", constraint="false")
    D(g, "h2", "j", "append frame", constraint="false")
    D(g, "h3", "j", "append end", constraint="false")
    g.edge("j", "nc", style="invis")
    legend(g, "h3")
    return g


for name, gb in (("flow_publisher_static", build_static), ("flow_publisher_live", build_live)):
    g = gb()
    g.format = "svg"; g.render(f"/home/claude/{name}", cleanup=True)
    g.format = "pdf"; g.render(f"/home/claude/{name}", cleanup=True)
    print("wrote", name)
