#!/usr/bin/env python3
"""
flow_s1.py — Scenario 1 (Baseline, no control) flow diagrams.

Same template as flow_s4.py: vertical swimlanes per domain, orthogonal
routing, tight spacing, bottom legend, three edge classes
(EXEC solid black / DATA dashed blue / COMMS dotted orange — no COMMS here,
baseline has no hardware).

Two diagrams:
  1. flow_s1_init  — initialization & teardown
  2. flow_s1_exec  — annual sweep (pandapower run_timeseries) + the separate
                     post-processing violation pass

Structural note (verified): Scenario 1 has NO manual control loop. ConstControl
feeds profiles, run_timeseries owns the per-step sweep (runpp via the
_timed_runpp hook, OutputWriter logs tables), then a SECOND loop in
run_scenario_1 post-processes the logged tables into per-step records.

Sources: scenario_1_baseline.py, scenario_result.py, violation_detector.py,
publisher.py.
"""
from graphviz import Digraph

F = {"m": "#E7F0FA", "s": "#EDF5E4", "eng": "#F1EFE9",
     "pub": "#EAE7F6", "gate": "#FCF5D6"}
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


def build_init():
    g = base("Scenario 1 — Baseline : Initialization & Teardown", ranksep="0.34")
    lane(g, "m", "Master script  ·  scenario_1_baseline.py", F["m"], [
        ("i1", "run_scenario_1(net, profiles, v_min/v_max)", F["m"]),
        ("i2", "align profiles to this net\\l-> ap : DER P, load P/Q, times, dt_s", F["m"]),
        ("i3", "drop stale controllers, clear results", F["m"]),
        ("i4", "attach ConstControl\\lsgen P ; load P, Q  (columns = element idx)", F["m"]),
        ("i5", "set up OutputWriter\\llog vm_pu, line/trafo loading, losses, extgrid P", F["m"]),
        ("t0", "== run_timeseries sweep + post-process ==\\l(Diagram 2)", F["gate"]),
        ("t2", "aggregate records -> ScenarioResult\\ln_violation_steps, VDI, max/min vm", F["m"]),
        ("t3", "return ScenarioResult", F["m"]),
    ])
    lane(g, "s", "Python sub-modules", F["s"], [
        ("s1", "adapt_profiles()\\lalign DER/load profiles, resolve dt_s", F["s"]),
        ("s2", "ScenarioResult.from_records()\\lreduce per-step records -> summary", F["s"]),
    ])
    lane(g, "eng", "pandapower engine", F["eng"], [
        ("e1", "DFData(profile df)\\lper-step data source", F["eng"]),
        ("e2", "ConstControl\\lwrites p_mw / q_mvar each step", F["eng"]),
        ("e3", "OutputWriter\\llogs res_bus / res_line / res_trafo", F["eng"]),
    ])
    lane(g, "pub", "Publisher", F["pub"], [
        ("p1", "on_scenario_start", F["pub"]),
        ("p2", "on_scenario_end", F["pub"]),
    ])
    E(g, "i1", "i2"); E(g, "i2", "i3"); E(g, "i3", "i4"); E(g, "i4", "i5")
    E(g, "i5", "t0"); E(g, "t0", "t2"); E(g, "t2", "t3")
    E(g, "i2", "s1", constraint="false")
    E(g, "i2", "p1", constraint="false")
    E(g, "i4", "e1", constraint="false")
    E(g, "i4", "e2", constraint="false")
    E(g, "i5", "e3", constraint="false")
    E(g, "t2", "s2", constraint="false")
    E(g, "t2", "p2", constraint="false")
    D(g, "s1", "i2", "ap", constraint="false")
    D(g, "s2", "t3", "ScenarioResult")
    legend(g, "t3")
    return g


def build_exec():
    g = base("Scenario 1 — Baseline : Annual Sweep + Post-Processing Pass", ranksep="0.34")
    lane(g, "eng", "pandapower engine  ·  run_timeseries (owns the sweep)", F["eng"], [
        ("g0", "run_timeseries(time_steps,\\lvoltage_depend_loads=False, run=_timed_runpp)", F["eng"]),
        ("g1", "step t: ConstControl writes\\lnet.sgen.p_mw, net.load P/Q  <- DFData[t]", F["eng"]),
        ("g2", "_timed_runpp -> pp.runpp (NR)\\lrecord t_total_ms per step", F["eng"]),
        ("g3", "OutputWriter logs res_bus.vm_pu,\\lline/trafo loading, losses, extgrid P", F["eng"]),
        ("g4", "next t  (loop until sweep done)", F["gate"]),
    ])
    lane(g, "m", "Master  ·  post-processing loop (run_scenario_1)", F["m"], [
        ("q0", "for each logged step t:", F["m"]),
        ("q1", "read logged vm_pu[t],\\lline/trafo loading[t] from OutputWriter", F["m"]),
        ("q2", "flag violations vs thresholds\\lvm outside band ; loading > max (+/- eps)", F["m"]),
        ("q3", "build TimestepRecord\\lov/uv buses, ol lines/trafos, losses, t_total_ms", F["m"]),
        ("q4", "publish on_timestep; append; next t", F["m"]),
    ])
    lane(g, "s", "Python sub-modules", F["s"], [
        ("d1", "violation_detector thresholds\\lV_MIN/V_MAX, LINE/TRAFO_MAX_LOADING, eps", "#EEEDFE"),
        ("d2", "make_record_from_report()\\lassemble TimestepRecord", F["s"]),
    ])
    lane(g, "pub", "Publisher", F["pub"], [
        ("pp1", "on_timestep -> live JSONL frame", F["pub"]),
    ])
    # engine sweep chain
    E(g, "g0", "g1"); E(g, "g1", "g2"); E(g, "g2", "g3"); E(g, "g3", "g4")
    E(g, "g4", "g1", "loop", style="dashed", constraint="false")
    # hand-off to post-processing
    E(g, "g4", "q0", "after sweep")
    # post-processing chain
    E(g, "q0", "q1"); E(g, "q1", "q2"); E(g, "q2", "q3"); E(g, "q3", "q4")
    E(g, "q4", "q0", "loop", style="dashed", constraint="false")
    # sub-module + publisher calls
    E(g, "q2", "d1", constraint="false")
    E(g, "q3", "d2", constraint="false")
    E(g, "q4", "pp1", constraint="false")
    # data
    D(g, "g3", "q1", "logged tables", constraint="false")
    D(g, "g2", "q3", "t_total_ms", constraint="false")
    D(g, "d2", "q3", "TimestepRecord", constraint="false")
    legend(g, "q4")
    return g


for name, gb in (("flow_s1_init", build_init), ("flow_s1_exec", build_exec)):
    g = gb()
    g.format = "svg"; g.render(f"/home/claude/{name}", cleanup=True)
    g.format = "pdf"; g.render(f"/home/claude/{name}", cleanup=True)
    print("wrote", name)
