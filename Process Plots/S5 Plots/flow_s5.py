#!/usr/bin/env python3
"""
flow_s5.py — Scenario 5 (AC OPF benchmark) flow diagrams. Same template.

  1. flow_s5_init  — OPF table setup & teardown
  2. flow_s5_loop  — per-timestep OPF-state build + runopp solve

Verified against scenario_5_opf.py, scenario_result.py, violation_detector.py.
Facts: "ideal controller" reference, NOT in the HIL loop; CIGRE MV only
(SimBench MV fails runopp on HV/MV Jacobian ill-conditioning); PYPOWER OPF
backend (cyipopt availability import). Objective maximises accepted DER P
(DER cost -1.0, ext_grid cost +0.001). Non-converged timestep is recorded as
non-converged, not physical infeasibility. publish_fn accepted but ignored.
"""
from graphviz import Digraph

F = {"m": "#E7F0FA", "s": "#EDF5E4", "solv": "#F1EFE9", "gate": "#FCF5D6"}
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
    g = base("Scenario 5 — AC OPF : Initialization & Teardown  (CIGRE MV only)", ranksep="0.34")
    lane(g, "m", "Master script  ·  scenario_5_opf.py", F["m"], [
        ("i1", "run_scenario_5(net, profiles, v_min/v_max,\\lline/trafo limits)", F["m"]),
        ("i2", "create_continuous_bus_index\\l-> align profiles -> ap", F["m"]),
        ("i3", "clear stale controllers + poly_cost", F["m"]),
        ("i4", "_setup_opf\\lsgen controllable=False; init P/Q bound cols", F["m"]),
        ("i5", "_prepare_ext_grid_for_opf\\lcontrollable=True; feeder-scale +/- P,Q bounds", F["m"]),
        ("i6", "_apply_network_constraints\\lbus min/max_vm; line/trafo max_loading", F["m"]),
        ("i7", "diagnostic; _compute_sn_rated\\lsn_mva -> |p_mw| -> peak profile", F["m"]),
        ("t0", "== per-timestep OPF loop ==  (Diagram 2)", F["gate"]),
        ("t2", "aggregate -> ScenarioResult", F["m"]),
        ("t3", "return ScenarioResult", F["m"]),
    ])
    lane(g, "s", "Python sub-modules", F["s"], [
        ("s1", "adapt_profiles()", F["s"]),
        ("s2", "ScenarioResult.from_records()", F["s"]),
    ])
    lane(g, "solv", "Solver", F["solv"], [
        ("dg", "pandapower diagnostic\\lpre-OPF sanity of tables/bounds", F["solv"]),
    ])
    E(g, "i1", "i2"); E(g, "i2", "i3"); E(g, "i3", "i4"); E(g, "i4", "i5")
    E(g, "i5", "i6"); E(g, "i6", "i7"); E(g, "i7", "t0"); E(g, "t0", "t2"); E(g, "t2", "t3")
    E(g, "i2", "s1", constraint="false")
    E(g, "i7", "dg", constraint="false")
    E(g, "t2", "s2", constraint="false")
    D(g, "s1", "i2", "ap", constraint="false")
    D(g, "s2", "t3", "ScenarioResult")
    legend(g, "t3")
    return g


def build_loop():
    g = base("Scenario 5 — AC OPF : Per-Timestep OPF-Solve Loop", ranksep="0.32")
    lane(g, "m", "Master  ·  timestep loop (run_scenario_5)", F["m"], [
        ("L1", "[1] p_bound = der_p[t].clip(>=0)\\lactive DERs = p_bound > eps", F["m"]),
        ("L2", "[2] Q limit per DER\\lq_lim = min( Q_RATIO x sn [VDE],\\lsqrt(sn^2 - p^2) [inverter circle] )", F["m"]),
        ("L3", "[3] set OPF operating point\\lreset all DERs (uncontrollable, 0);\\lactive: controllable, max_p=p_bound, +/-q_lim", F["m"]),
        ("L4", "[4] write loads; rebuild poly_cost\\lext_grid +0.001 ; each active DER -1.0", F["m"]),
        ("L5", "[5] solve AC OPF  runopp(net)", F["m"]),
        ("L6", "converged?\\lyes -> dispatch = benchmark,\\lcurtail = p_target - p_applied\\lno -> record non-converged", F["gate"]),
        ("L7", "[6] build TimestepRecord; append; next t", F["m"]),
    ])
    lane(g, "solv", "Solver", F["solv"], [
        ("opf", "pandapower runopp  (AC OPF, PYPOWER / cyipopt)\\lobjective: maximise SUM P_DER\\ls.t. V band, thermal, DER capability", F["solv"]),
    ])
    lane(g, "s", "Python sub-modules", F["s"], [
        ("d1", "violation thresholds\\lV band + line / trafo loading", "#EEEDFE"),
        ("d2", "TimestepRecord\\lp_target, p_applied (curtailment)", F["s"]),
    ])
    E(g, "L1", "L2"); E(g, "L2", "L3"); E(g, "L3", "L4"); E(g, "L4", "L5")
    E(g, "L5", "L6"); E(g, "L6", "L7"); E(g, "L7", "L1", "next t", constraint="false")
    E(g, "L5", "opf", "solve", constraint="false")
    E(g, "L7", "d1", constraint="false")
    E(g, "L7", "d2", constraint="false")
    D(g, "opf", "L6", "dispatch / status", constraint="false")
    D(g, "d2", "L7", "TimestepRecord", constraint="false")
    legend(g, "L7")
    return g


for name, gb in (("flow_s5_init", build_init), ("flow_s5_loop", build_loop)):
    g = gb()
    g.format = "svg"; g.render(f"/home/claude/{name}", cleanup=True)
    g.format = "pdf"; g.render(f"/home/claude/{name}", cleanup=True)
    print("wrote", name)
