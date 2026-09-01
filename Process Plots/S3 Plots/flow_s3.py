#!/usr/bin/env python3
"""
flow_s3.py — Scenario 3 (SVC) flow diagrams. Same template as flow_s4.

  1. flow_s3_init  — init (Q_MAX/k_q, bus selection, SVC sgen create) & teardown
  2. flow_s3_loop  — per-timestep deadbanded-droop control loop

Verified against scenario_3_svc.py, scenario_result.py, violation_detector.py.
Facts: single SVC sgen (p=0, controllable q) at one fixed MV bus chosen by
stress analysis; deadbanded droop toward 1.00 pu; Q_MAX=0.20xSUM(sn_mva),
k_q=Q_MAX/0.03; DER sgen Q forced 0; SVC sgen removed in a finally block.
"""
from graphviz import Digraph

F = {"m": "#E7F0FA", "s": "#EDF5E4", "solv": "#F1EFE9",
     "pub": "#EAE7F6", "gate": "#FCF5D6", "ctl": "#F3F8EC"}
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
    g = base("Scenario 3 — SVC : Initialization & Teardown", ranksep="0.34")
    lane(g, "m", "Master script  ·  scenario_3_svc.py", F["m"], [
        ("i1", "run_scenario_3(net, profiles, v_min/v_max)", F["m"]),
        ("i2", "align profiles to this net -> ap", F["m"]),
        ("i3", "drop stale controllers, clear results", F["m"]),
        ("i4", "compute SVC params\\lQ_MAX = 0.20 x SUM(trafo sn_mva)\\lk_q = Q_MAX / 0.03", F["m"]),
        ("i5", "select fixed SVC bus\\lstress -> lowest mean-vm MV bus\\l(fallback: violation scan, last MV bus)", F["m"]),
        ("i6", "create SVC sgen at bus\\lp=0, q=0, type=SVC", F["m"]),
        ("t0", "== per-timestep loop ==  (Diagram 2)", F["gate"]),
        ("t2", "finally: remove SVC sgen (clean net)", F["gate"]),
        ("t3", "aggregate -> ScenarioResult\\lsvc_bus, svc_q_max", F["m"]),
        ("t4", "return ScenarioResult", F["m"]),
    ])
    lane(g, "s", "Python sub-modules", F["s"], [
        ("s1", "adapt_profiles()", F["s"]),
        ("s2", "_compute_svc_params()", F["s"]),
        ("s3", "_select_svc_bus()\\l+ stress.apply_overvoltage_stress", F["s"]),
        ("s4", "ScenarioResult.from_records()", F["s"]),
    ])
    lane(g, "solv", "Solver", F["solv"], [
        ("pp", "pandapower runpp\\lstress solve for bus ranking", F["solv"]),
    ])
    lane(g, "pub", "Publisher", F["pub"], [
        ("p1", "on_scenario_start", F["pub"]),
        ("p2", "on_scenario_end", F["pub"]),
    ])
    E(g, "i1", "i2"); E(g, "i2", "i3"); E(g, "i3", "i4"); E(g, "i4", "i5")
    E(g, "i5", "i6"); E(g, "i6", "t0"); E(g, "t0", "t2"); E(g, "t2", "t3"); E(g, "t3", "t4")
    E(g, "i2", "s1", constraint="false")
    E(g, "i2", "p1", constraint="false")
    E(g, "i4", "s2", constraint="false")
    E(g, "i5", "s3", constraint="false")
    E(g, "s3", "pp", constraint="false")
    E(g, "t3", "s4", constraint="false")
    E(g, "t3", "p2", constraint="false")
    D(g, "s1", "i2", "ap", constraint="false")
    D(g, "pp", "i5", "bus vm ranking", constraint="false")
    D(g, "s4", "t4", "ScenarioResult")
    legend(g, "t4")
    return g


def build_loop():
    g = base("Scenario 3 — SVC : Per-Timestep Droop-Control Loop", ranksep="0.32")
    lane(g, "m", "Master  ·  timestep loop (run_scenario_3)", F["m"], [
        ("L1", "[1] write profiles\\lload P/Q ; DER sgen P=der_p[t], Q=0 ; SVC q=0", F["m"]),
        ("L2", "[2] pre-control PF (SVC q=0)", F["m"]),
        ("L3", "converged?\\lno -> SVC holds q=0, record, next", F["gate"]),
        ("L4", "[3] read SVC-bus voltage\\lvm_svc = res_bus[svc_bus]", F["m"]),
        ("L5", "[4] apply q_cmd to SVC sgen\\l|q|>0 -> post-PF ; else reuse pre-PF", F["m"]),
        ("L6", "[5] build TimestepRecord\\lvm/loading, violations, svc_q, saturated", F["m"]),
        ("L7", "publish on_timestep; append; next t", F["m"]),
    ])
    lane(g, "s", "Python sub-modules", F["s"], [
        ("dq", "_droop_q(vm, Q_MAX, k_q)\\lerror = 1.00 - vm\\l|err|<=0.01 -> q=0\\lerr>db: inject k_q(err-db)\\lerr<-db: absorb k_q(err+db)\\lclip +/-Q_MAX ; saturation flag", F["ctl"]),
        ("d1", "violation thresholds\\lV band + line / trafo loading", "#EEEDFE"),
        ("d2", "TimestepRecord\\lsvc_q_mvar, svc_saturated", F["s"]),
    ])
    lane(g, "solv", "Solver", F["solv"], [
        ("pp", "pandapower runpp (Newton-Raphson)\\lvoltage_depend_loads=False", F["solv"]),
    ])
    lane(g, "pub", "Publisher", F["pub"], [
        ("pb", "on_timestep -> live JSONL", F["pub"]),
    ])
    E(g, "L1", "L2"); E(g, "L2", "L3")
    E(g, "L3", "L4", "converged"); E(g, "L3", "L7", "diverged", style="dashed", constraint="false")
    E(g, "L4", "L5"); E(g, "L5", "L6"); E(g, "L6", "L7")
    E(g, "L7", "L1", "next t", constraint="false")
    E(g, "L4", "dq", "compute Q", constraint="false")
    E(g, "L2", "pp", "pre", constraint="false")
    E(g, "L5", "pp", "post (if q!=0)", constraint="false")
    E(g, "L6", "d1", constraint="false")
    E(g, "L6", "d2", constraint="false")
    E(g, "L7", "pb", constraint="false")
    D(g, "pp", "L4", "vm_svc", constraint="false")
    D(g, "dq", "L5", "q_cmd, saturated", constraint="false")
    D(g, "pp", "L6", "res tables", constraint="false")
    D(g, "d2", "L6", "TimestepRecord", constraint="false")
    legend(g, "L7")
    return g


for name, gb in (("flow_s3_init", build_init), ("flow_s3_loop", build_loop)):
    g = gb()
    g.format = "svg"; g.render(f"/home/claude/{name}", cleanup=True)
    g.format = "pdf"; g.render(f"/home/claude/{name}", cleanup=True)
    print("wrote", name)
