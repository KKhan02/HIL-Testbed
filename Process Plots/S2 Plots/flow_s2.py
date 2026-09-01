#!/usr/bin/env python3
"""
flow_s2.py — Scenario 2 (OLTC-only) flow diagrams. Same template as flow_s4.

Two diagrams:
  1. flow_s2_init  — init (incl. tap-sign calibration probe) & teardown
  2. flow_s2_loop  — per-timestep tap-control loop

Verified against scenario_2_oltc.py, scenario_result.py, violation_detector.py.
Key facts: manual loop; net.sgen.q_mvar forced 0 (pure OLTC, no DER Q); one tap
step/step when mean control-bus vm leaves [0.98,1.02]; tap sign calibrated once
by probing the PF; post-tap divergence rolls back to the previous tap.
"""
from graphviz import Digraph

F = {"m": "#E7F0FA", "s": "#EDF5E4", "solv": "#F1EFE9",
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
    g = base("Scenario 2 — OLTC : Initialization & Teardown", ranksep="0.34")
    lane(g, "m", "Master script  ·  scenario_2_oltc.py", F["m"], [
        ("i1", "run_scenario_2(net, profiles, v_min/v_max,\\ltap_metadata_override)", F["m"]),
        ("i2", "align profiles to this net -> ap", F["m"]),
        ("i3", "select HV/MV OLTC group (ganged)\\l+ control busbars (LV side)", F["m"]),
        ("i4", "complete / validate tap metadata\\l(fill defaults, audit print)", F["m"]),
        ("i5", "calibrate tap sign\\lprobe: move tap, runpp, measure d(vm)", F["m"]),
        ("i6", "init tap state\\lcurrent = neutral ; gang limits", F["m"]),
        ("t0", "== per-timestep loop ==  (Diagram 2)", F["gate"]),
        ("t2", "aggregate -> ScenarioResult\\l+ tap summary (moves, blocks, min/max)", F["m"]),
        ("t3", "return ScenarioResult", F["m"]),
    ])
    lane(g, "s", "Python sub-modules", F["s"], [
        ("s1", "adapt_profiles()", F["s"]),
        ("s2", "ScenarioResult.from_records()", F["s"]),
    ])
    lane(g, "solv", "Solver", F["solv"], [
        ("pp", "pandapower runpp\\lprobe solve for tap sign", F["solv"]),
    ])
    lane(g, "pub", "Publisher", F["pub"], [
        ("p1", "on_scenario_start", F["pub"]),
        ("p2", "on_scenario_end", F["pub"]),
    ])
    E(g, "i1", "i2"); E(g, "i2", "i3"); E(g, "i3", "i4"); E(g, "i4", "i5")
    E(g, "i5", "i6"); E(g, "i6", "t0"); E(g, "t0", "t2"); E(g, "t2", "t3")
    E(g, "i2", "s1", constraint="false")
    E(g, "i2", "p1", constraint="false")
    E(g, "i5", "pp", constraint="false")
    E(g, "t2", "s2", constraint="false")
    E(g, "t2", "p2", constraint="false")
    D(g, "s1", "i2", "ap", constraint="false")
    D(g, "pp", "i5", "d(vm) -> sign", constraint="false")
    D(g, "s2", "t3", "ScenarioResult")
    legend(g, "t3")
    return g


def build_loop():
    g = base("Scenario 2 — OLTC : Per-Timestep Tap-Control Loop", ranksep="0.32")
    lane(g, "m", "Master  ·  timestep loop (run_scenario_2)", F["m"], [
        ("L1", "[1] write profiles\\lload P/Q ; sgen P=der_p[t], sgen Q=0", F["m"]),
        ("L2", "[2] pre-action power flow", F["m"]),
        ("L3", "[3] converged?\\lno -> hold tap, log blocked, record", F["gate"]),
        ("L4", "[4] tap decision\\lvm_ctrl = mean(vm at control buses)\\l>1.02 up / <0.98 down / else hold", F["m"]),
        ("L5", "[5] apply tap  net.trafo.tap_pos = candidate\\l(clip to ganged min/max)", F["m"]),
        ("L6", "[5a] post-tap PF\\lconverged -> accept ; else rollback to prev", F["m"]),
        ("L7", "[6] build TimestepRecord\\lvm/loading, violations, tap flags, losses", F["m"]),
        ("L8", "publish on_timestep; append; next t", F["m"]),
    ])
    lane(g, "solv", "Solver", F["solv"], [
        ("pp", "pandapower runpp (Newton-Raphson)\\lvoltage_depend_loads=False", F["solv"]),
    ])
    lane(g, "s", "Python sub-modules", F["s"], [
        ("d1", "violation thresholds\\lV band + line / trafo loading", "#EEEDFE"),
        ("d2", "TimestepRecord\\ltap_pos / changed / attempted /\\lcandidate / blocked_reason", F["s"]),
    ])
    lane(g, "pub", "Publisher", F["pub"], [
        ("pb", "on_timestep -> live JSONL", F["pub"]),
    ])
    E(g, "L1", "L2"); E(g, "L2", "L3")
    E(g, "L3", "L4", "converged"); E(g, "L3", "L8", "diverged", style="dashed", constraint="false")
    E(g, "L4", "L5", "outside band"); E(g, "L4", "L7", "in band: hold", style="dashed", constraint="false")
    E(g, "L5", "L6"); E(g, "L6", "L7"); E(g, "L7", "L8")
    E(g, "L8", "L1", "next t", constraint="false")
    # solver calls
    E(g, "L2", "pp", "pre", constraint="false")
    E(g, "L6", "pp", "post / rollback", constraint="false")
    # sub-module calls
    E(g, "L7", "d1", constraint="false")
    E(g, "L7", "d2", constraint="false")
    E(g, "L8", "pb", constraint="false")
    # data
    D(g, "pp", "L4", "vm_ctrl", constraint="false")
    D(g, "pp", "L7", "res tables", constraint="false")
    D(g, "d2", "L7", "TimestepRecord", constraint="false")
    legend(g, "L8")
    return g


for name, gb in (("flow_s2_init", build_init), ("flow_s2_loop", build_loop)):
    g = gb()
    g.format = "svg"; g.render(f"/home/claude/{name}", cleanup=True)
    g.format = "pdf"; g.render(f"/home/claude/{name}", cleanup=True)
    print("wrote", name)
