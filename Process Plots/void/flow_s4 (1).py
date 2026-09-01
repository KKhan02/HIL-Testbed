#!/usr/bin/env python3
"""
flow_s4.py — Scenario 4 (Volt-Var HIL) flow diagrams, restructured.

Two diagrams:
  1. flow_s4_init  — initialization & teardown
  2. flow_s4_loop  — per-timestep execution cycle

Layout: vertical swimlanes (one cluster per domain), tight nodesep/ranksep,
orthogonal routing, bottom legend. Three edge classes:
  EXEC  solid black    execution order
  DATA  dashed blue    variable / array / dataframe crossing a domain
  COMMS dotted orange  serial transmission (explicit TX + RX pair)

Node text = verified functional summary, not docstring paste.
"""
from graphviz import Digraph

F = {"m": "#E7F0FA", "s": "#EDF5E4", "solv": "#F1EFE9",
     "hw": "#FBE9DF", "pub": "#EAE7F6", "co": "#F3F8EC", "gate": "#FCF5D6"}
INK = "#1B1B1A"


def E(g, a, b, t="", **k):
    g.edge(a, b, xlabel=t, color=INK, penwidth="2.0", arrowsize="0.75", **k)

def D(g, a, b, t="", **k):
    g.edge(a, b, xlabel=t, color="#1F6FB2", style="dashed", fontcolor="#1F6FB2",
           penwidth="1.25", arrowsize="0.65", **k)

def H(g, a, b, t="", **k):
    g.edge(a, b, xlabel=t, color="#D8641E", style="dotted", fontcolor="#B94F0C",
           penwidth="1.7", arrowsize="0.65", **k)


def base(title, ranksep, nodesep="0.22"):
    g = Digraph(format="svg")
    g.attr(rankdir="TB", splines="ortho", newrank="true", compound="true",
           forcelabels="true", pad="0.35", nodesep=nodesep, ranksep=ranksep,
           fontname="Helvetica-bold", labelloc="t", fontsize="16", label=title,
           bgcolor="transparent")
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica",
           fontsize="10", margin="0.13,0.075", color="#8A887F",
           fontcolor="#1B1B1A", height="0.3")
    g.attr("edge", fontname="Helvetica", fontsize="8.5")
    return g


def lane(g, cid, label, fill, nodes):
    with g.subgraph(name=f"cluster_{cid}") as c:
        c.attr(label=label, style="filled,rounded", fillcolor=fill,
               color="#8AB5D9", penwidth="1.2",
               fontname="Helvetica-bold", fontsize="12",
               fontcolor="#F4F7FF", margin="12")
        for nid, lbl, nf in nodes:
            c.node(nid, lbl, fillcolor=nf, fontcolor="#1B1B1A",
                   color="#D9EAFB", style="filled,rounded")


def legend(g, anchor):
    with g.subgraph(name="cluster_legend") as c:
        c.attr(label="Legend", style="filled,rounded", color="#9BB9D7",
               fillcolor="#DDEAF7", labeljust="l",
               fontname="Helvetica-bold", fontsize="10",
               fontcolor="#1B1B1A", bgcolor="transparent", margin="8")
        c.attr(rankdir="LR")

        c.node("lt1", "execution", shape="plaintext", fontname="Helvetica",
               fontcolor="#1B1B1A")
        c.node("lt2", "serial (HIL)", shape="plaintext", fontname="Helvetica",
               fontcolor="#1B1B1A")

        for n in ("le1", "le2", "lh1", "lh2"):
            c.node(n, "", shape="point", width="0.10", color="#1B1B1A",
                   fillcolor="#1B1B1A", style="filled")

        with c.subgraph() as r1:
            r1.attr(rank="same")
            r1.node("lt1"); r1.node("le1"); r1.node("le2")
        with c.subgraph() as r2:
            r2.attr(rank="same")
            r2.node("lt2"); r2.node("lh1"); r2.node("lh2")

        E(c, "le1", "le2")
        H(c, "lh1", "lh2")
        c.edge("lt1", "lt2", style="invis")

    g.edge(anchor, "lt1", style="invis", constraint="false")

'''def legend(g, anchor):
    with g.subgraph(name="cluster_legend") as c:
        c.attr(label="Legend", style="rounded", color="#8A887F", labeljust="l",
               fontname="Helvetica-bold", fontsize="10", bgcolor="#FFFFFF", margin="6")
        for n in ("le1", "le2", "ld1", "ld2", "lh1", "lh2"):
            c.node(n, "", shape="point", width="0.02", color="#FFFFFF")
        E(c, "le1", "le2", "execution")
        D(c, "ld1", "ld2", "data (array/df)")
        H(c, "lh1", "lh2", "serial TX/RX")
        c.edge("le2", "ld1", style="invis"); c.edge("ld2", "lh1", style="invis")
    g.edge(anchor, "le1", style="invis")'''


def build_init():
    g = base("Scenario 4 — Volt-Var HIL : Initialization & Teardown", ranksep="0.34")
    lane(g, "m", "Master script  ·  scenario_4_volt_var.py", F["m"], [
        ("i1", "run_scenario_4(net, profiles, port,\\ldry_run, coordination, v_min/v_max)", F["m"]),
        ("i2", "align profiles to this net\\l-> ap : DER P, load P/Q, times, dt_s", F["m"]),
        ("i3", "drop stale controllers, clear results", F["m"]),
        ("i4", "instantiate control objects\\lVoltVarController + SensitivityCoordinator", F["m"]),
        ("i5", "build DERDynamics  (Qmax = Q_RATIO x Sn)", F["m"]),
        ("i6", "seed dynamics  q_prev=0, p_prev=P[t0]", F["m"]),
        ("t1", "== per-timestep loop ==  (Diagram 2)", F["gate"]),
        ("t2", "aggregate records -> ScenarioResult\\lcoordination_rate, VDI, curtailment, energy", F["m"]),
        ("t3", "return ScenarioResult", F["m"]),
    ])
    lane(g, "s", "Python sub-modules", F["s"], [
        ("s1", "VoltVarController.configure()\\lresolve DER sgen idx / buses / installed MW", F["s"]),
        ("s2", "ScenarioResult.from_records()\\lreduce per-step records -> summary", F["s"]),
    ])
    lane(g, "pub", "Publisher", F["pub"], [
        ("p1", "on_scenario_start\\lwrite topology, open stream", F["pub"]),
        ("p2", "on_scenario_end\\lflush summary frame", F["pub"]),
    ])
    lane(g, "hw", "Hardware interface (HIL only)", F["hw"], [
        ("h0", "open serial /dev/ttyACM0 @115200", F["hw"]),
        ("h1", "handshake: n_ders", F["hw"]),
        ("h2", "handshake: installed P[]", F["hw"]),
        ("hf", "firmware: store n_ders + P\\lconfigured = true", F["hw"]),
        ("h3", "END -> firmware reset, close port", F["hw"]),
    ])
    E(g, "i1", "i2"); E(g, "i2", "i3"); E(g, "i3", "i4"); E(g, "i4", "i5")
    E(g, "i5", "i6"); E(g, "i6", "t1"); E(g, "t1", "t2"); E(g, "t2", "t3")
    E(g, "i2", "p1", constraint="false")
    E(g, "i4", "s1", constraint="false")
    E(g, "t2", "s2", constraint="false")
    E(g, "t2", "p2", constraint="false")
    D(g, "s2", "t3", "ScenarioResult")
    H(g, "s1", "h0", "if HIL")
    H(g, "h0", "h1"); H(g, "h1", "h2")
    H(g, "h1", "hf", "TX INIT:n")
    H(g, "hf", "h1", "RX ACK", constraint="false")
    H(g, "h2", "hf", "TX P:[]")
    H(g, "hf", "h2", "RX ACK", constraint="false")
    H(g, "t2", "h3", "teardown", constraint="false")
    legend(g, "i1")
    return g


def build_loop():
    g = base("Scenario 4 — Volt-Var HIL : Per-Timestep Execution Loop "
             "(4B; 4A skips coordinate)", ranksep="0.32")
    lane(g, "m", "Master  ·  _run_loop", F["m"], [
        ("A", "[A] P target for t\\lp_target = ap.der_p[t]", F["m"]),
        ("B", "[B] apply load profiles", F["m"]),
        ("C", "[C] run_coordinated_timestep", F["m"]),
        ("D", "[D] build TimestepRecord", F["m"]),
        ("Ee", "[E] curtail if violating\\lP -=10%/iter, re-solve, <=10x", F["m"]),
        ("Ff", "[F] stream frame, append; next t", F["m"]),
    ])
    lane(g, "rct", "sensitivity_coordinator.py  ·  run_coordinated_timestep", F["s"], [
        ("r01", "[0-1] set P=p_target, Q=0", F["s"]),
        ("r2",  "[2] pre-PF -> report_pre", F["s"]),
        ("rg",  "gate: violation?\\lno -> hold Q=0, skip control", F["gate"]),
        ("r3",  "[3] read vm_pu at DER buses (pre-PF)", F["s"]),
        ("r4",  "[4] Q setpoint\\l4B: coordinate() ; 4A: clip q_initial", F["s"]),
        ("r5",  "[5] apply DER dynamics", F["s"]),
        ("r6",  "[6] actuate net.sgen\\lP=p_applied, Q=clamp(q_applied)", F["s"]),
        ("r7",  "[7] post-PF -> report_post", F["s"]),
        ("r8",  "[8] curtailment flag = any_violations", F["s"]),
    ])
    lane(g, "co", "coordinate()  (4B maths)  ·  sensitivity_coordinator.py", F["co"], [
        ("c2", "extract NR Jacobian blocks\\lJ_PP, J_PQ, J_QP, J_QQ (net._ppc)", F["co"]),
        ("c3", "reduced Jacobian (Schur)\\lJ_red = J_QQ - J_QP J_PP^-1 J_PQ", F["co"]),
        ("c4", "voltage sensitivity\\lX = dV/dQ = J_red^-1", F["co"]),
        ("c5", "predict controlled voltage\\lvm_pred = vm + X (q_initial/Sbase)", F["co"]),
        ("c6", "residual violations?\\lbuses outside band +/- 1e-3", F["co"]),
        ("c7", "min-norm correction\\ldQ = lstsq(S_viol, V* - vm_pred)", F["co"]),
        ("c8", "q_adjusted = clip(q_initial + dQ, +/-Qmax)", F["co"]),
    ])
    lane(g, "hlp", "controller / dynamics / detector", F["s"], [
        ("qv", "QVCharacteristic.compute_setpoints\\lpiecewise Q(V)  (dry / fallback)", "#FBEFF4"),
        ("cl", "_clamp_to_net_limits\\l|Q| <= sqrt(Sn^2 - P^2)", "#FBEFF4"),
        ("dyn","DERDynamics.step\\lQ: exact PT1  a=1-exp(-dt/tau), tau=t95/3 (t95 def 10 s)\\lP: ramp dP_max = rate x p_rated x dt (base 0.5%/s)", "#E6F5EF"),
        ("vd", "detect_violations\\lV band + line / trafo loading", "#EEEDFE"),
    ])
    lane(g, "sh", "Solver  &  Hardware (HIL)", F["solv"], [
        ("pp", "pandapower runpp (Newton-Raphson)\\lvoltage_depend_loads=False", F["solv"]),
        ("tx", "exchange_batched\\lformat V:<vm> 4-dec ASCII", F["hw"]),
        ("fw", "Arduino firmware\\lparse V -> compute_q() f32 -> emit Q", F["hw"]),
    ])
    E(g, "A", "B"); E(g, "B", "C"); E(g, "C", "r01")
    E(g, "r01", "r2"); E(g, "r2", "rg"); E(g, "rg", "r3", "violation")
    E(g, "r3", "r4"); E(g, "r4", "r5"); E(g, "r5", "r6")
    E(g, "r6", "r7"); E(g, "r7", "r8"); E(g, "r8", "D")
    E(g, "D", "Ee"); E(g, "Ee", "Ff"); E(g, "Ff", "A", "next t", constraint="false")
    E(g, "rg", "r6", "clean", style="dashed", constraint="false")
    E(g, "r4", "c2", "4B", constraint="false")
    E(g, "c2", "c3"); E(g, "c3", "c4"); E(g, "c4", "c5")
    E(g, "c5", "c6"); E(g, "c6", "c7"); E(g, "c7", "c8")
    E(g, "r2", "pp", "pre", constraint="false")
    E(g, "r7", "pp", "post", constraint="false")
    E(g, "Ee", "pp", "curtail", constraint="false")
    E(g, "pp", "vd", constraint="false")
    D(g, "vd", "r2", "report_pre", constraint="false")
    D(g, "vd", "r7", "report_post", constraint="false")
    D(g, "r3", "qv", "vm (dry)", constraint="false")
    D(g, "qv", "r4", "q_initial", constraint="false")
    D(g, "tx", "r4", "q_initial", constraint="false")
    D(g, "c8", "r4", "q_adjusted", constraint="false")
    D(g, "r5", "dyn", "q_adjusted", constraint="false")
    D(g, "dyn", "r5", "q/p_applied", constraint="false")
    D(g, "r6", "cl", "clamp", constraint="false")
    D(g, "r8", "D", "CoordinatorResult", constraint="false")
    H(g, "r3", "tx", "vm (HIL)", constraint="false")
    H(g, "tx", "fw", "V: TX")
    H(g, "fw", "tx", "Q: RX", constraint="false")
    H(g, "tx", "qv", "err->fallback", constraint="false")
    legend(g, "Ff")
    return g


for name, gb in (("flow_s4_init_new", build_init), ("flow_s4_loop_new", build_loop)):
    g = gb()
    g.format = "svg"; g.render(f"D:/My Files/Personal Projects/HIL-Testbed/Process Plots/New Plots/{name}", cleanup=True)
    g.format = "pdf"; g.render(f"D:/My Files/Personal Projects/HIL-Testbed/Process Plots/New Plots/{name}", cleanup=True)
    print("wrote", name)
