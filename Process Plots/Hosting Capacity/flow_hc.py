#!/usr/bin/env python3
"""
flow_hc.py — hosting_capacity.py. Same template as flow_s4.

  1. flow_hc_baseline  — run_baseline_hc (Case A, no control)
  2. flow_hc_voltvar   — run_hc_with_volt_var (Case B, Q(V) fixed-point)

Verified from hosting_capacity.py function bodies (not the memory summary):
- deepcopy isolation; _RUNPP_BASE={voltage_depend_loads:False, algorithm:'nr'}.
- dist_kv = modal net.bus.vn_kv (HV slack excluded); HC_PARAMS MV 0/0.5/20 (40),
  LV 0/0.01/0.5 (50). end-of-feeder = max topological distance from slack.
- _set_worst_case_snapshot = MIN load x MAX existing PV (1.0 pu), applied once.
  (memory summary's '10%' is stress.py, a different module.)
- Case B: ctrl.configure() BEFORE the sweep -> HC sgens excluded from
  ctrl.sgen_indices (only pre-existing DERs get Q(V)). _qv_converge inner loop:
  MAX_QV_ITERS=10, Q_CONV_TOL=1e-4. HC gain = hc_mw(B) - hc_mw(A).
"""
from graphviz import Digraph

F = {"m": "#E7F0FA", "s": "#EDF5E4", "solv": "#F1EFE9",
     "qv": "#FBEFF4", "gate": "#FCF5D6", "out": "#F3F8EC", "note": "#EAE7F6"}
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


def build_baseline():
    g = base("hosting_capacity : run_baseline_hc (Case A, no control)", ranksep="0.3")
    lane(g, "m", "run_baseline_hc  ·  hosting_capacity.py", F["m"], [
        ("a1", "run_baseline_hc(net, network_id)", F["m"]),
        ("a2", "net = deepcopy; kwargs {vdl=False, nr}", F["m"]),
        ("a3", "dist_kv = _infer_dist_voltage\\l(modal vn_kv; HV slack excluded)", F["m"]),
        ("a4", "params = _hc_params_for(dist_kv)\\lMV 0/0.5/20 (40) ; LV 0/0.01/0.5 (50)", F["m"]),
        ("a5", "eof_bus = _find_endoffeeder_bus\\l(max topological distance from slack)", F["m"]),
        ("a6", "_set_worst_case_snapshot\\lMIN load x MAX existing PV (1.0 pu) [once]", F["m"]),
        ("a7", "while total_mw <= params.max:", F["gate"]),
        ("a8", "_add_pv_at_bus(eof_bus, step_mw)\\ltotal_mw += step (round 6)", F["m"]),
        ("a9", "runpp (except -> treat as violation)", F["m"]),
        ("a10","detect_violations 0.95-1.05 ?", F["gate"]),
        ("a11","violation -> violated_at/binding;\\lremove last sgen; break", F["m"]),
        ("a12","clean -> hc_mw = total_mw (last free);\\lappend sweep_curve; next step", F["m"]),
        ("a13","loop end w/o violation ->\\lhc_limit_reached=True; hc_mw=max", F["m"]),
        ("a14","return (HCResult, net @ hc_mw\\lviolating sgen removed)", F["out"]),
    ])
    lane(g, "s", "Deterministic helpers", F["s"], [
        ("hh", "_infer_dist_voltage / _hc_params_for /\\l_find_endoffeeder_bus (pptop) /\\l_set_worst_case_snapshot / _add_pv_at_bus", F["s"]),
        ("vd", "detect_violations (V band 0.95-1.05)", "#EEEDFE"),
    ])
    lane(g, "solv", "Solver", F["solv"], [
        ("pp", "pandapower runpp\\lvoltage_depend_loads=False, nr", F["solv"]),
    ])
    for a, b in [("a1","a2"),("a2","a3"),("a3","a4"),("a4","a5"),("a5","a6"),("a6","a7")]:
        E(g, a, b)
    E(g, "a7", "a8", "step"); E(g, "a8", "a9"); E(g, "a9", "a10")
    E(g, "a10", "a11", "violation"); E(g, "a10", "a12", "clean")
    E(g, "a12", "a7", "next", constraint="false")
    E(g, "a7", "a13", "max reached"); E(g, "a11", "a14"); E(g, "a13", "a14")
    E(g, "a3", "hh", constraint="false"); E(g, "a5", "hh", constraint="false")
    E(g, "a8", "hh", constraint="false")
    E(g, "a9", "pp", constraint="false"); E(g, "a10", "vd", constraint="false")
    D(g, "vd", "a10", "report", constraint="false")
    legend(g, "a14")
    return g


def build_voltvar():
    g = base("hosting_capacity : run_hc_with_volt_var (Case B, Q(V) fixed-point)",
             ranksep="0.3")
    lane(g, "m", "run_hc_with_volt_var  ·  hosting_capacity.py", F["m"], [
        ("b1", "run_hc_with_volt_var(net, network_id)", F["m"]),
        ("b2", "net=deepcopy; kwargs; dist_kv; params; eof_bus;\\l_set_worst_case_snapshot (same as Case A)", F["m"]),
        ("b3", "ctrl = VoltVarController(interface=None, dry_run=True)\\lctrl.configure()  [BEFORE sweep]\\lsgen_indices = pre-existing DERs ONLY", F["gate"]),
        ("b4", "while total_mw <= params.max:", F["gate"]),
        ("b5", "_add_pv_at_bus(eof_bus, step_mw)\\l(q_mvar=0; NOT in ctrl.sgen_indices)", F["m"]),
        ("b6", "_qv_converge(net, ctrl, kwargs)\\l[Q(V) fixed-point inner loop]", F["m"]),
        ("b7", "detect_violations 0.95-1.05 ?", F["gate"]),
        ("b8", "violation -> record; remove; break\\lclean -> hc_mw=total; append; next", F["m"]),
        ("b9", "return HCResult (case=volt_var,\\lqv_converged, qv_iters_max)", F["out"]),
    ])
    lane(g, "qv", "_qv_converge  (inner fixed-point loop)", F["qv"], [
        ("q1", "for n_iter in 1..MAX_QV_ITERS (10):", F["gate"]),
        ("q2", "runpp -> vm_pu at pre-existing DER buses", F["qv"]),
        ("q3", "q_new = QVCharacteristic.compute_setpoints(vm, p)\\lq_new = ctrl._clamp_to_net_limits(q_new)", F["qv"]),
        ("q4", "write q_new -> net.sgen.q_mvar (DER sgens only)", F["qv"]),
        ("q5", "max|q_new - q_prev| < Q_CONV_TOL (1e-4)?\\lyes -> converged/break ; no -> iterate", F["qv"]),
    ])
    lane(g, "s", "Sub-modules", F["s"], [
        ("ctrl", "VoltVarController (dry_run)\\lsgen_indices, p_installed, _clamp_to_net_limits", "#EDF5E4"),
        ("qc", "QVCharacteristic.compute_setpoints\\lpiecewise VDE Q(V)", "#FBEFF4"),
        ("vd", "detect_violations (V band)", "#EEEDFE"),
    ])
    lane(g, "note", "Verified note", F["note"], [
        ("nc", "HC gain = hc_mw(B) - hc_mw(A).\\lHC sgens excluded from Q(V) (added after configure()).\\l_qv_converge reuses last runpp (no redundant final PF).", F["note"]),
    ])
    for a, b in [("b1","b2"),("b2","b3"),("b3","b4")]:
        E(g, a, b)
    E(g, "b4", "b5", "step"); E(g, "b5", "b6"); E(g, "b6", "b7")
    E(g, "b7", "b8"); E(g, "b8", "b4", "next", constraint="false")
    E(g, "b4", "b9", "max reached")
    # inner loop
    E(g, "b6", "q1", constraint="false")
    E(g, "q1", "q2"); E(g, "q2", "q3"); E(g, "q3", "q4"); E(g, "q4", "q5")
    E(g, "q5", "q2", "iterate", constraint="false")
    E(g, "q3", "qc", constraint="false"); E(g, "q3", "ctrl", "clamp", constraint="false")
    E(g, "q2", "vd", style="invis")
    E(g, "b7", "vd", constraint="false")
    D(g, "ctrl", "b3", "sgen_indices", constraint="false")
    D(g, "q5", "b6", "converged, n_iters", constraint="false")
    g.edge("b9", "nc", style="invis")
    legend(g, "b9")
    return g


for name, gb in (("flow_hc_baseline", build_baseline), ("flow_hc_voltvar", build_voltvar)):
    g = gb()
    g.format = "svg"; g.render(f"/home/claude/{name}", cleanup=True)
    g.format = "pdf"; g.render(f"/home/claude/{name}", cleanup=True)
    print("wrote", name)
