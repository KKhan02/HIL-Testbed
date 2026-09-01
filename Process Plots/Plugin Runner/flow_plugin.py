#!/usr/bin/env python3
"""
flow_plugin.py — Custom controller plugin path. Same template as flow_s4.

  1. flow_plugin_reg   — plugin_runner.register_and_run: load / route / register
                         / run / cleanup  (the init+teardown analogue)
  2. flow_plugin_loop  — custom_controller.run_custom_controller_scenario:
                         the [A]-[F] per-timestep loop

Verified against plugin_runner.py, custom_controller.py, volt_var_controller.py.
Key facts: controller_fn is the pluggable Q(V) computation. Routing is decided
once at registration: hardware:true AND not dry_run AND port -> HardwareControllerFn
(serial to the researcher's own firmware) else the Python mirror. The loop is
the S4 loop MINUS coordinator, DER dynamics and the curtailment stage; Q is
applied directly (optional net-limit clamp). Clean-timestep gate is opt-in.
"""
from graphviz import Digraph

F = {"m": "#E7F0FA", "s": "#EDF5E4", "solv": "#F1EFE9",
     "hw": "#FBE9DF", "pub": "#EAE7F6", "gate": "#FCF5D6", "orch": "#EDE7F6"}
INK = "#1B1B1A"


def E(g, a, b, t="", **k):
    g.edge(a, b, xlabel=t, color=INK, penwidth="2.0", arrowsize="0.75", **k)

def D(g, a, b, t="", **k):
    g.edge(a, b, xlabel=t, color="#1F6FB2", style="dashed", fontcolor="#1F6FB2",
           penwidth="1.25", arrowsize="0.65", **k)

def H(g, a, b, t="", **k):
    g.edge(a, b, xlabel=t, color="#D8641E", style="dotted", fontcolor="#B94F0C",
           penwidth="1.7", arrowsize="0.65", **k)


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


def legend(g, anchor, hw=True):
    with g.subgraph(name="cluster_legend") as c:
        c.attr(label="Legend", style="rounded", color="#8A887F", labeljust="l",
               fontname="Helvetica-bold", fontsize="10", bgcolor="#FFFFFF", margin="6")
        ns = ["le1", "le2", "ld1", "ld2"] + (["lh1", "lh2"] if hw else [])
        for n in ns:
            c.node(n, "", shape="point", width="0.02", color="#FFFFFF")
        E(c, "le1", "le2", "execution")
        D(c, "ld1", "ld2", "data (array/df)")
        c.edge("le2", "ld1", style="invis")
        if hw:
            H(c, "lh1", "lh2", "serial TX/RX")
            c.edge("ld2", "lh1", style="invis")
    g.edge(anchor, "le1", style="invis")


def build_reg():
    g = base("Custom Controller Plugin : Registration, Routing & Cleanup "
             "(plugin_runner.py)", ranksep="0.34")
    lane(g, "m", "register_and_run  ·  plugin_runner.py", F["m"], [
        ("p1", "register_and_run(yaml, net, profiles,\\lnetwork_id, config, port)", F["m"]),
        ("p2", "load_plugin(yaml) -> cfg", F["m"]),
        ("p3", "_import_controller_fn(module_path, function) -> fn", F["m"]),
        ("p4", "bind kwargs\\lcontroller_fn = partial(fn, **cfg.kwargs)", F["m"]),
        ("p5", "hardware routing\\lhardware AND not dry_run AND port ?", F["gate"]),
        ("p6", "allocate num >= 10\\lScenarioSpec(runner=_plugin_runner)", F["m"]),
        ("p7", "extend config.scenarios (copy)\\lSCENARIO_REGISTRY[num] = spec", F["m"]),
        ("p8", "run_benchmark(...)  (plugin runs\\lalongside built-ins -> Diagram 2)", F["m"]),
        ("t1", "finally: registry.pop(num);\\lhw_fn.close() sends END (if HW)", F["gate"]),
        ("t2", "return ScenarioResult\\l(or + BenchmarkResult)", F["m"]),
    ])
    lane(g, "s", "Plugin loading", F["s"], [
        ("l1", "load_plugin\\lparse + validate YAML (paths YAML-relative;\\lfirmware .ino existence-checked)", F["s"]),
        ("l2", "_import_controller_fn\\limportlib load fn from FILE, check callable", F["s"]),
    ])
    lane(g, "orch", "Orchestration", F["orch"], [
        ("o1", "benchmark_runner.run_benchmark\\ldispatch num via SCENARIO_REGISTRY", F["orch"]),
    ])
    lane(g, "hw", "Controller routing target", F["hw"], [
        ("sw", "SW mirror: controller_fn = plugin fn\\l(Python, dry-run)", "#EDF5E4"),
        ("hf", "HW: HardwareControllerFn(port)\\lserial-backed controller_fn", F["hw"]),
    ])
    E(g, "p1", "p2"); E(g, "p2", "p3"); E(g, "p3", "p4"); E(g, "p4", "p5")
    E(g, "p5", "p6"); E(g, "p6", "p7"); E(g, "p7", "p8"); E(g, "p8", "t1"); E(g, "t1", "t2")
    E(g, "p2", "l1", constraint="false")
    E(g, "p3", "l2", constraint="false")
    E(g, "p5", "sw", "else (SW)", style="dashed", constraint="false")
    H(g, "p5", "hf", "HW", constraint="false")
    E(g, "p8", "o1", constraint="false")
    D(g, "l1", "p2", "cfg", constraint="false")
    legend(g, "t2", hw=True)
    return g


def build_loop():
    g = base("Custom Controller Plugin : Per-Timestep Loop "
             "(custom_controller.py)", ranksep="0.32")
    lane(g, "m", "Master  ·  run_custom_controller_scenario", F["m"], [
        ("A", "[A] p_target = ap.der_p[t]\\l(ctrl.sgen_indices order)", F["m"]),
        ("B", "[B] write net.load P/Q", F["m"]),
        ("C", "[0-1] net.sgen P=p_target, Q=0", F["m"]),
        ("D", "[2] pre-PF -> report_pre", F["m"]),
        ("G", "gate (opt-in): clean ->\\lhold Q=0, record, skip post-PF", F["gate"]),
        ("Ee", "[3] controller_fn(vm_pu, p_installed) -> q\\loptional _clamp_to_net_limits", F["m"]),
        ("Ff", "[4] net.sgen.q_mvar = q", F["m"]),
        ("Hh", "[5] post-PF -> report_post", F["m"]),
        ("Ii", "[F] make_record_from_report + fields;\\lpublish; append; next t", F["m"]),
    ])
    lane(g, "ctl", "Pluggable controller  (SW mirror OR HW firmware)", F["hw"], [
        ("sw", "controller_fn (plugin Python fn)\\lresearcher Q(V) algorithm", "#EDF5E4"),
        ("hw", "HardwareControllerFn\\lexchange_batched -> Arduino", F["hw"]),
        ("ard","Arduino firmware (researcher's own .ino)\\lV: -> compute Q -> Q:", F["hw"]),
    ])
    lane(g, "s", "Python sub-modules", F["s"], [
        ("m1", "VoltVarController (dry-run)\\lDER indices/buses/p_installed;\\l_clamp_to_net_limits", "#FBEFF4"),
        ("m2", "detect_violations\\lV band + line / trafo loading", "#EEEDFE"),
        ("m3", "make_record_from_report\\l+ losses / grid_import / t_total_ms", F["s"]),
    ])
    lane(g, "solv", "Solver", F["solv"], [
        ("pp", "pandapower runpp (Newton-Raphson)\\lvoltage_depend_loads=False", F["solv"]),
    ])
    lane(g, "pub", "Publisher", F["pub"], [
        ("pb", "on_timestep -> live JSONL", F["pub"]),
    ])
    E(g, "A", "B"); E(g, "B", "C"); E(g, "C", "D"); E(g, "D", "G")
    E(g, "G", "Ee", "violation"); E(g, "G", "Ii", "clean (gate)", style="dashed", constraint="false")
    E(g, "Ee", "Ff"); E(g, "Ff", "Hh"); E(g, "Hh", "Ii")
    E(g, "Ii", "A", "next t", constraint="false")
    # controller routing
    D(g, "Ee", "sw", "vm, p (SW)", constraint="false")
    H(g, "Ee", "hw", "vm, p (HW)", constraint="false")
    H(g, "hw", "ard", "V: TX")
    H(g, "ard", "hw", "Q: RX", constraint="false")
    D(g, "sw", "Ee", "q", constraint="false")
    D(g, "hw", "Ee", "q", constraint="false")
    # solver + submodules
    E(g, "D", "pp", "pre", constraint="false")
    E(g, "Hh", "pp", "post", constraint="false")
    E(g, "pp", "m2", constraint="false")
    E(g, "Ee", "m1", "clamp", constraint="false")
    E(g, "Ii", "m3", constraint="false")
    E(g, "Ii", "pb", constraint="false")
    D(g, "m2", "D", "report_pre", constraint="false")
    D(g, "m2", "Hh", "report_post", constraint="false")
    D(g, "m1", "A", "sgen_indices, p_installed", constraint="false")
    legend(g, "Ii", hw=True)
    return g


for name, gb in (("flow_plugin_reg", build_reg), ("flow_plugin_loop", build_loop)):
    g = gb()
    g.format = "svg"; g.render(f"/home/claude/{name}", cleanup=True)
    g.format = "pdf"; g.render(f"/home/claude/{name}", cleanup=True)
    print("wrote", name)
