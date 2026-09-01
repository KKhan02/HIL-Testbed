#!/usr/bin/env python3
"""
flow_orch.py — Orchestration core. Same template as flow_s4.

  1. flow_orch_script  — run_benchmark_script.py: config, network/profile
                         resolution, dispatch, publish
  2. flow_orch_runner  — benchmark_runner.run_benchmark: scenario loop with
                         per-scenario isolation + hosting-capacity dispatch

Verified against run_benchmark_script.py, benchmark_runner.py, plugin_runner.py.
Key facts: each scenario runs on copy.deepcopy(net) INSIDE the try (a copy
failure is a failed row, not an abort); _build_kwargs injects kwargs by
ScenarioSpec capability flags; a runner exception is caught into errors[n]
(NaN row) and does not abort; HC runs after the loop; run_hc_scenarios does a
recursive HC-stressed re-benchmark (run_hc/run_hc_scenarios forced False).
"""
from graphviz import Digraph

F = {"m": "#E7F0FA", "s": "#EDF5E4", "orch": "#EDE7F6",
     "pub": "#EAE7F6", "gate": "#FCF5D6", "out": "#F1EFE9"}
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


def build_script():
    g = base("Direct-script entry : run_benchmark_script.py", ranksep="0.34")
    lane(g, "m", "Script  ·  run_benchmark_script.py", F["m"], [
        ("e1", "parse args\\l--network / --controller / -y /\\l--q-ratio / --v-min/max / --line/trafo-max", F["m"]),
        ("e2", "apply CLI overrides\\lQ(V) params ; voltage / thermal limits", F["m"]),
        ("e3", "resolve network\\l--network: plugin YAML  else: hardcoded SimBench", F["gate"]),
        ("e4", "net_os = deepcopy(net); oversize_inverters(x1.3)", F["m"]),
        ("e5", "build profiles\\lplugin path: already built ; else build_annual_profiles", F["m"]),
        ("e6", "optional net mods\\lswitch toggle, load scale, DER inject, slice", F["m"]),
        ("e7", "BenchmarkConfig(scenarios, dry_run, HC flags,\\lprofile_factory, v_min/max) + PublishHandle", F["m"]),
        ("e8", "dispatch\\l--controller ? register_and_run : run_benchmark", F["gate"]),
        ("e9", "publish_result -> static JSON\\l(topology, profiles, hc, scenarios)", F["m"]),
    ])
    lane(g, "s", "Model-building sub-modules", F["s"], [
        ("s1", "network_plugin.load_network_from_yaml\\l/ simbench.get_simbench_net", F["s"]),
        ("s2", "profile_builder.build_annual_profiles\\l/ make_profile_factory", F["s"]),
        ("s3", "violation_detector / volt_var_controller\\loverride module constants", F["s"]),
    ])
    lane(g, "orch", "Orchestration", F["orch"], [
        ("o1", "benchmark_runner.run_benchmark\\l(Diagram 2)", F["orch"]),
        ("o2", "plugin_runner.register_and_run\\l(if --controller)", F["orch"]),
    ])
    lane(g, "pub", "Publisher", F["pub"], [
        ("p1", "publisher.publish_result\\lstatic JSON artefacts", F["pub"]),
    ])
    E(g, "e1", "e2"); E(g, "e2", "e3"); E(g, "e3", "e4"); E(g, "e4", "e5")
    E(g, "e5", "e6"); E(g, "e6", "e7"); E(g, "e7", "e8"); E(g, "e8", "e9")
    E(g, "e3", "s1", constraint="false")
    E(g, "e5", "s2", constraint="false")
    E(g, "e2", "s3", constraint="false")
    E(g, "e8", "o1", "built-ins", constraint="false")
    E(g, "e8", "o2", "plugin", style="dashed", constraint="false")
    E(g, "e9", "p1", constraint="false")
    D(g, "s1", "e4", "net", constraint="false")
    D(g, "s2", "e5", "profiles", constraint="false")
    legend(g, "e9")
    return g


def build_runner():
    g = base("Orchestrator : benchmark_runner.run_benchmark", ranksep="0.3")
    lane(g, "m", "run_benchmark  ·  benchmark_runner.py", F["m"], [
        ("r1", "run_benchmark(net, profiles, network_id, config)", F["m"]),
        ("r2", "validate config\\lv_min < v_max ; hc_scenarios needs factory", F["m"]),
        ("r3", "inspect net: is_lv? has PV/wind DERs?", F["m"]),
        ("r4", "for n in sorted(config.scenarios):", F["gate"]),
        ("r5", "LV + not supports_lv + no DERs?\\l-> skip (errors[n]='skipped')", F["gate"]),
        ("r6", "try: net_copy = deepcopy(net)\\lkwargs = _build_kwargs(spec, config)", F["m"]),
        ("r7", "result = spec.runner(net_copy, profiles, **kwargs)", F["m"]),
        ("r8", "except: errors[n]=traceback; results[n]=None\\l(NaN row) -- do NOT abort", F["gate"]),
        ("r9", "hosting capacity (run_hc)\\lrun_baseline_hc + run_hc_with_volt_var", F["m"]),
        ("r10","HC-stressed re-benchmark (run_hc_scenarios)\\lprofile_factory(net_hc) -> run_benchmark (recursive)", F["m"]),
        ("r11","build comparison_df; write CSV; print summary", F["m"]),
        ("r12","return BenchmarkResult", F["m"]),
    ])
    lane(g, "s", "Registry & dispatch", F["orch"], [
        ("b1", "SCENARIO_REGISTRY[n] -> ScenarioSpec\\lrunner + capability flags", F["orch"]),
        ("b2", "_build_kwargs\\lbase v_min/v_max; +dry_run/port/coordination (HW);\\l+verbose_opf (OPF); publish_fn (all)", F["orch"]),
        ("b3", "scenario runner\\lrun_scenario_1..5 / plugin  (their own diagrams)", F["s"]),
        ("b4", "hosting_capacity\\lrun_baseline_hc / run_hc_with_volt_var", F["s"]),
    ])
    lane(g, "out", "Result assembly", F["out"], [
        ("c1", "_build_comparison_df / _write_csv\\lNaN for failed rows (vs 0 = clean)", F["out"]),
        ("c2", "BenchmarkResult\\lresults, errors, comparison_df,\\lhc_results, hc_benchmark, net_hc", F["out"]),
    ])
    E(g, "r1", "r2"); E(g, "r2", "r3"); E(g, "r3", "r4")
    E(g, "r4", "r5"); E(g, "r5", "r6", "run"); E(g, "r5", "r4", "skip", style="dashed", constraint="false")
    E(g, "r6", "r7"); E(g, "r7", "r8", "on error", style="dashed"); E(g, "r7", "r4", "next n", constraint="false")
    E(g, "r8", "r4", "next n", style="dashed", constraint="false")
    E(g, "r4", "r9", "loop done")
    E(g, "r9", "r10"); E(g, "r10", "r11"); E(g, "r11", "r12")
    E(g, "r4", "b1", constraint="false")
    E(g, "r6", "b2", constraint="false")
    E(g, "r7", "b3", constraint="false")
    E(g, "r9", "b4", constraint="false")
    E(g, "r11", "c1", constraint="false")
    E(g, "r12", "c2", constraint="false")
    D(g, "b1", "r5", "supports_lv", constraint="false")
    D(g, "b2", "r7", "kwargs", constraint="false")
    D(g, "b3", "r7", "ScenarioResult", constraint="false")
    legend(g, "r12")
    return g


for name, gb in (("flow_orch_script", build_script), ("flow_orch_runner", build_runner)):
    g = gb()
    g.format = "svg"; g.render(f"/home/claude/{name}", cleanup=True)
    g.format = "pdf"; g.render(f"/home/claude/{name}", cleanup=True)
    print("wrote", name)
