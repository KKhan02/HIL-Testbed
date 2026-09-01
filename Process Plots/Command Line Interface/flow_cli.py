#!/usr/bin/env python3
"""
flow_cli.py — CLI path (wizard.py + executor.py + run_plan.py). Same template.

  1. flow_cli_wizard    — __main__ -> run_wizard (9 steps) -> RunPlan (+ configs)
  2. flow_cli_executor  — executor.execute: 5 phases with typed exit codes
  3. flow_cli_resolve   — build_net_and_profiles: network(4) x dataset(3) dispatch

Verified from function bodies (wizard.py, executor.py, run_plan.py) +
EXECUTOR_IMPLEMENTATION.md:
- run_wizard steps: Study, Network, Network mods, Dataset (skipped for plugin),
  Time window, Parameters, Hardware, Streaming, Controller plugin. Back-nav via
  BackRequested. Returns RunPlan.
- execute(plan)->int, 5 phases: config/channels, net+profiles, config+publish,
  run (register_and_run|run_benchmark), publish. Sanctioned channels:
  set_qv_parameters + rebind violation_detector constants.
- build_net_and_profiles: 4 network source types, 3 dataset source types,
  deliberate order (load -> mods BEFORE profiles -> build -> scale -> window).
- executor 'custom' dataset is the REAL custom (custom_path=data_dir + file_map/
  col_map); distinct from the rejected network-plugin strategy:custom.
"""
from graphviz import Digraph

F = {"m": "#E7F0FA", "w": "#EDF5E4", "rp": "#F1EFE9", "ch": "#FBEFF4",
     "fr": "#EDE7F6", "gate": "#FCF5D6", "out": "#F3F8EC", "note": "#FCEBEA"}
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


def legend(g, anchor, data_lbl="data (config)"):
    with g.subgraph(name="cluster_legend") as c:
        c.attr(label="Legend", style="rounded", color="#8A887F", labeljust="l",
               fontname="Helvetica-bold", fontsize="10", bgcolor="#FFFFFF", margin="6")
        for n in ("le1", "le2", "ld1", "ld2"):
            c.node(n, "", shape="point", width="0.02", color="#FFFFFF")
        E(c, "le1", "le2", "execution")
        D(c, "ld1", "ld2", data_lbl)
        c.edge("le2", "ld1", style="invis")
    g.edge(anchor, "le1", style="invis")


def build_wizard():
    g = base("CLI : __main__ -> run_wizard (9 steps) -> RunPlan", ranksep="0.28")
    lane(g, "mn", "__main__.py", F["m"], [
        ("a1", "run_wizard()", F["m"]),
        ("a2", "print_run_plan(plan) preview", F["m"]),
        ("a3", "optional: save preset JSON (plan.to_dict)", F["m"]),
        ("a4", "sys.exit(executor.execute(plan))  -> Diagram 2", F["m"]),
    ])
    lane(g, "w", "wizard.run_wizard  (index-walked; BackRequested -> prev step)", F["w"], [
        ("s1", "1 Study: _ask_study (+ _ask_hc_stressed if HC)", F["w"]),
        ("s2", "2 Network: _ask_network -> NetworkConfig", F["w"]),
        ("s3", "3 Network mods: der_placements, switches_to_flip", F["w"]),
        ("s4", "4 Dataset: plugin? skip (DatasetConfig('plugin'))\\lelse _ask_dataset", F["gate"]),
        ("s5", "5 Time window: time_period, time_index", F["w"]),
        ("s6", "6 Parameters: _ask_parameters -> ParameterConfig", F["w"]),
        ("s7", "7 Hardware: _ask_hardware -> hardware, port", F["w"]),
        ("s8", "8 Streaming: stream_every_k (default 4)", F["w"]),
        ("s9", "9 Controller plugin: controller_plugin_path", F["w"]),
    ])
    lane(g, "rp", "run_plan.py  (pure data; to_dict / from_dict)", F["rp"], [
        ("rp", "RunPlan\\lstudy, network, dataset, parameters, hardware, port,\\lstream_every_k, controller_plugin_path, hc_stressed,\\ltime_period/index, focus_buses, output_dir, run_id", F["rp"]),
        ("nc", "NetworkConfig\\lsource_type: preset|simbench_code|custom|plugin\\lpreset_name/family, simbench_code, simbench_selections,\\lcustom_path, custom_function_name, plugin_path,\\lder_placements, switches_to_flip", F["rp"]),
        ("dc", "DatasetConfig\\lsource_type: simbench_native|dwd|custom\\ldata_dir, station_id, year, file_map, col_map, custom_path", F["rp"]),
        ("pc", "ParameterConfig\\lv_min/v_max, line/trafo_max_loading, va_diff_max_degree,\\lunbalance_max_percent, der_scaling, load_scaling,\\ltimestep_resolution", F["rp"]),
    ])
    E(g, "a1", "a2"); E(g, "a2", "a3"); E(g, "a3", "a4")
    chain = ["s1","s2","s3","s4","s5","s6","s7","s8","s9"]
    for x, y in zip(chain, chain[1:]):
        E(g, x, y)
    D(g, "s9", "s1", "BackRequested -> prev step", constraint="false")
    D(g, "s9", "rp", "assemble", constraint="false")
    D(g, "s2", "nc", constraint="false"); D(g, "s4", "dc", constraint="false")
    D(g, "s6", "pc", constraint="false")
    D(g, "rp", "a1", "RunPlan", constraint="false")
    legend(g, "a4")
    return g


def build_executor():
    g = base("CLI : executor.execute (5 phases, typed exit codes)", ranksep="0.32")
    lane(g, "m", "executor.execute(plan) -> int", F["m"], [
        ("e0", "out_dir = output_dir/run_id;\\l_configure_cli_logging (Rich + session.log)", F["m"]),
        ("p0", "Phase 0 config + channels:\\lvalidate_plan; apply_qv_overrides; apply_violation_limits;\\lcheck_hardware_port; _confirm_plugin_firmware\\l-> CONFIG / HARDWARE / PLUGIN_ERROR", F["gate"]),
        ("p1", "Phase 1 net+profiles:\\lbuild_net_and_profiles(plan) [Diagram 3]\\l-> NETWORK_LOAD / DATASET_ERROR", F["m"]),
        ("p2", "Phase 2 config+publish:\\lbuild_benchmark_config(plan, factory, publish_fn);\\lPublishHandle(update_every_k=stream_every_k);\\lsave run_plan.json", F["m"]),
        ("p3", "Phase 3 run:\\lcontroller plugin? register_and_run : run_benchmark\\l-> SIMULATION_ERROR (broad except)", F["m"]),
        ("p4", "Phase 4 publish:\\lpublish_result (+ 2nd pass HC-stressed);\\lprint_summary_table -> PUBLISH_ERROR", F["m"]),
        ("ret","return ExitCode (0 = success)", F["out"]),
    ])
    lane(g, "ch", "Sanctioned runtime channels (only framework mutation)", F["ch"], [
        ("c1", "volt_var_controller.set_qv_parameters\\lper-run Q(V): dry curve + coordinator + Arduino CFG", F["ch"]),
        ("c2", "rebind violation_detector constants\\lV_MIN/V_MAX, line/trafo loading, angle, unbalance", F["ch"]),
    ])
    lane(g, "fr", "Framework (unmodified; also usable via run_benchmark_script)", F["fr"], [
        ("f1", "benchmark_runner.run_benchmark /\\lplugin_runner.register_and_run", F["fr"]),
        ("f2", "publisher.publish_result + PublishHandle", F["fr"]),
    ])
    E(g, "e0", "p0"); E(g, "p0", "p1"); E(g, "p1", "p2"); E(g, "p2", "p3")
    E(g, "p3", "p4"); E(g, "p4", "ret")
    E(g, "p0", "c1", "Q(V)", constraint="false")
    E(g, "p0", "c2", "limits", constraint="false")
    E(g, "p3", "f1", constraint="false")
    E(g, "p4", "f2", constraint="false")
    legend(g, "ret")
    return g


def build_resolve():
    g = base("CLI : build_net_and_profiles (network x dataset resolution)",
             ranksep="0.32")
    lane(g, "m", "build_net_and_profiles(plan) -> (net, profiles, network_id, factory)", F["m"], [
        ("r1", "[1] network dispatch on net_cfg.source_type (4 types)", F["gate"]),
        ("r2", "[2] _apply_network_modifications\\l(DER inject, switch flip) BEFORE profiles", F["m"]),
        ("r3", "[3] dataset dispatch on ds_cfg.source_type (3 types)\\l(non-plugin; plugin builds profiles at load)", F["gate"]),
        ("r4", "profile_factory: same builder_kwargs + '_hc_stressed'", F["m"]),
        ("r5", "[4/5] _apply_scaling (net tables + profile cols together);\\l_apply_time_window; _check_timestep_resolution;\\l_validate_focus_buses", F["m"]),
        ("r6", "return (net, profiles, network_id, profile_factory)", F["out"]),
    ])
    lane(g, "ns", "Network sources (4)", F["w"], [
        ("n1", "preset -> _preset_loaders()[name]()\\l(+ _PRESET_SIMBENCH_CODES for sb_code)", F["w"]),
        ("n2", "simbench_code -> sb.get_simbench_net(code)", F["w"]),
        ("n3", "custom -> importlib spec_from_file_location\\l(try/except cascade; duck-type bus/line/load/sgen)", F["w"]),
        ("n4", "plugin -> network_plugin.load_network_from_yaml\\l-> (net, profiles); validate warnings; make_profile_factory", F["w"]),
    ])
    lane(g, "ds", "Dataset sources (3, non-plugin)", F["rp"], [
        ("d1", "simbench_native -> build_annual_profiles(simbench_code)\\l[requires sb_code]", F["rp"]),
        ("d2", "dwd -> _validate_dwd_dir;\\lbuild_annual_profiles(data_dir or 'data/dwd')", F["rp"]),
        ("d3", "custom -> build_annual_profiles(data_dir=custom_path,\\lfile_map, col_map)  <- the REAL 'custom'", F["rp"]),
    ])
    lane(g, "note", "Verified note", F["note"], [
        ("nc", "executor 'custom' dataset = custom_path (data_dir) + file_map/col_map.\\lDistinct from network-plugin strategy:custom (rejected).\\lera5 CSVs enter via this 'custom' path.", F["note"]),
    ])
    E(g, "r1", "r2"); E(g, "r2", "r3"); E(g, "r3", "r4"); E(g, "r4", "r5"); E(g, "r5", "r6")
    E(g, "r1", "n1"); E(g, "r1", "n2", style="dashed"); E(g, "r1", "n3", style="dashed")
    E(g, "r1", "n4", style="dashed")
    E(g, "r3", "d1"); E(g, "r3", "d2", style="dashed"); E(g, "r3", "d3", style="dashed")
    D(g, "n4", "r3", "profiles at load (skip dataset)", constraint="false")
    D(g, "d2", "r4", "profiles", constraint="false")
    g.edge("r6", "nc", style="invis")
    legend(g, "r6")
    return g


for name, gb in (("flow_cli_wizard", build_wizard),
                 ("flow_cli_executor", build_executor),
                 ("flow_cli_resolve", build_resolve)):
    g = gb()
    g.format = "svg"; g.render(f"/home/claude/{name}", cleanup=True)
    g.format = "pdf"; g.render(f"/home/claude/{name}", cleanup=True)
    print("wrote", name)
