# HIL Testbed: Controller-Hardware-in-the-Loop Benchmarking of Volt-Var Control in Distribution Grids

This repository contains an open, reproducible **Controller-Hardware-in-the-Loop (CHIL)** testbed for benchmarking distribution-grid voltage-control strategies — with a focus on decentralised **Volt-Var (Q(V)) control** of inverter-based distributed energy resources (DERs).

A power-flow simulation (`pandapower`) runs a full year of a **benchmark distribution-grid model** while the **local Q(V) control law** is executed either **in Python (software / dry-run)** or **on a real Arduino microcontroller (hardware-in-the-loop)**. Running the same Q(V) law on real silicon, over a real serial link, exposes effects a pure software study hides — most importantly, the impact of **fixed-precision serial quantisation** on closed-loop control behaviour.

The work was developed in the context of an M.Sc. project in the **Renewable Energy programme at the Carl von Ossietzky Universität Oldenburg**.

The workflow combines:

- **pandapower** — steady-state AC power flow, OLTC, SVC, and AC optimal power flow (OPF)
- **SimBench / CIGRE / Kerber / Dickert** benchmark networks
- **A custom Volt-Var controller** (Python) mirrored by **matching Arduino firmware** (C++)
- **DWD open weather data + pvlib + BDEW** load profiles, alongside SimBench-native and synthetic profiles
- A **Streamlit dashboard**, a **CLI wizard**, and a **plugin system** for user-supplied controllers and networks

---

## 🎯 Project Objective

The central question is:

> How faithfully does a decentralised Volt-Var controller running on **real, resource-constrained hardware** reproduce its idealised software behaviour — and how do the standard grid-support strategies (OLTC, SVC, Volt-Var, OPF) compare on the same feeder model, same year, same limits?

The testbed answers two connected questions:

**Control-performance perspective** — for each strategy, how well are voltages kept inside the planning band, how much reactive power and active-power curtailment does it cost, and how does a rule-based local scheme compare to a coordinated scheme and to an AC-OPF reference?

**Hardware-fidelity perspective** — when the identical Q(V) law runs on an Arduino instead of in Python, what diverges, and why? The mechanism of interest is the **4-decimal ASCII + `float32` quantisation** on the serial channel, which perturbs the reactive setpoints just enough to change how often the (host-side) coordination layer engages.

```
   Benchmark network (SimBench / CIGRE / Kerber / Dickert)
                        │
                        ▼
        Annual load / PV / wind profiles
        (SimBench-native · DWD+pvlib+BDEW · flat · custom)
                        │
                        ▼
   ┌─────────────────────────────────────────────┐
   │  Simulation host  (Laptop  OR  Raspberry Pi) │
   │  pandapower AC power flow, per timestep       │
   │  coordination · dynamics · curtailment        │  ← all host-side
   │  → bus voltages at DER buses                  │
   └───────────────┬───────────────────────────────┘
                   │  "V:…"   (serial, 4-decimal ASCII)
                   ▼
   ┌─────────────────────────────────────────────┐
   │  Local Q(V) law  (Tier 1 only)                │
   │   • Python  (dry-run / software)              │
   │   • Arduino firmware  (hardware-in-the-loop)  │
   └───────────────┬───────────────────────────────┘
                   │  "Q:…"   (reactive setpoints)
                   ▼
   host applies Q, coordinates, (optionally) curtails P, re-solves
                   │
                   ▼
   TimestepRecord ── publisher (JSON + live/checkpoint JSONL)
                   │
        ┌──────────┼───────────────┐
        ▼          ▼               ▼
   report plots  Streamlit    student notebooks
                 dashboard     (dry-run vs HIL)
```

---

## 🚀 Main Features

✅ **Six registered benchmark scenarios across five control families**, using a common result model and consistent planning limits
✅ **Controller-hardware-in-the-loop** over serial — the local Q(V) law runs in Python *or* on Arduino firmware, transparently swappable
✅ **Dry-run vs HIL comparison** isolating the effect of serial quantisation on closed-loop control
✅ **Three-tier Volt-Var hierarchy**: local Q(V) → sensitivity-based coordination → active-power curtailment
✅ **DER response model** — first-order (PT1) reactive response + active-power ramp limiting
✅ **AC-OPF reference** (`runopp`, cyipopt) using the *same* reactive envelope as the rule-based schemes
✅ **Hosting-capacity analysis** — baseline vs Volt-Var, with a recursive re-benchmark on the HC-stressed network
✅ **Crash-resume checkpointing** — manual-loop scenarios resume from the last completed timestep; finished scenarios are skipped on restart (RPi-safe, SD-card-friendly)
✅ **Live streaming** to a Streamlit dashboard as the simulation runs
✅ **Plugin system** — bring your own controller (Python function *or* Arduino firmware) or your own network (JSON / pickle / loader function)
✅ **Interactive CLI wizard** + an editable direct-run script for fast iteration
✅ **Student templates** — Jupyter notebooks comparing Laptop/RPi × dry-run/Arduino runs
✅ **Reproducible outputs** — machine-readable JSON, comparison CSVs, and publication-quality figures

---

## ⚡ Quick Start

```bash
# 1. Environment
conda create -n hil python=3.12
conda activate hil
pip install -r requirements.txt

# 2. Run a benchmark interactively (dry-run needs no hardware)
python -m Command_Line_Interface

# 3. Explore the results
streamlit run Streamlit/Home.py
```

The wizard walks you through network, profiles, scenarios, parameters, and (optional) hardware, then runs and publishes the benchmark. Run outputs land under `runs/<run_id>/` (or `outputs/…` for the direct-run script). See [**How to Run**](#🖥️-how-to-run-detailed) for the direct script, plugins, and hardware setup.
Two short demo runs are committed with the repository (runs/demo_day_1-MV-rural--2-sw and runs/demo_week_1-MV-rural--2-sw), so the dashboard has something to open before you have run anything yourself. Point its project root at the repository root and it will discover them.

---

## 🔬 Benchmark Scenarios

Each scenario runs the same annual timeseries on an isolated copy of the network and produces a directly comparable `ScenarioResult`. The runtime registry numbers them **1–6**:

| # | Scenario | `scenario_id` | Control family | Notes |
|---|----------|---------------|----------------|-------|
| 1 | **Baseline** | `baseline` | none | No control — the reference case (declarative `run_timeseries`) |
| 2 | **OLTC** | `oltc` | transformer tap | On-load tap changer reacting to measured voltage |
| 3 | **SVC** | `svc` | central reactive | Static var compensator with a droop characteristic |
| 4 | **Volt-Var (local)** | `volt_var_local` | distributed reactive | Local Q(V) curve per inverter (the "4A" mode) |
| 5 | **Volt-Var (coordinated)** | `volt_var_coord` | reactive + coordination | Q(V) plus sensitivity-based coordination (the "4B" mode) |
| 6 | **AC-OPF reference** | `opf` | optimal reactive + curtailment | AC optimal power flow — an optimisation reference (CIGRE MV) |

A "scenario comparison" study typically runs **1–5**; the **OPF reference (6)** is run as its own study on **CIGRE MV** (SimBench MV ill-conditions the PYPOWER OPF backend and is excluded from that scenario). On **DER-free LV networks**, scenarios that require controllable DERs or MV-oriented control (SVC, both Volt-Var modes, OPF) are **skipped automatically** — Baseline and OLTC remain available.

---

## 🧠 The Volt-Var Control Hierarchy

The core scheme is **reactive-before-active**: use free reactive power first, curtail real power only as a last resort.

```
Measured voltage at DER buses
        │
   Tier 1 ── Local Q(V)          each inverter follows its own Q(V) curve
        │                        (this is the tier that runs on the Arduino
        │                         in HIL mode; everything below is host-side)
        │
   Tier 2 ── Coordination        if violations remain, a sensitivity model
        │                        (∂V/∂Q, least-squares) allocates extra Q to
        │                        the worst buses  (Scenario 5)
        │
   Tier 3 ── Active curtailment  if voltage still cannot be held, reduce P
                                 (Scenario 4/5; linear back-off)
```

**What runs where in HIL mode.** Only the **Tier-1 local Q(V) law** is computed on the Arduino (the firmware's `compute_q` mirrors the Python `QVCharacteristic`). The **power-flow solution, the sensitivity coordinator, the DER response model, and active-power curtailment all remain on the simulation host.** This is exactly what makes the dry-run-vs-HIL comparison meaningful: the *only* difference between the two paths is where — and at what numeric precision — the Q(V) law is evaluated.

### The Q(V) characteristic

A four-breakpoint piecewise-linear curve, continuous at every breakpoint, clamped to the inverter's reactive envelope. The default parameters are project planning values, selected with reference to the relevant grid-code / power-quality standards (VDE-AR-N 4110, EN 50160) rather than lifted verbatim from a specific clause:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `Q_RATIO` | **0.25** | max reactive as a fraction of inverter rating |
| `U1` | **0.96 pu** | below this → full reactive **injection** |
| `U2` | **0.99 pu** | lower deadband edge |
| `U3` | **1.01 pu** | upper deadband edge |
| `U4` | **1.04 pu** | above this → full reactive **absorption** |
| `V_MIN` / `V_MAX` | **0.95 / 1.05 pu** | project planning voltage band |

The per-DER reactive limit is `q_max = min(Q_RATIO · S_rated, √(S² − P²))` — the smaller of the reactive allowance and the inverter apparent-power circle. **Worked example:** a 0.5 MW DER → `q_max ≈ 0.125 MVAr`; a 1.5 MW DER → `q_max ≈ 0.375 MVAr`. All of these are per-run configurable via the CLI.

### DER response model

The framework models a **first-order (PT1) reactive-power response** and an **active-power ramp limiter**. At coarse annual-study resolutions (e.g. 15-minute / 900 s steps) the default response constants are effectively **non-binding**, so the dynamics are essentially invisible in the standard annual runs; they become significant at the finer control-loop resolutions the module is designed for. The response state is nonetheless tracked correctly and re-seeded on a crash-resume, so resuming a run never introduces a discontinuity.

---

## 🔌 The Hardware-in-the-Loop Loop

Each timestep, the simulation host and the controller exchange a compact serial message:

- **`V:`** — bus voltages at the DER buses, sent to the controller (4-decimal ASCII)
- **`Q:`** — reactive setpoints, returned by the controller
- **`CFG` handshake** — installed capacities and Q(V) parameters are pushed once at scenario start, so the firmware's Q(V) curve matches the Python one exactly

Two hosts and two control paths give four comparable configurations:

|                | Q(V) in Python (dry-run) | Q(V) on Arduino (HIL) |
|----------------|--------------------------|------------------------|
| **Laptop host** | fully in-software reference | laptop ↔ Arduino serial |
| **Raspberry Pi host** | RPi in-software | RPi ↔ Arduino serial |

**The mechanism under study.** The Arduino path introduces fixed-precision quantisation (4-decimal ASCII on the wire, `float32` on-device). These small perturbations are enough to shift *which* timesteps cross the coordination trigger, so the *coordination rate* — and the metrics downstream of it — can diverge between the dry-run and HIL paths. The quantitative characterisation of this effect is reported in the accompanying paper.

---

## 🌐 Networks & Profiles

**Benchmark networks**

| Network | Level | Role |
|---------|-------|------|
| SimBench `1-MV-rural--2-sw` | MV rural | primary MV benchmark |
| CIGRE MV (`pv_wind`) | MV | OPF-capable benchmark |
| Kerber / Dickert feeders | LV | LV distribution feeders |
| Synthetic LV | LV | LV screening / stress cases |

**Profile strategies**

| Strategy | Source |
|----------|--------|
| `simbench_native` | SimBench's embedded absolute profiles (portable via JSON export) |
| `dwd_pvlib` | DWD station **691 (Bremen)** irradiance + wind + temperature → **pvlib** PV + **BDEW** load |
| `flat` | constant rated capacity (worst-case simultaneity, no weather) |
| `custom` | user-supplied CSVs (e.g. **ERA5** reanalysis) via a file/column map |

**Resolution:** SimBench-native runs use **15-minute** profiles (35,136 steps for the 2016 leap year). DWD-based profiles are built from **10-minute** open-data (5-minute custom-station data is also supported), and the CLI permits other supported resolutions. Night-time PV is zeroed; extreme days (max load, min load, max PV, max wind) are auto-detected for focused plots.

---

## 📊 Metrics

`ScenarioResult` aggregates each run into directly comparable KPIs:

- **VDI** — voltage deviation index across all buses and timesteps
- **Violation duration** — hours per year outside the planning band
- **Coordination rate** — fraction of timesteps the coordination layer engaged (the HIL-sensitive metric)
- **Curtailed energy** — MWh of active power shed; zero in scenarios without active-power curtailment, potentially non-zero when Tier 3 activates in Volt-Var, and optimised by the AC-OPF reference
- **Voltage / loading extremes** — min/max bus voltage, max line & transformer loading
- **Energy balance** — losses, grid import, DER generation, served load
- **HIL timing** — per-timestep serial latency and total step time

A built-in **coverage-integrity check** rejects any gapped or duplicated timestep set, so a resumed run can never silently produce a corrupted annual metric.

---

## 🔋 Hosting-Capacity Analysis

Beyond the fixed-network benchmark, the testbed estimates *how much more DER the feeder can host* with and without Volt-Var:

- **Baseline HC** vs **Volt-Var HC**, reported with the **gain** (extra MW unlocked by reactive control)
- An optional **recursive re-benchmark** on the HC-stressed network: the feeder is loaded to its baseline hosting-capacity limit, fresh profiles are built for the added DERs (using the same profile strategy), and the scenarios are re-run to see how the control schemes behave at the edge.

---

## 💾 Crash-Resume Checkpointing

Annual runs are long and, on a Raspberry Pi, vulnerable to a dropped connection or power blip. The publisher writes **two per-scenario streams**:

- `live/<id>.jsonl` — compact dashboard frames, truncated fresh each run
- `checkpoint/<id>.jsonl` — full per-timestep records, **never truncated**, written to a **single held-open file handle** (the fix for an SD-card `mmc` stall caused by re-opening the file 100k+ times)

On restart, recovery operates at two levels: whole scenarios already completed are **skipped**, while the **manual-loop scenarios resume from their last checkpointed timestep** and re-seed stateful actuators where required (OLTC tap position, DER response state). **Scenario 1 (Baseline) is an exception** — its declarative `run_timeseries()` execution cannot currently resume mid-sweep and restarts if interrupted before completion. Run inside `tmux` and a dropped SSH session costs nothing.

---

## 🧩 Plugin System

Bring your own **controller** or **network** without touching the core.

**Custom controllers** — a YAML + a Python file exposing `compute_setpoints(vm_pu, p_installed) -> q_mvar`. Two worked examples are included (a proportional droop and a four-breakpoint deadband curve). A controller can also run as **real firmware on the Arduino** (`hardware: true` + an `.ino` sketch speaking the standard protocol), so a student's own algorithm can be benchmarked on real hardware alongside the built-ins.

**Custom networks** — a YAML pointing at a pandapower **JSON**, a **pickle**, or a zero-argument **loader function**, with a choice of profile strategy. The network is validated for benchmark compatibility (DER presence, transformer, voltage level, profile alignment) with actionable warnings rather than hard failures.

> ⚠️ **Security.** The plugin system **imports and executes user-supplied Python**, and can load **pickled** pandapower networks (`pp.from_pickle`). Treat controller/network plugins and pickle files as **trusted inputs** — only load them from sources you trust. Do not run untrusted plugin YAMLs, `.py` loaders, or `.pkl` networks.

---

## 🖥️ How to Run (detailed)

### Environment

```bash
conda create -n hil python=3.12
conda activate hil
pip install -r requirements.txt
```

Core dependencies (full list in [`requirements.txt`](requirements.txt)): `pandapower`, `simbench`, `numpy`, `pandas`, `scipy`, `pvlib`, `oemof.demand`, `pyserial`, `matplotlib`, `networkx`, `streamlit`,`plotly`, `rich`, `pyyaml`. Two dependencies are **optional** and left commented in `requirements.txt`: `contextily` (OSM basemaps for geographic network plots) and `cyipopt` (needed only for the OPF scenario — it requires a working Ipopt install). The Arduino firmware (`volt_var_arduino/volt_var_arduino.ino`) is flashed with the Arduino IDE / `arduino-cli`.

### Benchmark via the interactive wizard (recommended)

```bash
python -m Command_Line_Interface
```

The wizard covers study, network, profile strategy, parameters, hardware (dry-run vs a serial port), and streaming, then executes and publishes the run. Configurations can be saved/reloaded as presets.
The defaults are chosen to work on a fresh clone: step 4 of 9 (Dataset) uses SimBench native profiles, which ship with the simbench package and need no downloaded data. The one default to change deliberately is step 5 of 9 (Time window), which is Full annual — 35,136 timesteps at 15-minute resolution and several hours of runtime. Choose One day, One week or One month for anything exploratory.

### Benchmark via the direct script (fast iteration)

Edit the constants at the top of the direct-run script (network, scaling, DER injection, run name), then run it — optionally pointing at a plugin:

```bash
python run_benchmark_script.py --network <network.yaml> --controller <controller.yaml>
```

Parametric overrides can be passed on the command line without editing the file — `--q-ratio`, `--u1 … --u4` (Q(V) breakpoints), `--v-min` / `--v-max`, and `--line-max-loading` / `--trafo-max-loading`.

### Explore results — dashboard

```bash
streamlit run Streamlit/Home.py
```

The dashboard discovers runs and shows the scenario comparison, network view, profiles, hosting capacity, and live run health — reading the same JSON the publisher writes (it can follow a run live).

---

## ⚙️ Technical Stack

- **Python** — orchestration, control, analysis
- **pandapower** — AC power flow, OLTC/SVC, `runopp` OPF
- **SimBench** — MV benchmark networks & native profiles
- **pvlib + BDEW + DWD open data** — synthetic PV & load profiles
- **NumPy / pandas / SciPy** — numerics (SciPy `lstsq` for the ∂V/∂Q sensitivity model)
- **Matplotlib** — report figures
- **Streamlit + Rich** — dashboard & CLI console
- **Arduino (C++)** — Volt-Var (Q(V)) firmware, the hardware-in-the-loop endpoint
- **Raspberry Pi** — embedded simulation host for the HIL comparison

---

## 📁 Repository Structure

Run outputs (`outputs/`, `runs/`, `outputs (RPi)/`) are regenerable and are not committed, with two deliberate exceptions: `runs/demo_day_1-MV-rural--2-sw` and `runs/demo_week_1-MV-rural--2-sw`, short dry-run benchmarks kept in the repository so the dashboard is usable immediately. Everything else — run a benchmark to produce it

```
HIL-Testbed/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── volt_var_controller.py        # Q(V) characteristic, serial interface, control loop
├── sensitivity_coordinator.py    # ∂V/∂Q coordination (Tier 2)
├── der_dynamics.py               # PT1 reactive response + ramp limiter
├── violation_detector.py         # single source of violation truth (±ε thresholds)
├── hosting_capacity.py           # baseline & Volt-Var hosting-capacity analysis
├── publisher.py                  # JSON/JSONL output + crash-resume checkpointing
├── era5_to_csv.py                # ERA5 reanalysis → custom profile CSVs
│
├── scenario_runners/             # the benchmark core
│   ├── benchmark_runner.py       #   orchestrator (scenario registry, resume, HC)
│   ├── scenario_1_baseline.py … scenario_5_opf.py   # legacy filenames; OPF registers as scenario 6
│   ├── scenario_result.py        #   TimestepRecord / ScenarioResult / metrics
│   ├── plot_results.py           #   report figures
│   ├── custom_controller.py      #   generic user-controller loop
│   ├── plugin_runner.py          #   load/register a controller plugin
│   ├── network_plugin.py         #   load/validate a custom network
│   ├── example_plugins/          #   droop + deadband controller examples
│   └── example_networks/         #   network-plugin YAML examples
│
├── Command_Line_Interface/       # the CLI
│   ├── __main__.py               #   entry (preset / wizard → execute)
│   ├── wizard.py                 #   interactive configuration wizard
│   ├── run_plan.py               #   RunPlan / config dataclasses
│   └── executor.py               #   RunPlan → executed, published benchmark
│
├── Streamlit/                    # live/results dashboard
│   ├── Home.py, _data.py, render_frames.py
│   └── pages/                    #   scenario comparison, network, profiles, HC, health
│
├── volt_var_arduino/             # Arduino Volt-Var firmware (.ino) + protocol docs
│
├── hiltest/                      # earlier modular test framework (sections/*)
├── tests/                        # diagnostics & test suite
├── student_templates/            # Jupyter notebooks: dry-run vs HIL comparisons
├── Process Plots/                # architecture & per-component flowcharts
├── Walkthrough/                  # detailed line-by-line code documentation
├── Conference Paper/             # paper source (paper.tex)
├── data/                         # dwd/ · custom/ · era5/ · schematics/
└── presets/                      # saved run configurations
```

---

## 📤 Outputs

A benchmark writes, per network/run:

- `topology.json`, `profiles.json` — network geometry and annual profiles
- `scenarios/<id>.json` — full per-timestep record + summary for each scenario
- `comparison.json` / `comparison.csv` — the headline scenario comparison table
- `hc.json` — hosting-capacity results and gain
- `live/*.jsonl`, `checkpoint/*.jsonl` — streaming and resume data
- report figures (voltage heatmaps, Q(V) scatter, violation maps, HC sweep, benchmark summary, …)

---

## ⚠️ Model Limitations & Scope

**Steady-state, balanced power flow.** The benchmark uses balanced AC power flow; per-phase unbalance is supported in the violation model but not exercised by the six main scenarios (no runner performs a 3-phase flow).

**OPF scope and interpretation.** The AC-OPF scenario runs on CIGRE MV; SimBench MV ill-conditions the PYPOWER backend and is excluded from it. The OPF is an **optimisation reference** — an upper bound on DER active-power acceptance (equivalently a lower bound on curtailment) under the modelled constraints, *where it converges*. Non-convergence is a solver/conditioning outcome and does **not** by itself imply physical infeasibility.

**Benchmark networks, not field data.** The feeders are standard benchmark models (SimBench, CIGRE, Kerber, Dickert) and synthetic cases, chosen for reproducibility — not utility-derived field topologies.

**Synthetic profiles.** Where measured data is unavailable, profiles are reconstructed (DWD + pvlib + BDEW) or taken from SimBench; annual energy is anchored, exact sub-hourly timing is approximate.

**DER dynamics at annual resolution.** The PT1 reactive response and P ramp limiter are effectively non-binding at the standard 15-minute step; they matter only at finer control-loop resolutions.

**Known cosmetic / documentation items** (tracked, non-functional) are listed in the code walkthroughs — e.g. a few stale `hardware:`-related docstrings and a progress-percentage display quirk on resume in some runners.

---

## 🔜 Project Status

**Complete:** the full control stack (Q(V), coordination, DER response), all six scenarios, the HIL serial protocol and firmware, dry-run vs HIL comparison, hosting-capacity analysis, crash-resume checkpointing, the plugin system, the Streamlit dashboard, the CLI, student templates, and detailed line-by-line code documentation.

**Ongoing:** consolidation of the conference paper and figures; addressing the tracked cosmetic/documentation items.

---

## 📜 License

Released under the **MIT License** — see [`LICENSE`](LICENSE).

---

## ✍️ Acknowledgements

Developed as an M.Sc. project in the **Renewable Energy programme at the Carl von Ossietzky Universität Oldenburg**, Germany. The author holds a **DAAD scholarship**.

Benchmark data and tools are used under their respective licences: **SimBench**, the **CIGRE** benchmark models, **pandapower**, **pvlib**, and **DWD** open climate data.

---

## 📌 Citation / Reuse

The MIT licence governs reuse of the code. When reporting results derived from this testbed, please additionally distinguish clearly between:

- benchmark network data (SimBench / CIGRE / Kerber / Dickert)
- reconstructed / synthetic profiles (DWD + pvlib + BDEW, ERA5)
- the control-law parameters and engineering assumptions (Q(V) breakpoints, `Q_RATIO`, limits)
- measured hardware-in-the-loop timing and quantisation results

The goal of the repository is not only to reproduce the benchmark, but to make the control assumptions, the hardware effects, and their limitations transparent.

**Repository:** https://gitlab.uni-oldenburg.de/sebi0767/hil-tesbed
