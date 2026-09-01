# HIL Testbed — Side Projects Master List

Post-semester / time-permitting. **None of these are to be started before the main HIL project (five-scenario benchmark + hosting capacity + annual re-run) is complete.**

Scope guard: anything pulling toward transmission-level work, stochastic methods, or effort exceeding roughly one semester gets flagged here rather than mainlined into the core project.

---

## Standalone side projects (SP1–SP12, SP22, SP23)

| # | Title | Est. Effort | Description |
|---|---|---|---|
| 1 | PST Redispatch | ~30h | Hagenwerder-Mikułowa phase-shifting transformer. pandapower + PyPSA-Eur, Central European transmission. Strong portfolio piece for TenneT/50Hertz. Simulation-only, no HIL. Reference paper (63-page thesis) queued for review — check PST modelling approach (ideal phase shifter vs. tap-changer equivalent vs. full transformer model), quantified redispatch volumes, network scope, ENTSO-E validation. |
| 2 | ZIP Load Modelling | ~10h | Voltage-dependent load model in SimBench networks. |
| 3 | Short-Circuit with High PV | ~10h | `runsc()` analysis on SimBench MV with high PV penetration. |
| 4 | Distributed vs Single Slack | ~15h | Islanded microgrid slack bus architecture comparison. |
| 5 | HIL + State Estimation | ~25h | Extend HIL loop with a state estimator in the loop. mosaik orchestrates pandapower + Arduino + WLS estimator. Flow: real `runpp()` result → simulated measurement noise → `pp.create_measurement()` → `pp.estimate()` → estimated voltages → feed to controller → controller makes Q decision → compare against perfect-information case. Research question: how much measurement noise degrades Volt-Var controller performance before it under/over-compensates or misses violations. Standalone, not tied to a specific course. |
| 6 | N-1 Contingency Analysis | ~15h | Security assessment on SimBench MV network. No new software needed beyond existing pandapower infrastructure. |
| 7 | Stochastic Hosting Capacity | ~20h | Monte Carlo PV placement analysis. mosaik parallelises pandapower instances. Eye2Sky (DLR, 1-min GHI/DHI/DNI, April 2022–March 2023, stations near Oldenburg) preferred over DWD for spatially-correlated PV. Standalone, not tied to a specific course. |
| 8 | OPF Formulation Benchmarking | ~25h | `runopp()` vs `runpm_ac_opf()` vs linearized/SDP OPF on SimBench MV. |
| 9 | Multinetwork Storage Optimisation | ~45h | PandaModels.jl multi-period dispatch, PV+storage co-optimisation, hosting capacity extension. `pp.create_storage()`. |
| 10 | Flask Web Dashboard (RPi) | ~20h | Browser-accessible via WiFi. Mateo leads the web layer. Dropdowns for grid model + scenario, triggers pandapower, streams results/plots. `network_plotter.py` stays unchanged (static matplotlib, report figures) — live dashboard needs a separate `live_plotter.py`: a `pf_res_plotly` wrapper serialising `net.res_bus.vm_pu` + `net.res_line.loading_percent` as JSON per timestep, pushed via SSE/polling. Runner script fully automated, no mid-loop prompting; interactivity limited to scenario selector at launch. CLI POSTs to Flask `/api/runs/<run_id>/publish`. |
| 11 | UBOPF for LV Networks | ~30h | Unbalanced OPF: replace pandapower's balanced positive-sequence network model with a full three-phase unbalanced model (same optimisation framework, harder physics — not a new solver). Use PowerModelsDistribution.jl (`run_mc_acp_opf`) on Kerber or Synthetic VC LV network, compare against `runopp()` balanced OPF to quantify the error from the balanced assumption in LV. Reference: Espinosa Del Pozo et al. (2026), SEGAN. GitHub: github.com/swwjdodwe1/UBOPF-Case-Studies. Sequencing: after main project + hosting capacity, when moving into LV stretch networks — slots in as an LV-side extension of Scenario 5. |
| 12 | Harmonics Analysis | ~20h | (Flagged in memory; scope not yet elaborated in past chats — revisit description when picked up.) |
| 22 | Interactive Microgrid Islanding & BESS Restoration | ~40h | Depends on SP4 + SP9 + SP10. No solver until balanced — deficit computed analytically via `island_power_balance()`. Virtual slack solve for voltage preview (explicitly labelled as estimated). Actions: switch open/close, sgen setpoint change, load curtailment, BESS discharge. Files: `island_detector.py`, `power_balance.py`, `virtual_solver.py`, `restoration_actions.py`. UI has display and edit modes. Convention: `storage.p_mw > 0` = charging, `< 0` = discharging. |
| 23 | CSP-CAES Hybrid | ~40h | oemof + pandapower black-start modelling. References: Baigorri 2025, Roncolato, Santo thesis. Note: CSP itself flagged unsuitable for a Karachi siting context. Links to SP22 (shared island detection/restoration infrastructure), SP9 (multi-period dispatch), SP20 (Karachi geographic constraint). Synchronous inertia provision vs. BESS flagged as the publishable differentiator — potential CIRED or IEEE PES venue if pursued that far. |

---

## Course-ported side projects (dedicated chats hold full detail — cross-reference only)

### Wind Turbines course
| # | Title | Est. Effort | Description |
|---|---|---|---|
| 13 | RL Volt-Var (SimBench) | ~40h | Reinforcement learning replacing rule-based Volt-Var control. Independent of SP14/SP15. |
| 14 | APC + Q(V), CIGRE MV Feeder 2, Arduino HIL | ~35h | Simulink (course exercises 2/7/8) used for design + EMT validation; Arduino Uno R3 executes the physical firmware. Post-semester form of a Tier 2 wind farm collector control item originally scoped for the main semester project but removed due to firmware complexity. |
| 15 | RL replaces PI-APC | ~45h | Depends on SP14. Simulink validates. |

### Resilient Operations of Future Grids course (Summer 2026, DLR)
| # | Title | Est. Effort | Description |
|---|---|---|---|
| 16 | Resilient Ops (3-layer) | ~90h | MATLAB/Simulink. **Layer 1:** existing five-scenario HIL benchmark on pandapower (quasi-static). **Layer 2:** EMT validation via pandapower → `to_mpc()` → MATPOWER → Simscape Electrical, capturing RoCoF, frequency nadir, inertia response. **Layer 3:** VSM control law ported from the Grid Forming Control (GFC) course Simulink exercises to Arduino firmware, closing the dynamic HIL loop. A fault ride-through (LVRT) experiment unifies all three layers. |
| 21 | Data Centre + Virtual Inertia | ~25h | Extends SP16 Layer 3. UPS batteries modelled as BESS in pandapower and as load-side VSM in Simscape. Future work, contingent on SP16 completing in time. |

### DES Co-Simulation course (mosaik-based)
| # | Title | Est. Effort | Description |
|---|---|---|---|
| 16b | mosaik + ICT/Cybersecurity | ~35h | pandapower + Containernet network emulator via mosaik. Studies comms resilience effect on Volt-Var performance. OFFIS thesis direction. |
| 17 | Multi-Source EMS | ~40h | oemof.solph (hydro/gas/BESS/PV/wind, 1h dispatch) → mosaik → pandapower (15-min power flow) → Arduino HIL. Full architecture chain; flagged as publishable. |
| 18 | Virtual Power Plant (VPP) | ~35h | oemof DER dispatch vs. EPEX spot prices → mosaik → pandapower → Arduino. Benchmark against SP8 (`runopp()`). |
| 19 | Demand Response + Storage | ~30h | oemof flex-loads + BESS + heat pump → mosaik → pandapower hosting capacity. Maps to sector integration. |

### RE City project (Karachi) — separate from DES course
| # | Title | Est. Effort | Description |
|---|---|---|---|
| 20 | Karachi RE City | ~45h | oemof RE city optimisation → pandapower → HIL Volt-Var, KE feeder zones. Explicitly **not** part of the DES co-simulation course. |

---

## Mentioned but not yet numbered

| Title | Notes |
|---|---|
| Simscape EMT coupling | pandapower → `to_mpc()` → MATPOWER → Simscape Electrical. Enables Tier 2 dynamic initialisation, vectorised HC asset placement (SP7), impedance scanning for SP12 (as a DIgSILENT alternative) and Tier 3 protection work. Overlaps architecturally with SP16 Layer 2. |
| QSS (quasi-steady-state) + fast/EMT hybrid model | Pure testbed extension, not tied to the paper. pandapower/QSTS handles the bulk of timesteps (it is already the "fast" engine); a separate fast/dynamic model (ANDES, or a minimal custom ODE integrator scoped to DER + local-bus behaviour) handles fault/fast-transient windows, triggered by a rule built on top of existing violation-detection or rate-of-change logic. Open design questions before implementation: (1) trigger definition, (2) fast-model choice, (3) reconciliation protocol between the fast model's state and pandapower's next steady-state snapshot (decoupled vs. loosely coupled vs. tightly coupled handoff). Parked — flagged as needing its own design document before any code. |
| pandapower ↔ PyPSA OPF coupling | `pypsa.Network.import_from_pandapower_net(net, extra_line_data=True)` is the official import path. Caveats: import is still in beta; does not support three-winding transformers, switches, in-service status, or transformer tap positions. SimBench networks (e.g. `1-MV-rural--2-sw`) use switches for radial topology, which is exactly what would be lost — likely the biggest silent-failure risk. Known community-reported issues: deprecated variable mismatches (`s_mva`) across pandapower versions, and scrambled bus/node ordering after round-tripping through PYPOWER's `ppc` format. Possible upside: PyPSA's linearised OPF formulation may sidestep the PYPOWER admittance ill-conditioning that blocked Scenario 5 AC OPF on SimBench MV — but note this answers a different question (linear approximation) than the existing AC `runopp()` comparison. Recommended first step if picked back up: round-trip one static timestep and manually diff bus/line counts and total load before trusting it in any pipeline. Parked.|

---

## Reading list notes tied to specific side projects (from Papers & Resources chat)

- **Kerber 2011 (TU München thesis)** — Kerber network source; read the hosting-capacity chapter when SP7 or general HC work needs it.
- **Fan & Shojaee (NJIT)** — strongest hardware citation available; same RPi + Arduino Uno R3 + UART stack with measured latency data. Worth a read before any further HIL serial implementation work, independent of side projects.
- **Haidekker et al. (AC microgrid testbed paper)** — InfluxDB + Grafana data pipeline pattern. Flagged for Mateo specifically when SP10 (Flask dashboard) starts, as a reference architecture Flask could replace or sit alongside.
- **Laaksonen et al. (QU-droop for MV DER)** — Section IV.A only; background/citation use.
- **Dickert et al. (foundational LV network paper)** — citation for hosting-capacity write-up.

---

## Tier 2 / Tier 3 (post-semester extensions to the core Volt-Var algorithm stack)

Tier 1 (Volt-Var Q(V), sensitivity-based Q coordination, violation detection, ramp rate limiting, power curtailment) is the only tier in scope for the semester project itself. Tier 2 and Tier 3 are explicitly post-semester side-project material, not semester scope.

**Tier 2:**
- P-f droop (CIGRE MV islanded) — note: requires dynamic simulation; `runpp()` is quasi-static only, so there's a solver mismatch to resolve before this is workable.
- Islanding + mode transition — same dynamic-simulation mismatch as above.
- Battery storage (SimBench MV + LV).
- Wind farm collector control (CIGRE MV Feeder 2) — later re-scoped and ported forward as SP14 (Wind Turbines course) after firmware complexity forced its removal from the original semester-adjacent plan.
- HVDC dispatch — skip, transmission-level, out of scope entirely.

**Tier 3:**
- Protection relay coordination.
- Grid code VDE compliance (pandapower post-processing).

---

## Raw memory entries (verbatim, pre-consolidation)

Kept here in full for reference since these memory slots have now been cleared. Numbering/detail here is the authoritative historical record if anything above needs re-deriving.

**Original SP1–13 entry:**
> SP1–13: (1)PST ~30h (2)ZIP load ~10h (3)SC PV ~10h (4)Slack microgrids ~15h (5)HIL+state estimation ~25h — mosaik orchestrates pandapower+Arduino+estimator (6)N-1 ~15h (7)Stochastic HC ~20h — mosaik parallelises pandapower instances; Eye2Sky preferred (8)OPF bench ~25h (9)PandaModels.jl ~45h. SP10 Flask–Mateo. SP11 UBOPF ~30h. SP12 Harmonics ~20h. SP13 RL Volt-Var ~40h.

**SP22/SP23 entry:**
> SP23: CSP-CAES ~40h (oemof+pandapower black start; Baigorri 2025, Roncolato, Santo thesis; CSP unsuitable Karachi). SP22 Interactive Microgrid ~40h (deps SP4+SP9+SP10).

**SP22 detail entry:**
> SP22 Interactive Microgrid Islanding & BESS Restoration ~40h. Deps: SP4+SP9+SP10. No solver until balanced — deficit via island_power_balance() analytically. Virtual slack solve for voltage preview (labelled estimated). Actions: switch open/close, sgen setpoint, load curtailment, BESS discharge. Files: island_detector.py, power_balance.py, virtual_solver.py, restoration_actions.py. UI: display+edit modes. storage.p_mw>0=charging, <0=discharging. Post-semester only.

**SP13/14/15 entry:**
> SP13/14/15 (Wind Turbines course, separate chat): SP13 RL Volt-Var SimBench ~40h (parallel). SP14 APC+Q(V) CIGRE MV Feeder 2 Arduino HIL ~35h; Simulink (Ex2/7/8) = design+EMT validation. SP15 RL replaces PI-APC ~45h; Simulink validates. SP14→SP15 dependency, SP13 independent.

**Simscape EMT coupling entry:**
> Simscape EMT coupling (post-semester): pandapower→to_mpc()→MATPOWER→Simscape Electrical. Enables Tier 2 dynamic init, vectorised HC asset placement (SP7), impedance scanning for SP12 (alternative to DIgSILENT) and Tier 3 protection. Eye2Sky (DLR, NW Germany): 1-min GHI/DHI/DNI, April2022-March2023 only, xarray format, stations near Oldenburg. Better than DWD for SP7 spatially-correlated PV. Integrate via file_map/col_map override — no profile_builder changes needed.

**Tier 2+3 entry:**
> Tier 2+3 are post-semester only. Tier 2: P-f droop (CIGRE MV islanded — NOTE: requires dynamic sim, runpp() quasi-static only, solver mismatch), Islanding+mode transition (same dynamic sim issue), Battery storage (SimBench MV+LV), Wind farm collector control (CIGRE MV Feeder 2), HVDC dispatch (skip — transmission-level). Tier 3: Protection relay coordination, Grid code VDE compliance (pandapower post-proc).

**Course-ported SPs cross-reference entry:**
> Course-ported SPs (separate chats): SP16 Resilient Ops ~90h (MATLAB/Simulink: L1 HIL; L2 Simscape EMT; L3 VSM Arduino; LVRT; SP21 DataCentre ~25h extends L3). SP5/SP16b/SP17/SP18/SP19 = DES co-sim course (mosaik). SP20 Karachi ~45h = RE City project (oemof→pandapower→HIL). SP13/14/15 = Wind Turbines course.

**mosaik-tied SP table (from "Resilient power grid project ideas" chat):**

| SP | Topic | Course/Context |
|----|-------|---------------|
| SP5 | HIL + State Estimation | mosaik orchestrates pandapower + Arduino + WLS estimator |
| SP7 | Stochastic HC (Monte Carlo) | mosaik parallelises pandapower instances; Eye2Sky PV profiles |
| SP16b | mosaik + ICT/Cybersecurity | pandapower + Containernet via mosaik; OFFIS thesis direction |
| SP17 | Multi-Source EMS | oemof.solph → mosaik → pandapower → Arduino HIL |
| SP18 | VPP | oemof DER dispatch vs EPEX → mosaik → pandapower → Arduino |
| SP19 | DR + Storage | oemof flex-loads + BESS + heatpump → mosaik → pandapower HC |
| SP20 | Karachi oemof project | oemof RE city → mosaik → pandapower KE feeder zones |

SP5 and SP7 are standalone post-semester SPs, not tied to a specific course. SP16b and SP17–SP20 were earmarked for the DES co-simulation course. SP20 (Karachi) was later explicitly separated out and confirmed as belonging to a separate RE City project, not the DES course.

**SP16/SP16b/SP17–SP21 revised entry (after SP20 correction):**
> SP16 Resilient Ops course ~90h (MATLAB/Simulink): L1 quasi-static HIL; L2 pandapower→to_mpc()→MATPOWER→Simscape EMT; L3 VSM Arduino firmware (from GFC course); LVRT unifying experiment. SP16b mosaik+ICT ~35h. SP17 EMS ~40h: oemof→mosaik→pandapower→Arduino. SP18 VPP ~35h. SP19 DR+Storage ~30h. SP20 Karachi ~45h: oemof RE city optimisation → pandapower → HIL Volt-VAr — SEPARATE RE City project, NOT DES course. SP21 DataCentre+VirtualInertia ~25h: extends SP16 L3; future work if SP16 completes in time.

**CSP-CAES full context (from "Modeling compressed air energy storage" chat summary):**
> Links were established to SP22 (shared island detection and restoration infrastructure), SP9 (multi-period dispatch), and SP20 (Karachi geographic constraint). Synchronous inertia provision versus BESS was identified as the publishable differentiator, suitable for CIRED or IEEE PES venues. N-1 contingency scanning (SP6, ~15h) requires no new software beyond existing pandapower infrastructure. Hosting capacity analysis in the current HIL testbed uses deterministic worst-case end-of-feeder placement rather than Monte Carlo, with stochastic Monte Carlo HC reserved for SP7 using mosaik parallelisation and Eye2Sky spatially-correlated irradiance data.

**Electrisim assessment (tangential, side-project-adjacent, from same chat as the mosaik SP table above):**
> Electrisim (electrisim.com / github.com/electrisim) was assessed as a GUI frontend for pandapower/OpenDSS with no hardware loop capability, making it categorically different from and less research-relevant than the HIL testbed work itself.

---

*Last consolidated: July 26, 2026, from project memory and past chat history, then memory slots for side-project content were cleared since this file is now the sole tracking source. Effort estimates and scope descriptions are as last confirmed — re-verify against the relevant dedicated chat before starting any item, since some (e.g. SP12 Harmonics) have only a placeholder description.*
