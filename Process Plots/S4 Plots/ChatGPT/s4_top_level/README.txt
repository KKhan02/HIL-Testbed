Scenario 4 top-level figures, reverted to the initial design direction.

The IEEE figure is compact and single-column oriented.
The presentation figure uses three high-contrast conceptual lanes.

Logic: dry-run and HIL split during initialization, merge before dynamics.reset(), share the timestep loop, then split after the loop so only HIL executes close() -> TX END. Both paths then converge at ScenarioResult.from_records(), followed by on_scenario_end and return ScenarioResult.
