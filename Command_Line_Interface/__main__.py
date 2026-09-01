'''
Command-line entry point for the HIL CLI.

Flow:
1. Collect a RunPlan through the wizard
2. Print a human-readable summary.
3. Ask whether to proceed.
4. Optionally save the RunPlan as a preset JSON file.
5. Execute the run through the executor layer.
'''

from __future__ import annotations
import json
import sys
from pathlib import Path

from rich.prompt import Confirm, Prompt

from .wizard import run_wizard
from .helpers import print_run_plan, print_error_message
from ._console import console
from . import executor

from .run_plan import RunPlan
from .network_catalogue import get_preset_families, get_presets_for_family
from .executor import _preset_loaders

PRESET_DIR = Path("presets")

def _save_preset(plan) -> None:
    '''
    Save the selected RunPlan as a JSON preset
    '''

    preset_name = Prompt.ask("Preset name", console=console).strip()

    if not preset_name:
        console.print("[yellow]Preset not saved. Empty name provided.[/yellow]")
        return
    
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    
    preset_path = PRESET_DIR / f"{preset_name}.json"
    preset_json = json.dumps(plan.to_dict(), indent=2, sort_keys=True)

    preset_path.write_text(preset_json, encoding="utf-8")

    console.print(f"[green]Preset saved:[/green] {preset_path}")

def _validate_loaded_plan(plan) -> list[str]:
    """
    Semantic validation for a RunPlan reconstructed from a saved preset.
    from_dict() already guarantees structural safety (no KeyError/TypeError
    on missing or renamed fields) — this checks whether the loaded values
    still make sense against the CURRENT codebase, since a preset saved
    months ago may reference a preset name, custom file, or switch index
    that no longer exists or no longer applies to the same network.

    Returns a list of human-readable warning strings; an empty list means
    no issues were found. Does not raise — the caller decides whether to
    proceed, abort, or fall back to the wizard.
    """
    warnings: list[str] = []
    net_cfg = plan.network

    if net_cfg.source_type == "preset":
        loaders = _preset_loaders()
        if net_cfg.preset_name not in loaders:
            warnings.append(
                f"Preset network '{net_cfg.preset_name}' no longer exists "
                f"in the current preset catalogue."
            )

    elif net_cfg.source_type == "custom":
        if not net_cfg.custom_path or not Path(net_cfg.custom_path).exists():
            warnings.append(
                f"Custom network file '{net_cfg.custom_path}' does not "
                f"exist at this path anymore."
            )

    elif net_cfg.source_type == "plugin":
        if not net_cfg.plugin_path or not Path(net_cfg.plugin_path).exists():
            warnings.append(
                f"Network plugin YAML '{net_cfg.plugin_path}' does not "
                f"exist at this path anymore."
            )

    # Switch/DER modifications reference a specific network's structure —
    # only checkable by actually loading that network, which is more work
    # than a quick sanity pass should do. Instead, warn generically if
    # either is set, so the user knows to double-check them.
    if net_cfg.switches_to_flip:
        warnings.append(
            f"This preset flips switches {net_cfg.switches_to_flip} — "
            f"verify these indices are still valid for the target network "
            f"before proceeding (indices depend on the network's own "
            f"switch table, which may differ if the network has changed)."
        )
    if net_cfg.der_placements:
        warnings.append(
            f"This preset injects {len(net_cfg.der_placements)} DER(s) at "
            f"specific bus indices — verify those buses still exist in the "
            f"target network."
        )

    if plan.hardware and not plan.port:
        warnings.append(
            "This preset has hardware=True but no port set — the run will "
            "fail at the hardware step unless a port is configured."
        )

    return warnings

def _load_preset() -> "RunPlan | None":
    """
    Load a previously saved RunPlan preset from PRESET_DIR, with semantic
    validation against the current codebase. Returns None if the user
    cancels, no presets exist, or the chosen preset fails to load.
    """
    if not PRESET_DIR.exists():
        console.print(f"[yellow]No presets found — {PRESET_DIR} does not exist.[/yellow]")
        return None

    available = sorted(p.stem for p in PRESET_DIR.glob("*.json"))
    if not available:
        console.print(f"[yellow]No presets found in {PRESET_DIR}.[/yellow]")
        return None

    console.print("[muted]Available presets:[/muted]")
    for name in available:
        console.print(f"  {name}")

    chosen = Prompt.ask("Preset name to load", console=console).strip()
    preset_path = PRESET_DIR / f"{chosen}.json"

    if not preset_path.exists():
        console.print(f"[error]No preset named '{chosen}' found in {PRESET_DIR}.[/error]")
        return None

    try:
        data = json.loads(preset_path.read_text(encoding="utf-8"))
        plan = RunPlan.from_dict(data)
    except Exception as exc:
        console.print(f"[error]Could not load preset '{chosen}': {exc}[/error]")
        return None

    issues = _validate_loaded_plan(plan)
    if issues:
        console.print("[warning]Validation warnings for this preset:[/warning]")
        for w in issues:
            console.print(f"  [warning]- {w}[/warning]")
        if not Confirm.ask("Proceed with this preset anyway?", default=False, console=console):
            console.print("[yellow]Preset load cancelled.[/yellow]")
            return None

    return plan


def main() -> None:
    '''
    Run the HIL CLI workflow
    '''
    plan = None
    if Confirm.ask("Load a saved preset instead of running the wizard?", default=False, console=console):
        plan = _load_preset()
        if plan is None:
            console.print("[yellow]Falling back to the wizard.[/yellow]")

    if plan is None:
        plan = run_wizard()

    print_run_plan(plan)

    print_run_plan(plan)

    proceed = Confirm.ask("Proceed?", default=True, console=console)

    if not proceed:
        console.print("[yellow]Cancelled.[/yellow]")
        sys.exit(0)
    
    save_as_preset = Confirm.ask("Save as preset?", default=False, console=console)

    if save_as_preset:
        _save_preset(plan)
    
    # execute() owns all error handling and returns a typed exit code
    # (executor.ExitCode): 0=OK, 2=config, 3=network, 4=dataset, 5=plugin,
    # 6=hardware, 7=simulation, 8=publish, 130=interrupted. The previous
    # bare try/except-sys.exit(1) pattern is replaced by propagating that
    # code, so shell scripts and CI can distinguish failure classes. The
    # try/except here is only the net for a bug in the executor itself.
    try:
        sys.exit(executor.execute(plan))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(int(executor.ExitCode.INTERRUPTED))
    except Exception as exc:
        print_error_message(
            exc, context="Unexpected internal error in the executor layer",
        )
        sys.exit(1)


if __name__ == "__main__":
    main()