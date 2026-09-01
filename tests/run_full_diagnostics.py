#!/usr/bin/env python3
"""
run_full_diagnostics.py
========================
Consolidated post-run diagnostics for the HIL Testbed. Runs all five
existing analysis scripts against one pair of run directories (RPi/HIL and
laptop/dry-run), in a fixed order matching Stage H/J of
QUANTISATION_FINDING_AND_RERUN_PROCEDURE.md:

    1. analyze_timing.py              — structural validation + timing analysis
    2. diagnose_outliers.py           — per-scenario timing outlier detection
    3. compare_scenarios.py           — full electrical performance comparison
    4. diagnose_qinitial_clip.py      — H1/H2 q_initial clip diagnostic (Scenario 4B only)
    5. check_coordination_ground_truth.py — coordination ground-truth check (Scenario 4B only)

Each of the five is run exactly as documented for standalone use (same
--rpi-dir/--laptop-dir/--out arguments), so nothing about their own
validated logic changes — this script only orchestrates and summarises.

Scripts 4 and 5 are SKIPPED GRACEFULLY (not treated as failures) if the
run did not include a Volt-Var Coordinated (4B) scenario — e.g. a
Baseline/OLTC/SVC-only run, or a run using --scenarios without 4/5. This
is reported clearly in both the dashboard and the full report, not
silently omitted.

Outputs
-------
1. A condensed Rich dashboard printed to the terminal: headline numbers,
   pass/fail on structural validation, and a one-line status per script.
2. A full plain-text report (all five scripts' complete output,
   concatenated) — same content you'd get running them one by one.
3. A PDF version of the same full report, formatted with headers and
   tables, suitable for archival or as a thesis/paper appendix.

Usage
-----
    python run_full_diagnostics.py --rpi-dir <dir> --laptop-dir <dir> \\
        --out-dir <directory for report.txt / report.pdf>

--rpi-dir / --laptop-dir should point at the run's publisher output
directory (the one containing scenarios/<scenario_id>.json), i.e. the
same directories you'd pass to any of the five scripts individually.
--laptop-dir is optional — if omitted, only RPi/HIL data is analysed
(matches compare_scenarios.py's own single-environment mode).

No files are modified by this script or by anything it calls — every
diagnostic here is read-only. No .py files are edited.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

SCRIPT_DIR = Path(__file__).resolve().parent

SCENARIOS = ["baseline", "oltc", "svc", "volt_var_local", "volt_var_coord"]
SCENARIO_LABEL = {
    "baseline":       "Baseline",
    "oltc":           "OLTC",
    "svc":            "SVC",
    "volt_var_local": "Volt-Var Local (4A)",
    "volt_var_coord": "Volt-Var Coord (4B)",
}

console = Console()


# ===========================================================================
# Result container for one sub-script's run
# ===========================================================================

@dataclass
class ScriptResult:
    name: str
    ran: bool                      # False = gracefully skipped (e.g. no 4B data)
    ok: bool                       # False = actually failed (non-zero exit / exception)
    stdout: str = ""
    skip_reason: str = ""
    error: str = ""


# ===========================================================================
# Running the three subprocess-based scripts (analyze_timing, diagnose_outliers,
# compare_scenarios) — treated as black boxes, exactly as you'd run them by hand.
# ===========================================================================

def _run_subprocess_script(
    script_name: str, rpi_dir: Path, laptop_dir: Optional[Path], extra_args: list[str],
) -> ScriptResult:
    """
    NOTE on the scenarios/ subfolder: unlike _run_direct_diagnostic()'s
    two targets (diagnose_qinitial_clip.py, check_coordination_ground_
    truth.py), which append "scenarios" to whatever directory they're
    given internally, these three scripts' own --rpi-dir/--laptop-dir
    CLI arguments expect the scenarios/ folder directly (confirmed from
    each script's own default paths, which all end in ...\\scenarios).
    run_full_diagnostics.py's own --rpi-dir/--laptop-dir are documented
    as the publisher directory ONE LEVEL ABOVE scenarios/ -- so that
    join must happen here, at this call site, or every scenario lookup
    in these three scripts silently fails even on a perfectly valid run.
    """
    script_path = SCRIPT_DIR / script_name
    rpi_scenarios_dir = rpi_dir / "scenarios"
    cmd = [sys.executable, str(script_path),
           "--rpi-dir", str(rpi_scenarios_dir)]
    if laptop_dir is not None:
        cmd += ["--laptop-dir", str(laptop_dir / "scenarios")]
    cmd += extra_args

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return ScriptResult(script_name, ran=True, ok=False,
                             error="Timed out after 600s — unexpected for a "
                                   "read-only JSON analysis; check for a hang.")
    except Exception as exc:
        return ScriptResult(script_name, ran=True, ok=False, error=str(exc))

    output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        return ScriptResult(script_name, ran=True, ok=False, stdout=output,
                             error=f"Exited with code {proc.returncode}")
    return ScriptResult(script_name, ran=True, ok=True, stdout=output)


# ===========================================================================
# Running the two ad-hoc diagnostics directly (they now expose run() after
# today's refactor, so no subprocess needed — but we still go through their
# own graceful-skip return value rather than assuming success).
# ===========================================================================

def _run_direct_diagnostic(module_name: str, rpi_dir: Path, laptop_dir: Path) -> ScriptResult:
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        module = __import__(module_name)
    except Exception as exc:
        return ScriptResult(module_name, ran=True, ok=False, error=f"Import failed: {exc}")

    try:
        report_lines, ran = module.run(rpi_dir, laptop_dir)
    except Exception as exc:
        return ScriptResult(module_name, ran=True, ok=False, error=str(exc))

    output = "\n".join(report_lines)
    if not ran:
        # A graceful skip prints its own reason inside report_lines already —
        # surface a clean one-line reason for the dashboard (strip the
        # "SKIPPED — " prefix and any surrounding whitespace/newlines).
        raw = next((l for l in report_lines if "SKIPPED" in l), "skipped")
        skip_reason = raw.replace("SKIPPED", "").strip().lstrip("\u2014-").strip()
        skip_reason = skip_reason.splitlines()[0] if skip_reason else "skipped"
        return ScriptResult(module_name, ran=False, ok=True, stdout=output,
                             skip_reason=skip_reason)
    return ScriptResult(module_name, ran=True, ok=True, stdout=output)


# ===========================================================================
# Headline numbers for the dashboard — read directly from the publisher
# JSONs (small, targeted extraction; does not duplicate any of the five
# scripts' own analysis logic, just pulls a handful of summary fields for
# the condensed view).
# ===========================================================================

def _load_summary(run_dir: Path, sid: str) -> Optional[dict]:
    for name in (f"{sid}.json", f"{sid}_dry_run.json"):
        p = run_dir / "scenarios" / name
        if p.exists():
            with open(p) as f:
                payload = json.load(f)
            return payload.get("summary", {})
    return None


def _headline_table(rpi_dir: Path, laptop_dir: Optional[Path]) -> Table:
    table = Table(title="Headline scenario comparison", show_lines=False)
    table.add_column("Scenario", style="bold", min_width=20, no_wrap=True)
    table.add_column("Violations (RPi)", justify="right")
    if laptop_dir is not None:
        table.add_column("Violations (Laptop)", justify="right")
    table.add_column("max_vm_pu (RPi)", justify="right")
    if laptop_dir is not None:
        table.add_column("max_vm_pu (Laptop)", justify="right")
    table.add_column("Coord. rate (RPi)", justify="right")
    if laptop_dir is not None:
        table.add_column("Coord. rate (Laptop)", justify="right")

    for sid in SCENARIOS:
        rpi_summary = _load_summary(rpi_dir, sid)
        laptop_summary = _load_summary(laptop_dir, sid) if laptop_dir is not None else None
        if rpi_summary is None and laptop_summary is None:
            continue

        def fmt(summary, key, spec="{:.4f}"):
            if summary is None:
                return "[dim]—[/dim]"
            val = summary.get(key)
            if val is None:
                return "[dim]—[/dim]"
            try:
                return spec.format(val)
            except (ValueError, TypeError):
                return str(val)

        row = [SCENARIO_LABEL.get(sid, sid), fmt(rpi_summary, "n_violation_steps", "{:.0f}")]
        if laptop_dir is not None:
            row.append(fmt(laptop_summary, "n_violation_steps", "{:.0f}"))
        row.append(fmt(rpi_summary, "max_vm_pu"))
        if laptop_dir is not None:
            row.append(fmt(laptop_summary, "max_vm_pu"))
        row.append(fmt(rpi_summary, "coordination_rate"))
        if laptop_dir is not None:
            row.append(fmt(laptop_summary, "coordination_rate"))
        table.add_row(*row)

    return table


# ===========================================================================
# Dashboard rendering
# ===========================================================================

def _status_line(result: ScriptResult, label: str) -> Text:
    if not result.ran or (not result.ok and "SKIPPED" not in (result.skip_reason or "")):
        pass  # handled below explicitly
    if result.skip_reason:
        return Text(f"  \u25cb {label}: skipped — {result.skip_reason}", style="yellow")
    if not result.ok:
        return Text(f"  \u2717 {label}: FAILED — {result.error}", style="bold red")
    return Text(f"  \u2713 {label}: ok", style="green")


def render_dashboard(results: dict[str, ScriptResult], rpi_dir: Path,
                      laptop_dir: Optional[Path]) -> None:
    console.print()
    console.print(Panel.fit("HIL Testbed — Consolidated Diagnostics Summary",
                             style="bold cyan"))

    # --- Validation pass/fail, extracted from analyze_timing.py's own stdout ---
    timing_result = results["analyze_timing.py"]
    if timing_result.ok:
        text = timing_result.stdout
        if "STRUCTURAL VIOLATION" in text:
            console.print(Panel(
                "[bold red]STRUCTURAL VIOLATIONS DETECTED[/bold red] — "
                "analyze_timing.py found wiring-level issues, not just noise. "
                "See the full report for details before trusting this run's results.",
                style="red", title="Validation"
            ))
        elif "All validation checks passed" in text:
            console.print(Panel("[bold green]All validation checks passed[/bold green]",
                                 title="Validation"))
        else:
            console.print(Panel("[yellow]Validation ran, but pass/fail status could not be "
                                 "determined from output — check the full report.[/yellow]",
                                 title="Validation"))
    else:
        console.print(Panel(f"[bold red]Validation check itself failed to run: "
                             f"{timing_result.error}[/bold red]", title="Validation"))

    # --- Headline scenario comparison table ---
    console.print()
    try:
        console.print(_headline_table(rpi_dir, laptop_dir))
    except Exception as exc:
        console.print(f"[yellow]Could not build headline table: {exc}[/yellow]")

    # --- Per-script status lines ---
    console.print()
    console.print("[bold]Diagnostic scripts:[/bold]")
    labels = {
        "analyze_timing.py": "Timing validation + analysis",
        "diagnose_outliers.py": "Outlier detection",
        "compare_scenarios.py": "Scenario comparison",
        "diagnose_qinitial_clip": "Q_initial clip diagnostic (H1/H2)",
        "check_coordination_ground_truth": "Coordination ground-truth check",
    }
    for key, label in labels.items():
        if key in results:
            console.print(_status_line(results[key], label))

    console.print()
    console.print("[dim]Full detail for every script above is in the plain-text "
                   "and PDF reports.[/dim]")


# ===========================================================================
# Full plain-text report
# ===========================================================================

def build_full_text_report(results: dict[str, ScriptResult]) -> str:
    parts = []
    order = ["analyze_timing.py", "diagnose_outliers.py", "compare_scenarios.py",
             "diagnose_qinitial_clip", "check_coordination_ground_truth"]
    for key in order:
        if key not in results:
            continue
        r = results[key]
        parts.append("#" * 80)
        parts.append(f"# {key}")
        parts.append("#" * 80)
        if r.skip_reason:
            parts.append(f"SKIPPED — {r.skip_reason}")
        elif not r.ok:
            parts.append(f"FAILED — {r.error}")
            if r.stdout:
                parts.append(r.stdout)
        else:
            parts.append(r.stdout)
        parts.append("")
    return "\n".join(parts)


# ===========================================================================
# PDF report
# ===========================================================================

def build_pdf_report(results: dict[str, ScriptResult], rpi_dir: Path,
                      laptop_dir: Optional[Path], out_path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table as RLTable,
        TableStyle, Preformatted,
    )

    styles = getSampleStyleSheet()
    mono = ParagraphStyle("mono", parent=styles["Code"], fontSize=7, leading=8.5,
                           fontName="Courier")

    doc = SimpleDocTemplate(str(out_path), pagesize=letter,
                             topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    story = []

    story.append(Paragraph("HIL Testbed — Consolidated Diagnostics Report", styles["Title"]))
    story.append(Paragraph(f"RPi/HIL directory: {rpi_dir}", styles["Normal"]))
    if laptop_dir is not None:
        story.append(Paragraph(f"Laptop/dry-run directory: {laptop_dir}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # --- Status summary table ---
    labels = {
        "analyze_timing.py": "Timing validation + analysis",
        "diagnose_outliers.py": "Outlier detection",
        "compare_scenarios.py": "Scenario comparison",
        "diagnose_qinitial_clip": "Q_initial clip diagnostic (H1/H2)",
        "check_coordination_ground_truth": "Coordination ground-truth check",
    }
    rows = [["Script", "Status"]]
    for key, label in labels.items():
        if key not in results:
            continue
        r = results[key]
        if r.skip_reason:
            status = "Skipped"
        elif not r.ok:
            status = "FAILED"
        else:
            status = "OK"
        rows.append([label, status])

    status_table = RLTable(rows, colWidths=[3.5 * inch, 2 * inch])
    status_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(status_table)
    story.append(Spacer(1, 20))

    # --- Full output per script, monospaced ---
    order = ["analyze_timing.py", "diagnose_outliers.py", "compare_scenarios.py",
             "diagnose_qinitial_clip", "check_coordination_ground_truth"]
    for key in order:
        if key not in results:
            continue
        r = results[key]
        story.append(PageBreak())
        story.append(Paragraph(labels.get(key, key), styles["Heading1"]))
        if r.skip_reason:
            story.append(Paragraph(f"Skipped — {r.skip_reason}", styles["Normal"]))
        elif not r.ok:
            story.append(Paragraph(f"FAILED — {r.error}", styles["Normal"]))
            if r.stdout:
                story.append(Preformatted(r.stdout[:20000], mono))
        else:
            story.append(Preformatted(r.stdout[:40000], mono))

    doc.build(story)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rpi-dir", type=Path, required=True,
                         help="RPi/HIL run's publisher output directory "
                              "(contains scenarios/)")
    parser.add_argument("--laptop-dir", type=Path, default=None,
                         help="Laptop/dry-run publisher output directory. "
                              "Optional — omit for RPi-only analysis.")
    parser.add_argument("--out-dir", type=Path, default=Path("diagnostics_report"),
                         help="Directory to write report.txt and report.pdf into "
                              "(default: ./diagnostics_report)")
    parser.add_argument("--top-n-buses", type=int, default=10)
    parser.add_argument("--top-n-ders", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Running consolidated diagnostics...[/bold] "
                   "(this shells out to each analysis script in turn)")

    results: dict[str, ScriptResult] = {}

    with console.status("[bold]analyze_timing.py...[/bold]"):
        results["analyze_timing.py"] = _run_subprocess_script(
            "analyze_timing.py", args.rpi_dir, args.laptop_dir, [],
        )

    with console.status("[bold]diagnose_outliers.py...[/bold]"):
        results["diagnose_outliers.py"] = _run_subprocess_script(
            "diagnose_outliers.py", args.rpi_dir, args.laptop_dir, [],
        )

    with console.status("[bold]compare_scenarios.py...[/bold]"):
        results["compare_scenarios.py"] = _run_subprocess_script(
            "compare_scenarios.py", args.rpi_dir, args.laptop_dir,
            ["--top-n-buses", str(args.top_n_buses), "--top-n-ders", str(args.top_n_ders)],
        )

    laptop_dir_for_direct = args.laptop_dir if args.laptop_dir is not None else args.rpi_dir
    with console.status("[bold]diagnose_qinitial_clip...[/bold]"):
        results["diagnose_qinitial_clip"] = _run_direct_diagnostic(
            "diagnose_qinitial_clip", args.rpi_dir, laptop_dir_for_direct,
        )

    with console.status("[bold]check_coordination_ground_truth...[/bold]"):
        results["check_coordination_ground_truth"] = _run_direct_diagnostic(
            "check_coordination_ground_truth", args.rpi_dir, laptop_dir_for_direct,
        )

    # --- Dashboard ---
    render_dashboard(results, args.rpi_dir, args.laptop_dir)

    # --- Full text report ---
    text_report = build_full_text_report(results)
    text_path = args.out_dir / "report.txt"
    text_path.write_text(text_report, encoding="utf-8")

    # --- PDF report ---
    pdf_path = args.out_dir / "report.pdf"
    try:
        build_pdf_report(results, args.rpi_dir, args.laptop_dir, pdf_path)
    except Exception as exc:
        console.print(f"[bold red]PDF generation failed: {exc}[/bold red]")
        pdf_path = None

    console.print()
    console.print(f"[bold]Full text report:[/bold] {text_path}")
    if pdf_path:
        console.print(f"[bold]PDF report:[/bold] {pdf_path}")


if __name__ == "__main__":
    main()
