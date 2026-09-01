'''
Shared Rich Console for the HIL CLI.

This module owns the CLI theme and the single Console instance used by terminal-facing
CLI modules such as helpers.py and wizard.py
'''

from __future__ import annotations
from rich.console import Console
from rich.theme import Theme

CLI_THEME = Theme(
    {
        "header": "bold cyan",
        "rule": "cyan",
        "rule_end": "grey50",
        "stage": "bold yellow",
        "ok": "green",
        "warning": "yellow",
        "error": "bold red",
        "muted": "grey50",
        "algorithm": "cyan",
        "overvoltage": "yellow",
        "undervoltage": "blue",
        "overload": "magenta",
        "current": "bold default",
    }
)

console = Console(theme=CLI_THEME)