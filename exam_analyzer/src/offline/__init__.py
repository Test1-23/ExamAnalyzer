"""Offline analysis package."""

from .verbs import analyze_command_verbs
from .difficulty import assess_difficulty
from .dependencies import discover_dependencies
from .report import run_offline_analysis

__all__ = [
    "analyze_command_verbs", "assess_difficulty",
    "discover_dependencies", "run_offline_analysis",
]
