"""Backward-compat shim — all symbols now in offline/ package."""

from .offline import (
    analyze_command_verbs, assess_difficulty,
    discover_dependencies, run_offline_analysis,
)
__all__ = ["analyze_command_verbs", "assess_difficulty", "discover_dependencies", "run_offline_analysis"]
