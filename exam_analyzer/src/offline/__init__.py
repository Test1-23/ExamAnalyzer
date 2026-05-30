"""Offline analysis package.

Sub-modules (code currently in offline_analyzer.py — being migrated):
- verbs: analyze_command_verbs + all _phase* helpers + _write_verb_report
- difficulty: assess_difficulty + all _phase* helpers
- dependencies: discover_dependencies + all _phase* helpers

All public symbols are re-exported from the parent module for now.
"""

from ..offline_analyzer import (
    analyze_command_verbs,
    assess_difficulty,
    discover_dependencies,
    run_offline_analysis,
)

__all__ = [
    "analyze_command_verbs", "assess_difficulty",
    "discover_dependencies", "run_offline_analysis",
]
