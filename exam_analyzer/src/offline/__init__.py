"""Offline analysis — backward-compatible re-export.

Implementation moved to ``exam_analyzer/src/analysis/``.
"""

from ..analysis import (  # noqa: F401
    analyze_command_verbs, assess_difficulty,
    discover_dependencies, run_offline_analysis,
)
