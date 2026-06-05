"""Post-pipeline analysis — cross-paper checks, topic migration, pitfalls,
command-verb patterns, difficulty assessment, dependency discovery.

Previously spread across ``diagnostics/`` and ``offline/``.
All analysis functions re-exported from private ``_*.py`` modules.
"""

from ._cascade import _adjust_vectors_from_feedback, _compute_graph_centroid
from ._cross_paper import (
    compute_paper_signature, update_baselines, detect_anomalies,
    run_cross_paper_check, run_closed_loop,
)
from ._migration import run_phase2_cycle
from ._pitfalls import auto_discover_pitfalls, compute_exam_trends
from ._student import apply_student_feedback
from ._verbs import analyze_command_verbs
from ._difficulty import assess_difficulty
from ._dependencies import discover_dependencies
from ._report import run_offline_analysis

__all__ = [
    "_adjust_vectors_from_feedback", "_compute_graph_centroid",
    "compute_paper_signature", "update_baselines", "detect_anomalies",
    "run_cross_paper_check", "run_closed_loop",
    "run_phase2_cycle",
    "auto_discover_pitfalls", "compute_exam_trends",
    "apply_student_feedback",
    "analyze_command_verbs", "assess_difficulty",
    "discover_dependencies", "run_offline_analysis",
]
