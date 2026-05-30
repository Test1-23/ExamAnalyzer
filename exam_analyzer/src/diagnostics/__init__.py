"""Post-pipeline diagnostics package.

Sub-modules (code currently in pipeline_diagnostics.py — being migrated):
- pitfalls: auto_discover_pitfalls, compute_exam_trends
- cross_paper: compute_paper_signature, update_baselines, detect_anomalies
- student: apply_student_feedback
- migration: run_phase2_cycle, _run_migration_cycle, _update_topic_stats
- cascade: _adjust_vectors_from_feedback, _compute_graph_centroid, topic split/merge

All public symbols are re-exported from the parent module for now.
"""

from ..pipeline_diagnostics import (
    # pitfalls
    auto_discover_pitfalls,
    compute_exam_trends,
    # cross_paper
    compute_paper_signature,
    update_baselines,
    detect_anomalies,
    run_cross_paper_check,
    # student
    apply_student_feedback,
    # migration / closed-loop
    run_closed_loop,
    run_phase2_cycle,
    # cascade (internal, used by evolution)
    _adjust_vectors_from_feedback,
    _compute_graph_centroid,
)

__all__ = [
    "auto_discover_pitfalls", "compute_exam_trends",
    "compute_paper_signature", "update_baselines", "detect_anomalies",
    "run_cross_paper_check", "apply_student_feedback",
    "run_closed_loop", "run_phase2_cycle",
    "_adjust_vectors_from_feedback", "_compute_graph_centroid",
]
