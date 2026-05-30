"""Post-pipeline diagnostics package — split from pipeline_diagnostics.py."""

from .pitfalls import auto_discover_pitfalls, compute_exam_trends
from .cross_paper import compute_paper_signature, update_baselines, detect_anomalies, run_cross_paper_check
from .student import apply_student_feedback
from .migration import run_closed_loop, run_phase2_cycle
from .cascade import _adjust_vectors_from_feedback, _compute_graph_centroid

__all__ = [
    "auto_discover_pitfalls", "compute_exam_trends",
    "compute_paper_signature", "update_baselines", "detect_anomalies",
    "run_cross_paper_check", "apply_student_feedback",
    "run_closed_loop", "run_phase2_cycle",
    "_adjust_vectors_from_feedback", "_compute_graph_centroid",
]
