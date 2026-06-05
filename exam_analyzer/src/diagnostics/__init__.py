"""Post-pipeline diagnostics — backward-compatible re-export.

Implementation moved to ``exam_analyzer/src/analysis/``.
"""

from ..analysis import (  # noqa: F401
    auto_discover_pitfalls, compute_exam_trends,
    compute_paper_signature, update_baselines, detect_anomalies,
    run_cross_paper_check, run_closed_loop,
    apply_student_feedback, run_phase2_cycle,
    _adjust_vectors_from_feedback, _compute_graph_centroid,
)
