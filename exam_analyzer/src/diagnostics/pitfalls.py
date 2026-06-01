import json
import os
import statistics
import numpy as np
from collections import defaultdict

from ..deepseek_client import call_flash, create_client
from ..knowledge_base import QADatabase
from ..embedding_cluster import _get_model, TOPIC_EMBED_MODEL, detect_content_lang
from ..models import KPSpec
from ..constants import (
    SQLITE_PARAM_CHUNK, TOPIC_MERGE_COS_THRESHOLD, TOPIC_MERGE_AMBIGUOUS_THRESHOLD,
    ANOMALY_ZSCORE_SINGLE, ANOMALY_ZSCORE_SYSTEMIC, SYSTEMIC_DIMENSION_COUNT,
    DIFFICULTY_HARD_THRESHOLD, DIFFICULTY_EASY_THRESHOLD,
)
from ..logger import get_logger
from ..utils import get_worker_limit

_log = get_logger()

"""Post-pipeline diagnostics: closed-loop improvement + cross-paper consistency.

Two subsystems that run after pipeline completion:
1. Closed-loop: auto-pitfall discovery, exam trends, confusion detection, QA-KP challenge
2. Cross-paper: structural fingerprints, dimension baselines, anomaly detection
"""

_log = get_logger()


def auto_discover_pitfalls(db: QADatabase, kp_id: str, debug_cb=None) -> list[dict]:
    """Discover pitfalls from Phase 2 missed_points data for a KP."""
    qa_rows = db.conn.execute(
        "SELECT qa_id FROM qa_kp_membership WHERE kp_id = ?", (kp_id,)
    ).fetchall()
    if not qa_rows:
        return []
    qa_ids = [r["qa_id"] for r in qa_rows]
    placeholders = ",".join("?" * len(qa_ids))
    fb_rows = db.conn.execute(
        f"""SELECT missed_text, miss_categories FROM question_feedback
            WHERE qa_id IN ({placeholders}) AND missed_text != ''""",
        qa_ids,
    ).fetchall()
    missed_lines = []
    for r in fb_rows:
        for line in r["missed_text"].split("\n"):
            line = line.strip()
            if line:
                missed_lines.append(line)
    if len(missed_lines) < 3:
        return []
    try:
        model = _get_model(TOPIC_EMBED_MODEL)
        vecs = model.encode(missed_lines, normalize_embeddings=True, convert_to_numpy=True)
    except Exception as e:
        from ..error_utils import log_exception
        log_exception(debug_cb, "Pitfall embedding", f"kp={kp_id}", e)
        return []
    groups = cluster_by_cosine(vecs, 0.80, min_group_size=3)
    patterns = [{"pattern": missed_lines[g[0]], "count": len(g)} for g in groups]
    if patterns:
        patterns.sort(key=lambda x: -x["count"])
        kp = db.get_kp_by_id(kp_id)
        if kp:
            variations = kp.get("variations", "") or "{}"
            try:
                var_dict = json.loads(variations)
            except (json.JSONDecodeError, TypeError):
                var_dict = {}
            var_dict["pitfalls"] = patterns[:5]
            db.kp.upsert(KPSpec(
                kp_id=kp_id, name=kp.get("name", ""), description=kp.get("description", ""),
                core_concept=kp.get("core_concept", ""), core_detail=kp.get("core_detail", ""),
                variations=json.dumps(var_dict),
                cohesion=kp.get("cohesion"), evidence_count=kp.get("evidence_count", 0),
                quality=kp.get("quality", "draft"),
            ))
    if debug_cb and patterns:
        top = patterns[0]
        from ..error_utils import log_info
        log_info(debug_cb, "KP pitfalls", f"{kp_id}: {len(patterns)} pitfalls "
                 f"(top: \"{top['pattern'][:80]}\" x{top['count']})")
    return patterns


def compute_exam_trends(db: QADatabase, debug_cb=None) -> int:
    """Compute exam trends for all KPs with >= 2 years of data."""
    kps = db.kp.get_all()
    if not kps:
        return 0
    trend_count = 0
    for kp in kps:
        kp_id = kp["id"]
        rows = db.conn.execute(
            """SELECT s.year, s.season, COUNT(*) as cnt,
                      AVG(CASE WHEN q.difficulty_estimate = 'basic' THEN 1
                               WHEN q.difficulty_estimate = 'intermediate' THEN 2
                               WHEN q.difficulty_estimate = 'advanced' THEN 3 END) as avg_diff
               FROM qa_kp_membership m
               JOIN qa_pairs q ON m.qa_id = q.id
               JOIN exam_sessions s ON q.session_id = s.id
               WHERE m.kp_id = ? AND q.session_id IS NOT NULL
               GROUP BY s.year, s.season
               ORDER BY s.year, s.season""",
            (kp_id,),
        ).fetchall()
        if len(rows) < 2:
            continue
        years_seen = set()
        for r in rows:
            diff_label = "basic"
            avg = r["avg_diff"]
            if avg and avg > DIFFICULTY_HARD_THRESHOLD:
                diff_label = "advanced"
            elif avg and avg > DIFFICULTY_EASY_THRESHOLD:
                diff_label = "intermediate"
            db.analysis.upsert_exam_trend(
                kp_id=kp_id, year=r["year"], season=r["season"],
                occurrence_count=r["cnt"], avg_difficulty=diff_label,
            )
            years_seen.add(r["year"])
            trend_count += 1
        if len(years_seen) >= 3 and debug_cb:
            from ..error_utils import log_info
            log_info(debug_cb, "Trend", f"{kp_id}: {len(years_seen)} years, "
                     f"{sum(r['cnt'] for r in rows)} total occurrences")
    if debug_cb:
        multi_year = sum(1 for kp in kps if len(set(
            r["year"] for r in db.conn.execute(
                "SELECT year FROM exam_trends WHERE kp_id = ?", (kp["id"],)
            ).fetchall()
        )) >= 3)
        from ..error_utils import log_info
        log_info(debug_cb, "Trends", f"{trend_count} year-rows across {len(kps)} KPs "
                 f"({multi_year} KPs with >= 3 years data)")
    return trend_count


