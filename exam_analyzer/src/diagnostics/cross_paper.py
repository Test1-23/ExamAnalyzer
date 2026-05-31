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
)
from ..logger import get_logger
from ..utils import get_worker_limit

_log = get_logger()

# Cross-paper: signatures, baselines, anomaly detection
# ═══════════════════════════════════════════════════════════════


def _paper_filter(display_name):
    """Return (WHERE_or_AND_clause, params) for optional paper-scoped queries.

    Returns ("WHERE paper = ?", (name,)) when display_name is given,
    otherwise ("", ()) so queries work on all papers.
    """
    if display_name:
        return ("AND paper = ?", (display_name,))
    return ("", ())


def _query_paper_stats(db, display_name):
    """QA count, average answer length, and distinct topic count."""
    pf, params = _paper_filter(display_name)
    qa_rows = db.conn.execute(
        f"SELECT COUNT(*) as cnt, AVG(LENGTH(answer_text)) as avg_len "
        f"FROM qa_pairs WHERE 1=1 {pf}", params
    ).fetchone()
    qa_count = qa_rows["cnt"] if qa_rows else 0
    avg_len = qa_rows["avg_len"] or 0

    topic_rows = db.conn.execute(
        f"SELECT COUNT(DISTINCT topic) as cnt FROM qa_pairs "
        f"WHERE topic != '' AND topic != '(uncategorized)' {pf}", params,
    ).fetchone()
    topic_count = topic_rows["cnt"] if topic_rows else 0
    return qa_count, round(avg_len, 1) if avg_len else None, topic_count


def _query_paper_distributions(db, display_name):
    """Command verb and difficulty estimate distributions."""
    pf, params = _paper_filter(display_name)
    verb_rows = db.conn.execute(
        f"SELECT command_verb, COUNT(*) as cnt FROM qa_pairs "
        f"WHERE command_verb != '' AND command_verb != 'unknown' {pf} "
        f"GROUP BY command_verb", params
    ).fetchall()
    verb_dist = {r["command_verb"]: r["cnt"] for r in verb_rows}

    diff_rows = db.conn.execute(
        f"SELECT difficulty_estimate, COUNT(*) as cnt FROM qa_pairs "
        f"WHERE difficulty_estimate != '' {pf} "
        f"GROUP BY difficulty_estimate", params
    ).fetchall()
    difficulty_dist = {r["difficulty_estimate"]: r["cnt"] for r in diff_rows}
    return verb_dist, difficulty_dist


def _query_paper_miss_rate(db, display_name):
    """Average miss rate across all QAs with feedback data."""
    paper_clause = "AND q.paper = ?" if display_name else ""
    params = (display_name,) if display_name else ()
    miss_rows = db.conn.execute(
        f"SELECT AVG(CAST(missed_count AS REAL) / "
        f"CAST(covered_count + missed_count AS REAL)) as avg_miss "
        f"FROM question_feedback f JOIN qa_pairs q ON f.qa_id = q.id "
        f"{paper_clause} "
        f"WHERE covered_count + missed_count > 0", params,
    ).fetchone()
    if miss_rows and miss_rows["avg_miss"] is not None:
        return round(miss_rows["avg_miss"], 3)
    return None


def _query_paper_topic_purity(db, display_name):
    """Average topic purity (1 - std_len/avg_len) across topics with >= 2 QAs."""
    pf, params = _paper_filter(display_name)
    purity_rows = db.conn.execute(
        f"SELECT topic, COUNT(*) as n FROM qa_pairs "
        f"WHERE topic != '' AND topic != '(uncategorized)' {pf} "
        f"GROUP BY topic HAVING n >= 2", params
    ).fetchall()
    purities = []
    for r in purity_rows:
        var_rows = db.conn.execute(
            "SELECT LENGTH(answer_text) as l FROM qa_pairs "
            "WHERE topic = ?" + (f" {pf}" if display_name else ""),
            (r["topic"],) + (params if display_name else ()),
        ).fetchall()
        lens = [v["l"] for v in var_rows]
        if len(lens) >= 2:
            avg_len = sum(lens) / len(lens)
            if avg_len > 0:
                std = statistics.stdev(lens)
                purities.append(max(0.0, 1.0 - std / avg_len))
    if purities:
        return round(sum(purities) / len(purities), 3)
    return None


def compute_paper_signature(db: QADatabase, display_name: str = None) -> dict:
    """Compute structural fingerprint for a paper (or all papers if display_name is None)."""
    qa_count, avg_answer_length, topic_count = _query_paper_stats(db, display_name)
    verb_dist, difficulty_dist = _query_paper_distributions(db, display_name)
    avg_miss_rate = _query_paper_miss_rate(db, display_name)
    topic_purity_avg = _query_paper_topic_purity(db, display_name)

    return {
        "qa_count": qa_count,
        "topic_count": topic_count,
        "verb_dist": json.dumps(verb_dist),
        "difficulty_dist": json.dumps(difficulty_dist),
        "avg_miss_rate": avg_miss_rate,
        "avg_answer_length": avg_answer_length,
        "topic_purity_avg": topic_purity_avg,
    }


def update_baselines(db: QADatabase):
    """Update dimension baselines (median + MAD) from all paper_signatures."""
    dimensions = ["qa_count", "topic_count", "avg_miss_rate", "avg_answer_length"]
    papers = db.analysis.get_paper_signatures()
    sample_counts = []
    for dim in dimensions:
        values = [p[dim] for p in papers if p.get(dim) is not None]
        sample_counts.append(len(values))
        if len(values) < 3:
            continue
        med = statistics.median(values)
        mad = statistics.median([abs(v - med) for v in values])
        db.analysis.upsert_dimension_baseline(
            dimension=dim, mean=sum(values) / len(values),
            median=med, mad=mad, sample_count=len(values),
        )
    _log.info(f"Baselines updated: {len(dimensions)} dimensions, "
              f"sample counts: {dict(zip(dimensions, sample_counts))}")


def detect_anomalies(db: QADatabase, display_name: str) -> list[str]:
    """Detect anomalies for a paper by comparing against dimension baselines."""
    sig_rows = db.analysis.get_paper_signature(display_name)
    if not sig_rows:
        return []
    base_rows = db.analysis.get_dimension_baselines()
    baselines = {r["dimension"]: dict(r) for r in base_rows}
    anomalies = []
    dimensions = ["qa_count", "topic_count", "avg_miss_rate"]
    for dim in dimensions:
        bl = baselines.get(dim)
        if not bl or bl["sample_count"] < 3:
            continue
        val = sig_rows[dim]
        if val is None:
            continue
        med, mad = bl["median"], bl["mad"]
        if mad == 0:
            continue
        z = abs(val - med) / mad
        if z > ANOMALY_ZSCORE_SINGLE:
            anomalies.append(f"{dim} severely anomalous (z={z:.1f}, value={val}, median={med})")
        elif z > ANOMALY_ZSCORE_SYSTEMIC:
            anomalies.append(f"{dim} moderately anomalous (z={z:.1f}, value={val}, median={med})")
    multi_check_dims = ["qa_count", "topic_count", "avg_miss_rate", "avg_answer_length"]
    dims_over = sum(1 for dim in multi_check_dims
                    if (bl := baselines.get(dim)) and bl["sample_count"] >= 3
                    and (val := sig_rows[dim]) is not None and bl["mad"] != 0
                    and abs(val - bl["median"]) / bl["mad"] > 2.0)
    if dims_over >= SYSTEMIC_DIMENSION_COUNT:
        anomalies.append(f"SYSTEMIC: {dims_over}/{len(multi_check_dims)} dimensions anomalous")
    return anomalies


# ═══════════════════════════════════════════════════════════════
# Student feedback closed loop
# ═══════════════════════════════════════════════════════════════

def apply_student_feedback(db: QADatabase):
    """Close the loop: student confusion → KP difficulty/quality adjustment.

    Scans confusion_events and student_knowledge_state to detect patterns,
    then adjusts topic_difficulty and flags KPs for re-review.
    """
    # Count confusion events per topic
    confusion_counts = db.conn.execute(
        """SELECT topic, COUNT(*) as cnt
           FROM confusion_events
           WHERE created_at > datetime('now', '-30 days')
           GROUP BY topic"""
    ).fetchall()

    difficulty_map = {"basic": 1, "intermediate": 2, "advanced": 3}
    rev_difficulty_map = {1: "basic", 2: "intermediate", 3: "advanced"}

    for row in confusion_counts:
        topic = row["topic"]
        count = row["cnt"]

        # 学生混淆阈值: ≥5 次混淆才触发难度核查, ≥10 次升级到下一难度等级
        if count >= 5:
            current = db.conn.execute(
                "SELECT mode_difficulty FROM topic_difficulty WHERE topic=?",
                (topic,),
            ).fetchone()

            current_level = difficulty_map.get(
                (current["mode_difficulty"] if current else "basic"), 1
            )

            # 升级条件: count≥10→可升级到最高级, count≥5→可升级到中级
            if count >= 10 and current_level < 3:
                new_level = current_level + 1
            elif count >= 5 and current_level < 2:
                new_level = current_level + 1
            else:
                continue

            new_difficulty = rev_difficulty_map[new_level]
            with db.transaction():
                db.conn.execute(
                    """INSERT OR REPLACE INTO topic_difficulty
                       (topic, mode_difficulty, qa_count, assessed_at, assessment_method)
                       VALUES (?, ?, COALESCE((SELECT qa_count FROM topic_difficulty
                        WHERE topic=?), 1), datetime('now'), 'student_feedback')""",
                    (topic, new_difficulty, topic),
                )
            _log.info(
                f"Student feedback: '{topic}' difficulty {current_level}→{new_level} "
                f"({count} confusion events)"
            )

    # Check student_knowledge_state for mastery patterns
    mastery_rows = db.conn.execute(
        """SELECT topic, COUNT(*) as cnt, state
           FROM student_knowledge_state
           WHERE state = 'mastered'
           GROUP BY topic"""
    ).fetchall()

    for row in mastery_rows:
        if row["cnt"] >= 3:
            # Consolidate: mark KPs in this topic as more stable
            with db.transaction():
                db.conn.execute(
                    """UPDATE knowledge_points SET quality = 'verified'
                       WHERE id IN (
                           SELECT kp_id FROM qa_kp_membership
                           WHERE qa_id IN (
                               SELECT id FROM qa_pairs WHERE topic = ?
                           )
                       ) AND quality = 'accepted'""",
                    (row["topic"],),
                )

    _log.info(f"Student feedback applied: {len(confusion_counts)} topics with confusion, "
              f"{len(mastery_rows)} topics with mastery patterns")


# ═══════════════════════════════════════════════════════════════
# Unified entry points
# ═══════════════════════════════════════════════════════════════

def run_closed_loop(db, api_url: str, api_key: str, debug_callback=None):
    """Auto-discover pitfalls and compute exam trends."""
    def _debug(msg):
        if debug_callback:
            debug_callback(f"[DX] {msg}")
        else:
            print(f"[DX] {msg}")
    _debug("Starting closed-loop improvements...")
    kps = db.kp.get_all()
    if not kps:
        _debug("No KPs to improve")
        return
    client = create_client(api_url, api_key)
    pitfall_count = sum(1 for kp in kps if auto_discover_pitfalls(db, kp["id"], _debug))
    _debug(f"Auto-pitfalls: {pitfall_count} KPs updated")
    compute_exam_trends(db, _debug)
    _debug("Closed-loop improvements complete")


def run_cross_paper_check(db, display_name: str = None, debug_callback=None):
    """Compute signature, update baselines, detect anomalies."""
    def _debug(msg):
        if debug_callback:
            debug_callback(f"[DX] {msg}")
        else:
            print(f"[DX] {msg}")
    if db.count() < 10:
        _debug("Too few QAs for cross-paper check, skipping")
        return
    sig = compute_paper_signature(db, display_name)
    if display_name and sig["qa_count"] > 0:
        db.analysis.upsert_paper_signature(
            display_name=display_name, qa_count=sig["qa_count"],
            topic_count=sig["topic_count"], verb_dist=sig.get("verb_dist", ""),
            difficulty_dist=sig.get("difficulty_dist", ""),
            avg_miss_rate=sig["avg_miss_rate"],
            avg_answer_length=sig["avg_answer_length"],
            topic_purity_avg=sig["topic_purity_avg"],
        )
    update_baselines(db)
    if display_name:
        anomalies = detect_anomalies(db, display_name)
        if anomalies:
            db.analysis.update_paper_anomaly_flags(display_name, json.dumps(anomalies))
            for a in anomalies:
                _debug(f"ANOMALY: {a}")
        else:
            _debug("No anomalies detected")


