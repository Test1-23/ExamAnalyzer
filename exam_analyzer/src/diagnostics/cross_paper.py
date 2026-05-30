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

def compute_paper_signature(db: QADatabase, display_name: str = None) -> dict:
    """Compute structural fingerprint for a paper (or all papers if display_name is None)."""
    paper_filter = "WHERE paper = ?" if display_name else ""
    params = (display_name,) if display_name else ()

    qa_rows = db.conn.execute(
        f"SELECT COUNT(*) as cnt, AVG(LENGTH(answer_text)) as avg_len "
        f"FROM qa_pairs {paper_filter}", params
    ).fetchone()
    qa_count = qa_rows["cnt"] if qa_rows else 0
    avg_answer_length = qa_rows["avg_len"] or 0

    topic_rows = db.conn.execute(
        f"SELECT COUNT(DISTINCT topic) as cnt FROM qa_pairs "
        f"WHERE topic != '' AND topic != '(uncategorized)' "
        f"{'AND paper = ?' if display_name else ''}", params,
    ).fetchone()
    topic_count = topic_rows["cnt"] if topic_rows else 0

    verb_rows = db.conn.execute(
        f"SELECT command_verb, COUNT(*) as cnt FROM qa_pairs "
        f"WHERE command_verb != '' AND command_verb != 'unknown' "
        f"{'AND paper = ?' if display_name else ''} "
        f"GROUP BY command_verb", params
    ).fetchall()
    verb_dist = {r["command_verb"]: r["cnt"] for r in verb_rows}

    diff_rows = db.conn.execute(
        f"SELECT difficulty_estimate, COUNT(*) as cnt FROM qa_pairs "
        f"WHERE difficulty_estimate != '' "
        f"{'AND paper = ?' if display_name else ''} "
        f"GROUP BY difficulty_estimate", params
    ).fetchall()
    difficulty_dist = {r["difficulty_estimate"]: r["cnt"] for r in diff_rows}

    miss_rows = db.conn.execute(
        f"SELECT AVG(CAST(missed_count AS REAL) / CAST(covered_count + missed_count AS REAL)) as avg_miss "
        f"FROM question_feedback f JOIN qa_pairs q ON f.qa_id = q.id "
        f"{'AND q.paper = ?' if display_name else ''} "
        f"WHERE covered_count + missed_count > 0", params,
    ).fetchone()
    avg_miss_rate = miss_rows["avg_miss"] if miss_rows else None

    purity_rows = db.conn.execute(
        f"SELECT topic, COUNT(*) as n FROM qa_pairs "
        f"WHERE topic != '' AND topic != '(uncategorized)' "
        f"{'AND paper = ?' if display_name else ''} "
        f"GROUP BY topic HAVING n >= 2", params
    ).fetchall()
    purities = []
    for r in purity_rows:
        if display_name:
            var_rows = db.conn.execute(
                "SELECT LENGTH(answer_text) as l FROM qa_pairs WHERE topic = ? AND paper = ?",
                (r["topic"], display_name),
            ).fetchall()
        else:
            var_rows = db.conn.execute(
                "SELECT LENGTH(answer_text) as l FROM qa_pairs WHERE topic = ?", (r["topic"],)
            ).fetchall()
        lens = [v["l"] for v in var_rows]
        if len(lens) >= 2:
            avg_len = sum(lens) / len(lens)
            if avg_len > 0:
                std = statistics.stdev(lens)
                purities.append(max(0.0, 1.0 - std / avg_len))
    topic_purity_avg = sum(purities) / len(purities) if purities else None

    return {
        "qa_count": qa_count,
        "topic_count": topic_count,
        "verb_dist": json.dumps(verb_dist),
        "difficulty_dist": json.dumps(difficulty_dist),
        "avg_miss_rate": round(avg_miss_rate, 3) if avg_miss_rate is not None else None,
        "avg_answer_length": round(avg_answer_length, 1) if avg_answer_length else None,
        "topic_purity_avg": round(topic_purity_avg, 3) if topic_purity_avg is not None else None,
    }


def update_baselines(db: QADatabase):
    """Update dimension baselines (median + MAD) from all paper_signatures."""
    dimensions = ["qa_count", "topic_count", "avg_miss_rate", "avg_answer_length"]
    sample_counts = []
    for dim in dimensions:
        rows = db.conn.execute(
            f"SELECT {dim} FROM paper_signatures WHERE {dim} IS NOT NULL"
        ).fetchall()
        values = [r[dim] for r in rows if r[dim] is not None]
        sample_counts.append(len(values))
        if len(values) < 3:
            continue
        med = statistics.median(values)
        mad = statistics.median([abs(v - med) for v in values])
        db.conn.execute(
            """INSERT OR REPLACE INTO dimension_baselines
               (dimension, mean, median, mad, sample_count, last_updated)
               VALUES (?, ?, ?, ?, ?, datetime('now'))""",
            (dim, sum(values) / len(values), med, mad, len(values)),
        )
    db._db.maybe_commit()
    _log.info(f"Baselines updated: {len(dimensions)} dimensions, "
              f"sample counts: {dict(zip(dimensions, sample_counts))}")


def detect_anomalies(db: QADatabase, display_name: str) -> list[str]:
    """Detect anomalies for a paper by comparing against dimension baselines."""
    sig_rows = db.conn.execute(
        "SELECT * FROM paper_signatures WHERE display_name = ?", (display_name,)
    ).fetchone()
    if not sig_rows:
        return []
    base_rows = db.conn.execute("SELECT * FROM dimension_baselines").fetchall()
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
    kps = db.get_all_kps()
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
        with db.transaction():
            db.conn.execute(
                """INSERT OR REPLACE INTO paper_signatures
                   (display_name, qa_count, topic_count, verb_dist, difficulty_dist,
                    avg_miss_rate, avg_answer_length, topic_purity_avg, anomaly_flags)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (display_name, sig["qa_count"], sig["topic_count"],
                 sig["verb_dist"], sig["difficulty_dist"],
                 sig["avg_miss_rate"], sig["avg_answer_length"],
                 sig["topic_purity_avg"], ""),
            )
    update_baselines(db)
    if display_name:
        anomalies = detect_anomalies(db, display_name)
        if anomalies:
            with db.transaction():
                db.conn.execute(
                    "UPDATE paper_signatures SET anomaly_flags = ? WHERE display_name = ?",
                    (json.dumps(anomalies), display_name),
                )
            for a in anomalies:
                _debug(f"ANOMALY: {a}")
        else:
            _debug("No anomalies detected")


