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


