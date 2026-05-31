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

# Phase 2: Fragment migration + Topic stability
# ═══════════════════════════════════════════════════════════════

def _compute_loyalty(db: QADatabase, fragment_id: str, topic_id: str,
                     frag_helps: dict = None, topic_helps: dict = None) -> float:
    """How well does fragment P align with its current topic?
    Accepts pre-loaded frag_helps and topic_helps for batch efficiency."""
    # Use pre-loaded data if available
    if frag_helps is not None:
        p_helped_set = frag_helps.get(fragment_id, set())
    else:
        p_rows = db.conn.execute(
            "SELECT DISTINCT helped_qa_id FROM fragment_help_map WHERE fragment_id=?",
            (fragment_id,)
        ).fetchall()
        p_helped_set = {r["helped_qa_id"] for r in p_rows}
    p_helped = len(p_helped_set)
    if p_helped == 0:
        return 0.5

    if topic_helps is not None:
        topic_helped = topic_helps.get(topic_id, set())
    else:
        topic_helped = db.fragment.get_topic_helped_questions(topic_id)
    if not topic_helped:
        return 0.5

    return len(p_helped_set & topic_helped) / max(p_helped, 1)


def _compute_affinity(db: QADatabase, fragment_id: str, topic_id: str,
                      frag_helps: dict = None, topic_helps: dict = None) -> float:
    """How well does fragment P match a different topic?
    Accepts pre-loaded frag_helps and topic_helps for batch efficiency."""
    if frag_helps is not None:
        p_helped_set = frag_helps.get(fragment_id, set())
    else:
        p_rows = db.conn.execute(
            "SELECT DISTINCT helped_qa_id FROM fragment_help_map WHERE fragment_id=?",
            (fragment_id,)
        ).fetchall()
        p_helped_set = {r["helped_qa_id"] for r in p_rows}
    p_helped = len(p_helped_set)
    if p_helped == 0:
        return 0.0

    if topic_helps is not None:
        topic_helped = topic_helps.get(topic_id, set())
    else:
        topic_helped = db.fragment.get_topic_helped_questions(topic_id)
    if not topic_helped:
        return 0.0

    return len(p_helped_set & topic_helped) / max(min(p_helped, len(topic_helped)), 1)


def _get_migration_threshold(total_help_entries: int) -> float:
    """Adaptive threshold: higher when data is scarce."""
    base = 0.15
    if total_help_entries < 100:
        return base * 2.0
    elif total_help_entries < 300:
        return base * 1.5
    elif total_help_entries < 500:
        return base * 1.0
    else:
        return base * 0.7


def _run_migration_cycle(db: QADatabase, debug_cb=None) -> int:
    """Check all fragments for migration opportunities. Returns migration count."""
    total_help = db.conn.execute(
        "SELECT COUNT(*) as cnt FROM fragment_help_map"
    ).fetchone()["cnt"]
    if total_help < 20:
        if debug_cb:
            debug_cb(f"  Migration: too few help entries ({total_help}), skipping")
        return 0

    threshold = _get_migration_threshold(total_help)

    # Pre-load all fragment help data (one query, avoids O(fragments × topics) queries)
    all_help_rows = db.conn.execute(
        "SELECT fragment_id, helped_qa_id FROM fragment_help_map"
    ).fetchall()
    frag_helps = {}
    topic_helps = {}
    for r in all_help_rows:
        frag_helps.setdefault(r["fragment_id"], set()).add(r["helped_qa_id"])

    # Pre-load topic help sets
    rows = db.conn.execute(
        """SELECT fm.fragment_id, fm.topic_id as current_topic
           FROM fragment_membership fm
           JOIN ms_fragments mf ON fm.fragment_id = mf.point_id"""
    ).fetchall()
    all_topics = [r["topic_id"] for r in db.conn.execute(
        "SELECT topic_id FROM dynamic_topics"
    ).fetchall()]
    for topic_id in all_topics:
        topic_helps[topic_id] = db.fragment.get_topic_helped_questions(topic_id)

    migration_candidates = {}
    for row in rows:
        fid = row["fragment_id"]
        current = row["current_topic"]
        loyalty = _compute_loyalty(db, fid, current, frag_helps, topic_helps)
        best_affinity = 0.0
        best_topic = None
        for topic_id in all_topics:
            if topic_id == current:
                continue
            aff = _compute_affinity(db, fid, topic_id, frag_helps, topic_helps)
            if aff > best_affinity:
                best_affinity = aff
                best_topic = topic_id
        if best_topic and (best_affinity - loyalty) > threshold:
            key = (current, best_topic)
            migration_candidates.setdefault(key, []).append(fid)

    migrated = 0
    current_topic_mass = {}
    for (src_topic, dst_topic), fids in migration_candidates.items():
        if src_topic not in current_topic_mass:
            mass_row = db.conn.execute(
                "SELECT mass FROM dynamic_topics WHERE topic_id=?", (src_topic,)
            ).fetchone()
            current_topic_mass[src_topic] = mass_row["mass"] if mass_row else 0
        batch_threshold = 2 if current_topic_mass[src_topic] < 5 else 3
        if len(fids) >= batch_threshold:
            for fid in fids:
                db.topic.set_fragment_membership(fid, dst_topic, loyalty=0.5)
                migrated += 1
            if debug_cb:
                debug_cb(f"  Migration: {len(fids)} fragments {src_topic} -> {dst_topic}")
        elif debug_cb:
            debug_cb(f"  Migration deferred: {len(fids)} fragments {src_topic} -> "
                     f"{dst_topic} (need {batch_threshold}, have {len(fids)})")
    return migrated


def _update_topic_stats(db: QADatabase, debug_cb=None) -> int:
    """Update mass, cohesion, stability for all topics."""
    all_topics = [r["topic_id"] for r in db.conn.execute(
        "SELECT topic_id FROM dynamic_topics"
    ).fetchall()]

    # Pre-load fragment help data (avoid O(n) queries in loyalty loop)
    frag_helps = {}
    help_rows = db.conn.execute(
        "SELECT fragment_id, helped_qa_id FROM fragment_help_map"
    ).fetchall()
    for r in help_rows:
        frag_helps.setdefault(r["fragment_id"], set()).add(r["helped_qa_id"])

    topic_helps = {}
    for topic_id in all_topics:
        topic_helps[topic_id] = db.fragment.get_topic_helped_questions(topic_id)

    updated = 0
    for topic_id in all_topics:
        mass_row = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM fragment_membership WHERE topic_id=?",
            (topic_id,)
        ).fetchone()
        mass = mass_row["cnt"] if mass_row else 0
        if mass == 0:
            with db.transaction():
                db.conn.execute(
                    "UPDATE dynamic_topics SET mass=0, quality='dissolved' WHERE topic_id=?",
                    (topic_id,)
                )
            continue
        frags = db.topic.get_fragments(topic_id)
        if len(frags) < 2:
            cohesion = 1.0
        else:
            loyalties = [_compute_loyalty(db, fid, topic_id, frag_helps, topic_helps)
                         for fid in frags]
            cohesion = sum(loyalties) / len(loyalties)
        prev_rows = db.conn.execute(
            """SELECT COUNT(*) as cnt FROM fragment_membership
               WHERE topic_id=? AND previous_topic_id IS NOT NULL
               AND previous_topic_id != ?""",
            (topic_id, topic_id)
        ).fetchone()
        churn = prev_rows["cnt"] if prev_rows else 0
        stability = 1.0 - (churn / max(mass, 1))
        db.topic.update_stats(topic_id, mass, round(cohesion, 3),
                              round(max(0.0, stability), 3))
        if stability >= 0.8 and mass >= 4:
            with db.transaction():
                db.conn.execute(
                    "UPDATE dynamic_topics SET quality='forming' WHERE topic_id=? "
                    "AND quality='embryonic'", (topic_id,)
                )
        elif stability < 0.3 and mass < 3:
            with db.transaction():
                db.conn.execute(
                    "UPDATE dynamic_topics SET quality='dissolved' WHERE topic_id=?",
                    (topic_id,)
                )
        updated += 1

    if debug_cb:
        forming = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM dynamic_topics WHERE quality='forming'"
        ).fetchone()["cnt"]
        stable = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM dynamic_topics WHERE quality='stable'"
        ).fetchone()["cnt"]
        debug_cb(f"  Topic stats: {updated} topics, {forming} forming, {stable} stable")
    return updated


def run_phase2_cycle(db, debug_cb=None) -> dict:
    """Run Phase 2 self-organization: migration + topic stats + evolution."""
    migrated = _run_migration_cycle(db, debug_cb)
    updated = _update_topic_stats(db, debug_cb)

    # Phase 3: Topic evolution — only when help data has grown
    total_help = db.conn.execute(
        "SELECT COUNT(*) as cnt FROM fragment_help_map"
    ).fetchone()["cnt"]

    # Cooldown: only run evolution when help data has grown since last check
    prev_help = db.conn.execute(
        "SELECT COALESCE(MAX(qa_count_at_run), 0) as cnt FROM analysis_checkpoints "
        "WHERE task_name='phase3_evolution'"
    ).fetchone()["cnt"]

    splits = merges = dissolved = 0
    if total_help > prev_help + 20:  # at least 20 new help entries
        try:
            splits = _detect_topic_splits(db, debug_cb)
        except Exception as e:
            if debug_cb:
                debug_cb(f"  Split detection failed: {e}")
        try:
            merges = _detect_topic_merges(db, debug_cb)
        except Exception as e:
            if debug_cb:
                debug_cb(f"  Merge detection failed: {e}")
        try:
            dissolved = _process_dissolved_topics(db, debug_cb)
        except Exception as e:
            if debug_cb:
                debug_cb(f"  Dissolve processing failed: {e}")
        # Record checkpoint
        with db.transaction():
            db.conn.execute(
                """INSERT OR REPLACE INTO analysis_checkpoints (task_name, qa_count_at_run, status)
                   VALUES ('phase3_evolution', ?, 'completed')""",
                (total_help,),
            )

    # Phase 5: Vector cascade adjustment
    vectors_adjusted = 0
    try:
        v_result = _adjust_vectors_from_feedback(db, debug_cb)
        vectors_adjusted = sum(v_result.values())
    except Exception as e:
        if debug_cb:
            debug_cb(f"  Vector adjustment failed: {e}")

    return {"migrated": migrated, "topics_updated": updated,
            "splits": splits, "merges": merges, "dissolved": dissolved,
            "vectors_adjusted": vectors_adjusted}


