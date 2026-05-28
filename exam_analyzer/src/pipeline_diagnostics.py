"""Post-pipeline diagnostics: closed-loop improvement + cross-paper consistency.

Two subsystems that run after pipeline completion:
1. Closed-loop: auto-pitfall discovery, exam trends, confusion detection, QA-KP challenge
2. Cross-paper: structural fingerprints, dimension baselines, anomaly detection
"""

import json
import statistics
import numpy as np

from .deepseek_client import call_flash, create_client
from .knowledge_base import QADatabase
from .embedding_cluster import _get_model, TOPIC_EMBED_MODEL
from .logger import get_logger

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
        if debug_cb:
            debug_cb(f"  Pitfall embedding failed for {kp_id}: {e}")
        return []
    n = len(missed_lines)
    assigned = [False] * n
    patterns = []
    for i in range(n):
        if assigned[i]:
            continue
        group = [missed_lines[i]]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if float(np.dot(vecs[i], vecs[j])) >= 0.80:
                group.append(missed_lines[j])
                assigned[j] = True
        if len(group) >= 3:
            patterns.append({"pattern": group[0], "count": len(group)})
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
            db.upsert_kp(
                kp_id=kp_id, name=kp.get("name", ""), description=kp.get("description", ""),
                core_concept=kp.get("core_concept", ""), core_detail=kp.get("core_detail", ""),
                variations=json.dumps(var_dict),
                cohesion=kp.get("cohesion"), evidence_count=kp.get("evidence_count", 0),
                quality=kp.get("quality", "draft"),
            )
    if debug_cb and patterns:
        top = patterns[0]
        debug_cb(f"  [DX] KP {kp_id}: {len(patterns)} pitfalls "
                 f"(top: \"{top['pattern'][:80]}\" x{top['count']})")
    return patterns


def compute_exam_trends(db: QADatabase, client, debug_cb=None) -> int:
    """Compute exam trends for all KPs with >= 2 years of data."""
    kps = db.get_all_kps()
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
            if avg and avg > 2.0:
                diff_label = "advanced"
            elif avg and avg > 1.3:
                diff_label = "intermediate"
            db.upsert_exam_trend(
                kp_id=kp_id, year=r["year"], season=r["season"],
                occurrence_count=r["cnt"], avg_difficulty=diff_label,
            )
            years_seen.add(r["year"])
            trend_count += 1
        if len(years_seen) >= 3 and debug_cb:
            debug_cb(f"  Trend for {kp_id}: {len(years_seen)} years, "
                     f"{sum(r['cnt'] for r in rows)} total occurrences")
    if debug_cb:
        multi_year = sum(1 for kp in kps if len(set(
            r["year"] for r in db.conn.execute(
                "SELECT year FROM exam_trends WHERE kp_id = ?", (kp["id"],)
            ).fetchall()
        )) >= 3)
        debug_cb(f"  [DX] Trends: {trend_count} year-rows across {len(kps)} KPs "
                 f"({multi_year} KPs with >= 3 years data)")
    return trend_count


# ═══════════════════════════════════════════════════════════════
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
    db.conn.commit()
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
        if z > 3.0:
            anomalies.append(f"{dim} severely anomalous (z={z:.1f}, value={val}, median={med})")
        elif z > 2.0:
            anomalies.append(f"{dim} moderately anomalous (z={z:.1f}, value={val}, median={med})")
    multi_check_dims = ["qa_count", "topic_count", "avg_miss_rate", "avg_answer_length"]
    dims_over = sum(1 for dim in multi_check_dims
                    if (bl := baselines.get(dim)) and bl["sample_count"] >= 3
                    and (val := sig_rows[dim]) is not None and bl["mad"] != 0
                    and abs(val - bl["median"]) / bl["mad"] > 2.0)
    if dims_over >= 3:
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

        if count >= 5:
            current = db.conn.execute(
                "SELECT mode_difficulty FROM topic_difficulty WHERE topic=?",
                (topic,),
            ).fetchone()

            current_level = difficulty_map.get(
                (current["mode_difficulty"] if current else "basic"), 1
            )

            if count >= 10 and current_level < 3:
                new_level = current_level + 1
            elif count >= 5 and current_level < 2:
                new_level = current_level + 1
            else:
                continue

            new_difficulty = rev_difficulty_map[new_level]
            db.conn.execute(
                """INSERT OR REPLACE INTO topic_difficulty
                   (topic, mode_difficulty, qa_count, assessed_at, assessment_method)
                   VALUES (?, ?, COALESCE((SELECT qa_count FROM topic_difficulty
                    WHERE topic=?), 1), datetime('now'), 'student_feedback')""",
                (topic, new_difficulty, topic),
            )
            db.conn.commit()
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
            db.conn.commit()

    _log.info(f"Student feedback applied: {len(confusion_counts)} topics with confusion, "
              f"{len(mastery_rows)} topics with mastery patterns")


# ═══════════════════════════════════════════════════════════════
# Unified entry points
# ═══════════════════════════════════════════════════════════════

def run_closed_loop(db_path: str, api_url: str, api_key: str, debug_callback=None):
    """Auto-discover pitfalls and compute exam trends."""
    def _debug(msg):
        if debug_callback:
            debug_callback(f"[DX] {msg}")
        else:
            print(f"[DX] {msg}")
    _debug("Starting closed-loop improvements...")
    db = QADatabase(db_path)
    kps = db.get_all_kps()
    if not kps:
        _debug("No KPs to improve")
        db.close()
        return
    client = create_client(api_url, api_key)
    try:
        pitfall_count = sum(1 for kp in kps if auto_discover_pitfalls(db, kp["id"], _debug))
        _debug(f"Auto-pitfalls: {pitfall_count} KPs updated")
        compute_exam_trends(db, client, _debug)
        _debug("Closed-loop improvements complete")
    finally:
        db.close()


def run_cross_paper_check(db_path: str, display_name: str = None, debug_callback=None):
    """Compute signature, update baselines, detect anomalies."""
    def _debug(msg):
        if debug_callback:
            debug_callback(f"[DX] {msg}")
        else:
            print(f"[DX] {msg}")
    db = QADatabase(db_path)
    try:
        if db.count() < 10:
            _debug("Too few QAs for cross-paper check, skipping")
            return
        sig = compute_paper_signature(db, display_name)
        if display_name and sig["qa_count"] > 0:
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
            db.conn.commit()
        update_baselines(db)
        if display_name:
            anomalies = detect_anomalies(db, display_name)
            if anomalies:
                db.conn.execute(
                    "UPDATE paper_signatures SET anomaly_flags = ? WHERE display_name = ?",
                    (json.dumps(anomalies), display_name),
                )
                db.conn.commit()
                for a in anomalies:
                    _debug(f"ANOMALY: {a}")
            else:
                _debug("No anomalies detected")
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════
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
        topic_helped = db.get_topic_helped_questions(topic_id)
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
        topic_helped = db.get_topic_helped_questions(topic_id)
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
        topic_helps[topic_id] = db.get_topic_helped_questions(topic_id)

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
                db.set_fragment_membership(fid, dst_topic, loyalty=0.5)
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
        topic_helps[topic_id] = db.get_topic_helped_questions(topic_id)

    updated = 0
    for topic_id in all_topics:
        mass_row = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM fragment_membership WHERE topic_id=?",
            (topic_id,)
        ).fetchone()
        mass = mass_row["cnt"] if mass_row else 0
        if mass == 0:
            db.conn.execute(
                "UPDATE dynamic_topics SET mass=0, quality='dissolved' WHERE topic_id=?",
                (topic_id,)
            )
            db.conn.commit()
            continue
        frags = db.get_topic_fragments(topic_id)
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
        db.update_topic_stats(topic_id, mass, round(cohesion, 3),
                              round(max(0.0, stability), 3))
        if stability >= 0.8 and mass >= 4:
            db.conn.execute(
                "UPDATE dynamic_topics SET quality='forming' WHERE topic_id=? "
                "AND quality='embryonic'", (topic_id,)
            )
        elif stability < 0.3 and mass < 3:
            db.conn.execute(
                "UPDATE dynamic_topics SET quality='dissolved' WHERE topic_id=?",
                (topic_id,)
            )
        db.conn.commit()
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


def run_phase2_cycle(db_path: str, debug_cb=None) -> dict:
    """Run Phase 2 self-organization: migration + topic stats + evolution."""
    db = QADatabase(db_path)
    try:
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
            db.conn.execute(
                """INSERT OR REPLACE INTO analysis_checkpoints (task_name, qa_count_at_run, status)
                   VALUES ('phase3_evolution', ?, 'completed')""",
                (total_help,),
            )
            db.conn.commit()

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
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════
# Phase 3: Topic evolution — split, merge, dissolve
# ═══════════════════════════════════════════════════════════════

def _detect_topic_splits(db: QADatabase, debug_cb=None) -> int:
    """Detect topics with diverging behavioral subgroups and split them."""
    topics = db.conn.execute(
        "SELECT topic_id, mass FROM dynamic_topics WHERE mass >= 6 AND quality != 'dissolved'"
    ).fetchall()
    if not topics:
        return 0

    splits = 0
    for t in topics:
        topic_id = t["topic_id"]
        frags = db.get_topic_fragments(topic_id)
        if len(frags) < 6:
            continue

        # Batch-load all fragment help data for this topic (one query, not O(n²))
        placeholders = ",".join("?" * len(frags))
        help_rows = db.conn.execute(
            f"SELECT fragment_id, helped_qa_id FROM fragment_help_map "
            f"WHERE fragment_id IN ({placeholders})",
            frags
        ).fetchall()
        frag_helps = {}
        for r in help_rows:
            frag_helps.setdefault(r["fragment_id"], set()).add(r["helped_qa_id"])

        # Build pairwise behavioral similarity matrix
        n = len(frags)
        sim_matrix = {}
        for i in range(n):
            for j in range(i + 1, n):
                fi_set = frag_helps.get(frags[i], set())
                fj_set = frag_helps.get(frags[j], set())
                if fi_set and fj_set:
                    sim = len(fi_set & fj_set) / len(fi_set | fj_set)
                else:
                    sim = 0.0
                sim_matrix[(i, j)] = sim

        # Find two subgroups with high internal + low cross similarity
        # Greedy: pick first pair with low cross-sim, build groups around them
        best_i, best_j = -1, -1
        best_diff = 0.0
        for (i, j), cross in sim_matrix.items():
            # Compute avg intra vs cross for this candidate split
            intra_i = sum(sim_matrix.get((min(i, k), max(i, k)), 0)
                         for k in range(n) if k != i and k != j)
            intra_j = sum(sim_matrix.get((min(j, k), max(j, k)), 0)
                         for k in range(n) if k != i and k != j)
            diff = max(0.0, (intra_i + intra_j) / max(n - 2, 1)) - cross
            if diff > best_diff:
                best_diff = diff
                best_i, best_j = i, j

        if best_diff < 0.3:
            continue

        # Build subgroups
        s1 = {frags[best_i]}
        s2 = {frags[best_j]}
        for k in range(n):
            if k == best_i or k == best_j:
                continue
            sim_to_s1 = sum(sim_matrix.get((min(k, x), max(k, x)), 0)
                           for x in [best_i] + [idx for idx, f in enumerate(frags) if f in s1])
            sim_to_s2 = sum(sim_matrix.get((min(k, x), max(k, x)), 0)
                           for x in [best_j] + [idx for idx, f in enumerate(frags) if f in s2])
            if sim_to_s1 >= sim_to_s2:
                s1.add(frags[k])
            else:
                s2.add(frags[k])

        if len(s1) < 3 or len(s2) < 3:
            continue

        # Check internal cohesion of each subgroup
        s1_sims = [sim_matrix.get((min(i, j), max(i, j)), 0)
                   for i in range(n) for j in range(i + 1, n)
                   if frags[i] in s1 and frags[j] in s1]
        s2_sims = [sim_matrix.get((min(i, j), max(i, j)), 0)
                   for i in range(n) for j in range(i + 1, n)
                   if frags[i] in s2 and frags[j] in s2]
        s1_avg = sum(s1_sims) / len(s1_sims) if s1_sims else 0
        s2_avg = sum(s2_sims) / len(s2_sims) if s2_sims else 0

        if s1_avg < 0.4 or s2_avg < 0.4:
            continue

        # Execute split
        new_id_a = f"{topic_id}_a"
        new_id_b = f"{topic_id}_b"
        old_name = db.conn.execute(
            "SELECT name FROM dynamic_topics WHERE topic_id=?", (topic_id,)
        ).fetchone()
        old_name = old_name["name"] if old_name else topic_id

        db.upsert_dynamic_topic(new_id_a, name=f"{old_name} (A)", quality="embryonic")
        db.upsert_dynamic_topic(new_id_b, name=f"{old_name} (B)", quality="embryonic")
        for fid in s1:
            db.set_fragment_membership(fid, new_id_a, loyalty=0.5)
        for fid in s2:
            db.set_fragment_membership(fid, new_id_b, loyalty=0.5)

        db.conn.execute(
            """UPDATE dynamic_topics SET quality='dissolved',
               child_topics=?, last_evolved_at=datetime('now')
               WHERE topic_id=?""",
            (json.dumps([new_id_a, new_id_b]), topic_id),
        )
        db.conn.commit()
        splits += 1
        if debug_cb:
            debug_cb(f"  Split: {topic_id} -> [{new_id_a}]({len(s1)}) + "
                     f"[{new_id_b}]({len(s2)})")

    return splits


def _detect_topic_merges(db: QADatabase, debug_cb=None) -> int:
    """Detect topic pairs with high behavioral overlap and merge them."""
    topics = db.conn.execute(
        "SELECT topic_id, mass FROM dynamic_topics WHERE mass >= 2 AND quality != 'dissolved'"
    ).fetchall()
    if len(topics) < 2:
        return 0

    topic_ids = [t["topic_id"] for t in topics]

    # Pre-load fragment help data for affinity computation
    frag_helps = {}
    help_rows = db.conn.execute(
        "SELECT fragment_id, helped_qa_id FROM fragment_help_map"
    ).fetchall()
    for r in help_rows:
        frag_helps.setdefault(r["fragment_id"], set()).add(r["helped_qa_id"])
    topic_helps = {tid: db.get_topic_helped_questions(tid) for tid in topic_ids}

    merges = 0
    merged_set = set()

    for i in range(len(topic_ids)):
        if topic_ids[i] in merged_set:
            continue
        for j in range(i + 1, len(topic_ids)):
            if topic_ids[j] in merged_set:
                continue
            a, b = topic_ids[i], topic_ids[j]

            # Behavioral overlap: helped question sets
            a_helped = db.get_topic_helped_questions(a)
            b_helped = db.get_topic_helped_questions(b)
            if not a_helped or not b_helped:
                continue

            overlap = len(a_helped & b_helped)
            union = len(a_helped | b_helped)
            jaccard = overlap / union if union > 0 else 0

            if jaccard < 0.5:
                continue

            # Bidirectional fragment affinity
            a_frags = db.get_topic_fragments(a)
            b_frags = db.get_topic_fragments(b)
            a_aff = sum(_compute_affinity(db, fid, b, frag_helps, topic_helps)
                        for fid in a_frags)
            b_aff = sum(_compute_affinity(db, fid, a, frag_helps, topic_helps)
                        for fid in b_frags)
            avg_a = a_aff / len(a_frags) if a_frags else 0
            avg_b = b_aff / len(b_frags) if b_frags else 0

            if avg_a < 0.4 or avg_b < 0.4:
                continue

            # Execute merge
            new_id = f"{a}_m{b}"
            name_a = db.conn.execute(
                "SELECT name FROM dynamic_topics WHERE topic_id=?", (a,)
            ).fetchone()
            name_b = db.conn.execute(
                "SELECT name FROM dynamic_topics WHERE topic_id=?", (b,)
            ).fetchone()
            merged_name = f"{(name_a['name'] if name_a else a)} + {(name_b['name'] if name_b else b)}"

            db.upsert_dynamic_topic(new_id, name=merged_name, quality="embryonic")
            for fid in a_frags + b_frags:
                db.set_fragment_membership(fid, new_id, loyalty=0.5)

            db.conn.execute(
                """UPDATE dynamic_topics SET quality='dissolved',
                   last_evolved_at=datetime('now') WHERE topic_id IN (?, ?)""",
                (a, b),
            )
            db.conn.execute(
                "UPDATE dynamic_topics SET merged_from=? WHERE topic_id=?",
                (json.dumps([a, b]), new_id),
            )
            db.conn.commit()
            merged_set.add(a)
            merged_set.add(b)
            merges += 1
            if debug_cb:
                debug_cb(f"  Merge: {a} + {b} -> {new_id} (jaccard={jaccard:.2f})")

    return merges


def _process_dissolved_topics(db: QADatabase, debug_cb=None) -> int:
    """Redistribute orphan fragments from dissolved topics."""
    dissolved = db.conn.execute(
        "SELECT topic_id FROM dynamic_topics WHERE quality='dissolved'"
    ).fetchall()
    if not dissolved:
        return 0

    all_topics = [r["topic_id"] for r in db.conn.execute(
        "SELECT topic_id FROM dynamic_topics WHERE quality != 'dissolved'"
    ).fetchall()]

    redistributed = 0
    for row in dissolved:
        topic_id = row["topic_id"]
        frags = db.get_topic_fragments(topic_id)
        if not frags:
            continue

        for fid in frags:
            best_aff = 0.0
            best_topic = None
            for other in all_topics:
                aff = _compute_affinity(db, fid, other)
                if aff > best_aff:
                    best_aff = aff
                    best_topic = other

            if best_topic and best_aff > 0.1:
                db.set_fragment_membership(fid, best_topic, loyalty=0.3)
            else:
                # Orphan: mark with low loyalty to current (dissolved) topic
                db.set_fragment_membership(fid, topic_id, loyalty=0.0)
            redistributed += 1

    if debug_cb and redistributed:
        debug_cb(f"  Dissolved: {len(dissolved)} topics, {redistributed} fragments redistributed")
    return redistributed


# ═══════════════════════════════════════════════════════════════
# Phase 5: Fragment centrality + Graph centroid + Vector cascade
# ═══════════════════════════════════════════════════════════════

def _update_fragment_centrality(db: QADatabase, fragment_id: str, help_score: float,
                                  help_level: str, topic_id: str) -> dict:
    """Update a fragment's centrality scores after LLM feedback."""
    prev = db.get_fragment_centrality(fragment_id)
    prev_count = prev["verification_count"] if prev else 0
    prev_avg = prev["avg_help_score"] if prev else 0.0

    new_count = prev_count + 1
    new_avg = (prev_avg * prev_count + help_score) / new_count

    # Topic coherence: does this fragment help the same questions as its topic peers?
    p_helped = db.get_fragment_help_count(fragment_id)
    topic_helped = db.get_topic_helped_questions(topic_id)
    if p_helped > 0 and topic_helped:
        p_rows = db.conn.execute(
            "SELECT DISTINCT helped_qa_id FROM fragment_help_map WHERE fragment_id=?",
            (fragment_id,)
        ).fetchall()
        p_set = {r["helped_qa_id"] for r in p_rows}
        coherence = len(p_set & topic_helped) / max(p_helped, 1)
    else:
        coherence = 0.0

    # Variance: diversity of question types helped
    topic_rows = db.conn.execute(
        """SELECT DISTINCT q.topic FROM fragment_help_map fhm
           JOIN qa_pairs q ON fhm.helped_qa_id = q.id
           WHERE fhm.fragment_id = ?""", (fragment_id,)
    ).fetchall()
    variance = len(topic_rows) / max(p_helped, 1) if p_helped > 0 else 0

    centrality = (0.3 * min(1.0, new_count / 10)
                  + 0.4 * new_avg
                  + 0.3 * coherence)

    db.upsert_fragment_centrality(
        fragment_id, round(centrality, 3), round(new_avg, 3),
        round(coherence, 3), round(variance, 3))

    return {"centrality": centrality, "coherence": coherence, "variance": variance}


def _compute_graph_centroid(vectors: list[np.ndarray], weights: list[float],
                             max_iter: int = 20, tol: float = 1e-4) -> np.ndarray:
    """Geometric median weighted by eigenvector-like centrality.
    Uses Weiszfeld algorithm with centrality weights. Robust to outliers."""
    if not vectors:
        return np.zeros(384)
    if len(vectors) == 1:
        return vectors[0].copy()

    total_w = sum(weights) or 1.0
    centroid = sum(w * v for w, v in zip(weights, vectors)) / total_w

    for _ in range(max_iter):
        numerator = np.zeros_like(centroid)
        denominator = 0.0
        for v, w in zip(vectors, weights):
            dist = np.linalg.norm(centroid - v)
            if dist < tol:
                return centroid
            weight = w / dist
            numerator += weight * v
            denominator += weight
        new_centroid = numerator / denominator if denominator > 0 else centroid
        if np.linalg.norm(new_centroid - centroid) < tol:
            return new_centroid
        centroid = new_centroid
    return centroid


def _adjust_vectors_from_feedback(db: QADatabase, debug_cb=None) -> dict:
    """Three-layer cascade: adjust Fragment→KP→Topic vectors from LLM feedback."""
    result = {"fragments_adjusted": 0, "kps_adjusted": 0, "topics_adjusted": 0}

    # Layer 1: Adjust fragment centrality from LLM help data
    # Batch-read per topic via get_topic_fragment_centralities (avoid N+1 SELECTs)
    topics = db.conn.execute(
        "SELECT topic_id FROM dynamic_topics WHERE quality != 'dissolved'"
    ).fetchall()
    centrality_updates = []
    for t in topics:
        cent_rows = db.get_topic_fragment_centralities(t["topic_id"])
        for cent in cent_rows:
            if cent["verification_count"] < 1:
                continue
            cohesion = cent.get("topic_coherence", 0.5)
            # EMA: 60% carry-over + 30% help performance + 10% coherence
            new_centrality = (0.60 * cent["centrality_score"]
                              + 0.30 * cent["avg_help_score"]
                              + 0.10 * cohesion)
            centrality_updates.append((
                cent["fragment_id"],
                round(min(1.0, new_centrality), 3),
                cent["avg_help_score"],
                cohesion,
                cent.get("variance", 0),
            ))
            result["fragments_adjusted"] += 1

    if centrality_updates:
        with db._write_lock:
            db.conn.executemany(
                """INSERT OR REPLACE INTO fragment_centrality
                   (fragment_id, verification_count, avg_help_score, topic_coherence,
                    variance, centrality_score, updated_at)
                   VALUES (?, COALESCE((SELECT verification_count FROM fragment_centrality
                    WHERE fragment_id=?), 0) + 1, ?, ?, ?, ?, datetime('now'))""",
                [(fid, fid, ahs, tc, var, cs) for fid, cs, ahs, tc, var in centrality_updates],
            )
            db.conn.commit()

    # Layer 2: KP vectors (cascade from member QA embeddings)
    kp_rows = db.conn.execute(
        "SELECT id FROM knowledge_points WHERE quality != 'disputed'"
    ).fetchall()
    for kp_row in kp_rows:
        kp_id = kp_row["id"]
        member_rows = db.conn.execute(
            "SELECT qa_id FROM qa_kp_membership WHERE kp_id=?", (kp_id,)
        ).fetchall()
        if not member_rows:
            continue
        qa_ids = [r["qa_id"] for r in member_rows[:5]]
        model = _get_model(TOPIC_EMBED_MODEL)
        qa_texts = []
        for qid in qa_ids:
            qa = db.get(qid)
            if qa:
                qa_texts.append((qa.get("question_text", "") + " " + qa.get("answer_text", ""))[:500])
        if qa_texts:
            vecs = model.encode(qa_texts, normalize_embeddings=True, convert_to_numpy=True)
            centroid = _compute_graph_centroid(list(vecs), [1.0] * len(vecs))
            db.upsert_kp_vector(kp_id, centroid)
            result["kps_adjusted"] += 1

    # Layer 3: Topic vectors (cascade from KP vectors)
    for t in topics:
        topic_id = t["topic_id"]
        kp_ids = [r["id"] for r in db.conn.execute(
            """SELECT DISTINCT kp.id FROM knowledge_points kp
               JOIN qa_kp_membership qkm ON kp.id = qkm.kp_id
               JOIN qa_pairs q ON qkm.qa_id = q.id
               WHERE q.topic = (SELECT name FROM dynamic_topics WHERE topic_id=?)""",
            (topic_id,)
        ).fetchall()]
        if len(kp_ids) < 2:
            continue
        vecs = []
        weights = []
        for kp_id in kp_ids:
            v = db.get_kp_vector(kp_id)
            if v is not None:
                vecs.append(v)
                weights.append(1.0)
        if vecs:
            centroid = _compute_graph_centroid(vecs, weights)
            db.upsert_topic_vector(topic_id, centroid, len(kp_ids))
            result["topics_adjusted"] += 1

    if debug_cb and sum(result.values()) > 0:
        debug_cb(f"  Vector cascade: {result['fragments_adjusted']} fragments, "
                 f"{result['kps_adjusted']} KPs, {result['topics_adjusted']} topics")
    return result
