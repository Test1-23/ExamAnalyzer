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


# ═══════════════════════════════════════════════════════════════
# Closed-loop: confusion, QA challenge, pitfalls, trends
# ═══════════════════════════════════════════════════════════════

def detect_confusion(db: QADatabase, student_id: str, current_question: str,
                     kp_id: str, debug_cb=None) -> bool:
    """Detect if student's question indicates confusion about a KP."""
    row = db.conn.execute(
        """SELECT COUNT(*) as cnt FROM confusion_events
           WHERE student_id = ? AND topic = ? AND created_at > datetime('now', '-1 hour')""",
        (student_id, kp_id),
    ).fetchone()
    if row and row["cnt"] >= 2:
        if debug_cb:
            debug_cb(f"  Confusion threshold reached for KP {kp_id} (student {student_id})")
        return True
    confusion_keywords = ["confused", "don't understand", "why", "how come",
                          "不理解", "不懂", "为什么", "怎么回事", "不对吧"]
    q_lower = current_question.lower()
    return any(kw in q_lower for kw in confusion_keywords)


def trigger_kp_review_from_confusion(db: QADatabase, kp_id: str, debug_cb=None):
    """Mark a KP for adversarial re-review due to accumulated student confusions."""
    kp = db.get_kp_by_id(kp_id)
    if not kp:
        return
    db.upsert_kp(
        kp_id=kp_id, name=kp.get("name", ""), description=kp.get("description", ""),
        core_concept=kp.get("core_concept", ""), core_detail=kp.get("core_detail", ""),
        cohesion=kp.get("cohesion"), evidence_count=kp.get("evidence_count", 0),
        quality="draft", challenge_history=kp.get("challenge_history", ""),
    )
    if debug_cb:
        debug_cb(f"  KP {kp_id} marked for re-review due to student confusion")


def challenge_kp_with_new_qa(db: QADatabase, kp_id: str, new_qa_id: int,
                             client, debug_cb=None) -> bool:
    """Check if a new QA contradicts or refines an existing KP."""
    kp = db.get_kp_by_id(kp_id)
    qa = db.get(new_qa_id)
    if not kp or not qa:
        return False
    concept = kp.get("core_concept", "") or kp.get("description", "")
    if not concept:
        return False
    sys = "Compare a new QA with an existing knowledge point. Find contradictions or precision gaps. Output JSON."
    usr = (
        f"Knowledge Point concept: {concept}\n"
        f"KP detail: {kp.get('core_detail', '')}\n\n"
        f"New QA:\nQ: {qa['question_text']}\nA: {qa['answer_text']}\n\n"
        "Does the new QA contradict the KP? Does it reveal missing precision?\n"
        'Return: {"contradiction": true/false, "precision_gap": true/false, '
        '"detail": "explanation if any"}'
    )
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
    try:
        result = call_flash(client, messages, max_retries=1, debug_callback=debug_cb)
        result = result if isinstance(result, dict) else {}
    except Exception as e:
        if debug_cb:
            debug_cb(f"  QA challenge failed for KP {kp_id}: {e}")
        return False
    if result.get("contradiction") or result.get("precision_gap"):
        db.upsert_kp(
            kp_id=kp_id, name=kp.get("name", ""), description=kp.get("description", ""),
            core_concept=kp.get("core_concept", ""), core_detail=kp.get("core_detail", ""),
            cohesion=kp.get("cohesion"), evidence_count=kp.get("evidence_count", 0),
            quality="draft", challenge_history=kp.get("challenge_history", ""),
        )
        if debug_cb:
            debug_cb(f"  KP {kp_id} challenged by new QA {new_qa_id}: "
                     f"contradiction={result.get('contradiction')}, gap={result.get('precision_gap')}")
        return True
    return False


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
    values = None
    for dim in dimensions:
        rows = db.conn.execute(
            f"SELECT {dim} FROM paper_signatures WHERE {dim} IS NOT NULL"
        ).fetchall()
        values = [r[dim] for r in rows if r[dim] is not None]
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
    _log.info(f"Baselines updated: {len(dimensions)} dimensions, {len(values) if values else 0} samples")


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
    if db.count() < 10:
        _debug("Too few QAs for cross-paper check, skipping")
        db.close()
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
    db.close()
