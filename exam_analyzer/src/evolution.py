"""Self-evolving loop: KP drift detection, re-review, outlier QA detection.

Observes changes since last analysis, generates improvement proposals,
and auto-accepts low-risk refinements.
"""
import numpy as np

from .adversarial_refiner import refine_kp
from .deepseek_client import call_flash
from .embedding_cluster import detect_content_lang, _get_model, TOPIC_EMBED_MODEL
from .constants import SQLITE_PARAM_CHUNK
from .knowledge_base import QADatabase
from .diagnostics import run_phase2_cycle, apply_student_feedback


def run_evolution_cycle(db: QADatabase, client, debug) -> None:
    """Run self-evolution: detect drift, trigger re-review for degraded KPs."""
    # Phase 2: Fragment migration + topic stats
    try:
        result = run_phase2_cycle(db, debug_cb=debug)
        if result.get("migrated", 0) > 0:
            from .error_utils import log_info
            log_info(debug, "Evolution", f"{result['migrated']} fragments migrated")
    except Exception as e:
        from .error_utils import log_exception
        log_exception(debug, "Phase2 cycle", "", e)

    # Generate KP for stable/forming topics
    try:
        _generate_kp_for_stable_topics(db, client, debug)
    except Exception as e:
        from .error_utils import log_exception
        log_exception(debug, "KP generation", "stable_topics", e)

    kps = db.conn.execute("SELECT * FROM knowledge_points").fetchall()
    if not kps:
        return

    # Fix inconsistent attempt counters
    fixed = db.qa.fix_inconsistent_counters()
    if fixed:
        from .error_utils import log_info
        log_info(debug, "Evolution", f"fixed {fixed} inconsistent attempt counters")

    # Re-review disputed KPs
    disputed = [dict(k) for k in kps if k["quality"] == "disputed"]
    if disputed:
        from .error_utils import log_info
        log_info(debug, "Evolution", f"{len(disputed)} disputed KPs - queuing for re-review")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        targets = disputed[:3]
        with ThreadPoolExecutor(max_workers=min(len(targets), 3)) as executor:
            futures = {executor.submit(refine_kp, db, kp["id"], client, debug): kp["id"]
                       for kp in targets}
            for future in as_completed(futures):
                kp_id = futures[future]
                try:
                    result = future.result()
                    new_quality = result.get("quality", "unknown")
                    db.analysis.record_evolution(
                        kp_id=kp_id,
                        trigger_type="auto_re-review",
                        trigger_detail="disputed KP re-evaluated in evolution cycle",
                        old_state="quality=disputed",
                        new_state=f"quality={new_quality}",
                        outcome="completed",
                    )
                except Exception as e:
                    from .error_utils import log_exception
                    log_exception(debug, "Evolution re-review", f"kp={kp_id}", e)

    # Detect KPs with QA growth since last review
    for kp in kps:
        if kp["quality"] in ("draft", "accepted", "disputed"):
            current_members = db.kp.count_members(kp["id"])
            prev_evidence = kp.get("evidence_count", 0) or 0
            if current_members > prev_evidence and current_members >= 5:
                growth = current_members - prev_evidence
                if growth >= 3 or current_members >= prev_evidence * 1.5:
                    db.analysis.record_evolution(
                        kp_id=kp["id"],
                        trigger_type="evidence_growth",
                        trigger_detail=f"QA members: {prev_evidence} -> {current_members} (+{growth})",
                        old_state=f"evidence_count={prev_evidence}",
                        new_state=f"evidence_count={current_members}",
                        outcome="queued",
                    )
                    with db.transaction():
                        db.kp.update_evidence_count(kp["id"], current_members)

    pending_count = len(db.analysis.get_pending_evolutions())
    if pending_count:
        from .error_utils import log_info
        log_info(debug, "Evolution", f"{pending_count} improvement proposals queued")

    # Apply student feedback
    try:
        apply_student_feedback(db)
    except Exception as e:
        from .error_utils import log_exception
        log_exception(debug, "Student feedback", "", e)

    # Detect outlier QAs
    try:
        outlier_count = _detect_outlier_qas(db, debug)
        if outlier_count:
            from .error_utils import log_info
            log_info(debug, "Evolution", f"{outlier_count} outlier QAs flagged for review")
    except Exception as e:
        from .error_utils import log_exception
        log_exception(debug, "Outlier detection", "", e)


def _generate_kp_for_stable_topics(db: QADatabase, client, debug) -> None:
    """Generate KP text for topics that reached 'forming' quality via Flash."""
    topics = db.conn.execute(
        "SELECT * FROM dynamic_topics WHERE quality='forming'"
    ).fetchall()
    if not topics:
        return

    from .error_utils import log_info
    log_info(debug, "Phase 2", f"generating KP for {len(topics)} forming topics")

    for topic in topics:
        topic_id = topic["topic_id"]
        frags = db.topic.get_fragments(topic_id)
        if not frags:
            continue

        capped = frags[:SQLITE_PARAM_CHUNK]
        frag_texts = db.conn.execute(
            "SELECT point_text FROM ms_fragments WHERE point_id IN (%s)" % (
                ",".join("?" * len(capped))),
            capped
        ).fetchall()
        texts = [r["point_text"] for r in frag_texts]
        combined = "\n".join(f"- {t}" for t in texts[:20])

        lang = detect_content_lang(combined[:2000])

        if lang == 'en':
            sys = ("You are a knowledge distillation expert. These MS scoring points "
                   "were found by the system to help answer the same set of questions. "
                   "Name and explain the concept they describe. Output JSON.")
            usr = (f"MS scoring points (all test the same concept):\n{combined}\n\n"
                   "1. Name this concept (1 sentence)\n"
                   "2. Explain key details (2-3 sentences, based ONLY on the points above)\n"
                   'Return JSON: {"concept": "...", "detail": "..."}')
        else:
            sys = ("这些MS得分点被系统发现可互相帮助答题。请命名并解释它们描述的概念。Output JSON。")
            usr = (f"MS得分点（考查同一概念）:\n{combined}\n\n"
                   "1. 命名此概念（一句话）\n"
                   "2. 解释关键细节（2-3句，仅基于以上得分点）\n"
                   '返回 JSON: {"concept": "...", "detail": "..."}')

        messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
        try:
            result, _ = call_flash(client, messages, max_retries=1, debug=debug)
            concept = result.get("concept", topic["name"]) if isinstance(result, dict) else topic["name"]
            detail = result.get("detail", "") if isinstance(result, dict) else ""
        except Exception as e:
            from .error_utils import log_exception
            log_exception(debug, "KP generation", f"topic={topic_id}", e)
            concept = topic["name"] + "[auto]"
            detail = ""

        db.topic.set_kp(topic_id, concept, detail)
        from .error_utils import log_info
        log_info(debug, "KP generated", f"[{topic_id}] {concept[:80]}")


def _detect_outlier_qas(db: QADatabase, debug) -> int:
    """Detect QAs drifting from their topic centroid — potential new topics.

    For each topic with >= 5 QAs, computes the centroid of answer embeddings
    and flags QAs whose cosine distance to centroid exceeds 2 standard deviations.
    """
    groups = db.get_topic_groups()
    if not groups:
        return 0

    model = _get_model(TOPIC_EMBED_MODEL)
    flagged = 0

    for topic, qas in groups.items():
        if not topic or topic == "(uncategorized)" or len(qas) < 5:
            continue

        answer_texts = [qa["answer_text"] for qa in qas]
        if not any(answer_texts):
            continue

        try:
            vecs = model.encode(answer_texts, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            from .error_utils import log_exception
            log_exception(debug, "Outlier embedding", f"topic={topic}", e)
            continue

        centroid = np.mean(vecs, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)

        distances = [float(1.0 - np.dot(vecs[i], centroid)) for i in range(len(vecs))]
        if len(distances) < 3:
            continue

        mean_dist = sum(distances) / len(distances)
        stdev = (sum((d - mean_dist) ** 2 for d in distances) / len(distances)) ** 0.5
        if stdev < 0.01:
            continue

        for i, d in enumerate(distances):
            # 离群判定: 余弦距离>均值+2σ (离群倍数) 且 >0.25 (离群下限, 避免噪声)
            if d > mean_dist + 2.0 * stdev and d > 0.25:
                qa = qas[i]
                with db.transaction():
                    db.qa.set_failure_reason(qa["id"],
                        f"outlier: dist={d:.3f} from topic '{topic}' centroid")
                flagged += 1

    return flagged
