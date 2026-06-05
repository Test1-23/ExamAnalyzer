import json
import os
import re
from collections import deque

from ..deepseek_client import create_client, call_flash
from ..knowledge_base import QADatabase
from ..embedding_cluster import _get_model, detect_content_lang, TOPIC_EMBED_MODEL
from ..models import VerbPatternSpec, DependencySpec
from ..prompt_factory import PromptType, PromptBuilder
from ..logger import get_logger
from ..utils import get_worker_limit

_log = get_logger()

# ============================================================
# Task 3: Difficulty Assessment
# ============================================================

def assess_difficulty(db: QADatabase, client, verb_data: dict, debug, progress_cb=None) -> dict:
    """Assess difficulty per QA using hybrid Flash-calibrated signal approach.

    Returns: {qa_id: difficulty_label, ...}
    """
    _log.info("Offline Task 3: Difficulty assessment starting")
    debug("Task 3: Assessing question difficulty...")

    qas = db.get_all()
    if not qas:
        _log.info("Task 3: No QAs, skipping")
        return {}

    current_count = len(qas)
    prev = db.analysis.get_checkpoint("difficulty")
    if prev and prev.get("qa_count_at_run") == current_count:
        new_qas = [qa for qa in qas if not qa.get("difficulty_estimate", "")]
        if not new_qas:
            _log.info("Task 3: Already complete, skipping")
            return {}

    if progress_cb:
        progress_cb(0, "Sampling QAs for difficulty calibration...")

    # Phase 1: Flash benchmark on representative QAs
    anchors, anchor_labels = _phase1_difficulty_benchmark(db, qas, client, debug)

    if progress_cb:
        progress_cb(30, "Calibrating difficulty signals...")

    # Phase 2: Signal calibration
    boundaries = _phase2_calibrate_signals(db, qas, anchors, anchor_labels, verb_data, debug)

    if progress_cb:
        progress_cb(60, "Classifying remaining QAs...")

    # Phase 3: Classify all QAs + boundary Flash confirmation
    # Pre-load all QAs for verb percentile computation (avoids O(n^2) db.get_all() calls)
    all_qas_cache = qas
    _phase3_classify_and_confirm(db, all_qas_cache, boundaries, client, verb_data, debug)

    if progress_cb:
        progress_cb(90, "Aggregating topic difficulty...")

    # Phase 4: Topic-level aggregation
    _phase4_topic_aggregation(db)

    db.analysis.checkpoint("difficulty", current_count, "completed")

    # Difficulty distribution summary
    diff_counts = db.conn.execute(
        "SELECT difficulty_estimate, COUNT(*) as cnt FROM qa_pairs "
        "WHERE difficulty_estimate != '' GROUP BY difficulty_estimate"
    ).fetchall()
    diff_map = {r["difficulty_estimate"]: r["cnt"] for r in diff_counts}
    total_diff = sum(diff_map.values())
    if total_diff > 0:
        b = diff_map.get("basic", 0); i = diff_map.get("intermediate", 0); a = diff_map.get("advanced", 0)
        from ..error_utils import log_info
        log_info(debug, "QA difficulty", f"basic={b}, intermediate={i}, advanced={a} "
                 f"({b/total_diff*100:.0f}/{i/total_diff*100:.0f}/{a/total_diff*100:.0f})")
    topic_rows = db.conn.execute(
        "SELECT COUNT(*) as cnt FROM topic_difficulty"
    ).fetchone()
    mixed_rows = db.conn.execute(
        "SELECT COUNT(*) as cnt FROM topic_difficulty WHERE mode_difficulty = 'mixed'"
    ).fetchone()
    if topic_rows:
        from ..error_utils import log_info
        log_info(debug, "Topic difficulty", f"{topic_rows['cnt']} topics assessed, "
                 f"{mixed_rows['cnt']} mixed" if mixed_rows and mixed_rows['cnt'] > 0 else "")

    _log.info("Task 3: Complete")
    return {qa["id"]: qa.get("difficulty_estimate", "") for qa in db.get_all()}


def _phase1_difficulty_benchmark(db, qas, client, debug):
    """Flash rates ~30 representative QAs to establish difficulty baseline."""
    # Pick representative QAs: is_representative first, then highest Beta weight
    weights = db.qa.get_all_weights()
    # Select 1 QA per topic (highest weight), then sample up to 30
    groups = db.get_topic_groups()
    candidates = []
    for topic, t_qas in groups.items():
        if not topic or topic == "(uncategorized)":
            continue
        # Prefer representative QA
        rep = [qa for qa in t_qas if qa.get("is_representative")]
        if rep:
            candidates.append(rep[0])
        else:
            best = max(t_qas, key=lambda qa: weights.get(qa["id"], {}).get("mean", 0.5))
            candidates.append(best)

    # Sample uniformly: sort by miss_rate span if available, then take every Nth
    if len(candidates) > 30:
        step = max(1, len(candidates) // 30)
        candidates = candidates[::step][:30]

    labels = {}
    batch_size = 15  # 2 batches for 30 QAs

    for b in range(0, len(candidates), batch_size):
        batch = candidates[b:b+batch_size]
        lang = detect_content_lang(" ".join(qa["question_text"] for qa in batch))

        qa_block = ""
        if lang == 'en':
            qa_block += "Rate each question as basic, intermediate, or advanced:\n\n"
        else:
            qa_block += "评估每道题的难度：\n\n"
        for i, qa in enumerate(batch):
            qa_block += f"[{i}] Q: {qa['question_text']}\nA: {qa['answer_text'][:300]}\n\n"

        messages = PromptBuilder.build(PromptType.DIFFICULTY, lang=lang, qa_block=qa_block)
        try:
            result, _ = call_flash(client, messages, max_retries=1, debug=debug)
            ratings = result.get("ratings", []) if isinstance(result, dict) else []
        except Exception as e:
            from ..error_utils import log_exception
            log_exception(debug, "Difficulty benchmark", f"batch={b}", e)
            ratings = []

        for r in ratings:
            idx = r.get("question_index", -1)
            if 0 <= idx < len(batch):
                labels[batch[idx]["id"]] = r.get("difficulty", "intermediate")

    from ..error_utils import log_info
    log_info(debug, "Difficulty benchmark", f"{len(labels)} QAs rated by Flash")
    return candidates, labels


def _phase2_calibrate_signals(db, qas, anchors, anchor_labels, verb_data, debug):
    """Use Flash-anchored QAs to find difficulty thresholds for each signal.
    Uses effective_miss_rate (knowledge_gap + insufficient_detail only), not raw miss_rate."""

    def _compute_effective_miss_rate(qa):
        """Only knowledge_gap + insufficient_detail contribute to difficulty.
        misinterpretation and retrieval_quality are excluded."""
        return db.qa.get_effective_miss_rate(qa["id"])

    def _compute_cross_topic_degree(qa):
        topic = qa.get("topic", "")
        if not topic:
            return 0
        rows = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM topic_links WHERE src_topic = ? OR dst_topic = ?",
            (topic, topic),
        ).fetchone()
        return min(rows["cnt"] / 10, 1.0) if rows else 0

    def _compute_verb_length_percentile(qa):
        verb = qa.get("command_verb", "")
        if not verb or not verb_data:
            return None
        ans_len = len(qa.get("answer_text", ""))
        # Use pre-loaded qas parameter instead of db.get_all()
        all_lens = [
            len(q.get("answer_text", ""))
            for q in qas
            if q.get("command_verb", "") == verb
        ]
        if len(all_lens) < 2:
            return None
        rank = sum(1 for l in all_lens if l < ans_len)
        return rank / len(all_lens)

    # Compute signals for anchor QAs
    anchor_signals = {}
    for qa in anchors:
        if qa["id"] in anchor_labels:
            anchor_signals[qa["id"]] = {
                "effective_miss_rate": _compute_effective_miss_rate(qa),
                "cross_topic": _compute_cross_topic_degree(qa),
                "verb_percentile": _compute_verb_length_percentile(qa),
                "label": anchor_labels[qa["id"]],
            }

    # Group by difficulty label, compute median per signal
    groups = {"basic": [], "intermediate": [], "advanced": []}
    for qa_id, sigs in anchor_signals.items():
        groups[sigs["label"]].append(sigs)

    boundaries = {}
    for signal in ["effective_miss_rate", "cross_topic", "verb_percentile"]:
        medians = {}
        for level in ["basic", "intermediate", "advanced"]:
            vals = [s[signal] for s in groups[level] if s[signal] is not None]
            if vals:
                medians[level] = sorted(vals)[len(vals) // 2]

        if "basic" in medians and "intermediate" in medians:
            boundaries[f"{signal}_basic_inter"] = (medians["basic"] + medians["intermediate"]) / 2
        if "intermediate" in medians and "advanced" in medians:
            boundaries[f"{signal}_inter_adv"] = (medians["intermediate"] + medians["advanced"]) / 2

    from ..error_utils import log_info
    log_info(debug, "Difficulty boundaries", f"{json.dumps({k: round(v, 3) for k, v in boundaries.items()})}")
    return boundaries


def _get_signal(qa, signal_name, db, all_qas):
    """Compute a difficulty signal for a single QA.

    Extracted from _phase3_classify_and_confirm for testability.
    all_qas: pre-loaded list of all QAs (avoids repeated db.get_all() calls).
    """
    if signal_name == "effective_miss_rate":
        return db.qa.get_effective_miss_rate(qa["id"])
    elif signal_name == "cross_topic":
        topic = qa.get("topic", "")
        if not topic:
            return 0
        rows = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM topic_links WHERE src_topic = ? OR dst_topic = ?",
            (topic, topic),
        ).fetchone()
        return min(rows["cnt"] / 10, 1.0) if rows else 0
    elif signal_name == "verb_percentile":
        verb = qa.get("command_verb", "")
        if not verb:
            return None
        ans_len = len(qa.get("answer_text", ""))
        all_lens = [len(q.get("answer_text", "")) for q in all_qas if q.get("command_verb", "") == verb]
        if len(all_lens) < 2:
            return None
        rank = sum(1 for l in all_lens if l < ans_len)
        return rank / len(all_lens)
    return None


def _evaluate_signal(sig_val, bi_threshold, ia_threshold, margin):
    """Map a single signal value against calibrated boundaries.

    Returns (vote_label, is_boundary).  Pure function — independently testable.
    bi_threshold/ia_threshold can be None if that boundary is not calibrated.
    """
    # bi_threshold exists → classify basic vs intermediate/advanced
    if bi_threshold is not None:
        if sig_val < bi_threshold * (1 - margin):
            return "basic", False
        if sig_val < bi_threshold * (1 + margin):
            return "basic", True
        # Above bi_threshold — classify intermediate vs advanced
        if ia_threshold is not None:
            if sig_val < ia_threshold * (1 - margin):
                return "intermediate", False
            if sig_val < ia_threshold * (1 + margin):
                return "intermediate", True
            return "advanced", False
        return "intermediate", False

    # bi_threshold missing but ia_threshold exists → classify intermediate vs advanced
    if ia_threshold is not None:
        if sig_val < ia_threshold * (1 - margin):
            return "intermediate", False
        if sig_val < ia_threshold * (1 + margin):
            return "intermediate", True
        return "advanced", False

    # Neither boundary defined
    return "intermediate", False


def _classify_difficulty(signals, boundaries, margin=0.10):
    """Classify a QA's difficulty from signal values and calibrated boundaries.

    Returns (label, is_boundary).  Delegates per-signal evaluation to
    _evaluate_signal (pure function, independently testable).
    """
    votes = {"basic": 0, "intermediate": 0, "advanced": 0}
    is_boundary = False

    for sig_name, sig_val in signals.items():
        if sig_val is None:
            continue
        vote, boundary = _evaluate_signal(
            sig_val,
            boundaries.get(f"{sig_name}_basic_inter"),
            boundaries.get(f"{sig_name}_inter_adv"),
            margin,
        )
        votes[vote] += 1
        if boundary:
            is_boundary = True

    if votes["advanced"] > 0:
        return "advanced", is_boundary
    elif votes["intermediate"] > 0:
        return "intermediate", is_boundary
    elif votes["basic"] > 0:
        return "basic", is_boundary
    return "intermediate", False


def _phase3_classify_and_confirm(db, qas, boundaries, client, verb_data, debug):
    """Classify all QAs using calibrated signals. Flash confirms boundary cases.
    qas: pre-loaded list of all QAs (passed in to avoid repeated db.get_all() calls)."""
    boundary_cases = []

    classification_method = {}
    for qa in qas:
        sigs = {
            "effective_miss_rate": _get_signal(qa, "effective_miss_rate", db, qas),
            "cross_topic": _get_signal(qa, "cross_topic", db, qas),
            "verb_percentile": _get_signal(qa, "verb_percentile", db, qas),
        }
        has_signals = any(v is not None for v in sigs.values())

        if not has_signals:
            # No signals available → Flash later
            boundary_cases.append(qa)
            classification_method[qa["id"]] = "flash_only"
            continue

        label, is_boundary = _classify_difficulty(sigs, boundaries)

        if is_boundary:
            boundary_cases.append(qa)
        else:
            with db.transaction():
                db.conn.execute(
                    "UPDATE qa_pairs SET difficulty_estimate = ? WHERE id = ?",
                    (label, qa["id"]),
                )
            classification_method[qa["id"]] = "hybrid"

    # Flash confirm boundary cases
    if boundary_cases:
        from ..error_utils import log_info
        log_info(debug, "Flash confirm difficulty", f"{len(boundary_cases)} boundary cases...")
        for b in range(0, len(boundary_cases), 10):
            batch = boundary_cases[b:b+10]
            lang = detect_content_lang(" ".join(qa["question_text"] for qa in batch))

            qa_block = ""
            if lang == 'en':
                qa_block += "Rate:\n\n"
            else:
                qa_block += "评估:\n\n"
            for i, qa in enumerate(batch):
                qa_block += f"[{i}] Q: {qa['question_text']}\nA: {qa['answer_text'][:300]}\n\n"

            messages = PromptBuilder.build(PromptType.DIFFICULTY, lang=lang, qa_block=qa_block)
            try:
                result, _ = call_flash(client, messages, max_retries=1, debug=debug)
                ratings = result.get("ratings", []) if isinstance(result, dict) else []
            except Exception as e:
                from ..error_utils import log_exception
                log_exception(debug, "Boundary difficulty", f"batch={b}", e)
                ratings = []

            with db.transaction():
                for r in ratings:
                    idx = r.get("question_index", -1)
                    if 0 <= idx < len(batch):
                        qa = batch[idx]
                        label = r.get("difficulty", "intermediate")
                        db.conn.execute(
                            "UPDATE qa_pairs SET difficulty_estimate = ? WHERE id = ?",
                            (label, qa["id"]),
                        )
                        classification_method[qa["id"]] = "flash_boundary"

    from ..error_utils import log_info
    log_info(debug, "Difficulty", f"{len(qas)} QAs classified "
             f"(hybrid={sum(1 for v in classification_method.values() if v in ('hybrid', 'flash_boundary'))}, "
             f"flash_only={sum(1 for v in classification_method.values() if v == 'flash_only')})")


def _phase4_topic_aggregation(db):
    """Aggregate QA-level difficulty to topic level."""
    groups = db.get_topic_groups()
    for topic, qas in groups.items():
        if not topic or topic == "(uncategorized)":
            continue
        counts = {"basic": 0, "intermediate": 0, "advanced": 0}
        miss_rates = []
        for qa in qas:
            diff = qa.get("difficulty_estimate", "")
            if diff in counts:
                counts[diff] += 1
            row = db.conn.execute(
                "SELECT missed_count, covered_count FROM question_feedback WHERE qa_id = ? AND topic_match = 1",
                (qa["id"],),
            ).fetchone()
            if row and (row["missed_count"] + row["covered_count"]) > 0:
                miss_rates.append(row["missed_count"] / (row["missed_count"] + row["covered_count"]))

        mode = max(counts, key=counts.get) if any(counts.values()) else "intermediate"
        spread = (counts["basic"] > 0 and counts["advanced"] > 0)

        db.topic.upsert_difficulty(
            topic=topic,
            qa_count=len(qas),
            basic_count=counts["basic"],
            intermediate_count=counts["intermediate"],
            advanced_count=counts["advanced"],
            mode_difficulty="mixed" if spread else mode,
            avg_miss_rate=round(sum(miss_rates) / len(miss_rates), 2) if miss_rates else None,
            difficulty_spread=spread,
            assessment_method="hybrid" if any(qa.get("difficulty_estimate") for qa in qas) else "flash_only",
        )


