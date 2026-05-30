"""Offline analysis pipeline: command-verb patterns, difficulty assessment, dependency discovery.

Runs AFTER pipeline.run_pipeline() completes. All three tasks are batch offline processing
that does not touch the real-time chat path.

Execution order: Task 2 (verbs) -> Task 3 (difficulty) -> Task 1 (dependencies)
"""

import json
import os
import re
from collections import deque

from .deepseek_client import create_client, call_flash
from .knowledge_base import QADatabase
from .embedding_cluster import _get_model, detect_content_lang, TOPIC_EMBED_MODEL
from .models import VerbPatternSpec, DependencySpec
from .prompt_factory import VERB_PATTERN_SUMMARY, DIFFICULTY_RATE, DEPENDENCY_VALIDATE
from .logger import get_logger
from .utils import get_worker_limit

_log = get_logger()


# ============================================================
# Task 2: Command Verb Analysis
# ============================================================

def analyze_command_verbs(db: QADatabase, client, debug_cb, progress_cb=None) -> dict:
    """Extract command verbs from QA question_text, then analyze answer patterns per verb.

    Returns: {verb: {sample_count, avg_answer_length, pattern_summary, ...}, ...}
    """
    _log.info("Offline Task 2: Command verb analysis starting")
    debug_cb("Task 2: Extracting command verbs...")

    qas = db.get_all()
    if not qas:
        _log.info("Task 2: No QAs, skipping")
        return {}

    # Check incremental: only process QAs without command_verb
    new_qas = [qa for qa in qas if not qa.get("command_verb", "")]
    current_count = len(qas)
    prev = db.get_checkpoint("command_verbs")
    if prev and prev.get("qa_count_at_run") == current_count and not new_qas:
        _log.info("Task 2: Already complete, skipping")
        return {}

    if not new_qas:
        _log.info("Task 2: All QAs already have verb annotations, skipping extraction")
    else:
        _phase1_extract_verbs(new_qas, db, client, debug_cb, progress_cb)

    # Phase 2: Aggregate statistics per verb
    debug_cb("Task 2: Aggregating answer patterns...")
    qas = db.get_all()  # reload with updated command_verb
    verb_groups = _group_qas_by_verb(qas)
    verb_stats = _compute_verb_stats(verb_groups, db)

    # Phase 3: Flash pattern summary per verb
    debug_cb("Task 2: Summarizing verb patterns...")
    if progress_cb:
        progress_cb(0, "Analyzing command verb patterns...")
    _phase3_summarize_patterns(verb_groups, verb_stats, db, client, debug_cb)

    # Phase 4: Assign verb families
    _phase4_assign_families(db)

    db.checkpoint("command_verbs", current_count, "completed")
    # Verb coverage summary
    total = len(db.get_all())
    annotated = len([q for q in db.get_all() if q.get("command_verb", "")])
    unknown = len([q for q in db.get_all() if q.get("command_verb", "") == "unknown"])
    debug_cb(f"  [Verbs] Coverage: {annotated}/{total} QAs annotated ({annotated/total*100:.0f}%)" if total else "")
    if unknown:
        debug_cb(f"  [Verbs] Unknown: {unknown} QAs — Flash couldn't determine verb")
    # Top 10 verbs
    verb_counts = db.conn.execute(
        "SELECT command_verb, COUNT(*) as cnt FROM qa_pairs "
        "WHERE command_verb != '' AND command_verb != 'unknown' "
        "GROUP BY command_verb ORDER BY cnt DESC LIMIT 10"
    ).fetchall()
    if verb_counts:
        vstr = ", ".join(f"{r['command_verb']}={r['cnt']}" for r in verb_counts)
        debug_cb(f"  [Verbs] Distribution: {vstr}")

    _log.info(f"Task 2: Complete. {len(verb_stats)} verbs analyzed")
    return verb_stats


def _phase1_extract_verbs(qas, db, client, debug_cb, progress_cb):
    """Batch Flash extraction of command verbs from question_text. Parallelized."""
    batch_size = 20
    batches = [qas[i:i+batch_size] for i in range(0, len(qas), batch_size)]

    def _extract_batch(batch):
        lang = detect_content_lang(" ".join(qa["question_text"] for qa in batch))
        if lang == 'en':
            sys = (
                "You are an exam question analyst. Extract the command verb from each question. "
                "Standardize: state, explain, describe, compare, calculate, evaluate, "
                "identify, discuss, draw, convert, show, define, list, suggest, justify, outline. "
                "Domain verbs (normalize, compile) keep original. Compound instructions→primary+secondary. "
                'Implied ("What is X?")→infer. Output JSON.'
            )
            usr = "Questions:\n"
            for i, qa in enumerate(batch):
                usr += f"[{i}] {qa['question_text']}\n"
            usr += ('\nReturn: {"verbs": [{"question_index": 0, "primary_verb": "state", '
                     '"secondary_verb": null, "inferred": false}, ...]}')
        else:
            sys = '提取指令动词。复合→primary+secondary。隐含→推断。Output JSON。'
            usr = "题目列表：\n"
            for i, qa in enumerate(batch):
                usr += f"[{i}] {qa['question_text']}\n"
            usr += ('\n返回: {"verbs": [{"question_index": 0, "primary_verb": "state", '
                     '"secondary_verb": null, "inferred": false}, ...]}')

        messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
        flash_ok = True
        try:
            result = call_flash(client, messages, max_retries=1, debug_callback=debug_cb)
            verb_list = result.get("verbs", []) if isinstance(result, dict) else []
        except Exception as e:
            debug_cb(f"  Verb extraction batch failed: {e}")
            verb_list = []
            flash_ok = False

        with db.transaction():
            for v in verb_list:
                idx = v.get("question_index", -1)
                if 0 <= idx < len(batch):
                    qa = batch[idx]
                    db.conn.execute(
                        "UPDATE qa_pairs SET command_verb=?, command_verb_secondary=?, "
                        "command_verb_inferred=? WHERE id=?",
                        (v.get("primary_verb", ""), v.get("secondary_verb", ""),
                         int(v.get("inferred", False)), qa["id"]),
                    )
            if flash_ok:
                returned = {v.get("question_index") for v in verb_list}
                for i, qa in enumerate(batch):
                    if i not in returned:
                        db.conn.execute(
                            "UPDATE qa_pairs SET command_verb='unknown' WHERE id=?", (qa["id"],),
                        )

    if batches:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_w = get_worker_limit(len(batches), api_heavy=True)
        with ThreadPoolExecutor(max_workers=max_w) as executor:
            futures = [executor.submit(_extract_batch, b) for b in batches]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    debug_cb(f"  Verb extraction thread failed: {e}")

    if progress_cb:
        progress_cb(100, "Verb extraction complete")


def _group_qas_by_verb(qas):
    """Group QAs by primary command_verb."""
    groups: dict[str, list] = {}
    for qa in qas:
        verb = qa.get("command_verb", "") or "unknown"
        groups.setdefault(verb, []).append(qa)
    return groups


def _compute_verb_stats(verb_groups, db):
    """Compute per-verb statistics without API calls."""
    stats = {}
    for verb, qas in verb_groups.items():
        if not qas:
            continue
        n = len(qas)
        ans_lens = [len(qa["answer_text"]) for qa in qas if qa.get("answer_text")]
        avg_len = sum(ans_lens) / len(ans_lens) if ans_lens else 0
        sorted_lens = sorted(ans_lens)
        median_len = sorted_lens[len(sorted_lens) // 2] if sorted_lens else 0

        # Bullet detection (multiple formats)
        bullet_count = 0
        bullet_line_total = 0
        for qa in qas:
            ans = qa.get("answer_text", "")
            lines = [l.strip() for l in ans.split("\n") if l.strip()]
            bullet_lines = []
            for l in lines:
                if not l:
                    continue
                # Dash/unicode bullet: "- text", "• text", "‣ text"
                if l[0] in ('-', '•', '‣'):
                    bullet_lines.append(l)
                    continue
                # Numbered: "1.", "10.", "2)", "12)"
                if l[0].isdigit():
                    dot_or_paren = l.find('.')
                    if dot_or_paren == -1:
                        dot_or_paren = l.find(')')
                    if dot_or_paren > 0 and all(c.isdigit() for c in l[:dot_or_paren]):
                        bullet_lines.append(l)
                        continue
                # Parenthesized: "(a)", "(1)"
                if l[0] == '(' and len(l) > 2 and l[2] == ')':
                    bullet_lines.append(l)
            if bullet_lines:
                bullet_count += 1
                bullet_line_total += len(bullet_lines)

        bullet_ratio = bullet_count / n if n else 0
        avg_bullet = bullet_line_total / bullet_count if bullet_count else 0

        # Avg miss rate from question_feedback (topic_match=1 only)
        miss_rates = []
        qa_ids = [qa["id"] for qa in qas]
        if qa_ids:
            from .constants import SQLITE_PARAM_CHUNK
            for i in range(0, len(qa_ids), SQLITE_PARAM_CHUNK):
                chunk = qa_ids[i:i + SQLITE_PARAM_CHUNK]
                placeholders = ",".join("?" * len(chunk))
                rows = db.conn.execute(
                    f"""SELECT missed_count, covered_count FROM question_feedback
                        WHERE qa_id IN ({placeholders}) AND topic_match = 1""",
                    chunk,
                ).fetchall()
                for r in rows:
                    total = r["missed_count"] + r["covered_count"]
                    if total > 0:
                        miss_rates.append(r["missed_count"] / total)

        avg_miss_rate = sum(miss_rates) / len(miss_rates) if miss_rates else None

        stats[verb] = {
            "sample_count": n,
            "avg_answer_length": round(avg_len, 1),
            "median_answer_length": round(median_len, 1),
            "bullet_ratio": round(bullet_ratio, 2),
            "avg_bullet_count": round(avg_bullet, 1),
            "avg_miss_rate": round(avg_miss_rate, 2) if avg_miss_rate is not None else None,
        }
    return stats


def _phase3_summarize_patterns(verb_groups, verb_stats, db, client, debug_cb):
    """Flash summarizes answer patterns for each verb with >= 3 samples. Parallelized."""
    verbs_to_process = [(v, qas) for v, qas in verb_groups.items()
                        if verb_stats.get(v, {}).get("sample_count", 0) >= 3]
    if not verbs_to_process:
        return

    def _summarize_one(verb, qas):
        stat = verb_stats[verb]
        lang = detect_content_lang(" ".join(qa["question_text"] for qa in qas[:5]))
        qa_texts = ""
        for i, qa in enumerate(qas[:15]):
            qa_texts += f"Q{i+1}: {qa['question_text']}\nA: {qa['answer_text'][:300]}\n\n"

        messages = VERB_PATTERN_SUMMARY.build(
            lang=lang, verb=verb, sample_count=stat["sample_count"],
            avg_answer_length=stat["avg_answer_length"],
            bullet_pct=f"{stat['bullet_ratio']*100:.0f}%",
            avg_miss_rate=str(stat.get("avg_miss_rate", "N/A")),
            qa_texts=qa_texts,
        )
        try:
            result = call_flash(client, messages, max_retries=1, debug_callback=debug_cb)
            summary = result.get("pattern_summary", "") if isinstance(result, dict) else ""
        except Exception as e:
            debug_cb(f"  Pattern summary failed for '{verb}': {e}")
            return None

        # Topic-level variance
        topic_groups = {}
        for qa in qas:
            t = qa.get("topic", "")
            if t:
                topic_groups.setdefault(t, []).append(qa)
        topic_specific = {}
        if len(topic_groups) >= 2:
            topic_lens = {t: sum(len(qa["answer_text"]) for qa in g) / len(g) for t, g in topic_groups.items()}
            vals = list(topic_lens.values())
            if len(vals) >= 2:
                mean_val = sum(vals) / len(vals)
                variance = sum((v - mean_val) ** 2 for v in vals) / len(vals)
                if variance > (mean_val * 0.3):
                    for t, avg_len in topic_lens.items():
                        if abs(avg_len - mean_val) > mean_val * 0.2:
                            topic_specific[t] = (
                                f"Answers in this topic are {'longer' if avg_len > mean_val else 'shorter'} "
                                f"than typical for '{verb}' questions ({avg_len:.0f} vs {mean_val:.0f} chars)."
                            )

        db.upsert_verb_pattern(VerbPatternSpec(
            verb=verb, sample_count=stat["sample_count"],
            avg_answer_length=stat["avg_answer_length"],
            median_answer_length=stat["median_answer_length"],
            bullet_ratio=stat["bullet_ratio"], avg_bullet_count=stat["avg_bullet_count"],
            avg_miss_rate=stat["avg_miss_rate"], pattern_summary=summary,
            topic_specific_patterns=json.dumps(topic_specific) if topic_specific else "",
        ))
        debug_cb(f"  Verb '{verb}': {stat['sample_count']} samples, pattern generated")
        return verb

    from concurrent.futures import ThreadPoolExecutor, as_completed
    max_w = get_worker_limit(len(verbs_to_process), api_heavy=True)
    with ThreadPoolExecutor(max_workers=max_w) as executor:
        futures = {executor.submit(_summarize_one, v, q): v for v, q in verbs_to_process}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                debug_cb(f"  Verb pattern thread failed: {e}")


def _phase4_assign_families(db):
    """Assign low-sample verbs to families based on semantic similarity."""
    patterns = db.get_verb_patterns()
    if not patterns:
        return

    families = {
        "short-recall": ["state", "define", "identify", "name", "give", "list", "what"],
        "elaboration": ["explain", "describe", "discuss", "outline", "justify"],
        "comparative-analysis": ["compare", "evaluate", "contrast", "assess"],
        "procedural-computation": ["calculate", "convert", "show", "derive", "compute"],
        "visual-representation": ["draw", "sketch", "complete", "label"],
    }

    for p in patterns:
        verb = p["verb"]
        if p.get("verb_family", ""):
            continue  # already assigned
        assigned = None
        for family, members in families.items():
            if verb.lower() in members:
                assigned = family
                break
        if not assigned:
            # Domain-specific verb: keep as its own family
            assigned = verb.lower()
        with db.transaction():
            db.conn.execute(
                "UPDATE command_verb_patterns SET verb_family = ? WHERE verb = ?",
                (assigned, verb),
            )


# ============================================================
# Task 3: Difficulty Assessment
# ============================================================

def assess_difficulty(db: QADatabase, client, verb_data: dict, debug_cb, progress_cb=None) -> dict:
    """Assess difficulty per QA using hybrid Flash-calibrated signal approach.

    Returns: {qa_id: difficulty_label, ...}
    """
    _log.info("Offline Task 3: Difficulty assessment starting")
    debug_cb("Task 3: Assessing question difficulty...")

    qas = db.get_all()
    if not qas:
        _log.info("Task 3: No QAs, skipping")
        return {}

    current_count = len(qas)
    prev = db.get_checkpoint("difficulty")
    if prev and prev.get("qa_count_at_run") == current_count:
        new_qas = [qa for qa in qas if not qa.get("difficulty_estimate", "")]
        if not new_qas:
            _log.info("Task 3: Already complete, skipping")
            return {}

    if progress_cb:
        progress_cb(0, "Sampling QAs for difficulty calibration...")

    # Phase 1: Flash benchmark on representative QAs
    anchors, anchor_labels = _phase1_difficulty_benchmark(db, qas, client, debug_cb)

    if progress_cb:
        progress_cb(30, "Calibrating difficulty signals...")

    # Phase 2: Signal calibration
    boundaries = _phase2_calibrate_signals(db, qas, anchors, anchor_labels, verb_data, debug_cb)

    if progress_cb:
        progress_cb(60, "Classifying remaining QAs...")

    # Phase 3: Classify all QAs + boundary Flash confirmation
    # Pre-load all QAs for verb percentile computation (avoids O(n^2) db.get_all() calls)
    all_qas_cache = qas
    _phase3_classify_and_confirm(db, all_qas_cache, boundaries, client, verb_data, debug_cb)

    if progress_cb:
        progress_cb(90, "Aggregating topic difficulty...")

    # Phase 4: Topic-level aggregation
    _phase4_topic_aggregation(db)

    db.checkpoint("difficulty", current_count, "completed")

    # Difficulty distribution summary
    diff_counts = db.conn.execute(
        "SELECT difficulty_estimate, COUNT(*) as cnt FROM qa_pairs "
        "WHERE difficulty_estimate != '' GROUP BY difficulty_estimate"
    ).fetchall()
    diff_map = {r["difficulty_estimate"]: r["cnt"] for r in diff_counts}
    total_diff = sum(diff_map.values())
    if total_diff > 0:
        b = diff_map.get("basic", 0); i = diff_map.get("intermediate", 0); a = diff_map.get("advanced", 0)
        debug_cb(f"  [Diff] QA difficulty: basic={b}, intermediate={i}, advanced={a} "
                 f"({b/total_diff*100:.0f}/{i/total_diff*100:.0f}/{a/total_diff*100:.0f})")
    topic_rows = db.conn.execute(
        "SELECT COUNT(*) as cnt FROM topic_difficulty"
    ).fetchone()
    mixed_rows = db.conn.execute(
        "SELECT COUNT(*) as cnt FROM topic_difficulty WHERE mode_difficulty = 'mixed'"
    ).fetchone()
    if topic_rows:
        debug_cb(f"  [Diff] Topic difficulty: {topic_rows['cnt']} topics assessed, "
                 f"{mixed_rows['cnt']} mixed" if mixed_rows and mixed_rows['cnt'] > 0 else "")

    _log.info("Task 3: Complete")
    return {qa["id"]: qa.get("difficulty_estimate", "") for qa in db.get_all()}


def _phase1_difficulty_benchmark(db, qas, client, debug_cb):
    """Flash rates ~30 representative QAs to establish difficulty baseline."""
    # Pick representative QAs: is_representative first, then highest Beta weight
    weights = db.get_all_weights()
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

        messages = DIFFICULTY_RATE.build(lang=lang, qa_block=qa_block)
        try:
            result = call_flash(client, messages, max_retries=1, debug_callback=debug_cb)
            ratings = result.get("ratings", []) if isinstance(result, dict) else []
        except Exception as e:
            debug_cb(f"  Difficulty benchmark batch failed: {e}")
            ratings = []

        for r in ratings:
            idx = r.get("question_index", -1)
            if 0 <= idx < len(batch):
                labels[batch[idx]["id"]] = r.get("difficulty", "intermediate")

    debug_cb(f"  Difficulty benchmark: {len(labels)} QAs rated by Flash")
    return candidates, labels


def _phase2_calibrate_signals(db, qas, anchors, anchor_labels, verb_data, debug_cb):
    """Use Flash-anchored QAs to find difficulty thresholds for each signal.
    Uses effective_miss_rate (knowledge_gap + insufficient_detail only), not raw miss_rate."""

    def _compute_effective_miss_rate(qa):
        """Only knowledge_gap + insufficient_detail contribute to difficulty.
        misinterpretation and retrieval_quality are excluded."""
        return db.get_effective_miss_rate(qa["id"])

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

    debug_cb(f"  Difficulty boundaries: {json.dumps({k: round(v, 3) for k, v in boundaries.items()})}")
    return boundaries


def _get_signal(qa, signal_name, db, all_qas):
    """Compute a difficulty signal for a single QA.

    Extracted from _phase3_classify_and_confirm for testability.
    all_qas: pre-loaded list of all QAs (avoids repeated db.get_all() calls).
    """
    if signal_name == "effective_miss_rate":
        return db.get_effective_miss_rate(qa["id"])
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


def _classify_difficulty(signals, boundaries, margin=0.10):
    """Classify a QA's difficulty from signal values and calibrated boundaries.

    Returns (label, is_boundary).  Extracted from _phase3_classify_and_confirm
    to eliminate 5-level nesting.
    """
    votes = {"basic": 0, "intermediate": 0, "advanced": 0}
    is_boundary = False

    for sig_name, sig_val in signals.items():
        if sig_val is None:
            continue
        bi_key = f"{sig_name}_basic_inter"
        ia_key = f"{sig_name}_inter_adv"
        if bi_key in boundaries:
            mid = boundaries[bi_key]
            if sig_val < mid * (1 - margin):
                votes["basic"] += 1
            elif sig_val < mid * (1 + margin):
                votes["basic"] += 1
                is_boundary = True
            else:
                if ia_key in boundaries:
                    mid2 = boundaries[ia_key]
                    if sig_val < mid2 * (1 - margin):
                        votes["intermediate"] += 1
                    elif sig_val < mid2 * (1 + margin):
                        votes["intermediate"] += 1
                        is_boundary = True
                    else:
                        votes["advanced"] += 1
                else:
                    votes["intermediate"] += 1
        elif ia_key in boundaries:
            mid2 = boundaries[ia_key]
            if sig_val < mid2 * (1 - margin):
                votes["intermediate"] += 1
            elif sig_val < mid2 * (1 + margin):
                votes["intermediate"] += 1
                is_boundary = True
            else:
                votes["advanced"] += 1
        else:
            votes["intermediate"] += 1

    if votes["advanced"] > 0:
        return "advanced", is_boundary
    elif votes["intermediate"] > 0:
        return "intermediate", is_boundary
    elif votes["basic"] > 0:
        return "basic", is_boundary
    return "intermediate", False


def _phase3_classify_and_confirm(db, qas, boundaries, client, verb_data, debug_cb):
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
        debug_cb(f"  Flash confirming {len(boundary_cases)} boundary difficulty cases...")
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

            messages = DIFFICULTY_RATE.build(lang=lang, qa_block=qa_block)
            try:
                result = call_flash(client, messages, max_retries=1, debug_callback=debug_cb)
                ratings = result.get("ratings", []) if isinstance(result, dict) else []
            except Exception as e:
                debug_cb(f"  Boundary difficulty batch failed: {e}")
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

    debug_cb(f"  Difficulty: {len(qas)} QAs classified "
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

        db.upsert_topic_difficulty(
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


# ============================================================
# Task 1: Dependency Discovery
# ============================================================

def discover_dependencies(db: QADatabase, client, debug_cb, progress_cb=None) -> list:
    """Discover prerequisite relationships between topics.

    Returns: [(prerequisite, dependent, score, confidence), ...]
    """
    _log.info("Offline Task 1: Dependency discovery starting")
    debug_cb("Task 1: Discovering topic dependencies...")

    qas = db.get_all()
    if not qas:
        _log.info("Task 1: No QAs, skipping")
        return []

    current_count = len(qas)
    prev = db.get_checkpoint("dependencies")
    if prev and prev.get("qa_count_at_run") == current_count:
        _log.info("Task 1: Already complete, skipping")
        return []

    if progress_cb:
        progress_cb(0, "Generating dependency candidates...")

    # Phase 0: Candidate generation
    candidates = _phase0_generate_candidates(db, debug_cb)

    if progress_cb:
        progress_cb(40, f"Validating {len(candidates)} candidate pairs...")

    # Phase 1: Batch Flash validation
    validated = _phase1_validate_candidates(db, candidates, client, debug_cb)

    if progress_cb:
        progress_cb(70, "Post-processing dependency graph...")

    # Phase 2: Graph post-processing
    _phase2_postprocess_dependencies(db, validated, debug_cb)

    db.checkpoint("dependencies", current_count, "completed")

    # Dependency confidence distribution
    conf_counts = db.conn.execute(
        "SELECT confidence, COUNT(*) as cnt FROM topic_dependencies GROUP BY confidence"
    ).fetchall()
    conf_map = {r["confidence"]: r["cnt"] for r in conf_counts}
    if conf_map:
        cstr = ", ".join(f"{k}={v}" for k, v in sorted(conf_map.items()))
        node_count = db.conn.execute(
            "SELECT COUNT(DISTINCT prerequisite) + COUNT(DISTINCT dependent) FROM topic_dependencies"
        ).fetchone()
        # count unique nodes from both columns
        nodes = set()
        for r in db.conn.execute("SELECT prerequisite, dependent FROM topic_dependencies").fetchall():
            nodes.add(r["prerequisite"]); nodes.add(r["dependent"])
        edge_count = sum(conf_map.values())
        debug_cb(f"  [Dep] Dependencies: {cstr} — {len(nodes)} nodes, {edge_count} edges")

    _log.info(f"Task 1: Complete. {len(validated)} dependencies stored")
    return validated


def _phase0_generate_candidates(db, debug_cb):
    """Generate candidate dependency pairs from topic_links and embedding similarity."""
    topic_links = db.get_topic_links()
    topic_texts = db.get_topic_answer_texts()
    topics = list(topic_texts.keys())

    if len(topics) < 2:
        return []

    candidates = set()

    # Source A: topic_links (behavioral evidence from Phase 2)
    for (src, dst), count in topic_links.items():
        if src not in topic_texts or dst not in topic_texts:
            continue
        rev_count = topic_links.get((dst, src), 0)

        if count >= 2 and rev_count >= 2 and abs(count - rev_count) <= max(count, rev_count) * 0.5:
            # Bidirectional ≈ symmetric → corequisite, handled in post-processing
            candidates.add((dst, src, "topic_link_symmetric", min(count, rev_count)))
            candidates.add((src, dst, "topic_link_symmetric", min(count, rev_count)))
        elif count >= 2 and count > rev_count * 3:
            # Strong unidirectional: dependency = reverse of topic_link direction
            candidates.add((dst, src, "topic_link", count))
        elif count >= 1:
            # Weak: only use if embedding also supports
            pass  # handled by Source B below with higher threshold

    # Source B: embedding similarity
    if len(topics) >= 2:
        try:
            model = _get_model(TOPIC_EMBED_MODEL)
            texts = [topic_texts[t] for t in topics]
            vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            cos_matrix = vecs @ vecs.T
        except Exception as e:
            debug_cb(f"  Topic embedding failed: {e}")
            cos_matrix = None

        if cos_matrix is not None:
            for i in range(len(topics)):
                for j in range(i + 1, len(topics)):
                    cos = float(cos_matrix[i][j])
                    if cos < 0.5:
                        continue
                    a, b = topics[i], topics[j]
                    # Check if already in candidates from topic_links
                    already = any(
                        (c[0] == a and c[1] == b) or (c[0] == b and c[1] == a)
                        for c in candidates
                    )
                    if not already:
                        # Both directions are candidates (no directional evidence yet)
                        candidates.add((a, b, "embed_only", round(cos, 3)))
                        candidates.add((b, a, "embed_only", round(cos, 3)))

    debug_cb(f"  Dependency candidates: {len(candidates)} pairs "
             f"(topic_links={len(topic_links)}, topics={len(topics)})")
    return list(candidates)


def _phase1_validate_candidates(db, candidates, client, debug_cb):
    """Batch Flash validation of dependency candidates (5 pairs per call). Parallelized."""
    topic_texts = db.get_topic_answer_texts()
    batch_size = 5

    sorted_candidates = sorted(candidates, key=lambda c: (
        0 if c[2] in ("topic_link", "topic_link_symmetric") else 1, -c[3]
    ))
    batches = [sorted_candidates[i:i+batch_size] for i in range(0, len(sorted_candidates), batch_size)]

    def _validate_batch(batch):
        sample_text = ""
        for pre, dep, ev_type, strength in batch:
            sample_text += topic_texts.get(pre, "") + " " + topic_texts.get(dep, "") + " "
        lang = detect_content_lang(sample_text[:2000])

        pairs_block = ""
        if lang == 'en':
            pairs_block += "Evaluate these topic pairs:\n\n"
        else:
            pairs_block += "评估以下 topic 对：\n\n"
        for i, (pre, dep, ev_type, strength) in enumerate(batch):
            pairs_block += (f"Pair {i}: Topic A [{pre}] → Topic B [{dep}]\n"
                            f"  A QAs: {topic_texts.get(pre, '')[:600]}\n"
                            f"  B QAs: {topic_texts.get(dep, '')[:600]}\n\n")

        messages = DEPENDENCY_VALIDATE.build(lang=lang, pairs_block=pairs_block)
        try:
            result = call_flash(client, messages, max_retries=1, debug_callback=debug_cb)
            pairs = result.get("pairs", []) if isinstance(result, dict) else []
        except Exception as e:
            debug_cb(f"  Dependency validation batch failed: {e}")
            return []

        batch_results = []
        for p in pairs:
            idx = p.get("index", -1)
            if 0 <= idx < len(batch):
                pre, dep, ev_type, strength = batch[idx]
                score = p.get("score", 1)
                confidence = "low"
                if score == 2:
                    if ev_type == "topic_link" and strength >= 4: confidence = "high"
                    elif ev_type == "topic_link" and strength >= 2: confidence = "medium"
                    elif ev_type == "topic_link_symmetric": confidence = "medium"
                if score >= 1:
                    batch_results.append({
                        "prerequisite": pre, "dependent": dep,
                        "score": score, "reason": p.get("reason", ""),
                        "evidence_type": ev_type, "evidence_strength": strength,
                        "confidence": confidence,
                        "relationship_type": "corequisite" if ev_type == "topic_link_symmetric" else "prerequisite",
                    })
        return batch_results

    validated = []
    if batches:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_w = get_worker_limit(len(batches), api_heavy=True)
        with ThreadPoolExecutor(max_workers=max_w) as executor:
            futures = {executor.submit(_validate_batch, b): b for b in batches}
            for future in as_completed(futures):
                try:
                    validated.extend(future.result())
                except Exception as e:
                    debug_cb(f"  Dependency validation thread failed: {e}")

    debug_cb(f"  Flash validated: {len(validated)} dependencies (from {len(candidates)} candidates, {len(batches)} batches parallel)")
    return validated


def _phase2_postprocess_dependencies(db, validated, debug_cb):
    """Store dependencies, apply transitive reduction, detect cycles."""
    # Build adjacency for graph operations
    edges = {}  # (pre, dep) -> metadata
    for v in validated:
        key = (v["prerequisite"], v["dependent"])
        if key in edges:
            # Keep higher confidence
            if v["confidence"] == "high" or edges[key]["confidence"] == "low":
                edges[key] = v
        else:
            edges[key] = v

    # Transitive reduction: remove A→C if A→B→C exists
    topics_set = set()
    for pre, dep in edges:
        topics_set.add(pre)
        topics_set.add(dep)
    topics = list(topics_set)

    # Build adjacency list
    adj = {t: set() for t in topics}
    for pre, dep in edges:
        adj.setdefault(pre, set()).add(dep)

    # BFS from each node to find redundant edges
    redundant = set()
    for a in topics:
        if a not in adj:
            continue
        for b in list(adj.get(a, set())):
            # Check if there's a path a→...→b that doesn't use the direct edge
            visited = {a}
            frontier = deque([a])
            while frontier:
                curr = frontier.popleft()
                for nxt in adj.get(curr, set()):
                    if nxt == b and curr != a:
                        redundant.add((a, b))
                        break
                    if nxt not in visited and nxt != b:
                        visited.add(nxt)
                        frontier.append(nxt)
                if (a, b) in redundant:
                    break

    # Store non-redundant edges
    stored = 0
    for (pre, dep), v in edges.items():
        if (pre, dep) in redundant:
            debug_cb(f"  Transitive reduction: removed {pre}→{dep}")
            continue
        db.insert_dependency(DependencySpec(
            prerequisite=pre, dependent=dep,
            evidence_score=v["score"],
            evidence_reason=v.get("reason", ""),
            relationship_type=v.get("relationship_type", "prerequisite"),
            topic_link_count=v.get("evidence_strength", 0) if v["evidence_type"] != "embed_only" else 0,
            embedding_cos=v.get("evidence_strength", None) if v["evidence_type"] == "embed_only" else None,
            confidence=v["confidence"],
            validated_by="flash",
        ))
        stored += 1

    debug_cb(f"  Dependencies stored: {stored} (removed {len(redundant)} transitive)")

    # Detect remaining cycles (co-requisites) — check both new candidates and pre-existing DB edges
    cycles_found = 0
    with db.transaction():
        for pre, dep in list(edges.keys()):
            rev_key = (dep, pre)
            has_reverse = rev_key in edges
            if not has_reverse and (pre, dep) not in redundant:
                # Check DB for pre-existing reverse edge from a previous pipeline run
                db_row = db.conn.execute(
                    "SELECT 1 FROM topic_dependencies WHERE prerequisite=? AND dependent=?",
                    rev_key,
                ).fetchone()
                has_reverse = db_row is not None
            if has_reverse and (pre, dep) not in redundant and rev_key not in redundant:
                # Bidirectional dependency → update both to corequisite
                db.conn.execute(
                    """UPDATE topic_dependencies SET relationship_type = 'corequisite'
                       WHERE prerequisite = ? AND dependent = ?""",
                    (pre, dep),
                )
                cycles_found += 1
        debug_cb(f"  Co-requisite cycles found: {cycles_found}")


# ============================================================
# Orchestrator
# ============================================================

def _write_verb_report(db: QADatabase, output_dir: str, subject_code: str):
    """Write human-readable command verb analysis report to point/{subject}_verb_patterns.txt"""
    patterns = db.get_verb_patterns()
    if not patterns:
        return None

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{subject_code}_verb_patterns.txt")
    lines = [f"Command Verb Answer Patterns — {subject_code}", "=" * 60, ""]

    for p in patterns:
        verb = p["verb"]
        family = p.get("verb_family", "") or ""
        n = p["sample_count"]
        lines.append(f"{verb}  [{family}]  (n={n})")
        lines.append("-" * 40)
        if p.get("avg_answer_length"):
            lines.append(f"  Avg answer length: {p['avg_answer_length']:.0f} chars  "
                         f"(median: {p['median_answer_length']:.0f})")
        if p.get("bullet_ratio") is not None:
            lines.append(f"  Bullet/list usage: {p['bullet_ratio']*100:.0f}%  "
                         f"(avg {p['avg_bullet_count']:.1f} bullets)")
        if p.get("avg_miss_rate") is not None:
            lines.append(f"  Avg miss rate (AI): {p['avg_miss_rate']*100:.0f}%")
        if p.get("pattern_summary"):
            lines.append(f"  Pattern: {p['pattern_summary']}")
        if p.get("topic_specific_patterns"):
            try:
                tsp = json.loads(p["topic_specific_patterns"])
                for topic, note in tsp.items():
                    lines.append(f"    [{topic}] {note}")
            except Exception as e:
                _log.warning(f"verb pattern JSON parse: {e}")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path


def run_offline_analysis(db, api_url: str, api_key: str,
                         progress_callback=None, debug_callback=None,
                         output_dir: str = None):
    """Run all three offline analysis tasks in order.

    Called from pipeline.run_pipeline() after main processing completes.
    output_dir: directory for analysis report files (defaults to point/ adjacent to db_path)
    """
    def _debug(msg):
        if debug_callback:
            debug_callback(f"[Offline] {msg}")
        else:
            print(f"[Offline] {msg}")

    def _progress(pct, status):
        if progress_callback:
            progress_callback(pct, f"[Analysis] {status}")

    _debug("Starting offline analysis pipeline")

    if db.count() == 0:
        _debug("No QAs in database, skipping offline analysis")
        return

    # Derive subject code from db_path
    subject_code = "unknown"
    m = re.search(r'(\d+)_knowledge\.db', db.db_path)
    if m:
        subject_code = m.group(1)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(db.db_path) or ".", "..", "point")
        output_dir = os.path.normpath(output_dir)

    client = create_client(api_url, api_key)

    # Task 2: Command verb analysis (independent, runs first)
    _progress(0, "Extracting command verbs...")
    verb_data = analyze_command_verbs(db, client, _debug, _progress)

    # Task 2b: Write verb report
    report_path = _write_verb_report(db, output_dir, subject_code)
    if report_path:
        _debug(f"Verb pattern report: {report_path}")

    # Task 3: Difficulty assessment (depends on Task 2 for verb_length_percentile)
    _progress(33, "Assessing difficulty...")
    difficulty_data = assess_difficulty(db, client, verb_data, _debug, _progress)

    # Task 1: Dependency discovery (depends on Task 3 for cross_topic signal, Task 2 for verbs)
    _progress(66, "Discovering dependencies...")
    discover_dependencies(db, client, _debug, _progress)

    _progress(100, "Offline analysis complete")

    _debug("Offline analysis pipeline finished")
