import json
import os
import re
from collections import deque

from ..deepseek_client import create_client, call_flash
from ..knowledge_base import QADatabase
from ..embedding_cluster import _get_model, detect_content_lang, TOPIC_EMBED_MODEL
from ..models import VerbPatternSpec, DependencySpec
from ..prompt_factory import VERB_PATTERN_SUMMARY, DIFFICULTY_RATE, DEPENDENCY_VALIDATE
from ..logger import get_logger
from ..utils import get_worker_limit

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
    prev = db.analysis.get_checkpoint("command_verbs")
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

    db.analysis.checkpoint("command_verbs", current_count, "completed")
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
            from ..error_utils import log_exception
            log_exception(debug_cb, "Verb extraction", f"batch={b}", e)
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
                    from ..error_utils import log_exception
                    log_exception(debug_cb, "Verb extraction", f"thread", e)

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
            from ..error_utils import log_exception
            log_exception(debug_cb, "Pattern summary", f"verb={verb}", e)
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

        db.analysis.upsert_verb_pattern(VerbPatternSpec(
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
                from ..error_utils import log_exception
                log_exception(debug_cb, "Verb pattern", f"thread", e)


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


