"""Pipeline v4: QA-based knowledge accumulation with Flash summaries.

Phase 1: Store QA pairs + Flash-generated knowledge summaries.
Phase 2: Flash summary -> embedding retrieval -> Pro answer -> Flash grade -> store.
"""

import os
import json
import re
import time
import hashlib
import threading
import traceback
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from .deepseek_client import create_client, call_flash
from .file_pairer import pair_files
from .knowledge_base import QADatabase, QARetriever
from .pdf_extractor import extract_pdf
from .embedding_cluster import detect_content_lang, TOPIC_EMBED_MODEL


# ============================================================
# Unified error handling
# ============================================================

class PipelineError(Exception):
    """Pipeline error with severity: fatal | recoverable | degraded."""
    def __init__(self, message: str, severity: str = "recoverable"):
        super().__init__(message)
        self.severity = severity


def _log_stage_error(stage: str, debug: callable, exc: Exception):
    """Standardized error logging for any pipeline stage."""
    debug(f"[{stage}] {type(exc).__name__}: {exc}")
    debug(traceback.format_exc())


def _get_worker_limit(n: int, api_heavy: bool = False) -> int:
    """Dynamic worker limit: configurable via env, with sensible defaults.

    PIPELINE_MAX_WORKERS env var overrides the cap. Otherwise:
      - General (CPU work): cap at 16
      - API-heavy (rate limited): cap at 8
    """
    env_override = os.environ.get("PIPELINE_MAX_WORKERS", "")
    if env_override.isdigit():
        return max(1, min(n, int(env_override)))
    cap = 8 if api_heavy else 16
    return max(1, min(n, cap))


# ============================================================
# Stage 2: QA pairing (Flash)
# ============================================================

def stage2_qa_pairing(pair, client, debug: Callable) -> list:
    """Call Flash once per paper to match questions with answers."""
    from .models import ExtractedPair, QAPair
    qp_text = pair.qp.full_text
    ms_text = pair.ms.full_text
    debug(f"  QA pairing: {pair.display_name} ({len(qp_text)}c / {len(ms_text)}c)")

    # Detect language for bilingual prompts
    sample = (qp_text + ms_text)[:2000]
    lang = detect_content_lang(sample)

    if lang == 'en':
        sys = (
            "You are an exam paper question-answer pairing assistant. "
            "Match each question in the QP with its answer in the MS. "
            "Preserve the original question numbering. Output JSON."
        )
        usr = (
            f"Paper: {pair.display_name}\n\n"
            f"=== Question Paper (QP) ===\n{qp_text}\n\n"
            f"=== Mark Scheme (MS) ===\n{ms_text}\n\n"
            "Match each question with its answer.\n"
            'Return JSON: {"qa_pairs": [{"question_number": "2(a)", "parent_question": "2", "question_text": "...", "answer_text": "..."}]}\n\n'
            "Notes:\n"
            "1. question_number: original numbering (e.g. '1(a)', '2(b)(iii)')\n"
            "2. parent_question: parent question number (e.g. '1', '2'), same as question_number if no sub-questions\n"
            "3. question_text: use original question text\n"
            "4. answer_text: use the corresponding mark scheme text (do not omit any mark points)\n"
            "5. If a question has no answer in the mark scheme, set answer_text to empty string"
        )
    else:
        sys = (
            "你是一个考试试卷配题助手。将试卷(Question Paper)中的题目"
            "与答案(Markscheme)中的对应答案配对。保留试卷的原始题目编号结构。Output JSON。"
        )
        usr = (
            f"试卷名称：{pair.display_name}\n\n"
            f"=== 试卷内容 (QP) ===\n{qp_text}\n\n"
            f"=== 答案内容 (MS) ===\n{ms_text}\n\n"
            "请将每个题目与其答案配对。\n"
            '返回 JSON 格式：\n'
            '{"qa_pairs": [{"question_number": "2(a)", "parent_question": "2", "question_text": "...", "answer_text": "..."}]}\n\n'
            "注意：\n"
            "1. question_number 保留试卷原始编号（如 '1(a)', '2(b)(iii)'）\n"
            "2. parent_question 为大题编号（如 '1', '2'），无小问时与 question_number 相同\n"
            "3. question_text 使用题目原文\n"
            "4. answer_text 使用答案中对应的得分点原文（不要省略任何得分点）\n"
            "5. 如果某题目在 markscheme 中没有对应答案，answer_text 设为空字符串"
        )

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ]
    result = call_flash(client, messages, debug_callback=debug)
    raw_pairs = result.get("qa_pairs", [])
    debug(f"  QA pairing: {len(raw_pairs)} questions found")
    qa_list = []
    for i, rp in enumerate(raw_pairs):
        qn = rp.get("question_number", "")
        if not qn:
            qn = str(i + 1)
        qa_list.append(QAPair(
            question_number=str(qn),
            question_text=rp.get("question_text", ""),
            answer_text=rp.get("answer_text", ""),
            parent_question=rp.get("parent_question", str(qn)),
        ))
    return qa_list


# ============================================================
# Bilingual prompt sets for each API call
# ============================================================

# -- Summary + Topic (Flash) --

def _generate_summary(question_text: str, answer_text: str,
                      client, debug: Callable,
                      existing_topics: list = None) -> tuple[str, str]:
    lang = detect_content_lang(question_text + answer_text)
    topic_hint = ""
    if existing_topics:
        top_topics = sorted(existing_topics, key=lambda x: -x[1])[:15]  # top 15 by count
        topic_list = ", ".join(t[0] for t in top_topics)
        if lang == 'en':
            topic_hint = (f"\nExisting topics (reuse if applicable): {topic_list}\n"
                          "If this question matches an existing topic, use that exact name. "
                          "Only create a new topic name if none of the existing ones fit.\n")
        else:
            topic_hint = (f"\n已有主题（若适用请复用）: {topic_list}\n"
                          "如果这道题匹配已有主题，请使用完全相同的名称。"
                          "只有在已有主题都不适用时才创建新主题名。\n")

    if lang == 'en':
        sys = ("You are an exam knowledge classifier. Do two things. Output JSON. "
               "1. Describe the core concept tested in 1-2 sentences. "
               "2. Assign a concise topic name.")
        usr = (f"Question: {question_text}\n\nAnswer: {answer_text}\n\n"
               f"{topic_hint}"
               "Do not mention question-specific context (values, filenames, scenarios, names).\n"
               "Use standard terminology for topic names (e.g. 'Data Compression').\n"
               'Return JSON: {"summary": "core concept description", "topic": "Topic Name"}')
    else:
        sys = ("你是一个考试知识分类专家。同时完成两件事。Output JSON."
               "1. 用1-2句描述这道题考察的核心技术概念"
               "2. 分配一个简洁的主题名称")
        usr = (f"题目: {question_text}\n\n答案: {answer_text}\n\n"
               f"{topic_hint}"
               "不要提及题目特定上下文（具体数值、文件名、场景描述、人名）。\n"
               "主题名称应使用标准术语（如 'Data Compression', 'Interrupt Handling'）。\n"
               '返回 JSON: {"summary": "核心知识描述", "topic": "标准主题名"}')
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
    try:
        result = call_flash(client, messages, max_retries=1, debug_callback=debug)
        if isinstance(result, dict):
            return result.get("summary", question_text[:200]), result.get("topic", "(unnamed)")
        return str(result), "(unnamed)"
    except Exception as e:
        debug(f"  summary generation failed: {e}")
        return question_text[:200], "(unnamed)"


def _extract_ms_fragments(answer_text: str, qa_id: int, client, debug: Callable) -> list[dict]:
    """Split a mark scheme answer into individual scoring points. Preserves original wording."""
    lang = detect_content_lang(answer_text)

    if lang == 'en':
        sys = ("You are an exam marking expert. Split the given mark scheme answer "
               "into individual scoring points. Output JSON.")
        usr = (f"Mark scheme answer:\n{answer_text}\n\n"
               "Split into independent scoring points. Each point is a single, "
               "non-divisible requirement that a student must demonstrate.\n"
               "Preserve the original wording exactly. Do not rewrite or paraphrase.\n"
               'Return JSON: {"points": [{"text": "exact original wording", "marks": 1}, ...]}')
    else:
        sys = ("你是一个考试评分专家。将给定的评分标准答案拆分为独立的得分点。Output JSON。")
        usr = (f"评分标准答案:\n{answer_text}\n\n"
               "拆分为独立的得分点。每个得分点是一个不可再分的要求。\n"
               "保留原文措辞，不要改写。\n"
               '返回 JSON: {"points": [{"text": "原始措辞", "marks": 1}, ...]}')

    messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
    try:
        result = call_flash(client, messages, max_retries=1, debug_callback=debug)
        points = result.get("points", []) if isinstance(result, dict) else []
    except Exception as e:
        debug(f"  fragment extraction failed for QA {qa_id}: {e}")
        points = []

    fragments = []
    for i, p in enumerate(points):
        text = p.get("text", "") if isinstance(p, dict) else str(p)
        if text.strip():
            fragments.append({
                "point_id": f"f_{qa_id}_{i}",
                "qa_id": qa_id,
                "point_text": text.strip(),
                "marks": p.get("marks", 1) if isinstance(p, dict) else 1,
            })
    return fragments


# -- Round 1: Answer with past QAs (Pro) --

def _build_answer_prompt(question_text: str, similar_qas: list[dict]) -> list:
    lang = detect_content_lang(question_text)
    if similar_qas:
        qa_block = ""
        for i, qa in enumerate(similar_qas, 1):
            qa_block += (f"--- Past Q{i} (similarity: {qa.get('_score',0):.2f}) ---\n"
                         f"Q: {qa['question_text']}\nA: {qa['answer_text']}\n\n")
    else:
        qa_block = "(Knowledge base is empty, no past Q&As)\n\n"

    if lang == 'en':
        sys = "You are an exam answering system. Use past Q&A knowledge to answer new questions. Output JSON."
        usr = (f"{qa_block}=== New Question ===\n{question_text}\n\n"
               "Tasks:\n"
               "1. Answer the question using knowledge from past Q&As\n"
               "2. Mark which past Q&As you used (by number 1, 2...)\n"
               'Return JSON: {"answer": "...", "used_qa_indices": [1, 3]}')
    else:
        sys = "你是一个考试答题系统。利用历史题目的答案来回答新题。Output JSON."
        usr = (f"{qa_block}=== 新题目 ===\n{question_text}\n\n"
               "任务:\n"
               "1. 利用历史题目的知识回答这道新题\n"
               "2. 标注使用了哪些历史题目（用序号 1, 2...）\n"
               '返回 JSON: {"answer": "...", "used_qa_indices": [1, 3]}')
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


# -- Round 2: Grade + topic (Flash) --

def _build_grade_prompt(question_text: str, predicted_answer: str,
                        ms_answer: str) -> list:
    lang = detect_content_lang(question_text + ms_answer)
    if lang == 'en':
        sys = ("You are an exam grading expert. Compare student answer with markscheme point by point. "
               "Assign a topic name. For every missed point, classify WHY it was missed. Output JSON.")
        usr = (f"Question: {question_text}\n\n"
               f"Student Answer: {predicted_answer}\n\n"
               f"Markscheme: {ms_answer}\n\n"
               "Tasks:\n"
               "1. Compare each markscheme point against the student answer\n"
               "2. List covered points. For each missed point, provide the point text AND a reason:\n"
               "   - knowledge_gap: student did not mention this concept at all (missing knowledge)\n"
               "   - misinterpretation: student misunderstood what the question asked for\n"
               "   - insufficient_detail: student addressed the right idea but lacked precision/completeness\n"
               "   - retrieval_quality: student's answer was limited by poor reference material\n"
               "3. Assign a topic (e.g. 'Data Compression', 'Interrupt Handling')\n\n"
               'Return JSON: {"topic": "Topic Name", "covered_points": ["..."], '
               '"missed_points": [{"point": "...", "reason": "knowledge_gap"}, ...]}')
    else:
        sys = ("你是一个考试批改专家。对比学生答案和标准答案，逐得分点评分。"
               "对每个遗漏的得分点标注遗漏原因。为这道题分配一个主题名称。Output JSON.")
        usr = (f"题目: {question_text}\n\n"
               f"学生答案: {predicted_answer}\n\n"
               f"标准答案 (Markscheme): {ms_answer}\n\n"
               "任务:\n"
               "1. 对比学生答案和标准答案的每个得分点\n"
               "2. 列出覆盖的得分点。对每个遗漏的得分点，提供内容及遗漏原因：\n"
               "   - knowledge_gap: 学生答案未涉及此概念（缺乏相关知识）\n"
               "   - misinterpretation: 学生理解偏了题目要求\n"
               "   - insufficient_detail: 学生答了方向对但不够精确/完整\n"
               "   - retrieval_quality: 学生的回答受限于参考材料质量\n"
               "3. 为这道题分配一个主题（如 'Data Compression', 'Interrupt Handling'）\n\n"
               '返回 JSON: {"topic": "标准主题名", "covered_points": ["点1"], '
               '"missed_points": [{"point": "遗漏点A", "reason": "knowledge_gap"}, ...]}')
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


# ============================================================
# Progress tracker
# ============================================================

class ProgressTracker:
    """Tracks real progress based on completed work units. Thread-safe."""

    def __init__(self, total_units: int, callback: Callable, log_cb: Callable = None):
        self.total = total_units
        self.done = 0
        self._lock = threading.Lock()
        self._cb = callback
        self._log = log_cb

    def step(self, status: str = ""):
        with self._lock:
            self.done += 1
            pct = min(99, int(100 * self.done / self.total))
        self._cb(pct, status)
        if self._log and status:
            self._log(status, f"{self.done}/{self.total}")

    def set_status(self, status: str):
        pct = min(99, int(100 * self.done / self.total))
        self._cb(pct, status)


# ============================================================
# Pipeline
# ============================================================

def _ensure_session(db, display_name: str) -> Optional[int]:
    """Parse exam paper filename and ensure exam_sessions row exists."""
    m = re.match(r'^(\d+)_([smw])(\d{2})_(\d+)', display_name)
    if not m:
        return None
    season_map = {'s': 'S1', 'w': 'S2', 'm': 'S3'}
    subject = m.group(1)
    season = season_map.get(m.group(2), "Unknown")
    year = 2000 + int(m.group(3))
    db.conn.execute(
        """INSERT OR IGNORE INTO exam_sessions (subject_code, season, year, display_name)
           VALUES (?, ?, ?, ?)""",
        (subject, season, year, display_name),
    )
    db.conn.commit()
    row = db.conn.execute(
        "SELECT id FROM exam_sessions WHERE display_name = ?", (display_name,)
    ).fetchone()
    return row["id"] if row else None


def run_pipeline(
    api_url: str,
    api_key: str,
    input_dir: str,
    output_path: str,
    intermediate_dir: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    debug_callback: Optional[Callable] = None,
    log_callback: Optional[Callable] = None,
    shutdown_event=None,
) -> str:
    def _progress(pct: int, status: str):
        if progress_callback:
            progress_callback(pct, status)

    def _debug(msg: str):
        if debug_callback:
            debug_callback(msg)
        else:
            print(f"[DEBUG] {msg}")

    def _log(step: str, detail: str = ""):
        if log_callback:
            log_callback(step, detail)

    _progress(0, "Initializing...")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if intermediate_dir is None:
        intermediate_dir = os.path.join(os.path.dirname(input_dir) or ".", "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)

    _progress(2, "Pairing papers...")
    pairs = pair_files(input_dir)

    # Sort by year ascending: earliest papers first → Phase 1 gets foundational content
    def _parse_year(display_name):
        m = re.search(r'[smw](\d{2})_', display_name)
        return 2000 + int(m.group(1)) if m else 9999
    pairs.sort(key=lambda p: _parse_year(p[2]))

    # Log processing order
    order_desc = " → ".join(f"{p[2]}({_parse_year(p[2])})" for p in pairs[:10])
    if len(pairs) > 10:
        order_desc += f" ... (+{len(pairs)-10} more)"
    _debug(f"Pairs: {len(pairs)} — order: {order_desc}")
    _log("Pairing", f"{len(pairs)} pairs")
    if not pairs:
        raise RuntimeError(f"No paired papers found in {input_dir}")

    subject_code = "unknown"
    m = re.match(r'^(\d+)', pairs[0][2])
    if m:
        subject_code = m.group(1)
    else:
        _debug(f"Could not extract subject code from: {pairs[0][2]}, using 'unknown'")

    out_dir = os.path.dirname(output_path) or "."
    subject_output = os.path.join(out_dir, f"{subject_code}_points.txt")
    db_path = os.path.join(intermediate_dir, f"{subject_code}_knowledge.db")
    processed_path = os.path.join(intermediate_dir, f"{subject_code}_processed.json")

    _debug(f"Subject: {subject_code}, output: {subject_output}")
    _log("Subject", subject_code)

    db = QADatabase(db_path)
    retriever = QARetriever(db)

    # Log DB schema status
    from .knowledge_base import log_schema_status
    log_schema_status(db, _debug)

    processed: set = set()
    if os.path.exists(processed_path):
        try:
            with open(processed_path, "r") as f:
                processed = set(json.load(f))
            _debug(f"Processed files: {len(processed)}")
        except Exception as e:
            _debug(f"Failed to read processed.json: {e}")

    def _get_existing_topics():
        """Return [(topic_name, qa_count), ...] sorted by count desc."""
        groups = db.get_topic_groups()
        return [(t, len(qas)) for t, qas in groups.items()
                if t and t != "(uncategorized)"]

    client = create_client(api_url, api_key)
    is_first = (db.count() == 0)
    topic_links = db.get_topic_links()

    tracker = ProgressTracker(len(pairs) * 70 + 10, _progress, _log)

    for (qp_path, ms_path, display_name) in pairs:
        if shutdown_event and shutdown_event.is_set():
            _debug("Shutdown requested, stopping gracefully")
            break

        if display_name in processed:
            _debug(f"[{display_name}] Already processed, skipping")
            _log(f"  SKIP {display_name}", "already processed")
            continue

        tracker.set_status(f"Processing: {display_name}")

        # Resolve exam session for time-dimension queries
        session_id = _ensure_session(db, display_name)

        _debug(f"[{display_name}] PDF extraction + QA pairing...")
        try:
            qp_pdf = extract_pdf(qp_path)
            ms_pdf = extract_pdf(ms_path)
        except Exception as e:
            _debug(f"[{display_name}] PDF extraction failed: {e}, skipping")
            continue

        from .models import ExtractedPair
        pair = ExtractedPair(display_name=display_name, qp=qp_pdf, ms=ms_pdf)
        try:
            qa_pairs = stage2_qa_pairing(pair, client, _debug)
        except Exception as e:
            _debug(f"[{display_name}] QA pairing failed: {e}, skipping")
            continue

        if not qa_pairs:
            _debug(f"[{display_name}] No QA pairs found, skipping")
            continue

        if is_first:
            _debug(f"[{display_name}] Phase1: building KB ({len(qa_pairs)} questions)")

            existing_topics = _get_existing_topics() if db.count() > 0 else None

            def _phase1_worker(qa):
                t0 = time.time()
                summary, topic = _generate_summary(qa.question_text, qa.answer_text, client, _debug,
                                                   existing_topics=existing_topics)
                db.log_api_call("summary", "flash", display_name,
                                qa.question_number, int((time.time()-t0)*1000),
                                success=True, output_size=len(summary))
                qa_id = db.insert(question_text=qa.question_text, answer_text=qa.answer_text,
                                  knowledge_summary=summary, topic=topic,
                                  paper=display_name, question_number=qa.question_number,
                                  parent_question=qa.parent_question)
                db.record_attempt(qa_id, success=True)

                # Phase 1: Extract MS fragments + assign to initial topic
                try:
                    fragments = _extract_ms_fragments(
                        qa.answer_text, qa_id, client, _debug)
                    if fragments:
                        db.insert_fragments_batch(fragments)
                        topic_id = f"topic_{topic.replace(' ', '_').replace('/', '_')}"
                        db.upsert_dynamic_topic(
                            topic_id, name=topic, quality="embryonic")
                        for frag in fragments:
                            db.set_fragment_membership(
                                frag["point_id"], topic_id, loyalty=0.5)
                except Exception as e:
                    _debug(f"  fragment extraction failed for Q{qa.question_number}: {e}")

                tracker.step("")
                return qa_id

            tracker.step("QA pairing")  # Stage 2 complete
            with ThreadPoolExecutor(max_workers=_get_worker_limit(len(qa_pairs))) as executor:
                futures = {executor.submit(_phase1_worker, qa): qa for qa in qa_pairs}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        qa = futures[future]
                        _debug(f"  Q{qa.question_number} Phase1 failed: {e}")
                        tracker.step("")  # count failure too

            _debug(f"[{display_name}] KB: {db.count()} entries")
            is_first = False
            _debug("Loading embedding model for retrieval (first run may download ~80MB)...")
            retriever.rebuild()
            _debug("Embedding model ready")

        else:
            _debug(f"[{display_name}] Phase2: parallel test-learn ({len(qa_pairs)} questions)")

            # Pre-load QA weights + existing topics for retrieval-augmented summary
            weight_map = db.get_all_weights()
            existing_topics = _get_existing_topics() if db.count() > 0 else None

            def _process_one_question(qa):
                try:
                    return _process_one_question_inner(qa, weight_map, existing_topics)
                except Exception as e:
                    _debug(f"  Q{qa.question_number} thread error: {e}")
                    return (None, "", {})

            def _process_one_question_inner(qa, wmap, extopics):
                qn = qa.question_number

                # Step 0: Flash summary
                t0 = time.time()
                summary, step0_topic = _generate_summary(qa.question_text, qa.answer_text, client, _debug,
                                                         existing_topics=extopics)
                db.log_api_call("summary", "flash", display_name, qn,
                                int((time.time()-t0)*1000), success=True, output_size=len(summary))
                tracker.step("")

                all_similar = retriever.search(summary, threshold=0.5, min_k=3, max_cap=15)

                # Filter to top-4 by Beta weight for Pro context (reduces input tokens ~70%)
                top_similar = sorted(
                    all_similar,
                    key=lambda qa_ref: wmap.get(qa_ref["id"], {}).get("mean", 0.5),
                    reverse=True,
                )[:4]

                # Round 1: Flash answers (cheaper than Pro, richer miss signal for difficulty)
                t0 = time.time()
                r1_ok = True
                try:
                    messages = _build_answer_prompt(qa.question_text, top_similar)
                    result = call_flash(client, messages, max_retries=1, debug_callback=_debug)
                except Exception as e:
                    _debug(f"  Q{qn} Round1 failed: {e}")
                    result = {"answer": "", "used_qa_indices": []}
                    r1_ok = False
                db.log_api_call("answer", "flash", display_name, qn,
                                int((time.time()-t0)*1000), success=r1_ok,
                                output_size=len(result.get("answer","")))
                tracker.step("")

                used_indices = result.get("used_qa_indices", [])
                used_ids = set()
                for idx in used_indices:
                    if isinstance(idx, (int, float)) and 1 <= int(idx) <= len(top_similar):
                        qa_ref = top_similar[int(idx) - 1]
                        db.record_attempt(qa_ref["id"], success=True)
                        used_ids.add(qa_ref["id"])
                for qa_ref in top_similar:
                    if qa_ref["id"] not in used_ids:
                        ref_topic = qa_ref.get("topic", "")
                        if ref_topic and step0_topic and ref_topic != step0_topic:
                            reason = "topic_mismatch"
                        else:
                            reason = "retrieval_irrelevant"
                        db.record_attempt(qa_ref["id"], success=False, reason=reason)

                # Round 2: Flash grades
                t0 = time.time()
                r2_ok = True
                r2_topic = ""
                grade = ""
                try:
                    grade_msgs = _build_grade_prompt(qa.question_text, result.get("answer",""), qa.answer_text)
                    grade = call_flash(client, grade_msgs, max_retries=1, debug_callback=_debug)
                    if isinstance(grade, dict):
                        r2_topic = grade.get("topic", "")
                        covered = grade.get("covered_points", [])
                        missed_raw = grade.get("missed_points", [])
                    else:
                        covered, missed_raw = [], []
                        r2_ok = False
                except Exception as e:
                    _debug(f"  Q{qn} Round2 failed: {e}")
                    covered, missed_raw = [], []
                    r2_ok = False

                # Parse missed points: new format is [{"point":"...", "reason":"..."}, ...]
                # Old format was ["...", ...] — flatten for backward compat
                missed_texts = []
                miss_cats = {"knowledge_gap": 0, "misinterpretation": 0,
                             "insufficient_detail": 0, "retrieval_quality": 0}
                for mp in missed_raw:
                    if isinstance(mp, dict):
                        missed_texts.append(mp.get("point", ""))
                        reason = mp.get("reason", "")
                        if reason in miss_cats:
                            miss_cats[reason] += 1
                    elif isinstance(mp, str):
                        missed_texts.append(mp)
                miss_cats_json = json.dumps(miss_cats) if any(miss_cats.values()) else ""

                db.log_api_call("grade", "flash", display_name, qn,
                                int((time.time()-t0)*1000), success=r2_ok,
                                output_size=len(str(grade)))
                tracker.step("")

                topic = r2_topic or step0_topic

                # Collect cross-topic references for See also generation
                cross_refs = {}
                for idx in used_indices:
                    if isinstance(idx, (int, float)) and 1 <= int(idx) <= len(top_similar):
                        ref_topic = top_similar[int(idx) - 1].get("topic", "")
                        if ref_topic and ref_topic != topic:
                            key = (topic, ref_topic)
                            cross_refs[key] = cross_refs.get(key, 0) + 1

                qa_id = db.insert(question_text=qa.question_text, answer_text=qa.answer_text,
                                  knowledge_summary=summary, topic=topic,
                                  paper=display_name, question_number=qn,
                                  parent_question=qa.parent_question)
                db.record_attempt(qa_id, success=True)

                db.log_question_feedback(qa_id=qa_id, retrieval_count=len(all_similar),
                                         used_qa_count=len(used_indices),
                                         step0_topic=step0_topic, round2_topic=r2_topic,
                                         covered_count=len(covered), missed_count=len(missed_texts),
                                         missed_text="\n".join(missed_texts) if missed_texts else "",
                                         miss_categories=miss_cats_json)

                # Phase 1: Extract MS fragments + record behavior help + assign topic
                try:
                    fragments = _extract_ms_fragments(
                        qa.answer_text, qa_id, client, _debug)
                    if fragments:
                        db.insert_fragments_batch(fragments)
                        topic_id = f"topic_{topic.replace(' ', '_').replace('/', '_')}"
                        db.upsert_dynamic_topic(
                            topic_id, name=topic, quality="embryonic")
                        for frag in fragments:
                            db.set_fragment_membership(
                                frag["point_id"], topic_id, loyalty=0.5)

                    # Record behavior: fragments from used QAs helped this question
                    if used_ids:
                        for used_qa_id in used_ids:
                            used_frags = db.conn.execute(
                                "SELECT point_id FROM ms_fragments WHERE qa_id=?",
                                (used_qa_id,)
                            ).fetchall()
                            frag_ids = [r["point_id"] for r in used_frags]
                            if frag_ids:
                                help_effect = (len(covered) / max(
                                    len(covered) + len(missed_texts), 1))
                                db.record_fragment_help_batch(
                                    frag_ids, qa_id, round(help_effect, 3))
                except Exception as e:
                    _debug(f"  fragment extraction failed for Q{qn}: {e}")

                _debug(f"  Q{qn}: retrieved={len(all_similar)}, shown={len(top_similar)}, "
                       f"used={len(used_indices)}, topic={topic}, "
                       f"covered={len(covered)}, missed={len(missed_texts)}")
                return (qa_id, summary, cross_refs)

            tracker.step("QA pairing")  # Stage 2 complete
            qa_results = []
            with ThreadPoolExecutor(max_workers=_get_worker_limit(len(qa_pairs))) as executor:
                futures = {executor.submit(_process_one_question, qa): qa for qa in qa_pairs}
                for future in as_completed(futures):
                    try:
                        qa_id, summary, cross_refs = future.result()
                        qa_results.append((qa_id, summary))
                        for (src, dst), count in cross_refs.items():
                            db.upsert_topic_link(src, dst, count)
                            topic_links[(src, dst)] = topic_links.get((src, dst), 0) + count
                    except Exception as e:
                        qa = futures[future]
                        _debug(f"  Q{qa.question_number} thread failed: {e}")
                        tracker.step("")  # count failure too

            for qa_id, summary in qa_results:
                if qa_id is not None:
                    retriever.add_qa(qa_id, summary)

            _debug(f"[{display_name}] KB: {db.count()} entries")

            # Phase 2 summary: retrieval quality + miss categories for this paper
            if not is_first:
                try:
                    fb_rows = db.conn.execute(
                        """SELECT SUM(retrieval_count) as tot_ret, SUM(used_qa_count) as tot_used,
                                  SUM(covered_count) as tot_cov, SUM(missed_count) as tot_miss
                           FROM question_feedback WHERE qa_id IN
                           (SELECT id FROM qa_pairs WHERE paper = ?)""",
                        (display_name,)
                    ).fetchone()
                    if fb_rows and fb_rows["tot_ret"]:
                        utility = fb_rows["tot_used"] / fb_rows["tot_ret"] * 100 if fb_rows["tot_ret"] else 0
                        _debug(f"  [Phase2] {display_name}: retrieved={fb_rows['tot_ret']}, "
                               f"used={fb_rows['tot_used']} (utility={utility:.0f}%), "
                               f"covered={fb_rows['tot_cov']}, missed={fb_rows['tot_miss']}")

                    # Miss category breakdown
                    cat_rows = db.conn.execute(
                        """SELECT miss_categories FROM question_feedback
                           WHERE qa_id IN (SELECT id FROM qa_pairs WHERE paper = ?)
                           AND miss_categories != ''""",
                        (display_name,)
                    ).fetchall()
                    if cat_rows:
                        totals = {"knowledge_gap": 0, "misinterpretation": 0,
                                  "insufficient_detail": 0, "retrieval_quality": 0}
                        for r in cat_rows:
                            try:
                                cats = json.loads(r["miss_categories"])
                                for k in totals:
                                    totals[k] += cats.get(k, 0)
                            except Exception:
                                pass
                        _debug(f"  [MissCat] {display_name}: " +
                               ", ".join(f"{k}={v}" for k, v in totals.items() if v > 0))
                except Exception:
                    pass

        # Link QAs to exam session for time-dimension queries
        if session_id:
            db.conn.execute(
                "UPDATE qa_pairs SET session_id = ? WHERE paper = ? AND session_id IS NULL",
                (session_id, display_name),
            )
            db.conn.commit()

        processed.add(display_name)
        try:
            with open(processed_path, "w") as f:
                json.dump(sorted(processed), f)
        except Exception as e:
            _debug(f"Failed to write processed.json: {e}")

        # Cross-paper consistency check after each paper
        try:
            from .pipeline_diagnostics import run_cross_paper_check
            run_cross_paper_check(db_path, display_name, debug_callback=_debug)
        except Exception as e:
            _debug(f"Cross-paper check failed (non-fatal): {e}")

    # -- Post-processing core (wrapped: ensures db.close() on failure) --
    content = ""
    try:
        # -- Topic merge --
        tracker.set_status("Merging similar topics...")
        _merge_similar_topics(db, client, _debug)
        for _ in range(3):
            tracker.step("")

        # -- Distill --
        tracker.set_status("Distilling knowledge points...")
        content = _distill_points(db, client, _debug)
        for _ in range(5):
            tracker.step("")

        # -- Review + write --
        content = _review_distilled(content, client, topic_links, topic_related, _debug)
        tracker.set_status("Writing output...")
        with open(subject_output, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        _log("Output", f"{db.count()} QAs -> {subject_output}")
        _progress(95, "Output written")
    except Exception as e:
        _log_stage_error("Core post-processing", _debug, e)
        if not content:
            content = "; ".join(
                qa["answer_text"] for g in groups.values() if g
                for qa in g[:30]
            )[:10000] or "; No knowledge points extracted."

    # Mark representative and cross-topic QAs
    weights = db.get_all_weights()
    groups = db.get_topic_groups()
    for topic, qas in groups.items():
        if not topic or topic == "(uncategorized)" or not qas:
            continue
        # Representative: highest Beta weight in topic
        best_qa = max(qas, key=lambda qa: weights.get(qa["id"], {}).get("mean", 0.5))
        db.conn.execute("UPDATE qa_pairs SET is_representative = 1 WHERE id = ?", (best_qa["id"],))
        # Cross-topic: references or is referenced by other topics
        for qa in qas:
            if any(dst == topic and src != topic for (src, dst) in topic_links) or \
               any(src == topic and dst != topic for (src, dst) in topic_links):
                db.conn.execute("UPDATE qa_pairs SET is_cross_topic = 1 WHERE id = ?", (qa["id"],))
    db.conn.commit()

    # Compute retrieval-based topic relations for cold-start See also
    topic_related = {}
    groups = db.get_topic_groups()
    for topic, qas in groups.items():
        if not topic or topic == "(uncategorized)":
            continue
        if any(src == topic or dst == topic for (src, dst) in topic_links):
            continue
        query = qas[0].get("knowledge_summary", "") or qas[0]["question_text"]
        results = retriever.search(query, threshold=0.5, min_k=3, max_cap=15)
        counts = {}
        for r in results:
            rt = r.get("topic", "")
            if rt and rt != topic:
                counts[rt] = counts.get(rt, 0) + 1
        if counts:
            topic_related[topic] = sorted(counts.items(), key=lambda x: -x[1])

    # -- Knowledge graph: QA clustering → KP nodes → edge discovery --
    try:
        from .knowledge_graph import run_knowledge_graph
        _debug("Building knowledge graph (clustering QAs into KPs)...")
        _log("Knowledge graph", "Starting")
        run_knowledge_graph(db_path, api_url, api_key,
                            debug_callback=_debug,
                            progress_callback=_progress)
    except Exception as e:
        _log_stage_error("Knowledge graph", _debug, e)

    # -- Adversarial refinement: challenger/defender debate on KP quality --
    try:
        from .adversarial_refiner import run_adversarial_refinement, auto_split_kp, auto_merge_kps
        _debug("Running adversarial refinement on KPs...")
        run_adversarial_refinement(db_path, api_url, api_key,
                                   debug_callback=_debug)
        # Auto-split over-broad KPs
        kps = db.get_all_kps()
        for kp in kps:
            if kp.get("evidence_count", 0) >= 6:
                auto_split_kp(db, kp["id"], client, debug_cb=_debug)
        # Auto-merge KPs flagged by cross-consistency
        from .adversarial_refiner import cross_kp_consistency
        all_kp_ids = [k["id"] for k in db.get_all_kps()]
        if len(all_kp_ids) >= 2:
            consistency = cross_kp_consistency(db, all_kp_ids[:30], client, debug_cb=_debug)
            merged_count = auto_merge_kps(db, consistency.get("issues", []), debug_cb=_debug)
            if merged_count:
                _debug(f"Auto-merge: {merged_count} KP pairs merged")
    except Exception as e:
        _log_stage_error("Adversarial refinement", _debug, e)

    # -- Offline analysis (command verbs, difficulty, dependencies) --
    try:
        from .offline_analyzer import run_offline_analysis
        _debug("Starting offline analysis (verbs, difficulty, dependencies)...")
        _log("Offline analysis", "Starting")
        run_offline_analysis(db_path, api_url, api_key,
                             progress_callback=_progress,
                             debug_callback=_debug)
    except Exception as e:
        _log_stage_error("Offline analysis", _debug, e)

    # -- Pipeline diagnostics (closed-loop + cross-paper) --
    try:
        from .pipeline_diagnostics import run_closed_loop
        _debug("Running pipeline diagnostics (closed-loop)...")
        run_closed_loop(db_path, api_url, api_key,
                        debug_callback=_debug)
    except Exception as e:
        _log_stage_error("Closed-loop diagnostics", _debug, e)

    # -- Self-evolving loop: detect drift, trigger re-review --
    try:
        _run_evolution_cycle(db, client, _debug)
    except Exception as e:
        _log_stage_error("Evolution cycle", _debug, e)

    _progress(100, "Analysis complete")
    db.close()
    return content


# ============================================================
# Topic merge
# ============================================================

def _merge_similar_topics(db: QADatabase, client, debug: Callable):
    """Merge topics with similar answer content. Batch-encodes all topics once."""
    groups = db.get_topic_groups()
    topics = [(t, qas) for t, qas in groups.items() if t and t != "(uncategorized)"]
    n = len(topics)
    if n < 2:
        return

    topic_names = [t for t, _ in topics]
    all_answers = [" ".join(qa["answer_text"] for qa in qas) for _, qas in topics]

    try:
        from .embedding_cluster import EmbeddingClusterer
        clusterer = EmbeddingClusterer(all_answers)
        vecs = clusterer.vectors
        cos_matrix = vecs @ vecs.T
    except Exception as e:
        debug(f"  Topic merge encoding failed: {e}")
        return

    mergers = {}
    done = set()
    ambiguous = []
    for i in range(n):
        if topic_names[i] in done:
            continue
        for j in range(i + 1, n):
            if topic_names[j] in done:
                continue
            cos = float(cos_matrix[i][j])
            if cos >= 0.85:
                # Canonical = topic with more QAs (higher frequency, not shorter name)
                cnt_i = len(groups.get(topic_names[i], []))
                cnt_j = len(groups.get(topic_names[j], []))
                if cnt_i >= cnt_j:
                    canonical = topic_names[i]
                    mergers[topic_names[j]] = canonical
                    done.add(topic_names[j])
                else:
                    canonical = topic_names[j]
                    mergers[topic_names[i]] = canonical
                    done.add(topic_names[i])
                debug(f"  Topic merge: -> '{canonical}' (cos={cos:.2f})")
            elif cos >= 0.30:
                ambiguous.append((topic_names[i], topic_names[j], cos))

    if ambiguous:
        merged_by_flash = _flash_review_merges(ambiguous, db, client, debug)
        mergers.update(merged_by_flash)
        for t in merged_by_flash:
            done.add(t)

    if mergers:
        merged_count = 0
        for old_topic, new_topic in mergers.items():
            rows = db.conn.execute("UPDATE qa_pairs SET topic=? WHERE topic=?", (new_topic, old_topic))
            merged_count += rows.rowcount

        # Sync topic_links: read all → rename in memory → aggregate → rewrite.
        # In-memory merge avoids UNIQUE constraint violations that UPDATE would cause
        # when multiple old topics map to the same canonical name.
        all_links = db.conn.execute(
            "SELECT src_topic, dst_topic, count FROM topic_links"
        ).fetchall()
        merged = {}
        for r in all_links:
            src = mergers.get(r["src_topic"], r["src_topic"])
            dst = mergers.get(r["dst_topic"], r["dst_topic"])
            if src == dst:
                continue
            key = (src, dst)
            merged[key] = merged.get(key, 0) + r["count"]
        db.conn.execute("DELETE FROM topic_links")
        for (src, dst), total in merged.items():
            db.conn.execute(
                "INSERT INTO topic_links (src_topic, dst_topic, count) VALUES (?, ?, ?)",
                (src, dst, total),
            )
        db.conn.commit()
        debug(f"  Topic merge: {len(mergers)} groups, {merged_count} QAs affected, "
              f"{len(all_links)} links → {len(merged)} after merge")


def _flash_review_merges(ambiguous: list, db: QADatabase, client, debug: Callable) -> dict:
    """Send ambiguous topic pairs to Flash for merge review."""
    if not ambiguous:
        return {}

    topic_texts = {}
    for t1, t2, _ in ambiguous:
        for t in (t1, t2):
            if t not in topic_texts:
                rows = db.conn.execute("SELECT answer_text FROM qa_pairs WHERE topic=?", (t,)).fetchall()
                topic_texts[t] = " ".join(r["answer_text"] for r in rows) if rows else ""

    # Detect language for prompts
    sample_text = " ".join(topic_texts.values())[:2000]
    lang = detect_content_lang(sample_text)
    if lang == 'en':
        sys = "Decide whether two topics should be merged. Output JSON."
        usr_tpl = ("Topic A '%s': %s\n\nTopic B '%s': %s\n\n"
                   "Cosine similarity: %.2f\n"
                   "Should these topics be merged? Similar names but different content -> do NOT merge.\n"
                   'If merge: {"merge": true, "canonical": "chosen name"} '
                   'If no merge: {"merge": false, "canonical": ""}')
    else:
        sys = "判断两个主题是否应合并。Output JSON."
        usr_tpl = ("主题A '%s': %s\n\n主题B '%s': %s\n\n"
                   "余弦相似度: %.2f\n"
                   "这两个主题是否应该合并? 注意: 标题相似但考察内容不同则不应合并。\n"
                   '若合并: {"merge": true, "canonical": "保留的主题名"} '
                   '若不合并: {"merge": false, "canonical": ""}')

    mergers = {}

    def _review_one(t1, t2, cos):
        a1 = topic_texts.get(t1, "")
        a2 = topic_texts.get(t2, "")
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": usr_tpl % (t1, a1[:500], t2, a2[:500], cos)},
        ]
        try:
            result = call_flash(client, messages, max_retries=1, debug_callback=debug)
            if isinstance(result, dict) and result.get("merge"):
                canonical = result.get("canonical", "")
                if not canonical:
                    return None
                return (t2, canonical)
        except Exception as e:
            debug(f"  Flash merge review failed for '{t1}'/'{t2}': {e}")
        return None

    with ThreadPoolExecutor(max_workers=_get_worker_limit(len(ambiguous), api_heavy=True)) as executor:
        futures = {executor.submit(_review_one, t1, t2, cos): (t1, t2) for t1, t2, cos in ambiguous}
        for future in as_completed(futures):
            result = future.result()
            if result:
                t2, canonical = result
                if t2 not in mergers or len(canonical) < len(mergers[t2]):
                    mergers[t2] = canonical
                debug(f"  Flash merge: '{t2}' -> '{canonical}'")

    return mergers


# ============================================================
# Distillation
# ============================================================

def _build_missed_ref(db: QADatabase, topic: str, qas: list[dict], debug: Callable) -> str:
    """Build a reference text of recurring difficulty patterns from Phase 2 missed_points.
    Only returns patterns that pass MS similarity filter and appear >= 2 times."""
    raw_missed = db.get_missed_by_topic(topic)
    if len(raw_missed) < 3:
        return ""

    # Collect all answer_texts in this topic as filter reference
    all_answers = [qa["answer_text"] for qa in qas if qa.get("answer_text")]
    if not all_answers:
        return ""

    try:
        from .embedding_cluster import _get_model, MODEL_MAP, _detect_language
        model = _get_model(MODEL_MAP[_detect_language(raw_missed + all_answers)])
        missed_vecs = model.encode(raw_missed, normalize_embeddings=True, convert_to_numpy=True)
        answer_vecs = model.encode(all_answers, normalize_embeddings=True, convert_to_numpy=True)
    except Exception as e:
        debug(f"  missed embedding failed for '{topic}': {e}")
        return ""

    # Filter: keep missed lines similar to any answer_text (cos >= 0.60)
    filtered = []
    for i, line in enumerate(raw_missed):
        best_cos = max(float(np.dot(missed_vecs[i], av)) for av in answer_vecs)
        if best_cos >= 0.60:
            filtered.append(line)

    if len(filtered) < 2:
        return ""

    # Cluster: group by cos >= 0.80, keep groups with >= 2 members
    try:
        fvecs = model.encode(filtered, normalize_embeddings=True, convert_to_numpy=True)
    except Exception as e:
        debug(f"  missed cluster encoding failed for '{topic}': {e}")
        return ""
    n = len(filtered)
    assigned = [False] * n
    patterns = []
    for i in range(n):
        if assigned[i]:
            continue
        group = [filtered[i]]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if float(np.dot(fvecs[i], fvecs[j])) >= 0.80:
                group.append(filtered[j])
                assigned[j] = True
        if len(group) >= 2:
            patterns.append(group[0])

    if not patterns:
        return ""

    ref = ("\nReference: the following difficulty patterns were observed when answering "
           "similar questions. Use them to inform pitfalls ONLY IF they align with "
           "markscheme scoring criteria:\n")
    for p in patterns[:5]:
        ref += f"- {p}\n"
    debug(f"  missed patterns for '{topic}': {len(patterns)} from {len(filtered)}/{len(raw_missed)} lines")
    return ref


def _distill_points(db: QADatabase, client, debug: Callable) -> str:
    """Distill points.txt from accumulated QA pairs, grouped by topic."""
    groups = db.get_topic_groups()
    if not groups:
        return ""

    weights = db.get_all_weights()

    topic_meta = {}
    for topic, qas in groups.items():
        if not topic or topic == "(uncategorized)":
            continue
        qa_means = [weights.get(qa["id"], {}).get("mean", 0.5) for qa in qas]
        best_lb = max((weights.get(qa["id"], {}).get("lower_bound", 0) for qa in qas), default=0)
        best_mean = max(qa_means, default=0.5)
        variance = statistics.variance(qa_means) if len(qa_means) >= 2 else 0.0
        quality_flag = ""
        if best_mean < 0.4 and variance < 0.02 and len(qa_means) >= 2:
            quality_flag = " [low quality - review suggested]"
        topic_meta[topic] = {"lb": best_lb, "mean": best_mean, "n_qa": len(qas),
                             "variance": variance, "quality_flag": quality_flag}

    # Detect overall language for prompts
    all_text = " ".join(qa["question_text"] for g in groups.values() for qa in g)[:3000]
    lang = detect_content_lang(all_text)

    if lang == 'en':
        dist_sys = (
            "You are a knowledge distillation expert. Extract general knowledge points "
            "from exam answers. Output JSON. For each KP include: concept, detail, "
            "pitfall (common exam mistake, not extra knowledge), "
            "scoring (what earns marks + a sample full-mark answer phrase), "
            "confidence (\"high\" if from multiple/high-weight QAs, \"low\" if from single/low-weight QA). "
            "Prioritize high-weight QAs. "
            'Example: {"knowledge_points": ['
            '{"concept":"Binary addition proceeds column by column from LSB to MSB",'
            '"detail":"Per-bit rules: 0+0=0, 0+1=1, 1+1=0 carry 1",'
            '"pitfall":"Forgetting to add carry bits to the next column",'
            '"scoring":"Show working carries and final result. E.g. \'1+1=0 carry 1 to next column\'",'
            '"confidence":"high"}]}'
        )
        dist_usr = (
            "Topic: %s\n\n"
            "The following %d questions cover this topic:\n\n%s\n"
            "Task: distill general knowledge points. Format each as:\n"
            "  - concept: 1-sentence statement\n"
            "  - detail: key specifics or worked example\n"
            "  - pitfall: common exam mistake (not extra facts)\n"
            "  - scoring: what earns marks + sample answer phrase\n"
            "  - confidence: high|low\n\n"
            "Remove question-specific details (values, names). Keep common principles.\n"
            "Use consistent terminology: if source QAs use different names for the same concept, "
            "pick the most frequent term and use it throughout all KPs in this topic.\n"
            'Return JSON: {"knowledge_points": [{"concept":"...","detail":"...","pitfall":"...","scoring":"...","confidence":"high"}]}'
        )
    else:
        dist_sys = (
            "你是一个知识蒸馏专家。从题目的答案中提取通用知识点。Output JSON。"
            "每个知识点包含: concept(概念), detail(细节/例子), "
            "pitfall(常见考试错误，非额外知识点), "
            "scoring(得分要点 + 满分答案示例措辞), "
            "confidence(high=来自多个/高权重QA, low=来自单个/低权重QA)。"
            "优先从高权重QA提取。"
            '示例: {"knowledge_points": ['
            '{"concept":"二进制加法从LSB到MSB逐列进行",'
            '"detail":"逐位规则: 0+0=0, 0+1=1, 1+1=0进位1",'
            '"pitfall":"忘记将进位加到下一列",'
            '"scoring":"展示进位过程和最终结果。如 \'1+1=0 进位1至下一列\'",'
            '"confidence":"high"}]}'
        )
        dist_usr = (
            "主题: %s\n\n"
            "以下 %d 道题目涉及此主题:\n\n%s\n"
            "任务: 蒸馏出通用知识点。格式:\n"
            "  - concept: 1句话概念陈述\n"
            "  - detail: 关键细节或计算示例\n"
            "  - pitfall: 常见考试错误(非额外知识点)\n"
            "  - scoring: 得分要点 + 满分答案示例措辞\n"
            "  - confidence: high|low\n\n"
            "去除题目特定细节（数值、名称），保留共性技术原理。\n"
            "术语一致: 若多个题目的答案使用不同名称指代同一概念，"
            "选择出现最多的术语，在本主题的所有 KP 中统一使用。\n"
            '返回 JSON: {"knowledge_points": [{"concept":"...","detail":"...","pitfall":"...","scoring":"...","confidence":"high"}]}'
        )

    # Load distillation cache for incremental reuse
    cache = db.get_distillation_cache()

    # Pre-build topic list with precomputed text (no DB access needed in workers)
    topic_items = []       # topics to re-distill
    cached_results = {}    # topic -> cached content (reused as-is)
    skipped_count = 0
    for topic, qas in groups.items():
        if not topic or topic == "(uncategorized)":
            continue
        meta = topic_meta.get(topic)
        if not meta:
            continue
        if meta["lb"] < 0.25:
            if meta["n_qa"] > 1 and meta["lb"] < 0.15:
                continue
            marker = "  [needs review]"
        elif meta["lb"] >= 0.5:
            marker = "  [core]"
        else:
            marker = ""
        marker += meta.get("quality_flag", "")

        # Compute fingerprint of the topic's QA set
        qa_ids_sorted = sorted(qa["id"] for qa in qas)
        qa_ids_hash = hashlib.md5(",".join(map(str, qa_ids_sorted)).encode()).hexdigest()
        cached_state = db.get_cached_topic_state(topic)

        if (cached_state
                and cached_state["qa_count"] == len(qas)
                and cached_state["qa_ids_hash"] == qa_ids_hash
                and topic in cache):
            cached_results[topic] = cache[topic]
            skipped_count += 1
            continue

        qa_texts = ""
        # Sort by Beta weight descending: high-weight QAs first for model attention
        qas_sorted = sorted(
            qas,
            key=lambda qa: weights.get(qa["id"], {}).get("mean", 0.5),
            reverse=True,
        )
        for i, qa in enumerate(qas_sorted):
            w = weights.get(qa["id"], {})
            qa_texts += (f"Q{i+1} [{qa['paper']}] (weight={w.get('mean',0.5):.2f}): "
                         f"{qa['question_text']}\nA: {qa['answer_text']}\n\n")

        # Append missed-pattern reference if available
        missed_ref = _build_missed_ref(db, topic, qas, debug)
        if missed_ref:
            qa_texts += missed_ref

        # Low-weight topic hint: be conservative with limited data
        if meta["n_qa"] <= 2 and meta["mean"] < 0.5:
            if lang == 'en':
                qa_texts += ("\nNote: this topic has limited QA data. Only extract "
                             "knowledge points clearly and directly supported by the "
                             "answers above. Prefer fewer high-confidence KPs.\n")
            else:
                qa_texts += ("\n注意: 此主题的题目数据有限。请仅提取答案中明确直接支持的"
                             "知识点。宁可输出少量高置信度的知识点，也不要输出多个低置信度的。\n")

        topic_items.append((topic, len(qas), qa_texts, marker, qas, qa_ids_hash))

    if skipped_count:
        debug(f"Incremental: reusing {skipped_count} cached, distilling {len(topic_items)} topics")

    if not topic_items and not cached_results:
        return ""

    # Split topics: small topics (n_qa <= 3) get batched, large topics get individual calls
    SMALL_TOPIC_THRESHOLD = 3
    BATCH_SIZE = 5
    small_topics = [(t, n, txt, m, q, h) for t, n, txt, m, q, h in topic_items if n <= SMALL_TOPIC_THRESHOLD]
    large_topics = [(t, n, txt, m, q, h) for t, n, txt, m, q, h in topic_items if n > SMALL_TOPIC_THRESHOLD]

    batches = []
    for i in range(0, len(small_topics), BATCH_SIZE):
        batches.append(small_topics[i:i + BATCH_SIZE])

    if batches:
        debug(f"Batching: {len(small_topics)} small topics into {len(batches)} batches, "
              f"{len(large_topics)} large topics individually")

    def _distill_batch(batch_topics):
        """Distill multiple small topics in one API call."""
        # Build combined prompt: each topic with its QAs
        sections = []
        for topic, n_qa, qa_texts, marker, qas, qa_ids_hash in batch_topics:
            sections.append(f"=== Topic: {topic} ===\n{qa_texts}")
        combined = "\n".join(sections)

        if lang == 'en':
            batch_sys = (
                "You are a knowledge distillation expert. Extract knowledge points "
                "from MULTIPLE topics in one pass. Output JSON with a 'topics' array. "
                'Example: {"topics": [{"topic": "TopicA", "knowledge_points": ['
                '{"concept":"...","detail":"...","pitfall":"...","scoring":"...","confidence":"high"}]}]}'
            )
            batch_usr = (
                "Distill knowledge points for ALL topics below. "
                "For each topic, return 1-3 knowledge points. "
                "Remove question-specific details, keep common principles.\n\n%s\n\n"
                'Return JSON: {"topics": [{"topic": "exact topic name", '
                '"knowledge_points": [...]}, ...]}'
            )
        else:
            batch_sys = (
                "你是一个知识蒸馏专家。一次性从多个主题中提取知识点。Output JSON。"
                '格式: {"topics": [{"topic": "主题名", "knowledge_points": ['
                '{"concept":"...","detail":"...","pitfall":"...","scoring":"...","confidence":"high"}]}]}'
            )
            batch_usr = (
                "为以下所有主题蒸馏知识点。每个主题输出1-3个知识点。"
                "去除题目特定细节，保留共性技术原理。\n\n%s\n\n"
                '返回 JSON: {"topics": [{"topic": "确切的主题名", '
                '"knowledge_points": [...]}, ...]}'
            )

        messages = [
            {"role": "system", "content": batch_sys},
            {"role": "user", "content": batch_usr % combined},
        ]
        try:
            result = call_flash(client, messages, max_retries=1, debug_callback=debug)
            flash_result = result.get("topics", [])
        except Exception as e:
            debug(f"  batch distillation failed: {e}")
            flash_result = []

        batch_results = {}
        for td in flash_result:
            t_name = td.get("topic", "")
            kps = td.get("knowledge_points", [])
            if not t_name or not kps:
                continue
            # Find the matching batch item to get marker and qa_ids_hash
            marker = ""
            qa_ids_hash = ""
            n_qa = 0
            for bt_topic, bt_n_qa, _, bt_marker, _, bt_hash in batch_topics:
                if bt_topic.strip().lower() == t_name.strip().lower():
                    marker = bt_marker
                    n_qa = bt_n_qa
                    qa_ids_hash = bt_hash
                    break
            else:
                debug(f"  Batch topic name mismatch: model returned '{t_name}'")
            parts = [f"{t_name}{marker}"]
            for i, kp in enumerate(kps, 1):
                if isinstance(kp, str):
                    parts.append(f"{i}. {kp}")
                else:
                    conf = kp.get('confidence', 'high')
                    prefix = "(review) " if conf == 'low' else ""
                    parts.append(f"{i}. {prefix}{kp.get('concept') or kp.get('detail', str(kp))}")
                    if kp.get('detail'):
                        parts.append(f"   Detail: {kp['detail']}")
                    if kp.get('pitfall'):
                        parts.append(f"   Pitfall: {kp['pitfall']}")
                    if kp.get('scoring'):
                        parts.append(f"   Scoring: {kp['scoring']}")
            content = "\n".join(parts)
            db.upsert_distillation_cache(t_name, n_qa, qa_ids_hash, content)
            batch_results[t_name] = content
        # Log topics the model missed from the batch (will fall back to individual)
        returned_topics = {td.get("topic", "").strip().lower() for td in flash_result}
        for bt_topic, _, _, _, _, _ in batch_topics:
            if bt_topic.strip().lower() not in returned_topics:
                debug(f"  Batch missed topic '{bt_topic}', falling back to individual")

        return batch_results

    def _distill_one_topic(topic, n_qa, qa_texts, marker, qas, qa_ids_hash):
        messages = [
            {"role": "system", "content": dist_sys},
            {"role": "user", "content": dist_usr % (topic, n_qa, qa_texts)},
        ]
        try:
            result = call_flash(client, messages, max_retries=1, debug_callback=debug)
            kps = result.get("knowledge_points", [])
        except Exception as e:
            debug(f"  distillation failed for '{topic}': {e}")
            kps = []

        if not kps:
            # On failure, check cache — reuse old content rather than fabricating from raw answers
            if topic in cache:
                debug(f"  distillation failed for '{topic}', reusing cached content")
                return (topic, cache[topic])
            return (topic, "")
        parts = [f"{topic}{marker}"]
        for i, kp in enumerate(kps, 1):
            if isinstance(kp, str):
                parts.append(f"{i}. {kp}")
            else:
                conf = kp.get('confidence', 'high')
                prefix = "(review) " if conf == 'low' else ""
                parts.append(f"{i}. {prefix}{kp.get('concept') or kp.get('detail', str(kp))}")
                if kp.get('detail'):
                    parts.append(f"   Detail: {kp['detail']}")
                if kp.get('pitfall'):
                    parts.append(f"   Pitfall: {kp['pitfall']}")
                if kp.get('scoring'):
                    parts.append(f"   Scoring: {kp['scoring']}")
        content = "\n".join(parts)
        # Cache the successful result for future incremental runs
        db.upsert_distillation_cache(topic, n_qa, qa_ids_hash, content)
        return (topic, content)

    results = {}
    # Track which small topics were covered by batch results
    batched_topics = set()

    if batches or large_topics:
        with ThreadPoolExecutor(max_workers=_get_worker_limit(len(batches) + len(large_topics), api_heavy=True)) as executor:
            future_map = {}

            # Submit batch tasks
            for batch in batches:
                f = executor.submit(_distill_batch, batch)
                future_map[f] = ("batch", [t for t, _, _, _, _, _ in batch])

            # Submit individual tasks for large topics
            for item in large_topics:
                f = executor.submit(_distill_one_topic, *item)
                future_map[f] = ("single", [item[0]])

            for future in as_completed(future_map):
                try:
                    mode, topics_in_task = future_map[future]
                    result_val = future.result()
                    if mode == "batch":
                        for topic, text in result_val.items():
                            if text:
                                results[topic] = text
                                batched_topics.add(topic)
                    else:
                        topic, text = result_val
                        if text:
                            results[topic] = text
                except Exception as e:
                    _, topics_in_task = future_map.get(future, ("unknown", []))
                    debug(f"  distillation thread failed for {topics_in_task}: {e}")

        # Fallback: any small topic not covered by batch gets individual distillation
        missed_small = [item for item in small_topics if item[0] not in results]
        if missed_small:
            debug(f"  Batch missed {len(missed_small)} topics, running individual distillation")
            for item in missed_small:
                try:
                    topic, text = _distill_one_topic(*item)
                    if text:
                        results[topic] = text
                except Exception as e:
                    debug(f"  fallback distillation failed for '{item[0]}': {e}")

    # Assemble output in original topic order: newly distilled + cached
    all_lines = []
    for topic in groups:
        if topic in results:
            all_lines.append(results[topic])
        elif topic in cached_results:
            all_lines.append(cached_results[topic])

    return "\n\n".join(all_lines) + "\n"


# ============================================================
# Phase 2: KP generation for stable topics
# ============================================================

def _generate_kp_for_stable_topics(db: QADatabase, client, debug: Callable):
    """Generate KP text for topics that reached 'forming' quality via Flash."""
    topics = db.conn.execute(
        "SELECT * FROM dynamic_topics WHERE quality='forming'"
    ).fetchall()
    if not topics:
        return

    debug(f"Phase 2: generating KP for {len(topics)} forming topics")

    for topic in topics:
        topic_id = topic["topic_id"]
        frags = db.get_topic_fragments(topic_id)
        if not frags:
            continue

        # Collect fragment texts
        frag_texts = db.conn.execute(
            "SELECT point_text FROM ms_fragments WHERE point_id IN ({})".format(
                ",".join("?" * len(frags))),
            frags
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
            result = call_flash(client, messages, max_retries=1, debug_callback=debug)
            concept = result.get("concept", topic["name"]) if isinstance(result, dict) else topic["name"]
            detail = result.get("detail", "") if isinstance(result, dict) else ""
        except Exception as e:
            debug(f"  KP generation failed for {topic_id}: {e}")
            concept = topic["name"]
            detail = ""

        db.set_topic_kp(topic_id, concept, detail)
        debug(f"  KP generated: [{topic_id}] {concept[:80]}")


# ============================================================
# Self-evolving loop
# ============================================================

def _run_evolution_cycle(db: QADatabase, client, debug: Callable):
    """Run self-evolution: detect drift, trigger re-review for degraded KPs.

    Observes changes since last analysis, generates improvement proposals,
    and auto-accepts low-risk refinements.
    """
    # -- Phase 2: Fragment migration + topic stats --
    try:
        from .pipeline_diagnostics import run_phase2_cycle
        result = run_phase2_cycle(db_path, debug_cb=debug)
        if result.get("migrated", 0) > 0:
            debug(f"Evolution: {result['migrated']} fragments migrated")
    except Exception as e:
        debug(f"  Phase 2 cycle failed (non-fatal): {e}")

    # -- Phase 2: Generate KP for stable/forming topics --
    try:
        _generate_kp_for_stable_topics(db, client, debug)
    except Exception as e:
        debug(f"  KP generation for stable topics failed (non-fatal): {e}")

    kps = db.conn.execute("SELECT * FROM knowledge_points").fetchall()
    if not kps:
        return

    scores_above = db.conn.execute(
        "SELECT COUNT(*) as cnt FROM qa_pairs WHERE success_count > total_attempts"
    ).fetchone()
    if scores_above and scores_above["cnt"] > 0:
        debug("Evolution: fixing inconsistent attempt counters")
        db.conn.execute(
            "UPDATE qa_pairs SET success_count = total_attempts "
            "WHERE success_count > total_attempts"
        )
        db.conn.commit()

    # Detect KPs with quality 'disputed' that haven't been re-reviewed
    disputed = [dict(k) for k in kps if k["quality"] == "disputed"]
    if disputed:
        debug(f"Evolution: {len(disputed)} disputed KPs — queuing for re-review")
        from .adversarial_refiner import refine_kp
        from concurrent.futures import ThreadPoolExecutor, as_completed
        targets = disputed[:3]  # Cap at 3 per cycle to control API cost
        with ThreadPoolExecutor(max_workers=min(len(targets), 3)) as executor:
            futures = {executor.submit(refine_kp, db, kp["id"], client, debug): kp["id"]
                       for kp in targets}
            for future in as_completed(futures):
                kp_id = futures[future]
                try:
                    result = future.result()
                    new_quality = result.get("quality", "unknown")
                    db.record_evolution(
                        kp_id=kp_id,
                        trigger_type="auto_re-review",
                        trigger_detail="disputed KP re-evaluated in evolution cycle",
                        old_state="quality=disputed",
                        new_state=f"quality={new_quality}",
                        outcome="completed",
                    )
                except Exception as e:
                    debug(f"  Evolution re-review failed for {kp_id}: {e}")

    # Detect KPs that have accumulated new QAs since last review
    for kp in kps:
        if kp["quality"] in ("draft", "accepted", "disputed"):
            # Check if QA count grew significantly since last validation
            member_rows = db.conn.execute(
                "SELECT COUNT(*) as cnt FROM qa_kp_membership WHERE kp_id=?",
                (kp["id"],),
            ).fetchone()
            current_members = member_rows["cnt"] if member_rows else 0
            prev_evidence = kp.get("evidence_count", 0) or 0
            if current_members > prev_evidence and current_members >= 5:
                growth = current_members - prev_evidence
                if growth >= 3 or current_members >= prev_evidence * 1.5:
                    db.record_evolution(
                        kp_id=kp["id"],
                        trigger_type="evidence_growth",
                        trigger_detail=f"QA members: {prev_evidence} → {current_members} (+{growth})",
                        old_state=f"evidence_count={prev_evidence}",
                        new_state=f"evidence_count={current_members}",
                        outcome="queued",
                    )
                    db.conn.execute(
                        "UPDATE knowledge_points SET evidence_count=? WHERE id=?",
                        (current_members, kp["id"]),
                    )
                    db.conn.commit()

    pending_count = len(db.get_pending_evolutions())
    if pending_count:
        debug(f"Evolution: {pending_count} improvement proposals queued")

    # Apply student feedback: confusion events → difficulty adjustment
    try:
        from .pipeline_diagnostics import apply_student_feedback
        apply_student_feedback(db)
    except Exception as e:
        debug(f"  Student feedback loop failed (non-fatal): {e}")

    # Detect outlier QAs — potential new topics
    try:
        outlier_count = _detect_outlier_qas(db, debug)
        if outlier_count:
            debug(f"Evolution: {outlier_count} outlier QAs flagged for review")
    except Exception as e:
        debug(f"  Outlier detection failed (non-fatal): {e}")


# ============================================================
# Outlier QA detection
# ============================================================

def _detect_outlier_qas(db: QADatabase, debug: Callable) -> int:
    """Detect QAs drifting from their topic centroid — potential new topics.

    For each topic with >= 5 QAs, computes the centroid of answer embeddings
    and flags QAs whose cosine distance to centroid exceeds 2 standard deviations.
    """
    groups = db.get_topic_groups()
    if not groups:
        return 0

    from .embedding_cluster import TOPIC_EMBED_MODEL, _get_model
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
        except Exception:
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
            if d > mean_dist + 2.0 * stdev and d > 0.25:
                qa = qas[i]
                db.conn.execute(
                    "UPDATE qa_pairs SET last_failure_reason=? WHERE id=?",
                    (f"outlier: dist={d:.3f} from topic '{topic}' centroid", qa["id"]),
                )
                db.conn.commit()
                flagged += 1

    return flagged


# ============================================================
# Post-distillation review
# ============================================================

def _review_distilled(content: str, client, topic_links: dict, topic_related: dict, debug: Callable) -> str:
    """Post-distillation review in focused batches.

    Batch A (LLM): structural — duplicates, topic mismatches (checks 1-2)
    Batch B (LLM): content — scoring examples, calculation steps (checks 7-8)
    Rules-based: formatting normalization (checks 3-6)
    Then insert See also / Related from topic_links and topic_related.
    """
    if not content.strip() or len(content) < 500:
        return content

    lang = detect_content_lang(content)

    # ---- Batch A: Structural review (checks 1-2) ----
    if lang == 'en':
        sys_a = "You are a knowledge base reviewer. Find structural issues. Output JSON."
        usr_a = (f"{content}\n\n"
                 "Fix these structural issues:\n"
                 "1. Duplicate KPs across topics — merge into best topic\n"
                 "2. KP content contradicts its topic name OR topic name is too broad/narrow — fix topic or remove\n"
                 "Preserve everything else as-is.\n"
                 'Return JSON: {"content": "corrected file content"}')
    else:
        sys_a = "你是知识库结构审核专家。查找结构问题。Output JSON。"
        usr_a = (f"{content}\n\n"
                 "修复以下结构问题:\n"
                 "1. 不同topic下的重复KP — 合并到最合适的topic\n"
                 "2. KP内容与topic名矛盾或topic名不匹配 — 修正topic名或移除\n"
                 "保留其他内容不变。\n"
                 '返回 JSON: {"content": "修正后的完整文件"}')

    reviewed = content
    try:
        result = call_flash(client, [{"role": "system", "content": sys_a},
                                      {"role": "user", "content": usr_a}],
                           max_retries=1, debug_callback=debug)
        if isinstance(result, dict) and result.get("content"):
            reviewed = result["content"]
    except Exception as e:
        debug(f"  review batch A failed: {e}")

    # ---- Batch B: Content review (checks 7-8) ----
    if lang == 'en':
        sys_b = "You are a knowledge base reviewer. Fix scoring-related issues. Output JSON."
        usr_b = (f"{reviewed}\n\n"
                 "Fix these content issues:\n"
                 "7. Scoring fields missing concrete example answer sentence — add a sample full-mark answer in quotes\n"
                 "8. Calculation KPs scoring missing step-by-step mark allocation — add mark breakdown per step\n"
                 "Preserve everything else as-is.\n"
                 'Return JSON: {"content": "corrected file content"}')
    else:
        sys_b = "你是知识库内容审核专家。修复评分相关问题。Output JSON。"
        usr_b = (f"{reviewed}\n\n"
                 "修复以下内容问题:\n"
                 "7. Scoring字段缺少具体答案示例 — 补充带引号的完整答案范例\n"
                 "8. 计算类KP的Scoring缺少分步给分 — 补充每步分值\n"
                 "保留其他内容不变。\n"
                 '返回 JSON: {"content": "修正后的完整文件"}')

    try:
        result = call_flash(client, [{"role": "system", "content": sys_b},
                                      {"role": "user", "content": usr_b}],
                           max_retries=1, debug_callback=debug)
        if isinstance(result, dict) and result.get("content"):
            reviewed = result["content"]
    except Exception as e:
        debug(f"  review batch B failed: {e}")

    # ---- Rules-based pass: formatting normalization (checks 3-6) ----
    reviewed = _normalize_formatting(reviewed)

    # ---- Insert See also from accumulated topic_links + topic_related ----
    if topic_links or topic_related:
        reviewed = _insert_see_also(reviewed, topic_links, topic_related, debug)

    return reviewed


def _normalize_formatting(content: str) -> str:
    """Rules-based formatting normalization (no LLM call)."""
    lines = content.split("\n")
    result = []
    prev_blank = False
    for line in lines:
        # Normalize bullet styles: ensure consistent "  - " prefix
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            line = "  " + "- " + stripped[2:]
        # Strip trailing whitespace
        line = line.rstrip()
        # Normalize multiple blank lines to single
        is_blank = not line.strip()
        if is_blank:
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        result.append(line)
    return "\n".join(result)


def _insert_see_also(content: str, topic_links: dict, topic_related: dict, debug: Callable) -> str:
    """Insert See also / Related lines based on cross-topic references.
    topic_links (strong): Phase 2 runtime cross-topic QA usage → "See also: X (ref N)"
    topic_related (medium): retrieval co-occurrence → "Related: X"
    """
    if not topic_links and not topic_related:
        return content

    # Build per-topic annotation lists
    annotations: dict[str, list[str]] = {}

    for (src, dst), count in topic_links.items():
        if count >= 2 and src and dst:
            annotations.setdefault(src, []).append(f"See also: {dst} (ref {count})")

    for src, refs in topic_related.items():
        if src not in annotations:  # only if no stronger signal already
            parts = [f"{rt}" for rt, cnt in refs[:3] if cnt >= 2]
            if parts:
                annotations.setdefault(src, []).append(f"Related: {', '.join(parts)}")

    if not annotations:
        return content

    # Parse content lines, insert annotations after each topic section
    lines = content.split("\n")
    out: list[str] = []
    seen_sections: set[str] = set()
    # Sort by length descending to prevent prefix collisions (e.g. "Data" vs "Data Compression")
    topics_by_len = sorted(annotations.keys(), key=len, reverse=True)

    for line in lines:
        stripped = line.strip()
        current_topic = ""
        for topic in topics_by_len:
            if stripped.startswith(topic) and (stripped == topic or stripped[len(topic)] == ' '):
                current_topic = topic
                break
            if stripped.startswith(topic + "  [") or stripped.startswith(topic + " ["):
                current_topic = topic
                break

        if current_topic and current_topic not in seen_sections:
            out.append(line)
            for note in annotations.get(current_topic, []):
                out.append(f"   {note}")
            seen_sections.add(current_topic)
        else:
            out.append(line)

    debug(f"  See also: {len(seen_sections)} topics annotated "
          f"(links={len(topic_links)}, related={len(topic_related)})")
    return "\n".join(out)
