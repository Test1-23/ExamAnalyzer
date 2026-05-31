"""Pipeline v4: QA-based knowledge accumulation with Flash summaries.

Phase 1: Store QA pairs + Flash-generated knowledge summaries.
Phase 2: Flash summary -> embedding retrieval -> Pro answer -> Flash grade -> store.
"""

import os
import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

from .deepseek_client import create_client, call_flash
from .file_pairer import pair_files
from .adversarial_refiner import auto_split_kp, auto_merge_kps, cross_kp_consistency
from .knowledge_base import QADatabase, QARetriever, make_topic_id, log_schema_status
from .knowledge_graph import run_knowledge_graph
from .models import ExtractedPair, QAPair
from .offline import run_offline_analysis
from .pdf_extractor import extract_pdf
from .diagnostics import run_closed_loop, run_cross_paper_check
from .embedding_cluster import detect_content_lang
from .prompt_factory import FRAGMENT, QA_CLASSIFY, PromptType, PromptBuilder
from .prompts.pipeline_prompts import stage2_qa_pairing, _generate_summary, _build_answer_prompt, _build_grade_prompt
from .topic_merger import merge_similar_topics
from .reviewer import review_distilled
from .evolution import run_evolution_cycle
from .distiller import Distiller
from .utils import get_worker_limit, log_stage_error

from .constants import (
    RETRIEVAL_THRESHOLD, RETRIEVAL_MIN_K, RETRIEVAL_MAX_CAP,
    CENTRALITY_HIGH, CENTRALITY_MID, CENTRALITY_LOW,
    HELP_DIRECT_THRESHOLD, HELP_UNDERSTANDING_THRESHOLD,
    SEASON_MAP, YEAR_BASE_OFFSET,
)

# ============================================================


# ============================================================
# -- Data helpers --

def _parse_missed_points(missed_raw: list) -> tuple[list[str], str]:
    """Parse missed_points from grading result.

    New format: [{"point":"...", "reason":"..."}, ...]
    Old format: ["...", ...] — flatten for backward compat.
    Returns (missed_texts, miss_cats_json).
    """
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
    return missed_texts, miss_cats_json


def _record_fragment_help(used_ids: set, covered: list, missed_texts: list,
                          db: QADatabase, qa_id: int) -> None:
    """Record which fragments from used QAs helped answer this question."""
    if not used_ids:
        return
    for used_qa_id in used_ids:
        used_frags = db.conn.execute(
            "SELECT point_id FROM ms_fragments WHERE qa_id=?",
            (used_qa_id,)
        ).fetchall()
        frag_ids = [r["point_id"] for r in used_frags]
        if frag_ids:
            help_effect = len(covered) / max(len(covered) + len(missed_texts), 1)
            help_level = "direct" if help_effect >= HELP_DIRECT_THRESHOLD else (
                "understanding" if help_effect >= HELP_UNDERSTANDING_THRESHOLD else "none")
            for fid in frag_ids:
                db.record_fragment_help_with_level(
                    fid, qa_id, round(help_effect, 3), help_level)



def _extract_ms_fragments(answer_text: str, qa_id: int, client, debug: Callable) -> list[dict]:
    """Split a mark scheme answer into individual scoring points. Preserves original wording."""
    messages = FRAGMENT.build(lang_source=answer_text, answer_text=answer_text)
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


def _classify_qa_against_kps(qa_text: str, answer_text: str, kp_concepts: list[dict],
                              client, debug: Callable) -> dict[str, float]:
    """Layer 1: LLM judges QA relevance against all existing KPs.
    Returns {kp_id: relevance_score [0,1]}."""
    if not kp_concepts:
        return {}

    MAX_KPS = 50
    if len(kp_concepts) > MAX_KPS:
        # Keep top KPs by evidence_count (most validated)
        kp_concepts = sorted(kp_concepts, key=lambda k: k.get("evidence", 0), reverse=True)[:MAX_KPS]

    kp_list = "\n".join(f"[{k['id']}] {k['concept']}" for k in kp_concepts)
    messages = QA_CLASSIFY.build(lang_source=qa_text + answer_text,
                                  qa_text=qa_text[:500], answer_text=answer_text[:500],
                                  kp_list=kp_list)
    try:
        result = call_flash(client, messages, max_retries=1, debug_callback=debug)
        scores = result.get("kp_scores", {}) if isinstance(result, dict) else {}
    except Exception as e:
        debug(f"  KP classification failed for QA: {e}")
        scores = {}

    return {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}


def _place_qa_vector_from_kp_scores(db: QADatabase, qa_id: int,
                                      kp_scores: dict[str, float], debug: Callable):
    """Place a new QA's initial vector based on LLM KP relevance scores."""
    if not kp_scores:
        return

    # Store QA-KP relevance scores
    for kp_id, score in kp_scores.items():
        if score >= 0.3:
            db.upsert_qa_kp_score(qa_id, kp_id, round(score, 3))

    # Determine initial Topic and centrality
    best_kp = max(kp_scores, key=kp_scores.get)
    best_score = kp_scores[best_kp]
    kp_data = db.get_kp_by_id(best_kp)
    topic = kp_data.get("name", "") if kp_data else ""
    if not topic:
        # Name-less KP: derive from description or skip (don't pollute with raw kp_id)
        topic = kp_data.get("description", "") if kp_data else ""
        topic = topic[:80] if topic else ""

    centrality = CENTRALITY_HIGH if best_score >= CENTRALITY_HIGH else (CENTRALITY_MID if best_score >= CENTRALITY_MID else CENTRALITY_LOW)

    # Update QA's topic and centrality
    if topic:
        db.update_qa_topic(qa_id, topic)
    else:
        return

    # Initialize fragment centrality for this QA's fragments
    frag_rows = db.conn.execute(
        "SELECT point_id FROM ms_fragments WHERE qa_id=?", (qa_id,)
    ).fetchall()
    for fr in frag_rows:
        db.upsert_fragment_centrality(fr["point_id"], centrality, best_score, 0.5, 0.0)

    debug(f"  QA {qa_id}: Topic='{topic}', centrality={centrality}, best_kp={best_kp}({best_score})")



class ProgressTracker:
    """Minimal progress tracker with step counting for pipeline orchestration.

    Wraps a progress callback to auto-compute percentage from step count.
    ``step()`` increments the counter; ``set_status()`` updates the label only.
    """

    def __init__(self, total: int, progress_cb, log_cb):
        self._total = max(total, 1)
        self._current = 0
        self._progress = progress_cb
        self._log = log_cb

    def step(self, label: str = ""):
        self._current += 1
        pct = min(int(self._current / self._total * 100), 100)
        if label:
            self._progress(pct, label)

    def set_status(self, msg: str):
        pct = min(int(self._current / self._total * 100), 100)
        self._progress(pct, msg)


@dataclass(frozen=True)
class PipelineContext:
    """Immutable infrastructure bundle for Phase 1/2 step functions.

    Groups 6 repeatedly-passed parameters (client, db, debug, display_name,
    retriever, tracker) into a single ctx argument, reducing function
    signatures by 3-6 positional parameters each.
    """
    client: object
    db: object
    debug: Callable
    display_name: str
    retriever: object
    tracker: object


_FILENAME_RE = re.compile(r'^(\d+)_([smw])(\d{2})_(\d+)')


def _parse_year(display_name: str) -> int:
    """Extract sortable year from paper filename.  Unknown → 9999 (sorts last)."""
    m = _FILENAME_RE.search(display_name)
    return 2000 + int(m.group(3)) if m else 9999


def _get_existing_topics(db) -> list[tuple]:
    """Return [(topic_name, qa_count), ...] sorted by count desc."""
    groups = db.get_topic_groups()
    return [(t, len(qas)) for t, qas in groups.items()
            if t and t != "(uncategorized)"]


def _ensure_session(db, display_name: str) -> Optional[int]:
    """Parse exam paper filename and ensure exam_sessions row exists."""
    m = _FILENAME_RE.match(display_name)
    if not m:
        return None
    subject = m.group(1)
    season = SEASON_MAP.get(m.group(2), "Unknown")
    year = YEAR_BASE_OFFSET + int(m.group(3))
    with db.transaction():
        db.conn.execute(
            """INSERT OR IGNORE INTO exam_sessions (subject_code, season, year, display_name)
               VALUES (?, ?, ?, ?)""",
            (subject, season, year, display_name),
        )
        row = db.conn.execute(
            "SELECT id FROM exam_sessions WHERE display_name = ?", (display_name,)
        ).fetchone()
    return row["id"] if row else None


def _run_kp_refinement(db, client, debug):
    """KP structural refinement: auto-split + auto-merge (behavior-driven)."""
    kps = db.get_all_kps()
    for kp in kps:
        if kp.get("evidence_count", 0) >= 6:
            auto_split_kp(db, kp["id"], client, debug_cb=debug)
    kps = db.get_all_kps()
    all_kp_ids = [k["id"] for k in kps]
    if len(all_kp_ids) >= 2:
        consistency = cross_kp_consistency(db, all_kp_ids[:30], client, debug_cb=debug)
        merged_count = auto_merge_kps(db, consistency.get("issues", []), debug_cb=debug)
        if merged_count:
            debug(f"Auto-merge: {merged_count} KP pairs merged")
    debug("KP structural refinement complete (behavior-driven split/merge/consistency)")


# ============================================================
# Phase 1 worker — extracted from run_pipeline() closure
# ============================================================


def _phase1_worker(qa, existing_topics, ctx: PipelineContext):
    """Phase 1: summary → insert QA → extract MS fragments → assign topic."""
    t0 = time.time()
    summary, topic = _generate_summary(
        qa.question_text, qa.answer_text, ctx.client, ctx.debug,
        existing_topics=existing_topics)
    ctx.db.log_api_call("summary", "flash", ctx.display_name,
                        qa.question_number, int((time.time() - t0) * 1000),
                        success=True, output_size=len(summary))
    qa_id = ctx.db.insert(
        question_text=qa.question_text, answer_text=qa.answer_text,
        knowledge_summary=summary, topic=topic,
        paper=ctx.display_name, question_number=qa.question_number,
        parent_question=qa.parent_question)
    ctx.db.record_attempt(qa_id, success=True)

    try:
        fragments = _extract_ms_fragments(qa.answer_text, qa_id, ctx.client, ctx.debug)
        if fragments:
            with ctx.db.transaction():
                ctx.db.insert_fragments_batch(fragments)
                topic_id = make_topic_id(topic)
                ctx.db.upsert_dynamic_topic(topic_id, name=topic, quality="embryonic")
                for frag in fragments:
                    ctx.db.set_fragment_membership(
                        frag["point_id"], topic_id, loyalty=0.5)
    except Exception as e:
        ctx.debug(f"  fragment extraction failed for Q{qa.question_number}: {e}")

    ctx.tracker.step("")
    return qa_id


# ============================================================
# Phase 2 step functions — extracted from run_pipeline() closures
# ============================================================


def _step_summarize_retrieve(qa, wmap, extopics, ctx: PipelineContext):
    """Flash summary + dual-channel retrieval → top_similar list."""
    qn = qa.question_number
    t0 = time.time()
    summary, step0_topic = _generate_summary(
        qa.question_text, qa.answer_text, ctx.client, ctx.debug,
        existing_topics=extopics)
    ctx.db.log_api_call("summary", "flash", ctx.display_name, qn,
                        int((time.time() - t0) * 1000), success=True,
                        output_size=len(summary))
    ctx.tracker.step("")

    # Layer 2 dual-channel retrieval (embedding + structure + behavior)
    all_similar = ctx.retriever.search_dual_channel(
        summary, threshold=RETRIEVAL_THRESHOLD, min_k=RETRIEVAL_MIN_K,
        max_cap=RETRIEVAL_MAX_CAP, query_topic=step0_topic)

    # Filter to top-4 by Beta weight for Pro context
    top_similar = sorted(
        all_similar,
        key=lambda qa_ref: wmap.get(qa_ref["id"], {}).get("mean", 0.5),
        reverse=True,
    )[:4]

    # Phase 3: Include stable KP text as additional reference material
    stable_kps = ctx.db.get_stable_topics()
    if stable_kps:
        kp_refs = [{
            "id": -1,
            "question_text": f"[KP] {kp['kp_concept']}",
            "answer_text": kp["kp_detail"] or kp["kp_concept"],
            "topic": kp.get("name", ""),
            "_score": kp.get("stability", 0.8),
            "_is_kp": True,
        } for kp in stable_kps[:3]]
        top_similar = top_similar[:4] + kp_refs

    return summary, step0_topic, top_similar, all_similar


def _step_answer_and_grade(qa, top_similar, step0_topic, ctx: PipelineContext):
    """Round 1: Flash answers + Round 2: Flash grades → feedback data."""
    qn = qa.question_number
    # Round 1: Flash answers
    t0 = time.time()
    r1_ok = True
    try:
        messages = _build_answer_prompt(qa.question_text, top_similar)
        result = call_flash(ctx.client, messages, max_retries=1, debug_callback=ctx.debug)
    except Exception as e:
        ctx.debug(f"  Q{qn} Round1 failed: {e}")
        result = {"answer": "", "used_qa_indices": []}
        r1_ok = False
    ctx.db.log_api_call("answer", "flash", ctx.display_name, qn,
                        int((time.time() - t0) * 1000), success=r1_ok,
                        output_size=len(result.get("answer", "")))
    ctx.tracker.step("")

    used_indices = result.get("used_qa_indices", [])
    used_ids = set()
    for idx in used_indices:
        if isinstance(idx, (int, float)) and 1 <= int(idx) <= len(top_similar):
            qa_ref = top_similar[int(idx) - 1]
            if not qa_ref.get("_is_kp"):
                ctx.db.record_attempt(qa_ref["id"], success=True)
                used_ids.add(qa_ref["id"])
    for qa_ref in top_similar:
        if qa_ref.get("_is_kp"):
            continue
        if qa_ref["id"] not in used_ids:
            ref_topic = qa_ref.get("topic", "")
            reason = ("topic_mismatch" if (ref_topic and step0_topic
                      and ref_topic != step0_topic) else "retrieval_irrelevant")
            ctx.db.record_attempt(qa_ref["id"], success=False, reason=reason)

    # Round 2: Flash grades
    t0 = time.time()
    r2_ok = True
    r2_topic = ""
    grade = ""
    covered, missed_raw = [], []
    try:
        grade_msgs = _build_grade_prompt(
            qa.question_text, result.get("answer", ""), qa.answer_text)
        grade = call_flash(ctx.client, grade_msgs, max_retries=1, debug_callback=ctx.debug)
        if isinstance(grade, dict):
            r2_topic = grade.get("topic", "")
            covered = grade.get("covered_points", [])
            missed_raw = grade.get("missed_points", [])
        else:
            r2_ok = False
    except Exception as e:
        ctx.debug(f"  Q{qn} Round2 failed: {e}")
        r2_ok = False

    missed_texts, miss_cats_json = _parse_missed_points(missed_raw)
    ctx.db.log_api_call("grade", "flash", ctx.display_name, qn,
                        int((time.time() - t0) * 1000), success=r2_ok,
                        output_size=len(str(grade)))
    ctx.tracker.step("")
    return used_indices, used_ids, covered, missed_texts, miss_cats_json, r2_topic


def _step_insert_and_feedback(qa, summary, step0_topic, r2_topic,
                               all_similar, top_similar, used_indices,
                               covered, missed_texts, miss_cats_json,
                               ctx: PipelineContext):
    """DB insert + feedback + cross-topic ref collection."""
    topic = r2_topic or step0_topic
    cross_refs = {}
    for idx in used_indices:
        if isinstance(idx, (int, float)) and 1 <= int(idx) <= len(top_similar):
            ref_topic = top_similar[int(idx) - 1].get("topic", "")
            if ref_topic and ref_topic != topic:
                key = (topic, ref_topic)
                cross_refs[key] = cross_refs.get(key, 0) + 1

    qa_id = ctx.db.insert(
        question_text=qa.question_text, answer_text=qa.answer_text,
        knowledge_summary=summary, topic=topic,
        paper=ctx.display_name, question_number=qa.question_number,
        parent_question=qa.parent_question)
    ctx.db.record_attempt(qa_id, success=True)
    ctx.db.log_question_feedback(
        qa_id=qa_id, retrieval_count=len(all_similar),
        used_qa_count=len(used_indices),
        step0_topic=step0_topic, round2_topic=r2_topic,
        covered_count=len(covered), missed_count=len(missed_texts),
        missed_text="\n".join(missed_texts) if missed_texts else "",
        miss_categories=miss_cats_json)
    return qa_id, topic, cross_refs


def _step_fragment_and_kp(qa, qa_id, topic, used_ids, covered,
                           missed_texts, ctx: PipelineContext):
    """Extract MS fragments + LLM KP classification + behavior recording."""
    try:
        fragments = _extract_ms_fragments(qa.answer_text, qa_id, ctx.client, ctx.debug)
        if fragments:
            ctx.db.insert_fragments_batch(fragments)
            topic_id = make_topic_id(topic)
            ctx.db.upsert_dynamic_topic(topic_id, name=topic, quality="embryonic")
            for frag in fragments:
                ctx.db.set_fragment_membership(
                    frag["point_id"], topic_id, loyalty=0.5)

        # Layer 1: LLM KP classification (only when KPs exist)
        kps = ctx.db.get_all_kps()
        if kps:
            kp_concepts = [{
                "id": k["id"],
                "concept": k.get("core_concept", k.get("name", "")),
                "evidence": k.get("evidence_count", 0),
            } for k in kps]
            kp_scores = _classify_qa_against_kps(
                qa.question_text, qa.answer_text, kp_concepts, ctx.client, ctx.debug)
            if kp_scores:
                _place_qa_vector_from_kp_scores(ctx.db, qa_id, kp_scores, ctx.debug)

        # Record behavior: fragments from used QAs helped this question
        _record_fragment_help(used_ids, covered, missed_texts, ctx.db, qa_id)
    except Exception as e:
        ctx.debug(f"  fragment extraction failed for Q{qa.question_number}: {e}")
    ctx.tracker.step("")


def _process_one_question_inner(qa, wmap, extopics, ctx: PipelineContext):
    """Orchestrate the 4-step Phase 2 processing for a single question."""
    qn = qa.question_number

    # Step 0: Summarise + retrieve similar QAs
    summary, step0_topic, top_similar, all_similar = _step_summarize_retrieve(
        qa, wmap, extopics, ctx)

    # Step 1: Answer + grade via Flash
    used_indices, used_ids, covered, missed_texts, miss_cats_json, r2_topic = \
        _step_answer_and_grade(qa, top_similar, step0_topic, ctx)

    # Step 2: Insert QA + feedback into DB
    qa_id, topic, cross_refs = _step_insert_and_feedback(
        qa, summary, step0_topic, r2_topic,
        all_similar, top_similar, used_indices,
        covered, missed_texts, miss_cats_json, ctx)

    # Step 3: Fragment extraction + KP classification
    _step_fragment_and_kp(qa, qa_id, topic, used_ids, covered,
                           missed_texts, ctx)

    ctx.debug(f"  Q{qn}: retrieved={len(all_similar)}, shown={len(top_similar)}, "
              f"used={len(used_indices)}, topic={topic}, "
              f"covered={len(covered)}, missed={len(missed_texts)}")
    return (qa_id, summary, cross_refs)


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
    log_schema_status(db, _debug)

    processed: set = set()
    if os.path.exists(processed_path):
        try:
            with open(processed_path, "r") as f:
                processed = set(json.load(f))
            _debug(f"Processed files: {len(processed)}")
        except Exception as e:
            _debug(f"Failed to read processed.json: {e}")

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

            existing_topics = _get_existing_topics(db) if db.count() > 0 else None

            ctx = PipelineContext(client=client, db=db, debug=_debug,
                                    display_name=display_name,
                                    retriever=None, tracker=tracker)
            _p1_worker = partial(_phase1_worker, existing_topics=existing_topics, ctx=ctx)

            tracker.step("QA pairing")  # Stage 2 complete
            with ThreadPoolExecutor(max_workers=get_worker_limit(len(qa_pairs))) as executor:
                futures = {executor.submit(_p1_worker, qa): qa for qa in qa_pairs}
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
            existing_topics = _get_existing_topics(db) if db.count() > 0 else None

            ctx = PipelineContext(client=client, db=db, debug=_debug,
                                    display_name=display_name,
                                    retriever=retriever, tracker=tracker)
            _p2_processor = partial(_process_one_question_inner, ctx=ctx)

            tracker.step("QA pairing")  # Stage 2 complete
            qa_results = []
            with ThreadPoolExecutor(max_workers=get_worker_limit(len(qa_pairs))) as executor:
                futures = {executor.submit(_p2_processor, qa, weight_map, existing_topics): qa
                           for qa in qa_pairs}
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

            # Phase 2 summary: per-paper success/failure + retrieval quality
            total = len(qa_pairs)
            succeeded = len(qa_results)
            if succeeded < total:
                _debug(f"  [Phase2] {display_name}: {succeeded}/{total} questions succeeded, {total - succeeded} failed")

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
                        except Exception as e:
                            _debug(f"miss_categories parse: {e}")
                    _debug(f"  [MissCat] {display_name}: " +
                           ", ".join(f"{k}={v}" for k, v in totals.items() if v > 0))
            except Exception as e:
                _debug(f"Phase2 stats collection: {e}")

        # Link QAs to exam session for time-dimension queries
        if session_id:
            with db.transaction():
                db.conn.execute(
                    "UPDATE qa_pairs SET session_id = ? WHERE paper = ? AND session_id IS NULL",
                    (session_id, display_name),
                )

        processed.add(display_name)
        try:
            with open(processed_path, "w") as f:
                json.dump(sorted(processed), f)
        except Exception as e:
            _debug(f"Failed to write processed.json: {e}")

        # Cross-paper consistency check after each paper
        try:
            run_cross_paper_check(db, display_name, debug_callback=_debug)
        except Exception as e:
            _debug(f"Cross-paper check failed (non-fatal): {e}")

    # -- Compute topic_related + groups (read-only, before post-processing try block) --
    groups = db.get_topic_groups()
    topic_related = {}
    for topic, qas in groups.items():
        if not topic or topic == "(uncategorized)":
            continue
        if any(src == topic or dst == topic for (src, dst) in topic_links):
            continue
        query = qas[0].get("knowledge_summary", "") or qas[0]["question_text"]
        results = retriever.search(query, threshold=RETRIEVAL_THRESHOLD, min_k=RETRIEVAL_MIN_K, max_cap=RETRIEVAL_MAX_CAP)
        counts = {}
        for r in results:
            rt = r.get("topic", "")
            if rt and rt != topic:
                counts[rt] = counts.get(rt, 0) + 1
        if counts:
            topic_related[topic] = sorted(counts.items(), key=lambda x: -x[1])

    # -- Post-processing core (wrapped: ensures db.close() on failure) --
    content = ""
    try:
        # -- Topic merge --
        tracker.set_status("Merging similar topics...")
        merge_similar_topics(db, client, _debug)
        for _ in range(3):
            tracker.step("")

        # -- Distill --
        tracker.set_status("Distilling knowledge points...")
        content = Distiller(db, client, _debug).run()
        for _ in range(5):
            tracker.step("")

        # -- Review + write --
        content = review_distilled(content, client, topic_links, topic_related, _debug)
        tracker.set_status("Writing output...")
        with open(subject_output, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        _log("Output", f"{db.count()} QAs -> {subject_output}")
        _progress(95, "Output written")
    except Exception as e:
        log_stage_error("Core post-processing", _debug, e)
        if not content:
            content = "[FALLBACK] " + ("; ".join(
                qa["answer_text"] for g in groups.values() if g
                for qa in g[:30]
            )[:10000] or "; No knowledge points extracted.")

    # Mark representative and cross-topic QAs
    # Re-fetch groups after topic merge may have modified topic assignments
    groups = db.get_topic_groups()
    weights = db.get_all_weights()
    with db.transaction():
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

    # -- Post-processing stages --
    for label, stage_fn in [
        ("Knowledge graph", lambda: run_knowledge_graph(db, api_url, api_key, debug_callback=_debug)),
        ("KP structural refinement", lambda: _run_kp_refinement(db, client, _debug)),
        ("Offline analysis", lambda: run_offline_analysis(db, api_url, api_key, progress_callback=_progress, debug_callback=_debug)),
        ("Pipeline diagnostics", lambda: run_closed_loop(db, api_url, api_key, debug_callback=_debug)),
        ("Evolution cycle", lambda: run_evolution_cycle(db, client, _debug)),
    ]:
        try:
            _debug(f"Running {label.lower()}...")
            _log(label, "Starting")
            stage_fn()
        except Exception as e:
            log_stage_error(label, _debug, e)

    _progress(100, "Analysis complete")
    db.close()
    return content


