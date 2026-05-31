"""Chat context builders: KP cache loading and analysis context enrichment."""
import os
import re

from ..logger import get_logger

_log = get_logger()


def load_kp_from_db(db, qa_rows=None) -> list[dict]:
    """Read KP data from DB: Dynamic_Topics (priority) + qa_pairs (fallback).

    If *qa_rows* is provided (pre-loaded from retriever.rebuild()), the
    ``db.qa.get_all()`` scan is skipped — eliminating a redundant full-table
    read during chat warmup.
    """
    kps = []
    try:
        dt_rows = db.topic.get_stable_kps()
        for r in dt_rows:
            kps.append({
                "topic": r["name"] or "Topic",
                "concept": r["kp_concept"],
                "detail": r["kp_detail"],
                "pitfall": "",
                "scoring": "",
                "source": "dynamic_topic",
            })
    except Exception:
        _log.debug("Failed to load KPs from dynamic_topics", exc_info=True)

    try:
        rows = qa_rows if qa_rows is not None else db.qa.get_all()
        # Filter and sort like the original query: exclude empty/uncategorized, sort by representative
        rows = [r for r in rows if r.get("topic") and r["topic"] != "(uncategorized)"]
        rows.sort(key=lambda r: (-int(r.get("is_representative", 0)), -int(r.get("success_count", 0))))
        seen_topics = {k["topic"] for k in kps}
        for r in rows:
            topic = r["topic"]
            if topic in seen_topics:
                continue
            seen_topics.add(topic)
            kps.append({
                "topic": topic,
                "concept": r["knowledge_summary"] or r["question_text"][:200],
                "detail": r["answer_text"][:300],
                "pitfall": "",
                "scoring": "",
                "difficulty": r["difficulty_estimate"] or "",
                "source": "qa_pairs",
            })
    except Exception:
        _log.debug("Failed to load KPs from qa_pairs", exc_info=True)
    return kps


def load_kp_cache(db=None, points_file: str = "", qa_rows=None) -> list[dict]:
    """Load KP cache: DB first (structured), points.txt fallback (parsed)."""
    if db:
        kps = load_kp_from_db(db, qa_rows=qa_rows)
        if kps:
            return kps

    kps = []
    if not points_file or not os.path.exists(points_file):
        return kps
    try:
        with open(points_file, "r", encoding="utf-8") as f:
            content = f.read()
        current_topic = ""
        for block in content.split("\n\n"):
            lines = block.strip().split("\n")
            if not lines:
                continue
            first = lines[0].strip()
            if first and not first[0].isdigit() and not first.startswith("See also") and not first.startswith("Related:"):
                current_topic = first.split("  [")[0].strip()
            for idx, line in enumerate(lines):
                m = re.match(r"^(\d+)\.\s*(.*)", line.strip())
                if m:
                    concept = m.group(2)
                    detail = pitfall = scoring = ""
                    for j in range(idx + 1, len(lines)):
                        s = lines[j].strip()
                        if s.startswith("Detail:"):
                            detail = s[7:].strip()
                        elif s.startswith("Pitfall:"):
                            pitfall = s[8:].strip()
                        elif s.startswith("Scoring:"):
                            scoring = s[8:].strip()
                        elif s and (s[0].isdigit() or s.startswith("See also") or s.startswith("Related:")):
                            break
                    kps.append({"topic": current_topic, "concept": concept, "detail": detail, "pitfall": pitfall, "scoring": scoring})
    except Exception:
        _log.debug("Failed to parse points.txt KP cache", exc_info=True)
    return kps


def build_analysis_context(db, topics: list[str], student_verb: str, student_id: str) -> str:
    """Build enrichment context from offline analysis results.

    Reads verb_patterns, topic_difficulty, topic_dependencies, student state.
    Returns empty string if no data available (graceful degradation).
    """
    if not topics:
        return ""

    parts = []

    # 1. Topic difficulty
    try:
        difficulties = db.get_topic_difficulty()
        diff_map = {d["topic"]: d for d in difficulties}
        for t in topics[:2]:
            if t in diff_map:
                d = diff_map[t]
                if d.get("mode_difficulty") in ("advanced", "mixed"):
                    parts.append(f"Topic [{t}] is advanced — provide detailed explanation with first-principles build-up.")
                elif d.get("mode_difficulty") == "basic":
                    parts.append(f"Topic [{t}] is basic — keep explanation concise and direct.")
    except Exception:
        _log.debug("Failed to load topic difficulty", exc_info=True)

    # 2. Command verb pattern
    if student_verb:
        try:
            patterns = db.get_verb_patterns()
            for p in patterns:
                if p["verb"] == student_verb and p.get("pattern_summary"):
                    parts.append(f"Answer style for '{student_verb}' questions: {p['pattern_summary']}")
                    break
        except Exception:
            _log.debug("Failed to load verb patterns", exc_info=True)

    # 3. Prerequisites
    try:
        knowledge = db.get_knowledge_state(student_id)
        for t in topics[:2]:
            prereqs = db.get_direct_prerequisites(t)
            for pr in prereqs:
                pre_topic = pr["prerequisite"]
                if pre_topic not in knowledge or knowledge[pre_topic] != "mastered":
                    parts.append(
                        f"Student may not have mastered prerequisite [{pre_topic}] for [{t}]. "
                        f"Briefly recap {pre_topic} before explaining {t}."
                    )
    except Exception:
        _log.debug("Failed to load prerequisites", exc_info=True)

    # 4. Student confusion history
    try:
        confusions = db.get_student_confusions(student_id)
        confused_topics = {c["topic"] for c in confusions[:10] if not c.get("resolved")}
        relevant_confusions = confused_topics & set(topics)
        if relevant_confusions:
            parts.append(f"Student has shown confusion on: {', '.join(relevant_confusions)}. Address these carefully.")
    except Exception:
        _log.debug("Failed to load confusion history", exc_info=True)

    if parts:
        return "[Analysis Context]\n" + "\n".join(parts) + "\n\n"
    return ""
