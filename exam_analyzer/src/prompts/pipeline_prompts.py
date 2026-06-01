"""Prompt functions for pipeline stages — thin wrappers over PromptBuilder."""

from typing import Callable

from ..deepseek_client import call_flash
from ..embedding_cluster import detect_content_lang
from ..models import QAPair
from ..prompt_factory import PromptType, PromptBuilder


# ============================================================
# Stage 2: QA pairing (Flash)
# ============================================================

def stage2_qa_pairing(pair, client, debug: Callable) -> list:
    """Call Flash once per paper to match questions with answers via PromptBuilder."""
    qp_text = pair.qp.full_text
    ms_text = pair.ms.full_text
    debug(f"  QA pairing: {pair.display_name} ({len(qp_text)}c / {len(ms_text)}c)")

    messages = PromptBuilder.build(
        PromptType.QA_PAIRING,
        display_name=pair.display_name,
        qp_text=qp_text, ms_text=ms_text,
        lang=detect_content_lang((qp_text + ms_text)[:2000]))

    result, _ = call_flash(client, messages, debug_callback=debug)
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

# -- Summary + Topic (Flash) --

def _generate_summary(question_text: str, answer_text: str,
                      client, debug: Callable,
                      existing_topics: list = None) -> tuple[str, str]:
    """Generate Flash summary + topic, reusing existing topics when applicable."""
    lang = detect_content_lang(question_text + answer_text)
    topic_list = ""
    if existing_topics:
        top_topics = sorted(existing_topics, key=lambda x: -x[1])[:15]
        topic_list = ", ".join(t[0] for t in top_topics)

    messages = PromptBuilder.build(
        PromptType.SUMMARY, question_text=question_text,
        answer_text=answer_text, topic_list=topic_list, lang=lang)
    try:
        result, _ = call_flash(client, messages, max_retries=1, debug_callback=debug)
        if isinstance(result, dict):
            return result.get("summary", question_text[:200]), result.get("topic", "(unnamed)")
        return str(result), "(unnamed)"
    except Exception as e:
        from ..error_utils import log_exception
        log_exception(debug, "Summary generation", "", e)
        return question_text[:200], "(unnamed)[auto]"

# -- Round 1: Answer with past QAs --

def _build_answer_prompt(question_text: str, similar_qas: list[dict]) -> list:
    """Build Answer prompt via PromptBuilder (template: _ANSWER_TMPL)."""
    if similar_qas:
        qa_block = ""
        for i, qa in enumerate(similar_qas, 1):
            qa_block += (f"--- Past Q{i} (similarity: {qa.get('_score',0):.2f}) ---\n"
                         f"Q: {qa['question_text']}\nA: {qa['answer_text']}\n\n")
    else:
        qa_block = "(Knowledge base is empty, no past Q&As)\n\n"

    return PromptBuilder.build(
        PromptType.ANSWER, qa_block=qa_block, question_text=question_text,
        lang=detect_content_lang(question_text))

# -- Round 2: Grade --

def _build_grade_prompt(question_text: str, predicted_answer: str,
                        ms_answer: str) -> list:
    """Build Grade prompt via PromptBuilder (template: _GRADE_TMPL)."""
    return PromptBuilder.build(
        PromptType.GRADE, question_text=question_text,
        predicted_answer=predicted_answer, ms_answer=ms_answer,
        lang=detect_content_lang(question_text + ms_answer))

