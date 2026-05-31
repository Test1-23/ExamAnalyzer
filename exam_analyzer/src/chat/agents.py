"""Chat assistant agents: query analyst, answer generator, critic, suggester."""
from __future__ import annotations

from typing import Any

from ..deepseek_client import call_flash
from ..logger import get_logger
from ..prompt_factory import QUERY_ANALYST, CRITIC, SUGGEST, build_answer_generator_messages

_log = get_logger()


# ---- Agent 1: Query Analyst ----

def agent_query_analyst(question: str, lang: str, client: Any) -> dict[str, Any]:
    """Analyze question: rephrase for retrieval, classify type, extract command verb."""
    msgs = QUERY_ANALYST.build(lang=lang, question=question)
    try:
        result = call_flash(client, msgs, max_retries=1)
        return result if isinstance(result, dict) else {"keywords": [], "qtype": "explanation", "verb": ""}
    except Exception as e:
        from .error_utils import log_exception
        log_exception(_log, "Query analyst", "", e, level="warning")
        return {"keywords": [], "qtype": "explanation", "verb": ""}


# ---- Agent 3: Answer Generator (type-adaptive) ----

def agent_answer_generator(question: str, qtype: str, lang: str, ctx: str,
                           ctx_kp: str, history: list[dict[str, Any]], client: Any) -> dict[str, Any]:
    """Generate answer with type-adaptive prompt."""
    msgs = build_answer_generator_messages(question, qtype, lang, ctx, ctx_kp, history)
    try:
        result = call_flash(client, msgs, max_retries=1)
        return result if isinstance(result, dict) else {"answer": str(result)}
    except Exception as e:
        from .error_utils import log_exception
        log_exception(_log, "Answer generator", "", e, level="warning")
        return {"answer": ""}


# ---- Agent 4: Critic ----

def agent_critic(question: str, answer: str, similar: list[dict[str, Any]], lang: str, client: Any) -> dict[str, Any]:
    """Review answer quality. Returns {pass: bool, feedback: str}."""
    ctx = ""
    for i, qa in enumerate(similar[:3], 1):
        ctx += f"Q{i}: {qa['question_text']}\nA: {qa['answer_text']}\n\n"

    msgs = CRITIC.build(lang=lang, question=question, ctx=ctx, answer=answer)
    try:
        result = call_flash(client, msgs, max_retries=1)
        return result if isinstance(result, dict) else {"pass": True, "feedback": ""}
    except Exception as e:
        from .error_utils import log_exception
        log_exception(_log, "Critic", "", e, level="warning")
        return {"pass": False, "feedback": "Review unavailable (API error), retrying"}


# ---- Agent 5: Follow-up Suggester ----

def agent_suggest(question: str, answer: str, similar: list[dict[str, Any]], lang: str, client: Any,
                  db: Any = None, session_id: str = "") -> list[str]:
    """Generate follow-up question suggestions, enriched with dependency data."""
    topics = list(set(qa.get("topic", "") for qa in similar[:5] if qa.get("topic")))
    topic_str = ", ".join(topics) if topics else "various topics"

    prereq_hint = ""
    if db and topics and session_id:
        try:
            knowledge = db.student.get_knowledge_state(session_id)
            all_prereqs = set()
            for t in topics[:2]:
                for pr in db.analysis.get_direct_prerequisites(t):
                    pre_topic = pr["prerequisite"]
                    if pre_topic not in knowledge or knowledge[pre_topic] != "mastered":
                        all_prereqs.add(pre_topic)
            if all_prereqs:
                prereq_hint = f"Prerequisites not yet mastered: {', '.join(all_prereqs)}. Suggest reviewing these.\n"
        except Exception as e:
            from .error_utils import log_exception
            log_exception(_log, "Suggest prerequisites", "", e, level="warning")

    msgs = SUGGEST.build(lang=lang, question=question, topic_str=topic_str, prereq_hint=prereq_hint)
    try:
        result = call_flash(client, msgs, max_retries=1)
        return result.get("suggestions", []) if isinstance(result, dict) else []
    except Exception as e:
        from .error_utils import log_exception
        log_exception(_log, "Suggest agent", "", e, level="warning")
        return []
