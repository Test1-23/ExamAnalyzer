"""Chat assistant agents: query analyst, answer generator, critic, suggester."""
from __future__ import annotations

from typing import Any

from ..deepseek_client import call_flash
from ..logger import get_logger

_log = get_logger()


# ---- Agent 1: Query Analyst ----

def agent_query_analyst(question: str, lang: str, client: Any) -> dict[str, Any]:
    """Analyze question: rephrase for retrieval, classify type, extract command verb."""
    if lang == 'en':
        sys = "Analyze the student's question. Output JSON."
        usr = (
            f"Question: {question}\n\n"
            "1. Extract 3-5 English keywords/technical terms for knowledge base search\n"
            "2. Classify as: definition | calculation | comparison | explanation | exam_tip\n"
            "3. Identify the command verb (state, explain, describe, compare, calculate, evaluate, etc.)\n"
            'Return JSON: {"keywords": ["term1"], "qtype": "definition", "verb": "explain"}'
        )
    else:
        sys = "分析学生问题。Output JSON。"
        usr = (
            f"问题: {question}\n\n"
            "1. 提取3-5个英文关键词/技术术语用于知识库检索（将中文概念翻译为英文术语）\n"
            "2. 分类: definition(定义) | calculation(计算) | comparison(对比) | explanation(解释) | exam_tip(考试技巧)\n"
            "3. 识别指令动词(state/explain/describe/compare/calculate/evaluate...)\n"
            '返回 JSON: {"keywords": ["term1"], "qtype": "definition", "verb": "explain"}'
        )
    try:
        result = call_flash(client, [{"role": "system", "content": sys}, {"role": "user", "content": usr}], max_retries=1)
        return result if isinstance(result, dict) else {"keywords": [], "qtype": "explanation", "verb": ""}
    except Exception:
        _log.debug("Query analyst failed", exc_info=True)
        return {"keywords": [], "qtype": "explanation", "verb": ""}


# ---- Agent 3: Answer Generator (type-adaptive) ----

def agent_answer_generator(question: str, qtype: str, lang: str, ctx: str,
                           ctx_kp: str, history: list[dict[str, Any]], client: Any) -> dict[str, Any]:
    """Generate answer with type-adaptive prompt."""
    sys = (
        "You are a knowledgeable and patient tutor. "
        "Use the provided Q&A and knowledge points as reference. "
        "CRITICAL: NEVER translate Q&A or KP content — quote it verbatim. "
        "Technical terms stay in original language. "
        "Your explanation may be in the student's language. "
        "Mark each claim: prefix with [KB] if from provided references, [General] if from your own knowledge. "
        "Output JSON."
    )

    type_guide = {
        "definition": "Give a concise definition first, then explain with an example from the provided references.",
        "calculation": "Show the formula first, then apply it step-by-step. Use numbers from the references as examples. Show intermediate steps clearly.",
        "comparison": "Use a comparison table. For each difference, give an example from the references.",
        "explanation": "Explain from first principles. Build up the concept step by step, citing references at each stage.",
        "exam_tip": "Focus on common mistakes and scoring guidance from the references. List pitfalls with concrete wrong-answer examples.",
    }
    guide = type_guide.get(qtype, type_guide["explanation"])

    if lang == 'en':
        usr = (
            f"{ctx}\n{ctx_kp}\n"
            f"Student question: {question}\n\n"
            f"Question type: {qtype}\n"
            f"Style guide: {guide}\n\n"
            "Answer in English. Mark each claim with [KB] or [General].\n"
            "Also include:\n"
            "- A 1-question diagnostic quiz to check the student's understanding "
            "(with expected short answer, max 1 sentence)\n"
            "- A learning path hint: what related topic the student should explore next, and why\n"
            'Return JSON: {"answer": "your answer", '
            '"quiz": {"question": "...", "expected": "..."}, '
            '"path_hint": {"next_topic": "...", "reason": "..."}}'
        )
    else:
        usr = (
            f"{ctx}\n{ctx_kp}\n"
            f"学生问题: {question}\n\n"
            f"问题类型: {qtype}\n"
            f"回答风格: {guide}\n\n"
            "请用中文回答。标记每个论断: [KB]=来自参考资料, [General]=来自你自己。\n"
            "【关键规则】Q&A和KP内容是英文原文——必须逐字引用，绝对不要翻译成中文。\n"
            "技术术语保持英文原文。只有解释和评论部分使用中文。\n"
            "同时包含:\n"
            "- 一道诊断性小测题（检查学生是否理解，附带期望的简短答案）\n"
            "- 学习路径提示: 学生接下来应探索什么相关主题，为什么\n"
            '返回 JSON: {"answer": "你的回答", '
            '"quiz": {"question": "...", "expected": "..."}, '
            '"path_hint": {"next_topic": "...", "reason": "..."}}'
        )
    msgs = [{"role": "system", "content": sys}]
    for h in history[-6:]:
        role = "assistant" if h.get("role") == "assistant" else "user"
        msgs.append({"role": role, "content": h.get("content", "")})
    msgs.append({"role": "user", "content": usr})
    try:
        result = call_flash(client, msgs, max_retries=1)
        return result if isinstance(result, dict) else {"answer": str(result)}
    except Exception:
        _log.debug("Answer generator failed", exc_info=True)
        return {"answer": ""}


# ---- Agent 4: Critic ----

def agent_critic(question: str, answer: str, similar: list[dict[str, Any]], lang: str, client: Any) -> dict[str, Any]:
    """Review answer quality. Returns {pass: bool, feedback: str}."""
    ctx = ""
    for i, qa in enumerate(similar[:3], 1):
        ctx += f"Q{i}: {qa['question_text']}\nA: {qa['answer_text']}\n\n"

    if lang == 'en':
        sys = "Review this tutoring answer for quality. Output JSON."
        usr = (
            f"Question: {question}\n\n"
            f"References:\n{ctx}\n"
            f"Answer to review:\n{answer}\n\n"
            "Check:\n"
            "1. Are technical facts consistent with the references? (no hallucinations)\n"
            "2. Are the references quoted verbatim (not translated)?\n"
            "3. Is the explanation clear and complete?\n"
            "4. Are [KB]/[General] markers used correctly?\n"
            'Return JSON: {"pass": true/false, "feedback": "specific issues if any"}'
        )
    else:
        sys = "审查此教学回答的质量。Output JSON。"
        usr = (
            f"问题: {question}\n\n"
            f"参考资料:\n{ctx}\n"
            f"待审查回答:\n{answer}\n\n"
            "检查:\n"
            "1. 技术事实是否与参考资料一致？（无幻觉）\n"
            "2. 参考资料引用是否保持原文（未被翻译）？\n"
            "3. 解释是否清晰完整？\n"
            "4. [KB]/[General] 标记是否正确使用？\n"
            '返回 JSON: {"pass": true/false, "feedback": "具体问题（如有）"}'
        )
    try:
        result = call_flash(client, [{"role": "system", "content": sys}, {"role": "user", "content": usr}], max_retries=1)
        return result if isinstance(result, dict) else {"pass": True, "feedback": ""}
    except Exception:
        _log.debug("Critic failed", exc_info=True)
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
            knowledge = db.get_knowledge_state(session_id)
            all_prereqs = set()
            for t in topics[:2]:
                for pr in db.get_direct_prerequisites(t):
                    pre_topic = pr["prerequisite"]
                    if pre_topic not in knowledge or knowledge[pre_topic] != "mastered":
                        all_prereqs.add(pre_topic)
            if all_prereqs:
                prereq_hint = f"Prerequisites not yet mastered: {', '.join(all_prereqs)}. Suggest reviewing these.\n"
        except Exception:
            _log.debug("Failed to load prerequisites for suggestions", exc_info=True)

    if lang == 'en':
        sys = "Suggest 2-3 follow-up questions a student might ask. Output JSON."
        usr = (
            f"Student asked: {question}\n"
            f"Topics covered: {topic_str}\n"
            f"{prereq_hint}"
            "Suggest 2-3 natural follow-up questions.\n"
            'Include: (a) a deeper question on the same topic, '
            '(b) a question linking to a prerequisite or related topic.\n'
            'Return JSON: {"suggestions": ["question 1", "question 2"]}'
        )
    else:
        sys = "建议2-3个学生可能追问的问题。Output JSON。"
        usr = (
            f"学生问了: {question}\n"
            f"涉及主题: {topic_str}\n"
            f"{prereq_hint}"
            "建议2-3个自然追问。\n"
            "包含:(a)同一topic的深入问题, (b)关联前置知识的问题。\n"
            '返回 JSON: {"suggestions": ["问题1", "问题2"]}'
        )
    try:
        result = call_flash(client, [{"role": "system", "content": sys}, {"role": "user", "content": usr}], max_retries=1)
        return result.get("suggestions", []) if isinstance(result, dict) else []
    except Exception:
        _log.debug("Suggest agent failed", exc_info=True)
        return []
