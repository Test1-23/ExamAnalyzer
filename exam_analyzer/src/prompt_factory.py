"""Bilingual prompt factory — eliminates ~15 duplicated if-lang-else patterns.

Each PromptTemplate encodes a semantic intent (summary, classify, grade, etc.)
with English and Chinese variants.  ``.build(**kwargs)`` auto-selects language
from ``detect_content_lang(**kwargs.get('lang_source', ''))``.
"""

from __future__ import annotations

from typing import Any, Callable

from .embedding_cluster import detect_content_lang


class PromptTemplate:
    """A bilingual (en + zh) prompt pair, rendered via ``.build(**kwargs)``.

    If ``lang`` is provided as a kwarg it is used directly; otherwise the
    template detects language from ``lang_source`` (default ``""``).
    """

    __slots__ = ("en_system", "zh_system", "en_user", "zh_user")

    def __init__(self, en_system: str, zh_system: str, en_user: str, zh_user: str) -> None:
        self.en_system = en_system
        self.zh_system = zh_system
        self.en_user = en_user
        self.zh_user = zh_user

    def build(self, lang: str = "", lang_source: str = "", **kwargs: Any) -> list[dict]:
        """Return ``[{"role":"system",...}, {"role":"user",...}]``."""
        if not lang:
            lang = detect_content_lang(lang_source)
        if lang == "zh":
            sys = self.zh_system.format(**kwargs)
            usr = self.zh_user.format(**kwargs)
        else:
            sys = self.en_system.format(**kwargs)
            usr = self.en_user.format(**kwargs)
        return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

    def build_system_only(self, lang: str = "", **kwargs: Any) -> dict:
        """Return a bare system message dict (no user)."""
        if lang == "zh":
            return {"role": "system", "content": self.zh_system.format(**kwargs)}
        return {"role": "system", "content": self.en_system.format(**kwargs)}


# ============================================================
# Pre-defined templates — one per semantic intent
# ============================================================

SUMMARY = PromptTemplate(
    en_system="You are an exam knowledge classifier. Do two things. Output JSON. "
              "1. Describe the core concept tested in 1-2 sentences. "
              "2. Assign a concise topic name.",
    zh_system="你是一个考试知识分类专家。同时完成两件事。Output JSON."
              "1. 用1-2句描述这道题考察的核心技术概念"
              "2. 分配一个简洁的主题名称",
    en_user=("Question: {question_text}\n\nAnswer: {answer_text}\n\n"
             "{topic_hint}"
             "Do not mention question-specific context (values, filenames, scenarios, names).\n"
             "Use standard terminology for topic names (e.g. 'Data Compression').\n"
             'Return JSON: {{"summary": "core concept description", "topic": "Topic Name"}}'),
    zh_user=("题目: {question_text}\n\n答案: {answer_text}\n\n"
             "{topic_hint}"
             "不要提及题目特定上下文（具体数值、文件名、场景描述、人名）。\n"
             "主题名称应使用标准术语（如 'Data Compression', 'Interrupt Handling'）。\n"
             '返回 JSON: {{"summary": "核心知识描述", "topic": "标准主题名"}}'),
)

FRAGMENT = PromptTemplate(
    en_system="You are an exam marking expert. Split the given mark scheme answer "
              "into individual scoring points. Output JSON.",
    zh_system="你是一个考试评分专家。将给定的评分标准答案拆分为独立的得分点。Output JSON。",
    en_user=("Mark scheme answer:\n{answer_text}\n\n"
             "Split into independent scoring points. Each point is a single, "
             "non-divisible requirement that a student must demonstrate.\n"
             "Preserve the original wording exactly. Do not rewrite or paraphrase.\n"
             'Return JSON: {{"points": [{{"text": "exact original wording", "marks": 1}}, ...]}}'),
    zh_user=("评分标准答案:\n{answer_text}\n\n"
             "拆分为独立的得分点。每个得分点是一个不可再分的要求。\n"
             "保留原文措辞，不要改写。\n"
             '返回 JSON: {{"points": [{{"text": "原始措辞", "marks": 1}}, ...]}}'),
)

QA_CLASSIFY = PromptTemplate(
    en_system="You are an exam curriculum expert. Judge whether each listed knowledge "
              "point is tested by the given question. Score each KP. Output JSON.",
    zh_system="你是一个考试课程专家。判断每个知识点是否被给定的题目考察。逐项评分。Output JSON。",
    en_user=("Question: {qa_text}\n\nAnswer: {answer_text}\n\n"
             "Knowledge Points:\n{kp_list}\n\n"
             "For each KP, score:\n"
             "1.0 = core focus — the question directly tests this KP\n"
             "0.5 = indirectly involved — background knowledge helpful\n"
             "0.0 = unrelated\n"
             'Return JSON: {{"kp_scores": {{"kp_id": 1.0, ...}}}}'),
    zh_user=("题目: {qa_text}\n\n答案: {answer_text}\n\n"
             "知识点:\n{kp_list}\n\n"
             "对每个KPi评分:\n"
             "1.0 = 核心考察\n0.5 = 间接涉及\n0.0 = 无关\n"
             '返回 JSON: {{"kp_scores": {{"kp_id": 1.0, ...}}}}'),
)

ANSWER = PromptTemplate(
    en_system="You are a knowledgeable tutor. "
              "Use the provided references for accuracy; never guess. "
              "Answer clearly. Output JSON.",
    zh_system="你是一个知识渊博的辅导老师。"
              "利用提供的参考资料确保准确性；不要猜测。"
              "清晰地回答。Output JSON。",
    en_user=("References:\n{references}\n\n"
             "Question: {question}\n\n"
             'Return JSON: {{"answer": "your answer", '
             '"used_qa_indices": [], "r2_topic": ""}}'),
    zh_user=("参考资料:\n{references}\n\n"
             "问题: {question}\n\n"
             '返回 JSON: {{"answer": "你的回答", '
             '"used_qa_indices": [], "r2_topic": ""}}'),
)

GRADE = PromptTemplate(
    en_system="You are a strict exam grader. Compare the student's answer against "
              "the mark scheme. Output JSON.",
    zh_system="你是一个严格的考试评分员。将学生答案与评分标准逐项比对。Output JSON。",
    en_user=("Mark scheme (ground truth):\n{answer_text}\n\n"
             "Student answer to grade:\n{student_answer}\n\n"
             "1. List covered points (student got right)\n"
             '2. List missed points: {{"point": "...", "reason": "knowledge_gap|misinterpretation|insufficient_detail|retrieval_quality"}}\n'
             'Return JSON: {{"covered_points": [...], "missed_points": [...]}}'),
    zh_user=("评分标准（标准答案）:\n{answer_text}\n\n"
             "待评分的学生答案:\n{student_answer}\n\n"
             "1. 列出已覆盖的得分点\n"
             '2. 列出遗漏的得分点: {{"point": "...", "reason": "knowledge_gap|misinterpretation|insufficient_detail|retrieval_quality"}}\n'
             '返回 JSON: {{"covered_points": [...], "missed_points": [...]}}'),
)

QUERY_ANALYST = PromptTemplate(
    en_system="Analyze the student's question. Output JSON.",
    zh_system="分析学生问题。Output JSON。",
    en_user=("Question: {question}\n\n"
             "1. Extract 3-5 English keywords/technical terms for knowledge base search\n"
             "2. Classify as: definition | calculation | comparison | explanation | exam_tip\n"
             "3. Identify the command verb (state, explain, describe, compare, calculate, evaluate, etc.)\n"
             'Return JSON: {{"keywords": ["term1"], "qtype": "definition", "verb": "explain"}}'),
    zh_user=("问题: {question}\n\n"
             "1. 提取3-5个英文关键词/技术术语用于知识库检索（将中文概念翻译为英文术语）\n"
             "2. 分类: definition(定义) | calculation(计算) | comparison(对比) | explanation(解释) | exam_tip(考试技巧)\n"
             "3. 识别指令动词(state/explain/describe/compare/calculate/evaluate...)\n"
             '返回 JSON: {{"keywords": ["term1"], "qtype": "definition", "verb": "explain"}}'),
)

CRITIC = PromptTemplate(
    en_system="Review this tutoring answer for quality. Output JSON.",
    zh_system="审查此教学回答的质量。Output JSON。",
    en_user=("Question: {question}\n\n"
             "References:\n{ctx}\n"
             "Answer to review:\n{answer}\n\n"
             "Check:\n"
             "1. Are technical facts consistent with the references? (no hallucinations)\n"
             "2. Are the references quoted verbatim (not translated)?\n"
             "3. Is the explanation clear and complete?\n"
             "4. Are [KB]/[General] markers used correctly?\n"
             'Return JSON: {{"pass": true/false, "feedback": "specific issues if any"}}'),
    zh_user=("问题: {question}\n\n"
             "参考资料:\n{ctx}\n"
             "待审查回答:\n{answer}\n\n"
             "检查:\n"
             "1. 技术事实是否与参考资料一致？（无幻觉）\n"
             "2. 参考资料引用是否保持原文（未被翻译）？\n"
             "3. 解释是否清晰完整？\n"
             "4. [KB]/[General] 标记是否正确使用？\n"
             '返回 JSON: {{"pass": true/false, "feedback": "具体问题（如有）"}}'),
)

SUGGEST = PromptTemplate(
    en_system="Suggest 2-3 follow-up questions a student might ask. Output JSON.",
    zh_system="建议2-3个学生可能追问的问题。Output JSON。",
    en_user=("Student asked: {question}\n"
             "Topics covered: {topic_str}\n"
             "{prereq_hint}"
             "Suggest 2-3 natural follow-up questions.\n"
             'Include: (a) a deeper question on the same topic, '
             '(b) a question linking to a prerequisite or related topic.\n'
             'Return JSON: {{"suggestions": ["question 1", "question 2"]}}'),
    zh_user=("学生问: {question}\n"
             "涉及主题: {topic_str}\n"
             "{prereq_hint}"
             "建议2-3个自然的追问。\n"
             '包含: (a) 同一主题的深入问题, '
             '(b) 链接到先修或相关主题的问题。\n'
             '返回 JSON: {{"suggestions": ["问题 1", "问题 2"]}}'),
)


def build_answer_generator_messages(
    question: str, qtype: str, lang: str, ctx: str, ctx_kp: str,
    history: list[dict],
) -> list[dict]:
    """Build messages for answer generator with type-adaptive prompt.

    This is separate from PromptTemplate because it has dynamic type_guide
    and history stitching that don't fit the static template model cleanly.
    """
    type_guide = {
        "definition": "Give a concise definition first, then explain with an example from the provided references.",
        "calculation": "Show the formula first, then apply it step-by-step. Use numbers from the references as examples. Show intermediate steps clearly.",
        "comparison": "Use a comparison table. For each difference, give an example from the references.",
        "explanation": "Explain from first principles. Build up the concept step by step, citing references at each stage.",
        "exam_tip": "Focus on common mistakes and scoring guidance from the references. List pitfalls with concrete wrong-answer examples.",
    }
    guide = type_guide.get(qtype, type_guide["explanation"])

    sys = (
        "You are a knowledgeable and patient tutor. "
        "Use the provided Q&A and knowledge points as reference. "
        "CRITICAL: NEVER translate Q&A or KP content — quote it verbatim. "
        "Technical terms stay in original language. "
        "Your explanation may be in the student's language. "
        "Mark each claim: prefix with [KB] if from provided references, [General] if from your own knowledge. "
        "Output JSON."
    )

    if lang == "zh":
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
            '返回 JSON: {{"answer": "你的回答", '
            '"quiz": {{"question": "...", "expected": "..."}}, '
            '"path_hint": {{"next_topic": "...", "reason": "..."}}}}'
        )
    else:
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
            'Return JSON: {{"answer": "your answer", '
            '"quiz": {{"question": "...", "expected": "..."}}, '
            '"path_hint": {{"next_topic": "...", "reason": "..."}}}}'
        )

    msgs: list[dict] = [{"role": "system", "content": sys}]
    for h in history[-6:]:
        role = "assistant" if h.get("role") == "assistant" else "user"
        msgs.append({"role": role, "content": h.get("content", "")})
    msgs.append({"role": "user", "content": usr})
    return msgs
