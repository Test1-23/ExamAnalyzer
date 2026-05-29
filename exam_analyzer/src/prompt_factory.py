"""Bilingual prompt factory — eliminates duplicated if-lang-else patterns.

Uses ``%``-formatting (NOT ``str.format``) because exam text (mark schemes,
code snippets) frequently contains literal ``{`` ``}`` that would crash
``.format()``.  All string values are auto-escaped via ``_escape_pct`` so
that ``%`` characters in user content are safe.
"""

from __future__ import annotations

from typing import Any

from .embedding_cluster import detect_content_lang


def _escape_pct(text: str) -> str:
    """Double ``%`` signs so user text is safe for ``%``-format interpolation."""
    return text.replace("%", "%%")


class PromptTemplate:
    """A bilingual (en + zh) prompt pair, rendered via ``.build(**kwargs)``.

    Templates use ``%(name)s`` placeholders (NOT ``{name}``) so that exam
    text containing literal braces is safe.  String values are auto-escaped.
    """

    __slots__ = ("en_system", "zh_system", "en_user", "zh_user")

    def __init__(self, en_system: str, zh_system: str, en_user: str, zh_user: str) -> None:
        self.en_system = en_system
        self.zh_system = zh_system
        self.en_user = en_user
        self.zh_user = zh_user

    def build(self, lang: str = "", lang_source: str = "", **kwargs: Any) -> list[dict]:
        """Return ``[{"role":"system",...}, {"role":"user",...}]``.

        String values are auto-escaped for ``%``-formatting safety.
        """
        if not lang:
            lang = detect_content_lang(lang_source)
        safe = {k: _escape_pct(v) if isinstance(v, str) else v
                for k, v in kwargs.items()}
        if lang == "zh":
            sys = self.zh_system % safe
            usr = self.zh_user % safe
        else:
            sys = self.en_system % safe
            usr = self.en_user % safe
        return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

    def build_system_only(self, lang: str = "", **kwargs: Any) -> dict:
        """Return a bare system message dict (no user)."""
        safe = {k: _escape_pct(v) if isinstance(v, str) else v
                for k, v in kwargs.items()}
        if lang == "zh":
            return {"role": "system", "content": self.zh_system % safe}
        return {"role": "system", "content": self.en_system % safe}


# ============================================================
# Pre-defined templates — one per semantic intent
# ============================================================

FRAGMENT = PromptTemplate(
    en_system="You are an exam marking expert. Split the given mark scheme answer "
              "into individual scoring points. Output JSON.",
    zh_system="你是一个考试评分专家。将给定的评分标准答案拆分为独立的得分点。Output JSON。",
    en_user=("Mark scheme answer:\n%(answer_text)s\n\n"
             "Split into independent scoring points. Each point is a single, "
             "non-divisible requirement that a student must demonstrate.\n"
             "Preserve the original wording exactly. Do not rewrite or paraphrase.\n"
             'Return JSON: {"points": [{"text": "exact original wording", "marks": 1}}, ...]'),
    zh_user=("评分标准答案:\n%(answer_text)s\n\n"
             "拆分为独立的得分点。每个得分点是一个不可再分的要求。\n"
             "保留原文措辞，不要改写。\n"
             '返回 JSON: {"points": [{"text": "原始措辞", "marks": 1}}, ...]'),
)

QA_CLASSIFY = PromptTemplate(
    en_system="You are an exam curriculum expert. Judge whether each listed knowledge "
              "point is tested by the given question. Score each KP. Output JSON.",
    zh_system="你是一个考试课程专家。判断每个知识点是否被给定的题目考察。逐项评分。Output JSON。",
    en_user=("Question: %(qa_text)s\n\nAnswer: %(answer_text)s\n\n"
             "Knowledge Points:\n%(kp_list)s\n\n"
             "For each KP, score:\n"
             "1.0 = core focus — the question directly tests this KP\n"
             "0.5 = indirectly involved — background knowledge helpful\n"
             "0.0 = unrelated\n"
             'Return JSON: {"kp_scores": {"kp_id": 1.0, ...}}'),
    zh_user=("题目: %(qa_text)s\n\n答案: %(answer_text)s\n\n"
             "知识点:\n%(kp_list)s\n\n"
             "对每个KPi评分:\n"
             "1.0 = 核心考察\n0.5 = 间接涉及\n0.0 = 无关\n"
             '返回 JSON: {"kp_scores": {"kp_id": 1.0, ...}}'),
)

QUERY_ANALYST = PromptTemplate(
    en_system="Analyze the student's question. Output JSON.",
    zh_system="分析学生问题。Output JSON。",
    en_user=("Question: %(question)s\n\n"
             "1. Extract 3-5 English keywords/technical terms for knowledge base search\n"
             "2. Classify as: definition | calculation | comparison | explanation | exam_tip\n"
             "3. Identify the command verb (state, explain, describe, compare, calculate, evaluate, etc.)\n"
             "4. Identify the main subject topic (e.g. 'Data Compression', 'Interrupt Handling')\n"
             'Return JSON: {"keywords": ["term1"], "qtype": "definition", "verb": "explain", "topic": "Topic Name"}'),
    zh_user=("问题: %(question)s\n\n"
             "1. 提取3-5个英文关键词/技术术语用于知识库检索（将中文概念翻译为英文术语）\n"
             "2. 分类: definition(定义) | calculation(计算) | comparison(对比) | explanation(解释) | exam_tip(考试技巧)\n"
             "3. 识别指令动词(state/explain/describe/compare/calculate/evaluate...)\n"
             "4. 识别主题名称（如 'Data Compression', 'Interrupt Handling'）\n"
             '返回 JSON: {"keywords": ["term1"], "qtype": "definition", "verb": "explain", "topic": "主题名称"}'),
)

CRITIC = PromptTemplate(
    en_system="Review this tutoring answer for quality. Output JSON.",
    zh_system="审查此教学回答的质量。Output JSON。",
    en_user=("Question: %(question)s\n\n"
             "References:\n%(ctx)s\n"
             "Answer to review:\n%(answer)s\n\n"
             "Check:\n"
             "1. Are technical facts consistent with the references? (no hallucinations)\n"
             "2. Are the references quoted verbatim (not translated)?\n"
             "3. Is the explanation clear and complete?\n"
             "4. Are [KB]/[General] markers used correctly?\n"
             'Return JSON: {"pass": true/false, "feedback": "specific issues if any"}'),
    zh_user=("问题: %(question)s\n\n"
             "参考资料:\n%(ctx)s\n"
             "待审查回答:\n%(answer)s\n\n"
             "检查:\n"
             "1. 技术事实是否与参考资料一致？（无幻觉）\n"
             "2. 参考资料引用是否保持原文（未被翻译）？\n"
             "3. 解释是否清晰完整？\n"
             "4. [KB]/[General] 标记是否正确使用？\n"
             '返回 JSON: {"pass": true/false, "feedback": "具体问题（如有）"}'),
)

SUGGEST = PromptTemplate(
    en_system="Suggest 2-3 follow-up questions a student might ask. Output JSON.",
    zh_system="建议2-3个学生可能追问的问题。Output JSON。",
    en_user=("Student asked: %(question)s\n"
             "Topics covered: %(topic_str)s\n"
             "%(prereq_hint)s"
             "Suggest 2-3 natural follow-up questions.\n"
             'Include: (a) a deeper question on the same topic, '
             '(b) a question linking to a prerequisite or related topic.\n'
             'Return JSON: {"suggestions": ["question 1", "question 2"]}'),
    zh_user=("学生问: %(question)s\n"
             "涉及主题: %(topic_str)s\n"
             "%(prereq_hint)s"
             "建议2-3个自然的追问。\n"
             '包含: (a) 同一主题的深入问题, '
             '(b) 链接到先修或相关主题的问题。\n'
             '返回 JSON: {"suggestions": ["问题 1", "问题 2"]}'),
)

# ============================================================
# Offline analysis templates
# ============================================================

VERB_PATTERN_SUMMARY = PromptTemplate(
    en_system="You are an exam pattern analyst. Summarize how to answer this type of question. Output JSON.",
    zh_system="你是一个考试规律分析专家。总结这类题目的答题模式。Output JSON。",
    en_user=("Command verb: '%(verb)s'\nSample count: %(sample_count)s\n"
             "Avg answer length: %(avg_answer_length)s chars\n"
             "Bullet point usage: %(bullet_pct)s\n"
             "Avg miss rate (AI answering without markscheme): %(avg_miss_rate)s\n\n"
             "Example QAs:\n%(qa_texts)s\n\n"
             "Summarize: typical_structure, expected_depth, scoring_pattern, "
             "common_pitfalls, full_mark_formula.\n"
             'Return: {"pattern_summary": "concise structured description"}'),
    zh_user=("指令动词: '%(verb)s'\n样本数: %(sample_count)s\n"
             "平均答案长度: %(avg_answer_length)s 字符\n"
             "列表格式使用率: %(bullet_pct)s\n\n"
             "示例问答:\n%(qa_texts)s\n\n"
             '返回: {"pattern_summary": "结构化答题规律描述"}'),
)

DIFFICULTY_RATE = PromptTemplate(
    en_system=("You are an exam difficulty assessor. Rate each question's difficulty for students. "
               "Output JSON.\n\n"
               "Three levels:\n"
               "- basic: direct recall/recognition/simple formula. Student just needs to remember a definition or perform one operation.\n"
               "- intermediate: requires understanding relationships between concepts, multi-step reasoning, or accurate rule application.\n"
               "- advanced: requires synthesizing multiple concepts, evaluation/judgment, or precise complex procedures.\n\n"
               "Base your rating on the question text and the markscheme answer. Do not rely on your prior knowledge of the subject."),
    zh_system=("你是一个考试难度评估专家。评估每道题对学生的困难程度。Output JSON。\n"
               "三级：basic（直接回忆/简单操作），intermediate（理解关系/多步推理），advanced（综合/评估/复杂步骤）。"),
    en_user=("%(qa_block)s\n\n"
             'Return: {"ratings": [{"question_index": 0, "difficulty": "basic", "reasoning": "..."}, ...]}'),
    zh_user=("%(qa_block)s\n\n"
             '返回: {"ratings": [{"question_index": 0, "difficulty": "basic", "reasoning": "..."}, ...]}'),
)

DEPENDENCY_VALIDATE = PromptTemplate(
    en_system=("You are a curriculum design expert. For each topic pair, determine if topic A "
               "is a prerequisite for understanding topic B. Output JSON.\n\n"
               "Scoring: 2=prerequisite, 1=related, 0=independent.\n"
               "Base your judgment ONLY on the QA texts provided."),
    zh_system="判断每个 topic 对中 A 是否为 B 的前置知识。Output JSON。",
    en_user=("Evaluate these topic pairs:\n\n%(pairs_block)s\n\n"
             'Return: {"pairs": [{"index": 0, "score": 2, "reason": "..."}, ...]}'),
    zh_user=("评估以下 topic 对：\n\n%(pairs_block)s\n\n"
             '返回: {"pairs": [{"index": 0, "score": 2, "reason": "..."}, ...]}'),
)

# ============================================================
# Answer generator — kept as function due to dynamic type_guide
# ============================================================

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
            '返回 JSON: {"answer": "你的回答", '
            '"quiz": {"question": "...", "expected": "..."}, '
            '"path_hint": {"next_topic": "...", "reason": "..."}}'
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
            'Return JSON: {"answer": "your answer", '
            '"quiz": {"question": "...", "expected": "..."}, '
            '"path_hint": {"next_topic": "...", "reason": "..."}}'
        )

    msgs: list[dict] = [{"role": "system", "content": sys}]
    for h in history[-6:]:
        role = "assistant" if h.get("role") == "assistant" else "user"
        msgs.append({"role": role, "content": h.get("content", "")})
    msgs.append({"role": "user", "content": usr})
    return msgs
