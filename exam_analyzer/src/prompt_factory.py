"""Bilingual prompt factory — unified PromptBuilder entry point.

Uses ``%``-formatting (NOT ``str.format``) because exam text (mark schemes,
code snippets) frequently contains literal ``{`` ``}`` that would crash
``.format()``.  All string values are auto-escaped via ``_escape_pct`` so
that ``%`` characters in user content are safe.

Architecture::

    PromptBuilder.build(PromptType.QA_PAIRING, qp_text=..., ms_text=..., ...)
      └─ _build_qa_pairing(kwargs)
           └─ _QA_PAIRING_TMPL.build(lang=..., ...)

Callers import PromptType + PromptBuilder.  Template instances are private.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from .embedding_cluster import detect_content_lang


# ============================================================
# Utils
# ============================================================

def _escape_pct(text: str) -> str:
    """Double ``%`` signs so user text is safe for ``%``-format interpolation."""
    return text.replace("%", "%%")


# ============================================================
# PromptTemplate — bilingual (en+zh) pair with %-formatting
# ============================================================

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
# PromptType enum — one member per semantic intent
# ============================================================

class PromptType(Enum):
    """Every LLM call site maps to exactly one PromptType."""
    QA_PAIRING = "qa_pairing"        # Phase 1/2: pair QP questions with MS answers
    SUMMARY = "summary"              # Phase 1/2: summarise QA into knowledge summary + topic
    ANSWER = "answer"                # Phase 2: Flash answers new question using history
    GRADE = "grade"                  # Phase 2: grade predicted answer against MS
    FRAGMENT = "fragment"            # Phase 1: split MS answer into scoring points
    KP_CLASSIFY = "kp_classify"      # Layer 1: score new QA against all existing KPs
    QUERY_ANALYSIS = "query_analysis"  # Chat Agent 1: analyse student question
    ANSWER_GEN = "answer_gen"        # Chat Agent 3: generate tutoring answer
    CRITIC = "critic"                # Chat Agent 4: review answer quality
    SUGGEST = "suggest"              # Chat Agent 5: follow-up questions
    VERB_PATTERN = "verb_pattern"    # Offline: summarise command-verb patterns
    DIFFICULTY = "difficulty"        # Offline: rate question difficulty
    DEPENDENCY = "dependency"        # Offline: validate topic dependencies


# ============================================================
# Private template instances (prefixed with _ to discourage direct use)
# ============================================================

_QA_PAIRING_TMPL = PromptTemplate(
    en_system="You are an exam paper pairing assistant. Match each question in the "
              "Question Paper with its answer in the Mark Scheme. Preserve original "
              "question numbering. Output JSON.",
    zh_system="你是一个考试试卷配题助手。将试卷(Question Paper)中的题目"
              "与答案(Markscheme)中的对应答案配对。保留试卷的原始题目编号结构。Output JSON。",
    en_user=("Paper: %(display_name)s\n\n"
             "=== Question Paper (QP) ===\n%(qp_text)s\n\n"
             "=== Mark Scheme (MS) ===\n%(ms_text)s\n\n"
             "Match each question with its answer.\n"
             'Return JSON: {"qa_pairs": [{"question_number": "2(a)", '
             '"parent_question": "2", "question_text": "...", "answer_text": "..."}]}\n\n'
             "Notes:\n"
             "1. question_number: original numbering (e.g. '1(a)', '2(b)(iii)')\n"
             "2. parent_question: parent question number (e.g. '1', '2'), "
             "same as question_number if no sub-questions\n"
             "3. question_text: use original question text\n"
             "4. answer_text: use the corresponding mark scheme text "
             "(do not omit any mark points)\n"
             "5. If a question has no answer in the mark scheme, set answer_text to empty string"),
    zh_user=("试卷名称：%(display_name)s\n\n"
             "=== 试卷内容 (QP) ===\n%(qp_text)s\n\n"
             "=== 答案内容 (MS) ===\n%(ms_text)s\n\n"
             "请将每个题目与其答案配对。\n"
             '返回 JSON 格式：\n'
             '{"qa_pairs": [{"question_number": "2(a)", '
             '"parent_question": "2", "question_text": "...", "answer_text": "..."}]}\n\n'
             "注意：\n"
             "1. question_number 保留试卷原始编号（如 '1(a)', '2(b)(iii)'）\n"
             "2. parent_question 为大题编号（如 '1', '2'），无小问时与 question_number 相同\n"
             "3. question_text 使用题目原文\n"
             "4. answer_text 使用答案中对应的得分点原文（不要省略任何得分点）\n"
             "5. 如果某题目在 markscheme 中没有对应答案，answer_text 设为空字符串"),
)

_SUMMARY_TMPL = PromptTemplate(
    en_system="You are an exam knowledge classification expert. Do two things at once. "
              "Output JSON. 1. Describe the core technical concept tested in 1-2 sentences. "
              "2. Assign a concise topic name.",
    zh_system="你是一个考试知识分类专家。同时完成两件事。Output JSON."
              "1. 用1-2句描述这道题考察的核心技术概念"
              "2. 分配一个简洁的主题名称",
    en_user=("Question: %(question_text)s\n\nAnswer: %(answer_text)s\n\n"
             "%(topic_hint)s"
             "Do not mention exam-specific context (numbers, filenames, scenarios, names).\n"
             "Topic name should use standard terminology (e.g. 'Data Compression', 'Interrupt Handling').\n"
             'Return JSON: {"summary": "core knowledge description", "topic": "standard topic name"}'),
    zh_user=("题目: %(question_text)s\n\n答案: %(answer_text)s\n\n"
             "%(topic_hint)s"
             "不要提及题目特定上下文（具体数值、文件名、场景描述、人名）。\n"
             "主题名称应使用标准术语（如 'Data Compression', 'Interrupt Handling'）。\n"
             '返回 JSON: {"summary": "核心知识描述", "topic": "标准主题名"}'),
)

_ANSWER_TMPL = PromptTemplate(
    en_system="You are an exam answering system. Use historical Q&A to answer new questions. Output JSON.",
    zh_system="你是一个考试答题系统。利用历史题目的答案来回答新题。Output JSON。",
    en_user=("%(qa_block)s=== New Question ===\n%(question_text)s\n\n"
             "Task:\n"
             "1. Answer the new question using knowledge from the historical Q&A\n"
             "2. Mark which historical Q&As were used (by index 1, 2...)\n"
             'Return JSON: {"answer": "...", "used_qa_indices": [1, 3]}'),
    zh_user=("%(qa_block)s=== 新题目 ===\n%(question_text)s\n\n"
             "任务:\n"
             "1. 利用历史题目的知识回答这道新题\n"
             "2. 标注使用了哪些历史题目（用序号 1, 2...）\n"
             '返回 JSON: {"answer": "...", "used_qa_indices": [1, 3]}'),
)

_GRADE_TMPL = PromptTemplate(
    en_system="You are an exam grading expert. Compare student answer against mark scheme "
              "point by point. For each missed point, note the reason. Assign a topic. Output JSON.",
    zh_system="你是一个考试批改专家。对比学生答案和标准答案，逐得分点评分。"
              "对每个遗漏的得分点标注遗漏原因。为这道题分配一个主题名称。Output JSON。",
    en_user=("Question: %(question_text)s\n\n"
             "Student answer: %(predicted_answer)s\n\n"
             "Mark scheme answer: %(ms_answer)s\n\n"
             "Task:\n"
             "1. Compare student answer to mark scheme point by point\n"
             "2. List covered points. For each missed point, provide content and reason:\n"
             "   - knowledge_gap: student answer doesn't cover this concept (lacks knowledge)\n"
             "   - misinterpretation: student misunderstood the requirement\n"
             "   - insufficient_detail: student answer is on the right track but not precise/complete enough\n"
             "   - retrieval_quality: student answer was correct but the reference Q&A didn't provide this info\n"
             "3. Assign a topic name for the question (standard terminology)\n"
             'Return JSON: {"covered_points": ["covered"], '
             '"missed_points": [{"point": "exact wording", "reason": "knowledge_gap"}], '
             '"topic": "standard topic name", "help_level": "direct|understanding|none"}'),
    zh_user=("题目: %(question_text)s\n\n"
             "学生答案: %(predicted_answer)s\n\n"
             "标准答案 (Markscheme): %(ms_answer)s\n\n"
             "任务:\n"
             "1. 对比学生答案和标准答案的每个得分点\n"
             "2. 列出覆盖的得分点。对每个遗漏的得分点，提供内容及遗漏原因：\n"
             "   - knowledge_gap: 学生答案未涉及此概念（缺乏相关知识）\n"
             "   - misinterpretation: 学生理解偏了题目要求\n"
             "   - insufficient_detail: 学生答了方向对但不够精确/完整\n"
             "   - retrieval_quality: 学生答案对但参考QA未提供此信息\n"
             "3. 为这道题分配一个主题名称（标准术语）\n"
             '返回 JSON: {"covered_points": ["已覆盖"], '
             '"missed_points": [{"point": "原措辞", "reason": "knowledge_gap"}], '
             '"topic": "标准主题名", "help_level": "direct|understanding|none"}'),
)

_FRAGMENT_TMPL = PromptTemplate(
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

_KP_CLASSIFY_TMPL = PromptTemplate(
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

_QUERY_ANALYSIS_TMPL = PromptTemplate(
    en_system="Analyze the student's question. Output JSON.",
    zh_system="分析学生问题。Output JSON。",
    en_user=("Question: %(question)s\n\n"
             "1. Extract 3-5 English keywords/technical terms for knowledge base search\n"
             "2. Classify as: definition | calculation | comparison | explanation | exam_tip\n"
             "3. Identify the command verb (state, explain, describe, compare, calculate, evaluate, etc.)\n"
             "4. Identify the main subject topic (e.g. 'Data Compression', 'Interrupt Handling')\n"
             'Return JSON: {"keywords": ["term1"], "qtype": "definition", '
             '"verb": "explain", "topic": "Topic Name"}'),
    zh_user=("问题: %(question)s\n\n"
             "1. 提取3-5个英文关键词/技术术语用于知识库检索（将中文概念翻译为英文术语）\n"
             "2. 分类: definition(定义) | calculation(计算) | comparison(对比) | explanation(解释) | exam_tip(考试技巧)\n"
             "3. 识别指令动词(state/explain/describe/compare/calculate/evaluate...)\n"
             "4. 识别主题名称（如 'Data Compression', 'Interrupt Handling'）\n"
             '返回 JSON: {"keywords": ["term1"], "qtype": "definition", '
             '"verb": "explain", "topic": "主题名称"}'),
)

_CRITIC_TMPL = PromptTemplate(
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

_SUGGEST_TMPL = PromptTemplate(
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

_VERB_PATTERN_TMPL = PromptTemplate(
    en_system="You are an exam pattern analyst. Summarise how to answer this type of question. Output JSON.",
    zh_system="你是一个考试规律分析专家。总结这类题目的答题模式。Output JSON。",
    en_user=("Command verb: '%(verb)s'\nSample count: %(sample_count)s\n"
             "Avg answer length: %(avg_answer_length)s chars\n"
             "Bullet point usage: %(bullet_pct)s\n"
             "Avg miss rate (AI answering without markscheme): %(avg_miss_rate)s\n\n"
             "Example QAs:\n%(qa_texts)s\n\n"
             "Summarise: typical_structure, expected_depth, scoring_pattern, "
             "common_pitfalls, full_mark_formula.\n"
             'Return: {"pattern_summary": "concise structured description"}'),
    zh_user=("指令动词: '%(verb)s'\n样本数: %(sample_count)s\n"
             "平均答案长度: %(avg_answer_length)s 字符\n"
             "列表格式使用率: %(bullet_pct)s\n\n"
             "示例问答:\n%(qa_texts)s\n\n"
             '返回: {"pattern_summary": "结构化答题规律描述"}'),
)

_DIFFICULTY_TMPL = PromptTemplate(
    en_system=("You are an exam difficulty assessor. Rate each question's difficulty for students. "
               "Output JSON.\n\n"
               "Three levels:\n"
               "- basic: direct recall/recognition/simple formula. Student just needs to "
               "remember a definition or perform one operation.\n"
               "- intermediate: requires understanding relationships between concepts, "
               "multi-step reasoning, or accurate rule application.\n"
               "- advanced: requires synthesising multiple concepts, evaluation/judgment, "
               "or precise complex procedures.\n\n"
               "Base your rating on the question text and the markscheme answer. "
               "Do not rely on your prior knowledge of the subject."),
    zh_system=("你是一个考试难度评估专家。评估每道题对学生的困难程度。Output JSON。\n"
               "三级：basic（直接回忆/简单操作），intermediate（理解关系/多步推理），advanced（综合/评估/复杂步骤）。"),
    en_user=("%(qa_block)s\n\n"
             'Return: {"ratings": [{"question_index": 0, "difficulty": "basic", "reasoning": "..."}, ...]}'),
    zh_user=("%(qa_block)s\n\n"
             '返回: {"ratings": [{"question_index": 0, "difficulty": "basic", "reasoning": "..."}, ...]}'),
)

_DEPENDENCY_TMPL = PromptTemplate(
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
# Backward-compatible aliases (so existing callers don't break immediately)
# ============================================================

FRAGMENT = _FRAGMENT_TMPL
QA_CLASSIFY = _KP_CLASSIFY_TMPL
QUERY_ANALYST = _QUERY_ANALYSIS_TMPL
CRITIC = _CRITIC_TMPL
SUGGEST = _SUGGEST_TMPL
VERB_PATTERN_SUMMARY = _VERB_PATTERN_TMPL
DIFFICULTY_RATE = _DIFFICULTY_TMPL
DEPENDENCY_VALIDATE = _DEPENDENCY_TMPL


# ============================================================
# PromptBuilder — single entry point for all prompt construction
# ============================================================

class PromptBuilder:
    """Builds ``list[dict]`` messages for any prompt type via ``match/case`` dispatch.

    Usage::

        from .prompt_factory import PromptType, PromptBuilder
        msgs = PromptBuilder.build(PromptType.QA_PAIRING,
                                   display_name=pair.display_name,
                                   qp_text=..., ms_text=...)
    """

    @staticmethod
    def build(prompt_type: PromptType, **kwargs) -> list[dict]:
        """Return ``[{"role":"system",...}, {"role":"user",...}]`` for *prompt_type*.

        All keyword arguments are forwarded to the type-specific builder.
        """
        match prompt_type:
            case PromptType.QA_PAIRING:
                return PromptBuilder._build_qa_pairing(kwargs)
            case PromptType.SUMMARY:
                return PromptBuilder._build_summary(kwargs)
            case PromptType.ANSWER:
                return PromptBuilder._build_answer(kwargs)
            case PromptType.GRADE:
                return PromptBuilder._build_grade(kwargs)
            case PromptType.FRAGMENT:
                return PromptBuilder._build_fragment(kwargs)
            case PromptType.KP_CLASSIFY:
                return PromptBuilder._build_kp_classify(kwargs)
            case PromptType.QUERY_ANALYSIS:
                return PromptBuilder._build_query_analysis(kwargs)
            case PromptType.ANSWER_GEN:
                return PromptBuilder._build_answer_gen(kwargs)
            case PromptType.CRITIC:
                return PromptBuilder._build_critic(kwargs)
            case PromptType.SUGGEST:
                return PromptBuilder._build_suggest(kwargs)
            case PromptType.VERB_PATTERN:
                return PromptBuilder._build_verb_pattern(kwargs)
            case PromptType.DIFFICULTY:
                return PromptBuilder._build_difficulty(kwargs)
            case PromptType.DEPENDENCY:
                return PromptBuilder._build_dependency(kwargs)
            case _:
                raise ValueError(f"Unknown PromptType: {prompt_type}")

    # ── QA_PAIRING ───────────────────────────────────────────

    @staticmethod
    def _build_qa_pairing(kw: dict) -> list[dict]:
        display_name = kw.get("display_name", "")
        qp_text = kw.get("qp_text", "")
        ms_text = kw.get("ms_text", "")
        lang = kw.get("lang", "") or detect_content_lang(qp_text + ms_text)
        return _QA_PAIRING_TMPL.build(
            lang=lang, display_name=display_name,
            qp_text=qp_text, ms_text=ms_text,
        )

    # ── SUMMARY ──────────────────────────────────────────────

    @staticmethod
    def _build_summary(kw: dict) -> list[dict]:
        question_text = kw.get("question_text", "")
        answer_text = kw.get("answer_text", "")
        topic_list = kw.get("topic_list", "")
        lang = kw.get("lang", "") or detect_content_lang(question_text + answer_text)
        topic_hint = ""
        if topic_list:
            if lang == "zh":
                topic_hint = (f"\n已有主题（若适用请复用）: {topic_list}\n"
                              "如果这道题匹配已有主题，请使用完全相同的名称。"
                              "只有在已有主题都不适用时才创建新主题名。\n")
            else:
                topic_hint = (f"\nExisting topics (reuse if applicable): {topic_list}\n"
                              "If this question matches an existing topic, use the exact same name. "
                              "Only create a new topic if none of the existing ones fit.\n")
        return _SUMMARY_TMPL.build(
            lang=lang, question_text=question_text,
            answer_text=answer_text, topic_hint=topic_hint,
        )

    # ── ANSWER ───────────────────────────────────────────────

    @staticmethod
    def _build_answer(kw: dict) -> list[dict]:
        qa_block = kw.get("qa_block", "")
        question_text = kw.get("question_text", "")
        lang = kw.get("lang", "") or detect_content_lang(question_text + qa_block)
        return _ANSWER_TMPL.build(
            lang=lang, qa_block=qa_block, question_text=question_text,
        )

    # ── GRADE ────────────────────────────────────────────────

    @staticmethod
    def _build_grade(kw: dict) -> list[dict]:
        question_text = kw.get("question_text", "")
        predicted_answer = kw.get("predicted_answer", "")
        ms_answer = kw.get("ms_answer", "")
        lang = kw.get("lang", "") or detect_content_lang(question_text + ms_answer)
        return _GRADE_TMPL.build(
            lang=lang, question_text=question_text,
            predicted_answer=predicted_answer, ms_answer=ms_answer,
        )

    # ── FRAGMENT ─────────────────────────────────────────────

    @staticmethod
    def _build_fragment(kw: dict) -> list[dict]:
        answer_text = kw.get("answer_text", "")
        lang = kw.get("lang", "") or detect_content_lang(answer_text)
        return _FRAGMENT_TMPL.build(lang=lang, answer_text=answer_text)

    # ── KP_CLASSIFY ──────────────────────────────────────────

    @staticmethod
    def _build_kp_classify(kw: dict) -> list[dict]:
        qa_text = kw.get("qa_text", "")
        answer_text = kw.get("answer_text", "")
        kp_list = kw.get("kp_list", "")
        lang = kw.get("lang", "") or detect_content_lang(qa_text + answer_text)
        return _KP_CLASSIFY_TMPL.build(
            lang=lang, qa_text=qa_text,
            answer_text=answer_text, kp_list=kp_list,
        )

    # ── QUERY_ANALYSIS ───────────────────────────────────────

    @staticmethod
    def _build_query_analysis(kw: dict) -> list[dict]:
        question = kw.get("question", "")
        lang = kw.get("lang", "") or detect_content_lang(question)
        return _QUERY_ANALYSIS_TMPL.build(lang=lang, question=question)

    # ── ANSWER_GEN (Chat Agent 3) ────────────────────────────

    @staticmethod
    def _build_answer_gen(kw: dict) -> list[dict]:
        """Build answer-generator messages with type-adaptive prompt + history.

        Merged from ``build_answer_generator_messages``.
        """
        question = kw.get("question", "")
        qtype = kw.get("qtype", "explanation")
        lang = kw.get("lang", "en")
        ctx = kw.get("ctx", "")
        ctx_kp = kw.get("ctx_kp", "")
        history = kw.get("history", [])

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

    # ── CRITIC ───────────────────────────────────────────────

    @staticmethod
    def _build_critic(kw: dict) -> list[dict]:
        question = kw.get("question", "")
        ctx = kw.get("ctx", "")
        answer = kw.get("answer", "")
        lang = kw.get("lang", "") or detect_content_lang(question + ctx)
        return _CRITIC_TMPL.build(
            lang=lang, question=question, ctx=ctx, answer=answer,
        )

    # ── SUGGEST ──────────────────────────────────────────────

    @staticmethod
    def _build_suggest(kw: dict) -> list[dict]:
        question = kw.get("question", "")
        topic_str = kw.get("topic_str", "")
        prereq_hint = kw.get("prereq_hint", "")
        lang = kw.get("lang", "") or detect_content_lang(question)
        return _SUGGEST_TMPL.build(
            lang=lang, question=question,
            topic_str=topic_str, prereq_hint=prereq_hint,
        )

    # ── VERB_PATTERN ─────────────────────────────────────────

    @staticmethod
    def _build_verb_pattern(kw: dict) -> list[dict]:
        verb = kw.get("verb", "")
        sample_count = kw.get("sample_count", "")
        avg_answer_length = kw.get("avg_answer_length", "")
        bullet_pct = kw.get("bullet_pct", "")
        avg_miss_rate = kw.get("avg_miss_rate", "")
        qa_texts = kw.get("qa_texts", "")
        lang = kw.get("lang", "en")
        return _VERB_PATTERN_TMPL.build(
            lang=lang, verb=verb, sample_count=sample_count,
            avg_answer_length=avg_answer_length, bullet_pct=bullet_pct,
            avg_miss_rate=avg_miss_rate, qa_texts=qa_texts,
        )

    # ── DIFFICULTY ───────────────────────────────────────────

    @staticmethod
    def _build_difficulty(kw: dict) -> list[dict]:
        qa_block = kw.get("qa_block", "")
        lang = kw.get("lang", "") or detect_content_lang(qa_block)
        return _DIFFICULTY_TMPL.build(lang=lang, qa_block=qa_block)

    # ── DEPENDENCY ───────────────────────────────────────────

    @staticmethod
    def _build_dependency(kw: dict) -> list[dict]:
        pairs_block = kw.get("pairs_block", "")
        lang = kw.get("lang", "en")
        return _DEPENDENCY_TMPL.build(lang=lang, pairs_block=pairs_block)


# ============================================================
# build_answer_generator_messages — backward-compatible wrapper
# ============================================================

def build_answer_generator_messages(
    question: str, qtype: str, lang: str, ctx: str, ctx_kp: str,
    history: list[dict],
) -> list[dict]:
    """Backward-compatible wrapper — delegates to PromptBuilder."""
    return PromptBuilder.build(
        PromptType.ANSWER_GEN,
        question=question, qtype=qtype, lang=lang,
        ctx=ctx, ctx_kp=ctx_kp, history=history,
    )
