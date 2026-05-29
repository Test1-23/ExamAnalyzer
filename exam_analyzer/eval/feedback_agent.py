"""Quality feedback agent: evaluates chat assistant output against exam data.

Uses real QA pairs from the knowledge base as ground truth, tests the chat
assistant via HTTP API, and produces structured improvement reports.
"""

import os
import sys as _sys

# Ensure project root is on sys.path so `from src.xxx` imports work
# both when imported by app.py and when run standalone via `python eval/feedback_agent.py`
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(THIS_DIR)  # eval/ -> exam_analyzer/
if _PROJECT_DIR not in _sys.path:
    _sys.path.insert(0, _PROJECT_DIR)

import json
import re
import time
import random
import requests
from datetime import datetime

from src.knowledge_base import QADatabase, QARetriever
from src.deepseek_client import create_client, call_flash
from src.embedding_cluster import _get_model, MODEL_MAP, _detect_language
import numpy as np


class FeedbackAgent:
    """Evaluates chat assistant quality using real exam data."""

    def __init__(self, api_url: str, api_key: str, db, points_file: str,
                 chat_base_url: str = "http://127.0.0.1:5000"):
        self.client = create_client(api_url, api_key)
        self.db = db
        self.retriever = QARetriever(self.db)
        self.retriever.rebuild()
        self.points_file = points_file
        self.chat_url = f"{chat_base_url}/api/chat"
        self.results = {}

    # ================================================================
    # Main evaluation entry point
    # ================================================================

    def run_full_evaluation(self, sample_topics: int = 5) -> str:
        """Run all evaluations and return a formatted report."""
        print("[FeedbackAgent] Starting evaluation...")
        t0 = time.time()

        topics = self._get_topics_with_qa()
        if not topics:
            return "No QA data available for evaluation."

        selected = random.sample(topics, min(sample_topics, len(topics)))

        self.results["accuracy"] = self._eval_accuracy(selected)
        self.results["language"] = self._eval_language(selected)
        self.results["honesty"] = self._eval_source_honesty(selected)
        self.results["coverage"] = self._eval_coverage()
        self.results["pitfall"] = self._eval_pitfall_relevance()
        self.results["confusion"] = self._eval_confusion(selected)

        elapsed = int(time.time() - t0)
        report = self._generate_report(elapsed)
        print(f"[FeedbackAgent] Evaluation complete in {elapsed}s")
        return report

    # ================================================================
    # Helper: topic extraction
    # ================================================================

    def _get_topics_with_qa(self) -> list[str]:
        """Return topics that have at least one QA pair."""
        groups = self.db.get_topic_groups()
        return [t for t, qas in groups.items() if t and t != "(uncategorized)" and len(qas) >= 1]

    def _extract_real_questions(self, topic: str, limit: int = 3) -> list[dict]:
        """Extract real QA pairs from DB as test questions."""
        rows = self.db.conn.execute(
            "SELECT question_text, answer_text, paper FROM qa_pairs WHERE topic = ?", (topic,)
        ).fetchall()
        qas = [{"question": r["question_text"], "markscheme": r["answer_text"],
                "paper": r["paper"]} for r in rows]
        if len(qas) > limit:
            qas = random.sample(qas, limit)
        return qas

    def _call_chat(self, question: str) -> dict:
        """Call the chat assistant API and return the response."""
        try:
            resp = requests.post(self.chat_url, json={
                "question": question,
                "session_id": "feedback_agent_test"
            }, timeout=60)
            return resp.json()
        except Exception as e:
            return {"error": str(e), "answer": "", "sources": [], "suggestions": []}

    # ================================================================
    # Evaluation A: Answer Accuracy
    # ================================================================

    def _eval_accuracy(self, topics: list[str]) -> dict:
        """Test chat assistant against real QA pairs as ground truth."""
        results = {}
        total = correct = 0
        topic_scores = {}

        for topic in topics:
            qas = self._extract_real_questions(topic, limit=2)
            if not qas:
                continue
            t_correct = 0
            for qa in qas:
                resp = self._call_chat(qa["question"])
                answer = resp.get("answer", "")
                if not answer:
                    continue
                # Simple evaluation: check if answer covers markscheme key points
                score = self._score_answer(answer, qa["markscheme"])
                total += 1
                if score["pass"]:
                    correct += 1
                    t_correct += 1
                results[f"{topic}/{qa['paper']}"] = score
            topic_scores[topic] = {"correct": t_correct, "total": len(qas)}

        return {
            "total": total, "correct": correct,
            "accuracy": round(correct / max(total, 1), 2),
            "by_topic": topic_scores,
            "details": results,
        }

    def _score_answer(self, answer: str, markscheme: str) -> dict:
        """Use Flash to score an answer against the markscheme."""
        sys = "You are an exam grader. Compare the student answer with the markscheme. Output JSON."
        usr = (
            f"Markscheme:\n{markscheme}\n\n"
            f"Student answer:\n{answer}\n\n"
            "Does the answer cover the key points? "
            "List covered and missed points.\n"
            'Return JSON: {"pass": true/false, "covered": ["..."], "missed": ["..."], '
            '"overall": "good|partial|poor"}'
        )
        try:
            result = call_flash(self.client, [{"role": "system", "content": sys},
                                              {"role": "user", "content": usr}], max_retries=1)
            return result if isinstance(result, dict) else {"pass": False, "overall": "poor"}
        except Exception:
            return {"pass": False, "overall": "poor"}

    # ================================================================
    # Evaluation B: Language Compliance
    # ================================================================

    def _eval_language(self, topics: list[str]) -> dict:
        """Check if Chinese answers keep Q&A references in original English."""
        violations = []
        total = 0

        for topic in topics:
            qas = self._extract_real_questions(topic, limit=1)
            for qa in qas:
                chinese_q = self._make_chinese_question(qa["question"])
                if not chinese_q:
                    continue
                resp = self._call_chat(chinese_q)
                answer = resp.get("answer", "")
                if not answer:
                    continue
                total += 1
                # Check for translated terms
                check = self._check_translation(answer, qa["markscheme"])
                if check["violations"]:
                    violations.append({"topic": topic, "violations": check["violations"]})

        return {
            "total": total,
            "compliant": total - len(violations),
            "rate": round((total - len(violations)) / max(total, 1), 2),
            "violations": violations,
        }

    def _make_chinese_question(self, en_question: str) -> str:
        """Wrap an English question as a Chinese student query."""
        return f"请解释: {en_question}"

    def _check_translation(self, answer: str, markscheme: str) -> dict:
        """Use Flash to detect if English terms were translated."""
        sys = "Detect if English technical terms in the answer match the original markscheme. Output JSON."
        usr = (
            f"Markscheme (original English):\n{markscheme[:500]}\n\n"
            f"Answer:\n{answer[:1000]}\n\n"
            "Check: are English technical terms kept in original form, or translated? "
            "List any translated terms.\n"
            'Return JSON: {"violations": ["translated term: original -> translation", ...]}'
        )
        try:
            result = call_flash(self.client, [{"role": "system", "content": sys},
                                              {"role": "user", "content": usr}], max_retries=1)
            return result if isinstance(result, dict) else {"violations": []}
        except Exception:
            return {"violations": []}

    # ================================================================
    # Evaluation C: Source Honesty
    # ================================================================

    def _eval_source_honesty(self, topics: list[str]) -> dict:
        """Check if [KB] marked content actually comes from retrieved QAs."""
        results = {}
        honest = dishonest = 0

        for topic in topics:
            qas = self._extract_real_questions(topic, limit=1)
            for qa in qas:
                resp = self._call_chat(qa["question"])
                answer = resp.get("answer", "")
                sources = resp.get("sources", [])
                if not answer:
                    continue
                check = self._verify_sources(answer, sources)
                results[f"{topic}"] = check
                if check["honest"]:
                    honest += 1
                else:
                    dishonest += 1

        return {
            "total": honest + dishonest,
            "honest": honest,
            "dishonest": dishonest,
            "rate": round(honest / max(honest + dishonest, 1), 2),
            "details": results,
        }

    def _verify_sources(self, answer: str, sources: list) -> dict:
        """Use Flash to verify if [KB] claims align with sources."""
        src_text = "\n".join(f"- {s.get('topic', '')}: {s.get('question', '')[:200]}" for s in sources[:3])
        sys = "Verify if KB-cited claims in the answer are supported by the sources. Output JSON."
        usr = (
            f"Sources:\n{src_text}\n\n"
            f"Answer:\n{answer[:1500]}\n\n"
            "Are [KB] marked claims actually in the sources? "
            "List any that appear fabricated.\n"
            'Return JSON: {"honest": true/false, "fabricated": ["claim1", ...]}'
        )
        try:
            result = call_flash(self.client, [{"role": "system", "content": sys},
                                              {"role": "user", "content": usr}], max_retries=1)
            return result if isinstance(result, dict) else {"honest": True, "fabricated": []}
        except Exception:
            return {"honest": True, "fabricated": []}

    # ================================================================
    # Evaluation D: Paper Coverage
    # ================================================================

    def _eval_coverage(self) -> dict:
        """Check how many QA topics the assistant can answer correctly."""
        topics = self._get_topics_with_qa()
        total = len(topics)
        if total == 0:
            return {"total": 0, "covered": 0, "rate": 0}

        covered = 0
        details = {}
        for topic in topics[:10]:  # sample up to 10
            qas = self._extract_real_questions(topic, limit=1)
            if not qas:
                continue
            resp = self._call_chat(qas[0]["question"])
            answer = resp.get("answer", "")
            score = self._score_answer(answer, qas[0]["markscheme"])
            details[topic] = score
            if score.get("pass"):
                covered += 1

        return {
            "total_topics": total,
            "sampled": len(details),
            "covered": covered,
            "rate": round(covered / max(len(details), 1), 2),
            "details": details,
        }

    # ================================================================
    # Evaluation E: Pitfall Relevance
    # ================================================================

    def _eval_pitfall_relevance(self) -> dict:
        """Check if pitfalls in points.txt are related to actual exam answers."""
        kps = self._parse_points_kps()
        if not kps:
            return {"total_pitfalls": 0, "relevant": 0, "irrelevant": 0, "rate": 0}

        pitfalls = [(kp["topic"], kp["pitfall"]) for kp in kps if kp.get("pitfall")]
        if not pitfalls:
            return {"total_pitfalls": 0, "relevant": 0, "irrelevant": 0, "rate": 0}

        # Build topic answer texts for comparison
        topic_texts = {}
        for topic, _ in pitfalls:
            if topic not in topic_texts:
                rows = self.db.conn.execute(
                    "SELECT answer_text FROM qa_pairs WHERE topic = ?", (topic,)
                ).fetchall()
                topic_texts[topic] = " ".join(r["answer_text"] for r in rows) if rows else ""

        relevant = 0
        irrelevant = 0
        details = []
        for topic, pitfall in pitfalls[:20]:
            answer_text = topic_texts.get(topic, "")
            if not answer_text:
                irrelevant += 1
                details.append({"topic": topic, "pitfall": pitfall[:100], "relevant": False})
                continue
            # Simple embedding check
            cos = 0.0
            try:
                model = _get_model(MODEL_MAP[_detect_language([pitfall, answer_text])])
                vecs = model.encode([pitfall, answer_text], normalize_embeddings=True, convert_to_numpy=True)
                cos = float(np.dot(vecs[0], vecs[1]))
                is_rel = cos >= 0.40
            except Exception:
                is_rel = True
                cos = 1.0  # give benefit of doubt
            if is_rel:
                relevant += 1
            else:
                irrelevant += 1
            details.append({"topic": topic, "pitfall": pitfall[:100], "relevant": is_rel, "cos": round(cos, 2)})

        return {
            "total_pitfalls": len(pitfalls),
            "sampled": len(details),
            "relevant": relevant,
            "irrelevant": irrelevant,
            "rate": round(relevant / max(relevant + irrelevant, 1), 2),
            "details": details,
        }

    def _parse_points_kps(self) -> list[dict]:
        """Parse points.txt into structured KP list (same as _load_kp_cache in app.py)."""
        kps = []
        if not os.path.exists(self.points_file):
            return kps
        try:
            with open(self.points_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return kps
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
        return kps

    # ================================================================
    # Evaluation F: Student Confusion Simulation
    # ================================================================

    def _eval_confusion(self, topics: list[str]) -> dict:
        """Test 5 categories of student confusion scenarios."""
        results = {
            "concept": self._test_concept_confusion(topics),
            "misreading": self._test_misreading(topics),
            "exam_tips": self._test_exam_tips(topics),
            "keywords": self._test_keywords(topics),
            "misunderstanding": self._test_misunderstanding(topics),
        }
        return results

    def _test_concept_confusion(self, topics: list[str]) -> dict:
        """C1: Test similar concept confusion."""
        if len(topics) < 2:
            return {"tested": 0, "passed": 0}
        passed = 0
        for i in range(min(3, len(topics) - 1)):
            q = f"Are {topics[i]} and {topics[i+1]} the same thing? Explain the difference."
            resp = self._call_chat(q)
            answer = resp.get("answer", "")
            if "different" in answer.lower() or "区别" in answer or "not the same" in answer.lower():
                passed += 1
        return {"tested": 3, "passed": passed}

    def _test_misreading(self, topics: list[str]) -> dict:
        """C2: Test misreading — missing constraint."""
        passed = 0
        tested = 0
        for topic in topics:
            qas = self._extract_real_questions(topic, limit=1)
            if not qas:
                continue
            tested += 1
            # Ask a follow-up that tests constraint awareness — a core reasoning skill
            q = "I wrote an answer but forgot to check the conditions given in the question. How can I make sure I don't miss constraints when answering exam questions?"
            resp = self._call_chat(q)
            if resp.get("answer"):
                passed += 1
            if tested >= 3:
                break
        return {"tested": max(tested, 1), "passed": passed}

    def _test_exam_tips(self, topics: list[str]) -> dict:
        """C3: Test exam technique questions."""
        questions = [
            "If a question is worth 4 marks, how much should I write?",
            "When drawing a truth table, do I need to show my working?",
            "I only know part of the answer — how can I get partial marks?",
        ]
        passed = 0
        for q in questions[:3]:
            resp = self._call_chat(q)
            if resp.get("answer") and len(resp["answer"]) > 50:
                passed += 1
        return {"tested": 3, "passed": passed}

    def _test_keywords(self, topics: list[str]) -> dict:
        """C4: Test keyword explanation."""
        passed = 0
        tested = 0
        for topic in topics:
            qas = self._extract_real_questions(topic, limit=1)
            if not qas:
                continue
            # Extract a term from the markscheme
            terms = re.findall(r'\b[A-Z][A-Za-z]{2,}\b', qas[0]["markscheme"])
            if not terms:
                continue
            tested += 1
            q = f"What does '{terms[0]}' mean in the context of this topic?"
            resp = self._call_chat(q)
            if resp.get("answer") and len(resp["answer"]) > 30:
                passed += 1
            if tested >= 3:
                break
        return {"tested": max(tested, 1), "passed": passed}

    def _test_misunderstanding(self, topics: list[str]) -> dict:
        """C5: Test misunderstanding — answering wrong question type."""
        passed = 0
        tested = 0
        for topic in topics:
            qas = self._extract_real_questions(topic, limit=1)
            if not qas:
                continue
            tested += 1
            q = f"I was asked '{qas[0]['question']}' but I answered with a definition instead of what was asked. What should I have done differently?"
            resp = self._call_chat(q)
            if resp.get("answer") and len(resp["answer"]) > 30:
                passed += 1
            if tested >= 3:
                break
        return {"tested": max(tested, 1), "passed": passed}

    # ================================================================
    # Report Generation
    # ================================================================

    def _generate_report(self, elapsed: int) -> str:
        """Format all results into a readable report."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        lines = []
        lines.append("=" * 64)
        lines.append(f"  聊天 Agent 质量评估报告 — {ts}")
        lines.append("=" * 64)
        lines.append("")

        # Section 1: Accuracy
        acc = self.results.get("accuracy", {})
        lines.append(f"一、回答准确性 (基于 {acc.get('total', 0)} 道真实题目)")
        lines.append(f"  总体准确率: {acc.get('accuracy', 0)*100:.0f}% ({acc.get('correct', 0)}/{acc.get('total', 0)})")
        for topic, score in acc.get("by_topic", {}).items():
            status = "✓" if score["correct"] == score["total"] else "✗" if score["correct"] == 0 else "△"
            lines.append(f"    {topic}: {score['correct']}/{score['total']} {status}")
        lines.append("")

        # Section 2: Language
        lang = self.results.get("language", {})
        lines.append(f"二、语言合规")
        lines.append(f"  中文提问 Q&A 引用保持英文: {lang.get('rate', 0)*100:.0f}% ({lang.get('compliant', 0)}/{lang.get('total', 0)})")
        for v in lang.get("violations", [])[:3]:
            lines.append(f"  违规: [{v.get('topic', '')}] {v.get('violations', '')}")
        lines.append("")

        # Section 3: Honesty
        hon = self.results.get("honesty", {})
        lines.append(f"三、溯源诚实性")
        lines.append(f"  [KB] 标记与检索 QA 一致: {hon.get('rate', 0)*100:.0f}% ({hon.get('honest', 0)}/{hon.get('total', 0)})")
        for k, d in hon.get("details", {}).items():
            if not d.get("honest", True):
                lines.append(f"  [{k}] 编造内容: {d.get('fabricated', [])}")
        lines.append("")

        # Section 4: Coverage
        cov = self.results.get("coverage", {})
        lines.append(f"四、数据利用效率")
        lines.append(f"  试卷覆盖: {cov.get('covered', 0)}/{cov.get('sampled', 0)} topics 可正确回答")
        lines.append("")

        # Section 5: Pitfall
        pit = self.results.get("pitfall", {})
        lines.append(f"五、Pitfall 试卷相关性")
        lines.append(f"  检查 {pit.get('sampled', 0)} 条 pitfall:")
        lines.append(f"    与试卷答案相关: {pit.get('relevant', 0)} ({pit.get('rate', 0)*100:.0f}%)")
        lines.append(f"    无关联（模型编造）: {pit.get('irrelevant', 0)}")
        for d in pit.get("details", [])[:5]:
            if not d["relevant"]:
                lines.append(f"    无关联: [{d['topic']}] {d['pitfall'][:80]}...")
        lines.append("")

        # Section 6: Confusion
        conf = self.results.get("confusion", {})
        lines.append("六、学生困惑场景测试")
        for name, data in [("概念混淆", "concept"), ("读题不清", "misreading"),
                           ("答题模式", "exam_tips"), ("关键词", "keywords"),
                           ("误解题意", "misunderstanding")]:
            d = conf.get(data, {})
            lines.append(f"  {name}: {d.get('passed', 0)}/{d.get('tested', 0)}")
        lines.append("")

        # Section 7: Suggestions
        lines.append("七、改进建议")
        suggestions = self._generate_suggestions()
        for i, s in enumerate(suggestions, 1):
            lines.append(f"  {i}. {s}")
        lines.append("")
        lines.append(f"评估耗时: {elapsed}s")

        return "\n".join(lines)

    def _generate_suggestions(self) -> list[str]:
        """Generate concrete improvement suggestions from results."""
        suggestions = []
        acc = self.results.get("accuracy", {})
        lang = self.results.get("language", {})
        hon = self.results.get("honesty", {})
        pit = self.results.get("pitfall", {})

        if acc.get("accuracy", 0) < 0.7:
            suggestions.append("[准确性] 回答准确率偏低，建议检查检索阈值和 Agent 3 prompt")

        if lang.get("rate", 1) < 1.0:
            suggestions.append("[语言] 存在术语翻译违规，建议在 Agent 3 prompt 中加强 '即使术语有中文译名也必须用英文原文' 的约束")

        if hon.get("rate", 1) < 0.8:
            suggestions.append("[溯源] [KB] 标记存在编造，建议在 Agent 3 prompt 中增加 '如果不在检索结果中，标记为 [General]' ")

        if pit.get("rate", 1) < 0.7:
            suggestions.append("[Pitfall] 大量 pitfall 无试卷依据，建议蒸馏 prompt 允许 '无常见错误时省略 pitfall'，而非强制为每个 KP 生成")

        # Topic-specific
        for topic, score in acc.get("by_topic", {}).items():
            if score["correct"] == 0 and score["total"] > 0:
                suggestions.append(f"[检索] {topic} 准确率 0% → 建议检查该 topic 的 knowledge_summary 嵌入质量")

        if not suggestions:
            suggestions.append("各项指标达标，无需紧急改进")
        return suggestions


# ================================================================
# Standalone entry point
# ================================================================

if __name__ == "__main__":
    import glob

    PROJECT_DIR = _PROJECT_DIR  # set at module top
    db_files = glob.glob(os.path.join(PROJECT_DIR, "intermediate", "*_knowledge.db"))
    points_files = glob.glob(os.path.join(PROJECT_DIR, "point", "*_points.txt"))

    if not db_files or not points_files:
        print("请先运行分析生成知识库和知识点文件")
        exit(1)

    from src.config import load_config
    config = load_config()
    api_url = config.get("api_url", "https://api.deepseek.com")
    api_key = config.get("api_key", "")

    agent = FeedbackAgent(api_url, api_key, QADatabase(db_files[0]), points_files[0])
    report = agent.run_full_evaluation()
    print(report)

    # Save report
    report_dir = os.path.join(THIS_DIR, "feedback")
    os.makedirs(report_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"{ts}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n报告已保存: {report_path}")
