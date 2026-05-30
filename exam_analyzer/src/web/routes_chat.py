"""Chat assistant endpoints — multi-agent pipeline, history, student state (7 routes)."""

import json

from flask import Blueprint, request, jsonify

from . import state
from src.config import load_config
from src.deepseek_client import create_client
from src.embedding_cluster import detect_content_lang
from src.chat.agents import agent_query_analyst, agent_answer_generator, agent_critic, agent_suggest
from src.chat.context import build_analysis_context
from src.logger import get_logger

_log = get_logger()
chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/api/chat/status")
def chat_status():
    retriever = state.get_chat_retriever()
    if retriever is None:
        return jsonify({"available": False, "qa_count": 0})
    return jsonify({"available": True, "qa_count": retriever.count()})


@chat_bp.route("/api/chat/history", methods=["GET", "DELETE"])
def chat_history_endpoint():
    session_id = request.args.get("session_id", "default")
    retriever = state.get_chat_retriever()
    if retriever is None:
        return jsonify({"history": []}) if request.method == "GET" else jsonify({"success": False})
    if request.method == "DELETE":
        retriever.clear_chat_history(session_id)
        return jsonify({"success": True})
    history = retriever._db.get_chat_history(session_id)
    return jsonify({"history": history})


@chat_bp.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing question"}), 400
    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    session_id = data.get("session_id", "default")
    lang = detect_content_lang(question)

    retriever = state.get_chat_retriever()
    if retriever is None:
        return jsonify({"error": "知识库尚未建立，请先运行分析"}), 400

    config = load_config()
    if not config.get("api_url") or not config.get("api_key"):
        return jsonify({"error": "请先配置 API"}), 400

    try:
        client = create_client(config["api_url"], config["api_key"])
    except Exception as e:
        return jsonify({"error": f"创建 API 客户端失败: {e}"}), 500

    history = []
    try:
        history = retriever._db.get_chat_history(session_id)
    except Exception:
        _log.debug("Failed to load chat history", exc_info=True)

    analysis = agent_query_analyst(question, lang, client)
    keywords = analysis.get("keywords", [])
    qtype = analysis.get("qtype", "explanation")
    student_verb = analysis.get("verb", "")
    enriched_query = question + " " + " ".join(keywords)

    query_topic = analysis.get("topic", "")
    similar = retriever.search_dual_channel(
        enriched_query, threshold=0.5, min_k=2, max_cap=5, query_topic=query_topic)
    relevant = [qa for qa in similar if qa.get("_score", 0) >= 0.5]

    ctx = ""
    if relevant:
        ctx = "Relevant past Q&A from this subject:\n\n"
        for i, qa in enumerate(relevant[:5], 1):
            ctx += f"Q{i}: {qa['question_text']}\nA: {qa['answer_text']}\n\n"
        ctx += "Use the Q&A above as your PRIMARY reference.\n\n"
    else:
        ctx = "(No directly relevant Q&A in knowledge base. State this and answer from general knowledge.)\n\n"

    ctx_kp = ""
    kp_cache = state.get_kp_cache()
    if kp_cache and relevant:
        seen = set()
        kp_lines = []
        for qa in relevant[:3]:
            topic = qa.get("topic", "")
            if topic in seen:
                continue
            seen.add(topic)
            for kp in kp_cache:
                if kp["topic"] == topic:
                    kp_lines.append(f"[{topic}] {kp['concept']}")
                    if kp.get("pitfall"):
                        kp_lines.append(f"   Pitfall: {kp['pitfall']}")
                    if kp.get("scoring"):
                        kp_lines.append(f"   Scoring: {kp['scoring']}")
        if kp_lines:
            ctx_kp = "Relevant knowledge points from the curriculum:\n" + "\n".join(kp_lines) + "\n\n"

    relevant_topics = list(set(qa.get("topic", "") for qa in relevant[:3] if qa.get("topic")))
    analysis_ctx = build_analysis_context(retriever._db, relevant_topics, student_verb, session_id)
    if analysis_ctx:
        has_diff = "difficulty" in analysis_ctx.lower()
        has_verb = "answer style" in analysis_ctx.lower()
        has_prereq = "prerequisite" in analysis_ctx.lower()
        has_conf = "confusion" in analysis_ctx.lower()
        print(f"[Chat] Analysis context: {len(analysis_ctx)} chars "
              f"(diff={has_diff}, verb={has_verb}, prereq={has_prereq}, confusion={has_conf})")
    ctx = analysis_ctx + ctx

    answer_raw = agent_answer_generator(question, qtype, lang, ctx, ctx_kp, history, client)
    answer_text = answer_raw.get("answer", "") if isinstance(answer_raw, dict) else str(answer_raw)
    quiz = answer_raw.get("quiz") if isinstance(answer_raw, dict) else None
    path_hint = answer_raw.get("path_hint") if isinstance(answer_raw, dict) else None

    for _ in range(2):
        review = agent_critic(question, answer_text, relevant, lang, client)
        if review.get("pass", True):
            break
        if lang == 'en':
            ctx += f"\nPrevious answer issues: {review.get('feedback', '')}\nPlease fix these issues.\n"
        else:
            ctx += f"\n上次回答问题: {review.get('feedback', '')}\n请修正这些问题。\n"
        answer_raw2 = agent_answer_generator(question, qtype, lang, ctx, ctx_kp, history, client)
        answer_text = answer_raw2.get("answer", "") if isinstance(answer_raw2, dict) else str(answer_raw2)

    suggestions = agent_suggest(question, answer_text, relevant, lang, client,
                                db=retriever._db, session_id=session_id)

    try:
        retriever._db.save_chat_message(session_id, "user", question, "")
        retriever._db.save_chat_message(session_id, "assistant", answer_text,
                                        json.dumps([{"topic": qa.get("topic", ""), "question": qa.get("question_text", "")[:120]} for qa in relevant[:3]]))
        for qa in relevant[:2]:
            topic = qa.get("topic", "")
            if topic:
                retriever._db.save_student_memory(session_id, "question", topic, question[:500])
                retriever._db.upsert_knowledge_state(session_id, topic, "learning")
                try:
                    rows = retriever._db.conn.execute(
                        "SELECT kp_id FROM qa_kp_membership WHERE qa_id = ?", (qa["id"],)
                    ).fetchall()
                    for r in rows:
                        retriever._db.record_trajectory(session_id, r["kp_id"], "new", "learning", "chat_question")
                except Exception:
                    _log.debug("Failed to record trajectory", exc_info=True)
    except Exception:
        _log.debug("Failed to load student knowledge state", exc_info=True)

    sources = []
    for qa in relevant[:5]:
        sources.append({
            "topic": qa.get("topic", ""),
            "question": qa.get("question_text", "")[:200],
            "score": round(qa.get("_score", 0), 2),
        })

    result = {"answer": answer_text, "sources": sources, "suggestions": suggestions}
    if quiz:
        result["quiz"] = quiz
    if path_hint:
        result["path_hint"] = path_hint
    return jsonify(result)


@chat_bp.route("/api/chat/exam-stats")
def exam_stats():
    topic = request.args.get("topic", "")
    if not topic:
        return jsonify({"error": "Missing topic"}), 400
    retriever = state.get_chat_retriever()
    if retriever is None:
        return jsonify({"error": "知识库未就绪"}), 400
    stats = retriever._db.get_exam_stats(topic)
    return jsonify({"topic": topic, "stats": stats})


@chat_bp.route("/api/chat/student-state")
def student_state():
    sid = request.args.get("student_id", "default")
    retriever = state.get_chat_retriever()
    if retriever is None:
        return jsonify({"state": {}})
    return jsonify({"state": retriever._db.get_knowledge_state(sid)})


@chat_bp.route("/api/chat/student-confusions")
def student_confusions():
    sid = request.args.get("student_id", "default")
    retriever = state.get_chat_retriever()
    if retriever is None:
        return jsonify({"confusions": []})
    confusions = retriever._db.get_student_confusions(sid)
    topic = request.args.get("topic", "")
    if topic:
        confusions = [c for c in confusions if c["topic"] == topic]
    return jsonify({"confusions": confusions})


@chat_bp.route("/api/chat/topic-questions")
def topic_questions():
    topic = request.args.get("topic", "")
    level = request.args.get("level", "")
    if not topic:
        return jsonify({"error": "Missing topic"}), 400
    retriever = state.get_chat_retriever()
    if retriever is None:
        return jsonify({"error": "知识库未就绪"}), 400
    qs = []
    rows = retriever._db.qa.get_by_topic(topic, difficulty=level, order_by="is_representative DESC, success_count DESC")
    for r in rows:
        qs.append({
            "question_number": r["question_number"],
            "question_text": r["question_text"][:200],
            "paper": r["paper"],
            "is_representative": bool(r["is_representative"]),
            "is_cross_topic": bool(r["is_cross_topic"]),
            "difficulty": r["difficulty_estimate"],
        })
    return jsonify({"topic": topic, "questions": qs})
