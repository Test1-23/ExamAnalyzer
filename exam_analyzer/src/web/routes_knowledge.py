"""Knowledge & practice endpoints — graph, verbs, difficulty, question generation (5 routes)."""

import os
import json

from flask import Blueprint, request, jsonify

from . import state
from src.config import load_config
from src.logger import get_logger

_log = get_logger()
knowledge_bp = Blueprint("knowledge", __name__)


@knowledge_bp.route("/api/practice/generate", methods=["POST"])
def practice_generate():
    data = request.get_json()
    if not data or "kp_id" not in data:
        return jsonify({"error": "Missing kp_id"}), 400
    config = load_config()
    if not config.get("api_url"):
        return jsonify({"error": "请先配置 API"}), 400
    try:
        import glob as _glob
        from src.question_generator import generate_questions
        db_files = _glob.glob(os.path.join(state.THIS_DIR, "intermediate", "*_knowledge.db"))
        if not db_files:
            return jsonify({"error": "no_data"}), 400
        questions = generate_questions(
            db_files[0], data["kp_id"],
            count=data.get("count", 3),
            difficulty=data.get("difficulty", "intermediate"),
            api_url=config.get("api_url", ""), api_key=config.get("api_key", ""),
        )
        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route("/api/practice/grade", methods=["POST"])
def practice_grade():
    data = request.get_json()
    if not data or "question" not in data or "student_answer" not in data:
        return jsonify({"error": "Missing question or student_answer"}), 400
    config = load_config()
    if not config.get("api_url"):
        return jsonify({"error": "请先配置 API"}), 400
    try:
        from src.deepseek_client import create_client, call_flash
        client = create_client(config["api_url"], config["api_key"])
        sys = "Compare student answer with model answer. Output JSON."
        usr = (
            f"Question: {data['question']}\n"
            f"Student Answer: {data['student_answer']}\n"
            f"Model Answer: {data.get('model_answer', '')}\n\n"
            "List covered and missed points. Give brief feedback.\n"
            'Return JSON: {"covered_points": [...], "missed_points": [...], '
            '"feedback": "brief feedback", "score_pct": 0-100}'
        )
        result = call_flash(client, [{"role": "system", "content": sys}, {"role": "user", "content": usr}], max_retries=1)
        return jsonify(result if isinstance(result, dict) else {"feedback": str(result)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route("/api/knowledge-graph")
def knowledge_graph():
    retriever = state.get_chat_retriever()
    if retriever is None:
        return jsonify({"error": "no_data", "message": "Analyze exam papers first"}), 400

    db = retriever.db
    graph = db.get_dependency_graph()
    difficulties = {d["topic"]: d for d in db.get_topic_difficulty()}

    nodes = []
    added = set()
    for topic, info in graph.items():
        if topic in added:
            continue
        added.add(topic)
        diff = difficulties.get(topic, {})
        nodes.append({
            "name": topic, "qa_count": diff.get("qa_count", 0),
            "difficulty": diff.get("mode_difficulty", ""),
            "prerequisites": info.get("prerequisites", []),
            "dependents": info.get("dependents", []),
        })

    for topic, diff in difficulties.items():
        if topic not in added:
            nodes.append({"name": topic, "qa_count": diff.get("qa_count", 0),
                          "difficulty": diff.get("mode_difficulty", ""),
                          "prerequisites": [], "dependents": []})

    edges = []
    for topic, info in graph.items():
        for pre in info.get("prerequisites", []):
            pre_name = pre["topic"] if isinstance(pre, dict) else pre
            edges.append({"source": pre_name, "target": topic, "type": "prerequisite", "confidence": "medium"})
        for dep in info.get("dependents", []):
            dep_name = dep["topic"] if isinstance(dep, dict) else dep
            edges.append({"source": topic, "target": dep_name, "type": "prerequisite", "confidence": "medium"})

    try:
        dt_rows = db.conn.execute(
            "SELECT topic_id, name, kp_concept, kp_detail, mass, stability, quality, "
            "parent_topic, child_topics, merged_from "
            "FROM dynamic_topics WHERE quality IN ('stable', 'forming')"
        ).fetchall()
        for r in dt_rows:
            nodes.append({"name": r["name"] or r["topic_id"], "qa_count": r["mass"] or 0,
                          "difficulty": "", "prerequisites": [], "dependents": [],
                          "source": "dynamic_topic", "stability": r["stability"]})
            if r["parent_topic"]:
                edges.append({"source": r["parent_topic"], "target": r["name"] or r["topic_id"],
                              "type": "parent_of", "confidence": "medium"})
            child_topics = json.loads(r["child_topics"] or "[]")
            for child in child_topics:
                edges.append({"source": r["name"] or r["topic_id"], "target": child,
                              "type": "split_into", "confidence": "medium"})
            merged = json.loads(r["merged_from"] or "[]")
            for src in merged:
                edges.append({"source": src, "target": r["name"] or r["topic_id"],
                              "type": "merged_from", "confidence": "medium"})
    except Exception:
        _log.debug("Failed to load knowledge graph evolution data", exc_info=True)

    return jsonify({"nodes": nodes, "edges": edges})


@knowledge_bp.route("/api/command-verbs")
def command_verbs():
    retriever = state.get_chat_retriever()
    if retriever is None:
        return jsonify({"error": "no_data", "message": "Analyze exam papers first"}), 400
    patterns = retriever.db.get_verb_patterns()
    return jsonify({"verbs": patterns})


@knowledge_bp.route("/api/topic-difficulty")
def topic_difficulty_api():
    retriever = state.get_chat_retriever()
    if retriever is None:
        return jsonify({"error": "no_data", "message": "Analyze exam papers first"}), 400
    topic = request.args.get("topic", "")
    difficulties = retriever.db.get_topic_difficulty(topic if topic else None)
    return jsonify({"difficulties": difficulties})
