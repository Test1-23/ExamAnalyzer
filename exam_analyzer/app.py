import os
import re
import signal
import sys
import threading
import json
import time
from datetime import datetime
from flask import Flask, request, jsonify, render_template

from src.logger import get_logger
from src.deepseek_client import create_client, call_flash
from src.embedding_cluster import detect_content_lang

_log = get_logger()

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,
            template_folder=os.path.join(THIS_DIR, "templates"),
            static_folder=os.path.join(THIS_DIR, "static") if os.path.isdir(os.path.join(THIS_DIR, "static")) else None)

@app.before_request
def _log_request():
    request._start_time = time.time()

@app.after_request
def _log_response(response):
    elapsed = int((time.time() - getattr(request, '_start_time', time.time())) * 1000)
    _log.debug(f"HTTP {request.method} {request.path} → {response.status_code} ({elapsed}ms)")
    return response

# ---- Paths ----
INPUT_DIR = os.path.join(THIS_DIR, "input")
POINTS_DIR = os.path.join(THIS_DIR, "point")
POINTS_FILE = os.path.join(POINTS_DIR, "points.txt")


def _find_points_file() -> str:
    """Find the most recent points output file (subject-specific naming)."""
    if os.path.isdir(POINTS_DIR):
        candidates = []
        for f in os.listdir(POINTS_DIR):
            if f.endswith("_points.txt") and f != "points.txt":
                candidates.append(os.path.join(POINTS_DIR, f))
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
    return POINTS_FILE
CONFIG_FILE = os.path.join(THIS_DIR, "config.json")

# ---- Global state ----
analysis_state = {
    "running": False,
    "progress": 0,
    "status": "",
    "error": None,
    "result": None,
    "debug_log": [],
    "timeline": [],
}
_state_lock = threading.Lock()
_analysis_thread = None  # set by start_analysis

# ---- Chat assistant (multi-agent pipeline) ----
_chat_retriever = None
_chat_retriever_db_path = None
_chat_retriever_lock = threading.Lock()
_kp_cache = None


def _warmup_chat_retriever():
    """Pre-load embedding model in background so first chat request is fast."""
    def _warmup():
        try:
            _get_chat_retriever()  # model loads inside rebuild()
        except Exception:
            pass
    t = threading.Thread(target=_warmup, daemon=True)
    t.start()


def _get_chat_retriever():
    global _chat_retriever, _chat_retriever_db_path, _kp_cache
    import glob
    db_files = glob.glob(os.path.join(THIS_DIR, "intermediate", "*_knowledge.db"))
    if not db_files:
        return None
    db_path = db_files[0]
    if _chat_retriever is not None and db_path == _chat_retriever_db_path:
        return _chat_retriever
    with _chat_retriever_lock:
        if _chat_retriever is not None and db_path == _chat_retriever_db_path:
            return _chat_retriever
        from src.knowledge_base import QADatabase, QARetriever
        db = QADatabase(db_path)
        retriever = QARetriever(db)
        retriever.rebuild()
        _chat_retriever = retriever
        _chat_retriever_db_path = db_path
        _kp_cache = _load_kp_cache(db)
        source = "points.txt" if _kp_cache and any(k.get("pitfall") for k in _kp_cache[:3]) else "DB"
        print(f"[Chat] KP cache loaded: {len(_kp_cache)} entries (source: {source})")
    return retriever


def _load_kp_from_db(db) -> list[dict]:
    """Read KP data from DB: Dynamic_Topics (priority) + qa_pairs (fallback)."""
    kps = []
    try:
        # Phase 4: Dynamic_Topics first (behavior-validated KPs)
        dt_rows = db.conn.execute(
            "SELECT name, kp_concept, kp_detail, mass "
            "FROM dynamic_topics WHERE quality='stable' AND kp_concept != '' "
            "ORDER BY mass DESC"
        ).fetchall()
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
        pass

    # Fallback: qa_pairs representative QAs (legacy)
    try:
        rows = db.conn.execute("""
            SELECT topic, knowledge_summary, question_text, answer_text,
                   is_representative, difficulty_estimate
            FROM qa_pairs
            WHERE topic != '' AND topic != '(uncategorized)'
            ORDER BY is_representative DESC, success_count DESC
        """).fetchall()
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
        pass
    return kps


def _load_kp_cache(db=None) -> list[dict]:
    """Load KP cache: DB first (structured), points.txt fallback (parsed)."""
    # Primary: DB read
    if db:
        kps = _load_kp_from_db(db)
        if kps:
            return kps

    # Fallback: parse points.txt
    kps = []
    points_file = _find_points_file()
    if not os.path.exists(points_file):
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
        pass
    return kps


# ---- Agent 1: Query Analyst ----

def _build_analysis_context(db, topics: list[str], student_verb: str, student_id: str) -> str:
    """Build enrichment context from offline analysis results.
    Reads verb_patterns, topic_difficulty, topic_dependencies, student state.
    Returns empty string if no data available (graceful degradation)."""
    if not topics:
        return ""

    parts = []

    # 1. Topic difficulty — adjust answer depth
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
        pass

    # 2. Command verb pattern — adapt answer structure
    if student_verb:
        try:
            patterns = db.get_verb_patterns()
            for p in patterns:
                if p["verb"] == student_verb and p.get("pattern_summary"):
                    parts.append(f"Answer style for '{student_verb}' questions: {p['pattern_summary']}")
                    break
        except Exception:
            pass

    # 3. Prerequisites — warn if student hasn't mastered them
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
        pass

    # 4. Student confusion history — personalize
    try:
        confusions = db.get_student_confusions(student_id)
        confused_topics = {c["topic"] for c in confusions[:10] if not c.get("resolved")}
        relevant_confusions = confused_topics & set(topics)
        if relevant_confusions:
            parts.append(f"Student has shown confusion on: {', '.join(relevant_confusions)}. Address these carefully.")
    except Exception:
        pass

    if parts:
        return "[Analysis Context]\n" + "\n".join(parts) + "\n\n"
    return ""


def _agent_query_analyst(question: str, lang: str, client) -> dict:
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
        sys = "分析学生问题。Output JSON."
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
        return {"keywords": [], "qtype": "explanation", "verb": ""}


# ---- Agent 3: Answer Generator (type-adaptive) ----

def _agent_answer_generator(question: str, qtype: str, lang: str, ctx: str, ctx_kp: str, history: list, client) -> dict:
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

    # Type-adaptive instructions
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
        return {"answer": ""}


# ---- Agent 4: Critic ----

def _agent_critic(question: str, answer: str, similar: list, lang: str, client) -> dict:
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
        sys = "审查此教学回答的质量。Output JSON."
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
        return {"pass": False, "feedback": "Review unavailable (API error), retrying"}


# ---- Agent 5: Follow-up Suggester ----

def _agent_suggest(question: str, answer: str, similar: list, lang: str, client,
                   db=None, session_id: str = "") -> list[str]:
    """Generate follow-up question suggestions, enriched with dependency data."""
    topics = list(set(qa.get("topic", "") for qa in similar[:5] if qa.get("topic")))
    topic_str = ", ".join(topics) if topics else "various topics"

    # Build prerequisite hints from dependency graph
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
            pass

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
        sys = "建议2-3个学生可能追问的问题。Output JSON."
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
        return []


def _debug(msg: str):
    """Log debug message with timestamp to both console and analysis state."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    with _state_lock:
        analysis_state.setdefault("debug_log", []).append(f"[{ts}] {msg}")


def _log_step(step: str, detail: str = ""):
    """Log a timestamped timeline entry."""
    now = datetime.now().strftime("%H:%M:%S")
    entry = {"time": now, "step": step, "detail": detail}
    print(f"[{now}] {step}" + (f" — {detail}" if detail else ""))
    with _state_lock:
        analysis_state.setdefault("timeline", []).append(entry)


# Graceful shutdown: set by Ctrl+C handler, checked in _run_analysis
shutdown_event = threading.Event()


def _run_analysis(config: dict):
    """Run the new pipeline in a background thread."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(THIS_DIR, "point"), exist_ok=True)

    try:
        from src.pipeline import run_pipeline

        api_url = config.get("api_url", "")
        api_key = config.get("api_key", "")

        _debug(f"输出路径: {POINTS_FILE}")
        _debug(f"输入目录: {INPUT_DIR}")
        _log_step("初始化", "分析流程启动")

        def progress_callback(pct: int, status: str):
            with _state_lock:
                analysis_state["progress"] = pct
                analysis_state["status"] = status

        def log_callback(step: str, detail: str = ""):
            _log_step(step, detail)

        result_content = run_pipeline(
            api_url=api_url,
            api_key=api_key,
            input_dir=INPUT_DIR,
            output_path=POINTS_FILE,
            intermediate_dir=os.path.join(THIS_DIR, "intermediate"),
            progress_callback=progress_callback,
            debug_callback=_debug,
            log_callback=log_callback,
            shutdown_event=shutdown_event,
        )

        _log_step("分析完成",
                  f"知识点已写入 {POINTS_FILE}")
        with _state_lock:
            analysis_state["progress"] = 100
            analysis_state["status"] = "分析完成"
            analysis_state["result"] = result_content
            analysis_state["error"] = None
        # Pre-warm chat retriever so first question is instant
        _warmup_chat_retriever()

    except Exception as e:
        _log_step("分析失败", str(e))
        _debug(f"异常: {e}")
        import traceback
        _debug(traceback.format_exc())
        with _state_lock:
            analysis_state["error"] = str(e)
            analysis_state["status"] = "分析失败"
    finally:
        with _state_lock:
            analysis_state["running"] = False


# ---- API Routes ----


def _load_config() -> dict:
    config = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            pass
    config["api_url"] = os.environ.get("DEEPSEEK_API_URL", config.get("api_url", ""))
    config["api_key"] = os.environ.get("DEEPSEEK_API_KEY", config.get("api_key", ""))
    return config


def _save_config(data: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@app.route("/api/config", methods=["GET", "POST"])
def config_endpoint():
    if request.method == "POST":
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON body"}), 400
        _save_config(data)
        return jsonify({"success": True})
    return jsonify(_load_config())


@app.route("/api/files")
def list_input_files():
    """List PDF files in the project input directory."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    return jsonify(files)


@app.route("/api/analyze", methods=["POST"])
def start_analysis():
    """Start analysis in background thread."""
    global analysis_state, _analysis_thread

    with _state_lock:
        if analysis_state["running"]:
            return jsonify({"error": "分析正在进行中"}), 400

        config = _load_config()
        if not config.get("api_url") or not config.get("api_key"):
            return jsonify({"error": "请先配置 API URL 和 API Key"}), 400

        # Check input has files
        os.makedirs(INPUT_DIR, exist_ok=True)
        pdfs = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
        if not pdfs:
            return jsonify({"error": f"input 目录中({INPUT_DIR})没有 PDF 文件"}), 400

        shutdown_event.clear()

        # Invalidate chat retriever cache — new analysis will add QAs
        global _chat_retriever, _chat_retriever_db_path
        _chat_retriever = None
        _chat_retriever_db_path = None

        analysis_state = {
            "running": True,
            "progress": 0,
            "status": "初始化...",
            "error": None,
            "result": None,
            "debug_log": [],
            "timeline": [],
        }

    thread = threading.Thread(target=_run_analysis, args=(config,), daemon=False)
    thread.start()
    _analysis_thread = thread
    return jsonify({"success": True, "thread_id": thread.ident})


@app.route("/api/status")
def get_status():
    return jsonify(analysis_state)


@app.route("/api/points")
def get_points():
    """Read points content from the actual output file."""
    points_path = _find_points_file()
    if os.path.exists(points_path):
        with open(points_path, "r", encoding="utf-8") as f:
            content = f.read()
        return jsonify({"content": content, "exists": True})
    return jsonify({"content": "", "exists": False})


@app.route("/api/input-files", methods=["DELETE"])
def clear_input():
    """Clear all files from input directory."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    for f in os.listdir(INPUT_DIR):
        fp = os.path.join(INPUT_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)
    return jsonify({"success": True})


@app.route("/api/chat/status")
def chat_status():
    """Check if chat is available (knowledge base exists)."""
    retriever = _get_chat_retriever()
    if retriever is None:
        return jsonify({"available": False, "qa_count": 0})
    return jsonify({"available": True, "qa_count": retriever.count()})


@app.route("/api/chat/history", methods=["GET", "DELETE"])
def chat_history_endpoint():
    """Get or clear chat history for a session."""
    session_id = request.args.get("session_id", "default")
    retriever = _get_chat_retriever()
    if retriever is None:
        return jsonify({"history": []}) if request.method == "GET" else jsonify({"success": False})
    if request.method == "DELETE":
        retriever.clear_chat_history(session_id)
        return jsonify({"success": True})
    history = retriever._db.get_chat_history(session_id)
    return jsonify({"history": history})


@app.route("/api/chat", methods=["POST"])
def chat():
    """Multi-agent conversational knowledge assistant."""
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing question"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    session_id = data.get("session_id", "default")
    lang = detect_content_lang(question)

    retriever = _get_chat_retriever()
    if retriever is None:
        return jsonify({"error": "知识库尚未建立，请先运行分析"}), 400

    config = _load_config()
    if not config.get("api_url") or not config.get("api_key"):
        return jsonify({"error": "请先配置 API"}), 400

    try:
        client = create_client(config["api_url"], config["api_key"])
    except Exception as e:
        return jsonify({"error": f"创建 API 客户端失败: {e}"}), 500

    # Load history from DB
    history = []
    try:
        history = retriever._db.get_chat_history(session_id)
    except Exception:
        pass

    # Agent 1: Query Analyst — rephrase + classify + extract verb
    analysis = _agent_query_analyst(question, lang, client)
    keywords = analysis.get("keywords", [])
    qtype = analysis.get("qtype", "explanation")
    student_verb = analysis.get("verb", "")
    # Build enriched query: original + extracted English keywords
    enriched_query = question + " " + " ".join(keywords)

    # Hybrid retrieval: embedding search with keyword-boosted query
    similar = retriever.search(enriched_query, threshold=0.5, min_k=2, max_cap=5)
    relevant = [qa for qa in similar if qa.get("_score", 0) >= 0.5]

    # Build QA context
    ctx = ""
    if relevant:
        ctx = "Relevant past Q&A from this subject:\n\n"
        for i, qa in enumerate(relevant[:5], 1):
            ctx += f"Q{i}: {qa['question_text']}\nA: {qa['answer_text']}\n\n"
        ctx += "Use the Q&A above as your PRIMARY reference.\n\n"
    else:
        ctx = "(No directly relevant Q&A in knowledge base. State this and answer from general knowledge.)\n\n"

    # KP structured context
    ctx_kp = ""
    if _kp_cache and relevant:
        seen = set()
        kp_lines = []
        for qa in relevant[:3]:
            topic = qa.get("topic", "")
            if topic in seen:
                continue
            seen.add(topic)
            for kp in _kp_cache:
                if kp["topic"] == topic:
                    kp_lines.append(f"[{topic}] {kp['concept']}")
                    if kp.get("pitfall"):
                        kp_lines.append(f"   Pitfall: {kp['pitfall']}")
                    if kp.get("scoring"):
                        kp_lines.append(f"   Scoring: {kp['scoring']}")
        if kp_lines:
            ctx_kp = "Relevant knowledge points from the curriculum:\n" + "\n".join(kp_lines) + "\n\n"

    # Enrichment from offline analysis: difficulty, verb patterns, prerequisites, student state
    relevant_topics = list(set(qa.get("topic", "") for qa in relevant[:3] if qa.get("topic")))
    analysis_ctx = _build_analysis_context(retriever._db, relevant_topics, student_verb, session_id)
    if analysis_ctx:
        has_diff = "difficulty" in analysis_ctx.lower()
        has_verb = "answer style" in analysis_ctx.lower()
        has_prereq = "prerequisite" in analysis_ctx.lower()
        has_conf = "confusion" in analysis_ctx.lower()
        print(f"[Chat] Analysis context: {len(analysis_ctx)} chars "
              f"(diff={has_diff}, verb={has_verb}, prereq={has_prereq}, confusion={has_conf})")
    ctx = analysis_ctx + ctx  # prepend: analysis context goes before Q&A context

    # Agent 3: Answer Generator (type-adaptive)
    answer_raw = _agent_answer_generator(question, qtype, lang, ctx, ctx_kp, history, client)
    answer_text = answer_raw.get("answer", "") if isinstance(answer_raw, dict) else str(answer_raw)
    quiz = answer_raw.get("quiz") if isinstance(answer_raw, dict) else None
    path_hint = answer_raw.get("path_hint") if isinstance(answer_raw, dict) else None

    # Agent 4: Critic — review and optionally regenerate
    for _ in range(2):
        review = _agent_critic(question, answer_text, relevant, lang, client)
        if review.get("pass", True):
            break
        if lang == 'en':
            ctx += f"\nPrevious answer issues: {review.get('feedback', '')}\nPlease fix these issues.\n"
        else:
            ctx += f"\n上次回答问题: {review.get('feedback', '')}\n请修正这些问题。\n"
        answer_raw2 = _agent_answer_generator(question, qtype, lang, ctx, ctx_kp, history, client)
        answer_text = answer_raw2.get("answer", "") if isinstance(answer_raw2, dict) else str(answer_raw2)

    # Agent 5: Follow-up Suggester (enriched with prerequisite data)
    suggestions = _agent_suggest(question, answer_text, relevant, lang, client,
                                 db=retriever._db, session_id=session_id)

    # Save to history + record student trajectory
    try:
        retriever._db.save_chat_message(session_id, "user", question, "")
        retriever._db.save_chat_message(session_id, "assistant", answer_text,
                                        json.dumps([{"topic": qa.get("topic", ""), "question": qa.get("question_text", "")[:120]} for qa in relevant[:3]]))
        # Record student memory + trajectory per KP
        for qa in relevant[:2]:
            topic = qa.get("topic", "")
            if topic:
                retriever._db.save_student_memory(session_id, "question", topic, question[:500])
                retriever._db.upsert_knowledge_state(session_id, topic, "learning")
                # Record trajectory if we have KP mapping
                try:
                    rows = retriever._db.conn.execute(
                        "SELECT kp_id FROM qa_kp_membership WHERE qa_id = ?", (qa["id"],)
                    ).fetchall()
                    for r in rows:
                        retriever._db.record_trajectory(session_id, r["kp_id"], "new", "learning", "chat_question")
                except Exception:
                    pass
    except Exception:
        pass

    # Build sources for frontend
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


_eval_state = {"running": False, "progress": 0, "report": "", "error": None}
_eval_lock = threading.Lock()


@app.route("/api/evaluate", methods=["POST"])
def start_evaluation():
    """Start feedback agent evaluation in background thread."""
    global _eval_state
    with _eval_lock:
        if _eval_state["running"]:
            return jsonify({"error": "评估正在进行中"}), 400
        # Check analysis state under its own lock to avoid race
        with _state_lock:
            analysis_running = analysis_state.get("running", False)
        if analysis_running:
            return jsonify({"error": "分析正在进行中，请等待完成"}), 400
        # Double-check retriever still exists
        retriever = _get_chat_retriever()
        if retriever is None:
            return jsonify({"error": "知识库尚未建立"}), 400
        _eval_state = {"running": True, "progress": 0, "report": "", "error": None}
    config = _load_config()
    if not config.get("api_url") or not config.get("api_key"):
        return jsonify({"error": "请先配置 API"}), 400

    def _run_eval():
        global _eval_state
        try:
            import glob
            db_files = glob.glob(os.path.join(THIS_DIR, "intermediate", "*_knowledge.db"))
            if not db_files:
                _eval_state["error"] = "未找到知识库文件，请先运行分析"
                _eval_state["running"] = False
                return
            points_file = _find_points_file()
            from eval.feedback_agent import FeedbackAgent
            agent = FeedbackAgent(config["api_url"], config["api_key"], db_files[0], points_file)
            _eval_state["progress"] = 30
            report = agent.run_full_evaluation()
            _eval_state["progress"] = 100
            _eval_state["report"] = report
            _eval_state["error"] = None
        except Exception as e:
            _eval_state["error"] = str(e)
        finally:
            _eval_state["running"] = False

    t = threading.Thread(target=_run_eval, daemon=True)
    t.start()
    return jsonify({"success": True})


@app.route("/api/evaluate/status")
def evaluation_status():
    """Get evaluation progress and report."""
    return jsonify(_eval_state)


@app.route("/api/timeline")
def get_timeline():
    """Return the timeline log from the last analysis run."""
    return jsonify(analysis_state.get("timeline", []))


@app.route("/api/chat/exam-stats")
def exam_stats():
    """Return exam session distribution for a topic."""
    topic = request.args.get("topic", "")
    if not topic:
        return jsonify({"error": "Missing topic"}), 400
    retriever = _get_chat_retriever()
    if retriever is None:
        return jsonify({"error": "知识库未就绪"}), 400
    stats = retriever._db.get_exam_stats(topic)
    return jsonify({"topic": topic, "stats": stats})


@app.route("/api/chat/student-state")
def student_state():
    """Return knowledge state for a student."""
    sid = request.args.get("student_id", "default")
    retriever = _get_chat_retriever()
    if retriever is None:
        return jsonify({"state": {}})
    return jsonify({"state": retriever._db.get_knowledge_state(sid)})


@app.route("/api/chat/student-confusions")
def student_confusions():
    """Return confusion events for a student, optionally filtered by topic."""
    sid = request.args.get("student_id", "default")
    retriever = _get_chat_retriever()
    if retriever is None:
        return jsonify({"confusions": []})
    confusions = retriever._db.get_student_confusions(sid)
    topic = request.args.get("topic", "")
    if topic:
        confusions = [c for c in confusions if c["topic"] == topic]
    return jsonify({"confusions": confusions})


@app.route("/api/chat/topic-questions")
def topic_questions():
    """Return QAs for a topic, optionally filtered by difficulty."""
    topic = request.args.get("topic", "")
    level = request.args.get("level", "")
    if not topic:
        return jsonify({"error": "Missing topic"}), 400
    retriever = _get_chat_retriever()
    if retriever is None:
        return jsonify({"error": "知识库未就绪"}), 400
    qs = []
    extra = "AND difficulty_estimate = ?" if level else ""
    params = (topic,) + ((level,) if level else ())
    rows = retriever._db.conn.execute(
        f"""SELECT question_number, question_text, paper, is_representative, is_cross_topic, difficulty_estimate
            FROM qa_pairs WHERE topic = ? {extra}
            ORDER BY is_representative DESC, success_count DESC""",
        params,
    ).fetchall()
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


@app.route("/api/practice/generate", methods=["POST"])
def practice_generate():
    """Generate practice questions for a KP."""
    data = request.get_json()
    if not data or "kp_id" not in data:
        return jsonify({"error": "Missing kp_id"}), 400
    config = _load_config()
    if not config.get("api_url"):
        return jsonify({"error": "请先配置 API"}), 400
    try:
        import glob
        from src.question_generator import generate_questions
        db_files = glob.glob(os.path.join(THIS_DIR, "intermediate", "*_knowledge.db"))
        if not db_files:
            return jsonify({"error": "no_data"}), 400
        questions = generate_questions(
            db_files[0], data["kp_id"],
            count=data.get("count", 3),
            difficulty=data.get("difficulty", "intermediate"),
            api_url=config["api_url"], api_key=config.get("api_key", ""),
        )
        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/practice/grade", methods=["POST"])
def practice_grade():
    """Grade a student's answer to a practice question."""
    data = request.get_json()
    if not data or "question" not in data or "student_answer" not in data:
        return jsonify({"error": "Missing question or student_answer"}), 400
    config = _load_config()
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


@app.route("/api/knowledge-graph")
def knowledge_graph():
    """Return dependency graph: nodes (topics) + edges (prerequisites/corequisites)."""
    retriever = _get_chat_retriever()
    if retriever is None:
        return jsonify({"error": "no_data", "message": "Analyze exam papers first"}), 400

    db = retriever._db
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
            "name": topic,
            "qa_count": diff.get("qa_count", 0),
            "difficulty": diff.get("mode_difficulty", ""),
            "prerequisites": info.get("prerequisites", []),
            "dependents": info.get("dependents", []),
        })

    # Add topics with difficulty but no dependencies
    for topic, diff in difficulties.items():
        if topic not in added:
            nodes.append({
                "name": topic,
                "qa_count": diff.get("qa_count", 0),
                "difficulty": diff.get("mode_difficulty", ""),
                "prerequisites": [],
                "dependents": [],
            })

    # Derive edges from already-loaded graph (avoid duplicate query)
    for topic, info in graph.items():
        for pre in info.get("prerequisites", []):
            edges.append({"source": pre, "target": topic, "type": "prerequisite", "confidence": "medium"})
        for dep in info.get("dependents", []):
            edges.append({"source": topic, "target": dep, "type": "prerequisite", "confidence": "medium"})

    # Phase 4: Include Dynamic_Topics nodes and derived edges
    try:
        dt_rows = db.conn.execute(
            "SELECT topic_id, name, kp_concept, kp_detail, mass, stability, quality, "
            "parent_topic, child_topics, merged_from "
            "FROM dynamic_topics WHERE quality IN ('stable', 'forming')"
        ).fetchall()
        for r in dt_rows:
            nodes.append({
                "name": r["name"] or r["topic_id"],
                "qa_count": r["mass"] or 0,
                "difficulty": "",
                "prerequisites": [],
                "dependents": [],
                "source": "dynamic_topic",
                "stability": r["stability"],
            })
            # Derive edges from parent/child relationships
            if r["parent_topic"]:
                edges.append({
                    "source": r["parent_topic"], "target": r["name"] or r["topic_id"],
                    "type": "parent_of", "confidence": "medium",
                })
            child_topics = json.loads(r["child_topics"] or "[]")
            for child in child_topics:
                edges.append({
                    "source": r["name"] or r["topic_id"], "target": child,
                    "type": "split_into", "confidence": "medium",
                })
            merged = json.loads(r["merged_from"] or "[]")
            for src in merged:
                edges.append({
                    "source": src, "target": r["name"] or r["topic_id"],
                    "type": "merged_from", "confidence": "medium",
                })
    except Exception:
        pass

    return jsonify({"nodes": nodes, "edges": edges})


@app.route("/api/command-verbs")
def command_verbs():
    """Return command verb patterns for chat assistant reference."""
    retriever = _get_chat_retriever()
    if retriever is None:
        return jsonify({"error": "no_data", "message": "Analyze exam papers first"}), 400

    patterns = retriever._db.get_verb_patterns()
    return jsonify({"verbs": patterns})


@app.route("/api/topic-difficulty")
def topic_difficulty_api():
    """Return topic difficulty assessments."""
    retriever = _get_chat_retriever()
    if retriever is None:
        return jsonify({"error": "no_data", "message": "Analyze exam papers first"}), 400

    topic = request.args.get("topic", "")
    difficulties = retriever._db.get_topic_difficulty(topic if topic else None)
    return jsonify({"difficulties": difficulties})


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    # Ensure all required directories exist
    for d in [INPUT_DIR, POINTS_DIR, os.path.join(THIS_DIR, "intermediate"),
              os.path.join(THIS_DIR, "logs"), os.path.join(THIS_DIR, "TestReport")]:
        os.makedirs(d, exist_ok=True)

    # Silence Werkzeug access logs (keep error logs visible)
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    # Graceful shutdown on Ctrl+C: signal the analysis thread to stop at the next
    # safe point. System-level crashes (power loss, OOM) are not catchable —
    # persistence mechanisms (WAL, topic_links DB, processed.json) handle those.
    def _handle_sigint(signum, frame):
        if not shutdown_event.is_set():
            print("\n收到中断信号，正在安全退出... (再次 Ctrl+C 强制退出)")
            shutdown_event.set()
            # If no analysis is running, exit immediately
            if not analysis_state.get("running"):
                print("无活动分析，退出")
                sys.exit(0)
            # Analysis is running — wait for the thread (non-blocking check)
            if _analysis_thread and _analysis_thread.is_alive():
                print("等待分析线程完成...")
                _analysis_thread.join(timeout=30)
            sys.exit(0)
        else:
            print("\n强制退出")
            os._exit(1)  # bypass Werkzeug debugger which catches SystemExit
    signal.signal(signal.SIGINT, _handle_sigint)

    print(f"启动 UI 服务器...")
    print(f"  脚本目录: {THIS_DIR}")
    print(f"  模板目录: {os.path.join(THIS_DIR, 'templates')}")
    print(f"  项目输入: {INPUT_DIR}")
    print(f"  输出文件: {POINTS_FILE}")
    print(f"  打开浏览器访问: http://127.0.0.1:5000")
    # Pre-warm embedding model if knowledge base already exists
    _warmup_chat_retriever()
    # use_reloader=False: prevents Werkzeug from forking a child process that traps
    # signals, ensuring Ctrl+C reaches the main thread and terminates the process.
    app.run(debug=False, use_reloader=False, port=5000)
