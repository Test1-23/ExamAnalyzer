"""Quality evaluation endpoints — feedback agent lifecycle (2 routes)."""

import os
import threading

from flask import Blueprint, request, jsonify

from . import state
from src.config import load_config

eval_bp = Blueprint("eval", __name__)


@eval_bp.route("/api/evaluate", methods=["POST"])
def start_evaluation():
    with state._eval_lock:
        if state._eval_state["running"]:
            return jsonify({"error": "评估正在进行中"}), 400
        with state._state_lock:
            analysis_running = state.analysis_state.get("running", False)
        if analysis_running:
            return jsonify({"error": "分析正在进行中，请等待完成"}), 400
        retriever = state.get_chat_retriever()
        if retriever is None:
            return jsonify({"error": "知识库尚未建立"}), 400
        config = load_config()
        if not config.get("api_url") or not config.get("api_key"):
            return jsonify({"error": "请先配置 API"}), 400
        state.start_eval_run()

    def _run_eval():
        try:
            import glob as _glob
            db_files = _glob.glob(os.path.join(state.THIS_DIR, "intermediate", "*_knowledge.db"))
            if not db_files:
                state.update_eval_state(error="未找到知识库文件，请先运行分析")
                state.finish_eval_run()
                return
            points_file = state.find_points_file()
            from eval.feedback_agent import FeedbackAgent
            agent = FeedbackAgent(config["api_url"], config["api_key"], retriever.db, points_file)
            state.update_eval_state(progress=30)
            report = agent.run_full_evaluation()
            state.update_eval_state(progress=100, report=report, error=None)
        except Exception as e:
            state.update_eval_state(error=str(e))
        finally:
            state.finish_eval_run()

    t = threading.Thread(target=_run_eval, daemon=True)
    t.start()
    return jsonify({"success": True})


@eval_bp.route("/api/evaluate/status")
def evaluation_status():
    return jsonify(state.get_eval_state())
