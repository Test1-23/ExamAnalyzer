"""Analysis endpoints — config, file management, pipeline lifecycle, timeline (7 routes)."""

import os
import json
import threading

from flask import Blueprint, request, jsonify

from . import state
from src.config import load_config

analysis_bp = Blueprint("analysis", __name__)

CONFIG_FILE = os.path.join(state.THIS_DIR, "config.json")


def _save_config(data: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@analysis_bp.route("/api/config", methods=["GET", "POST"])
def config_endpoint():
    if request.method == "POST":
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON body"}), 400
        _save_config(data)
        return jsonify({"success": True})
    return jsonify(load_config())


@analysis_bp.route("/api/files")
def list_input_files():
    os.makedirs(state.INPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(state.INPUT_DIR) if f.lower().endswith(".pdf")]
    return jsonify(files)


@analysis_bp.route("/api/analyze", methods=["POST"])
def start_analysis():
    with state._state_lock:
        if state.analysis_state["running"]:
            return jsonify({"error": "分析正在进行中"}), 400

        config = load_config()
        if not config.get("api_url") or not config.get("api_key"):
            return jsonify({"error": "请先配置 API URL 和 API Key"}), 400

        os.makedirs(state.INPUT_DIR, exist_ok=True)
        pdfs = [f for f in os.listdir(state.INPUT_DIR) if f.lower().endswith(".pdf")]
        if not pdfs:
            return jsonify({"error": f"input 目录中({state.INPUT_DIR})没有 PDF 文件"}), 400

        state.shutdown_event.clear()
        state.invalidate_chat_retriever()

        state.analysis_state = {
            "running": True, "progress": 0, "status": "初始化...",
            "error": None, "result": None, "debug_log": [], "timeline": [],
        }

    def _run_analysis():
        os.makedirs(state.INPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(state.THIS_DIR, "point"), exist_ok=True)
        try:
            from src.pipeline import run_pipeline
            api_url = config.get("api_url", "")
            api_key = config.get("api_key", "")
            state.debug(f"输出路径: {state.POINTS_FILE}")
            state.debug(f"输入目录: {state.INPUT_DIR}")
            state.log_step("初始化", "分析流程启动")

            def progress_callback(pct, status_text):
                with state._state_lock:
                    state.analysis_state["progress"] = pct
                    state.analysis_state["status"] = status_text

            result = run_pipeline(
                api_url=api_url, api_key=api_key,
                input_dir=state.INPUT_DIR, output_path=state.POINTS_FILE,
                intermediate_dir=os.path.join(state.THIS_DIR, "intermediate"),
                progress_callback=progress_callback,
                debug=state.debug, log_callback=state.log_step,
                shutdown_event=state.shutdown_event,
            )
            if result.healthy:
                state.log_step("分析完成", f"知识点已写入 {state.POINTS_FILE}")
            else:
                failures = result.counters.total_failures()
                state.log_step("分析完成", f"知识点已写入 {state.POINTS_FILE} (⚠️ {failures} 个阶段有失败)")
            with state._state_lock:
                state.analysis_state["progress"] = 100
                state.analysis_state["status"] = "分析完成"
                state.analysis_state["result"] = result.content
                state.analysis_state["error"] = None
            state.warmup_chat_retriever()
        except Exception as e:
            state.log_step("分析失败", str(e))
            state.debug(f"异常: {e}")
            import traceback
            state.debug(traceback.format_exc())
            with state._state_lock:
                state.analysis_state["error"] = str(e)
                state.analysis_state["status"] = "分析失败"
        finally:
            with state._state_lock:
                state.analysis_state["running"] = False

    thread = threading.Thread(target=_run_analysis, daemon=False)
    thread.start()
    state._analysis_thread = thread
    return jsonify({"success": True, "thread_id": thread.ident})


@analysis_bp.route("/api/status")
def get_status():
    return jsonify(state.get_analysis_state())


@analysis_bp.route("/api/points")
def get_points():
    points_path = state.find_points_file()
    if os.path.exists(points_path):
        with open(points_path, "r", encoding="utf-8") as f:
            content = f.read()
        return jsonify({"content": content, "exists": True})
    return jsonify({"content": "", "exists": False})


@analysis_bp.route("/api/input-files", methods=["DELETE"])
def clear_input():
    os.makedirs(state.INPUT_DIR, exist_ok=True)
    for f in os.listdir(state.INPUT_DIR):
        fp = os.path.join(state.INPUT_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)
    return jsonify({"success": True})


@analysis_bp.route("/api/timeline")
def get_timeline():
    with state._state_lock:
        timeline_copy = list(state.analysis_state.get("timeline", []))
    return jsonify(timeline_copy)
