"""Web UI entry point — thin wrapper over create_app()."""

import os
import sys
import signal

from src.web.app_factory import create_app
from src.web import state

app = create_app()


# ---- Static page ----
@app.route("/")
def index():
    from flask import render_template
    return render_template("index.html")


# ---- Main ----
if __name__ == "__main__":
    for d in [state.INPUT_DIR, state.POINTS_DIR,
              os.path.join(state.THIS_DIR, "intermediate"),
              os.path.join(state.THIS_DIR, "TestReport")]:
        os.makedirs(d, exist_ok=True)

    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    def _handle_sigint(signum, frame):
        if not state.shutdown_event.is_set():
            print("\n收到中断信号，正在安全退出... (再次 Ctrl+C 强制退出)")
            state.shutdown_event.set()
            if not state.analysis_state.get("running"):
                print("无活动分析，退出")
                sys.exit(0)
            if state._analysis_thread and state._analysis_thread.is_alive():
                print("等待分析线程完成...")
                state._analysis_thread.join(timeout=30)
            sys.exit(0)
        else:
            print("\n强制退出")
            os._exit(1)
    signal.signal(signal.SIGINT, _handle_sigint)

    print(f"启动 UI 服务器...")
    print(f"  脚本目录: {state.THIS_DIR}")
    print(f"  模板目录: {os.path.join(state.THIS_DIR, 'templates')}")
    print(f"  项目输入: {state.INPUT_DIR}")
    print(f"  输出文件: {state.POINTS_FILE}")
    print(f"  打开浏览器访问: http://127.0.0.1:5000")
    state.warmup_chat_retriever()
    app.run(debug=False, use_reloader=False, port=5000)
