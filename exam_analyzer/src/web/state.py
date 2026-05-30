"""Global application state — extracted from app.py.

All mutable state accessed by Flask routes lives here with explicit locks.
This prevents the God Object anti-pattern where state, routes, and lifecycle
management were all mixed into app.py.
"""

import threading

# ---- Analysis state ----
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


def get_analysis_state() -> dict:
    """Thread-safe snapshot of analysis state."""
    with _state_lock:
        return dict(analysis_state)


def update_analysis_state(**kwargs):
    """Thread-safe partial update of analysis_state."""
    with _state_lock:
        analysis_state.update(kwargs)


def append_debug_log(message: str):
    """Thread-safe append to debug log."""
    with _state_lock:
        analysis_state["debug_log"].append(message)


def append_timeline(step: str, detail: str = ""):
    """Thread-safe append to execution timeline."""
    from datetime import datetime
    with _state_lock:
        analysis_state["timeline"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "step": step,
            "detail": detail,
        })


# ---- Chat assistant state ----
_chat_retriever = None
_chat_retriever_db_path = None
_chat_retriever_lock = threading.Lock()
_kp_cache = None

# ---- Shutdown ----
shutdown_event = threading.Event()
