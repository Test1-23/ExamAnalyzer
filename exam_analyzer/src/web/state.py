"""Shared global state for the web layer — thread-safe accessors.

All module-level globals that were previously in app.py live here so that
Blueprint route handlers can import them without circular dependencies.
"""

import os
import threading
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/web/ → src/ → exam_analyzer/
INPUT_DIR = os.path.join(THIS_DIR, "input")
POINTS_DIR = os.path.join(THIS_DIR, "point")
POINTS_FILE = os.path.join(POINTS_DIR, "points.txt")


def find_points_file() -> str:
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


# ---------------------------------------------------------------------------
# Analysis state
# ---------------------------------------------------------------------------

analysis_state: dict = {
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
shutdown_event = threading.Event()


def get_analysis_state() -> dict:
    """Thread-safe snapshot of analysis state."""
    with _state_lock:
        return dict(analysis_state)


def update_analysis_state(**kwargs):
    """Thread-safe partial update of analysis_state."""
    with _state_lock:
        analysis_state.update(kwargs)


def append_debug_log(message: str):
    """Append a timestamped debug message."""
    ts = datetime.now().strftime("%H:%M:%S")
    with _state_lock:
        analysis_state.setdefault("debug_log", []).append(f"[{ts}] {message}")


def append_timeline(step: str, detail: str = ""):
    """Append a timestamped timeline entry."""
    now = datetime.now().strftime("%H:%M:%S")
    entry = {"time": now, "step": step, "detail": detail}
    with _state_lock:
        analysis_state.setdefault("timeline", []).append(entry)


def debug(msg: str):
    """Log debug message to console + analysis state (replaces app._debug)."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    with _state_lock:
        analysis_state.setdefault("debug_log", []).append(f"[{ts}] {msg}")


def log_step(step: str, detail: str = ""):
    """Log a timestamped timeline entry (replaces app._log_step)."""
    now = datetime.now().strftime("%H:%M:%S")
    entry = {"time": now, "step": step, "detail": detail}
    print(f"[{now}] {step}" + (f" — {detail}" if detail else ""))
    with _state_lock:
        analysis_state.setdefault("timeline", []).append(entry)


# ---------------------------------------------------------------------------
# Chat retriever
# ---------------------------------------------------------------------------

_chat_retriever = None
_chat_retriever_db_path = None
_chat_retriever_lock = threading.Lock()
_kp_cache = None


def get_chat_retriever():
    """Lazy-init the QARetriever with double-checked locking. Returns None if no DB."""
    global _chat_retriever, _chat_retriever_db_path, _kp_cache
    import glob as _glob
    db_files = _glob.glob(os.path.join(THIS_DIR, "intermediate", "*_knowledge.db"))
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
        qa_rows = retriever.rebuild()
        _chat_retriever = retriever
        _chat_retriever_db_path = db_path
        _load_kp_cache_into_global(db, find_points_file(), qa_rows)
        source = _kp_cache[0].get("source", "DB") if _kp_cache else "DB"
        print(f"[Chat] KP cache loaded: {len(_kp_cache)} entries (source: {source})")
    return retriever


def _load_kp_cache_into_global(db, points_file: str, qa_rows=None):
    """Load KP cache and store in module global. Internal helper."""
    global _kp_cache
    from src.chat.context import load_kp_cache
    _kp_cache = load_kp_cache(db, points_file=points_file, qa_rows=qa_rows)


def warmup_chat_retriever():
    """Pre-load embedding model in background so first chat request is fast."""
    def _warmup():
        try:
            from src.embedding_cluster import _get_model, TOPIC_EMBED_MODEL
            _get_model(TOPIC_EMBED_MODEL)  # trigger download if not cached
            get_chat_retriever()
        except Exception as e:
            from src.logger import get_logger
            _log = get_logger()
            from src.error_utils import log_exception
            log_exception(_log, "Chat warmup", "", e, level="warning")
    t = threading.Thread(target=_warmup, daemon=True)
    t.start()


def invalidate_chat_retriever():
    """Clear cached retriever + release embedding models (called before new analysis)."""
    global _chat_retriever, _chat_retriever_db_path
    with _chat_retriever_lock:
        _chat_retriever = None
        _chat_retriever_db_path = None
    try:
        from src.embedding_cluster import clear_model_cache
        clear_model_cache()
    except Exception as e:
        from src.error_utils import log_exception
        log_exception(print, "Model cache clear", "", e)


def get_kp_cache():
    """Return the cached KP list (or None)."""
    return _kp_cache


# ---------------------------------------------------------------------------
# Eval state
# ---------------------------------------------------------------------------

_eval_state: dict = {"running": False, "progress": 0, "report": "", "error": None}
_eval_lock = threading.Lock()


def get_eval_state() -> dict:
    """Thread-safe snapshot of eval state."""
    with _eval_lock:
        return dict(_eval_state)


def start_eval_run():
    """Mark eval as running. Caller must have validated preconditions."""
    global _eval_state
    with _eval_lock:
        _eval_state = {"running": True, "progress": 0, "report": "", "error": None}


def update_eval_state(**kwargs):
    """Thread-safe partial update of eval state."""
    global _eval_state
    with _eval_lock:
        _eval_state.update(kwargs)


def finish_eval_run():
    """Mark eval as complete."""
    global _eval_state
    with _eval_lock:
        _eval_state["running"] = False
