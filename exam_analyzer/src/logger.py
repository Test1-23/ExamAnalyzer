"""Unified logging: timestamped file per session + console, with rotation."""

import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

_logger = None
_logger_lock = threading.Lock()


def get_logger(name: str = "exam_analyzer") -> logging.Logger:
    """Get or create the shared logger. New file per session. Thread-safe."""
    global _logger
    if _logger is not None:
        return _logger

    with _logger_lock:
        if _logger is not None:
            return _logger

        os.makedirs(LOG_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOG_DIR, f"run_{timestamp}.log")

        _logger = logging.getLogger(name)
        _logger.setLevel(logging.DEBUG)

        fh = RotatingFileHandler(
            log_file, encoding="utf-8",
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=5,
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        _logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        _logger.addHandler(ch)

        _logger.info(f"Log session started: {log_file}")
        return _logger
