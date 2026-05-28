"""Unified graceful degradation: safe_call decorator + safe context manager.

Replaces the pervasive ``except Exception: pass`` anti-pattern with logged fallbacks.
"""
import functools
import traceback
from typing import Any, Callable

from .logger import get_logger

_log = get_logger()

_LOG_LEVEL_MAP = {"debug": 10, "info": 20, "warning": 30, "error": 40}


def safe_call(default: Any = None, log_level: str = "debug") -> Callable:
    """Decorator: catch any Exception, log with traceback, return *default*.

    ``default`` may be a callable (factory) or a static value.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception:
                _log.log(
                    _LOG_LEVEL_MAP.get(log_level, 10),
                    "%s failed:\n%s",
                    func.__name__,
                    traceback.format_exc(),
                )
                return default() if callable(default) else default
        return wrapper
    return decorator


class _SafeContext:
    """Context manager that suppresses Exception and logs it."""

    def __init__(self, default: Any = None, log_level: str = "debug") -> None:
        self.default = default
        self.log_level = log_level
        self.value: Any = default

    def __enter__(self) -> "_SafeContext":
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None,
                 exc_tb: Any) -> bool:
        if exc_type is not None and issubclass(exc_type, Exception):
            _log.log(
                _LOG_LEVEL_MAP.get(self.log_level, 10),
                "SafeContext suppressed %s: %s",
                exc_type.__name__,
                exc_val,
            )
            self.value = self.default
            return True  # suppress
        return False


def safe(default: Any = None, log_level: str = "debug") -> _SafeContext:
    """Context manager: ``with safe(default=[]): ...`` logs and suppresses Exception."""
    return _SafeContext(default, log_level)
