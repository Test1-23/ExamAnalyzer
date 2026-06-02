"""Lightweight error-handling helpers for consistent exception logging.

Provides ``log_exception`` — a single-call replacement for the 90 scattered
``except Exception`` blocks across the codebase.  Produces grep-friendly
one-line error messages with stage name, context identifier, and exception
type::

    [ERROR] QA pairing | paper=2024_s_01 | ValueError: invalid format

Usage::

    from .error_utils import log_exception

    try:
        ...
    except Exception as e:
        log_exception(debug, "Verb extraction", f"batch={b}", e)
        continue  # fallback behavior unchanged

The function accepts both callable loggers (e.g. ``debug``, ``print``)
and logger objects with ``.debug()`` / ``.warning()`` methods.
"""

import traceback


def log_exception(
    logger,
    stage: str,
    context: str = "",
    exc: Exception = None,
    level: str = "debug",
) -> None:
    """Log an exception with consistent formatting.

    Args:
        logger: A callable ``(msg) -> None``, or an object with
                ``.debug()`` / ``.warning()`` / ``.error()`` methods.
        stage: Pipeline stage or operation name (e.g. ``'QA pairing'``).
        context: What was being processed (e.g. ``'paper=2024_s_01'``,
                 ``'Q=5'``, ``'batch=3'``).  Empty string is fine.
        exc: The caught exception.  ``type(exc).__name__`` is included.
        level: ``'debug'`` for non-fatal, ``'warning'`` for degraded
               operation, ``'error'`` for fatal-but-caught.

    Does NOT re-raise — the caller retains control of fallback behavior
    (return default, continue, pass).
    """
    exc_name = type(exc).__name__ if exc else "unknown"
    exc_msg = str(exc) if exc else ""
    detail = f"{exc_name}: {exc_msg}" if exc_msg else exc_name
    if context:
        msg = f"[ERROR] {stage} | {context} | {detail}"
    else:
        msg = f"[ERROR] {stage} | {detail}"

    if hasattr(logger, level):
        getattr(logger, level)(msg)
    elif callable(logger):
        logger(msg)

    # Include traceback for warning/error levels
    if level in ("warning", "error") and exc:
        tb = traceback.format_exc()
        if hasattr(logger, level):
            getattr(logger, level)(tb)
        elif callable(logger):
            logger(tb)


def log_info(logger, stage: str, message: str) -> None:
    """Log an informational message with consistent formatting.

    Produces ``[INFO] stage | message`` — mirroring the ``[ERROR] stage | ctx | exc``
    format used by ``log_exception`` so all log output shares the same
    grep-friendly prefix convention.

    Usage::

        log_info(debug, "Evolution", f"{count} fragments migrated")
        # → [INFO] Evolution | 15 fragments migrated

        log_info(_debug, display_name, "PDF extraction complete")
        # → [INFO] 2024_s_01 | PDF extraction complete
    """
    if callable(logger):
        logger(f"[INFO] {stage} | {message}")
