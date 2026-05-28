"""Shared pipeline utilities: worker limits, error logging."""
import os
import traceback


def get_worker_limit(n: int, api_heavy: bool = False) -> int:
    """Dynamic worker limit: configurable via env, with sensible defaults.

    PIPELINE_MAX_WORKERS env var overrides the cap. Otherwise:
      - General (CPU work): cap at 16
      - API-heavy (rate limited): cap at 8
    """
    env_override = os.environ.get("PIPELINE_MAX_WORKERS", "")
    if env_override.isdigit():
        return max(1, min(n, int(env_override)))
    cap = 8 if api_heavy else 16
    return max(1, min(n, cap))


def log_stage_error(stage: str, debug, exc: Exception):
    """Standardized error logging for any pipeline stage."""
    debug(f"[{stage}] {type(exc).__name__}: {exc}")
    debug(traceback.format_exc())
