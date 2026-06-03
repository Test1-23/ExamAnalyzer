"""Backward-compat shim — all symbols now in pipeline/ package."""
from .pipeline import run_pipeline

__all__ = ["run_pipeline"]
