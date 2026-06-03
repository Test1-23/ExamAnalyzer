"""PipelineContext — immutable infrastructure bundle for pipeline step functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..knowledge_base import QADatabase, QARetriever
    from .tracker import ProgressTracker


@dataclass(frozen=True)
class PipelineContext:
    """Immutable infrastructure bundle for Phase 1/2 step functions.

    Groups 6 repeatedly-passed parameters (client, db, debug, display_name,
    retriever, tracker) into a single ctx argument, reducing function
    signatures by 3-6 positional parameters each.
    """
    client: object            # DeepSeek API client
    db: QADatabase             # QADatabase instance
    debug: Callable[[str], None]  # debug callback
    display_name: str           # current paper display name
    retriever: QARetriever      # QARetriever instance
    tracker: ProgressTracker    # ProgressTracker instance
