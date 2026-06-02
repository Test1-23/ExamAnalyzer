"""QADatabase -- SQLite QA knowledge base with Store-based CRUD."""

import sqlite3
import numpy as np
from typing import Optional

from .connection_manager import ConnectionMgr
from .logger import get_logger
from .models import KPSpec, VerbPatternSpec, KpEdgeSpec, DependencySpec
from .query_builder import QueryBuilder

_log = get_logger()

from .constants import SQLITE_PARAM_CHUNK
from .retriever import QARetriever, make_topic_id, log_schema_status  # re-export for backward compat

# ============================================================
# QADatabase
# ============================================================

class QADatabase:
    """Facade assembling ConnectionMgr → QueryBuilder → 8 Domain Stores.

    Interface freeze policy (2026-05-31): No new methods shall be added to this
    class.  New data access should use the Store directly: ``db.qa.insert(...)``,
    ``db.topic.upsert_link(...)``, etc.

    All proxy methods have been removed (55 methods deleted across WSD-005,
    WSD-009a, WSD-022).  Only 7 stable API methods remain — these are
    widely-used convenience wrappers (4+ caller files each) validated across
    ~50 call sites:

    ==================== ================================================
    ``get(qa_id)``        Single QA lookup — ~10 caller files
    ``count()``           Total QA count — 6 caller files
    ``get_all()``         All QAs ordered by id — 6 caller files
    ``get_kp_by_id(id)``  Single KP lookup — 4 caller files
    ``get_topic_groups()`` Topic→QA grouping — 5 caller files
    ``get_verb_patterns()`` Command verb statistics — 4 caller files
    ``checkpoint(...)``   Analysis progress checkpoint — 4 caller files
    ==================== ================================================
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db = ConnectionMgr(db_path)
        self._qb = QueryBuilder(self._db)

        from .stores.qa_store import QaStore
        from .stores.kp_store import KpStore
        from .stores.topic_store import TopicStore
        from .stores.fragment_store import FragmentStore
        from .stores.chat_store import ChatStore
        from .stores.student_store import StudentStore
        from .stores.analysis_store import AnalysisStore
        from .stores.vector_store import VectorStore

        self.qa = QaStore(self._qb)
        self.kp = KpStore(self._qb)
        self.topic = TopicStore(self._qb)
        self.fragment = FragmentStore(self._qb)
        self.chat = ChatStore(self._qb)
        self.student = StudentStore(self._qb)
        self.analysis = AnalysisStore(self._qb)
        self.vector = VectorStore(self._qb)

    @property
    def conn(self) -> sqlite3.Connection:
        return self._db.conn

    @property
    def _write_lock(self):
        return self._db._write_lock

    def _commit(self):
        """Commit only when NOT inside a transaction. Safe to call inside ``with db.transaction()``."""
        self._db.maybe_commit()

    def transaction(self):
        """Context manager for atomic multi-step writes. See ConnectionMgr.transaction."""
        return self._db.transaction()

    # -- Stable API (7 methods) --

    def get(self, qa_id: int) -> Optional[dict]:
        """(stable API) Single QA lookup by id."""
        return self._qb.get("qa_pairs", qa_id)

    def get_all(self) -> list[dict]:
        """(stable API) All QAs ordered by id."""
        return self._qb.get_all("qa_pairs", order_by="id")

    def count(self) -> int:
        """(stable API) Total QA count."""
        return self._qb.count("qa_pairs")

    def get_topic_groups(self) -> dict[str, list[dict]]:
        """(stable API) Topic→QA grouping."""
        return self.qa.get_topic_groups()

    def get_kp_by_id(self, kp_id: str) -> dict:
        """(stable API) Single KP lookup by id."""
        return self.kp.get_by_id(kp_id)

    def get_verb_patterns(self) -> list[dict]:
        """(stable API) Command verb statistics."""
        return self.analysis.get_verb_patterns()

    def checkpoint(self, task_name: str, qa_count: int, status: str = "completed"):
        """(stable API) Analysis progress checkpoint."""
        self.analysis.checkpoint(task_name, qa_count, status)

    # -- Lifecycle --

    def close(self):
        self._db.close()


# ============================================================
# QARetriever

