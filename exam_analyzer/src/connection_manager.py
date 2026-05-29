"""ConnectionMgr — sole holder of sqlite3.Connection with WAL, migrations, and transaction support."""

import sqlite3
import os
import threading
from typing import Optional

from .schema import SCHEMA_DDL, SCHEMA_INDEXES, SCHEMA_MIGRATIONS
from .logger import get_logger

_log = get_logger()


class ConnectionMgr:
    """Sole holder of the sqlite3.Connection for a QADatabase.

    Provides:
    - Lazy connection with double-checked locking + WAL mode
    - DDL execution and versioned schema migrations
    - ``transaction()`` context manager for atomic multi-step writes
    - Reentrant write lock (RLock) so individual methods that acquire the
      lock can be safely nested inside a ``transaction()`` block
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._write_lock = threading.RLock()
        self._tx_depth = 0

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            with self._write_lock:
                if self._conn is None:
                    os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
                    self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
                    self._conn.execute("PRAGMA journal_mode=WAL")
                    self._conn.row_factory = sqlite3.Row
                    self._init_tables()
        return self._conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_tables(self):
        for _name, ddl in SCHEMA_DDL:
            self.conn.execute(ddl)
        for idx_ddl in SCHEMA_INDEXES:
            self.conn.execute(idx_ddl)
        self.conn.commit()
        self._run_migrations()

    def _run_migrations(self):
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, "
            "description TEXT, applied_at TEXT DEFAULT (datetime('now')))"
        )
        current = self.conn.execute(
            "SELECT COALESCE(MAX(version), 0) as v FROM schema_version"
        ).fetchone()["v"]

        for version, description, statements in SCHEMA_MIGRATIONS:
            if version <= current:
                continue
            for stmt in statements:
                try:
                    self.conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
            self.conn.execute(
                "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                (version, description),
            )
            _log.info(f"DB migration v{version}: {description}")
        self.conn.commit()

    # ------------------------------------------------------------------
    # Transaction
    # ------------------------------------------------------------------

    @property
    def in_transaction(self) -> bool:
        return self._tx_depth > 0

    def maybe_commit(self):
        """Commit only when NOT inside a transaction context.
        Individual CRUD methods call this instead of ``self.conn.commit()``
        so they are safe to nest inside a ``with db.transaction():`` block.
        """
        if self._tx_depth == 0:
            self.conn.commit()

    def transaction(self):
        """Context manager for atomic multi-step writes.

        Usage::

            with db.transaction():
                db.qa.insert(...)
                db.kp.set_membership(...)

        Acquires the write lock, issues ``BEGIN IMMEDIATE`` (only for the
        outermost call — nested transactions are flattened), yields the
        connection, and on exit either commits or rolls back.
        Individual Store methods that also acquire ``_write_lock`` are safe
        to call inside the block because the lock is reentrant (RLock).
        """
        return _TransactionContext(self)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        with self._write_lock:
            if self._conn:
                self._conn.close()
                self._conn = None


class _TransactionContext:
    def __init__(self, mgr: ConnectionMgr):
        self._mgr = mgr
        self._lock = mgr._write_lock
        self._outer = False

    def __enter__(self):
        self._lock.acquire()
        if self._mgr._tx_depth == 0:
            self._outer = True
            self._mgr.conn.execute("BEGIN IMMEDIATE")
        self._mgr._tx_depth += 1
        return self._mgr.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._mgr._tx_depth -= 1
        try:
            if self._outer:
                if exc_type is None:
                    self._mgr.conn.commit()
                else:
                    self._mgr.conn.rollback()
        finally:
            self._lock.release()
