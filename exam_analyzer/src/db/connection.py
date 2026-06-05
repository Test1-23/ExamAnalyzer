"""ConnectionMgr — sole holder of sqlite3.Connection with WAL, migrations, and transaction support."""

import sqlite3
import os
import threading
from typing import Optional

from ..schema import SCHEMA_DDL, SCHEMA_INDEXES, SCHEMA_MIGRATIONS
from ..logger import get_logger

_log = get_logger()


class ConnectionMgr:
    """Holder of sqlite3 connections for a QADatabase — per-thread pooling.

    Provides:
    - Per-thread lazy connections with WAL mode (concurrent reads, serial writes)
    - DDL execution and versioned schema migrations (once per database)
    - ``transaction()`` context manager for atomic multi-step writes
    - Reentrant write lock (RLock) so individual methods that acquire the
      lock can be safely nested inside a ``transaction()`` block
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._connections = threading.local()
        self._all_connections: list = []  # for close()
        self._connections_lock = threading.Lock()
        self._write_lock = threading.RLock()
        self._tx_depth = 0
        self._schema_initialized = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._connections, 'conn', None)
        if conn is None:
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            self._connections.conn = conn
            with self._connections_lock:
                self._all_connections.append(conn)
            # Schema initialization runs once (on whichever thread connects first)
            if not self._schema_initialized:
                with self._write_lock:
                    if not self._schema_initialized:
                        self._init_tables(conn)
                        self._schema_initialized = True
        return conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_tables(self, conn):
        for _name, ddl in SCHEMA_DDL:
            conn.execute(ddl)
        for idx_ddl in SCHEMA_INDEXES:
            conn.execute(idx_ddl)
        conn.commit()
        self._run_migrations(conn)

    def _run_migrations(self, conn):
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, "
            "description TEXT, applied_at TEXT DEFAULT (datetime('now')))"
        )
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) as v FROM schema_version"
        ).fetchone()
        current = row["v"] if row else 0

        for version, description, statements in SCHEMA_MIGRATIONS:
            if version <= current:
                continue
            for stmt in statements:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
            conn.execute(
                "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                (version, description),
            )
            _log.info(f"DB migration v{version}: {description}")
        conn.commit()

    # ------------------------------------------------------------------
    # Transaction
    # ------------------------------------------------------------------

    def maybe_commit(self):
        """Commit only when NOT inside a transaction context.
        Individual CRUD methods call this instead of ``self.conn.commit()``
        so they are safe to nest inside a ``with db.transaction():`` block.
        """
        if self._tx_depth == 0:
            self.conn.commit()

    def _assert_write_locked(self):
        """Debug-only: raise AssertionError if write lock is not held by current thread.

        Store write methods call this as their first line inside
        ``with self._mgr._write_lock:`` so that a missing lock is caught
        immediately instead of causing silent data corruption.

        Uses ``RLock._is_owned()`` (CPython) which correctly tests whether
        the *current* thread holds the reentrant lock — ``acquire(False)``
        always succeeds from the same thread on an RLock, so it cannot be
        used for this check.

        The entire body is guarded by ``if __debug__:`` so ``python -O``
        eliminates it — zero overhead in production.
        """
        if __debug__:
            if not self._write_lock._is_owned():
                raise AssertionError(
                    "Store write operation called without holding _write_lock"
                )

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
            with self._connections_lock:
                conns = list(self._all_connections)
                self._all_connections.clear()
            for conn in conns:
                try:
                    conn.close()
                except Exception:
                    pass


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
