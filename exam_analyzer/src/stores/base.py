"""BaseStore — shared write-lock + commit boilerplate for all domain stores.

Every Store writes through this base.  Individual methods call
``self._write(sql, params)`` or ``self._read_all(sql, params)`` instead of
manually acquiring ``self._mgr._write_lock`` and calling
``self._mgr.maybe_commit()``.

Subclasses must set ``self._qb`` (QueryBuilder) and ``self._mgr``
(ConnectionMgr) in ``__init__`` via ``super().__init__(qb)``.
"""

import sqlite3


class BaseStore:
    """Abstract base for domain stores that write through ConnectionMgr.

    Provides:

    * ``_write(sql, params)`` — execute a write, auto-commit (unless inside tx)
    * ``_write_many(sql, params_list)`` — execute many writes in one go
    * ``_write_insert(sql, params)`` — execute INSERT, return lastrowid
    * ``_read(sql, params)`` — execute a read, return cursor
    * ``_read_one(sql, params)`` — fetch one row as dict or None
    * ``_read_all(sql, params)`` — fetch all rows as list[dict]
    """

    def __init__(self, qb: "QueryBuilder"):  # noqa: F821
        self._qb = qb
        self._mgr = qb._mgr

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._qb.conn

    # ── Write helpers ──────────────────────────────────────────

    def _write(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a write SQL statement with auto-commit.

        Acquires the write lock, asserts it's held, executes, and
        calls maybe_commit() — safe to call inside or outside a
        ``with db.transaction():`` block.
        """
        with self._mgr._write_lock:
            self._mgr._assert_write_locked()
            cur = self._conn.execute(sql, params)
            self._mgr.maybe_commit()
            return cur

    def _write_many(self, sql: str, params_list: list[tuple]) -> sqlite3.Cursor:
        """Execute many writes with auto-commit."""
        with self._mgr._write_lock:
            self._mgr._assert_write_locked()
            cur = self._conn.executemany(sql, params_list)
            self._mgr.maybe_commit()
            return cur

    def _write_insert(self, sql: str, params: tuple = ()) -> int:
        """Execute INSERT and return lastrowid."""
        return self._write(sql, params).lastrowid

    def _write_locked(self):
        """Context manager holding the write lock for a multi-statement block.

        Use for read-then-write sequences where the read and write must be
        atomic (e.g. check-then-insert).  Calls ``maybe_commit()`` on exit.

        Usage::

            with self._write_locked():
                existing = self._read_one("SELECT ...")
                if not existing:
                    self._qb.insert("table", ...)
        """
        return _WriteLockedBlock(self._mgr)

    # ── QueryBuilder wrappers (acquire write lock) ─────────────

    def _qb_insert(self, table: str, /, **values) -> int:
        """Insert via QueryBuilder, with write lock acquired. Returns lastrowid."""
        with self._mgr._write_lock:
            self._mgr._assert_write_locked()
            return self._qb.insert(table, **values)

    def _qb_delete(self, table: str, id_value, id_col: str = "id") -> int:
        """Delete via QueryBuilder, with write lock acquired. Returns rowcount."""
        with self._mgr._write_lock:
            self._mgr._assert_write_locked()
            return self._qb.delete(table, id_value, id_col=id_col)

    def _qb_delete_where(self, table: str, /, **filters) -> int:
        """Delete via QueryBuilder WHERE, with write lock acquired. Returns rowcount."""
        with self._mgr._write_lock:
            self._mgr._assert_write_locked()
            return self._qb.delete_where(table, **filters)

    def _qb_update(self, table: str, id_value, /, id_col: str = "id", **kv) -> int:
        """Update via QueryBuilder, with write lock acquired. Returns rowcount."""
        with self._mgr._write_lock:
            self._mgr._assert_write_locked()
            return self._qb.update(table, id_value, id_col=id_col, **kv)

    # ── Read helpers ───────────────────────────────────────────

    def _read(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a read query. Returns cursor for fetchone/fetchall."""
        return self._conn.execute(sql, params)

    def _read_one(self, sql: str, params: tuple = ()) -> dict | None:
        """Fetch one row as dict, or None."""
        row = self._read(sql, params).fetchone()
        return dict(row) if row else None

    def _read_all(self, sql: str, params: tuple = ()) -> list[dict]:
        """Fetch all rows as list[dict]."""
        return [dict(r) for r in self._read(sql, params).fetchall()]


class _WriteLockedBlock:
    """Context manager that holds the write lock for a multi-statement block.

    Calls ``maybe_commit()`` on exit so individual statements inside the
    block don't need to.
    """

    def __init__(self, mgr):
        self._mgr = mgr

    def __enter__(self):
        self._mgr._write_lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                self._mgr.maybe_commit()
        finally:
            self._mgr._write_lock.release()
