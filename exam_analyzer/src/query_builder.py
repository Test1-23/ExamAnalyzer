"""QueryBuilder — lightweight SQL generator using %%-formatting.

Uses ``%%``-formatting (NOT ``.format()``) because mark-scheme text
frequently contains ``{`` ``}`` characters that would crash ``.format()``.

User values are **always** passed as ``?`` parameters — only table names,
column names, and SQL keywords are interpolated via ``%%s`` / ``%%d``.
"""

import sqlite3
from typing import Any, Optional


class QueryBuilder:
    """Lightweight SQL builder over a ConnectionMgr.

    Provides single-table CRUD, aggregation, and a ``raw()`` escape hatch
    for complex JOIN / subquery statements.
    """

    def __init__(self, mgr: "ConnectionMgr"):
        self._mgr = mgr

    @property
    def conn(self) -> sqlite3.Connection:
        return self._mgr.conn

    # ------------------------------------------------------------------
    # Single-table CRUD
    # ------------------------------------------------------------------

    def insert(self, table: str, /, **values) -> int:
        """INSERT INTO table (cols) VALUES (?, ...). Returns lastrowid."""
        cols = list(values.keys())
        placeholders = ", ".join(["?"] * len(cols))
        sql = ("INSERT INTO %s (%s) VALUES (%s)"
               % (table, ", ".join(cols), placeholders))
        cur = self.conn.execute(sql, list(values.values()))
        self._mgr.maybe_commit()
        return cur.lastrowid

    def upsert(self, table: str, /, keys: Optional[list[str]] = None, **values) -> None:
        """INSERT OR REPLACE INTO table (cols) VALUES (?, ...).

        For ON CONFLICT DO UPDATE patterns, use ``raw()`` instead.
        """
        cols = list(values.keys())
        placeholders = ", ".join(["?"] * len(cols))
        sql = ("INSERT OR REPLACE INTO %s (%s) VALUES (%s)"
               % (table, ", ".join(cols), placeholders))
        self.conn.execute(sql, list(values.values()))
        self._mgr.maybe_commit()

    def get(self, table: str, id_value: Any, id_col: str = "id") -> Optional[dict]:
        """SELECT * FROM table WHERE id_col = ?."""
        sql = "SELECT * FROM %s WHERE %s=?" % (table, id_col)
        row = self.conn.execute(sql, (id_value,)).fetchone()
        return dict(row) if row else None

    def get_all(self, table: str, order_by: str = "", limit: int = 0) -> list[dict]:
        """SELECT * FROM table [ORDER BY ...] [LIMIT ...]."""
        sql = "SELECT * FROM %s" % table
        if order_by:
            sql += " ORDER BY " + order_by
        if limit:
            sql += " LIMIT %d" % limit
        rows = self.conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def get_where(self, table: str, /, order_by: str = "", limit: int = 0,
                  **filters) -> list[dict]:
        """SELECT * FROM table WHERE col1=? AND col2=? ..."""
        if not filters:
            return self.get_all(table, order_by, limit)
        clauses = ["%s=?" % k for k in filters]
        sql = "SELECT * FROM %s WHERE %s" % (table, " AND ".join(clauses))
        if order_by:
            sql += " ORDER BY " + order_by
        if limit:
            sql += " LIMIT %d" % limit
        rows = self.conn.execute(sql, list(filters.values())).fetchall()
        return [dict(r) for r in rows]

    def count(self, table: str, /, **filters) -> int:
        """SELECT COUNT(*) as cnt FROM table [WHERE ...]."""
        if filters:
            clauses = ["%s=?" % k for k in filters]
            sql = "SELECT COUNT(*) as cnt FROM %s WHERE %s" % (table, " AND ".join(clauses))
            row = self.conn.execute(sql, list(filters.values())).fetchone()
        else:
            sql = "SELECT COUNT(*) as cnt FROM %s" % table
            row = self.conn.execute(sql).fetchone()
        return row["cnt"] if row else 0

    def count_group(self, table: str, group_col: str, /, **filters) -> list[dict]:
        """SELECT group_col, COUNT(*) as cnt FROM table [WHERE ...] GROUP BY group_col."""
        params = []
        filter_clause = ""
        if filters:
            clauses = ["%s=?" % k for k in filters]
            filter_clause = " WHERE " + " AND ".join(clauses)
            params = list(filters.values())
        sql = ("SELECT %s, COUNT(*) as cnt FROM %s%s GROUP BY %s"
               % (group_col, table, filter_clause, group_col))
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def aggregate(self, table: str, func: str, col: str, /, **filters) -> float:
        """SELECT FUNC(col) as val FROM table [WHERE ...]."""
        params = []
        filter_clause = ""
        if filters:
            clauses = ["%s=?" % k for k in filters]
            filter_clause = " WHERE " + " AND ".join(clauses)
            params = list(filters.values())
        sql = "SELECT %s(%s) as val FROM %s%s" % (func, col, table, filter_clause)
        row = self.conn.execute(sql, params).fetchone()
        return row["val"] if row and row["val"] is not None else 0.0

    def update(self, table: str, id_value: Any, /, id_col: str = "id", **kv) -> int:
        """UPDATE table SET col1=?, col2=? WHERE id_col=?. Returns rowcount."""
        sets = ["%s=?" % k for k in kv]
        sql = ("UPDATE %s SET %s WHERE %s=?"
               % (table, ", ".join(sets), id_col))
        params = list(kv.values()) + [id_value]
        rows = self.conn.execute(sql, params)
        self._mgr.maybe_commit()
        return rows.rowcount

    def delete(self, table: str, id_value: Any, id_col: str = "id") -> int:
        """DELETE FROM table WHERE id_col = ?. Returns rowcount."""
        sql = "DELETE FROM %s WHERE %s=?" % (table, id_col)
        rows = self.conn.execute(sql, (id_value,))
        self._mgr.maybe_commit()
        return rows.rowcount

    def delete_where(self, table: str, /, **filters) -> int:
        """DELETE FROM table WHERE col1=? AND col2=? ..."""
        clauses = ["%s=?" % k for k in filters]
        sql = "DELETE FROM %s WHERE %s" % (table, " AND ".join(clauses))
        rows = self.conn.execute(sql, list(filters.values()))
        self._mgr.maybe_commit()
        return rows.rowcount

    # ------------------------------------------------------------------
    # Escape hatch
    # ------------------------------------------------------------------

    def raw(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        """Execute a raw SQL query and return fetched rows.

        For complex JOIN, subquery, or UNION statements that QueryBuilder
        cannot express. Caller is responsible for iterating the result.

        Unlike the CRUD helpers, ``raw()`` does **not** call ``maybe_commit()``
        — write operations via ``raw()`` should use ``ConnectionMgr.transaction()``
        or commit explicitly.
        """
        return self.conn.execute(sql, params).fetchall()
