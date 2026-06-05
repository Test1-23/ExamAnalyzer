"""SqlDialect — abstract SQL dialect differences for future DB portability.

Current implementation: SqliteDialect (SQLite 3.x with WAL mode).
To add PostgreSQL support, implement ``PostgresDialect(SqlDialect)``.
"""

from abc import ABC, abstractmethod


class SqlDialect(ABC):
    """Abstract SQL dialect — placeholder for future non-SQLite backends."""

    @property
    @abstractmethod
    def param_placeholder(self) -> str:
        """The parameter placeholder character ('?' for SQLite, '%s' for pg)."""
        ...

    @abstractmethod
    def quote_identifier(self, name: str) -> str:
        """Quote a table or column identifier."""
        ...

    @abstractmethod
    def last_rowid_sql(self) -> str:
        """SQL to retrieve last inserted row id."""
        ...


class SqliteDialect(SqlDialect):
    """SQLite 3.x dialect."""

    param_placeholder = "?"

    def quote_identifier(self, name: str) -> str:
        return f'"{name}"'

    def last_rowid_sql(self) -> str:
        return "SELECT last_insert_rowid()"
