"""Database layer — ConnectionMgr, QueryBuilder, SqlDialect, BaseRepository."""

from .connection import ConnectionMgr, _TransactionContext
from .dialect import SqlDialect, SqliteDialect
from .query_builder import QueryBuilder
from .repository import BaseRepository

__all__ = [
    "ConnectionMgr", "_TransactionContext",
    "SqlDialect", "SqliteDialect",
    "QueryBuilder", "BaseRepository",
]
