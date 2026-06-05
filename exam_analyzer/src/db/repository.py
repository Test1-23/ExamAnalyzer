"""BaseRepository — generic typed CRUD base for simple single-table stores.

Stores with complex JOIN logic (KpStore, AnalysisStore, FragmentStore)
should stay hand-written.  Stores that are mostly single-table CRUD can
inherit from this to get free methods.
"""

from typing import Any, TypeVar

from .query_builder import QueryBuilder

T = TypeVar("T")


class BaseRepository:
    """Typed CRUD base for single-table domain stores.

    Usage::

        class ChatStore(BaseRepository):
            table = "chat_history"
            id_col = "id"

            def __init__(self, qb: QueryBuilder):
                super().__init__(qb)
    """

    table: str = ""
    id_col: str = "id"

    def __init__(self, qb: QueryBuilder):
        self._qb: QueryBuilder = qb

    def get(self, id_value: Any) -> dict | None:
        return self._qb.get(self.table, id_value, id_col=self.id_col)

    def get_all(self, order_by: str = "", limit: int = 0) -> list[dict]:
        return self._qb.get_all(self.table, order_by=order_by, limit=limit)

    def count(self, **filters) -> int:
        return self._qb.count(self.table, **filters)

    def insert(self, **values) -> int:
        return self._qb.insert(self.table, **values)

    def update(self, id_value: Any, **kv) -> int:
        return self._qb.update(self.table, id_value, id_col=self.id_col, **kv)

    def delete(self, id_value: Any) -> int:
        return self._qb.delete(self.table, id_value, id_col=self.id_col)
