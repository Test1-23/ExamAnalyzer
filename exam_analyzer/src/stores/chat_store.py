"""ChatStore — domain store for chat_history table."""

from .base import BaseStore


class ChatStore(BaseStore):
    """Operations for chat history persistence."""

    def save_message(self, session_id: str, role: str, content: str, sources: str = ""):
        self._qb_insert("chat_history",
                        session_id=session_id, role=role,
                        content=content, sources=sources)

    def get_history(self, session_id: str, limit: int = 50) -> list[dict]:
        rows = self._qb.get_where("chat_history", session_id=session_id,
                                  order_by="created_at ASC", limit=limit)
        return [{"role": r["role"], "content": r["content"], "sources": r["sources"]} for r in rows]

    def clear_history(self, session_id: str):
        self._qb_delete_where("chat_history", session_id=session_id)
