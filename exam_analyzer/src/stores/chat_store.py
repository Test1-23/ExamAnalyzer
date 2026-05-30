"""ChatStore — domain store for chat_history table."""


class ChatStore:
    """Operations for chat history persistence."""

    def __init__(self, qb: "QueryBuilder"):
        self._qb = qb
        self._mgr = qb._mgr

    def save_message(self, session_id: str, role: str, content: str, sources: str = ""):
        with self._mgr._write_lock:
            self._mgr._assert_write_locked()
            self._qb.insert("chat_history",
                           session_id=session_id, role=role,
                           content=content, sources=sources)

    def get_history(self, session_id: str, limit: int = 50) -> list[dict]:
        rows = self._qb.get_where("chat_history", session_id=session_id,
                                  order_by="created_at ASC", limit=limit)
        return [{"role": r["role"], "content": r["content"], "sources": r["sources"]} for r in rows]

    def clear_history(self, session_id: str):
        with self._mgr._write_lock:
            self._mgr._assert_write_locked()
            self._qb.delete_where("chat_history", session_id=session_id)
