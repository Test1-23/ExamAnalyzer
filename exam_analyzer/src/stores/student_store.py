"""StudentStore — domain store for student_memory, student_knowledge_state, confusion_events, student_trajectory."""


class StudentStore:
    """Operations for student memory and knowledge state."""

    def __init__(self, qb: "QueryBuilder"):
        self._qb = qb
        self._mgr = qb._mgr

    # -- Student Memory --

    def save_memory(self, student_id: str, memory_type: str, topic: str, content: str):
        with self._mgr._write_lock:
            self._mgr._assert_write_locked()
            self._qb.insert("student_memory",
                           student_id=student_id, memory_type=memory_type,
                           topic=topic, content=content)

    def get_memories(self, student_id: str, limit: int = 20) -> list[dict]:
        return self._qb.get_where("student_memory", student_id=student_id,
                                  order_by="created_at DESC", limit=limit)

    def get_confusions(self, student_id: str) -> list[dict]:
        return self._qb.get_where("confusion_events", student_id=student_id,
                                  order_by="created_at DESC")

    def record_confusion(self, student_id: str, topic: str, trigger: str, ctype: str):
        with self._mgr._write_lock:
            self._mgr._assert_write_locked()
            self._qb.insert("confusion_events",
                           student_id=student_id, topic=topic,
                           trigger_question=trigger, confusion_type=ctype)

    # -- Knowledge State --

    def upsert_knowledge_state(self, student_id: str, topic: str, state: str):
        with self._mgr._write_lock:
            self._mgr._assert_write_locked()
            self._qb.conn.execute(
                """INSERT INTO student_knowledge_state (student_id, topic, state, evidence_count)
                   VALUES (?, ?, ?, 1)
                   ON CONFLICT(student_id, topic) DO UPDATE SET
                   state = excluded.state,
                   evidence_count = evidence_count + 1,
                   updated_at = datetime('now')""",
                (student_id, topic, state),
            )
            self._mgr.maybe_commit()

    def get_knowledge_state(self, student_id: str) -> dict[str, str]:
        rows = self._qb.get_where("student_knowledge_state", student_id=student_id)
        return {r["topic"]: r["state"] for r in rows}

    # -- Student Trajectory --

    def record_trajectory(self, student_id: str, kp_id: str,
                          from_state: str, to_state: str, trigger: str = ""):
        with self._mgr._write_lock:
            self._mgr._assert_write_locked()
            self._qb.insert("student_trajectory",
                           student_id=student_id, kp_id=kp_id,
                           from_state=from_state, to_state=to_state,
                           trigger=trigger)
