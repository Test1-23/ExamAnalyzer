"""VectorStore — domain store for kp_vectors, qa_kp_scores, topic_vectors."""

import numpy as np


class VectorStore:
    """Operations for KP vectors, QA-KP scores, and topic vectors."""

    def __init__(self, qb: "QueryBuilder"):
        self._qb = qb
        self._mgr = qb._mgr

    # -- KP Vectors --

    def upsert_kp_vector(self, kp_id: str, vector: np.ndarray):
        with self._mgr._write_lock:
            self._mgr._assert_write_locked()
            self._qb.conn.execute(
                """INSERT OR REPLACE INTO kp_vectors
                   (kp_id, vector, adjustment_count, updated_at)
                   VALUES (?, ?, COALESCE((SELECT adjustment_count FROM kp_vectors
                    WHERE kp_id=?), 0) + 1, datetime('now'))""",
                (kp_id, vector.astype(np.float32).tobytes(), kp_id),
            )
            self._mgr.maybe_commit()

    def get_kp_vector(self, kp_id: str) -> np.ndarray | None:
        row = self._qb.conn.execute(
            "SELECT vector FROM kp_vectors WHERE kp_id=?", (kp_id,)
        ).fetchone()
        return np.frombuffer(row["vector"], dtype=np.float32) if row else None

    # -- QA-KP Scores --

    def upsert_qa_kp_score(self, qa_id: int, kp_id: str, relevance_score: float):
        with self._mgr._write_lock:
            self._mgr._assert_write_locked()
            self._qb.conn.execute(
                """INSERT OR REPLACE INTO qa_kp_scores (qa_id, kp_id, relevance_score)
                   VALUES (?, ?, ?)""",
                (qa_id, kp_id, relevance_score),
            )
            self._mgr.maybe_commit()

    def get_qa_kp_scores(self, qa_id: int) -> dict[str, float]:
        rows = self._qb.conn.execute(
            "SELECT kp_id, relevance_score FROM qa_kp_scores WHERE qa_id=?", (qa_id,)
        ).fetchall()
        return {r["kp_id"]: r["relevance_score"] for r in rows}

    # -- Topic Vectors --

    def upsert_topic_vector(self, topic_id: str, vector: np.ndarray, member_count: int = 0):
        with self._mgr._write_lock:
            self._mgr._assert_write_locked()
            self._qb.conn.execute(
                """INSERT OR REPLACE INTO topic_vectors
                   (topic_id, vector, member_kp_count, updated_at)
                   VALUES (?, ?, ?, datetime('now'))""",
                (topic_id, vector.astype(np.float32).tobytes(), member_count),
            )
            self._mgr.maybe_commit()

    def get_topic_vector(self, topic_id: str) -> np.ndarray | None:
        row = self._qb.conn.execute(
            "SELECT vector FROM topic_vectors WHERE topic_id=?", (topic_id,)
        ).fetchone()
        return np.frombuffer(row["vector"], dtype=np.float32) if row else None
