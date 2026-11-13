"""VectorStore — domain store for kp_vectors, qa_kp_scores, topic_vectors."""

import numpy as np
from .base import BaseStore


class VectorStore(BaseStore):
    """Operations for KP vectors, QA-KP scores, and topic vectors."""

    # -- KP Vectors --

    def upsert_kp_vector(self, kp_id: str, vector: np.ndarray):
        self._write(
            """INSERT OR REPLACE INTO kp_vectors
               (kp_id, vector, adjustment_count, updated_at)
               VALUES (?, ?, COALESCE((SELECT adjustment_count FROM kp_vectors
                WHERE kp_id=?), 0) + 1, datetime('now'))""",
            (kp_id, vector.astype(np.float32).tobytes(), kp_id),
        )

    def get_kp_vector(self, kp_id: str) -> np.ndarray | None:
        row = self._read_one(
            "SELECT vector FROM kp_vectors WHERE kp_id=?", (kp_id,)
        )
        return np.frombuffer(row["vector"], dtype=np.float32) if row else None

    # -- QA-KP Scores --

    def upsert_qa_kp_score(self, qa_id: int, kp_id: str, relevance_score: float):
        self._write(
            """INSERT OR REPLACE INTO qa_kp_scores (qa_id, kp_id, relevance_score)
               VALUES (?, ?, ?)""",
            (qa_id, kp_id, relevance_score),
        )

    def get_qa_kp_scores(self, qa_id: int) -> dict[str, float]:
        rows = self._read_all(
            "SELECT kp_id, relevance_score FROM qa_kp_scores WHERE qa_id=?", (qa_id,)
        )
        return {r["kp_id"]: r["relevance_score"] for r in rows}

    # -- Layout Positions (migration v5) --

    def upsert_positions_batch(self, rows: list[dict]) -> int:
        """Persist relaxed positions (object_type, object_id, position, anchor)."""
        if not rows:
            return 0
        payload = [
            (
                r["object_type"],
                str(r["object_id"]),
                np.asarray(r["position"], dtype=np.float32).tobytes(),
                np.asarray(r["anchor"], dtype=np.float32).tobytes(),
                r.get("encoder_version", ""),
                int(r.get("epoch", 0)),
            )
            for r in rows
        ]
        self._write_many(
            """INSERT OR REPLACE INTO vector_positions
               (object_type, object_id, position, anchor,
                encoder_version, epoch, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
            payload,
        )
        return len(payload)

    def get_positions(self, object_type: str) -> dict[str, np.ndarray]:
        rows = self._read_all(
            "SELECT object_id, position FROM vector_positions WHERE object_type=?",
            (object_type,),
        )
        return {
            r["object_id"]: np.frombuffer(r["position"], dtype=np.float32)
            for r in rows
        }

    def get_position(self, object_type: str, object_id: str) -> np.ndarray | None:
        row = self._read_one(
            "SELECT position FROM vector_positions WHERE object_type=? AND object_id=?",
            (object_type, str(object_id)),
        )
        return np.frombuffer(row["position"], dtype=np.float32) if row else None

    def count_positions(self, object_type: str | None = None) -> int:
        if object_type is None:
            row = self._read_one("SELECT COUNT(*) AS cnt FROM vector_positions")
        else:
            row = self._read_one(
                "SELECT COUNT(*) AS cnt FROM vector_positions WHERE object_type=?",
                (object_type,),
            )
        return row["cnt"] if row else 0

    def delete_positions(self, encoder_version: str | None = None) -> int:
        if encoder_version is None:
            return self._write("DELETE FROM vector_positions").rowcount
        return self._write(
            "DELETE FROM vector_positions WHERE encoder_version=?",
            (encoder_version,),
        ).rowcount

    # -- Topic Vectors --

    def upsert_topic_vector(self, topic_id: str, vector: np.ndarray, member_count: int = 0):
        self._write(
            """INSERT OR REPLACE INTO topic_vectors
               (topic_id, vector, member_kp_count, updated_at)
               VALUES (?, ?, ?, datetime('now'))""",
            (topic_id, vector.astype(np.float32).tobytes(), member_count),
        )

    def prune_kp_vectors(self, keep_ids: list[str]) -> int:
        """Delete KP vectors that no longer correspond to a current KP."""
        if not keep_ids:
            return self._write("DELETE FROM kp_vectors").rowcount
        placeholders = ",".join("?" * len(keep_ids))
        return self._write(
            f"DELETE FROM kp_vectors WHERE kp_id NOT IN ({placeholders})",
            tuple(keep_ids),
        ).rowcount

    def get_topic_vector(self, topic_id: str) -> np.ndarray | None:
        row = self._read_one(
            "SELECT vector FROM topic_vectors WHERE topic_id=?", (topic_id,)
        )
        return np.frombuffer(row["vector"], dtype=np.float32) if row else None
