"""FragmentStore — domain store for ms_fragments, fragment_help_map, fragment_centrality."""


class FragmentStore:
    """Operations for MS fragments, help records, and centrality."""

    def __init__(self, qb: "QueryBuilder"):
        self._qb = qb
        self._mgr = qb._mgr

    # -- MS Fragments --

    def insert_batch(self, fragments: list[dict]) -> int:
        with self._mgr._write_lock:
            count = 0
            for f in fragments:
                self._qb.conn.execute(
                    """INSERT OR IGNORE INTO ms_fragments (point_id, qa_id, point_text, marks)
                       VALUES (?, ?, ?, ?)""",
                    (f["point_id"], f["qa_id"], f["point_text"], f.get("marks", 1)),
                )
                count += 1
            self._mgr.maybe_commit()
        return count

    # -- Fragment Help Map --

    def record_help_with_level(self, fragment_id: str, helped_qa_id: int,
                               help_effect: float = 0.0, help_level: str = ""):
        with self._mgr._write_lock:
            self._qb.conn.execute(
                """INSERT OR REPLACE INTO fragment_help_map
                   (fragment_id, helped_qa_id, help_effect, help_level)
                   VALUES (?, ?, ?, ?)""",
                (fragment_id, helped_qa_id, help_effect, help_level),
            )
            self._mgr.maybe_commit()

    # -- Fragment Centrality --

    def upsert_centrality(self, fragment_id: str, centrality_score: float,
                          avg_help_score: float = 0.0, topic_coherence: float = 0.0,
                          variance: float = 0.0):
        with self._mgr._write_lock:
            self._qb.conn.execute(
                """INSERT OR REPLACE INTO fragment_centrality
                   (fragment_id, verification_count, avg_help_score, topic_coherence,
                    variance, centrality_score, updated_at)
                   VALUES (?, COALESCE((SELECT verification_count FROM fragment_centrality
                    WHERE fragment_id=?), 0) + 1, ?, ?, ?, ?, datetime('now'))""",
                (fragment_id, fragment_id, avg_help_score, topic_coherence, variance, centrality_score),
            )
            self._mgr.maybe_commit()

    def get_topic_centralities(self, topic_id: str) -> list[dict]:
        rows = self._qb.conn.execute(
            """SELECT fc.* FROM fragment_centrality fc
               JOIN fragment_membership fm ON fc.fragment_id = fm.fragment_id
               WHERE fm.topic_id = ?""", (topic_id,)
        ).fetchall()
        return [dict(r) for r in rows]
