"""FragmentStore — domain store for ms_fragments, fragment_help_map, fragment_centrality."""

from ..constants import SQLITE_PARAM_CHUNK
from .base import BaseStore


class FragmentStore(BaseStore):
    """Operations for MS fragments, help records, and centrality."""

    # -- MS Fragments --

    def insert_batch(self, fragments: list[dict]) -> int:
        count = 0
        for f in fragments:
            self._write(
                """INSERT OR IGNORE INTO ms_fragments (point_id, qa_id, point_text, marks)
                   VALUES (?, ?, ?, ?)""",
                (f["point_id"], f["qa_id"], f["point_text"], f.get("marks", 1)),
            )
            count += 1
        return count

    # -- Fragment Help Map --

    def record_help_with_level(self, fragment_id: str, helped_qa_id: int,
                               help_effect: float = 0.0, help_level: str = ""):
        self._write(
            """INSERT OR REPLACE INTO fragment_help_map
               (fragment_id, helped_qa_id, help_effect, help_level)
               VALUES (?, ?, ?, ?)""",
            (fragment_id, helped_qa_id, help_effect, help_level),
        )

    def get_second_order_qas(self, source_qa_ids: list[int], limit: int = 30) -> list[int]:
        if not source_qa_ids:
            return []
        ph = ",".join("?" * len(source_qa_ids))
        rows = self._read_all(
            "SELECT DISTINCT f2.qa_id FROM ("
            "SELECT DISTINCT fhm1.helped_qa_id FROM fragment_help_map fhm1 "
            "JOIN ms_fragments mf1 ON fhm1.fragment_id = mf1.point_id "
            "WHERE mf1.qa_id IN (%s)"
            ") shared_helps "
            "JOIN fragment_help_map fhm2 ON shared_helps.helped_qa_id = fhm2.helped_qa_id "
            "JOIN ms_fragments f2 ON fhm2.fragment_id = f2.point_id "
            "LIMIT %d" % (ph, limit),
            source_qa_ids,
        )
        return [r["qa_id"] for r in rows]

    def get_helped_qas(self, source_qa_ids: list[int]) -> set[int]:
        if not source_qa_ids:
            return set()
        ph = ",".join("?" * len(source_qa_ids))
        rows = self._read_all(
            "SELECT DISTINCT helped_qa_id FROM fragment_help_map fhm2 "
            "JOIN ms_fragments mf2 ON fhm2.fragment_id = mf2.point_id "
            "WHERE mf2.qa_id IN (%s)" % ph,
            source_qa_ids,
        )
        return {r["helped_qa_id"] for r in rows}

    def get_behavior_scores(self, candidate_ids: list[int],
                            helped_ids: list[int]) -> dict[int, float]:
        if not candidate_ids or not helped_ids:
            return {}
        c_ph = ",".join("?" * len(candidate_ids))
        h_ph = ",".join("?" * len(helped_ids))
        rows = self._read_all(
            "SELECT mf.qa_id, COUNT(*) as cnt FROM fragment_help_map fhm "
            "JOIN ms_fragments mf ON fhm.fragment_id = mf.point_id "
            "WHERE mf.qa_id IN (%s) AND fhm.helped_qa_id IN (%s) "
            "GROUP BY mf.qa_id" % (c_ph, h_ph),
            candidate_ids + helped_ids,
        )
        return {r["qa_id"]: min(1.0, r["cnt"] / 10.0) for r in rows}

    # -- Fragment Centrality --

    def upsert_centrality(self, fragment_id: str, centrality_score: float,
                          avg_help_score: float = 0.0, topic_coherence: float = 0.0,
                          variance: float = 0.0):
        self._write(
            """INSERT OR REPLACE INTO fragment_centrality
               (fragment_id, verification_count, avg_help_score, topic_coherence,
                variance, centrality_score, updated_at)
               VALUES (?, COALESCE((SELECT verification_count FROM fragment_centrality
                WHERE fragment_id=?), 0) + 1, ?, ?, ?, ?, datetime('now'))""",
            (fragment_id, fragment_id, avg_help_score, topic_coherence, variance, centrality_score),
        )

    def get_topic_centralities(self, topic_id: str) -> list[dict]:
        return self._read_all(
            """SELECT fc.* FROM fragment_centrality fc
               JOIN fragment_membership fm ON fc.fragment_id = fm.fragment_id
               WHERE fm.topic_id = ?""", (topic_id,)
        )

    def get_topic_helped_questions(self, topic_id: str) -> set[int]:
        rows = self._read_all(
            "SELECT DISTINCT fhm.helped_qa_id "
            "FROM fragment_help_map fhm "
            "JOIN fragment_membership fm ON fhm.fragment_id = fm.fragment_id "
            "WHERE fm.topic_id = ?",
            (topic_id,),
        )
        return {r["helped_qa_id"] for r in rows}

    def get_texts_by_ids(self, point_ids: list[str]) -> list[str]:
        if not point_ids:
            return []
        capped = point_ids[:SQLITE_PARAM_CHUNK]
        ph = ",".join("?" * len(capped))
        rows = self._read_all(
            "SELECT point_text FROM ms_fragments WHERE point_id IN (%s)" % ph, capped
        )
        return [r["point_text"] for r in rows]
