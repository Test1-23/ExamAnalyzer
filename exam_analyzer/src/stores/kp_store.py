"""KpStore — domain store for knowledge_points, kp_edges, qa_kp_membership."""

from ..models import KPSpec, KpEdgeSpec
from .base import BaseStore


class KpStore(BaseStore):
    """Operations for knowledge points, edges, and QA-KP membership."""

    # -- Knowledge Points --

    def upsert(self, spec: KPSpec):
        existing = self._read_one(
            "SELECT id FROM knowledge_points WHERE id = ?", (spec.kp_id,)
        )
        if existing:
            self._write(
                """UPDATE knowledge_points SET name=?, description=?,
                   cluster_id=?, centroid_vector=?, core_concept=?,
                   core_detail=?, variations=?, scoring_pattern=?,
                   typical_marks=?, cohesion=?, evidence_count=?,
                   quality=?, challenge_history=?,
                   last_validated_at=datetime('now')
                   WHERE id=?""",
                (spec.name, spec.description, spec.cluster_id, spec.centroid_vector,
                 spec.core_concept, spec.core_detail, spec.variations, spec.scoring_pattern,
                 spec.typical_marks, spec.cohesion, spec.evidence_count, spec.quality,
                 spec.challenge_history, spec.kp_id),
            )
        else:
            self._write(
                """INSERT INTO knowledge_points
                   (id, name, description, cluster_id, centroid_vector,
                    core_concept, core_detail, variations, scoring_pattern,
                    typical_marks, cohesion, evidence_count, quality,
                    challenge_history)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (spec.kp_id, spec.name, spec.description, spec.cluster_id, spec.centroid_vector,
                 spec.core_concept, spec.core_detail, spec.variations, spec.scoring_pattern,
                 spec.typical_marks, spec.cohesion, spec.evidence_count, spec.quality,
                 spec.challenge_history),
            )

    def get_all(self) -> list[dict]:
        return self._read_all(
            "SELECT id, name, description, core_concept, core_detail, "
            "cohesion, evidence_count, quality FROM knowledge_points "
            "ORDER BY evidence_count DESC"
        )

    def get_by_id(self, kp_id: str) -> dict:
        row = self._qb.get("knowledge_points", kp_id, id_col="id")
        return row if row else {}

    def get_representative_qas(self, kp_id: str, limit: int = 3) -> list[dict]:
        rows = self._read_all(
            """SELECT q.* FROM qa_pairs q
               JOIN qa_kp_membership m ON q.id = m.qa_id
               WHERE m.kp_id = ? AND m.is_representative = 1
               LIMIT ?""",
            (kp_id, limit),
        )
        if not rows:
            rows = self._read_all(
                """SELECT q.* FROM qa_pairs q
                   JOIN qa_kp_membership m ON q.id = m.qa_id
                   WHERE m.kp_id = ?
                   ORDER BY m.membership_strength DESC LIMIT ?""",
                (kp_id, limit),
            )
        return rows

    def get_kp_qas(self, kp_id: str) -> list[dict]:
        return self._read_all(
            """SELECT q.*, m.membership_strength, m.is_representative
               FROM qa_pairs q
               JOIN qa_kp_membership m ON q.id = m.qa_id
               WHERE m.kp_id = ?""",
            (kp_id,),
        )

    # -- KP Edges --

    def upsert_edge(self, spec: KpEdgeSpec):
        self._write(
            """INSERT OR REPLACE INTO kp_edges
               (source_kp, target_kp, edge_type, retrieval_weight,
                semantic_weight, sequential_weight, learning_path_weight,
                combined_strength, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (spec.source_kp, spec.target_kp, spec.edge_type, spec.retrieval_weight,
             spec.semantic_weight, spec.sequential_weight, spec.learning_path_weight,
             spec.combined_strength, spec.confidence),
        )

    def replace_edge(self, spec: KpEdgeSpec):
        """Atomically DELETE old edges for (source, target) then INSERT fused edge."""
        self._write(
            "DELETE FROM kp_edges WHERE source_kp = ? AND target_kp = ?",
            (spec.source_kp, spec.target_kp),
        )
        self._write(
            """INSERT INTO kp_edges
               (source_kp, target_kp, edge_type, retrieval_weight,
                semantic_weight, sequential_weight, learning_path_weight,
                combined_strength, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (spec.source_kp, spec.target_kp, spec.edge_type, spec.retrieval_weight,
             spec.semantic_weight, spec.sequential_weight, spec.learning_path_weight,
             spec.combined_strength, spec.confidence),
        )

    def get_edges(self, kp_id: str = None) -> list[dict]:
        if kp_id:
            return self._read_all(
                "SELECT * FROM kp_edges WHERE source_kp = ? OR target_kp = ?",
                (kp_id, kp_id),
            )
        return self._qb.get_all("kp_edges")

    def get_graph(self) -> dict:
        rows = self._read_all(
            "SELECT source_kp, target_kp, edge_type, confidence FROM kp_edges"
        )
        graph: dict[str, dict] = {}
        for r in rows:
            s, t = r["source_kp"], r["target_kp"]
            for k in (s, t):
                if k not in graph:
                    graph[k] = {"prerequisites": [], "dependents": []}
            if r["edge_type"] in ("prerequisite", "corequisite"):
                graph[t]["prerequisites"].append(
                    {"kp": s, "type": r["edge_type"], "confidence": r["confidence"]}
                )
                graph[s]["dependents"].append(
                    {"kp": t, "type": r["edge_type"], "confidence": r["confidence"]}
                )
        return graph

    def get_edge_counts(self) -> list[dict]:
        return self._read_all(
            "SELECT edge_type, COUNT(*) as cnt FROM kp_edges GROUP BY edge_type"
        )

    def get_duplicate_edges(self) -> list[dict]:
        return self._read_all(
            "SELECT source_kp, target_kp, COUNT(*) as cnt "
            "FROM kp_edges GROUP BY 1, 2 HAVING COUNT(*) > 1"
        )

    # -- QA-KP Membership --

    def set_membership(self, qa_id: int, kp_id: str,
                       membership_strength: float = 1.0,
                       is_representative: bool = False):
        self._write(
            """INSERT OR REPLACE INTO qa_kp_membership
               (qa_id, kp_id, membership_strength, is_representative)
               VALUES (?, ?, ?, ?)""",
            (qa_id, kp_id, membership_strength, int(is_representative)),
        )

    def get_kp_ids_for_qa(self, qa_id: int) -> list[str]:
        rows = self._read_all(
            "SELECT kp_id FROM qa_kp_membership WHERE qa_id = ?", (qa_id,)
        )
        return [r["kp_id"] for r in rows]

    def count_members(self, kp_id: str) -> int:
        row = self._read_one(
            "SELECT COUNT(*) as cnt FROM qa_kp_membership WHERE kp_id=?", (kp_id,)
        )
        return row["cnt"] if row else 0

    def update_evidence_count(self, kp_id: str, count: int) -> None:
        self._write(
            "UPDATE knowledge_points SET evidence_count=? WHERE id=?", (count, kp_id)
        )

    def get_member_qa_ids(self, kp_id: str) -> list[int]:
        rows = self._read_all(
            "SELECT qa_id FROM qa_kp_membership WHERE kp_id=?", (kp_id,)
        )
        return [r["qa_id"] for r in rows]

    def move_memberships(self, from_kp: str, to_kp: str) -> None:
        self._write(
            "UPDATE qa_kp_membership SET kp_id=? WHERE kp_id=?", (to_kp, from_kp)
        )

    def delete_kp(self, kp_id: str) -> None:
        self._write("DELETE FROM knowledge_points WHERE id=?", (kp_id,))
