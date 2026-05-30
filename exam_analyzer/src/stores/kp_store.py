"""KpStore — domain store for knowledge_points, kp_edges, qa_kp_membership."""

from ..models import KPSpec, KpEdgeSpec


class KpStore:
    """Operations for knowledge points, edges, and QA-KP membership."""

    def __init__(self, qb: "QueryBuilder"):
        self._qb = qb
        self._mgr = qb._mgr

    # -- Knowledge Points --

    def upsert(self, spec: KPSpec):
        with self._mgr._write_lock:
            existing = self._qb.conn.execute(
                "SELECT id FROM knowledge_points WHERE id = ?", (spec.kp_id,)
            ).fetchone()
            if existing:
                self._qb.conn.execute(
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
                self._qb.conn.execute(
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
            self._mgr.maybe_commit()

    def get_all(self) -> list[dict]:
        rows = self._qb.conn.execute(
            "SELECT id, name, description, core_concept, core_detail, "
            "cohesion, evidence_count, quality FROM knowledge_points "
            "ORDER BY evidence_count DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_id(self, kp_id: str) -> dict:
        row = self._qb.get("knowledge_points", kp_id, id_col="id")
        return row if row else {}

    def get_representative_qas(self, kp_id: str, limit: int = 3) -> list[dict]:
        rows = self._qb.conn.execute(
            """SELECT q.* FROM qa_pairs q
               JOIN qa_kp_membership m ON q.id = m.qa_id
               WHERE m.kp_id = ? AND m.is_representative = 1
               LIMIT ?""",
            (kp_id, limit),
        ).fetchall()
        if not rows:
            rows = self._qb.conn.execute(
                """SELECT q.* FROM qa_pairs q
                   JOIN qa_kp_membership m ON q.id = m.qa_id
                   WHERE m.kp_id = ?
                   ORDER BY m.membership_strength DESC LIMIT ?""",
                (kp_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_kp_qas(self, kp_id: str) -> list[dict]:
        rows = self._qb.conn.execute(
            """SELECT q.*, m.membership_strength, m.is_representative
               FROM qa_pairs q
               JOIN qa_kp_membership m ON q.id = m.qa_id
               WHERE m.kp_id = ?""",
            (kp_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- KP Edges --

    def upsert_edge(self, spec: KpEdgeSpec):
        with self._mgr._write_lock:
            self._qb.conn.execute(
                """INSERT OR REPLACE INTO kp_edges
                   (source_kp, target_kp, edge_type, retrieval_weight,
                    semantic_weight, sequential_weight, learning_path_weight,
                    combined_strength, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (spec.source_kp, spec.target_kp, spec.edge_type, spec.retrieval_weight,
                 spec.semantic_weight, spec.sequential_weight, spec.learning_path_weight,
                 spec.combined_strength, spec.confidence),
            )
            self._mgr.maybe_commit()

    def get_edges(self, kp_id: str = None) -> list[dict]:
        if kp_id:
            rows = self._qb.conn.execute(
                "SELECT * FROM kp_edges WHERE source_kp = ? OR target_kp = ?",
                (kp_id, kp_id),
            ).fetchall()
        else:
            rows = self._qb.get_all("kp_edges")
        return [dict(r) for r in rows]

    def get_graph(self) -> dict:
        rows = self._qb.conn.execute(
            "SELECT source_kp, target_kp, edge_type, confidence FROM kp_edges"
        ).fetchall()
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

    # -- QA-KP Membership --

    def set_membership(self, qa_id: int, kp_id: str,
                       membership_strength: float = 1.0,
                       is_representative: bool = False):
        with self._mgr._write_lock:
            self._qb.conn.execute(
                """INSERT OR REPLACE INTO qa_kp_membership
                   (qa_id, kp_id, membership_strength, is_representative)
                   VALUES (?, ?, ?, ?)""",
                (qa_id, kp_id, membership_strength, int(is_representative)),
            )
            self._mgr.maybe_commit()
