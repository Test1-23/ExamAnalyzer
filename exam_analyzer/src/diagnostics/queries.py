"""DiagnosticQueries — reusable analytical query patterns for diagnostics/offline.

These queries are cross-table JOINs, full-table pre-loads, and aggregations
that don't fit into single-table Store CRUD methods.  They serve as *shared*
building blocks — each diagnostic module factors out its raw SQL into a named
method here only when the same query pattern is used by at least two modules.

Design principle: this class does NOT aim to cover every raw SQL in the
diagnostics/offline packages.  One-off analytical queries specific to a single
module stay in that module.  The purpose is to eliminate *duplicated* raw SQL
and give the most common cross-table pre-loads explicit contracts.

Tables accessed (all read-only unless noted):
    fragment_help_map, fragment_membership, ms_fragments, dynamic_topics,
    knowledge_points, qa_kp_membership, kp_vectors, topic_vectors,
    qa_pairs, question_feedback, student_knowledge_state, confusion_events
"""

from ..knowledge_base import QADatabase


class DiagnosticQueries:
    """Named analytical queries shared across diagnostic/offline modules."""

    def __init__(self, db: QADatabase):
        self._db = db

    # -- Fragment/Help pre-loads (used by migration.py + cascade.py) --

    def load_fragment_helps(self) -> dict[str, set]:
        """Pre-load entire fragment_help_map → {fragment_id: {helped_qa_id, ...}}."""
        rows = self._db.conn.execute(
            "SELECT fragment_id, helped_qa_id FROM fragment_help_map"
        ).fetchall()
        result: dict[str, set] = {}
        for r in rows:
            result.setdefault(r["fragment_id"], set()).add(r["helped_qa_id"])
        return result

    def load_fragment_topic_map(self) -> dict[str, str]:
        """fragment_membership ⋈ ms_fragments → {fragment_id: topic_id}."""
        rows = self._db.conn.execute(
            "SELECT fm.fragment_id, fm.topic_id AS current_topic "
            "FROM fragment_membership fm "
            "JOIN ms_fragments mf ON fm.fragment_id = mf.point_id"
        ).fetchall()
        return {r["fragment_id"]: r["current_topic"] for r in rows}

    def load_topic_helps(self) -> dict[str, set]:
        """Pre-compute {topic_id: {helped_qa_id}} from fragment help data.

        Used by migration.py and cascade.py for topic overlap/loyalty analysis.
        """
        help_rows = self._db.conn.execute(
            "SELECT fragment_id, helped_qa_id FROM fragment_help_map"
        ).fetchall()
        topic_rows = self._db.conn.execute(
            "SELECT fm.fragment_id, fm.topic_id AS current_topic "
            "FROM fragment_membership fm "
            "JOIN ms_fragments mf ON fm.fragment_id = mf.point_id"
        ).fetchall()
        frag_to_topic = {r["fragment_id"]: r["current_topic"] for r in topic_rows}
        topic_helps: dict[str, set] = {}
        for r in help_rows:
            tid = frag_to_topic.get(r["fragment_id"])
            if tid:
                topic_helps.setdefault(tid, set()).add(r["helped_qa_id"])
        return topic_helps

    # -- Topic state (used by migration.py + cascade.py) --

    def get_active_topics(self) -> list[dict]:
        """topic_id, mass FROM dynamic_topics WHERE mass >= 2 AND quality != 'dissolved'."""
        return [
            dict(r) for r in self._db.conn.execute(
                "SELECT topic_id, mass FROM dynamic_topics "
                "WHERE mass >= 2 AND quality != 'dissolved'"
            ).fetchall()
        ]

    def get_all_topic_ids(self) -> list[str]:
        """All topic_ids from dynamic_topics (for iteration)."""
        return [
            r["topic_id"] for r in self._db.conn.execute(
                "SELECT topic_id FROM dynamic_topics"
            ).fetchall()
        ]

    def get_topic_mass(self, topic_id: str) -> int:
        """Fragment count for a single topic."""
        row = self._db.conn.execute(
            "SELECT COUNT(*) AS cnt FROM fragment_membership WHERE topic_id = ?",
            (topic_id,),
        ).fetchone()
        return row["cnt"] if row else 0

    def count_topic_churn(self, topic_id: str) -> int:
        """Count fragments that migrated *away* from this topic."""
        row = self._db.conn.execute(
            "SELECT COUNT(*) AS cnt FROM fragment_membership "
            "WHERE topic_id = ? AND previous_topic_id IS NOT NULL "
            "AND previous_topic_id != ?",
            (topic_id, topic_id),
        ).fetchone()
        return row["cnt"] if row else 0

    # -- Topic vectors (used by cascade.py + dependencies.py) --

    def load_topic_vectors(self) -> dict[str, "np.ndarray"]:
        """{topic_id: vector} — all topic vectors as numpy arrays."""
        import numpy as np
        rows = self._db.conn.execute(
            "SELECT topic_id, vector FROM topic_vectors"
        ).fetchall()
        return {
            r["topic_id"]: np.frombuffer(r["vector"], dtype=np.float32)
            for r in rows if r["vector"] is not None
        }

    def load_kp_vectors(self, exclude_disputed: bool = True) -> dict[str, "np.ndarray"]:
        """{kp_id: vector} — KP vectors, optionally excluding disputed."""
        import numpy as np
        qual_filter = "WHERE quality != 'disputed'" if exclude_disputed else ""
        rows = self._db.conn.execute(
            f"SELECT id, vector FROM kp_vectors {qual_filter}"
        ).fetchall()
        return {
            r["id"]: np.frombuffer(r["vector"], dtype=np.float32)
            for r in rows if r["vector"] is not None
        }

    def load_topic_vectors_for_topics(self, topic_ids: list[str]) -> dict[str, "np.ndarray"]:
        """{topic_id: vector} for a subset of topic IDs only."""
        import numpy as np
        if not topic_ids:
            return {}
        ph = ",".join("?" * len(topic_ids))
        rows = self._db.conn.execute(
            f"SELECT topic_id, vector FROM topic_vectors WHERE topic_id IN ({ph})",
            topic_ids,
        ).fetchall()
        return {
            r["topic_id"]: np.frombuffer(r["vector"], dtype=np.float32)
            for r in rows if r["vector"] is not None
        }

    # -- Student confusion & mastery (used by student.py + cross_paper.py) --

    def get_confusion_counts(self) -> dict[str, int]:
        """{topic: count} — confusion events grouped by topic."""
        rows = self._db.conn.execute(
            "SELECT topic, COUNT(*) AS cnt FROM confusion_events "
            "GROUP BY topic HAVING COUNT(*) >= 3"
        ).fetchall()
        return {r["topic"]: r["cnt"] for r in rows}

    def get_mastery_topics(self, min_count: int = 3) -> list[dict]:
        """Topics where >= min_count students have reached 'mastered' state."""
        return [
            dict(r) for r in self._db.conn.execute(
                "SELECT topic, COUNT(*) AS cnt, state "
                "FROM student_knowledge_state "
                "WHERE state = 'mastered' "
                "GROUP BY topic"
            ).fetchall()
            if r["cnt"] >= min_count
        ]

    # -- Pre-migration state (used by migration.py) --

    def get_total_help_count(self) -> int:
        """Total rows in fragment_help_map — used for threshold calculation."""
        row = self._db.conn.execute(
            "SELECT COUNT(*) AS cnt FROM fragment_help_map"
        ).fetchone()
        return row["cnt"] if row else 0

    def get_topic_qa_distribution(self) -> dict[str, int]:
        """{topic_id: qa_count} — how many QAs are associated with each topic."""
        rows = self._db.conn.execute(
            "SELECT topic, COUNT(*) AS cnt FROM qa_pairs "
            "WHERE topic != '' AND topic != '(uncategorized)' "
            "GROUP BY topic"
        ).fetchall()
        return {r["topic"]: r["cnt"] for r in rows}

    # -- KP member helpers (used by cascade.py) --

    def get_kp_member_qas(self, kp_id: str) -> list[dict]:
        """QA rows belonging to a specific KP."""
        return [
            dict(r) for r in self._db.conn.execute(
                "SELECT q.*, m.membership_strength, m.is_representative "
                "FROM qa_pairs q "
                "JOIN qa_kp_membership m ON q.id = m.qa_id "
                "WHERE m.kp_id = ?",
                (kp_id,),
            ).fetchall()
        ]
