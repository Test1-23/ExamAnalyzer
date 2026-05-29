"""TopicStore — domain store for dynamic_topics, fragment_membership, topic_links, topic_difficulty."""


class TopicStore:
    """Operations for dynamic topics, fragment memberships, and topic links."""

    def __init__(self, qb: "QueryBuilder"):
        self._qb = qb
        self._mgr = qb._mgr

    # -- Dynamic Topics --

    def upsert(self, topic_id: str, name: str = "", quality: str = "embryonic"):
        with self._mgr._write_lock:
            self._qb.conn.execute(
                """INSERT OR REPLACE INTO dynamic_topics
                   (topic_id, name, quality, last_evolved_at)
                   VALUES (?, ?, ?, datetime('now'))""",
                (topic_id, name, quality),
            )
            self._mgr.maybe_commit()

    def update_stats(self, topic_id: str, mass: int, cohesion: float, stability: float):
        with self._mgr._write_lock:
            self._qb.conn.execute(
                """UPDATE dynamic_topics SET mass=?, cohesion=?, stability=?,
                   last_evolved_at=datetime('now') WHERE topic_id=?""",
                (mass, cohesion, stability, topic_id),
            )
            self._mgr.maybe_commit()

    def set_kp(self, topic_id: str, kp_concept: str, kp_detail: str):
        with self._mgr._write_lock:
            self._qb.conn.execute(
                """UPDATE dynamic_topics SET kp_concept=?, kp_detail=?,
                   quality='stable', last_evolved_at=datetime('now')
                   WHERE topic_id=?""",
                (kp_concept, kp_detail, topic_id),
            )
            self._mgr.maybe_commit()

    def get_stable(self) -> list[dict]:
        return self._qb.get_where("dynamic_topics", quality="stable")

    def get_stable_kps(self) -> list[dict]:
        rows = self._qb.conn.execute(
            "SELECT name, kp_concept, kp_detail, mass "
            "FROM dynamic_topics WHERE quality='stable' AND kp_concept != '' "
            "ORDER BY mass DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_helped_questions(self, topic_id: str) -> set:
        rows = self._qb.conn.execute(
            """SELECT DISTINCT fhm.helped_qa_id
               FROM fragment_help_map fhm
               JOIN fragment_membership fm ON fhm.fragment_id = fm.fragment_id
               WHERE fm.topic_id = ?""",
            (topic_id,),
        ).fetchall()
        return {r["helped_qa_id"] for r in rows}

    # -- Fragment Membership --

    def set_fragment_membership(self, fragment_id: str, topic_id: str, loyalty: float = 0.5):
        with self._mgr._write_lock:
            self._qb.conn.execute(
                """INSERT OR REPLACE INTO fragment_membership
                   (fragment_id, topic_id, loyalty, joined_at, previous_topic_id)
                   VALUES (?, ?, ?, datetime('now'),
                    (SELECT topic_id FROM fragment_membership WHERE fragment_id=?))""",
                (fragment_id, topic_id, loyalty, fragment_id),
            )
            self._mgr.maybe_commit()

    def get_fragments(self, topic_id: str) -> list[str]:
        rows = self._qb.get_where("fragment_membership", topic_id=topic_id)
        return [r["fragment_id"] for r in rows]

    # -- Topic Links --

    def upsert_link(self, src_topic: str, dst_topic: str, count: int = 1):
        if not src_topic or not dst_topic or src_topic == dst_topic:
            return
        with self._mgr._write_lock:
            self._qb.conn.execute(
                """INSERT INTO topic_links (src_topic, dst_topic, count)
                   VALUES (?, ?, ?)
                   ON CONFLICT(src_topic, dst_topic) DO UPDATE SET count = count + ?""",
                (src_topic, dst_topic, count, count),
            )
            self._mgr.maybe_commit()

    def get_links(self) -> dict:
        rows = self._qb.get_all("topic_links")
        return {(r["src_topic"], r["dst_topic"]): r["count"] for r in rows}

    def get_adjacent_topics(self, topic: str, limit: int = 5) -> list[str]:
        rows = self._qb.conn.execute(
            "SELECT DISTINCT dst_topic FROM topic_links WHERE src_topic=? UNION "
            "SELECT DISTINCT src_topic FROM topic_links WHERE dst_topic=?",
            (topic, topic),
        ).fetchall()
        return [r["dst_topic"] for r in rows[:limit]]

    # -- Topic Difficulty --

    def upsert_difficulty(self, topic: str, qa_count: int = 0,
                          basic_count: int = 0, intermediate_count: int = 0,
                          advanced_count: int = 0, mode_difficulty: str = "",
                          avg_miss_rate: float = None,
                          difficulty_spread: bool = False,
                          assessment_method: str = "hybrid"):
        with self._mgr._write_lock:
            self._qb.conn.execute(
                """INSERT OR REPLACE INTO topic_difficulty
                   (topic, qa_count, basic_count, intermediate_count, advanced_count,
                    mode_difficulty, avg_miss_rate, difficulty_spread,
                    assessed_at, assessment_method)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)""",
                (topic, qa_count, basic_count, intermediate_count, advanced_count,
                 mode_difficulty, avg_miss_rate, int(difficulty_spread),
                 assessment_method),
            )
            self._mgr.maybe_commit()

    def get_difficulty(self, topic: str = None) -> list[dict]:
        if topic:
            return self._qb.get_where("topic_difficulty", topic=topic)
        return self._qb.get_all("topic_difficulty", order_by="mode_difficulty, topic")

    # -- Topic Dependencies --

    def get_direct_prerequisites(self, topic: str) -> list[dict]:
        rows = self._qb.conn.execute(
            """SELECT prerequisite, evidence_score, confidence, relationship_type
               FROM topic_dependencies WHERE dependent = ?""",
            (topic,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_transitive_prerequisites(self, topic: str, max_depth: int = 5) -> list[str]:
        seen = set()
        frontier = [topic]
        for _ in range(max_depth):
            if not frontier:
                break
            next_frontier = []
            for t in frontier:
                rows = self._qb.conn.execute(
                    "SELECT prerequisite FROM topic_dependencies WHERE dependent = ?", (t,)
                ).fetchall()
                for r in rows:
                    pre = r["prerequisite"]
                    if pre not in seen:
                        seen.add(pre)
                        next_frontier.append(pre)
            frontier = next_frontier
        return list(seen)
