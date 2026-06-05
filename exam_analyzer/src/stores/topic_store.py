"""TopicStore — domain store for dynamic_topics, fragment_membership, topic_links, topic_difficulty."""

from .base import BaseStore


class TopicStore(BaseStore):
    """Operations for dynamic topics, fragment memberships, and topic links."""

    # -- Dynamic Topics --

    def upsert(self, topic_id: str, name: str = "", quality: str = "embryonic"):
        self._write(
            """INSERT OR REPLACE INTO dynamic_topics
               (topic_id, name, quality, last_evolved_at)
               VALUES (?, ?, ?, datetime('now'))""",
            (topic_id, name, quality),
        )

    def update_stats(self, topic_id: str, mass: int, cohesion: float, stability: float):
        self._write(
            """UPDATE dynamic_topics SET mass=?, cohesion=?, stability=?,
               last_evolved_at=datetime('now') WHERE topic_id=?""",
            (mass, cohesion, stability, topic_id),
        )

    def set_kp(self, topic_id: str, kp_concept: str, kp_detail: str):
        self._write(
            """UPDATE dynamic_topics SET kp_concept=?, kp_detail=?,
               quality='stable', last_evolved_at=datetime('now')
               WHERE topic_id=?""",
            (kp_concept, kp_detail, topic_id),
        )

    def get_stable(self) -> list[dict]:
        return self._qb.get_where("dynamic_topics", quality="stable")

    def get_stable_kps(self) -> list[dict]:
        return self._read_all(
            "SELECT name, kp_concept, kp_detail, mass "
            "FROM dynamic_topics WHERE quality='stable' AND kp_concept != '' "
            "ORDER BY mass DESC"
        )

    # -- Fragment Membership --

    def set_fragment_membership(self, fragment_id: str, topic_id: str, loyalty: float = 0.5):
        self._write(
            """INSERT OR REPLACE INTO fragment_membership
               (fragment_id, topic_id, loyalty, joined_at, previous_topic_id)
               VALUES (?, ?, ?, datetime('now'),
                (SELECT topic_id FROM fragment_membership WHERE fragment_id=?))""",
            (fragment_id, topic_id, loyalty, fragment_id),
        )

    def get_fragments(self, topic_id: str) -> list[str]:
        rows = self._qb.get_where("fragment_membership", topic_id=topic_id)
        return [r["fragment_id"] for r in rows]

    def get_adjacent_topics(self, topic: str) -> list[str]:
        rows = self._read_all(
            "SELECT DISTINCT dst_topic FROM topic_links WHERE src_topic=? UNION "
            "SELECT DISTINCT src_topic FROM topic_links WHERE dst_topic=?",
            (topic, topic),
        )
        return [r["dst_topic"] for r in rows]

    def get_linked_mask(self, query_topic: str, candidates: list[str]) -> set[str]:
        if not candidates:
            return set()
        ph = ",".join("?" * len(candidates))
        params = (query_topic,) + tuple(candidates) + tuple(candidates) + (query_topic,)
        rows = self._read_all(
            f"SELECT DISTINCT dst_topic FROM topic_links "
            f"WHERE src_topic=? AND dst_topic IN ({ph}) UNION "
            f"SELECT DISTINCT src_topic FROM topic_links "
            f"WHERE src_topic IN ({ph}) AND dst_topic=?",
            params,
        )
        return {r["dst_topic"] for r in rows}

    # -- Topic Links --

    def upsert_link(self, src_topic: str, dst_topic: str, count: int = 1):
        if not src_topic or not dst_topic or src_topic == dst_topic:
            return
        self._write(
            """INSERT INTO topic_links (src_topic, dst_topic, count)
               VALUES (?, ?, ?)
               ON CONFLICT(src_topic, dst_topic) DO UPDATE SET count = count + ?""",
            (src_topic, dst_topic, count, count),
        )

    def get_links(self) -> dict:
        rows = self._qb.get_all("topic_links")
        return {(r["src_topic"], r["dst_topic"]): r["count"] for r in rows}

    # -- Topic Difficulty --

    def upsert_difficulty(self, topic: str, qa_count: int = 0,
                          basic_count: int = 0, intermediate_count: int = 0,
                          advanced_count: int = 0, mode_difficulty: str = "",
                          avg_miss_rate: float = None,
                          difficulty_spread: bool = False,
                          assessment_method: str = "hybrid"):
        self._write(
            """INSERT OR REPLACE INTO topic_difficulty
               (topic, qa_count, basic_count, intermediate_count, advanced_count,
                mode_difficulty, avg_miss_rate, difficulty_spread,
                assessed_at, assessment_method)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)""",
            (topic, qa_count, basic_count, intermediate_count, advanced_count,
             mode_difficulty, avg_miss_rate, int(difficulty_spread),
             assessment_method),
        )

    def get_difficulty(self, topic: str = None) -> list[dict]:
        if topic:
            return self._qb.get_where("topic_difficulty", topic=topic)
        return self._qb.get_all("topic_difficulty", order_by="mode_difficulty, topic")

    def get_by_quality(self, quality: str) -> list[dict]:
        return self._qb.get_where("dynamic_topics", quality=quality)

    def get_by_qualities(self, qualities: list[str]) -> list[dict]:
        if not qualities:
            return []
        ph = ",".join("?" * len(qualities))
        return self._read_all(
            "SELECT * FROM dynamic_topics WHERE quality IN (%s)" % ph, qualities
        )

    def get_all_links(self) -> list[dict]:
        return self._qb.get_all("topic_links")

    def replace_all_links(self, links: list[tuple]) -> None:
        self._write("DELETE FROM topic_links")
        for src, dst, total in links:
            self._write(
                "INSERT INTO topic_links (src_topic, dst_topic, count) VALUES (?, ?, ?)",
                (src, dst, total),
            )
