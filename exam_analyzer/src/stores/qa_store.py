"""QaStore — domain store for qa_pairs table."""

from typing import Optional
from ..constants import SQLITE_PARAM_CHUNK
from .base import BaseStore


class QaStore(BaseStore):
    """CRUD operations for the ``qa_pairs`` table."""

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, qa_id: int) -> Optional[dict]:
        return self._qb.get("qa_pairs", qa_id)

    def get_all(self) -> list[dict]:
        return self._qb.get_all("qa_pairs", order_by="id")

    def get_by_ids(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        row_map = {}
        for i in range(0, len(ids), SQLITE_PARAM_CHUNK):
            chunk = ids[i:i + SQLITE_PARAM_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = self._read_all(
                "SELECT * FROM qa_pairs WHERE id IN (%s)" % placeholders, chunk
            )
            for r in rows:
                row_map[r["id"]] = r
        return [row_map[i] for i in ids if i in row_map]

    def get_by_topic(self, topic: str, difficulty: str = "",
                     order_by: str = "is_representative DESC, success_count DESC",
                     limit: int = 0) -> list[dict]:
        extra = "AND difficulty_estimate = ?" if difficulty else ""
        params = (topic,) + ((difficulty,) if difficulty else ())
        sql = ("SELECT * FROM qa_pairs WHERE topic = ? %s ORDER BY %s"
               % (extra, order_by))
        if limit:
            sql += " LIMIT %d" % limit
        return self._read_all(sql, params)

    def get_topic_groups(self) -> dict[str, list[dict]]:
        from collections import defaultdict
        groups: dict[str, list[dict]] = defaultdict(list)
        for qa in self.get_all():
            topic = qa.get("topic", "") or "(uncategorized)"
            groups[topic].append(qa)
        return dict(groups)

    def get_topic_answer_texts(self) -> dict[str, str]:
        rows = self._read_all(
            "SELECT topic, answer_text FROM qa_pairs "
            "WHERE topic != '' AND topic != '(uncategorized)'"
        )
        texts: dict[str, list[str]] = {}
        for r in rows:
            texts.setdefault(r["topic"], []).append(r["answer_text"])
        return {t: " ".join(parts)[:2000] for t, parts in texts.items()}

    def count(self) -> int:
        return self._qb.count("qa_pairs")

    def get_all_weights(self) -> dict[int, dict]:
        rows = self._read_all(
            "SELECT id, success_count, total_attempts FROM qa_pairs"
        )
        result = {}
        for r in rows:
            s, t = r["success_count"], r["total_attempts"]
            mean = (s + 1) / (t + 2)
            if t > 0:
                a, b = s + 1, t - s + 1
                n_post = a + b
                p_hat = a / n_post
                z = 1.282
                z2 = z * z
                denom = 1.0 + z2 / n_post
                center = (p_hat + z2 / (2.0 * n_post)) / denom
                margin = z * ((p_hat * (1.0 - p_hat) + z2 / (4.0 * n_post)) / n_post) ** 0.5 / denom
                lb = max(0.0, center - margin)
            else:
                lb = 0.0
            result[r["id"]] = {"mean": round(mean, 3), "lower_bound": round(lb, 3), "total": t}
        return result

    def get_effective_miss_rate(self, qa_id: int) -> Optional[float]:
        import json
        row = self._read_one(
            "SELECT covered_count, missed_count, miss_categories "
            "FROM question_feedback WHERE qa_id = ?", (qa_id,)
        )
        if not row:
            return None
        total = row["covered_count"] + row["missed_count"]
        if total == 0:
            return None
        cats_json = row["miss_categories"] or ""
        if cats_json:
            try:
                cats = json.loads(cats_json)
                effective = cats.get("knowledge_gap", 0) + cats.get("insufficient_detail", 0) * 0.5
                return effective / total
            except (json.JSONDecodeError, TypeError):
                pass
        return row["missed_count"] / total

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert(self, question_text: str, answer_text: str,
               topic: str = "", paper: str = "",
               question_number: str = "", parent_question: str = "",
               knowledge_summary: str = "") -> int:
        with self._write_locked():
            existing = self._read_one(
                "SELECT id FROM qa_pairs WHERE question_text = ? AND answer_text = ? LIMIT 1",
                (question_text, answer_text),
            )
            if existing:
                return existing["id"]
            qa_id = self._qb_insert(
                "qa_pairs",
                question_text=question_text,
                answer_text=answer_text,
                knowledge_summary=knowledge_summary,
                topic=topic,
                paper=paper,
                question_number=question_number,
                parent_question=parent_question,
            )
            return qa_id

    def record_attempt(self, qa_id: int, success: bool, reason: str = ""):
        if success:
            self._write(
                "UPDATE qa_pairs SET success_count=success_count+1, "
                "total_attempts=total_attempts+1, last_failure_reason='' WHERE id=?",
                (qa_id,),
            )
        else:
            self._write(
                "UPDATE qa_pairs SET total_attempts=total_attempts+1, "
                "last_failure_reason=? WHERE id=?",
                (reason, qa_id),
            )

    def update_topic(self, qa_id: int, topic: str):
        self._qb_update("qa_pairs", qa_id, topic=topic)

    def rename_topic(self, new_topic: str, old_topic: str) -> int:
        rows = self._write(
            "UPDATE qa_pairs SET topic=? WHERE topic=?", (new_topic, old_topic)
        )
        return rows.rowcount

    def get_qa_difficulty(self, qa_id: int) -> str:
        row = self._read_one(
            "SELECT difficulty_estimate FROM qa_pairs WHERE id = ?", (qa_id,)
        )
        return row["difficulty_estimate"] if row else ""

    def get_verb_for_qa(self, qa_id: int) -> dict:
        row = self._read_one(
            "SELECT command_verb, command_verb_secondary, command_verb_inferred "
            "FROM qa_pairs WHERE id = ?", (qa_id,)
        )
        return row if row else {}

    # ------------------------------------------------------------------
    # Data integrity helpers
    # ------------------------------------------------------------------

    def fix_inconsistent_counters(self) -> int:
        row = self._read_one(
            "SELECT COUNT(*) as cnt FROM qa_pairs "
            "WHERE success_count > total_attempts"
        )
        if not row or row["cnt"] == 0:
            return 0
        cur = self._write(
            "UPDATE qa_pairs SET success_count = total_attempts "
            "WHERE success_count > total_attempts"
        )
        return cur.rowcount

    def set_failure_reason(self, qa_id: int, reason: str) -> None:
        self._write(
            "UPDATE qa_pairs SET last_failure_reason=? WHERE id=?", (reason, qa_id)
        )

    def set_representative(self, qa_id: int) -> None:
        self._write(
            "UPDATE qa_pairs SET is_representative = 1 WHERE id = ?", (qa_id,)
        )

    def set_cross_topic(self, qa_id: int) -> None:
        self._write(
            "UPDATE qa_pairs SET is_cross_topic = 1 WHERE id = ?", (qa_id,)
        )
