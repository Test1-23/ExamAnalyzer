"""AnalysisStore — domain store for api_call_log, question_feedback, distillation_cache,
analysis_checkpoints, evolution_history, exam_sessions, exam_trends, command_verb_patterns,
topic_dependencies."""


class AnalysisStore:
    """Operations for analysis metadata: logs, feedback, checkpoints, evolution, exam data."""

    def __init__(self, qb: "QueryBuilder"):
        self._qb = qb
        self._mgr = qb._mgr

    # -- API Call Log --

    def log_api_call(self, stage: str, model: str, paper: str = "",
                     question_number: str = "", latency_ms: int = 0,
                     success: bool = True, output_size: int = 0):
        with self._mgr._write_lock:
            self._qb.insert("api_call_log",
                           stage=stage, model=model, paper=paper,
                           question_number=question_number, latency_ms=latency_ms,
                           success=int(success), output_size=output_size)

    # -- Question Feedback --

    def log_question_feedback(self, qa_id: int, retrieval_count: int = 0,
                              used_qa_count: int = 0, step0_topic: str = "",
                              round2_topic: str = "", covered_count: int = 0,
                              missed_count: int = 0, missed_text: str = "",
                              miss_categories: str = ""):
        with self._mgr._write_lock:
            total = covered_count + missed_count
            ratio = (covered_count / total) if total > 0 else 0.0
            match = 1 if (step0_topic and round2_topic
                          and step0_topic.lower() == round2_topic.lower()) else 0
            self._qb.insert("question_feedback",
                           qa_id=qa_id, retrieval_count=retrieval_count,
                           used_qa_count=used_qa_count, step0_topic=step0_topic,
                           round2_topic=round2_topic, topic_match=match,
                           covered_count=covered_count, missed_count=missed_count,
                           missed_text=missed_text, coverage_ratio=ratio,
                           miss_categories=miss_categories)

    def get_missed_by_topic(self, topic: str) -> list[str]:
        rows = self._qb.conn.execute(
            """SELECT f.missed_text FROM question_feedback f
               JOIN qa_pairs q ON f.qa_id = q.id
               WHERE q.topic = ? AND f.missed_text != ''""",
            (topic,),
        ).fetchall()
        missed = []
        for r in rows:
            for line in r["missed_text"].split("\n"):
                line = line.strip()
                if line:
                    missed.append(line)
        return missed

    # -- Distillation Cache --

    def get_distillation_cache(self) -> dict[str, str]:
        rows = self._qb.get_all("distillation_cache")
        return {r["topic"]: r["distilled_content"] for r in rows}

    def get_cached_topic_state(self, topic: str) -> dict | None:
        row = self._qb.get("distillation_cache", topic, id_col="topic")
        return {"qa_count": row["qa_count"], "qa_ids_hash": row["qa_ids_hash"]} if row else None

    def upsert_distillation_cache(self, topic: str, qa_count: int,
                                   qa_ids_hash: str, content: str):
        with self._mgr._write_lock:
            self._qb.conn.execute(
                """INSERT OR REPLACE INTO distillation_cache
                   (topic, qa_count, qa_ids_hash, distilled_content, distilled_at)
                   VALUES (?, ?, ?, ?, datetime('now'))""",
                (topic, qa_count, qa_ids_hash, content),
            )
            self._mgr.maybe_commit()

    def invalidate_distillation_cache(self, topic: str):
        with self._mgr._write_lock:
            self._qb.delete("distillation_cache", topic, id_col="topic")

    # -- Evolution History --

    def record_evolution(self, kp_id: str, trigger_type: str,
                         trigger_detail: str = "", old_state: str = "",
                         new_state: str = "", outcome: str = "pending"):
        with self._mgr._write_lock:
            self._qb.insert("evolution_history",
                           kp_id=kp_id, trigger_type=trigger_type,
                           trigger_detail=trigger_detail, old_state=old_state,
                           new_state=new_state, outcome=outcome)

    def get_pending_evolutions(self, kp_id: str = None) -> list[dict]:
        if kp_id:
            rows = self._qb.conn.execute(
                "SELECT * FROM evolution_history WHERE kp_id=? AND outcome='pending' "
                "ORDER BY created_at", (kp_id,)
            ).fetchall()
        else:
            rows = self._qb.conn.execute(
                "SELECT * FROM evolution_history WHERE outcome='pending' ORDER BY created_at"
            ).fetchall()
        return [dict(r) for r in rows]

    # -- Checkpoints --

    def checkpoint(self, task_name: str, qa_count: int, status: str = "completed"):
        with self._mgr._write_lock:
            self._qb.conn.execute(
                """INSERT OR REPLACE INTO analysis_checkpoints
                   (task_name, qa_count_at_run, completed_at, status)
                   VALUES (?, ?, datetime('now'), ?)""",
                (task_name, qa_count, status),
            )
            self._mgr.maybe_commit()

    def get_checkpoint(self, task_name: str) -> dict:
        row = self._qb.get("analysis_checkpoints", task_name, id_col="task_name")
        return row if row else {}

    def clear_checkpoint(self, task_name: str):
        with self._mgr._write_lock:
            self._qb.delete("analysis_checkpoints", task_name, id_col="task_name")

    # -- Exam Sessions --

    def get_exam_stats(self, topic: str) -> list[dict]:
        rows = self._qb.conn.execute(
            """SELECT s.year, s.season, COUNT(*) as cnt
               FROM qa_pairs q JOIN exam_sessions s ON q.session_id = s.id
               WHERE q.topic = ? GROUP BY s.year, s.season
               ORDER BY s.year DESC, s.season""",
            (topic,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Exam Trends --

    def upsert_exam_trend(self, kp_id: str, year: int, season: str,
                          occurrence_count: int = 0, avg_difficulty: str = "",
                          trend_summary: str = ""):
        with self._mgr._write_lock:
            self._qb.conn.execute(
                """INSERT OR REPLACE INTO exam_trends
                   (kp_id, year, season, occurrence_count, avg_difficulty, trend_summary)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (kp_id, year, season, occurrence_count, avg_difficulty, trend_summary),
            )
            self._mgr.maybe_commit()

    def get_exam_trends(self, kp_id: str = None) -> list[dict]:
        if kp_id:
            rows = self._qb.conn.execute(
                "SELECT * FROM exam_trends WHERE kp_id = ? ORDER BY year, season",
                (kp_id,),
            ).fetchall()
        else:
            rows = self._qb.get_all("exam_trends", order_by="kp_id, year, season")
        return [dict(r) for r in rows]

    # -- Command Verb Patterns --

    def upsert_verb_pattern(self, verb: str, sample_count: int = 0,
                            avg_answer_length: float = None,
                            median_answer_length: float = None,
                            bullet_ratio: float = None,
                            avg_bullet_count: float = None,
                            avg_miss_rate: float = None,
                            common_missed_patterns: str = "",
                            pattern_summary: str = "",
                            topic_specific_patterns: str = "",
                            verb_family: str = ""):
        with self._mgr._write_lock:
            self._qb.conn.execute(
                """INSERT OR REPLACE INTO command_verb_patterns
                   (verb, sample_count, avg_answer_length, median_answer_length,
                    bullet_ratio, avg_bullet_count, avg_miss_rate,
                    common_missed_patterns, pattern_summary,
                    topic_specific_patterns, verb_family, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                (verb, sample_count, avg_answer_length, median_answer_length,
                 bullet_ratio, avg_bullet_count, avg_miss_rate,
                 common_missed_patterns, pattern_summary,
                 topic_specific_patterns, verb_family),
            )
            self._mgr.maybe_commit()

    def get_verb_patterns(self) -> list[dict]:
        return self._qb.get_all("command_verb_patterns", order_by="sample_count DESC")

    # -- Topic Dependencies --

    def insert_dependency(self, prerequisite: str, dependent: str,
                          evidence_score: int = None, evidence_reason: str = "",
                          relationship_type: str = "prerequisite",
                          topic_link_count: int = 0, embedding_cos: float = None,
                          confidence: str = "low", validated_by: str = "flash"):
        with self._mgr._write_lock:
            existing = self._qb.conn.execute(
                "SELECT first_seen_at FROM topic_dependencies "
                "WHERE prerequisite = ? AND dependent = ?",
                (prerequisite, dependent),
            ).fetchone()
            if existing:
                self._qb.conn.execute(
                    """UPDATE topic_dependencies SET
                       evidence_score = ?, evidence_reason = ?,
                       relationship_type = ?, topic_link_count = ?,
                       embedding_cos = ?, confidence = ?,
                       validated_at = datetime('now'),
                       last_validated_at = datetime('now'),
                       validated_by = ?
                       WHERE prerequisite = ? AND dependent = ?""",
                    (evidence_score, evidence_reason, relationship_type,
                     topic_link_count, embedding_cos, confidence, validated_by,
                     prerequisite, dependent),
                )
            else:
                self._qb.conn.execute(
                    """INSERT INTO topic_dependencies
                       (prerequisite, dependent, evidence_score, evidence_reason,
                        relationship_type, topic_link_count, embedding_cos,
                        confidence, validated_at, first_seen_at, last_validated_at, validated_by)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'),
                                datetime('now'), ?)""",
                    (prerequisite, dependent, evidence_score, evidence_reason,
                     relationship_type, topic_link_count, embedding_cos,
                     confidence, validated_by),
                )
            self._mgr.maybe_commit()

    def get_dependencies(self, confidence: str = None) -> list[dict]:
        if confidence:
            rows = self._qb.conn.execute(
                "SELECT * FROM topic_dependencies WHERE confidence = ? ORDER BY prerequisite",
                (confidence,),
            ).fetchall()
        else:
            rows = self._qb.get_all("topic_dependencies", order_by="prerequisite")
        return [dict(r) for r in rows]

    def get_dependency_graph(self) -> dict:
        rows = self._qb.conn.execute(
            "SELECT prerequisite, dependent, relationship_type, confidence FROM topic_dependencies"
        ).fetchall()
        graph: dict[str, dict] = {}
        for r in rows:
            pre, dep = r["prerequisite"], r["dependent"]
            for t in (pre, dep):
                if t not in graph:
                    graph[t] = {"prerequisites": [], "dependents": []}
            graph[dep]["prerequisites"].append({
                "topic": pre, "type": r["relationship_type"], "confidence": r["confidence"]
            })
            graph[pre]["dependents"].append({
                "topic": dep, "type": r["relationship_type"], "confidence": r["confidence"]
            })
        return graph
