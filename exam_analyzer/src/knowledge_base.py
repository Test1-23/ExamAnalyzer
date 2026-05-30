"""SQLite QA knowledge base + embedding retrieval.

QADatabase: stores question-answer pairs.
QARetriever: embedding-based similarity search over the QA database.
"""

import sqlite3
import threading
import numpy as np
from typing import List, Optional

from .connection_manager import ConnectionMgr
from .embedding_cluster import _get_model, MODEL_MAP, _detect_language
from .logger import get_logger
from .models import KPSpec, VerbPatternSpec, KpEdgeSpec, DependencySpec
from .query_builder import QueryBuilder

_log = get_logger()

from .constants import (
    SQLITE_PARAM_CHUNK,
    CHANNEL_A_RECALL, BEHAVIOR_CHUNK,
    WEIGHT_EMBEDDING, WEIGHT_TOPIC, WEIGHT_BEHAVIOR, WEIGHT_KEYWORD,
)


# ============================================================
# Helpers
# ============================================================

def make_topic_id(topic: str) -> str:
    """Sanitize a topic name into a stable ID: ``topic_<sanitized_name>``."""
    sanitized = topic.replace(" ", "_").replace("/", "_")
    return f"topic_{sanitized}"


# ============================================================
# QADatabase
# ============================================================

class QADatabase:
    """Stores and retrieves question-answer pairs with embedding support."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db = ConnectionMgr(db_path)
        self._qb = QueryBuilder(self._db)

        from .stores.qa_store import QaStore
        from .stores.kp_store import KpStore
        from .stores.topic_store import TopicStore
        from .stores.fragment_store import FragmentStore
        from .stores.chat_store import ChatStore
        from .stores.student_store import StudentStore
        from .stores.analysis_store import AnalysisStore
        from .stores.vector_store import VectorStore

        self.qa = QaStore(self._qb)
        self.kp = KpStore(self._qb)
        self.topic = TopicStore(self._qb)
        self.fragment = FragmentStore(self._qb)
        self.chat = ChatStore(self._qb)
        self.student = StudentStore(self._qb)
        self.analysis = AnalysisStore(self._qb)
        self.vector = VectorStore(self._qb)

    @property
    def conn(self) -> sqlite3.Connection:
        return self._db.conn

    @property
    def _write_lock(self):
        return self._db._write_lock

    def _commit(self):
        """Commit only when NOT inside a transaction. Safe to call inside ``with db.transaction()``."""
        self._db.maybe_commit()

    def transaction(self):
        """Context manager for atomic multi-step writes. See ConnectionMgr.transaction."""
        return self._db.transaction()

    def update_qa_topic(self, qa_id: int, topic: str):
        self.qa.update_topic(qa_id, topic)

    def rename_topic(self, new_topic: str, old_topic: str) -> int:
        return self.qa.rename_topic(new_topic, old_topic)

    def insert(self, question_text: str, answer_text: str,
               topic: str = "", paper: str = "",
               question_number: str = "",
               parent_question: str = "",
               knowledge_summary: str = "") -> int:
        return self.qa.insert(question_text=question_text, answer_text=answer_text,
                              topic=topic, paper=paper, question_number=question_number,
                              parent_question=parent_question, knowledge_summary=knowledge_summary)

    def get(self, qa_id: int) -> Optional[dict]:
        return self._qb.get("qa_pairs", qa_id)

    def get_all(self) -> list[dict]:
        return self._qb.get_all("qa_pairs", order_by="id")

    def get_by_ids(self, ids: list[int]) -> list[dict]:
        return self.qa.get_by_ids(ids)

    def count(self) -> int:
        return self._qb.count("qa_pairs")

    def record_attempt(self, qa_id: int, success: bool, reason: str = ""):
        self.qa.record_attempt(qa_id, success, reason)

    def get_topic_groups(self) -> dict[str, list[dict]]:
        return self.qa.get_topic_groups()

    def log_api_call(self, stage: str, model: str, paper: str = "",
                     question_number: str = "", latency_ms: int = 0,
                     success: bool = True, output_size: int = 0):
        self.analysis.log_api_call(stage, model, paper=paper, question_number=question_number,
                                   latency_ms=latency_ms, success=success, output_size=output_size)

    def log_question_feedback(self, qa_id: int, retrieval_count: int = 0,
                              used_qa_count: int = 0, step0_topic: str = "",
                              round2_topic: str = "", covered_count: int = 0,
                              missed_count: int = 0, missed_text: str = "",
                              miss_categories: str = ""):
        self.analysis.log_question_feedback(qa_id, retrieval_count=retrieval_count,
                                            used_qa_count=used_qa_count, step0_topic=step0_topic,
                                            round2_topic=round2_topic, covered_count=covered_count,
                                            missed_count=missed_count, missed_text=missed_text,
                                            miss_categories=miss_categories)

    def get_missed_by_topic(self, topic: str) -> list[str]:
        return self.analysis.get_missed_by_topic(topic)

    def get_all_weights(self) -> dict[int, dict]:
        return self.qa.get_all_weights()

    # ============================================================
    # Distillation cache — enables incremental distillation
    # ============================================================

    def get_distillation_cache(self) -> dict[str, str]:
        return self.analysis.get_distillation_cache()

    def get_cached_topic_state(self, topic: str) -> dict | None:
        return self.analysis.get_cached_topic_state(topic)

    def upsert_distillation_cache(self, topic: str, qa_count: int,
                                   qa_ids_hash: str, content: str):
        self.analysis.upsert_distillation_cache(topic, qa_count, qa_ids_hash, content)

    def invalidate_distillation_cache(self, topic: str):
        self.analysis.invalidate_distillation_cache(topic)

    # ============================================================
    # Evolution history — tracks KP self-improvement events
    # ============================================================

    def record_evolution(self, kp_id: str, trigger_type: str,
                         trigger_detail: str = "", old_state: str = "",
                         new_state: str = "", outcome: str = "pending"):
        self.analysis.record_evolution(kp_id, trigger_type, trigger_detail=trigger_detail,
                                       old_state=old_state, new_state=new_state, outcome=outcome)

    def get_pending_evolutions(self, kp_id: str = None) -> list[dict]:
        return self.analysis.get_pending_evolutions(kp_id)

    # ============================================================
    # Phase 1: MS Fragments + Dynamic Topics
    # ============================================================

    def insert_fragments_batch(self, fragments: list[dict]) -> int:
        return self.fragment.insert_batch(fragments)

    def set_fragment_membership(self, fragment_id: str, topic_id: str,
                                 loyalty: float = 0.5):
        self.topic.set_fragment_membership(fragment_id, topic_id, loyalty)

    def upsert_dynamic_topic(self, topic_id: str, name: str = "",
                               quality: str = "embryonic"):
        self.topic.upsert(topic_id, name=name, quality=quality)

    def update_topic_stats(self, topic_id: str, mass: int, cohesion: float,
                            stability: float):
        self.topic.update_stats(topic_id, mass, cohesion, stability)

    def set_topic_kp(self, topic_id: str, kp_concept: str, kp_detail: str):
        self.topic.set_kp(topic_id, kp_concept, kp_detail)

    def get_stable_topics(self) -> list[dict]:
        """Return all topics with quality='stable' and their KP text."""
        rows = self._qb.get_where("dynamic_topics", quality="stable")
        return [dict(r) for r in rows]

    def get_topic_fragments(self, topic_id: str) -> list[str]:
        """Return fragment IDs belonging to a topic."""
        rows = self._qb.get_where("fragment_membership", topic_id=topic_id)
        return [r["fragment_id"] for r in rows]

    def get_fragment_help_count(self, fragment_id: str) -> int:
        """Return how many questions a fragment has helped."""
        return self._qb.count("fragment_help_map", fragment_id=fragment_id)

    def get_topic_helped_questions(self, topic_id: str) -> set:
        """Return the set of QA IDs that this topic's fragments helped."""
        rows = self.conn.execute(
            """SELECT DISTINCT fhm.helped_qa_id
               FROM fragment_help_map fhm
               JOIN fragment_membership fm ON fhm.fragment_id = fm.fragment_id
               WHERE fm.topic_id = ?""",
            (topic_id,),
        ).fetchall()
        return {r["helped_qa_id"] for r in rows}

    # ============================================================
    # Phase 5: Fragment centrality + vector infrastructure
    # ============================================================

    def upsert_fragment_centrality(self, fragment_id: str, centrality_score: float,
                                    avg_help_score: float = 0.0, topic_coherence: float = 0.0,
                                    variance: float = 0.0):
        self.fragment.upsert_centrality(fragment_id, centrality_score,
                                        avg_help_score=avg_help_score,
                                        topic_coherence=topic_coherence, variance=variance)

    def get_fragment_centrality(self, fragment_id: str) -> dict | None:
        return self._qb.get("fragment_centrality", fragment_id, id_col="fragment_id")

    def get_topic_fragment_centralities(self, topic_id: str) -> list[dict]:
        return self.fragment.get_topic_centralities(topic_id)

    def upsert_kp_vector(self, kp_id: str, vector: np.ndarray):
        self.vector.upsert_kp_vector(kp_id, vector)

    def get_kp_vector(self, kp_id: str) -> np.ndarray | None:
        return self.vector.get_kp_vector(kp_id)

    def upsert_qa_kp_score(self, qa_id: int, kp_id: str, relevance_score: float):
        self.vector.upsert_qa_kp_score(qa_id, kp_id, relevance_score)

    def get_qa_kp_scores(self, qa_id: int) -> dict[str, float]:
        return self.vector.get_qa_kp_scores(qa_id)

    def upsert_topic_vector(self, topic_id: str, vector: np.ndarray, member_count: int = 0):
        self.vector.upsert_topic_vector(topic_id, vector, member_count)

    def get_topic_vector(self, topic_id: str) -> np.ndarray | None:
        return self.vector.get_topic_vector(topic_id)

    def record_fragment_help_with_level(self, fragment_id: str, helped_qa_id: int,
                                         help_effect: float = 0.0, help_level: str = ""):
        self.fragment.record_help_with_level(fragment_id, helped_qa_id, help_effect, help_level)

    def upsert_topic_link(self, src_topic: str, dst_topic: str, count: int = 1):
        self.topic.upsert_link(src_topic, dst_topic, count)

    def get_topic_links(self) -> dict:
        return self.topic.get_links()

    # ---- Chat history ----

    def save_chat_message(self, session_id: str, role: str, content: str, sources: str = ""):
        self.chat.save_message(session_id, role, content, sources)

    def get_chat_history(self, session_id: str, limit: int = 50) -> list[dict]:
        return self.chat.get_history(session_id, limit)

    def clear_chat_history(self, session_id: str):
        self.chat.clear_history(session_id)

    # ---- Student memory ----

    def save_student_memory(self, student_id: str, memory_type: str, topic: str, content: str):
        self.student.save_memory(student_id, memory_type, topic, content)

    def get_student_memories(self, student_id: str, limit: int = 20) -> list[dict]:
        return self.student.get_memories(student_id, limit)

    def get_student_confusions(self, student_id: str) -> list[dict]:
        return self.student.get_confusions(student_id)

    def record_confusion(self, student_id: str, topic: str, trigger: str, ctype: str):
        self.student.record_confusion(student_id, topic, trigger, ctype)

    def upsert_knowledge_state(self, student_id: str, topic: str, state: str):
        self.student.upsert_knowledge_state(student_id, topic, state)

    def get_knowledge_state(self, student_id: str) -> dict[str, str]:
        return self.student.get_knowledge_state(student_id)

    # ---- Exam stats ----

    def get_exam_stats(self, topic: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT s.year, s.season, COUNT(*) as cnt
               FROM qa_pairs q
               JOIN exam_sessions s ON q.session_id = s.id
               WHERE q.topic = ?
               GROUP BY s.year, s.season
               ORDER BY s.year DESC, s.season""",
            (topic,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- Topic dependencies ----

    def insert_dependency(self, spec: DependencySpec):
        self.analysis.insert_dependency(
            spec.prerequisite, spec.dependent,
            evidence_score=spec.evidence_score, evidence_reason=spec.evidence_reason,
            relationship_type=spec.relationship_type, topic_link_count=spec.topic_link_count,
            embedding_cos=spec.embedding_cos, confidence=spec.confidence,
            validated_by=spec.validated_by)

    def get_dependencies(self, confidence: str = None) -> list[dict]:
        if confidence:
            rows = self.conn.execute(
                "SELECT * FROM topic_dependencies WHERE confidence = ? ORDER BY prerequisite",
                (confidence,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM topic_dependencies ORDER BY prerequisite"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_dependency_graph(self) -> dict:
        """Return {topic: {prerequisites: [...], dependents: [...]}} for all topics."""
        rows = self.conn.execute(
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

    def get_direct_prerequisites(self, topic: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT prerequisite, evidence_score, confidence, relationship_type
               FROM topic_dependencies WHERE dependent = ?""",
            (topic,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_transitive_prerequisites(self, topic: str, max_depth: int = 5) -> list[str]:
        """BFS to find all transitive prerequisites of a topic."""
        seen = set()
        frontier = [topic]
        for _ in range(max_depth):
            if not frontier:
                break
            next_frontier = []
            for t in frontier:
                rows = self.conn.execute(
                    "SELECT prerequisite FROM topic_dependencies WHERE dependent = ?",
                    (t,),
                ).fetchall()
                for r in rows:
                    pre = r["prerequisite"]
                    if pre not in seen:
                        seen.add(pre)
                        next_frontier.append(pre)
            frontier = next_frontier
        return list(seen)

    # ---- Command verb patterns ----

    def upsert_verb_pattern(self, spec: VerbPatternSpec):
        self.analysis.upsert_verb_pattern(
            spec.verb, sample_count=spec.sample_count,
            avg_answer_length=spec.avg_answer_length,
            median_answer_length=spec.median_answer_length,
            bullet_ratio=spec.bullet_ratio, avg_bullet_count=spec.avg_bullet_count,
            avg_miss_rate=spec.avg_miss_rate,
            common_missed_patterns=spec.common_missed_patterns,
            pattern_summary=spec.pattern_summary,
            topic_specific_patterns=spec.topic_specific_patterns,
            verb_family=spec.verb_family)

    def get_verb_patterns(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM command_verb_patterns ORDER BY sample_count DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_verb_for_qa(self, qa_id: int) -> dict:
        return self.qa.get_verb_for_qa(qa_id)

    # ---- Topic difficulty ----

    def upsert_topic_difficulty(self, topic: str, qa_count: int = 0,
                                basic_count: int = 0, intermediate_count: int = 0,
                                advanced_count: int = 0, mode_difficulty: str = "",
                                avg_miss_rate: float = None,
                                difficulty_spread: bool = False,
                                assessment_method: str = "hybrid"):
        self.topic.upsert_difficulty(topic, qa_count=qa_count,
                                     basic_count=basic_count, intermediate_count=intermediate_count,
                                     advanced_count=advanced_count, mode_difficulty=mode_difficulty,
                                     avg_miss_rate=avg_miss_rate, difficulty_spread=difficulty_spread,
                                     assessment_method=assessment_method)

    def get_topic_difficulty(self, topic: str = None) -> list[dict]:
        if topic:
            rows = self.conn.execute(
                "SELECT * FROM topic_difficulty WHERE topic = ?", (topic,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM topic_difficulty ORDER BY mode_difficulty, topic"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_qa_difficulty(self, qa_id: int) -> str:
        return self.qa.get_qa_difficulty(qa_id)

    def get_effective_miss_rate(self, qa_id: int) -> Optional[float]:
        return self.qa.get_effective_miss_rate(qa_id)

    # ---- Analysis checkpoints ----

    def checkpoint(self, task_name: str, qa_count: int, status: str = "completed"):
        self.analysis.checkpoint(task_name, qa_count, status)

    def get_checkpoint(self, task_name: str) -> dict:
        return self.analysis.get_checkpoint(task_name)

    def clear_checkpoint(self, task_name: str):
        self.analysis.clear_checkpoint(task_name)

    # ---- Topic vectors helper (for dependency candidate generation) ----

    def get_topic_answer_texts(self) -> dict[str, str]:
        return self.qa.get_topic_answer_texts()

    # ---- Knowledge Points (KP graph) ----

    def upsert_kp(self, spec: KPSpec):
        self.kp.upsert(spec)

    def get_all_kps(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, name, description, core_concept, core_detail, "
            "cohesion, evidence_count, quality FROM knowledge_points "
            "ORDER BY evidence_count DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_kp_by_id(self, kp_id: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM knowledge_points WHERE id = ?", (kp_id,)
        ).fetchone()
        return dict(row) if row else {}

    def get_kp_representative_qas(self, kp_id: str, limit: int = 3) -> list[dict]:
        rows = self.conn.execute(
            """SELECT q.* FROM qa_pairs q
               JOIN qa_kp_membership m ON q.id = m.qa_id
               WHERE m.kp_id = ? AND m.is_representative = 1
               LIMIT ?""",
            (kp_id, limit),
        ).fetchall()
        if not rows:
            rows = self.conn.execute(
                """SELECT q.* FROM qa_pairs q
                   JOIN qa_kp_membership m ON q.id = m.qa_id
                   WHERE m.kp_id = ?
                   ORDER BY m.membership_strength DESC LIMIT ?""",
                (kp_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- KP Edges ----

    def upsert_kp_edge(self, spec: KpEdgeSpec):
        self.kp.upsert_edge(spec)

    def get_kp_edges(self, kp_id: str = None) -> list[dict]:
        if kp_id:
            rows = self.conn.execute(
                "SELECT * FROM kp_edges WHERE source_kp = ? OR target_kp = ?",
                (kp_id, kp_id),
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM kp_edges").fetchall()
        return [dict(r) for r in rows]

    def get_kp_graph(self) -> dict:
        """Return {kp_id: {prerequisites: [...], dependents: [...]}}."""
        rows = self.conn.execute(
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

    # ---- QA-KP Membership ----

    def set_qa_kp_membership(self, qa_id: int, kp_id: str,
                             membership_strength: float = 1.0,
                             is_representative: bool = False):
        self.kp.set_membership(qa_id, kp_id, membership_strength, is_representative)

    def get_kp_qas(self, kp_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT q.*, m.membership_strength, m.is_representative
               FROM qa_pairs q
               JOIN qa_kp_membership m ON q.id = m.qa_id
               WHERE m.kp_id = ?""",
            (kp_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- Student Trajectory ----

    def record_trajectory(self, student_id: str, kp_id: str,
                          from_state: str, to_state: str, trigger: str = ""):
        self.student.record_trajectory(student_id, kp_id, from_state, to_state, trigger)

    def get_student_trajectory(self, student_id: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM student_trajectory
               WHERE student_id = ? ORDER BY recorded_at DESC LIMIT ?""",
            (student_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- Exam Trends ----

    def upsert_exam_trend(self, kp_id: str, year: int, season: str,
                          occurrence_count: int = 0, avg_difficulty: str = "",
                          trend_summary: str = ""):
        self.analysis.upsert_exam_trend(kp_id, year, season,
                                        occurrence_count=occurrence_count,
                                        avg_difficulty=avg_difficulty,
                                        trend_summary=trend_summary)

    def get_exam_trends(self, kp_id: str = None) -> list[dict]:
        if kp_id:
            rows = self.conn.execute(
                "SELECT * FROM exam_trends WHERE kp_id = ? ORDER BY year, season",
                (kp_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM exam_trends ORDER BY kp_id, year, season"
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self._db.close()


# ============================================================
# QARetriever
# ============================================================

def log_schema_status(db, debug_cb=None):
    """Diagnostic: log DB schema state for test verification."""
    def _d(msg):
        if debug_cb:
            debug_cb(f"[DB] {msg}")
    try:
        tables = [r["name"] for r in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()]
        new_tables = ["topic_dependencies", "command_verb_patterns", "topic_difficulty",
                      "analysis_checkpoints", "knowledge_points", "kp_edges",
                      "qa_kp_membership", "student_trajectory", "exam_trends"]
        present = [t for t in new_tables if t in tables]
        missing = [t for t in new_tables if t not in tables]
        _d(f"Tables: {len(tables)} total, {len(present)} new tables present"
           + (f", MISSING: {missing}" if missing else ""))

        qa_cols = [c["name"] for c in db.conn.execute("PRAGMA table_info(qa_pairs)").fetchall()]
        new_qa_cols = ["command_verb", "command_verb_secondary", "command_verb_inferred",
                       "difficulty_estimate", "is_representative", "is_cross_topic", "session_id"]
        qa_present = [c for c in new_qa_cols if c in qa_cols]
        qa_missing = [c for c in new_qa_cols if c not in qa_cols]
        _d(f"qa_pairs: {len(qa_cols)} columns"
           + (f", MISSING: {qa_missing}" if qa_missing else " (all expected present)"))

        fb_cols = [c["name"] for c in db.conn.execute("PRAGMA table_info(question_feedback)").fetchall()]
        _d(f"question_feedback has miss_categories: {'miss_categories' in fb_cols}")
    except Exception as e:
        _d(f"Schema status check failed: {e}")


class QARetriever:
    """Retrieve similar past questions from the QA database via embedding similarity."""

    def __init__(self, db: QADatabase):
        self._db = db
        self._embeddings: Optional[np.ndarray] = None
        self._id_map: dict[int, int] = {}
        self._embed_model_name: Optional[str] = None
        self._add_lock = threading.Lock()
        """Lock for add_qa to prevent race conditions when parallel pipeline workers
        modify _embeddings (ndarray) and _id_map (dict) concurrently."""

    def _ensure_embeddings(self):
        with self._add_lock:
            self._ensure_embeddings_locked()

    def _ensure_embeddings_locked(self):
        qas = self._db.get_all()
        if not qas:
            self._embeddings = np.empty((0, 384))
            self._id_map = {}
            self._embed_model_name = None
            return
        # Use raw QA text for corpus embeddings (ground truth, no Flash distortion).
        # knowledge_summary is only used as fallback if QA text is unavailable.
        texts = [
            (qa["question_text"] + " " + qa["answer_text"])
            if (qa.get("question_text") or qa.get("answer_text"))
            else (qa.get("knowledge_summary", ""))
            for qa in qas
        ]
        lang = _detect_language(texts)
        self._embed_model_name = MODEL_MAP[lang]
        model = _get_model(self._embed_model_name)
        self._embeddings = model.encode(
            texts, batch_size=64, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        self._id_map = {qa["id"]: i for i, qa in enumerate(qas)}

    def search(self, query: str, threshold: float = 0.5,
               min_k: int = 3, max_cap: int = 15) -> List[dict]:
        if self._embeddings is None or len(self._embeddings) == 0:
            self._ensure_embeddings()
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        # Use the SAME model as the corpus to keep vectors compatible
        if self._embed_model_name is None:
            lang = _detect_language([query])
            self._embed_model_name = MODEL_MAP[lang]
        model = _get_model(self._embed_model_name)
        query_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = np.dot(self._embeddings, query_vec)

        n = min(len(scores), max_cap)
        if len(scores) <= n:
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, n)[:n]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        id_list = list(self._id_map.keys())
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= threshold or len(results) < min_k:
                qa = self._db.get(id_list[idx])
                if qa:
                    qa["_score"] = score
                    results.append(qa)

        _log.debug(f"Retrieval: qlen={len(query)}, results={len(results)}, threshold={threshold}")
        return results

    def search_dual_channel(self, query: str, threshold: float = 0.5,
                             min_k: int = 3, max_cap: int = 15,
                             query_topic: str = "", query_kp_scores: dict = None) -> List[dict]:
        """Layer 2 dual-channel retrieval: embedding + structure + behavior.

        Channel A (semantic): embedding top-30
        Channel B (structure): topic affiliation + behavioral graph walk (walk-1)
        Mixed ranking: 0.35×embedding + 0.35×topic_match + 0.20×behavior + 0.10×keyword
        """
        # Ensure embeddings are ready
        if self._embeddings is None or len(self._embeddings) == 0:
            self._ensure_embeddings()
        if self._embeddings is None or len(self._embeddings) == 0:
            return self.search(query, threshold, min_k, max_cap)

        # Channel A: embedding top-30 (high recall)
        if self._embed_model_name is None:
            lang = _detect_language([query])
            self._embed_model_name = MODEL_MAP[lang]
        model = _get_model(self._embed_model_name)
        query_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = np.dot(self._embeddings, query_vec)

        channel_a_size = min(len(scores), CHANNEL_A_RECALL)
        top_a = np.argpartition(-scores, channel_a_size - 1)[:channel_a_size]
        top_a = top_a[np.argsort(-scores[top_a])]

        id_list = list(self._id_map.keys())
        channel_a_ids = {id_list[idx] for idx in top_a}
        channel_a_qas = [self._db.get(id_list[idx]) for idx in top_a]
        channel_a_qas = [q for q in channel_a_qas if q]

        # Channel B: structure — topic affiliation + graph walk
        channel_b_ids = set()
        keyword_query = set(query.lower().split())

        # B1: Topic affiliation — QAs in the same or adjacent topics
        if query_topic:
            topic_rows = self._db.conn.execute(
                "SELECT id, topic FROM qa_pairs WHERE topic=? AND topic!='' LIMIT 20",
                (query_topic,)
            ).fetchall()
            channel_b_ids.update(r["id"] for r in topic_rows)

            # Adjacent topics via topic_links
            adj_rows = self._db.conn.execute(
                "SELECT DISTINCT dst_topic FROM topic_links WHERE src_topic=? UNION "
                "SELECT DISTINCT src_topic FROM topic_links WHERE dst_topic=?",
                (query_topic, query_topic)
            ).fetchall()
            for adj in adj_rows[:5]:
                adj_rows2 = self._db.conn.execute(
                    "SELECT id FROM qa_pairs WHERE topic=? LIMIT 10",
                    (adj["dst_topic"],)
                ).fetchall()
                channel_b_ids.update(r["id"] for r in adj_rows2)

        # B2: Graph walk — QAs whose fragments helped same questions
        walk_source = list(channel_a_ids)[:15] if channel_a_ids else []  # single conversion, reused
        if walk_source:
            placeholders_a = ",".join("?" * len(walk_source))
            walk_rows = self._db.conn.execute(
                f"SELECT DISTINCT f2.qa_id FROM ("
                f"SELECT DISTINCT fhm1.helped_qa_id FROM fragment_help_map fhm1 "
                f"JOIN ms_fragments mf1 ON fhm1.fragment_id = mf1.point_id "
                f"WHERE mf1.qa_id IN ({placeholders_a})"
                f") shared_helps "
                f"JOIN fragment_help_map fhm2 ON shared_helps.helped_qa_id = fhm2.helped_qa_id "
                f"JOIN ms_fragments f2 ON fhm2.fragment_id = f2.point_id "
                f"LIMIT 30",
                walk_source
            ).fetchall()
            channel_b_ids.update(r["qa_id"] for r in walk_rows)

        # Remove Channel A overlap
        channel_b_ids -= channel_a_ids

        # Build candidate pool
        candidates = []
        for idx in top_a:
            qa_id = id_list[idx]
            qa = self._db.get(qa_id)
            if qa:
                qa["_score"] = float(scores[idx])
                qa["_channel"] = "embedding"
                candidates.append(qa)

        for qa_id in channel_b_ids:
            qa = self._db.get(qa_id)
            if qa:
                qa["_score"] = 0.0
                qa["_channel"] = "structure"
                candidates.append(qa)

        # Pre-load topic adjacency for fast lookup (avoid O(N) DB queries)
        candidate_topics = {qa.get("topic", "") for qa in candidates if qa.get("topic")}
        adjacency_map = {}
        if query_topic and candidate_topics:
            for ct in candidate_topics:
                adj_row = self._db.conn.execute(
                    "SELECT COUNT(*) as cnt FROM topic_links "
                    "WHERE (src_topic=? AND dst_topic=?) OR (src_topic=? AND dst_topic=?)",
                    (query_topic, ct, ct, query_topic)
                ).fetchone()
                if adj_row and adj_row["cnt"] > 0:
                    adjacency_map[ct] = True

        # Pre-load helped question set from Channel A (reuse walk_source, one query)
        helped_qa_set = set()
        if walk_source:
            bh_all_rows = self._db.conn.execute(
                f"SELECT DISTINCT helped_qa_id FROM fragment_help_map fhm2 "
                f"JOIN ms_fragments mf2 ON fhm2.fragment_id = mf2.point_id "
                f"WHERE mf2.qa_id IN ({','.join('?' * len(walk_source))})",
                walk_source
            ).fetchall()
            helped_qa_set = {r["helped_qa_id"] for r in bh_all_rows}

        # Behavior scores: chunked batch to avoid SQLite 999-param limit
        candidate_qa_ids = [qa["id"] for qa in candidates]
        behavior_scores = {}
        if helped_qa_set and candidate_qa_ids:
            helped_list = list(helped_qa_set)
            CHUNK = BEHAVIOR_CHUNK   # leave room for candidate_qa_ids chunk
            for i in range(0, len(candidate_qa_ids), CHUNK):
                c_chunk = candidate_qa_ids[i:i + CHUNK]
                for j in range(0, len(helped_list), CHUNK):
                    h_chunk = helped_list[j:j + CHUNK]
                    bh_batch_rows = self._db.conn.execute(
                        f"SELECT mf.qa_id, COUNT(*) as cnt FROM fragment_help_map fhm "
                        f"JOIN ms_fragments mf ON fhm.fragment_id = mf.point_id "
                        f"WHERE mf.qa_id IN ({','.join('?' * len(c_chunk))})"
                        f" AND fhm.helped_qa_id IN ({','.join('?' * len(h_chunk))})"
                        f" GROUP BY mf.qa_id",
                        c_chunk + h_chunk
                    ).fetchall()
                    for r in bh_batch_rows:
                        behavior_scores[r["qa_id"]] = min(1.0, r["cnt"] / 10.0)

        # Mixed ranking (pre-loaded data, zero DB queries)
        for qa in candidates:
            emb_score = qa.get("_score", 0.0)
            qa_topic = qa.get("topic", "")

            topic_score = 0.0
            if query_topic and qa_topic:
                if qa_topic == query_topic:
                    topic_score = 1.0
                elif qa_topic in adjacency_map:
                    topic_score = 0.5

            behavior_score = behavior_scores.get(qa["id"], 0.0)

            qa_text = (qa.get("question_text", "") + " " + qa.get("answer_text", "")).lower()
            qa_keywords = set(qa_text.split())
            kw_jaccard = len(keyword_query & qa_keywords) / max(len(keyword_query | qa_keywords), 1)

            composite = (WEIGHT_EMBEDDING * emb_score + WEIGHT_TOPIC * topic_score
                         + WEIGHT_BEHAVIOR * behavior_score + WEIGHT_KEYWORD * kw_jaccard)
            qa["_score"] = composite

        # Sort by composite score and return top-k
        candidates.sort(key=lambda q: q["_score"], reverse=True)
        results = []
        for qa in candidates:
            if qa["_score"] >= threshold or len(results) < min_k:
                results.append(qa)

        _log.debug(f"Dual-channel retrieval: qlen={len(query)}, "
                   f"chA={len(channel_a_ids)}, chB={len(channel_b_ids)}, results={len(results)}")
        return results[:max_cap]

    def add_qa(self, qa_id: int, summary_text: str):
        """Add a new QA vector to the index. Uses raw QA text for embedding
        (consistent with _ensure_embeddings), falls back to summary_text."""
        with self._add_lock:
            if qa_id in self._id_map:
                return
            # Fetch QA for raw text; fall back to summary_text if DB unavailable
            qa = self._db.get(qa_id)
            if qa and (qa.get("question_text") or qa.get("answer_text")):
                encode_text = qa["question_text"] + " " + qa["answer_text"]
            else:
                encode_text = summary_text or ""
            # Use the same model as the corpus to keep vectors compatible
            if self._embed_model_name is None:
                lang = _detect_language([encode_text])
                self._embed_model_name = MODEL_MAP[lang]
            model = _get_model(self._embed_model_name)
            new_vec = model.encode([encode_text], normalize_embeddings=True, convert_to_numpy=True)[0]
            if self._embeddings is None or len(self._embeddings) == 0:
                self._embeddings = new_vec.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, new_vec])
            self._id_map[qa_id] = len(self._embeddings) - 1

    def rebuild(self):
        self._embeddings = None
        self._id_map = {}
        self._embed_model_name = None
        self._ensure_embeddings()

    def count(self) -> int:
        """Return the number of QAs in the underlying database."""
        return self._db.count()

    def clear_chat_history(self, session_id: str):
        """Proxy to clear chat history in underlying DB."""
        self._db.clear_chat_history(session_id)
