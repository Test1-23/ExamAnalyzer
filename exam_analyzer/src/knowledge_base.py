"""QADatabase -- SQLite QA knowledge base with Store-based CRUD."""

import sqlite3
import numpy as np
from typing import Optional

from .connection_manager import ConnectionMgr
from .logger import get_logger
from .models import KPSpec, VerbPatternSpec, KpEdgeSpec, DependencySpec
from .query_builder import QueryBuilder

_log = get_logger()

from .constants import SQLITE_PARAM_CHUNK
from .retriever import QARetriever, make_topic_id, log_schema_status  # re-export for backward compat

# ============================================================
# QADatabase
# ============================================================

class QADatabase:
    """Facade assembling ConnectionMgr → QueryBuilder → 8 Domain Stores.

    Interface freeze policy (2026-05-31): No new methods shall be added to this
    class.  New data access should use the Store directly: ``db.qa.insert(...)``,
    ``db.topic.upsert_link(...)``, etc.  Existing proxy methods remain for
    backward compatibility but are conceptually deprecated.
    """


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
        """(deprecated — use db.qa.update_topic instead)"""
        self.qa.update_topic(qa_id, topic)

    def rename_topic(self, new_topic: str, old_topic: str) -> int:
        """(deprecated — use db.qa.rename_topic instead)"""
        return self.qa.rename_topic(new_topic, old_topic)

    def insert(self, question_text: str, answer_text: str,
               topic: str = "", paper: str = "",
               question_number: str = "",
               parent_question: str = "",
               knowledge_summary: str = "") -> int:
        """(deprecated — use db.qa.insert instead) Insert a QA pair."""
        return self.qa.insert(question_text=question_text, answer_text=answer_text,
                              topic=topic, paper=paper, question_number=question_number,
                              parent_question=parent_question, knowledge_summary=knowledge_summary)

    def get(self, qa_id: int) -> Optional[dict]:
        """(deprecated — use db.qa.get instead)"""
        return self._qb.get("qa_pairs", qa_id)

    def get_all(self) -> list[dict]:
        """(deprecated — use db.qa.get_all instead)"""
        return self._qb.get_all("qa_pairs", order_by="id")

    def get_by_ids(self, ids: list[int]) -> list[dict]:
        """(deprecated — use db.qa.get_by_ids instead)"""
        return self.qa.get_by_ids(ids)

    def count(self) -> int:
        """(deprecated — use db.qa.count instead)"""
        return self._qb.count("qa_pairs")

    def record_attempt(self, qa_id: int, success: bool, reason: str = ""):
        """(deprecated — use db.qa.record_attempt instead)"""
        self.qa.record_attempt(qa_id, success, reason)

    def get_topic_groups(self) -> dict[str, list[dict]]:
        """(deprecated — use db.qa.get_topic_groups instead)"""
        return self.qa.get_topic_groups()

    def log_api_call(self, stage: str, model: str, paper: str = "",
                     question_number: str = "", latency_ms: int = 0,
                     success: bool = True, output_size: int = 0):
        """(deprecated — use db.analysis.log_api_call instead)"""
        self.analysis.log_api_call(stage, model, paper=paper, question_number=question_number,
                                   latency_ms=latency_ms, success=success, output_size=output_size)

    def log_question_feedback(self, qa_id: int, retrieval_count: int = 0,
                              used_qa_count: int = 0, step0_topic: str = "",
                              round2_topic: str = "", covered_count: int = 0,
                              missed_count: int = 0, missed_text: str = "",
                              miss_categories: str = ""):
        """(deprecated — use db.analysis.log_question_feedback instead)"""
        self.analysis.log_question_feedback(qa_id, retrieval_count=retrieval_count,
                                            used_qa_count=used_qa_count, step0_topic=step0_topic,
                                            round2_topic=round2_topic, covered_count=covered_count,
                                            missed_count=missed_count, missed_text=missed_text,
                                            miss_categories=miss_categories)

    def get_missed_by_topic(self, topic: str) -> list[str]:
        """(deprecated — use db.analysis.get_missed_by_topic instead)"""
        return self.analysis.get_missed_by_topic(topic)

    def get_all_weights(self) -> dict[int, dict]:
        """(deprecated — use db.qa.get_all_weights instead)"""
        return self.qa.get_all_weights()

    # ============================================================
    # Distillation cache — enables incremental distillation
    # ============================================================

    def get_distillation_cache(self) -> dict[str, str]:
        """(deprecated — use db.analysis.get_distillation_cache instead)"""
        return self.analysis.get_distillation_cache()

    def get_cached_topic_state(self, topic: str) -> dict | None:
        """(deprecated — use db.analysis.get_cached_topic_state instead)"""
        return self.analysis.get_cached_topic_state(topic)

    def upsert_distillation_cache(self, topic: str, qa_count: int,
                                   qa_ids_hash: str, content: str):
        """(deprecated — use db.analysis.upsert_distillation_cache instead)"""
        self.analysis.upsert_distillation_cache(topic, qa_count, qa_ids_hash, content)

    def invalidate_distillation_cache(self, topic: str):
        """(deprecated — use db.analysis.invalidate_distillation_cache instead)"""
        self.analysis.invalidate_distillation_cache(topic)

    # ============================================================
    # Evolution history — tracks KP self-improvement events
    # ============================================================

    def record_evolution(self, kp_id: str, trigger_type: str,
                         trigger_detail: str = "", old_state: str = "",
                         new_state: str = "", outcome: str = "pending"):
        """(deprecated — use db.analysis.record_evolution instead)"""
        self.analysis.record_evolution(kp_id, trigger_type, trigger_detail=trigger_detail,
                                       old_state=old_state, new_state=new_state, outcome=outcome)

    def get_pending_evolutions(self, kp_id: str = None) -> list[dict]:
        """(deprecated — use db.analysis.get_pending_evolutions instead)"""
        return self.analysis.get_pending_evolutions(kp_id)

    # ============================================================
    # Phase 1: MS Fragments + Dynamic Topics
    # ============================================================

    def insert_fragments_batch(self, fragments: list[dict]) -> int:
        """(deprecated — use db.fragment.insert_batch instead)"""
        return self.fragment.insert_batch(fragments)

    def set_fragment_membership(self, fragment_id: str, topic_id: str,
                                 loyalty: float = 0.5):
        """(deprecated — use db.topic.set_fragment_membership instead)"""
        self.topic.set_fragment_membership(fragment_id, topic_id, loyalty)

    def upsert_dynamic_topic(self, topic_id: str, name: str = "",
                               quality: str = "embryonic"):
        """(deprecated — use db.topic.upsert instead)"""
        self.topic.upsert(topic_id, name=name, quality=quality)

    def update_topic_stats(self, topic_id: str, mass: int, cohesion: float,
                            stability: float):
        """(deprecated — use db.topic.update_stats instead)"""
        self.topic.update_stats(topic_id, mass, cohesion, stability)

    def set_topic_kp(self, topic_id: str, kp_concept: str, kp_detail: str):
        """(deprecated — use db.topic.set_kp instead)"""
        self.topic.set_kp(topic_id, kp_concept, kp_detail)

    def get_stable_topics(self) -> list[dict]:
        """(deprecated — use db.topic.get_stable instead)"""
        return self.topic.get_stable()

    def get_topic_fragments(self, topic_id: str) -> list[str]:
        """(deprecated — use db.topic.get_fragments instead)"""
        return self.topic.get_fragments(topic_id)

    def get_topic_helped_questions(self, topic_id: str) -> set:
        """(deprecated — use db.fragment.get_topic_helped_questions instead)"""
        return self.fragment.get_topic_helped_questions(topic_id)

    # ============================================================
    # Phase 5: Fragment centrality + vector infrastructure
    # ============================================================

    def upsert_fragment_centrality(self, fragment_id: str, centrality_score: float,
                                    avg_help_score: float = 0.0, topic_coherence: float = 0.0,
                                    variance: float = 0.0):
        """(deprecated — use db.fragment.upsert_centrality instead)"""
        self.fragment.upsert_centrality(fragment_id, centrality_score,
                                        avg_help_score=avg_help_score,
                                        topic_coherence=topic_coherence, variance=variance)

    def get_topic_fragment_centralities(self, topic_id: str) -> list[dict]:
        """(deprecated — use db.fragment.get_topic_centralities instead)"""
        return self.fragment.get_topic_centralities(topic_id)

    def upsert_kp_vector(self, kp_id: str, vector: np.ndarray):
        """(deprecated — use db.vector.upsert_kp_vector instead)"""
        self.vector.upsert_kp_vector(kp_id, vector)

    def get_kp_vector(self, kp_id: str) -> np.ndarray | None:
        """(deprecated — use db.vector.get_kp_vector instead)"""
        return self.vector.get_kp_vector(kp_id)

    def upsert_qa_kp_score(self, qa_id: int, kp_id: str, relevance_score: float):
        """(deprecated — use db.vector.upsert_qa_kp_score instead)"""
        self.vector.upsert_qa_kp_score(qa_id, kp_id, relevance_score)

    def get_qa_kp_scores(self, qa_id: int) -> dict[str, float]:
        """(deprecated — use db.vector.get_qa_kp_scores instead)"""
        return self.vector.get_qa_kp_scores(qa_id)

    def upsert_topic_vector(self, topic_id: str, vector: np.ndarray, member_count: int = 0):
        """(deprecated — use db.vector.upsert_topic_vector instead)"""
        self.vector.upsert_topic_vector(topic_id, vector, member_count)

    def get_topic_vector(self, topic_id: str) -> np.ndarray | None:
        """(deprecated — use db.vector.get_topic_vector instead)"""
        return self.vector.get_topic_vector(topic_id)

    def record_fragment_help_with_level(self, fragment_id: str, helped_qa_id: int,
                                         help_effect: float = 0.0, help_level: str = ""):
        """(deprecated — use db.fragment.record_help_with_level instead)"""
        self.fragment.record_help_with_level(fragment_id, helped_qa_id, help_effect, help_level)

    def upsert_topic_link(self, src_topic: str, dst_topic: str, count: int = 1):
        """(deprecated — use db.topic.upsert_link instead)"""
        self.topic.upsert_link(src_topic, dst_topic, count)

    def get_topic_links(self) -> dict:
        """(deprecated — use db.topic.get_links instead)"""
        return self.topic.get_links()

    # ---- Chat history ----

    def save_chat_message(self, session_id: str, role: str, content: str, sources: str = ""):
        """(deprecated — use db.chat.save_message instead)"""
        self.chat.save_message(session_id, role, content, sources)

    def get_chat_history(self, session_id: str, limit: int = 50) -> list[dict]:
        """(deprecated — use db.chat.get_history instead)"""
        return self.chat.get_history(session_id, limit)

    def clear_chat_history(self, session_id: str):
        """(deprecated — use db.chat.clear_history instead)"""
        self.chat.clear_history(session_id)

    # ---- Student memory ----

    def save_student_memory(self, student_id: str, memory_type: str, topic: str, content: str):
        """(deprecated — use db.student.save_memory instead)"""
        self.student.save_memory(student_id, memory_type, topic, content)

    def get_student_confusions(self, student_id: str) -> list[dict]:
        """(deprecated — use db.student.get_confusions instead)"""
        return self.student.get_confusions(student_id)

    def record_confusion(self, student_id: str, topic: str, trigger: str, ctype: str):
        """(deprecated — use db.student.record_confusion instead)"""
        self.student.record_confusion(student_id, topic, trigger, ctype)

    def upsert_knowledge_state(self, student_id: str, topic: str, state: str):
        """(deprecated — use db.student.upsert_knowledge_state instead)"""
        self.student.upsert_knowledge_state(student_id, topic, state)

    def get_knowledge_state(self, student_id: str) -> dict[str, str]:
        """(deprecated — use db.student.get_knowledge_state instead)"""
        return self.student.get_knowledge_state(student_id)

    # ---- Exam stats ----

    def get_exam_stats(self, topic: str) -> list[dict]:
        """(deprecated — use db.analysis.get_exam_stats instead)"""
        return self.analysis.get_exam_stats(topic)

    # ---- Topic dependencies ----

    def insert_dependency(self, spec: DependencySpec):
        """(deprecated — use db.analysis.insert_dependency instead)"""
        self.analysis.insert_dependency(
            spec.prerequisite, spec.dependent,
            evidence_score=spec.evidence_score, evidence_reason=spec.evidence_reason,
            relationship_type=spec.relationship_type, topic_link_count=spec.topic_link_count,
            embedding_cos=spec.embedding_cos, confidence=spec.confidence,
            validated_by=spec.validated_by)

    def get_dependencies(self, confidence: str = None) -> list[dict]:
        """(deprecated — use db.analysis.get_dependencies instead)"""
        return self.analysis.get_dependencies(confidence)

    def get_dependency_graph(self) -> dict:
        """(deprecated — use db.analysis.get_dependency_graph instead)"""
        return self.analysis.get_dependency_graph()

    def get_direct_prerequisites(self, topic: str) -> list[dict]:
        """(deprecated — use db.analysis.get_direct_prerequisites instead)"""
        return self.analysis.get_direct_prerequisites(topic)

    def get_transitive_prerequisites(self, topic: str, max_depth: int = 5) -> list[str]:
        """(deprecated — use db.analysis.get_transitive_prerequisites instead)"""
        return self.analysis.get_transitive_prerequisites(topic, max_depth)

    # ---- Command verb patterns ----

    def upsert_verb_pattern(self, spec: VerbPatternSpec):
        """(deprecated — use db.analysis.upsert_verb_pattern instead)"""
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
        """(deprecated — use db.analysis.get_verb_patterns instead)"""
        return self.analysis.get_verb_patterns()

    def get_verb_for_qa(self, qa_id: int) -> dict:
        """(deprecated — use db.qa.get_verb_for_qa instead)"""
        return self.qa.get_verb_for_qa(qa_id)

    # ---- Topic difficulty ----

    def upsert_topic_difficulty(self, topic: str, qa_count: int = 0,
                                basic_count: int = 0, intermediate_count: int = 0,
                                advanced_count: int = 0, mode_difficulty: str = "",
                                avg_miss_rate: float = None,
                                difficulty_spread: bool = False,
                                assessment_method: str = "hybrid"):
        """(deprecated — use db.topic.upsert_difficulty instead)"""
        self.topic.upsert_difficulty(topic, qa_count=qa_count,
                                     basic_count=basic_count, intermediate_count=intermediate_count,
                                     advanced_count=advanced_count, mode_difficulty=mode_difficulty,
                                     avg_miss_rate=avg_miss_rate, difficulty_spread=difficulty_spread,
                                     assessment_method=assessment_method)

    def get_topic_difficulty(self, topic: str = None) -> list[dict]:
        """(deprecated — use db.topic.get_difficulty instead)"""
        return self.topic.get_difficulty(topic)

    def get_qa_difficulty(self, qa_id: int) -> str:
        """(deprecated — use db.qa.get_qa_difficulty instead)"""
        return self.qa.get_qa_difficulty(qa_id)

    def get_effective_miss_rate(self, qa_id: int) -> Optional[float]:
        """(deprecated — use db.qa.get_effective_miss_rate instead)"""
        return self.qa.get_effective_miss_rate(qa_id)

    # ---- Analysis checkpoints ----

    def checkpoint(self, task_name: str, qa_count: int, status: str = "completed"):
        """(deprecated — use db.analysis.checkpoint instead)"""
        self.analysis.checkpoint(task_name, qa_count, status)

    def get_checkpoint(self, task_name: str) -> dict:
        """(deprecated — use db.analysis.get_checkpoint instead)"""
        return self.analysis.get_checkpoint(task_name)

    def clear_checkpoint(self, task_name: str):
        """(deprecated — use db.analysis.clear_checkpoint instead)"""
        self.analysis.clear_checkpoint(task_name)

    # ---- Topic vectors helper (for dependency candidate generation) ----

    def get_topic_answer_texts(self) -> dict[str, str]:
        """(deprecated — use db.qa.get_topic_answer_texts instead)"""
        return self.qa.get_topic_answer_texts()

    # ---- Knowledge Points (KP graph) ----

    def upsert_kp(self, spec: KPSpec):
        """(deprecated — use db.kp.upsert instead)"""
        self.kp.upsert(spec)

    def get_all_kps(self) -> list[dict]:
        """(deprecated — use db.kp.get_all instead)"""
        return self.kp.get_all()

    def get_kp_by_id(self, kp_id: str) -> dict:
        """(deprecated — use db.kp.get_by_id instead)"""
        return self.kp.get_by_id(kp_id)

    def get_kp_representative_qas(self, kp_id: str, limit: int = 3) -> list[dict]:
        """(deprecated — use db.kp.get_representative_qas instead)"""
        return self.kp.get_representative_qas(kp_id, limit)

    # ---- KP Edges ----

    def upsert_kp_edge(self, spec: KpEdgeSpec):
        """(deprecated — use db.kp.upsert_edge instead)"""
        self.kp.upsert_edge(spec)

    def get_kp_edges(self, kp_id: str = None) -> list[dict]:
        """(deprecated — use db.kp.get_edges instead)"""
        return self.kp.get_edges(kp_id)

    # ---- QA-KP Membership ----

    def set_qa_kp_membership(self, qa_id: int, kp_id: str,
                             membership_strength: float = 1.0,
                             is_representative: bool = False):
        """(deprecated — use db.kp.set_membership instead)"""
        self.kp.set_membership(qa_id, kp_id, membership_strength, is_representative)

    def get_kp_qas(self, kp_id: str) -> list[dict]:
        """(deprecated — use db.kp.get_kp_qas instead)"""
        return self.kp.get_kp_qas(kp_id)

    # ---- Student Trajectory ----

    def record_trajectory(self, student_id: str, kp_id: str,
                          from_state: str, to_state: str, trigger: str = ""):
        """(deprecated — use db.student.record_trajectory instead)"""
        self.student.record_trajectory(student_id, kp_id, from_state, to_state, trigger)

    # ---- Exam Trends ----

    def upsert_exam_trend(self, kp_id: str, year: int, season: str,
                          occurrence_count: int = 0, avg_difficulty: str = "",
                          trend_summary: str = ""):
        """(deprecated — use db.analysis.upsert_exam_trend instead)"""
        self.analysis.upsert_exam_trend(kp_id, year, season,
                                        occurrence_count=occurrence_count,
                                        avg_difficulty=avg_difficulty,
                                        trend_summary=trend_summary)

    def get_exam_trends(self, kp_id: str = None) -> list[dict]:
        """(deprecated — use db.analysis.get_exam_trends instead)"""
        return self.analysis.get_exam_trends(kp_id)

    def close(self):
        self._db.close()


# ============================================================
# QARetriever

