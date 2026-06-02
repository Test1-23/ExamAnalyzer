"""QARetriever -- embedding-based similarity search over the QA database."""

from __future__ import annotations

import threading
import numpy as np
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .knowledge_base import QADatabase

from .embedding_cluster import _get_model, MODEL_MAP, _detect_language
from .logger import get_logger

_log = get_logger()

from .constants import (
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
# Diagnostic
# ============================================================

def log_schema_status(db, debug=None):
    """Diagnostic: log DB schema state for test verification."""
    def _d(msg):
        if debug:
            debug(f"[DB] {msg}")
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
        from .error_utils import log_exception
        log_exception(_d, "Schema status", "", e)

# ============================================================
# QARetriever
# ============================================================

class QARetriever:
    """Retrieve similar past questions from the QA database via embedding similarity."""

    def __init__(self, db: QADatabase):
        self._db = db
        self._embeddings: Optional[np.ndarray] = None
        self._id_map: dict[int, int] = {}
        self._embed_model_name: Optional[str] = None
        self._add_lock = threading.Lock()

    @property
    def db(self) -> QADatabase:
        """Public accessor for the underlying QADatabase (replaces _db)."""
        return self._db

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
            topic_rows = self._db.qa.get_by_topic(query_topic, limit=20)
            channel_b_ids.update(r["id"] for r in topic_rows)

            # Adjacent topics via topic_links
            for adj_topic in self._db.topic.get_adjacent_topics(query_topic)[:5]:
                adj_qas = self._db.qa.get_by_topic(adj_topic, limit=10)
                channel_b_ids.update(r["id"] for r in adj_qas)

        # B2: Graph walk — QAs whose fragments helped same questions
        walk_source = list(channel_a_ids)[:15] if channel_a_ids else []
        if walk_source:
            walk_qa_ids = self._db.fragment.get_second_order_qas(walk_source, limit=30)
            channel_b_ids.update(walk_qa_ids)

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

        # Pre-load topic adjacency for fast lookup (one query, not O(N))
        candidate_topics = [qa.get("topic", "") for qa in candidates if qa.get("topic")]
        linked_topics = self._db.topic.get_linked_mask(query_topic, candidate_topics) if query_topic and candidate_topics else set()

        # Pre-load helped question set from Channel A (reuse walk_source, one query)
        helped_qa_set = self._db.fragment.get_helped_qas(walk_source) if walk_source else set()

        # Behavior scores: chunked batch to avoid SQLite 999-param limit
        candidate_qa_ids = [qa["id"] for qa in candidates]
        behavior_scores = {}
        if helped_qa_set and candidate_qa_ids:
            helped_list = list(helped_qa_set)
            CHUNK = BEHAVIOR_CHUNK
            for i in range(0, len(candidate_qa_ids), CHUNK):
                c_chunk = candidate_qa_ids[i:i + CHUNK]
                for j in range(0, len(helped_list), CHUNK):
                    h_chunk = helped_list[j:j + CHUNK]
                    batch_scores = self._db.fragment.get_behavior_scores(c_chunk, h_chunk)
                    behavior_scores.update(batch_scores)

        # Mixed ranking (pre-loaded data, zero DB queries)
        for qa in candidates:
            emb_score = qa.get("_score", 0.0)
            qa_topic = qa.get("topic", "")

            topic_score = 0.0
            if query_topic and qa_topic:
                if qa_topic == query_topic:
                    topic_score = 1.0
                elif qa_topic in linked_topics:
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
        """Clear and rebuild the embedding index. Returns loaded QA rows.

        Callers can reuse the returned rows to avoid a second full-table scan
        when building a KP cache or analysis context immediately after rebuild.
        """
        self._embeddings = None
        self._id_map = {}
        self._embed_model_name = None
        self._ensure_embeddings()
        return self._db.get_all()

    def count(self) -> int:
        """Return the number of QAs in the underlying database."""
        return self._db.count()

    def clear_chat_history(self, session_id: str):
        """Proxy to clear chat history in underlying DB."""
        self._db.chat.clear_history(session_id)
