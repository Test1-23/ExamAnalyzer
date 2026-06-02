"""Store CRUD tests using temporary SQLite DB (mock_db fixture).

Covers QaStore, TopicStore, KpStore, ChatStore — the most-used Stores.
"""

import pytest
import numpy as np

from src.stores.qa_store import QaStore
from src.stores.topic_store import TopicStore
from src.stores.kp_store import KpStore
from src.stores.chat_store import ChatStore
from src.stores.vector_store import VectorStore
from src.models import KPSpec, KpEdgeSpec
from src.query_builder import QueryBuilder


class TestQaStore:
    def test_insert_and_get(self, mock_db):
        store = QaStore(QueryBuilder(mock_db._db))
        qid = store.insert("What is binary?", "Base-2 system", topic="Binary", paper="9618_s21_qp", question_number="1")
        assert qid > 0
        qa = store.get(qid)
        assert qa["question_text"] == "What is binary?"
        assert qa["topic"] == "Binary"

    def test_insert_dedup(self, mock_db):
        store = QaStore(QueryBuilder(mock_db._db))
        qid1 = store.insert("Q", "A", topic="T")
        qid2 = store.insert("Q", "A", topic="T2")
        assert qid1 == qid2  # same Q+A -> same ID

    def test_get_all(self, mock_db):
        store = QaStore(QueryBuilder(mock_db._db))
        store.insert("Q1", "A1", topic="T1")
        store.insert("Q2", "A2", topic="T2")
        assert store.count() == 2

    def test_get_by_topic(self, mock_db):
        store = QaStore(QueryBuilder(mock_db._db))
        store.insert("Q1", "A1", topic="Binary")
        store.insert("Q2", "A2", topic="Hex")
        rows = store.get_by_topic("Binary")
        assert len(rows) == 1
        assert rows[0]["topic"] == "Binary"

    def test_get_by_ids(self, mock_db):
        store = QaStore(QueryBuilder(mock_db._db))
        ids = [store.insert(f"Q{i}", f"A{i}", topic="T") for i in range(5)]
        result = store.get_by_ids(ids[:3])
        assert len(result) == 3

    def test_record_attempt_success(self, mock_db):
        store = QaStore(QueryBuilder(mock_db._db))
        qid = store.insert("Q", "A")
        store.record_attempt(qid, success=True)
        qa = store.get(qid)
        assert qa["success_count"] == 1
        assert qa["total_attempts"] == 1
        assert qa["last_failure_reason"] == ""

    def test_record_attempt_failure(self, mock_db):
        store = QaStore(QueryBuilder(mock_db._db))
        qid = store.insert("Q", "A")
        store.record_attempt(qid, success=False, reason="wrong")
        qa = store.get(qid)
        assert qa["success_count"] == 0
        assert qa["total_attempts"] == 1
        assert qa["last_failure_reason"] == "wrong"

    def test_rename_topic(self, mock_db):
        store = QaStore(QueryBuilder(mock_db._db))
        store.insert("Q1", "A1", topic="Old")
        store.insert("Q2", "A2", topic="Old")
        store.rename_topic("New", "Old")
        assert len(store.get_by_topic("Old")) == 0
        assert len(store.get_by_topic("New")) == 2

    def test_get_topic_groups(self, mock_db):
        store = QaStore(QueryBuilder(mock_db._db))
        store.insert("Q1", "A1", topic="Binary")
        store.insert("Q2", "A2", topic="Hex")
        groups = store.get_topic_groups()
        assert "Binary" in groups
        assert "Hex" in groups

    def test_get_all_weights(self, mock_db):
        store = QaStore(QueryBuilder(mock_db._db))
        qid = store.insert("Q", "A")
        store.record_attempt(qid, success=True)
        weights = store.get_all_weights()
        assert qid in weights
        assert "mean" in weights[qid]
        assert "lower_bound" in weights[qid]


class TestTopicStore:
    def test_upsert_and_get(self, mock_db):
        store = TopicStore(QueryBuilder(mock_db._db))
        store.upsert("topic_test", name="Test Topic", quality="embryonic")
        kps = store.get_stable_kps()
        assert isinstance(kps, list)  # may be empty (not stable yet)

    def test_upsert_link(self, mock_db):
        store = TopicStore(QueryBuilder(mock_db._db))
        store.upsert_link("A", "B", count=3)
        links = store.get_links()
        assert ("A", "B") in links
        assert links[("A", "B")] >= 1

    def test_upsert_link_self_skip(self, mock_db):
        store = TopicStore(QueryBuilder(mock_db._db))
        store.upsert_link("A", "A")  # should be no-op
        links = store.get_links()
        assert ("A", "A") not in links

    def test_set_fragment_membership(self, mock_db):
        store = TopicStore(QueryBuilder(mock_db._db))
        # Insert a fragment first (need ms_fragments row)
        mock_db._db.conn.execute(
            "INSERT OR IGNORE INTO ms_fragments (point_id, qa_id, point_text, marks) "
            "VALUES ('f_test', 1, 'test point', 1)")
        mock_db._db.conn.commit()
        store.set_fragment_membership("f_test", "topic_x", loyalty=0.5)
        frags = store.get_fragments("topic_x")
        assert "f_test" in frags


class TestKpStore:
    def test_upsert_and_get(self, mock_db):
        store = KpStore(QueryBuilder(mock_db._db))
        spec = KPSpec(kp_id="kp_001", name="Binary", description="Base-2",
                      core_concept="Binary numbers", core_detail="0s and 1s",
                      cohesion=0.9, evidence_count=5, quality="accepted")
        store.upsert(spec)
        all_kps = store.get_all()
        assert any(k["id"] == "kp_001" for k in all_kps)

    def test_get_by_id(self, mock_db):
        store = KpStore(QueryBuilder(mock_db._db))
        spec = KPSpec(kp_id="kp_002", name="Hex", description="Base-16",
                      core_concept="Hexadecimal", core_detail="0-F",
                      cohesion=0.8, evidence_count=3, quality="draft")
        store.upsert(spec)
        kp = store.get_by_id("kp_002")
        assert kp["name"] == "Hex"

    def test_upsert_edge(self, mock_db):
        store = KpStore(QueryBuilder(mock_db._db))
        spec = KpEdgeSpec(source_kp="A", target_kp="B", edge_type="prerequisite",
                          retrieval_weight=0.5, semantic_weight=0.6,
                          sequential_weight=0.3, learning_path_weight=0.4,
                          combined_strength=0.7, confidence="medium")
        store.upsert_edge(spec)
        edges = store.get_edges("A")
        assert any(e["target_kp"] == "B" for e in edges)

    def test_set_membership(self, mock_db):
        store = KpStore(QueryBuilder(mock_db._db))
        # Need a qa_pairs row first
        mock_db._db.conn.execute(
            "INSERT INTO qa_pairs (id, question_text, answer_text) VALUES (99, 'Q', 'A')")
        mock_db._db.conn.execute(
            "INSERT INTO knowledge_points (id, name, description, quality) "
            "VALUES ('kp_x', 'Test', 'Test KP', 'draft')")
        mock_db._db.conn.commit()
        store.set_membership(99, "kp_x", membership_strength=0.9, is_representative=True)


# ═══════════════════════════════════════════════════════════════
# WSD-027 — per-thread connection pool concurrency stress test
# ═══════════════════════════════════════════════════════════════


class TestConcurrentWrites:
    """Verify per-thread connection pool handles concurrent writes correctly."""

    def test_concurrent_writes_no_errors(self, mock_db):
        """50 QAs across 10 threads → zero SQLITE_BUSY, zero lost rows."""
        from concurrent.futures import ThreadPoolExecutor

        def write(n):
            mock_db.qa.insert(f"Q{n}", f"A{n}", topic="Test")

        with ThreadPoolExecutor(max_workers=10) as ex:
            list(ex.map(write, range(50)))

        assert mock_db.qa.count() == 50, f"Expected 50, got {mock_db.qa.count()}"


class TestChatStore:
    def test_save_and_get(self, mock_db):
        store = ChatStore(QueryBuilder(mock_db._db))
        store.save_message("s1", "user", "Hello")
        store.save_message("s1", "assistant", "Hi there", sources="[]")
        history = store.get_history("s1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_clear(self, mock_db):
        store = ChatStore(QueryBuilder(mock_db._db))
        store.save_message("s1", "user", "Hello")
        store.clear_history("s1")
        assert len(store.get_history("s1")) == 0


class TestVectorStore:
    def test_upsert_and_get_kp_vector(self, mock_db):
        store = VectorStore(QueryBuilder(mock_db._db))
        mock_db._db.conn.execute(
            "INSERT INTO knowledge_points (id, name, description, quality) "
            "VALUES ('v_kp', 'Test', 'Test', 'draft')")
        mock_db._db.conn.commit()
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        store.upsert_kp_vector("v_kp", vec)
        result = store.get_kp_vector("v_kp")
        assert result is not None
        assert len(result) == 3
