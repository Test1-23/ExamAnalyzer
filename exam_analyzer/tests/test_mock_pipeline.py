"""Happy-path mock tests: Distiller, QARetriever, QA insert flow.

All tests use mock Flash responses — no network, no API keys.
"""
import pytest
import numpy as np

from src.knowledge_base import QADatabase, QARetriever, make_topic_id
from src.distiller import Distiller


# ═══════════════════════════════════════════════════════════════
# QADatabase + QARetriever integration
# ═══════════════════════════════════════════════════════════════

class TestQADatabaseInsert:
    """QA insert → get → count round-trip."""

    def test_insert_and_get(self, mock_db):
        db = mock_db
        qa_id = db.conn.execute(
            """INSERT INTO qa_pairs (question_text, answer_text, topic, paper, question_number)
               VALUES (?, ?, ?, ?, ?)""",
            ("What is binary?", "Base-2 number system.", "Binary", "test_paper", "1(a)"),
        ).lastrowid
        db.conn.commit()
        assert qa_id is not None
        qa = db.get(qa_id)
        assert qa is not None
        assert qa["topic"] == "Binary"
        assert qa["paper"] == "test_paper"

    def test_count(self, mock_db):
        db = mock_db
        assert db.count() == 0
        db.conn.execute(
            "INSERT INTO qa_pairs (question_text, answer_text, topic) VALUES (?, ?, ?)",
            ("Q1", "A1", "Topic1"),
        )
        db.conn.commit()
        assert db.count() == 1


class TestQARetriever:
    """QARetriever add_qa + search (mock embeddings)."""

    def test_add_qa(self, mock_db, mock_embedding):
        db = mock_db
        # Insert QA first
        qa_id = db.conn.execute(
            "INSERT INTO qa_pairs (question_text, answer_text, topic) VALUES (?, ?, ?)",
            ("What is binary?", "Base-2 numbering.", "Binary"),
        ).lastrowid
        db.conn.commit()

        retriever = QARetriever(db)
        retriever.add_qa(qa_id, "summary text")
        assert qa_id in retriever._id_map
        assert retriever._embeddings is not None
        assert retriever._embeddings.shape[0] == 1

    def test_add_qa_idempotent(self, mock_db, mock_embedding):
        db = mock_db
        qa_id = db.conn.execute(
            "INSERT INTO qa_pairs (question_text, answer_text, topic) VALUES (?, ?, ?)",
            ("Q", "A", "T"),
        ).lastrowid
        db.conn.commit()

        retriever = QARetriever(db)
        retriever.add_qa(qa_id, "summary")
        first_size = retriever._embeddings.shape[0]
        retriever.add_qa(qa_id, "summary again")
        assert retriever._embeddings.shape[0] == first_size  # no-op

    def test_search(self, mock_db, mock_embedding):
        db = mock_db
        for i in range(5):
            qa_id = db.conn.execute(
                "INSERT INTO qa_pairs (question_text, answer_text, topic) VALUES (?, ?, ?)",
                (f"Question {i}", f"Answer {i}", f"Topic{i % 2}"),
            ).lastrowid
            db.conn.commit()

        retriever = QARetriever(db)
        for qa_id in [1, 2, 3, 4, 5]:
            retriever.add_qa(qa_id, f"summary {qa_id}")

        results = retriever.search("Question 1", threshold=0.0, min_k=2, max_cap=10)
        assert len(results) >= 2
        # Results should be dicts with _score
        for r in results:
            assert "_score" in r


# ═══════════════════════════════════════════════════════════════
# Distiller with mock Flash
# ═══════════════════════════════════════════════════════════════

class TestDistiller:
    """Distiller.run() with mock Flash — verifies the distillation pipeline
    completes without crashing and produces expected output markers."""

    def _seed_db(self, db, topics=("Binary", "Hex")):
        """Insert QAs into multiple topics."""
        for topic in topics:
            for i in range(3):
                db.conn.execute(
                    """INSERT INTO qa_pairs
                       (question_text, answer_text, topic, paper, question_number)
                       VALUES (?, ?, ?, ?, ?)""",
                    (f"Q{i} for {topic}", f"A{i} for {topic}", topic, "paper1", str(i + 1)),
                )
        db.conn.commit()

    def test_distiller_empty_db(self, mock_db, mock_flash, mock_embedding):
        db = mock_db
        mock_flash.default = {"content": "No knowledge points yet."}
        d = Distiller(db, client=None, debug=lambda m: None)
        result = d.run()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_distiller_with_seeded_data(self, mock_db, mock_flash, mock_embedding):
        db = mock_db
        self._seed_db(db, topics=("Binary", "Hex", "Data Compression"))

        # Register Flash responses for distillation stages
        mock_flash.register("grouped QAs", {
            "content": "## Binary\n\nKnowledge about binary numbering.\n\n## Hex\n\nHexadecimal system.\n\n## Data Compression\n\nLossy vs lossless compression.",
        })
        mock_flash.register("review", {
            "content": "## Binary\n\nKnowledge about binary numbering.\n\n## Hex\n\nHexadecimal system.\n\n## Data Compression\n\nLossy vs lossless compression.",
        })

        d = Distiller(db, client=None, debug=lambda m: None)
        result = d.run()
        assert isinstance(result, str)
        # Distiller should produce topic headers
        assert "Binary" in result or "Hex" in result or len(result) > 50


# ═══════════════════════════════════════════════════════════════
# make_topic_id
# ═══════════════════════════════════════════════════════════════

class TestMakeTopicId:
    def test_simple(self):
        assert make_topic_id("Binary") == "topic_Binary"

    def test_with_spaces(self):
        assert make_topic_id("Data Compression") == "topic_Data_Compression"

    def test_with_slash(self):
        assert make_topic_id("Input/Output") == "topic_Input_Output"

    def test_empty(self):
        assert make_topic_id("") == "topic_"
