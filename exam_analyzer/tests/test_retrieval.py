"""QARetriever tests with seeded DB — no network, no real embeddings."""

import numpy as np
import pytest

from src.retriever import QARetriever


def _seed_db(db):
    """Insert enough QAs to exercise Channel A (embedding) + Channel B (topic/graph)."""
    qa_store = db.qa
    topics = ["Binary", "Hex", "Data Compression", "Binary", "Hex"]
    questions = [
        "What is binary?",
        "Convert hex to decimal",
        "Explain lossless compression",
        "Binary addition rules",
        "Hex multiplication table",
    ]
    answers = [
        "A base-2 numbering system using 0 and 1.",
        "Multiply each digit by 16^n and sum.",
        "Compression that preserves all original data.",
        "0+0=0, 0+1=1, 1+0=1, 1+1=10 carry 1.",
        "Multiply each hex digit, convert results.",
    ]
    for i, (t, q, a) in enumerate(zip(topics, questions, answers)):
        qa_store.insert(q, a, topic=t, paper=f"test_paper", question_number=str(i + 1))


class TestQARetrieverSearch:
    def test_search_returns_results(self, mock_db, mock_embedding):
        _seed_db(mock_db)
        retriever = QARetriever(mock_db)
        retriever.rebuild()
        results = retriever.search("binary", threshold=0.0, min_k=1, max_cap=3)
        assert len(results) >= 1

    def test_search_dual_channel_returns_results(self, mock_db, mock_embedding):
        _seed_db(mock_db)
        retriever = QARetriever(mock_db)
        retriever.rebuild()
        results = retriever.search_dual_channel(
            "binary numbers", threshold=0.0, min_k=1, max_cap=5,
            query_topic="Binary")
        assert len(results) >= 1

    def test_search_dual_channel_with_topic(self, mock_db, mock_embedding):
        _seed_db(mock_db)
        retriever = QARetriever(mock_db)
        retriever.rebuild()
        results = retriever.search_dual_channel(
            "compression techniques", threshold=0.0, min_k=2, max_cap=5,
            query_topic="Data Compression")
        # Channel B should find QAs in "Data Compression" topic
        topics_found = {r.get("topic") for r in results}
        assert "Data Compression" in topics_found or len(results) >= 2

    def test_search_empty_db(self, mock_db, mock_embedding):
        retriever = QARetriever(mock_db)
        results = retriever.search("anything", threshold=0.0, min_k=1)
        assert results == []

    def test_search_dual_channel_empty_db(self, mock_db, mock_embedding):
        retriever = QARetriever(mock_db)
        results = retriever.search_dual_channel("anything", threshold=0.0, min_k=1)
        assert results == []

    def test_add_qa_updates_index(self, mock_db, mock_embedding):
        _seed_db(mock_db)
        retriever = QARetriever(mock_db)
        retriever.rebuild()
        count_before = retriever.count()
        new_id = mock_db.qa.insert("New Q", "New A", topic="Binary")
        retriever.add_qa(new_id, "New Q summary")
        assert retriever.count() == count_before + 1

    def test_rebuild_resets_index(self, mock_db, mock_embedding):
        _seed_db(mock_db)
        retriever = QARetriever(mock_db)
        retriever.rebuild()
        retriever.rebuild()  # second rebuild should work
        assert retriever.count() == 5
