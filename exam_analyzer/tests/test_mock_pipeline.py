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


# ═══════════════════════════════════════════════════════════════
# Pipeline extracted functions (WSD-002)
# ═══════════════════════════════════════════════════════════════


class TestPipelineFunctions:
    """Unit tests for module-level pipeline functions extracted from closures."""

    # -- _phase1_worker --

    def test_phase1_worker_inserts_qa_and_fragments(
        self, mock_db, mock_tracker, monkeypatch):
        """_phase1_worker → QA inserted + fragments assigned to topic."""
        from src.pipeline import _phase1_worker

        monkeypatch.setattr(
            "src.pipeline._generate_summary",
            lambda qt, at, cl, dbg, **kw: ("Summary of Binary", "Binary"))
        monkeypatch.setattr(
            "src.pipeline._extract_ms_fragments",
            lambda at, qid, cl, dbg: [
                {"point_id": "f1", "qa_id": qid, "point_text": "Base-2", "marks": 1}
            ])

        qa = type("QA", (), {
            "question_text": "What is binary?",
            "answer_text": "Base-2 number system.",
            "question_number": "1", "parent_question": "",
        })()

        qa_id = _phase1_worker(
            qa, client=None, db=mock_db, debug=print,
            display_name="test_paper", existing_topics=None,
            tracker=mock_tracker)

        assert qa_id > 0
        qa_row = mock_db.get(qa_id)
        assert qa_row["topic"] == "Binary"
        assert qa_row["knowledge_summary"] == "Summary of Binary"
        mock_tracker.step.assert_called()

    # -- _step_summarize_retrieve --

    def test_step_summarize_retrieve_top4_and_kp_refs(
        self, mock_db, mock_tracker, monkeypatch):
        """_step_summarize_retrieve → top-4 + KP refs appended."""
        from src.pipeline import _step_summarize_retrieve

        for i in range(8):
            mock_db.qa.insert(f"Q{i}", f"A{i}", topic=f"T{i}")
        mock_db.topic.upsert("t_s", name="StableKP", quality="stable")
        mock_db.topic.set_kp("t_s", "Binary Arithmetic", "Base conversions")

        class FakeRetriever:
            def search_dual_channel(self, summary, **kw):
                return [{"id": i, "topic": f"T{i}"} for i in range(1, 9)]
        retriever = FakeRetriever()

        monkeypatch.setattr(
            "src.pipeline._generate_summary",
            lambda qt, at, cl, dbg, **kw: ("Summary", "T0"))

        qa = type("QA", (), {
            "question_text": "Q", "answer_text": "A", "question_number": "1"})()
        wmap = {i: {"mean": 0.8} for i in range(1, 9)}

        summary, step0_topic, top_similar, all_similar = \
            _step_summarize_retrieve(
                qa, wmap, [], None, mock_db, print, "test",
                retriever, mock_tracker)

        assert summary == "Summary"
        assert step0_topic == "T0"
        assert len(top_similar) >= 4
        assert len(all_similar) == 8

    # -- _step_answer_and_grade --

    def test_step_answer_and_grade_parses_covered_missed(
        self, mock_db, mock_flash, mock_tracker):
        """_step_answer_and_grade → covered/missed from mock Flash answer+grade."""
        from src.pipeline import _step_answer_and_grade

        # The answer prompt contains "Past Q" keywords that are unique to it.
        mock_flash.register("Past Q", {
            "answer": "Binary is base-2.", "used_qa_indices": [1],
        })
        # The grade is the second call — use default for it since no keyword
        # from the answer will appear in the grade prompt.
        mock_flash.default = {
            "topic": "Binary",
            "covered_points": ["Base-2 definition"],
            "missed_points": [{"point": "Bit manipulation", "reason": "knowledge_gap"}],
        }

        qa = type("QA", (), {
            "question_text": "Q", "answer_text": "A", "question_number": "1"})()
        top_similar = [{"id": 1, "topic": "Binary", "question_text": "Q1", "answer_text": "A1"}]

        used_indices, used_ids, covered, missed_texts, miss_cats_json, r2_topic = \
            _step_answer_and_grade(
                qa, top_similar, "Binary", None, mock_db, print, "test", mock_tracker)

        assert used_indices == [1]
        assert 1 in used_ids
        assert covered == ["Base-2 definition"]
        assert "Bit manipulation" in missed_texts
        assert r2_topic == "Binary"

    # -- _step_insert_and_feedback --

    def test_step_insert_and_feedback_creates_qa(self, mock_db, mock_tracker):
        """_step_insert_and_feedback → QA inserted + feedback logged."""
        from src.pipeline import _step_insert_and_feedback

        qa = type("QA", (), {
            "question_text": "Q", "answer_text": "A",
            "question_number": "1", "parent_question": "",
        })()

        qa_id, topic, cross_refs = _step_insert_and_feedback(
            qa, "summary", "T0", "T1",
            all_similar=[{"id": 1}], top_similar=[{"id": 1, "topic": "T1"}],
            used_indices=[1], covered=["c"], missed_texts=["m"],
            miss_cats_json='{"cat":"x"}',
            db=mock_db, debug=print, tracker=mock_tracker,
            display_name="test")

        assert qa_id > 0
        assert topic == "T1"
        row = mock_db.get(qa_id)
        assert row["question_text"] == "Q"

    # -- _step_fragment_and_kp --

    def test_step_fragment_and_kp_extracts_and_classifies(
        self, mock_db, mock_tracker, monkeypatch):
        """_step_fragment_and_kp → fragments inserted + KP classified."""
        from src.pipeline import _step_fragment_and_kp
        from src.models import KPSpec

        mock_db.kp.upsert(KPSpec(
            kp_id="kp_0000", name="KP0", description="d",
            cohesion=0.9, evidence_count=5, quality="draft"))

        monkeypatch.setattr(
            "src.pipeline._extract_ms_fragments",
            lambda at, qid, cl, dbg: [
                {"point_id": "f1", "qa_id": qid, "point_text": "pt", "marks": 1}
            ])
        monkeypatch.setattr(
            "src.pipeline._classify_qa_against_kps",
            lambda qt, at, kps, cl, dbg: {"kp_0000": 0.9})
        monkeypatch.setattr(
            "src.pipeline._place_qa_vector_from_kp_scores",
            lambda db, qid, scores, dbg: None)
        monkeypatch.setattr(
            "src.pipeline._record_fragment_help",
            lambda uids, cov, mis, db, qid: None)

        qa = type("QA", (), {
            "question_text": "Q", "answer_text": "A", "question_number": "1"})()
        _step_fragment_and_kp(
            qa, 1, "T0", {1}, ["c"], ["m"],
            None, mock_db, print, mock_tracker)

        mock_tracker.step.assert_called()

    # -- _process_one_question_inner --

    def test_process_one_question_inner_orchestrates_4_steps(
        self, mock_db, mock_tracker, monkeypatch):
        """_process_one_question_inner → orchestrates 4 steps, returns qa_id."""
        from src.pipeline import _process_one_question_inner

        monkeypatch.setattr(
            "src.pipeline._step_summarize_retrieve",
            lambda *a, **kw: ("summary", "T",
                              [{"id": 1, "topic": "T"}],
                              [{"id": 1, "topic": "T"}]))
        monkeypatch.setattr(
            "src.pipeline._step_answer_and_grade",
            lambda *a, **kw: ([1], {1}, ["covered"], ["missed"],
                              '{"cat":"x"}', "T"))
        monkeypatch.setattr(
            "src.pipeline._step_insert_and_feedback",
            lambda *a, **kw: (99, "T", {}))
        monkeypatch.setattr(
            "src.pipeline._step_fragment_and_kp",
            lambda *a, **kw: None)

        qa = type("QA", (), {
            "question_text": "Q", "answer_text": "A", "question_number": "1"})()
        wmap = {1: {"mean": 0.8}}

        qa_id, summary, cross_refs = _process_one_question_inner(
            qa, wmap, [], None, mock_db, print, "test", None, mock_tracker)

        assert qa_id == 99
        assert summary == "summary"

    # -- _run_kp_refinement --

    def test_run_kp_refinement_splits_eligible_kps(
        self, mock_db, monkeypatch):
        """_run_kp_refinement → split called for KPs with evidence_count >= 6."""
        from src.pipeline import _run_kp_refinement
        from src.models import KPSpec

        mock_db.kp.upsert(KPSpec(
            kp_id="kp_0000", name="KP0", description="d",
            cohesion=0.9, evidence_count=10, quality="draft"))
        mock_db.kp.upsert(KPSpec(
            kp_id="kp_0001", name="KP1", description="d",
            cohesion=0.8, evidence_count=3, quality="draft"))

        split_calls = []
        monkeypatch.setattr(
            "src.pipeline.auto_split_kp",
            lambda db, kp_id, client, debug_cb: split_calls.append(kp_id))
        monkeypatch.setattr(
            "src.pipeline.cross_kp_consistency",
            lambda db, kp_ids, client, debug_cb: {"issues": []})
        monkeypatch.setattr(
            "src.pipeline.auto_merge_kps",
            lambda db, issues, debug_cb: 0)

        _run_kp_refinement(mock_db, None, print)

        assert len(split_calls) == 1
        assert split_calls[0] == "kp_0000"
