"""Integration test for the knowledge-space layout (mock encoder, temp DB)."""

import re
import zlib

import numpy as np
import pytest

from src.knowledge._geometry import LayoutConfig, normalize_rows
from src.knowledge._layout import run_layout


class FakeEncoder:
    """Deterministic bag-of-tokens encoder: shared words -> similar vectors."""

    def __init__(self, dim: int = 32):
        self.dim = dim
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        rows = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float64)
            for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
                vec[zlib.crc32(token.encode()) % self.dim] += 1.0
            rows.append(vec)
        return normalize_rows(np.stack(rows))


def _seed_db(db):
    """Two papers; the same knowledge point recurs across both."""
    specs = [
        # (paper, question, answer, topic)
        ("p_s23_11", "convert binary 1010 to denary",
         "1 mark for correct answer / showing working / 1010 binary equals 10 denary", "Binary Conversion"),
        ("p_s23_11", "explain database primary key",
         "primary key uniquely identifies a record / no duplicates allowed", "Database Keys"),
        ("p_s24_11", "convert denary 10 to binary",
         "1 mark for correct answer / 10 denary equals 1010 binary", "Binary Conversion"),
        ("p_s24_11", "describe a foreign key",
         "foreign key links two tables / references primary key", "Database Keys"),
    ]
    ids = []
    for paper, question, answer, topic in specs:
        ids.append(db.qa.insert(
            question_text=question,
            answer_text=answer,
            topic=topic,
            paper=paper,
            question_number="1",
            parent_question="1",
            knowledge_summary=question,
        ))
    return ids


def test_run_layout_persists_and_is_idempotent(mock_db):
    from src.knowledge._fragments import split_fragments  # noqa: F401  (sanity)

    qa_ids = _seed_db(mock_db)

    # derive fragments deterministically, then layout from DB fragments
    first = run_layout(
        mock_db,
        FakeEncoder(),
        LayoutConfig(epochs=6, radius=0.35, seed=0),
        persist=True,
        split_missing=True,
        encoder_version="fake:v1",
    )

    assert first.fragment_count > 0
    assert first.qa_count == len(qa_ids)
    assert first.kp_count >= 1
    assert first.persisted is True
    assert first.collapsed is False

    assert mock_db.vector.count_positions("fragment") == first.fragment_count
    assert mock_db.vector.count_positions("qa") == first.qa_count
    assert mock_db.vector.count_positions("kp") == first.kp_count

    memberships = mock_db.kp.get_kp_ids_for_qa(qa_ids[0])
    assert memberships, "every QA must belong to a KP"
    topics = mock_db.topic.get_by_quality("stable") + mock_db.topic.get_by_quality("embryonic")
    assert len(topics) >= first.kp_count

    # rerun: fragments are now in the DB, so no re-splitting; ids stay stable
    second = run_layout(
        mock_db,
        FakeEncoder(),
        LayoutConfig(epochs=6, radius=0.35, seed=0),
        persist=True,
        split_missing=False,
        encoder_version="fake:v1",
    )
    assert second.fragment_count == first.fragment_count
    assert second.kp_count == first.kp_count
    assert second.metrics["n_clusters"] == first.metrics["n_clusters"]
    assert mock_db.kp.get_kp_ids_for_qa(qa_ids[0]) == memberships


def test_run_layout_dry_run_writes_nothing(mock_db):
    _seed_db(mock_db)
    result = run_layout(
        mock_db,
        FakeEncoder(),
        LayoutConfig(epochs=3, seed=1),
        persist=False,
        split_missing=True,
    )
    assert result.persisted is False
    assert mock_db.vector.count_positions() == 0
    assert mock_db.kp.get_all() == []


def test_run_layout_rejects_empty_input(mock_db):
    with pytest.raises(RuntimeError):
        run_layout(mock_db, FakeEncoder(), LayoutConfig(epochs=2), persist=False)
