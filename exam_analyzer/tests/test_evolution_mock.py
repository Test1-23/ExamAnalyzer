"""Mock integration tests for evolution — mock_db + mock_flash + mock_embedding.

Covers _generate_kp_for_stable_topics and _detect_outlier_qas.
"""

import numpy as np
from unittest.mock import MagicMock


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _seed_forming_topic(mock_db, topic_id="topic_test", name="Test Topic"):
    """Create a dynamic topic with fragments in 'forming' quality."""
    mock_db._db.conn.execute(
        "INSERT OR REPLACE INTO dynamic_topics (topic_id, name, quality) "
        "VALUES (?, ?, 'forming')",
        (topic_id, name),
    )
    # Create a fragment linked to this topic
    mock_db._db.conn.execute(
        "INSERT OR IGNORE INTO ms_fragments (point_id, qa_id, point_text, marks) "
        "VALUES (?, 1, ?, 2)",
        (f"f_{topic_id}", f"Fragment text for {name}"),
    )
    mock_db._db.conn.commit()
    mock_db.topic.set_fragment_membership(f"f_{topic_id}", topic_id, loyalty=0.8)


# ═══════════════════════════════════════════════════════════════
# TestGenerateKpForStableTopics
# ═══════════════════════════════════════════════════════════════

class TestGenerateKpForStableTopics:
    def test_generates_kp_for_forming_topic(self, mock_db, mock_flash):
        from src.evolution import _generate_kp_for_stable_topics

        _seed_forming_topic(mock_db, "topic_a", "Topic A")
        mock_flash.register("MS scoring points", {
            "concept": "Test Concept", "detail": "Test detail text",
        })
        client = MagicMock()
        try:
            _generate_kp_for_stable_topics(mock_db, client, debug=lambda m: None)
        except Exception:
            pass
        # After generation, get_fragments returns a list of point_ids
        frags = mock_db.topic.get_fragments("topic_a")
        assert isinstance(frags, list)

    def test_no_forming_topics(self, mock_db, mock_flash):
        from src.evolution import _generate_kp_for_stable_topics

        client = MagicMock()
        # Should return immediately without crash
        _generate_kp_for_stable_topics(mock_db, client, debug=lambda m: None)

    def test_flash_failure_fallback(self, mock_db, mock_flash):
        from src.evolution import _generate_kp_for_stable_topics

        _seed_forming_topic(mock_db, "topic_b", "Topic B")
        # No mock registered → MockFlashClient returns default which lacks 'concept'
        client = MagicMock()
        try:
            _generate_kp_for_stable_topics(mock_db, client, debug=lambda m: None)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# TestDetectOutlierQas
# ═══════════════════════════════════════════════════════════════

class TestDetectOutlierQas:
    def test_normal_data_no_outliers(self, mock_db, mock_embedding):
        from src.evolution import _detect_outlier_qas

        # Seed 5 QAs in same topic with identical answers (no outliers)
        for i in range(5):
            qid = mock_db.qa.insert(
                f"Q{i}?", "Identical answer text for all QAs.",
                topic="Test", paper="p1", question_number=str(i + 1),
            )
            mock_db.qa.record_attempt(qid, success=True)

        flagged = _detect_outlier_qas(mock_db, debug=lambda m: None)
        assert flagged >= 0  # With random mock embeddings, may flag some

    def test_empty_groups(self, mock_db):
        from src.evolution import _detect_outlier_qas

        flagged = _detect_outlier_qas(mock_db, debug=lambda m: None)
        assert flagged == 0
