"""Unit tests for pure functions (no API/DB dependencies)."""
import os
import sys
import json
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---- src/config.py ----

class TestConfig:
    def test_load_config_returns_dict(self):
        from src.config import load_config
        config = load_config()
        assert isinstance(config, dict)
        assert "api_url" in config
        assert "api_key" in config

    def test_load_config_env_override(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-123")
        from src.config import load_config
        config = load_config()
        assert config["api_key"] == "test-key-123"

    def test_load_config_missing_file(self, monkeypatch):
        monkeypatch.setattr("src.config.os.path.exists", lambda p: False)
        from src.config import load_config
        config = load_config()
        assert isinstance(config, dict)


# ---- src/file_pairer.py ----

class TestFilePairer:
    def test_parse_exam_filename_valid(self):
        from src.file_pairer import parse_exam_filename
        subject, paper_type, variant = parse_exam_filename("9618_s23_qp_11.pdf")
        assert subject == "9618_s23"
        assert paper_type == "qp"
        assert variant == "11"

    def test_parse_exam_filename_ms(self):
        from src.file_pairer import parse_exam_filename
        subject, paper_type, _ = parse_exam_filename("9618_s23_ms_11.pdf")
        assert paper_type == "ms"

    def test_parse_exam_filename_invalid(self):
        from src.file_pairer import parse_exam_filename
        with pytest.raises(ValueError):
            parse_exam_filename("invalid.pdf")


# ---- src/embedding_cluster.py (cluster_by_cosine) ----

class TestClusterByCosine:
    def test_empty_vectors(self):
        import numpy as np
        from src.embedding_cluster import cluster_by_cosine
        groups = cluster_by_cosine(np.empty((0, 3)), threshold=0.5)
        assert groups == []

    def test_single_vector(self):
        import numpy as np
        from src.embedding_cluster import cluster_by_cosine
        vecs = np.array([[1.0, 0.0, 0.0]])
        groups = cluster_by_cosine(vecs, threshold=0.5)
        assert groups == []  # single element can't form group of 2

    def test_two_similar_vectors(self):
        import numpy as np
        from src.embedding_cluster import cluster_by_cosine
        vecs = np.array([[1.0, 0.0], [0.99, 0.01]])
        groups = cluster_by_cosine(vecs, threshold=0.5)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_two_dissimilar_vectors(self):
        import numpy as np
        from src.embedding_cluster import cluster_by_cosine
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        groups = cluster_by_cosine(vecs, threshold=0.9)
        assert groups == []

    def test_min_group_size(self):
        import numpy as np
        from src.embedding_cluster import cluster_by_cosine
        vecs = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
        groups = cluster_by_cosine(vecs, threshold=0.5, min_group_size=3)
        assert groups == []  # only 2 similar, need 3


# ---- src/knowledge_base.py (make_topic_id) ----

class TestMakeTopicId:
    def test_simple_topic(self):
        from src.knowledge_base import make_topic_id
        assert make_topic_id("Data Compression") == "topic_Data_Compression"

    def test_topic_with_slash(self):
        from src.knowledge_base import make_topic_id
        assert make_topic_id("Input/Output Devices") == "topic_Input_Output_Devices"

    def test_topic_no_special_chars(self):
        from src.knowledge_base import make_topic_id
        assert make_topic_id("Binary") == "topic_Binary"
