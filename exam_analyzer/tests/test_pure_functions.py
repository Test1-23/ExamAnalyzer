"""Unit tests for pure functions — zero dependencies, zero mocks needed.

Covers:
- _evaluate_signal: 12 boundary combinations for difficulty classification
- make_topic_id: topic name sanitization
"""

import pytest

from src.offline.difficulty import _evaluate_signal
from src.retriever import make_topic_id


class TestEvaluateSignal:
    """12 boundary combinations — independently testable pure function."""

    # -- bi_threshold present, ia_threshold present --
    def test_clear_basic(self):
        v, b = _evaluate_signal(0.20, 0.30, 0.60, 0.10)
        assert v == "basic" and not b

    def test_boundary_basic(self):
        v, b = _evaluate_signal(0.31, 0.30, 0.60, 0.10)
        assert v == "basic" and b

    def test_clear_intermediate(self):
        v, b = _evaluate_signal(0.45, 0.30, 0.60, 0.10)
        assert v == "intermediate" and not b

    def test_boundary_intermediate(self):
        v, b = _evaluate_signal(0.61, 0.30, 0.60, 0.10)
        assert v == "intermediate" and b

    def test_clear_advanced(self):
        v, b = _evaluate_signal(0.75, 0.30, 0.60, 0.10)
        assert v == "advanced" and not b

    # -- bi_threshold present, ia_threshold missing --
    def test_bi_only_clear_basic(self):
        v, b = _evaluate_signal(0.20, 0.30, None, 0.10)
        assert v == "basic" and not b

    def test_bi_only_boundary_basic(self):
        v, b = _evaluate_signal(0.31, 0.30, None, 0.10)
        assert v == "basic" and b

    def test_bi_only_above_becomes_intermediate(self):
        v, b = _evaluate_signal(0.45, 0.30, None, 0.10)
        assert v == "intermediate" and not b

    # -- bi_threshold missing, ia_threshold present --
    def test_ia_only_clear_intermediate(self):
        v, b = _evaluate_signal(0.45, None, 0.60, 0.10)
        assert v == "intermediate" and not b

    def test_ia_only_boundary_intermediate(self):
        v, b = _evaluate_signal(0.61, None, 0.60, 0.10)
        assert v == "intermediate" and b

    def test_ia_only_clear_advanced(self):
        v, b = _evaluate_signal(0.75, None, 0.60, 0.10)
        assert v == "advanced" and not b

    # -- neither boundary present --
    def test_no_boundaries_defaults_intermediate(self):
        v, b = _evaluate_signal(0.50, None, None, 0.10)
        assert v == "intermediate" and not b

    # -- margin=0 edge cases --
    def test_zero_margin_exact_match(self):
        v, b = _evaluate_signal(0.30, 0.30, 0.60, 0.0)
        assert v == "basic" and b  # sig_val == bi * (1+0) -> boundary


class TestMakeTopicId:
    def test_simple_name(self):
        assert make_topic_id("Data Compression") == "topic_Data_Compression"

    def test_with_slash(self):
        assert make_topic_id("I/O Devices") == "topic_I_O_Devices"

    def test_single_word(self):
        assert make_topic_id("Binary") == "topic_Binary"
