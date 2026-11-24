"""Integration tests for attention-based fragment extraction.

These require the local Qwen3 weights and are skipped automatically when
the model is not configured (EXAM_ATTENTION_MODEL unset or path missing).
"""

import os

import numpy as np
import pytest

from src.constants import ATTENTION_MODEL_PATH
from src.knowledge._attention import QwenAttentionExtractor


MODEL_READY = bool(ATTENTION_MODEL_PATH) and os.path.isdir(ATTENTION_MODEL_PATH)

pytestmark = pytest.mark.skipif(
    not MODEL_READY,
    reason="attention model not configured (set EXAM_ATTENTION_MODEL)",
)


QP = (
    "A student is designing a network for a small office. "
    "Describe two advantages of using a wireless network instead of a wired one."
)

MS = (
    "1 mark per bullet point to max 2 2 "
    "Wireless: Freedom of movement - can move between rooms and still receive data. "
    "Less cabling / expertise is needed. "
    "Easily expanded if more devices join the network."
)


def test_extractor_reports_available():
    extractor = QwenAttentionExtractor()
    assert extractor.available() is True


def test_extract_returns_scores():
    extractor = QwenAttentionExtractor()
    result = extractor.extract(QP, MS)

    assert result is not None
    assert result.qp_end > 0
    assert result.n_ms_tokens > 0
    assert len(result.scores) == result.n_ms_tokens
    assert result.scores.max() == pytest.approx(1.0, abs=1e-5)
    assert np.all(result.scores >= 0)


def test_extract_rejects_short_input():
    extractor = QwenAttentionExtractor()
    assert extractor.extract("short", MS) is None
    assert extractor.extract(QP, "short") is None


def test_extract_is_deterministic():
    extractor = QwenAttentionExtractor()
    a = extractor.extract(QP, MS)
    b = extractor.extract(QP, MS)
    assert a is not None and b is not None
    assert np.allclose(a.scores, b.scores, atol=1e-6)


def test_activation_units_from_real_scores():
    from src.constants import (
        ATTENTION_GATE_MAX_GAP,
        ATTENTION_GATE_MIN_LEN,
        ATTENTION_GATE_THRESHOLD,
    )
    from src.knowledge._attention import build_activation_units, gate_text

    extractor = QwenAttentionExtractor()
    result = extractor.extract(QP, MS)
    assert result is not None

    units = build_activation_units(
        result.tokens, result.scores, result.qp_end,
        threshold=ATTENTION_GATE_THRESHOLD,
        max_gap=ATTENTION_GATE_MAX_GAP,
        min_len=ATTENTION_GATE_MIN_LEN,
    )
    assert units, "expected at least one activation unit"
    for unit in units:
        assert unit["text"].strip()
        assert unit["end"] > unit["start"]
        assert 0.0 <= unit["content"] <= 1.0

    text = gate_text(units)
    assert text.strip()
