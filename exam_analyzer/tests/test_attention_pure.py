"""Pure-function tests for attention-based fragment extraction.

No model weights required: every test here exercises the deterministic
math and token helpers only.
"""

import numpy as np
import pytest

from src.knowledge._attention import (
    AttentionResult,
    alignment_scores,
    build_activation_units,
    clean_token,
    gate_text,
    is_content_token,
    is_filler_token,
    projected_attention,
)


# ============================================================
# Token helpers
# ============================================================


@pytest.mark.parametrize("token,expected", [
    ("\u0120graph", " graph"),
    ("\u2581nodes", " nodes"),
    ("plain", "plain"),
    ("a\nb", "a b"),
])
def test_clean_token(token, expected):
    assert clean_token(token) == expected


@pytest.mark.parametrize("token,expected", [
    ("\u0120graph", True),
    ("\u0120the", False),
    ("\u0120of", False),
    (",", False),
    ("\u0120123", False),
    ("\u0120vertices", True),
])
def test_is_content_token(token, expected):
    assert is_content_token(token) is expected


@pytest.mark.parametrize("token,expected", [
    ("...", True),
    ("\u2022", True),
    ("MARK", True),
    ("SCHEME", True),
    ("7(b)(i)", True),
    ("\u0120graph", False),
])
def test_is_filler_token(token, expected):
    assert is_filler_token(token) is expected


# ============================================================
# Projected attention
# ============================================================


def test_projected_attention_shape():
    n_heads, n_tok, head_dim, hidden = 2, 5, 4, 8
    rng = np.random.default_rng(0)
    attn = rng.random((n_heads, n_tok, n_tok)).astype(np.float32)
    attn /= attn.sum(axis=-1, keepdims=True)
    values = rng.random((n_heads, n_tok, head_dim)).astype(np.float32)
    w_o = rng.random((n_heads, head_dim, hidden)).astype(np.float32)

    out = projected_attention(attn, values, w_o)

    assert out.shape == (n_tok, n_tok)
    assert np.all(out >= 0)
    assert np.all(np.isfinite(out))


def test_projected_attention_matches_naive():
    """Gram-matrix form must equal the explicit vector form."""
    n_heads, n_tok, head_dim, hidden = 2, 4, 3, 5
    rng = np.random.default_rng(1)
    attn = rng.random((n_heads, n_tok, n_tok)).astype(np.float32)
    attn /= attn.sum(axis=-1, keepdims=True)
    values = rng.random((n_heads, n_tok, head_dim)).astype(np.float32)
    w_o = rng.random((n_heads, head_dim, hidden)).astype(np.float32)

    fast = projected_attention(attn, values, w_o)

    # Naive: build the hidden-dim vector per (i, j) and take its norm.
    projected_values = np.einsum("hcd,hdo->hco", values, w_o)  # (H, T, hidden)
    naive = np.zeros((n_tok, n_tok), dtype=np.float32)
    for i in range(n_tok):
        for j in range(n_tok):
            acc = np.zeros(hidden, dtype=np.float32)
            for h in range(n_heads):
                acc += attn[h, i, j] * projected_values[h, j]
            naive[i, j] = np.linalg.norm(acc)

    assert np.allclose(fast, naive, rtol=1e-4, atol=1e-4)


def test_projected_attention_chunking_is_stable():
    """Result must not depend on the column block size."""
    n_heads, n_tok, head_dim, hidden = 2, 7, 4, 6
    rng = np.random.default_rng(2)
    attn = rng.random((n_heads, n_tok, n_tok)).astype(np.float32)
    attn /= attn.sum(axis=-1, keepdims=True)
    values = rng.random((n_heads, n_tok, head_dim)).astype(np.float32)
    w_o = rng.random((n_heads, head_dim, hidden)).astype(np.float32)

    a = projected_attention(attn, values, w_o, chunk=2)
    b = projected_attention(attn, values, w_o, chunk=64)

    assert np.allclose(a, b, rtol=1e-5, atol=1e-5)


# ============================================================
# Alignment scores
# ============================================================


def test_alignment_scores_normalised():
    tokens = ["\u0120graph", "\u0120nodes", "\u0120the", "\u0120edges",
              "\u0120weighted", "\u0120paths"]
    qp_end = 3
    projected = np.zeros((6, 6), dtype=np.float32)
    # QP content rows (0, 1) align to MS tokens 3 and 4.
    projected[0, 3] = 5.0
    projected[1, 4] = 2.0

    scores = alignment_scores(projected, qp_end, tokens)

    assert len(scores) == 3
    assert scores.max() == pytest.approx(1.0)
    assert scores[0] == pytest.approx(1.0)   # token 3 is the peak
    assert scores[1] == pytest.approx(0.4)   # token 4 is 2/5
    assert scores[2] == pytest.approx(0.0)


def test_alignment_scores_empty_when_no_content_rows():
    tokens = ["\u0120the", "\u0120of", "\u0120a", "\u0120graph"]
    projected = np.ones((4, 4), dtype=np.float32)
    scores = alignment_scores(projected, 3, tokens)
    assert len(scores) == 1
    assert scores[0] == 0.0


def test_alignment_scores_handles_degenerate_bounds():
    tokens = ["\u0120graph"] * 4
    projected = np.ones((4, 4), dtype=np.float32)
    assert len(alignment_scores(projected, 0, tokens)) == 0
    assert len(alignment_scores(projected, 4, tokens)) == 0


# ============================================================
# Activation units
# ============================================================


def test_build_activation_units_groups_contiguous():
    tokens = ["<s>", "\u0120q", "\u0120q2", "\u0120graph", "\u0120nodes",
              "\u0120c", "\u0120d", "\u0120e"]
    qp_end = 3
    scores = np.array([0.9, 0.8, 0.1, 0.05, 0.7], dtype=np.float32)

    units = build_activation_units(tokens, scores, qp_end,
                                   threshold=0.4, max_gap=1, min_len=2)

    assert len(units) == 1
    assert units[0]["start"] == 0
    assert units[0]["end"] == 2
    assert "graph" in units[0]["text"] and "nodes" in units[0]["text"]


def test_build_activation_units_respects_min_len():
    tokens = ["<s>", "\u0120q", "\u0120graph", "\u0120nodes"]
    scores = np.array([0.9, 0.1], dtype=np.float32)
    units = build_activation_units(tokens, scores, 2,
                                   threshold=0.4, max_gap=1, min_len=2)
    assert units == []


def test_build_activation_units_empty_when_below_threshold():
    tokens = ["<s>", "\u0120q", "\u0120a"]
    scores = np.array([0.1], dtype=np.float32)
    assert build_activation_units(tokens, scores, 2, 0.4, 1, 2) == []


def test_build_activation_units_merges_across_gap():
    # MS tokens: graph(0) x(1) nodes(2); graph and nodes are hot, x is cold.
    # max_gap=1 lets the run bridge the single cold token.
    tokens = ["<s>", "\u0120q", "\u0120graph", "\u0120x", "\u0120nodes"]
    scores = np.array([0.9, 0.1, 0.8], dtype=np.float32)
    units = build_activation_units(tokens, scores, 2,
                                   threshold=0.4, max_gap=1, min_len=2)
    assert len(units) == 1
    assert units[0]["start"] == 0 and units[0]["end"] == 3


def test_build_activation_units_keeps_hot_pair_without_gap():
    # Two adjacent hot content tokens form a run of length 2 with max_gap=0.
    tokens = ["<s>", "\u0120q", "\u0120graph", "\u0120nodes"]
    scores = np.array([0.9, 0.8], dtype=np.float32)
    units = build_activation_units(tokens, scores, 2,
                                   threshold=0.4, max_gap=0, min_len=2)
    assert len(units) == 1
    assert units[0]["start"] == 0 and units[0]["end"] == 2


def test_build_activation_units_splits_when_gap_too_large():
    # Same tokens but max_gap=0 forces a split into two single-token runs,
    # both below min_len, so nothing survives.
    tokens = ["<s>", "\u0120q", "\u0120graph", "\u0120x", "\u0120nodes"]
    scores = np.array([0.9, 0.1, 0.8], dtype=np.float32)
    units = build_activation_units(tokens, scores, 2,
                                   threshold=0.4, max_gap=0, min_len=2)
    assert units == []


def test_gate_text_joins_units():
    units = [{"text": "alpha"}, {"text": "beta"}, {"text": ""}]
    assert gate_text(units) == "alpha ... beta"
    assert gate_text(units, separator=" | ") == "alpha | beta"


# ============================================================
# AttentionResult
# ============================================================


def test_attention_result_slices_ms_tokens():
    result = AttentionResult(
        tokens=["<s>", "q1", "q2", "SEP", "m1", "m2"],
        qp_end=4,
        scores=np.array([0.5, 0.6], dtype=np.float32),
    )
    assert result.ms_tokens == ["m1", "m2"]
    assert result.n_ms_tokens == 2


def test_attention_result_handles_empty():
    result = AttentionResult()
    assert result.ms_tokens == []
    assert result.n_ms_tokens == 0


# ============================================================
# Layer-major batch extraction
# ============================================================


def test_extract_batch_empty_input():
    from src.knowledge._attention import extract_batch
    assert extract_batch(None, []) == []


def test_extract_batch_returns_one_slot_per_pair():
    """Every input pair must map to exactly one output slot."""
    from src.knowledge._attention import QwenAttentionExtractor, extract_batch

    ex = QwenAttentionExtractor(model_path="/nonexistent/path")
    pairs = [("q" * 50, "a" * 50), ("q" * 50, "a" * 50), ("q" * 50, "a" * 50)]
    out = extract_batch(ex, pairs)
    assert len(out) == len(pairs)
    assert all(r is None for r in out)  # model unavailable -> all None


def test_extract_batch_skips_short_inputs():
    """Short QP/MS pairs are skipped without raising."""
    from src.knowledge._attention import QwenAttentionExtractor, extract_batch

    ex = QwenAttentionExtractor(model_path="/nonexistent/path")
    out = extract_batch(ex, [("short", "short")])
    assert out == [None]


def test_extract_batch_preserves_order():
    """Output order must match input order even with skipped pairs."""
    from src.knowledge._attention import QwenAttentionExtractor, extract_batch

    ex = QwenAttentionExtractor(model_path="/nonexistent/path")
    pairs = [("x" * 50, "y" * 50), ("short", "short"), ("z" * 50, "w" * 50)]
    out = extract_batch(ex, pairs)
    assert len(out) == 3
    assert out[1] is None


def test_cross_alignment_matches_full_projection():
    """cross_alignment_scores must equal alignment_scores(projected_attention)."""
    from src.knowledge._attention import (
        alignment_scores, cross_alignment_scores, projected_attention,
    )

    rng = np.random.default_rng(11)
    for n_tok, qp_end in ((60, 25), (150, 60)):
        n_heads, head_dim, hidden = 3, 8, 16
        attn = rng.random((n_heads, n_tok, n_tok)).astype(np.float32)
        attn /= attn.sum(-1, keepdims=True)
        values = rng.random((n_heads, n_tok, head_dim)).astype(np.float32)
        w_o = rng.random((n_heads, head_dim, hidden)).astype(np.float32)
        tokens = ["graph"] * n_tok
        for i in range(0, n_tok, 3):
            tokens[i] = "the"

        full = alignment_scores(projected_attention(attn, values, w_o), qp_end, tokens)
        cross = cross_alignment_scores(attn, values, w_o, qp_end, tokens)
        assert np.allclose(full, cross, atol=1e-5)


def test_cross_alignment_handles_degenerate_bounds():
    from src.knowledge._attention import cross_alignment_scores

    rng = np.random.default_rng(12)
    attn = rng.random((2, 10, 10)).astype(np.float32)
    values = rng.random((2, 10, 4)).astype(np.float32)
    w_o = rng.random((2, 4, 8)).astype(np.float32)
    tokens = ["graph"] * 10
    assert len(cross_alignment_scores(attn, values, w_o, 0, tokens)) == 0
    assert len(cross_alignment_scores(attn, values, w_o, 10, tokens)) == 0


def test_cross_alignment_returns_normalised_scores():
    from src.knowledge._attention import cross_alignment_scores

    rng = np.random.default_rng(13)
    n_tok, qp_end = 40, 15
    attn = rng.random((2, n_tok, n_tok)).astype(np.float32)
    attn /= attn.sum(-1, keepdims=True)
    values = rng.random((2, n_tok, 4)).astype(np.float32)
    w_o = rng.random((2, 4, 8)).astype(np.float32)
    tokens = ["graph"] * n_tok

    scores = cross_alignment_scores(attn, values, w_o, qp_end, tokens)
    assert len(scores) == n_tok - qp_end
    assert scores.max() == pytest.approx(1.0, abs=1e-5)
    assert np.all(scores >= 0)