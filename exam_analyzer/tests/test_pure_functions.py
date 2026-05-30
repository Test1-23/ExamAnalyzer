"""Unit tests for pure functions — zero dependencies, zero mocks needed.

Covers:
- _evaluate_signal: 12 boundary combinations for difficulty classification
- make_topic_id: topic name sanitization
- _qn_sort_key: natural question-number sort key
- _compute_sequential_transitions: KP transition counts from exam ordering
- _compute_semantic_edges: cosine similarity edge candidates
- _compute_retrieval_candidates: topic-link KP mapping
"""

import numpy as np
import pytest

from src.offline.difficulty import _evaluate_signal
from src.retriever import make_topic_id
from src.knowledge_graph import (
    _qn_sort_key,
    _compute_sequential_transitions,
    _compute_semantic_edges,
    _compute_retrieval_candidates,
    _transition_weight,
)


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


# ============================================================
# P1-4 extracted pure functions from knowledge_graph.py
# ============================================================


class TestQnSortKey:
    """Natural sort key for question numbers like "1(a)", "10(b)"."""

    def test_simple_number(self):
        assert _qn_sort_key("1") == (1, "")

    def test_number_with_suffix(self):
        assert _qn_sort_key("1(a)") == (1, "(a)")

    def test_multi_digit(self):
        assert _qn_sort_key("10(b)") == (10, "(b)")

    def test_natural_ordering(self):
        """Verify natural sort: 2 < 10, not lexicographic "10" < "2"."""
        items = ["10(a)", "2(a)", "1(a)"]
        assert sorted(items, key=_qn_sort_key) == ["1(a)", "2(a)", "10(a)"]

    def test_empty_string(self):
        assert _qn_sort_key("") == (0, "")

    def test_no_number(self):
        assert _qn_sort_key("abc") == (0, "abc")


class TestComputeSequentialTransitions:
    """Pure function: QA list + clusters → KP transitions across papers."""

    @staticmethod
    def _make_kp_id(cluster_idx):
        return f"kp_{cluster_idx:04d}"

    def test_single_paper_two_transitions(self):
        """Paper P1: QA order Q1→Q2→Q3 (KPs: A, B, A). Transitions: A→B, B→A."""
        qa_list = [
            {"id": 1, "paper": "P1", "question_number": "1"},
            {"id": 2, "paper": "P1", "question_number": "2"},
            {"id": 3, "paper": "P1", "question_number": "3"},
        ]
        clusters = [[0, 2], [1]]  # QA 1&3→cluster 0, QA 2→cluster 1
        kp_ids = [self._make_kp_id(0), self._make_kp_id(1)]

        transitions, paper_kp_pairs = _compute_sequential_transitions(
            qa_list, clusters, kp_ids)

        assert transitions == {(kp_ids[0], kp_ids[1]): 1, (kp_ids[1], kp_ids[0]): 1}
        assert paper_kp_pairs[(kp_ids[0], kp_ids[1])] == {"P1"}
        assert paper_kp_pairs[(kp_ids[1], kp_ids[0])] == {"P1"}

    def test_two_papers_same_transition(self):
        """Two papers each with A→B transition → count=2, papers={P1,P2}."""
        qa_list = [
            {"id": 1, "paper": "P1", "question_number": "1"},
            {"id": 2, "paper": "P1", "question_number": "2"},
            {"id": 3, "paper": "P2", "question_number": "1"},
            {"id": 4, "paper": "P2", "question_number": "2"},
        ]
        clusters = [[0, 2], [1, 3]]  # QA 1&3→cluster 0, QA 2&4→cluster 1
        kp_ids = [self._make_kp_id(0), self._make_kp_id(1)]

        transitions, paper_kp_pairs = _compute_sequential_transitions(
            qa_list, clusters, kp_ids)

        assert transitions[(kp_ids[0], kp_ids[1])] == 2
        assert paper_kp_pairs[(kp_ids[0], kp_ids[1])] == {"P1", "P2"}

    def test_consecutive_same_kp_collapsed(self):
        """Consecutive QAs in same KP are collapsed: Q1(A)→Q2(A)→Q3(B) → A→B only."""
        qa_list = [
            {"id": 1, "paper": "P1", "question_number": "1"},
            {"id": 2, "paper": "P1", "question_number": "2"},
            {"id": 3, "paper": "P1", "question_number": "3"},
        ]
        clusters = [[0, 1], [2]]  # QA 1&2→cluster 0, QA 3→cluster 1
        kp_ids = [self._make_kp_id(0), self._make_kp_id(1)]

        transitions, _ = _compute_sequential_transitions(
            qa_list, clusters, kp_ids)

        # Only one transition: A→B (A→A is collapsed)
        assert len(transitions) == 1
        assert transitions[(kp_ids[0], kp_ids[1])] == 1

    def test_natural_sort_order(self):
        """QAs are sorted naturally: Q10 after Q2, not before."""
        qa_list = [
            {"id": 1, "paper": "P1", "question_number": "10"},
            {"id": 2, "paper": "P1", "question_number": "2"},
        ]
        clusters = [[0], [1]]
        kp_ids = [self._make_kp_id(0), self._make_kp_id(1)]

        transitions, _ = _compute_sequential_transitions(
            qa_list, clusters, kp_ids)

        # Q2 (KP B) → Q10 (KP A), not Q10→Q2
        assert transitions[(kp_ids[1], kp_ids[0])] == 1

    def test_no_paper_info(self):
        """QAs without paper info produce no transitions."""
        qa_list = [
            {"id": 1, "paper": "", "question_number": "1"},
            {"id": 2, "paper": "P1", "question_number": ""},
        ]
        clusters = [[0], [1]]
        kp_ids = [self._make_kp_id(0), self._make_kp_id(1)]

        transitions, _ = _compute_sequential_transitions(
            qa_list, clusters, kp_ids)

        assert transitions == {}


class TestComputeSemanticEdges:
    """Pure function: KP centroids → cosine similarity edge candidates."""

    def test_two_centroids_high_similarity(self):
        """Two nearly identical centroids → one edge with high confidence."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.99, 0.01, 0.0], dtype=np.float32)
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        centroids = {"kp_0000": v1, "kp_0001": v2}

        edges = _compute_semantic_edges(centroids)

        assert len(edges) == 1
        a, b, cos, conf = edges[0]
        assert {a, b} == {"kp_0000", "kp_0001"}
        assert cos >= 0.5
        # With cos ~0.99 → confidence should be "medium" (>= 0.65)
        assert conf == "medium"

    def test_orthogonal_centroids_no_edge(self):
        """Orthogonal unit vectors → cos ≈ 0 → no edge (below 0.5 threshold)."""
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        centroids = {"kp_0000": v1, "kp_0001": v2}

        edges = _compute_semantic_edges(centroids)

        assert edges == []

    def test_empty_centroids(self):
        assert _compute_semantic_edges({}) == []

    def test_single_centroid(self):
        centroids = {"kp_0000": np.array([1.0, 0.0], dtype=np.float32)}
        assert _compute_semantic_edges(centroids) == []

    def test_confidence_boundary(self):
        """cos=0.64 → "low", cos=0.65 → "medium" (with explicit normalization)."""
        v1 = np.array([1.0, 0.0], dtype=np.float32)

        # cos=0.64 → low
        cos_val = 0.64
        v2 = np.array([cos_val, np.sqrt(1 - cos_val ** 2)], dtype=np.float32)
        v2 = v2 / np.linalg.norm(v2)
        edges = _compute_semantic_edges({"kp_0000": v1, "kp_0001": v2})
        assert len(edges) == 1
        assert edges[0][3] == "low"

        # cos=0.65 → medium
        cos_val = 0.65
        v2 = np.array([cos_val, np.sqrt(1 - cos_val ** 2)], dtype=np.float32)
        v2 = v2 / np.linalg.norm(v2)
        edges = _compute_semantic_edges({"kp_0000": v1, "kp_0001": v2})
        assert len(edges) == 1
        assert edges[0][3] == "medium"


class TestComputeRetrievalCandidates:
    """Pure function: topic_links + QAs → retrieval edge candidates."""

    def test_single_topic_pair(self):
        """One topic pair with matching QAs → one candidate."""
        topic_links = {("Algebra", "Calculus"): 3}
        qa_list = [
            {"id": 1, "topic": "Algebra"},
            {"id": 2, "topic": "Calculus"},
        ]
        qa_to_kp = {1: "kp_0000", 2: "kp_0001"}

        candidates = _compute_retrieval_candidates(topic_links, qa_list, qa_to_kp)

        assert len(candidates) == 1
        dk, sk, count = candidates[0]
        assert dk == "kp_0001"  # reversed: dst_topic→target
        assert sk == "kp_0000"
        assert count == 3

    def test_count_below_threshold_filtered(self):
        """Topic pairs with count < 2 are skipped."""
        topic_links = {("Algebra", "Calculus"): 1}
        qa_list = [
            {"id": 1, "topic": "Algebra"},
            {"id": 2, "topic": "Calculus"},
        ]
        qa_to_kp = {1: "kp_0000", 2: "kp_0001"}

        candidates = _compute_retrieval_candidates(topic_links, qa_list, qa_to_kp)

        assert candidates == []

    def test_same_kp_filtered(self):
        """Source and destination KPs are the same → filtered out."""
        topic_links = {("Algebra", "Geometry"): 2}
        qa_list = [
            {"id": 1, "topic": "Algebra"},
            {"id": 2, "topic": "Geometry"},
        ]
        qa_to_kp = {1: "kp_0000", 2: "kp_0000"}  # same KP

        candidates = _compute_retrieval_candidates(topic_links, qa_list, qa_to_kp)

        assert candidates == []

    def test_empty_topic_links(self):
        assert _compute_retrieval_candidates({}, [], {}) == []

    def test_multiple_qas_per_topic(self):
        """Topics with multiple QAs → Cartesian product of KPs (cross-topic only)."""
        topic_links = {("A", "B"): 4}
        qa_list = [
            {"id": 1, "topic": "A"},
            {"id": 2, "topic": "A"},
            {"id": 3, "topic": "B"},
        ]
        qa_to_kp = {1: "kp_0000", 2: "kp_0001", 3: "kp_0002"}

        candidates = _compute_retrieval_candidates(topic_links, qa_list, qa_to_kp)

        # src_kps = {kp_0000, kp_0001}, dst_kps = {kp_0002}
        # cross product: 2 × 1 = 2, both filtered by sk!=dk
        assert len(candidates) == 2
        dk_values = {c[0] for c in candidates}
        assert dk_values == {"kp_0002"}  # reversed: dst_topic → target_kp


class TestTransitionWeight:
    """Edge transition count → normalized weight."""

    def test_zero(self):
        assert _transition_weight(0) == 0.0

    def test_half_divisor(self):
        """count = EDGE_TRANSITION_DIVISOR/2 → weight = 0.5."""
        from src.constants import EDGE_TRANSITION_DIVISOR
        assert _transition_weight(EDGE_TRANSITION_DIVISOR // 2) == 0.5

    def test_at_divisor(self):
        from src.constants import EDGE_TRANSITION_DIVISOR
        assert _transition_weight(EDGE_TRANSITION_DIVISOR) == 1.0

    def test_capped_at_one(self):
        from src.constants import EDGE_TRANSITION_DIVISOR
        assert _transition_weight(EDGE_TRANSITION_DIVISOR * 2) == 1.0
