"""Pure function tests for knowledge_graph — zero DB, zero network, zero embedding.

All functions are tested with synthetic numpy arrays and plain dicts.
"""

import numpy as np
import pytest

from src.constants import KG_SEMANTIC_EDGE_THRESHOLD, KG_SEMANTIC_EDGE_MEDIUM_THRESHOLD
from src.knowledge_graph import (
    _build_qa_to_kp_mapping,
    _build_similarity_graph,
    _compute_centroid,
    _compute_cohesion,
    _compute_retrieval_candidates,
    _compute_semantic_edges,
    _compute_sequential_transitions,
    _cosine_to_centroid,
    _find_clusters,
    _parse_kp_cluster_idx,
    _qn_sort_key,
    _transition_weight,
)


# ═══════════════════════════════════════════════════════════════
# Helper — make normalized fake embedding vectors
# ═══════════════════════════════════════════════════════════════

def _make_vecs(*rows):
    """Create a normalized (n, 4) ndarray from list-of-lists."""
    arr = np.array(rows, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr / norms


# ═══════════════════════════════════════════════════════════════
# _qn_sort_key
# ═══════════════════════════════════════════════════════════════

class TestQnSortKey:
    def test_simple_number(self):
        assert _qn_sort_key("1") == (1, "")
        assert _qn_sort_key("10") == (10, "")

    def test_number_with_suffix(self):
        assert _qn_sort_key("1(a)") == (1, "(a)")
        assert _qn_sort_key("10(b)") == (10, "(b)")

    def test_pure_suffix(self):
        key = _qn_sort_key("(a)")
        assert key == (0, "(a)")

    def test_sort_order(self):
        items = ["10(a)", "2(a)", "1(b)", "1(a)"]
        assert sorted(items, key=_qn_sort_key) == ["1(a)", "1(b)", "2(a)", "10(a)"]


# ═══════════════════════════════════════════════════════════════
# _parse_kp_cluster_idx
# ═══════════════════════════════════════════════════════════════

class TestParseKpClusterIdx:
    def test_normal_kp_id(self):
        assert _parse_kp_cluster_idx("kp_0000") == 0
        assert _parse_kp_cluster_idx("kp_0042") == 42

    def test_split_suffix(self):
        assert _parse_kp_cluster_idx("kp_0000s0") == 0
        assert _parse_kp_cluster_idx("kp_0003s1") == 3

    def test_non_kp_format(self):
        assert _parse_kp_cluster_idx("not_a_kp") is None
        assert _parse_kp_cluster_idx("") is None
        assert _parse_kp_cluster_idx("topic_0001") is None


# ═══════════════════════════════════════════════════════════════
# _transition_weight
# ═══════════════════════════════════════════════════════════════

class TestTransitionWeight:
    def test_zero_count(self):
        assert _transition_weight(0) == 0.0

    def test_half_divisor(self):
        w = _transition_weight(5)  # EDGE_TRANSITION_DIVISOR = 10
        assert w == 0.5

    def test_at_divisor(self):
        w = _transition_weight(10)
        assert w == 1.0

    def test_above_divisor_capped(self):
        w = _transition_weight(100)
        assert w == 1.0


# ═══════════════════════════════════════════════════════════════
# _build_similarity_graph
# ═══════════════════════════════════════════════════════════════

class TestBuildSimilarityGraph:
    def test_empty(self):
        adj, cos_mat = _build_similarity_graph(np.empty((0, 4)))
        assert adj == {}
        assert len(cos_mat) == 0

    def test_identical_vectors_connected(self):
        v = _make_vecs([1, 0, 0, 0], [1, 0, 0, 0])
        adj, _ = _build_similarity_graph(v, threshold=0.7)
        assert 1 in adj[0]

    def test_orthogonal_vectors_disconnected(self):
        v = _make_vecs([1, 0, 0, 0], [0, 1, 0, 0])
        adj, _ = _build_similarity_graph(v, threshold=0.7)
        assert 1 not in adj[0]

    def test_threshold_boundary(self):
        # vectors with cos ≈ 0.5 — should connect only when threshold <= 0.5
        v = _make_vecs([1, 0, 0, 0], [1, 1, 0, 0])  # ~0.707
        adj_low, _ = _build_similarity_graph(v, threshold=0.5)
        adj_high, _ = _build_similarity_graph(v, threshold=0.8)
        assert 1 in adj_low[0]
        assert 1 not in adj_high[0]


# ═══════════════════════════════════════════════════════════════
# _find_clusters
# ═══════════════════════════════════════════════════════════════

class TestFindClusters:
    def test_empty_graph(self):
        clusters, noise = _find_clusters({}, 0)
        assert clusters == []
        assert noise == []

    def test_single_cluster(self):
        adj = {0: {1}, 1: {0}}
        clusters, noise = _find_clusters(adj, 2)
        assert len(clusters) == 1
        assert set(clusters[0]) == {0, 1}
        assert noise == []

    def test_two_clusters(self):
        adj = {0: {1}, 1: {0}, 2: {3}, 3: {2}}
        clusters, noise = _find_clusters(adj, 4)
        assert len(clusters) == 2

    def test_all_noise(self):
        adj = {0: set(), 1: set(), 2: set()}
        clusters, noise = _find_clusters(adj, 3)
        assert clusters == []
        assert sorted(noise) == [0, 1, 2]


# ═══════════════════════════════════════════════════════════════
# _compute_centroid
# ═══════════════════════════════════════════════════════════════

class TestComputeCentroid:
    def test_two_vectors(self):
        v = _make_vecs([1, 0, 0, 0], [1, 0, 0, 0])
        c = _compute_centroid(v, [0, 1])
        assert c is not None
        assert c.shape == (4,)
        assert abs(float(c[0]) - 1.0) < 0.001

    def test_empty_indices(self):
        v = _make_vecs([1, 0, 0, 0])
        c = _compute_centroid(v, [])
        assert c is None


# ═══════════════════════════════════════════════════════════════
# _compute_cohesion
# ═══════════════════════════════════════════════════════════════

class TestComputeCohesion:
    def test_identical_vectors(self):
        v = _make_vecs([1, 0, 0, 0], [1, 0, 0, 0])
        c = _compute_centroid(v, [0, 1])
        cohesion = _compute_cohesion(v, [0, 1], c)
        assert abs(cohesion - 1.0) < 0.001

    def test_empty_indices(self):
        v = _make_vecs([1, 0, 0, 0])
        c = _compute_centroid(v, [0])
        cohesion = _compute_cohesion(v, [], c)
        assert cohesion == 0.0


# ═══════════════════════════════════════════════════════════════
# _cosine_to_centroid
# ═══════════════════════════════════════════════════════════════

class TestCosineToCentroid:
    def test_same_direction(self):
        v = _make_vecs([1, 0, 0, 0])
        c = np.array([1, 0, 0, 0], dtype=np.float32)
        sim = _cosine_to_centroid(v, 0, c)
        assert abs(sim - 1.0) < 0.001

    def test_none_centroid(self):
        v = _make_vecs([1, 0, 0, 0])
        assert _cosine_to_centroid(v, 0, None) == 0.0


# ═══════════════════════════════════════════════════════════════
# _build_qa_to_kp_mapping
# ═══════════════════════════════════════════════════════════════

class TestBuildQaToKpMapping:
    def test_single_cluster(self):
        qa_list = [{"id": 10}, {"id": 20}]
        clusters = [[0, 1]]
        kp_ids = ["kp_0000"]
        mapping = _build_qa_to_kp_mapping(clusters, qa_list, kp_ids)
        assert mapping == {10: "kp_0000", 20: "kp_0000"}

    def test_multi_cluster(self):
        qa_list = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
        clusters = [[0, 1], [2, 3]]
        kp_ids = ["kp_0000", "kp_0001"]
        mapping = _build_qa_to_kp_mapping(clusters, qa_list, kp_ids)
        assert mapping[1] == "kp_0000"
        assert mapping[3] == "kp_0001"

    def test_empty(self):
        mapping = _build_qa_to_kp_mapping([], [], [])
        assert mapping == {}


# ═══════════════════════════════════════════════════════════════
# _compute_sequential_transitions
# ═══════════════════════════════════════════════════════════════

class TestComputeSequentialTransitions:
    def test_single_paper_two_kps(self):
        qa_list = [
            {"id": 1, "paper": "p1", "question_number": "1"},
            {"id": 2, "paper": "p1", "question_number": "2"},
        ]
        clusters = [[0], [1]]
        kp_ids = ["kp_0000", "kp_0001"]
        transitions, paper_pairs = _compute_sequential_transitions(
            qa_list, clusters, kp_ids)
        assert ("kp_0000", "kp_0001") in transitions
        assert transitions[("kp_0000", "kp_0001")] == 1

    def test_multi_paper_same_pair(self):
        qa_list = [
            {"id": 1, "paper": "p1", "question_number": "1"},
            {"id": 2, "paper": "p1", "question_number": "2"},
            {"id": 3, "paper": "p2", "question_number": "1"},
            {"id": 4, "paper": "p2", "question_number": "2"},
        ]
        clusters = [[0], [1], [2], [3]]
        kp_ids = ["kp_0000", "kp_0001", "kp_0002", "kp_0003"]
        transitions, paper_pairs = _compute_sequential_transitions(
            qa_list, clusters, kp_ids)
        assert transitions[("kp_0000", "kp_0001")] == 1
        assert transitions[("kp_0002", "kp_0003")] == 1
        assert paper_pairs[("kp_0000", "kp_0001")] == {"p1"}

    def test_duplicate_consecutive_kp_skipped(self):
        """Consecutive QAs in same KP should not produce transitions."""
        qa_list = [
            {"id": 1, "paper": "p1", "question_number": "1"},
            {"id": 2, "paper": "p1", "question_number": "2"},
        ]
        clusters = [[0, 1]]  # same cluster
        kp_ids = ["kp_0000"]
        transitions, _ = _compute_sequential_transitions(
            qa_list, clusters, kp_ids)
        assert transitions == {}


# ═══════════════════════════════════════════════════════════════
# _compute_retrieval_candidates
# ═══════════════════════════════════════════════════════════════

class TestComputeRetrievalCandidates:
    def test_empty_links(self):
        qa_list = [{"id": 1}, {"id": 2}]
        qa_to_kp = {1: "kp_0000", 2: "kp_0001"}
        candidates = _compute_retrieval_candidates({}, qa_list, qa_to_kp)
        assert candidates == []

    def test_single_link_produces_candidate(self):
        """Topic link count > 0 produces a reversed-direction candidate."""
        topic_links = {("A", "B"): 5}
        qa_list = [
            {"id": 1, "topic": "A"}, {"id": 2, "topic": "B"},
        ]
        qa_to_kp = {1: "kp_0000", 2: "kp_0001"}
        candidates = _compute_retrieval_candidates(topic_links, qa_list, qa_to_kp)
        assert len(candidates) >= 1
        # Reversed direction: retrieval from dst→src
        # Reversed direction: (dst_kp, src_kp, count)
        assert any(c[0] == "kp_0001" and c[1] == "kp_0000" for c in candidates)


# ═══════════════════════════════════════════════════════════════
# _compute_semantic_edges
# ═══════════════════════════════════════════════════════════════

class TestComputeSemanticEdges:
    def test_less_than_two_kps(self):
        centroids = {"kp_0000": np.array([1.0, 0.0], dtype=np.float32)}
        edges = _compute_semantic_edges(centroids)
        assert edges == []

    def test_high_similarity_produces_edge(self):
        centroids = {
            "kp_0000": np.array([1.0, 0.0], dtype=np.float32),
            "kp_0001": np.array([1.0, 0.0], dtype=np.float32),
        }
        edges = _compute_semantic_edges(centroids)
        assert len(edges) >= 1
        kp_a, kp_b, score, confidence = edges[0]
        assert abs(score - 1.0) < 0.01
        assert confidence == "medium"

    def test_low_similarity_no_edge(self):
        centroids = {
            "kp_0000": np.array([1.0, 0.0], dtype=np.float32),
            "kp_0001": np.array([0.0, 1.0], dtype=np.float32),
        }
        edges = _compute_semantic_edges(centroids)
        assert edges == []

    def test_medium_confidence_boundary(self):
        """cos ≈ 0.6 → confidence should be 'low' (below 0.65)."""
        centroids = {
            "kp_0000": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "kp_0001": np.array([0.6, 0.8, 0.0], dtype=np.float32),
        }
        edges = _compute_semantic_edges(centroids)
        assert len(edges) >= 1
        assert edges[0][3] == "low"
