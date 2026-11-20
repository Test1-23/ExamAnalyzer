"""Behavioural contract for src/knowledge/_geometry.py (pure, hermetic)."""

import numpy as np
import pytest

from src.knowledge._geometry import (
    Edge,
    LayoutConfig,
    build_edges,
    cluster_positions,
    dbscan_clusters,
    effective_dim,
    layout_metrics,
    merge_similar_clusters,
    mean_pairwise_cos,
    mutual_knn_adjacency,
    normalize_rows,
    partition_ari,
    propagate_qa_membership,
    relax,
)

DIM = 32


# ── helpers ─────────────────────────────────────────────────


def random_anchors(n: int, seed: int = 0, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return normalize_rows(rng.normal(size=(n, dim)))


def blobs(centres: int, per_centre: int, seed: int = 0, noise: float = 0.02,
          dim: int = DIM):
    rng = np.random.default_rng(seed)
    base = normalize_rows(rng.normal(size=(centres, dim)))
    rows, labels = [], []
    for c in range(centres):
        for _ in range(per_centre):
            rows.append(base[c] + rng.normal(0.0, noise, dim))
            labels.append(c)
    return normalize_rows(np.stack(rows)), np.array(labels)


def within_cluster_edges(labels: np.ndarray, weight: float = 1.0) -> list[Edge]:
    return [
        Edge(i, j, weight, "synthetic")
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
        if labels[i] == labels[j]
    ]


def edge_cosine(positions: np.ndarray, edges: list[Edge]) -> float:
    return float(np.mean([
        float(np.dot(positions[e.src], positions[e.dst])) for e in edges
    ]))


# ── normalisation / collapse ────────────────────────────────


def test_normalize_rows_handles_zero_row():
    m = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = normalize_rows(m)
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(out[1], 0.0)
    assert not np.isnan(out).any()


def test_effective_dim_detects_collapse():
    rng = np.random.default_rng(0)
    spread = normalize_rows(rng.normal(size=(50, DIM)))
    collapsed = normalize_rows(np.tile(normalize_rows(rng.normal(size=(1, DIM))), (50, 1))
                               + rng.normal(0, 1e-6, (50, DIM)))
    assert effective_dim(spread) > 8.0
    assert effective_dim(collapsed) < 2.0
    assert mean_pairwise_cos(collapsed) > 0.9


# ── relaxation ──────────────────────────────────────────────


def test_relax_pulls_edges_together():
    anchors = random_anchors(40, seed=1)
    labels = np.array([i // 8 for i in range(40)])
    edges = within_cluster_edges(labels)
    before = edge_cosine(anchors, edges)
    positions, diag = relax(anchors, edges, LayoutConfig(epochs=10, seed=0))
    after = edge_cosine(positions, edges)
    assert diag["collapsed"] is False
    assert after > before + 0.05
    assert diag["epochs_run"] == 10


def test_relax_without_anchor_collapses_and_falls_back():
    anchors = random_anchors(30, seed=2)
    # a connected graph (here: the complete graph) has one attractor, so
    # pure attraction without an anchor must collapse the space
    edges = [Edge(i, j, 1.0, "complete")
             for i in range(30) for j in range(i + 1, 30)]
    cfg = LayoutConfig(anchor_strength=0.0, epochs=30, eta=0.3, seed=0)
    positions, diag = relax(anchors, edges, cfg)
    assert diag["collapsed"] is True
    assert diag["epochs_run"] <= 20
    # fallback returns the anchors untouched
    assert np.allclose(positions, anchors)


def test_relax_is_deterministic():
    anchors = random_anchors(30, seed=3)
    edges = within_cluster_edges(np.array([i // 5 for i in range(30)]))
    cfg = LayoutConfig(epochs=5, seed=7)
    a, _ = relax(anchors, edges, cfg)
    b, _ = relax(anchors, edges, cfg)
    assert np.array_equal(a, b)


def test_relax_is_robust_to_edge_order():
    anchors = random_anchors(30, seed=4)
    labels = np.array([i // 5 for i in range(30)])
    edges = within_cluster_edges(labels)
    rng = np.random.default_rng(0)
    shuffled = [edges[i] for i in rng.permutation(len(edges))]

    cfg = LayoutConfig(epochs=8, seed=5)
    pos_a, _ = relax(anchors, edges, cfg)
    pos_b, _ = relax(anchors, shuffled, cfg)
    assert partition_ari(cluster_positions(pos_a), cluster_positions(pos_b)) >= 0.95


def test_relax_without_edges_returns_anchors():
    anchors = random_anchors(10, seed=5)
    positions, diag = relax(anchors, [], LayoutConfig())
    assert np.allclose(positions, anchors)
    assert diag["edges"] == 0


# ── clustering ──────────────────────────────────────────────


def test_cluster_recovers_synthetic_blobs():
    positions, truth = blobs(centres=5, per_centre=12, seed=6, noise=0.02)
    labels = cluster_positions(positions, LayoutConfig(knn_k=8, radius=0.5))
    assert partition_ari(truth, labels) >= 0.95
    assert len(np.unique(labels)) >= 5


def _two_blobs_plus_outlier():
    positions, truth = blobs(centres=2, per_centre=16, seed=7, noise=0.01)
    rng = np.random.default_rng(11)
    outlier = rng.normal(size=DIM)
    for centre in (positions[truth == 0].mean(axis=0),
                   positions[truth == 1].mean(axis=0)):
        centre = centre / np.linalg.norm(centre)
        outlier = outlier - np.dot(outlier, centre) * centre
    outlier = normalize_rows(outlier.reshape(1, -1))[0]
    return np.vstack([positions, outlier])


def test_dbscan_marks_sparse_points_as_noise():
    positions = _two_blobs_plus_outlier()
    labels, core = dbscan_clusters(positions, eps=0.3, min_samples=4)
    assert labels[-1] == -1
    assert core[-1] == np.False_


def test_cluster_positions_covers_every_object():
    positions = _two_blobs_plus_outlier()
    labels = cluster_positions(positions, LayoutConfig(radius=0.3, min_samples=4))
    # sparse points are absorbed by their nearest cluster, not dropped
    assert np.all(labels >= 0)
    assert np.sum(labels == labels[-1]) >= 1


def test_merge_similar_clusters_joins_duplicates():
    rng = np.random.default_rng(8)
    base = normalize_rows(rng.normal(size=(1, DIM)))
    pos = normalize_rows(np.vstack([base + rng.normal(0, 0.005, (5, DIM)),
                                    base + rng.normal(0, 0.005, (5, DIM))]))
    labels = np.array([0] * 5 + [1] * 5)
    merged = merge_similar_clusters(pos, labels, threshold=0.9)
    assert len(np.unique(merged)) == 1


def test_mutual_knn_is_symmetric():
    positions = random_anchors(20, seed=9)
    adj = mutual_knn_adjacency(positions, k=3, radius=None)
    assert np.array_equal(adj, adj.T)
    assert not np.any(np.diagonal(adj))


# ── ARI / metrics ───────────────────────────────────────────


def test_partition_ari_edges():
    a = np.array([0, 0, 1, 1])
    assert partition_ari(a, a) == pytest.approx(1.0)
    # anti-correlated partitions score below zero in the adjusted index
    assert partition_ari(a, np.array([0, 1, 0, 1])) == pytest.approx(-0.5)
    assert partition_ari(np.array([0]), np.array([0])) == pytest.approx(1.0)


def test_layout_metrics_shape():
    positions, truth = blobs(centres=3, per_centre=6, seed=10, noise=0.01)
    labels = cluster_positions(positions, LayoutConfig(radius=0.5))
    m = layout_metrics(positions, labels, anchors=positions)
    for key in ("n_objects", "n_clusters", "singleton_ratio", "cohesion",
                "separation", "effective_dim", "mean_pairwise_cos",
                "anchor_drift_mean"):
        assert key in m
    assert m["n_objects"] == len(positions)
    assert m["cohesion"] > m["separation"]


# ── edge building ───────────────────────────────────────────


def test_build_edges_creates_expected_kinds():
    index = {("qa", "1"): 0, ("fragment", "f1"): 1, ("fragment", "f2"): 2,
             ("qa", "2"): 3}
    edges = build_edges(index, {1: ["f1", "f2"]}, [("f1", 2, 0.8)])
    kinds = sorted(e.kind for e in edges)
    assert kinds.count("qa_fragment") == 2
    assert kinds.count("fragment_sibling") == 1
    assert kinds.count("fragment_help") == 1
    help_edge = next(e for e in edges if e.kind == "fragment_help")
    assert help_edge.weight == pytest.approx(0.8)


def test_build_edges_dedups_and_skips_unknown_ids():
    index = {("qa", "1"): 0, ("fragment", "f1"): 1}
    edges = build_edges(index, {1: ["f1", "missing"], 99: ["f1"]}, [("nope", 1, 1.0)])
    assert len(edges) == 1
    assert edges[0].kind == "qa_fragment"


def test_build_edges_never_links_same_paper_questions():
    """Design invariant: no same-paper QA-QA edges are created."""
    index = {("qa", str(i)): i for i in range(4)}
    edges = build_edges(index, {})
    assert edges == []


def test_propagate_qa_membership_pulls_same_question_together():
    rng = np.random.default_rng(0)
    base = normalize_rows(rng.normal(size=(1, DIM)))[0]
    # three fragments of one question, scattered around a shared centre
    pos = normalize_rows(np.vstack([base + rng.normal(0, 0.05, (3, DIM)),
                                    normalize_rows(rng.normal(size=(1, DIM)))]))
    qa_of_row = np.array([1, 1, 1, 2])
    before = float(pos[0] @ pos[1])
    out = propagate_qa_membership(pos, qa_of_row, weight=0.5, rounds=1)
    after = float(out[0] @ out[1])
    assert after > before
    # the unrelated fragment must not be dragged in
    assert float(out[0] @ out[3]) < 0.9


def test_propagate_qa_membership_weight_zero_is_identity():
    pos = random_anchors(10, seed=3)
    qa_of_row = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    out = propagate_qa_membership(pos, qa_of_row, weight=0.0)
    assert np.allclose(out, pos)


def test_propagate_qa_membership_keeps_unit_norm():
    pos = random_anchors(12, seed=4)
    qa_of_row = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    out = propagate_qa_membership(pos, qa_of_row, weight=0.6, rounds=2)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_propagate_qa_membership_rejects_length_mismatch():
    pos = random_anchors(4, seed=5)
    with pytest.raises(ValueError):
        propagate_qa_membership(pos, np.array([1, 1]), weight=0.5)
