"""Vector-space geometry: anchored relaxation and geometric clustering.

This module replaces threshold-based QA assignment with a continuous layout:

1. Every object (QA / MS fragment / KP centroid) has an **anchor** - its
   embedding-derived starting position on the unit sphere.
2. Knowledge-level co-occurrence edges pull co-occurring objects slightly
   closer over several sweeps (epochs).
3. A pull-back term keeps every object near its anchor, so the space cannot
   collapse into a single point.
4. Clusters are read off the relaxed geometry (mutual kNN + radius), not
   decided by a similarity threshold at insert time.

All functions are pure numpy: no DB access, no I/O, no model loading.
See tests/test_geometry.py for the behavioural contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class LayoutConfig:
    """Tunable parameters for the anchored relaxation."""

    # Pull-back strength towards the embedding anchor (0 == pure attraction).
    anchor_strength: float = 0.20

    # Initial step size and per-epoch decay: eta_t = eta / (1 + decay * t)
    eta: float = 0.20
    eta_decay: float = 0.10
    epochs: int = 20

    # Deterministic edge ordering.
    seed: int = 0

    # Collapse guard: abort and fall back to anchors when the space degenerates.
    # Absolute effective-dimension floors mis-fire on small graphs, so the
    # dimension check is relative to the anchor geometry's own dimension.
    collapse_mean_cos: float = 0.85
    collapse_dim_ratio: float = 0.20

    # QA co-membership propagation (Mygo-006).
    # Fragments of the same question are pulled towards their shared centroid
    # before clustering.  Measured effect: cross-paper same-topic AUC
    # 0.698 -> 0.777 at weight 0.4 (docs/research/association-structure-embedding.md).
    qa_propagation_weight: float = 0.40
    qa_propagation_rounds: int = 1

    # Clustering geometry.
    # radius=None means "derive eps from the data" (median k-th neighbour
    # similarity), then raise it until no single cluster exceeds
    # max_cluster_share.  Set a float to pin eps explicitly.
    knn_k: int = 8
    radius: float | None = None
    min_samples: int = 4
    max_cluster_share: float = 0.35
    radius_step: float = 0.03
    radius_max: float = 0.95
    merge_cos: float = 0.92
    min_cluster_size: int = 2


@dataclass
class Edge:
    """A weighted pull between two rows of the position matrix."""

    src: int
    dst: int
    weight: float
    kind: str = ""

    def key(self) -> tuple[int, int, str]:
        return (min(self.src, self.dst), max(self.src, self.dst), self.kind)


# ============================================================
# Basic array helpers
# ============================================================


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise every row; zero rows are left as zeros (never NaN)."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, EPS)


def mean_pairwise_cos(positions: np.ndarray, sample_size: int = 4000,
                      seed: int = 0) -> float:
    """Mean cosine over a random sample of distinct pairs (collapse detector)."""
    n = len(positions)
    if n < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, sample_size * 2)
    j = rng.integers(0, n, sample_size * 2)
    mask = i != j
    if not mask.any():
        return 1.0
    sims = np.sum(positions[i[mask]] * positions[j[mask]], axis=1)
    return float(np.mean(sims))


def effective_dim(positions: np.ndarray) -> float:
    """Participation ratio of the covariance spectrum (effective dimension).

    Collapsed spaces (all vectors pointing the same way) report ~1.0.
    """
    n = len(positions)
    if n < 2:
        return 1.0
    centred = positions - positions.mean(axis=0)
    cov = np.cov(centred.T)
    eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    denom = float(np.sum(eig ** 2))
    if denom <= EPS:
        return 1.0
    return float(np.sum(eig) ** 2 / denom)


# ============================================================
# Edge construction
# ============================================================


def build_edges(
    index: dict[tuple[str, str], int],
    qa_fragments: dict[int, list[str]],
    fragment_helps: list[tuple[str, int, float]] | None = None,
    same_qa_fragment_weight: float = 0.1,
    qa_fragment_weight: float = 1.0,
    help_weight_scale: float = 1.0,
) -> list[Edge]:
    """Build knowledge-level co-occurrence edges.

    Deliberately does NOT create same-paper QA-QA edges: the design-doc
    experiment shows those organise the space by paper instead of by
    knowledge point.

    index maps ("qa", str(qa_id)) / ("fragment", point_id) to a row.
    """
    edges: dict[tuple[int, int, str], float] = {}

    def add(src: int, dst: int, weight: float, kind: str) -> None:
        if src == dst or weight <= 0:
            return
        key = (min(src, dst), max(src, dst), kind)
        if weight > edges.get(key, 0.0):
            edges[key] = float(weight)

    for qa_id, frag_ids in qa_fragments.items():
        qa_key = ("qa", str(qa_id))
        if qa_key not in index:
            continue
        qa_row = index[qa_key]
        rows = []
        for fid in frag_ids:
            frow = index.get(("fragment", fid))
            if frow is None:
                continue
            rows.append(frow)
            add(qa_row, frow, qa_fragment_weight, "qa_fragment")
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                add(rows[a], rows[b], same_qa_fragment_weight, "fragment_sibling")

    for frag_id, helped_qa_id, effect in fragment_helps or []:
        frow = index.get(("fragment", frag_id))
        qrow = index.get(("qa", str(helped_qa_id)))
        if frow is None or qrow is None:
            continue
        add(frow, qrow, max(float(effect), 0.0) * help_weight_scale, "fragment_help")

    return [Edge(s, d, w, k) for (s, d, k), w in sorted(edges.items())]


def build_cross_paper_edges(
    embeddings: np.ndarray,
    paper_of: list[str],
    topk: int = 3,
    cos_min: float = 0.90,
) -> list[Edge]:
    """Connect near-duplicate fragments that recur in DIFFERENT papers.

    This is the knowledge-level co-occurrence signal: the same scoring point
    ("state one benefit of compression") reappearing in another paper. Edges
    inside one paper are deliberately skipped - those encode exam context, not
    concepts (see the design-plan experiment table).
    """
    n = len(embeddings)
    if n < 2:
        return []
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -1.0)
    k_eff = min(30, n - 1)
    candidates = np.argpartition(-sim, k_eff, axis=1)[:, :k_eff]

    edges: dict[tuple[int, int], float] = {}
    for i in range(n):
        added = 0
        for j in candidates[i]:
            j = int(j)
            if sim[i, j] < cos_min or paper_of[i] == paper_of[j]:
                continue
            key = (min(i, j), max(i, j))
            edges[key] = max(edges.get(key, 0.0), float(sim[i, j]))
            added += 1
            if added >= topk:
                break
    return [Edge(a, b, w, "cross_paper") for (a, b), w in sorted(edges.items())]


# ============================================================
# Relaxation
# ============================================================


def relax(
    anchors: np.ndarray,
    edges: list[Edge],
    config: LayoutConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """Run anchored, tangential relaxation over the edge list.

    Update rule (per edge, per epoch)::

        d    = x_j - x_i            (projected onto the tangent plane of x_i)
        x_i <- normalize(x_i + eta*w*d + lambda*(a_i - x_i))
        x_j <- normalize(x_j - eta*w*d + lambda*(a_j - x_j))

    Returns (positions, diagnostics). diagnostics["collapsed"] is True when the
    space degenerated (near-total similarity or a collapse of the effective
    dimensionality relative to the anchors), in which case the anchors are
    returned unchanged.
    """
    cfg = config or LayoutConfig()
    anchors = normalize_rows(anchors)
    positions = anchors.copy()

    anchor_dim = effective_dim(anchors)
    diag: dict = {
        "collapsed": False,
        "epochs_run": 0,
        "effdim": [],
        "mean_cos": [],
        "anchor_dim": anchor_dim,
        "edges": len(edges),
    }
    rng = np.random.default_rng(cfg.seed)

    if positions.shape[0] < 2 or not edges:
        diag["mean_cos"] = [mean_pairwise_cos(positions, seed=cfg.seed)]
        diag["effdim"] = [effective_dim(positions)]
        return positions, diag

    valid = [(e.src, e.dst, e.weight) for e in edges
             if 0 <= e.src < len(positions) and 0 <= e.dst < len(positions)]
    if not valid:
        return positions, diag

    for epoch in range(cfg.epochs):
        eta = cfg.eta / (1.0 + cfg.eta_decay * epoch)
        for k in rng.permutation(len(valid)):
            i, j, w = valid[k]
            d = positions[j] - positions[i]
            d -= np.dot(positions[i], d) * positions[i]
            positions[i] = (positions[i] + eta * w * d
                            + cfg.anchor_strength * (anchors[i] - positions[i]))
            d = positions[i] - positions[j]
            d -= np.dot(positions[j], d) * positions[j]
            positions[j] = (positions[j] + eta * w * d
                            + cfg.anchor_strength * (anchors[j] - positions[j]))
            positions[i] /= max(np.linalg.norm(positions[i]), EPS)
            positions[j] /= max(np.linalg.norm(positions[j]), EPS)

        cos = mean_pairwise_cos(positions, seed=cfg.seed)
        dim = effective_dim(positions)
        diag["epochs_run"] = epoch + 1
        diag["mean_cos"].append(cos)
        diag["effdim"].append(dim)

        dim_ratio = dim / anchor_dim if anchor_dim > 1.5 else 1.0
        if cos > cfg.collapse_mean_cos or dim_ratio < cfg.collapse_dim_ratio:
            diag["collapsed"] = True
            diag["epochs_run"] = epoch + 1
            return anchors.copy(), diag

    return positions, diag


# ============================================================
# Clustering
# ============================================================


def similarity_matrix(positions: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix with a neutral (1.0) diagonal."""
    sim = positions @ positions.T
    np.fill_diagonal(sim, 1.0)
    return sim


def adaptive_radius(positions: np.ndarray, k: int = 4, sample: int = 500,
                    seed: int = 0) -> float:
    """Data-driven eps: median cosine distance to the k-th nearest neighbour.

    Uses the same model-dependent similarity scale the embeddings produce, so
    the default needs no dataset-specific tuning.
    """
    n = len(positions)
    if n < 2:
        return 0.5
    rng = np.random.default_rng(seed)
    rows = rng.choice(n, min(n, sample), replace=False)
    sims = positions[rows] @ positions.T
    sims[np.arange(len(rows)), rows] = -np.inf
    k_eff = min(k, n - 1)
    kth = np.partition(-sims, k_eff - 1, axis=1)[:, k_eff - 1]
    return float(np.median(-kth))


def dbscan_clusters(positions: np.ndarray, eps: float, min_samples: int = 4,
                    sim: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Cosine DBSCAN. Returns (labels (-1 = noise), core_mask)."""
    n = len(positions)
    sim = similarity_matrix(positions) if sim is None else sim
    neighbourhood = sim >= eps
    core = neighbourhood.sum(axis=1) >= min_samples

    labels = np.full(n, -1, dtype=np.int32)
    label = 0
    for start in range(n):
        if not core[start] or labels[start] != -1:
            continue
        stack = [start]
        labels[start] = label
        while stack:
            node = stack.pop()
            if not core[node]:
                continue
            for nxt in np.flatnonzero(neighbourhood[node]):
                if labels[nxt] == -1:
                    labels[nxt] = label
                    stack.append(int(nxt))
        label += 1
    return labels, core


def assign_unclustered(positions: np.ndarray, labels: np.ndarray,
                       sim: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Assign noise points to their nearest cluster centroid.

    Every object ends up in a knowledge point: dropping sparse fragments would
    hide knowledge, and making each of them a singleton KP would explode the
    KP table. Returns (labels, confidence per row).
    """
    out = labels.copy()
    unique = [int(l) for l in np.unique(labels) if l >= 0]
    if not unique:
        return np.zeros(len(labels), dtype=np.int32), np.zeros(len(labels))

    sim = similarity_matrix(positions) if sim is None else sim
    centroids = np.stack([
        positions[labels == l].mean(axis=0) for l in unique
    ])
    centroids = normalize_rows(centroids)
    sims = positions @ centroids.T
    nearest = np.argmax(sims, axis=1)
    confidence = sims[np.arange(len(positions)), nearest]

    mask = out < 0
    out[mask] = np.array([unique[i] for i in nearest[mask]], dtype=np.int32)
    return out, confidence


def _largest_share(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels[labels >= 0])
    return float(counts.max() / len(labels)) if counts.size else 0.0


def mutual_knn_adjacency(positions: np.ndarray, k: int = 8,
                         radius: float | None = 0.40) -> np.ndarray:
    """Boolean adjacency: mutual kNN edges plus an optional cosine radius."""
    n = len(positions)
    if n < 2:
        return np.zeros((n, n), dtype=bool)

    sim = positions @ positions.T
    np.fill_diagonal(sim, -np.inf)
    k_eff = min(k, n - 1)

    knn = np.argpartition(-sim, k_eff, axis=1)[:, :k_eff]
    adj = np.zeros((n, n), dtype=bool)
    adj[np.repeat(np.arange(n), k_eff), knn.ravel()] = True
    adj &= adj.T

    if radius is not None:
        sims = positions @ positions.T
        np.fill_diagonal(sims, -np.inf)
        adj |= sims >= radius
    return adj


def _connected_components(adj: np.ndarray) -> np.ndarray:
    """Label connected components, deterministically ordered by first row."""
    n = len(adj)
    labels = np.full(n, -1, dtype=np.int32)
    label = 0
    for start in range(n):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = label
        while stack:
            node = stack.pop()
            for nxt in np.flatnonzero(adj[node]):
                if labels[nxt] == -1:
                    labels[nxt] = label
                    stack.append(int(nxt))
        label += 1
    return labels


def merge_similar_clusters(positions: np.ndarray, labels: np.ndarray,
                           threshold: float = 0.92) -> np.ndarray:
    """Merge clusters whose centroids are near-identical (cos >= threshold)."""
    unique = [int(l) for l in np.unique(labels) if l >= 0]
    if len(unique) < 2:
        return labels

    centroids = normalize_rows(np.stack([
        positions[labels == l].mean(axis=0) for l in unique
    ]))
    sim = centroids @ centroids.T
    remap = {l: l for l in unique}
    for a in range(len(unique)):
        for b in range(a + 1, len(unique)):
            if sim[a, b] >= threshold:
                keep, drop = sorted((unique[a], unique[b]))
                remap[drop] = remap.get(keep, keep)
    out = labels.copy()
    for old, new in remap.items():
        out[labels == old] = new
    # Re-number sequentially, keeping order stable.
    uniq, first_idx = np.unique(out, return_index=True)
    order = np.argsort(first_idx)
    relabel = {int(uniq[o]): new for new, o in enumerate(order)}
    return np.array([relabel.get(int(l), -1) for l in out], dtype=np.int32)


def propagate_qa_membership(
    positions: np.ndarray,
    qa_of_row: np.ndarray,
    weight: float = 0.40,
    rounds: int = 1,
) -> np.ndarray:
    """Pull fragments of the same question towards their shared centroid.

    This is the QA co-membership association structure: the scoring points of
    one question describe the same knowledge unit, so their vectors should be
    mutually consistent.  Unlike a same-paper edge it does not organise the
    space by paper - it only tightens each question own fragment group.

    weight blends the original position with the question centroid:
    x <- normalize((1-w)*x + w*mean(x over the same qa)).  weight=0 returns
    the input unchanged.
    """
    if weight <= 0 or rounds <= 0:
        return positions

    out = normalize_rows(positions)
    qa_of_row = np.asarray(qa_of_row)
    if len(qa_of_row) != len(out):
        raise ValueError("qa_of_row must have one entry per position row")

    for _ in range(rounds):
        centroids: dict[int, np.ndarray] = {}
        for qa_id in np.unique(qa_of_row):
            rows = np.flatnonzero(qa_of_row == qa_id)
            centroids[int(qa_id)] = out[rows].mean(axis=0)
        blended = np.empty_like(out)
        for i, qa_id in enumerate(qa_of_row):
            blended[i] = (1.0 - weight) * out[i] + weight * centroids[int(qa_id)]
        out = normalize_rows(blended)
    return out


def cluster_positions(
    positions: np.ndarray,
    config: LayoutConfig | None = None,
) -> np.ndarray:
    """Cluster relaxed positions into knowledge-point candidates.

    Small clusters are kept (marked embryonic upstream) rather than discarded:
    the project treats early KPs as always visible.
    """
    cfg = config or LayoutConfig()
    positions = normalize_rows(positions)
    sim = similarity_matrix(positions)

    eps = cfg.radius if cfg.radius is not None else adaptive_radius(
        positions, k=cfg.min_samples, seed=cfg.seed)

    labels: np.ndarray
    while True:
        labels, _core = dbscan_clusters(
            positions, eps=eps, min_samples=cfg.min_samples, sim=sim)
        if _largest_share(labels) <= cfg.max_cluster_share or eps >= cfg.radius_max:
            break
        eps = min(eps + cfg.radius_step, cfg.radius_max)

    labels, _confidence = assign_unclustered(positions, labels, sim=sim)
    return merge_similar_clusters(positions, labels, cfg.merge_cos)


# ============================================================
# Diagnostics / metrics
# ============================================================


def partition_ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand Index between two partitions (seeded stability metric)."""
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if len(a) != len(b):
        raise ValueError("partitions must have equal length")
    n = len(a)
    if n < 2:
        return 1.0

    _, a_inv = np.unique(a, return_inverse=True)
    _, b_inv = np.unique(b, return_inverse=True)
    contingency = np.zeros((a_inv.max() + 1, b_inv.max() + 1), dtype=np.int64)
    np.add.at(contingency, (a_inv, b_inv), 1)

    def c2(x: np.ndarray) -> np.ndarray:
        return x * (x - 1) / 2.0

    sum_ij = float(c2(contingency).sum())
    sum_a = float(c2(contingency.sum(axis=1)).sum())
    sum_b = float(c2(contingency.sum(axis=0)).sum())
    total = c2(np.array([n], dtype=np.int64))[0]

    expected = sum_a * sum_b / total if total else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if abs(denom) < EPS:
        return 1.0
    return (sum_ij - expected) / denom


def layout_metrics(positions: np.ndarray, labels: np.ndarray,
                   anchors: np.ndarray | None = None) -> dict:
    """Summary metrics used by the relayout report."""
    positions = normalize_rows(positions)
    labels = np.asarray(labels)
    unique = [int(l) for l in np.unique(labels)]

    sizes = [int(np.sum(labels == l)) for l in unique]
    centroid_rows = [positions[labels == l].mean(axis=0) for l in unique]
    centroids = (normalize_rows(np.stack(centroid_rows)) if unique
                 else np.zeros((0, positions.shape[1])))
    sim = centroids @ centroids.T
    off_diag = sim[~np.eye(len(sim), dtype=bool)] if len(sim) > 1 else np.array([])

    cohesion = []
    for l in unique:
        members = positions[labels == l]
        if len(members) > 1:
            m = normalize_rows(members)
            s = m @ m.T
            cohesion.append(float(s[~np.eye(len(m), dtype=bool)].mean()))

    metrics = {
        "n_objects": int(len(positions)),
        "n_clusters": len(unique),
        "singleton_clusters": int(sum(1 for s in sizes if s == 1)),
        "singleton_ratio": float(np.mean([s == 1 for s in sizes])) if sizes else 0.0,
        "mean_cluster_size": float(np.mean(sizes)) if sizes else 0.0,
        "cohesion": float(np.mean(cohesion)) if cohesion else 0.0,
        "separation": float(off_diag.mean()) if off_diag.size else 0.0,
        "effective_dim": effective_dim(positions),
        "mean_pairwise_cos": mean_pairwise_cos(positions),
    }
    if anchors is not None:
        anchors = normalize_rows(anchors)
        drift = 1.0 - np.sum(positions * anchors, axis=1)
        metrics["anchor_drift_mean"] = float(drift.mean())
        metrics["anchor_drift_p95"] = float(np.percentile(drift, 95))
    return metrics
