"""EmergentClusterer — KPs form naturally via vector similarity.

No preset K.  No fixed threshold.  Clusters emerge as QAs accumulate.

Design:
- Each QA vector either joins an existing KP or creates a new embryonic one
- Per-KP independent threshold based on internal member distribution
- Embryonic KPs always visible, decay if no new QAs join
- Centroid adjusted via signal extension points (help feedback, post-process, user input)

Replaces the old ``cluster_qas()`` / ``_build_similarity_graph()`` /
``_find_clusters()`` in knowledge_graph.py (marked for deprecation in concern.md).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from ..knowledge_base import QADatabase
from ._encoding import VectorEncoder, VectorComparison


# ============================================================
# Configuration
# ============================================================

@dataclass
class ClusterConfig:
    """Tunable parameters for emergent clustering."""

    # Absolute floor — below this, QA always creates a new KP
    min_similarity: float = 0.50

    # Embryonic KPs need this many QAs to reach 'stable' quality
    embryonic_evidence_threshold: int = 3

    # Decay factor per re-evaluation without new members
    # centroid *= (1 - decay) each check → gradually fades
    embryonic_decay: float = 0.05

    # MAD multiplier for per-KP threshold
    # threshold = max(min_similarity, median - mad_multiplier * MAD)
    mad_multiplier: float = 1.0

    # Minimum members before MAD-based threshold kicks in
    # (small KPs use min_similarity directly)
    min_members_for_stats: int = 3


# ============================================================
# Signal extension points
# ============================================================

@dataclass
class CentroidSignal:
    """A signal that nudges a KP's centroid vector.

    Extension point — new signal types can be added by creating
    new CentroidSignal instances with different sources.
    """

    source: str           # "help_feedback", "post_process", "user_input", ...
    kp_id: str
    vector: np.ndarray    # signal direction
    weight: float         # -1.0 (push away) to 1.0 (pull toward)
    metadata: dict = field(default_factory=dict)


# ============================================================
# EmergentClusterer
# ============================================================


class EmergentClusterer:
    """Assign QAs to KPs via emergent clustering.

    Usage::

        encoder = EmbeddingEncoder()
        clusterer = EmergentClusterer(db, encoder)

        # Per-QA ingestion
        qa_vec = encoder.encode_single(qa_text)
        kp_id = clusterer.assign_qa(qa_id, qa_vec)

        # Signal-based adjustment
        signal = CentroidSignal(
            source="help_feedback", kp_id=kp_id,
            vector=qa_vec, weight=0.05,
        )
        clusterer.apply_signal(signal)
    """

    def __init__(
        self,
        db: QADatabase,
        encoder: VectorEncoder,
        config: ClusterConfig | None = None,
    ):
        self._db = db
        self._encoder = encoder
        self._config = config or ClusterConfig()

        # Registered signal handlers — extensible
        self._signal_handlers: dict[str, Callable[[CentroidSignal], None]] = {
            "help_feedback": self._handle_help_signal,
            "post_process": self._handle_post_signal,
            "user_input": self._handle_user_signal,
        }

    # ── Signal handler registration ────────────────────────────

    def register_signal_handler(
        self, source: str, handler: Callable[[CentroidSignal], None]
    ):
        """Register a new signal handler for future signal types."""
        self._signal_handlers[source] = handler

    def apply_signal(self, signal: CentroidSignal):
        """Apply a centroid adjustment signal via registered handler."""
        handler = self._signal_handlers.get(signal.source)
        if handler:
            handler(signal)

    # ── Core: QA assignment ─────────────────────────────────────

    def assign_qa(self, qa_id: int, qa_vector: np.ndarray) -> str:
        """Assign a QA to an existing KP or create a new embryonic one.

        Returns the kp_id.
        """
        existing_kps = self._db.kp.get_all_with_vectors()

        if not existing_kps:
            return self._create_embryonic(qa_id, qa_vector)

        # Get centroid vectors for all KPs
        kp_data = []
        kp_vectors = []
        for kp in existing_kps:
            centroid = self._db.vector.get_kp_vector(kp["id"])
            if centroid is not None:
                kp_data.append(kp)
                kp_vectors.append(centroid)

        if not kp_vectors:
            return self._create_embryonic(qa_id, qa_vector)

        kp_matrix = np.stack(kp_vectors)

        # Compute similarities to all KPs
        comparisons = self._encoder.compare_batch(qa_vector, kp_matrix)

        # Find best match
        best_idx = int(np.argmax([c.similarity for c in comparisons]))
        best_sim = comparisons[best_idx].similarity
        best_kp = kp_data[best_idx]

        # Get per-KP threshold
        threshold = self._compute_threshold(best_kp["id"])

        if best_sim >= threshold:
            # Join existing KP
            self._db.kp.set_membership(
                best_kp["id"], qa_id, float(best_sim)
            )
            self._update_centroid(best_kp["id"])
            return best_kp["id"]
        else:
            # Create new embryonic KP
            return self._create_embryonic(qa_id, qa_vector)

    # ── Per-KP threshold ────────────────────────────────────────

    def _compute_threshold(self, kp_id: str) -> float:
        """Compute adaptive threshold for a single KP.

        threshold = max(min_similarity, median - MAD)
        Falls back to min_similarity for KPs with too few members.
        """
        cfg = self._config
        member_vecs = self._get_member_vectors(kp_id)
        if len(member_vecs) < cfg.min_members_for_stats:
            return cfg.min_similarity

        # Compute pairwise similarities within the cluster
        sims = self._pairwise_similarities(member_vecs)
        if len(sims) == 0:
            return cfg.min_similarity

        median = float(np.median(sims))
        mad = float(np.median(np.abs(np.array(sims) - median)))

        return max(cfg.min_similarity, median - cfg.mad_multiplier * mad)

    def _get_member_vectors(self, kp_id: str) -> np.ndarray:
        """Get all member QA vectors for a KP."""
        qa_ids = self._db.kp.get_member_qa_ids(kp_id)
        if not qa_ids:
            return np.array([])
        vectors = []
        for qa_id in qa_ids:
            # Re-encode QA text on demand — in production this would
            # come from a pre-computed cache
            qa = self._db.get(qa_id)
            if qa:
                text = f"{qa.get('question_text', '')} {qa.get('answer_text', '')}"
                vec = self._encoder.encode_single(text)
                vectors.append(vec)
        if not vectors:
            return np.array([])
        return np.stack(vectors)

    def _pairwise_similarities(self, vectors: np.ndarray) -> list[float]:
        """Compute all pairwise cosine similarities."""
        n = len(vectors)
        if n < 2:
            return []
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / (norms + 1e-10)
        # All-pairs cosine
        sim_matrix = normalized @ normalized.T
        # Upper triangle only
        return [
            float(sim_matrix[i, j])
            for i in range(n) for j in range(i + 1, n)
        ]

    # ── Centroid update ─────────────────────────────────────────

    def _update_centroid(self, kp_id: str):
        """Recalculate KP centroid as mean of member QA vectors."""
        member_vecs = self._get_member_vectors(kp_id)
        if len(member_vecs) == 0:
            return
        new_centroid = np.mean(member_vecs, axis=0)
        self._db.vector.upsert_kp_vector(kp_id, new_centroid)

    def _create_embryonic(self, qa_id: int, qa_vector: np.ndarray) -> str:
        """Create a new embryonic KP for a QA that doesn't match any existing KP."""
        kp_id = f"kp_emb_{qa_id}"
        from ..models import KPSpec
        spec = KPSpec(kp_id=kp_id, quality="embryonic")
        self._db.kp.upsert(spec)
        self._db.kp.set_membership(kp_id, qa_id, 1.0)
        self._db.vector.upsert_kp_vector(kp_id, qa_vector)
        return kp_id

    # ── Signal handlers ─────────────────────────────────────────

    def _handle_help_signal(self, signal: CentroidSignal):
        """Help feedback: success pulls closer, failure pushes away."""
        centroid = self._db.vector.get_kp_vector(signal.kp_id)
        if centroid is None:
            return
        new_centroid = centroid + signal.weight * signal.vector
        # Normalize to prevent unbounded growth
        new_centroid = new_centroid / (np.linalg.norm(new_centroid) + 1e-10)
        self._db.vector.upsert_kp_vector(signal.kp_id, new_centroid)

    def _handle_post_signal(self, signal: CentroidSignal):
        """Post-process signal: same EMA adjustment, smaller default weight."""
        self._handle_help_signal(signal)

    def _handle_user_signal(self, signal: CentroidSignal):
        """User input signal: explicit correction, full weight."""
        centroid = self._db.vector.get_kp_vector(signal.kp_id)
        if centroid is None:
            return
        # User corrections are authoritative — blend at 50%
        new_centroid = 0.5 * centroid + 0.5 * signal.vector
        self._db.vector.upsert_kp_vector(signal.kp_id, new_centroid)

    # ── Decay check ─────────────────────────────────────────────

    def apply_embryonic_decay(self):
        """Periodic check: decay embryonic KPs that haven't grown.

        Call after each paper processing cycle.  KPs with
        ``quality='embryonic'`` that haven't gained members
        have their centroid slightly faded.
        """
        cfg = self._config
        embryonic = self._db.kp.get_all()
        for kp in embryonic:
            if kp.get("quality") != "embryonic":
                continue
            count = self._db.kp.count_members(kp["id"])
            if count < cfg.embryonic_evidence_threshold:
                centroid = self._db.vector.get_kp_vector(kp["id"])
                if centroid is not None:
                    faded = centroid * (1.0 - cfg.embryonic_decay)
                    self._db.vector.upsert_kp_vector(kp["id"], faded)

    # ── Quality promotion ───────────────────────────────────────

    def promote_stable_kps(self):
        """Promote embryonic KPs with enough evidence to 'stable'."""
        cfg = self._config
        embryonic = self._db.kp.get_all()
        for kp in embryonic:
            if kp.get("quality") != "embryonic":
                continue
            count = self._db.kp.count_members(kp["id"])
            if count >= cfg.embryonic_evidence_threshold:
                self._db.topic.upsert(
                    kp["id"],
                    name=kp.get("name", kp["id"]),
                    quality="stable",
                )
