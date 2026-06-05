"""KpSystem — single entry point for all knowledge-point operations.

Encapsulates: clustering, encoding, signal processing, event publishing.
External callers (pipeline, chat, web routes) interact ONLY through KpSystem.

Usage::

    from .knowledge import KpSystem
    from .knowledge._encoding import EmbeddingEncoder

    kp_sys = KpSystem(db, EmbeddingEncoder())

    # Per-QA ingestion (called from pipeline)
    kp_id = kp_sys.ingest_qa(qa_id, qa_vector)

    # Post-paper hook
    kp_sys.on_paper_completed(display_name, qa_count)

    # Signal-based vector adjustment
    kp_sys.apply_feedback_signal(qa_id, success=True)
"""

from __future__ import annotations

import numpy as np

from ..knowledge_base import QADatabase
from ..events.bus_client import EventBusClient
from ..events.events import EventType
from ._encoding import VectorEncoder
from ._clustering import EmergentClusterer, ClusterConfig, CentroidSignal


class KpSystem:
    """Facade for the complete knowledge-point subsystem.

    Wires together:
    - ``EmergentClusterer`` — QA-to-KP assignment
    - ``VectorEncoder`` — text → vector encoding
    - ``EventBusClient`` — cross-module event publishing
    """

    def __init__(
        self,
        db: QADatabase,
        encoder: VectorEncoder,
        bus: EventBusClient | None = None,
        config: ClusterConfig | None = None,
    ):
        self._db = db
        self._encoder = encoder
        self._bus = bus or EventBusClient.main()
        self._clusterer = EmergentClusterer(db, encoder, config)

    # ── Ingestion (pipeline → KpSystem) ─────────────────────────

    def ingest_qa(self, qa_id: int, qa_text: str = "") -> str:
        """Encode QA text and assign to a KP. Returns kp_id.

        Called by the pipeline for every QA, regardless of whether
        it's the first or Nth paper — no special-casing.
        """
        if qa_text:
            qa_vector = self._encoder.encode_single(qa_text)
        else:
            qa = self._db.get(qa_id)
            qa_text = f"{qa.get('question_text', '')} {qa.get('answer_text', '')}"
            qa_vector = self._encoder.encode_single(qa_text)

        kp_id = self._clusterer.assign_qa(qa_id, qa_vector)

        # Publish event
        is_new = kp_id.startswith("kp_emb_")
        if is_new:
            self._bus.publish(EventType.KP_CREATED, {
                "kp_id": kp_id,
                "qa_id": qa_id,
                "quality": "embryonic",
            })
        self._bus.publish(EventType.KP_ASSIGNED, {
            "kp_id": kp_id, "qa_id": qa_id,
        })

        return kp_id

    # ── Post-paper hooks ────────────────────────────────────────

    def on_paper_completed(self, display_name: str, qa_count: int = 0):
        """Called after each paper is fully processed.

        Triggers: decay check, quality promotion, paper_completed event.
        """
        self._clusterer.apply_embryonic_decay()
        self._clusterer.promote_stable_kps()

        new_kps = sum(
            1 for kp in self._db.kp.get_all()
            if kp.get("quality") == "embryonic"
        )

        self._bus.publish(EventType.PAPER_COMPLETED, {
            "display_name": display_name,
            "qa_count": qa_count,
            "new_kps": new_kps,
        })

    # ── Signal-based vector adjustment ──────────────────────────

    def apply_feedback_signal(self, qa_id: int, success: bool):
        """Nudge KP vector based on QA help success/failure.

        Called when a QA helped (or failed to help) answer a student question.
        """
        kp_ids = self._db.kp.get_kp_ids_for_qa(qa_id)
        if not kp_ids:
            return

        qa = self._db.get(qa_id)
        if not qa:
            return
        qa_text = f"{qa.get('question_text', '')} {qa.get('answer_text', '')}"
        qa_vec = self._encoder.encode_single(qa_text)

        for kp_id in kp_ids:
            weight = 0.05 if success else -0.02
            signal = CentroidSignal(
                source="help_feedback",
                kp_id=kp_id,
                vector=qa_vec,
                weight=weight,
                metadata={"qa_id": qa_id, "success": success},
            )
            self._clusterer.apply_signal(signal)

            self._bus.publish(EventType.KP_VECTOR_SHIFTED, {
                "kp_id": kp_id,
                "signal_source": "help_feedback",
                "shift_magnitude": abs(weight),
            })

    def apply_post_signal(self, kp_id: str, adjustment_vector: np.ndarray):
        """Apply a post-processing adjustment to a KP's centroid."""
        signal = CentroidSignal(
            source="post_process",
            kp_id=kp_id,
            vector=adjustment_vector,
            weight=0.03,
        )
        self._clusterer.apply_signal(signal)

    def apply_user_correction(self, kp_id: str, correction_vector: np.ndarray):
        """Apply a user-provided correction to a KP's centroid."""
        signal = CentroidSignal(
            source="user_input",
            kp_id=kp_id,
            vector=correction_vector,
            weight=1.0,
        )
        self._clusterer.apply_signal(signal)

    # ── Queries ─────────────────────────────────────────────────

    def get_kp_vector(self, kp_id: str) -> np.ndarray | None:
        """Get a KP's centroid vector."""
        return self._db.vector.get_kp_vector(kp_id)

    def compare_kps(self, kp_id_a: str, kp_id_b: str) -> "VectorComparison":
        """Compare two KPs via the encoder."""
        vec_a = self._db.vector.get_kp_vector(kp_id_a)
        vec_b = self._db.vector.get_kp_vector(kp_id_b)
        if vec_a is None or vec_b is None:
            from ._encoding import VectorComparison
            return VectorComparison()
        return self._encoder.compare(vec_a, vec_b)
