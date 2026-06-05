"""VectorEncoder — pluggable encoding backend for KP vector operations.

Two implementations planned:
- ``EmbeddingEncoder`` — sentence-transformer (current, default)
- ``SaeEncoder`` — sparse autoencoder (future, when trained)

The protocol is designed so SAE can be swapped in without changing any
caller code.  Feature activation storage columns exist in the schema
(``feature_activations BLOB`` on kp_vectors) but are unused until SAE
is ready.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


# ============================================================
# VectorEncoder protocol
# ============================================================


class VectorEncoder(ABC):
    """Abstract encoder — produces vectors and compares KPs.

    EmbeddingEncoder implements this with cosine similarity.
    SaeEncoder will implement this with sparse feature overlap.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output vector dimension."""
        ...

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts → [N, dim] array."""
        ...

    @abstractmethod
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text → [dim] array."""
        ...

    @abstractmethod
    def compare(self, vec_a: np.ndarray, vec_b: np.ndarray) -> "VectorComparison":
        """Compare two vectors — returns a structured comparison.

        For embedding: cosine similarity.
        For SAE: shared feature count + overlap score.
        """
        ...

    @abstractmethod
    def compare_batch(
        self, query_vec: np.ndarray, candidates: np.ndarray
    ) -> list["VectorComparison"]:
        """Compare one query vector against a batch of candidates."""
        ...


# ============================================================
# VectorComparison — structured comparison result
# ============================================================


@dataclass
class VectorComparison:
    """Result of comparing two KP vectors.

    Embedding backend fills ``similarity``.
    SAE backend fills ``shared_features`` and ``feature_overlap_score``.
    """

    similarity: float = 0.0
    shared_features: list[int] = field(default_factory=list)
    feature_overlap_score: float = 0.0
    method: str = "cosine"  # "cosine" or "sae_overlap"
    metadata: dict = field(default_factory=dict)


# ============================================================
# EmbeddingEncoder — current default
# ============================================================


class EmbeddingEncoder(VectorEncoder):
    """Sentence-transformer based encoder."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self._model_name = model_name
        self._model = None

    @property
    def dim(self) -> int:
        return 384

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        self._ensure_model()
        return self._model.encode(texts, show_progress_bar=False)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def compare(self, vec_a: np.ndarray, vec_b: np.ndarray) -> VectorComparison:
        sim = float(np.dot(vec_a, vec_b) / (
            np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-10
        ))
        return VectorComparison(similarity=sim, method="cosine")

    def compare_batch(
        self, query_vec: np.ndarray, candidates: np.ndarray
    ) -> list[VectorComparison]:
        norms_q = np.linalg.norm(query_vec)
        norms_c = np.linalg.norm(candidates, axis=1)
        sims = np.dot(candidates, query_vec) / (norms_c * norms_q + 1e-10)
        return [
            VectorComparison(similarity=float(s), method="cosine")
            for s in sims
        ]
