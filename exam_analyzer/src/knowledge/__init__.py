"""Knowledge point system — encoding, clustering, graph, lifecycle.

exports:
- VectorEncoder / EmbeddingEncoder — pluggable encoding backend
- EmergentClusterer — QA-to-KP assignment via emergent clustering
"""

from ._encoding import VectorEncoder, EmbeddingEncoder, VectorComparison
from ._clustering import EmergentClusterer, ClusterConfig, CentroidSignal
from .system import KpSystem
from ._attention import (
    AttentionResult, QwenAttentionExtractor, alignment_scores,
    build_activation_units, cross_alignment_scores, extract_batch, gate_text,
    projected_attention,
)

__all__ = [
    "VectorEncoder",
    "EmbeddingEncoder",
    "VectorComparison",
    "EmergentClusterer",
    "ClusterConfig",
    "CentroidSignal",
    "KpSystem",
    "AttentionResult",
    "QwenAttentionExtractor",
    "alignment_scores",
    "build_activation_units",
    "cross_alignment_scores",
    "extract_batch",
    "gate_text",
    "projected_attention",
]
