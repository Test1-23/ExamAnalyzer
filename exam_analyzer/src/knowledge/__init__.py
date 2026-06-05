"""Knowledge point system — encoding, clustering, graph, lifecycle.

exports:
- VectorEncoder / EmbeddingEncoder — pluggable encoding backend
- EmergentClusterer — QA-to-KP assignment via emergent clustering
"""

from ._encoding import VectorEncoder, EmbeddingEncoder, VectorComparison
from ._clustering import EmergentClusterer, ClusterConfig, CentroidSignal

__all__ = [
    "VectorEncoder",
    "EmbeddingEncoder",
    "VectorComparison",
    "EmergentClusterer",
    "ClusterConfig",
    "CentroidSignal",
]
