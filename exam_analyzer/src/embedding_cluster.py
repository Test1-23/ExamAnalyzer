"""Semantic embedding utilities: model caching, language detection, vector encoding."""

import os as _os
_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import time
import threading
import unicodedata
import numpy as np
from typing import List, Optional

from .logger import get_logger

_log = get_logger()

from .constants import CJK_DETECTION_THRESHOLD, EMBEDDING_BATCH_SIZE


# ---------------------------------------------------------------------------
# Global model cache — load each model only once per process
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict = {}  # model_name -> SentenceTransformer
_MODEL_CACHE_LOCK = threading.Lock()


def _get_model(model_name: str):
    """Load (or retrieve cached) SentenceTransformer model. Thread-safe."""
    if model_name not in _MODEL_CACHE:
        with _MODEL_CACHE_LOCK:
            if model_name not in _MODEL_CACHE:
                t0 = time.time()
                from sentence_transformers import SentenceTransformer
                # Disable download progress bar noise in logs
                _MODEL_CACHE[model_name] = SentenceTransformer(
                    model_name,
                    trust_remote_code=True,
                )
                elapsed = int((time.time() - t0) * 1000)
                _log.info(f"Embedding model loaded: {model_name}, {elapsed}ms")
    return _MODEL_CACHE[model_name]


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def _detect_language(texts: List[str]) -> str:
    """Detect whether texts are primarily Chinese or English.
    Returns 'zh' if >5% CJK characters, else 'en'.
    """
    def _is_cjk(ch: str) -> bool:
        if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
            return True
        return unicodedata.name(ch, '').startswith('CJK')

    total = 0
    cjk = 0
    for t in texts:
        for ch in t:
            total += 1
            if _is_cjk(ch):
                cjk += 1
    if total == 0:
        return 'en'
    ratio = cjk / total
    return 'zh' if ratio > CJK_DETECTION_THRESHOLD else 'en'


def detect_content_lang(text: str) -> str:
    """Return 'zh' if significant Chinese content, else 'en'.
    Uses same criteria as _detect_language for consistency."""
    return _detect_language([text])


MODEL_MAP = {
    'zh': 'paraphrase-multilingual-MiniLM-L12-v2',
    'en': 'all-MiniLM-L6-v2',
}

# Topic-level embeddings use the multilingual model for cross-language compatibility
TOPIC_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------------------------
# EmbeddingClusterer — vector encoding for topic merge
# ---------------------------------------------------------------------------

class EmbeddingClusterer:
    """Compute and cache sentence embeddings for a list of texts."""

    def __init__(self, texts: List[str]):
        self._texts = texts
        self._language = _detect_language(texts)
        self._model_name = MODEL_MAP[self._language]
        self._vectors: Optional[np.ndarray] = None

    @property
    def vectors(self) -> np.ndarray:
        if self._vectors is None:
            self._vectors = self._encode(self._texts)
        return self._vectors

    def _encode(self, texts: List[str]) -> np.ndarray:
        t0 = time.time()
        model = _get_model(self._model_name)
        result = model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        elapsed = int((time.time() - t0) * 1000)
        _log.debug(f"Embedding encode: model={self._model_name}, texts={len(texts)}, {elapsed}ms")
        return result


# ---------------------------------------------------------------------------
# Shared clustering utility — replaces duplicated threshold-based grouping
# ---------------------------------------------------------------------------

def cluster_by_cosine(vectors: np.ndarray, threshold: float,
                      min_group_size: int = 2) -> list[list[int]]:
    """Greedy single-pass cosine clustering. Returns list of index-groups."""
    n = len(vectors)
    assigned = [False] * n
    groups: list[list[int]] = []
    for i in range(n):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if float(np.dot(vectors[i], vectors[j])) >= threshold:
                group.append(j)
                assigned[j] = True
        if len(group) >= min_group_size:
            groups.append(group)
    return groups
