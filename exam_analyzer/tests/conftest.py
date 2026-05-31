"""Shared fixtures for mock LLM testing — no network, no real API keys.

Provides:
- MockFlashClient: returns predefined JSON responses by prompt type
- mock_db: temporary SQLite QADatabase with full schema, auto-cleaned
- mock_flash: monkeypatch that replaces call_flash with MockFlashClient
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.knowledge_base import QADatabase
from src.schema import SCHEMA_DDL, SCHEMA_INDEXES, SCHEMA_MIGRATIONS


# ═══════════════════════════════════════════════════════════════
# MockFlashClient — predefined JSON responses for each prompt type
# ═══════════════════════════════════════════════════════════════

class MockFlashClient:
    """Drop-in replacement for ``call_flash`` in tests.

    Callers register expected responses by message content keyword, or use
    a default response.  The `respond_with` dict maps a substring of the
    user message to the JSON dict that should be returned.
    """

    def __init__(self, default: dict | None = None):
        self.default = default or {"result": "ok"}
        self.respond_with: dict[str, dict] = {}
        self.calls: list[dict] = []  # record of all calls for assertions

    def register(self, keyword: str, response: dict):
        """Map a message-content substring to a JSON response."""
        self.respond_with[keyword] = response

    def __call__(self, client, messages, max_retries=1, debug_callback=None, json_mode=False):
        """Mimic call_flash(client, messages, ...) → dict."""
        user_content = ""
        for m in messages:
            if m.get("role") == "user":
                user_content += m.get("content", "")
        result = self.default
        for keyword, response in self.respond_with.items():
            if keyword in user_content:
                result = response
                break
        self.calls.append({
            "messages": messages,
            "user_content": user_content,
            "result": result,
        })
        return dict(result)  # shallow copy so tests can mutate originals


# ═══════════════════════════════════════════════════════════════
# DB fixture — temp SQLite with full schema
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_db():
    """Temporary QADatabase with full schema, cleaned up after test."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="test_exam_")
    os.close(fd)
    try:
        db = QADatabase(path)
        yield db
    finally:
        db.close()
        try:
            os.unlink(path)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════
# monkeypatch fixture — inject MockFlashClient
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_flash(monkeypatch):
    """Replace ``call_flash`` with MockFlashClient for the duration of a test.

    Returns the MockFlashClient instance so tests can register responses
    and inspect recorded calls.
    """
    mock = MockFlashClient()
    # Patch every call site — each module imports call_flash into its own namespace
    for mod_name in (
        "src.deepseek_client",
        "src.pipeline",
        "src.distiller",
        "src.offline_analyzer",
        "src.evolution",
        "src.adversarial_refiner",
        "src.knowledge_graph",
        "src.prompt_factory",
    ):
        try:
            monkeypatch.setattr(f"{mod_name}.call_flash", mock)
        except AttributeError:
            pass  # module may not import call_flash
    # Also patch the authoritative definition
    monkeypatch.setattr("src.deepseek_client.call_flash", mock)
    return mock


# ═══════════════════════════════════════════════════════════════
# Embedding mock — return fake normalized vectors
# ═══════════════════════════════════════════════════════════════

class _FakeModel:
    """Returns a tiny random normalized vector per encode call."""
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True,
               batch_size=64, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        rng = np.random.RandomState(hash(str(texts)) % (2 ** 31))
        vecs = rng.randn(len(texts), 384).astype(np.float32)
        if normalize_embeddings:
            vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        if convert_to_numpy:
            return vecs
        return vecs.tolist()


@pytest.fixture
def mock_embedding(monkeypatch):
    """Replace ``_get_model`` so tests never download sentence-transformers."""
    model = _FakeModel()
    for mod_name in ("src.embedding_cluster", "src.knowledge_base", "src.distiller",
                     "src.evolution", "src.offline_analyzer", "src.pipeline_diagnostics",
                     "src.adversarial_refiner", "src.knowledge_graph"):
        try:
            monkeypatch.setattr(f"{mod_name}._get_model", lambda name, model=model: model)
        except AttributeError:
            pass
    monkeypatch.setattr("src.embedding_cluster._get_model", lambda name, model=model: model)
    return model


# ═══════════════════════════════════════════════════════════════
# ProgressTracker mock
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def mock_tracker():
    """Fake ProgressTracker that records step/set_status calls for assertions."""
    from unittest.mock import MagicMock
    return MagicMock()
