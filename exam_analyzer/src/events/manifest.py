"""Module manifest — declares publishes/subscribes for each module.

Loaded at startup to validate event contracts.  Each module has a
``ModuleManifest`` that lists which events it publishes and subscribes to.

The event bus itself does NOT enforce manifests (it's pub/sub by design).
This module exists for:

1. **Documentation** — visible contract for each module
2. **Runtime validation** — ``validate_all()`` checks that every subscribed
   event has at least one publisher registered
3. **Startup check** — warns if event flow has missing links
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .events import EventType


@dataclass
class ModuleManifest:
    """Declares a module's event contract."""
    module: str
    description: str = ""
    publishes: list[str] = field(default_factory=list)
    subscribes: list[str] = field(default_factory=list)


# ============================================================
# Module manifests — single source of truth for event contracts
# ============================================================

MANIFESTS: list[ModuleManifest] = [
    ModuleManifest(
        module="pipeline",
        description="PDF extraction, QA pairing, and analysis orchestration",
        publishes=[
            EventType.QA_INSERTED,
            EventType.QA_VECTOR_READY,
            EventType.PAPER_COMPLETED,
        ],
        subscribes=[
            EventType.ANALYSIS_COMPLETED,
            EventType.KP_CREATED,
            EventType.SYSTEM_SHUTDOWN,
        ],
    ),
    ModuleManifest(
        module="knowledge",
        description="KP lifecycle: clustering, graph, distillation, refinement, evolution",
        publishes=[
            EventType.KP_CREATED,
            EventType.KP_UPDATED,
            EventType.KP_ASSIGNED,
            EventType.KP_MERGED,
            EventType.KP_SPLIT,
            EventType.KP_DISSOLVED,
            EventType.KP_VECTOR_SHIFTED,
            EventType.EDGE_DISCOVERED,
            EventType.EDGE_UPDATED,
            EventType.EDGE_REMOVED,
            EventType.CLUSTER_CHANGED,
        ],
        subscribes=[
            EventType.QA_INSERTED,
            EventType.QA_VECTOR_READY,
            EventType.PAPER_COMPLETED,
            EventType.ANOMALY_DETECTED,
            EventType.DRIFT_DETECTED,
            EventType.STUDENT_CONFUSION,
        ],
    ),
    ModuleManifest(
        module="analysis",
        description="Post-pipeline diagnostics: cross-paper, migration, pitfalls, trends",
        publishes=[
            EventType.ANOMALY_DETECTED,
            EventType.DRIFT_DETECTED,
            EventType.TOPIC_MIGRATED,
            EventType.ANALYSIS_COMPLETED,
        ],
        subscribes=[
            EventType.PAPER_COMPLETED,
            EventType.EDGE_DISCOVERED,
            EventType.KP_MERGED,
            EventType.KP_SPLIT,
            EventType.KP_CREATED,
        ],
    ),
    ModuleManifest(
        module="chat",
        description="Multi-agent chat assistant with dual-channel knowledge retrieval",
        publishes=[
            EventType.CHAT_QUERY,
            EventType.CHAT_RESPONSE,
            EventType.STUDENT_CONFUSION,
        ],
        subscribes=[
            EventType.KP_CREATED,
            EventType.KP_UPDATED,
            EventType.EDGE_DISCOVERED,
        ],
    ),
    ModuleManifest(
        module="web",
        description="Flask web layer: routes, state management, SSE push",
        publishes=[
            EventType.SYSTEM_STARTUP,
            EventType.SYSTEM_SHUTDOWN,
        ],
        subscribes=[
            EventType.PAPER_COMPLETED,
            EventType.KP_CREATED,
            EventType.ANALYSIS_COMPLETED,
            EventType.SYSTEM_STARTUP,
            EventType.SYSTEM_SHUTDOWN,
        ],
    ),
]


# ============================================================
# Validation
# ============================================================

def validate_all() -> list[str]:
    """Validate that all subscribed events have at least one publisher.

    Returns list of warnings (empty = all good).
    """
    warnings: list[str] = []
    all_publishes: set[str] = set()
    for m in MANIFESTS:
        all_publishes.update(m.publishes)

    for m in MANIFESTS:
        for sub in m.subscribes:
            if sub not in all_publishes:
                warnings.append(
                    f"[{m.module}] subscribes to '{sub}' but no module publishes it"
                )

    # Check for orphan publishes (no subscribers)
    all_subscribes: set[str] = set()
    for m in MANIFESTS:
        all_subscribes.update(m.subscribes)

    for m in MANIFESTS:
        for pub in m.publishes:
            if pub not in all_subscribes:
                warnings.append(
                    f"[{m.module}] publishes '{pub}' but no module subscribes to it"
                )

    return warnings
