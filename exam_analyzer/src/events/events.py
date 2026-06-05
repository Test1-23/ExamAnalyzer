"""Event type constants and payload dataclasses for the KP lifecycle.

All modules publish and subscribe using these well-known event types.
New event types MUST be registered here first — no ad-hoc strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ============================================================
# Event type constants (single source of truth)
# ============================================================

class EventType:
    """Well-known event types for the KP lifecycle bus.

    Naming convention: ``{domain}.{action_past_tense}``
    """

    # -- Pipeline → Knowledge --
    QA_INSERTED           = "qa.inserted"
    QA_VECTOR_READY       = "qa.vector_ready"
    PAPER_COMPLETED       = "paper.completed"

    # -- Knowledge (KP lifecycle) --
    KP_ASSIGNED           = "kp.assigned"
    KP_CREATED            = "kp.created"
    KP_UPDATED            = "kp.updated"
    KP_MERGED             = "kp.merged"
    KP_SPLIT              = "kp.split"
    KP_DISSOLVED          = "kp.dissolved"
    KP_VECTOR_SHIFTED     = "kp.vector_shifted"
    EDGE_DISCOVERED       = "edge.discovered"
    EDGE_UPDATED          = "edge.updated"
    EDGE_CANDIDATE        = "edge.candidate"       # SAE suggests a possible edge
    EDGE_REMOVED          = "edge.removed"

    # -- Knowledge → Analysis --
    CLUSTER_CHANGED       = "cluster.changed"
    TOPIC_MIGRATED        = "topic.migrated"

    # -- Analysis → Knowledge (feedback loop) --
    ANOMALY_DETECTED      = "anomaly.detected"
    DRIFT_DETECTED        = "drift.detected"
    ANALYSIS_COMPLETED    = "analysis.completed"

    # -- Chat --
    CHAT_QUERY            = "chat.query"
    CHAT_RESPONSE         = "chat.response"
    STUDENT_CONFUSION     = "student.confusion"

    # -- System --
    SYSTEM_STARTUP        = "system.startup"
    SYSTEM_SHUTDOWN       = "system.shutdown"


# ============================================================
# Payload dataclasses
# ============================================================

@dataclass
class BaseEvent:
    """Base event with common metadata."""
    type: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_module: str = ""


@dataclass
class QaInserted(BaseEvent):
    qa_id: int = 0
    topic: str = ""
    paper: str = ""


@dataclass
class KpCreated(BaseEvent):
    kp_id: str = ""
    name: str = ""
    quality: str = "embryonic"


@dataclass
class KpUpdated(BaseEvent):
    kp_id: str = ""
    changed_fields: list[str] = field(default_factory=list)


@dataclass
class KpMerged(BaseEvent):
    from_kp: str = ""
    into_kp: str = ""
    reason: str = ""


@dataclass
class KpSplit(BaseEvent):
    original_kp: str = ""
    resulting_kps: list[str] = field(default_factory=list)


@dataclass
class EdgeDiscovered(BaseEvent):
    source_kp: str = ""
    target_kp: str = ""
    edge_type: str = ""
    confidence: str = "low"
    discovery_method: str = ""  # "graph", "vector", "sae", "sequential"


@dataclass
class EdgeCandidate(BaseEvent):
    """SAE or vector suggests a possible edge — needs LLM verification."""
    source_kp: str = ""
    target_kp: str = ""
    suggested_type: str = "related"
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class PaperCompleted(BaseEvent):
    display_name: str = ""
    qa_count: int = 0
    new_kps: int = 0


@dataclass
class KpVectorShifted(BaseEvent):
    kp_id: str = ""
    signal_source: str = ""     # "help_feedback", "post_process", "user_input"
    shift_magnitude: float = 0.0
