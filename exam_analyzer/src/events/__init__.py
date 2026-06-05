"""Event system — Node.js event bus client and type definitions.

Usage::

    from .events import EventType
    from .events.bus_client import EventBusClient

    bus = EventBusClient.main()
    bus.publish(EventType.KP_CREATED, {"kp_id": "kp_001"})
"""

from .events import EventType
from .bus_client import EventBusClient
from .manifest import ModuleManifest, MANIFESTS, validate_all

__all__ = [
    "EventType",
    "EventBusClient",
    "ModuleManifest",
    "MANIFESTS",
    "validate_all",
]
