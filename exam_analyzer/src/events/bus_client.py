"""EventBusClient — publish events to Node.js event bus via HTTP.

Provides:
- ``publish(event_type, payload)`` — fire-and-forget HTTP POST
- ``EventBusClient.main()`` — connect to main bus (port 3030)
- ``EventBusClient.sub(knowledge)`` — connect to sub-bus (ports 3031-3035)

Usage::

    bus = EventBusClient.main()
    bus.publish(EventType.KP_CREATED, {"kp_id": "...", "name": "..."})

    # Or to a specific sub-bus:
    knowledge_bus = EventBusClient.sub("knowledge")
"""

from __future__ import annotations

import json
import threading
import urllib.request
import urllib.error
from typing import Any

from .events import EventType


# Sub-bus port offsets from main bus port
_SUB_BUS_PORTS: dict[str, int] = {
    "main":      0,
    "knowledge": 1,
    "analysis":  2,
    "pipeline":  3,
    "chat":      4,
    "web":       5,
}


class EventBusClient:
    """Publish events to the Node.js event bus.

    Falls back gracefully if bus is unreachable — publishes are fire-and-forget.
    Subscriptions use WebSocket (optional — import ws_client for that).
    """

    def __init__(self, bus_url: str, bus_name: str = "main"):
        self._bus_url = bus_url.rstrip("/")
        self._bus_name = bus_name
        self._lock = threading.Lock()
        self._fail_count = 0
        self._max_fail_log = 10  # only log first N failures

    # ── Factory methods ────────────────────────────────────────

    @classmethod
    def main(cls, base_url: str = "http://127.0.0.1:3030") -> "EventBusClient":
        """Connect to the main event bus."""
        return cls(base_url, "main")

    @classmethod
    def sub(cls, module: str, base_port: int = 3030) -> "EventBusClient":
        """Connect to a module's sub-bus.

        Args:
            module: One of "knowledge", "analysis", "pipeline", "chat", "web"
            base_port: Main bus port (sub-buses are at base_port + offset)
        """
        offset = _SUB_BUS_PORTS.get(module, 0)
        port = base_port + offset
        return cls(f"http://127.0.0.1:{port}", module)

    # ── Publish ─────────────────────────────────────────────────

    def publish(self, event_type: str, payload: dict[str, Any] | None = None) -> bool:
        """Publish an event. Returns True if accepted, False if bus unreachable.

        Fire-and-forget — does not raise on network errors.
        """
        data = json.dumps(payload or {}).encode("utf-8")
        url = f"{self._bus_url}/publish/{event_type}"

        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                body = json.loads(resp.read())
                return body.get("ok", False)
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            self._record_failure()
            return False

    def publish_or_raise(self, event_type: str, payload: dict[str, Any] | None = None):
        """Publish an event. Raises RuntimeError if bus is unreachable.

        Use this for critical events that MUST be delivered.
        """
        if not self.publish(event_type, payload):
            raise RuntimeError(
                f"Event bus unreachable: {self._bus_url} (event: {event_type})"
            )

    # ── Health ──────────────────────────────────────────────────

    def health(self) -> dict[str, Any] | None:
        """Check bus health. Returns stats dict or None if unreachable."""
        try:
            url = f"{self._bus_url}/health"
            with urllib.request.urlopen(url, timeout=2) as resp:
                return json.loads(resp.read())
        except Exception:
            return None

    def is_alive(self) -> bool:
        """Quick check: is the bus reachable?"""
        return self.health() is not None

    # ── Internals ──────────────────────────────────────────────

    def _record_failure(self):
        with self._lock:
            self._fail_count += 1
            if self._fail_count <= self._max_fail_log:
                import logging
                logging.getLogger("exam_analyzer").warning(
                    f"EventBusClient[{self._bus_name}]: publish failed "
                    f"(#{self._fail_count}, url={self._bus_url})"
                )
