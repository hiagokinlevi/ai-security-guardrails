from __future__ import annotations

import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


# Request-scoped correlation id, set by middleware and consumed by audit emitters.
_correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def set_correlation_id(correlation_id: str | None) -> None:
    """Set request correlation id in context.

    If None is supplied, a new UUIDv4 is generated.
    """
    _correlation_id_ctx.set(correlation_id or str(uuid.uuid4()))


def get_correlation_id() -> str:
    """Return current request correlation id, generating one if missing."""
    current = _correlation_id_ctx.get()
    if current:
        return current
    generated = str(uuid.uuid4())
    _correlation_id_ctx.set(generated)
    return generated


class AuditEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: Literal["input_validation", "policy_decision", "output_filtering"]
    correlation_id: str
    request_id: str | None = None
    user_id: str | None = None
    action: str
    decision: str
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditLogger:
    """Structured audit logger.

    Existing call sites can omit correlation_id; it is injected from request context.
    """

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    @property
    def events(self) -> list[AuditEvent]:
        return self._events

    def emit(self, **payload: Any) -> AuditEvent:
        payload.setdefault("correlation_id", get_correlation_id())
        event = AuditEvent(**payload)
        self._events.append(event)
        return event
