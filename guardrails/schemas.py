from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DecisionCode(str, Enum):
    """Deterministic decision codes emitted in structured audit events."""

    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"
    INPUT_TOO_LARGE = "INPUT_TOO_LARGE"


class AuditEvent(BaseModel):
    """Structured audit event emitted by guardrails components."""

    decision: DecisionCode = Field(..., description="Deterministic decision code")
    reason: str = Field(..., description="Human-readable reason for the decision")
    metadata: dict[str, Any] = Field(default_factory=dict)
