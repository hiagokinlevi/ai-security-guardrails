from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class PolicyDecisionCode(str, Enum):
    """Machine-stable policy/audit classification codes."""

    INPUT_INJECTION_BLOCK = "INPUT_INJECTION_BLOCK"
    OUTPUT_PII_REDACTED = "OUTPUT_PII_REDACTED"
    TOKEN_BUDGET_EXCEEDED = "TOKEN_BUDGET_EXCEEDED"
    POLICY_LOAD_FAILURE = "POLICY_LOAD_FAILURE"


class SecurityEvent(BaseModel):
    """Structured audit event emitted by guardrail decisions."""

    event_type: str
    action: Literal["allow", "warn", "block"]
    decision_code: PolicyDecisionCode = Field(
        ...,
        description="Deterministic machine-readable policy decision classification.",
    )
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
