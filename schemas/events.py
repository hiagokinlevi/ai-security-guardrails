"""
Pydantic Event Schemas
=======================
Defines the canonical data models for all guardrail events.

These schemas are used for:
- API request/response validation
- Audit log serialization
- Integration with external SIEM / observability tools

All models use Pydantic v2. Fields use snake_case for consistency with
Python conventions and JSON serialization targets.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------


class DecisionType(str, Enum):
    """Generic decision enumeration used across input and output events."""

    ALLOW = "allow"
    ALLOW_WITH_WARNING = "allow_with_warning"
    SEND_TO_REVIEW = "send_to_review"
    BLOCK = "block"
    PASS = "pass"
    PASS_REDACTED = "pass_redacted"


class PolicyActionType(str, Enum):
    """Actions available in the policy engine."""

    ALLOW = "allow"
    ALLOW_WITH_WARNING = "allow_with_warning"
    SEND_TO_REVIEW = "send_to_review"
    BLOCK = "block"


# ---------------------------------------------------------------------------
# Input validation event
# ---------------------------------------------------------------------------


class InputValidationEvent(BaseModel):
    """
    Schema for an input validation result.

    Emitted before the user's message is sent to the model.
    """

    request_id: str = Field(description="Unique request identifier for correlation.")
    decision: DecisionType = Field(description="Validation decision.")
    risk_score: float = Field(ge=0.0, le=1.0, description="Risk score between 0.0 and 1.0.")
    risk_flags: list[str] = Field(
        default_factory=list,
        description="List of risk signal names detected.",
    )
    input_length: int = Field(ge=0, description="Character length of the original input.")
    reason: Optional[str] = Field(None, description="Human-readable reason for the decision.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the event was recorded.",
    )

    @field_validator("risk_score")
    @classmethod
    def round_risk_score(cls, v: float) -> float:
        """Ensure risk scores are stored with consistent precision."""
        return round(v, 4)

    model_config = {"json_schema_extra": {
        "example": {
            "request_id": "req_abc123def456",
            "decision": "allow_with_warning",
            "risk_score": 0.35,
            "risk_flags": ["email_address"],
            "input_length": 142,
            "reason": None,
        }
    }}


# ---------------------------------------------------------------------------
# Output filtering event
# ---------------------------------------------------------------------------


class OutputFilterEvent(BaseModel):
    """
    Schema for an output filter result.

    Emitted after the model responds, before the response is returned to the user.
    """

    request_id: str = Field(description="Unique request identifier for correlation.")
    decision: DecisionType = Field(description="Filter decision.")
    risk_score: float = Field(ge=0.0, le=1.0, description="Risk score between 0.0 and 1.0.")
    risk_flags: list[str] = Field(default_factory=list)
    was_redacted: bool = Field(
        default=False,
        description="True if any content was redacted from the output.",
    )
    reason: Optional[str] = Field(None)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("risk_score")
    @classmethod
    def round_risk_score(cls, v: float) -> float:
        return round(v, 4)


# ---------------------------------------------------------------------------
# Policy decision event
# ---------------------------------------------------------------------------


class PolicyDecisionEvent(BaseModel):
    """
    Schema for a policy engine evaluation result.

    Captures which policy was applied and what action it produced.
    """

    request_id: str
    action: PolicyActionType
    policy_name: str
    policy_version: str
    applied_rules: list[str] = Field(default_factory=list)
    reason: str = ""
    latency_ms: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Full audit event
# ---------------------------------------------------------------------------


class AuditEvent(BaseModel):
    """
    Complete audit record for a single user interaction.

    Combines input validation, output filter, and policy decision into a
    single correlated record. This is the primary unit of audit data.
    """

    # Identifiers
    event_id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="Unique identifier for this audit event.",
    )
    request_id: str = Field(description="Request identifier shared across all events for this turn.")
    session_id: Optional[str] = Field(None, description="Session identifier (optional).")
    user_id: Optional[str] = Field(None, description="User identifier (consider pseudonymization).")

    # Decisions
    input_decision: DecisionType
    output_decision: DecisionType
    policy_action: Optional[PolicyActionType] = None
    policy_name: Optional[str] = None
    policy_version: Optional[str] = None

    # Scores (only included when log_risk_scores is enabled)
    input_risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    output_risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    input_risk_flags: list[str] = Field(default_factory=list)
    output_risk_flags: list[str] = Field(default_factory=list)

    # Timing
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Extensible metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata for application-specific context.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "event_id": "a1b2c3d4e5f6",
            "request_id": "req_abc123def456",
            "session_id": "sess_xyz",
            "user_id": None,
            "input_decision": "allow",
            "output_decision": "pass_redacted",
            "policy_action": "allow",
            "policy_name": "default",
            "policy_version": "1.0",
            "input_risk_score": 0.0,
            "output_risk_score": 0.2,
            "input_risk_flags": [],
            "output_risk_flags": ["email_in_output"],
            "total_latency_ms": 312.5,
        }
    }}
