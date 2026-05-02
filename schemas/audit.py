from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class PolicyDecisionReasonCode(str, Enum):
    prompt_injection_detected = "prompt_injection_detected"
    pii_redacted = "pii_redacted"
    token_budget_exceeded = "token_budget_exceeded"
    regex_rule_match = "regex_rule_match"
    tool_depth_exceeded = "tool_depth_exceeded"
    content_type_blocked = "content_type_blocked"


class PolicyDecisionEvent(BaseModel):
    decision: Literal["allow", "warn", "block"]
    policy_decision_reason_code: PolicyDecisionReasonCode = Field(
        ..., description="Canonical deterministic reason code for policy decision"
    )
