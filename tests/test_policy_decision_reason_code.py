import pytest
from pydantic import ValidationError

from schemas.audit import PolicyDecisionEvent


@pytest.mark.parametrize(
    "code",
    [
        "prompt_injection_detected",
        "pii_redacted",
        "token_budget_exceeded",
        "regex_rule_match",
        "tool_depth_exceeded",
        "content_type_blocked",
    ],
)
def test_policy_decision_reason_code_accepts_only_canonical_values(code: str) -> None:
    event = PolicyDecisionEvent(decision="warn", policy_decision_reason_code=code)
    assert event.policy_decision_reason_code.value == code


def test_policy_decision_reason_code_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError):
        PolicyDecisionEvent(decision="block", policy_decision_reason_code="some_new_reason")
