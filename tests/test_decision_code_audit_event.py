from schemas.security_event import PolicyDecisionCode, SecurityEvent


def test_security_event_requires_stable_decision_code_values():
    blocked = SecurityEvent(
        event_type="input_scan",
        action="block",
        decision_code=PolicyDecisionCode.INPUT_INJECTION_BLOCK,
        reason="prompt injection signature detected",
        metadata={"rule": "ignore_previous_instructions"},
    )
    warned = SecurityEvent(
        event_type="output_filter",
        action="warn",
        decision_code=PolicyDecisionCode.OUTPUT_PII_REDACTED,
        reason="redacted possible SSN",
    )
    allowed = SecurityEvent(
        event_type="token_budget",
        action="allow",
        decision_code=PolicyDecisionCode.TOKEN_BUDGET_EXCEEDED,
        reason="soft threshold reached; request truncated",
    )

    assert blocked.decision_code.value == "INPUT_INJECTION_BLOCK"
    assert warned.decision_code.value == "OUTPUT_PII_REDACTED"
    assert allowed.decision_code.value == "TOKEN_BUDGET_EXCEEDED"
