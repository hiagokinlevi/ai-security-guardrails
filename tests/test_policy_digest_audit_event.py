from guardrails.audit import build_audit_event, canonical_policy_digest


def test_policy_digest_attached_and_deterministic_for_equivalent_policy_shapes():
    policy_a = {
        "rules": [
            {"id": "prompt_injection", "action": "block", "enabled": True},
            {"id": "pii_redaction", "action": "warn", "enabled": True},
        ],
        "thresholds": {"input_risk": 0.8, "output_risk": 0.7},
    }

    # Same semantic content with different key ordering
    policy_b = {
        "thresholds": {"output_risk": 0.7, "input_risk": 0.8},
        "rules": [
            {"enabled": True, "action": "block", "id": "prompt_injection"},
            {"enabled": True, "id": "pii_redaction", "action": "warn"},
        ],
    }

    digest_a = canonical_policy_digest(policy_a)
    digest_b = canonical_policy_digest(policy_b)

    assert digest_a == digest_b
    assert len(digest_a) == 64

    event = build_audit_event(
        event_type="decision",
        action="block",
        message="Blocked due to prompt injection signature",
        metadata={"source": "middleware"},
        policy_digest=digest_a,
    )

    assert event.policy_digest == digest_a
