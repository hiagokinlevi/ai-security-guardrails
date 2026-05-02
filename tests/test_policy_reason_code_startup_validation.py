from __future__ import annotations

import yaml
import pytest

from guardrails.policy_loader import (
    POLICY_LOAD_VALIDATION_EXIT_CODE,
    load_policy,
)


def test_load_policy_exits_on_invalid_reason_code(tmp_path, monkeypatch):
    policy_data = {
        "rules": [
            {
                "name": "bad-rule",
                "policy_decision_reason_code": "NOT_A_CANONICAL_REASON",
            }
        ]
    }
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(policy_data), encoding="utf-8")

    captured = {}

    def _fake_emit(event):
        captured["event"] = event

    monkeypatch.setattr("guardrails.policy_loader.emit_audit_event", _fake_emit)

    with pytest.raises(SystemExit) as exc:
        load_policy(str(policy_path))

    assert exc.value.code == POLICY_LOAD_VALIDATION_EXIT_CODE
    assert captured["event"]["event_type"] == "policy_load_validation"
    assert captured["event"]["reason"] == "invalid_policy_decision_reason_code"
    assert "invalid_reason_code_paths" in captured["event"]["details"]
    assert captured["event"]["details"]["invalid_reason_code_paths"]


def test_load_policy_accepts_valid_reason_code(tmp_path):
    # Use first canonical enum value dynamically to avoid coupling to specific literals.
    from guardrails.schemas.audit import PolicyDecisionReasonCode

    valid_code = next(iter(PolicyDecisionReasonCode)).value

    policy_data = {
        "rules": [
            {
                "name": "good-rule",
                "policy_decision_reason_code": valid_code,
            }
        ]
    }
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(policy_data), encoding="utf-8")

    loaded = load_policy(str(policy_path))
    assert loaded["rules"][0]["policy_decision_reason_code"] == valid_code
