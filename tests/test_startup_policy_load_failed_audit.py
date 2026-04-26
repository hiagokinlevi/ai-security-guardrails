from __future__ import annotations

from pathlib import Path

import pytest

from guardrails.policy import PolicyLoadError, load_policy_or_fail_closed


class _AuditSpy:
    def __init__(self) -> None:
        self.events = []

    def log_event(self, event_type, payload):
        self.events.append({"event_type": event_type, "payload": payload})


def test_startup_policy_load_failed_emits_audit_event_before_raise(tmp_path: Path):
    bad_policy = tmp_path / "policy.yaml"
    bad_policy.write_text("rules: [\n", encoding="utf-8")  # malformed YAML

    audit = _AuditSpy()

    with pytest.raises(PolicyLoadError):
        load_policy_or_fail_closed(bad_policy, audit)

    assert len(audit.events) == 1
    event = audit.events[0]
    assert event["event_type"] == "startup_policy_load_failed"

    payload = event["payload"]
    assert payload["reason_code"] == "POLICY_SCHEMA_VALIDATION_FAILED"
    assert payload["policy_path"] == str(bad_policy)
    assert isinstance(payload["validation_error_summary"], str)
    assert payload["validation_error_summary"]
