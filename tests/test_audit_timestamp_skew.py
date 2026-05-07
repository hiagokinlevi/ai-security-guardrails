from datetime import datetime, timedelta, timezone

import pytest

from guardrails.audit import AuditTimestampSkewError, create_audit_event


class _Policy:
    def __init__(self, settings):
        self.settings = settings


def test_audit_timestamp_skew_within_window_passes_and_records_skew():
    now = datetime.now(timezone.utc)
    policy = _Policy(
        {
            "audit_timestamp_skew_window_seconds": 120,
            "audit_timestamp_skew_action": "block",
        }
    )

    event = {
        "timestamp": now - timedelta(seconds=30),
        "decision_payload": {},
    }

    out = create_audit_event(event, policy)
    assert out["decision_payload"]["timestamp_skew_exceeded"] is False
    assert out["decision_payload"]["timestamp_skew_seconds"] == pytest.approx(30, abs=2)


def test_audit_timestamp_skew_exceeds_window_warn_mode_does_not_raise():
    now = datetime.now(timezone.utc)
    policy = _Policy(
        {
            "audit_timestamp_skew_window_seconds": 120,
            "audit_timestamp_skew_action": "warn",
        }
    )

    event = {
        "timestamp": now + timedelta(seconds=300),
        "decision_payload": {},
    }

    out = create_audit_event(event, policy)
    assert out["decision_payload"]["timestamp_skew_exceeded"] is True
    assert out["decision_payload"]["timestamp_skew_action"] == "warn"
    assert out["decision_payload"]["timestamp_skew_seconds"] == pytest.approx(300, abs=2)


def test_audit_timestamp_skew_exceeds_window_block_mode_raises():
    now = datetime.now(timezone.utc)
    policy = _Policy(
        {
            "audit_timestamp_skew_window_seconds": 120,
            "audit_timestamp_skew_action": "block",
        }
    )

    event = {
        "timestamp": now - timedelta(seconds=1000),
        "decision_payload": {},
    }

    with pytest.raises(AuditTimestampSkewError):
        create_audit_event(event, policy)
