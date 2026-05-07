from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from guardrails.policy import PolicyEngine


class AuditTimestampSkewError(ValueError):
    """Raised when audit event timestamp skew exceeds policy in block mode."""


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def validate_audit_timestamp_skew(
    event_timestamp: datetime,
    *,
    policy: PolicyEngine,
    decision_payload: Dict[str, Any],
    now: datetime | None = None,
) -> None:
    """
    Validate audit event timestamp skew at ingestion time.

    Policy settings used:
      - audit_timestamp_skew_window_seconds (int, default: 120)
      - audit_timestamp_skew_action ("warn" | "block", default: "warn")

    Enriches `decision_payload` with:
      - timestamp_skew_seconds (float)
      - timestamp_skew_window_seconds (int)
      - timestamp_skew_action (str)
      - timestamp_skew_exceeded (bool)
    """
    settings = getattr(policy, "settings", {}) or {}
    window = int(settings.get("audit_timestamp_skew_window_seconds", 120))
    action = str(settings.get("audit_timestamp_skew_action", "warn")).lower().strip()
    if action not in {"warn", "block"}:
        action = "warn"

    current = _as_utc(now or datetime.now(timezone.utc))
    ts = _as_utc(event_timestamp)

    skew_seconds = abs((ts - current).total_seconds())
    exceeded = skew_seconds > window

    decision_payload["timestamp_skew_seconds"] = skew_seconds
    decision_payload["timestamp_skew_window_seconds"] = window
    decision_payload["timestamp_skew_action"] = action
    decision_payload["timestamp_skew_exceeded"] = exceeded

    if exceeded and action == "block":
        raise AuditTimestampSkewError(
            f"audit timestamp skew {skew_seconds:.3f}s exceeds allowed window ±{window}s"
        )


# Existing audit event creation flow uses this branch at ingestion time.
def create_audit_event(event: Dict[str, Any], policy: PolicyEngine) -> Dict[str, Any]:
    payload = dict(event.get("decision_payload") or {})
    ts = event.get("timestamp")
    if not isinstance(ts, datetime):
        raise ValueError("audit event timestamp must be a datetime")

    validate_audit_timestamp_skew(ts, policy=policy, decision_payload=payload)

    out = dict(event)
    out["decision_payload"] = payload
    return out
