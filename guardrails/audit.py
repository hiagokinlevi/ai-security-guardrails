from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class AuditLogger:
    """Structured audit logger with timestamp integrity checks.

    Guard behavior:
    - Ensures emitted event timestamps are timezone-aware UTC.
    - Ensures non-decreasing timestamps within this process.
    - On violation, emits an `audit.timestamp_integrity_warning` event and
      marks the affected event with `timestamp_integrity=False`.
    """

    _last_event_ts: Optional[datetime] = field(default=None, init=False)

    def _is_aware_utc(self, ts: datetime) -> bool:
        return ts.tzinfo is not None and ts.utcoffset() == timezone.utc.utcoffset(ts)

    def _normalize_or_now(self, ts: Optional[datetime]) -> datetime:
        if ts is None:
            return datetime.now(timezone.utc)
        return ts

    def _emit(self, event: Dict[str, Any]) -> Dict[str, Any]:
        # Replace with the repository's sink/handler integration if present.
        return event

    def log_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None, *, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        payload = dict(payload or {})
        ts = self._normalize_or_now(timestamp)

        integrity_ok = True
        violation_reasons = []

        if not self._is_aware_utc(ts):
            integrity_ok = False
            violation_reasons.append("timestamp_not_aware_utc")

        if self._last_event_ts is not None and ts < self._last_event_ts:
            integrity_ok = False
            violation_reasons.append("timestamp_not_monotonic")

        payload["timestamp_integrity"] = integrity_ok

        event = {
            "event_type": event_type,
            "timestamp": ts.isoformat(),
            "payload": payload,
        }

        if not integrity_ok:
            warning = {
                "event_type": "audit.timestamp_integrity_warning",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {
                    "timestamp_integrity": False,
                    "violations": violation_reasons,
                    "affected_event_type": event_type,
                    "affected_event_timestamp": ts.isoformat(),
                },
            }
            self._emit(warning)

        # Update monotonic reference only with UTC-aware timestamps to avoid
        # poisoning state with invalid values.
        if self._is_aware_utc(ts):
            if self._last_event_ts is None or ts >= self._last_event_ts:
                self._last_event_ts = ts

        return self._emit(event)
