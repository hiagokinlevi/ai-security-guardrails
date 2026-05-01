from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _extract_sequence_id(event: dict[str, Any]) -> int | None:
    """Return sequence_id (or equivalent) if present and integer-like."""
    for key in ("sequence_id", "sequence", "index"):
        if key in event:
            value = event.get(key)
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
            return None
    return None


def verify_audit_events(events: list[dict[str, Any]]) -> tuple[bool, str]:
    """Verify audit event sequence monotonicity.

    Requires strictly increasing sequence IDs with no gaps.
    """
    if not events:
        return True, "ok"

    prev_seq: int | None = None
    for idx, event in enumerate(events, start=1):
        seq = _extract_sequence_id(event)
        if seq is None:
            return False, f"event {idx} missing valid sequence_id"

        if prev_seq is None:
            prev_seq = seq
            continue

        if seq <= prev_seq:
            return False, (
                f"sequence_id out of order or duplicate at event {idx}: "
                f"{seq} after {prev_seq}"
            )

        if seq != prev_seq + 1:
            return False, (
                f"sequence_id gap at event {idx}: expected {prev_seq + 1}, got {seq}"
            )

        prev_seq = seq

    return True, "ok"


def verify_audit_file(path: str | Path) -> tuple[bool, str]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "events" in data:
        events = data["events"]
    else:
        events = data

    if not isinstance(events, list):
        return False, "audit payload must be a list of events or contain an 'events' list"

    if not all(isinstance(e, dict) for e in events):
        return False, "all audit events must be objects"

    return verify_audit_events(events)
