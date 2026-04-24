from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Tuple


def canonicalize_event_payload(event: dict[str, Any]) -> str:
    """Return canonical JSON for hashing (stable keys, compact separators)."""
    return json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_audit_event_hash(prev_hash: str, event_without_hash: dict[str, Any]) -> str:
    """Compute hash-chain link for an audit event."""
    material = f"{prev_hash}|{canonicalize_event_payload(event_without_hash)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def verify_audit_chain_events(events: Iterable[dict[str, Any]]) -> Tuple[bool, int | None, str | None]:
    """
    Verify a sequence of newline-delimited audit events.

    Returns:
        (ok, bad_index, reason)
        - ok=True if all links validate.
        - bad_index is zero-based index of first mismatch/invalid event.
    """
    prev_hash = "GENESIS"

    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            return False, idx, "event is not an object"

        stored_hash = event.get("hash")
        if not isinstance(stored_hash, str) or not stored_hash:
            return False, idx, "missing or invalid hash"

        event_wo_hash = dict(event)
        event_wo_hash.pop("hash", None)

        expected = compute_audit_event_hash(prev_hash, event_wo_hash)
        if expected != stored_hash:
            return False, idx, "hash mismatch"

        prev_hash = stored_hash

    return True, None, None


def verify_audit_chain_file(path: str | Path) -> Tuple[bool, int | None, str | None]:
    """Verify newline-delimited JSON audit log file integrity."""
    p = Path(path)
    events: list[dict[str, Any]] = []

    with p.open("r", encoding="utf-8") as f:
        for idx, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                return False, idx, "invalid JSON"
            events.append(parsed)

    return verify_audit_chain_events(events)
