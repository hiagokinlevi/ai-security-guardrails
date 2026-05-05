import hashlib
import json
from typing import Any, Dict, List

import pytest


def _canonical_event_payload(event: Dict[str, Any]) -> str:
    payload = {k: v for k, v in event.items() if k not in {"event_hash", "prev_event_hash"}}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _compute_event_hash(event: Dict[str, Any], prev_hash: str) -> str:
    canonical = _canonical_event_payload(event)
    data = f"{prev_hash}:{canonical}".encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _genesis_hash(process_id: str, session_id: str) -> str:
    seed = f"audit-genesis:{process_id}:{session_id}".encode("utf-8")
    return hashlib.sha256(seed).hexdigest()


def _build_chain(events: List[Dict[str, Any]], process_id: str = "p1", session_id: str = "s1") -> List[Dict[str, Any]]:
    prev = _genesis_hash(process_id, session_id)
    out = []
    for e in events:
        evt = dict(e)
        evt["prev_event_hash"] = prev
        evt["event_hash"] = _compute_event_hash(evt, prev)
        prev = evt["event_hash"]
        out.append(evt)
    return out


def _verify_chain(events: List[Dict[str, Any]], process_id: str = "p1", session_id: str = "s1") -> bool:
    expected_prev = _genesis_hash(process_id, session_id)
    for idx, event in enumerate(events):
        if event.get("prev_event_hash") != expected_prev:
            raise ValueError(f"broken link at index {idx}")
        expected_hash = _compute_event_hash(event, expected_prev)
        if event.get("event_hash") != expected_hash:
            raise ValueError(f"tamper detected at index {idx}")
        expected_prev = event["event_hash"]
    return True


def test_valid_chain_passes_verification():
    events = _build_chain(
        [
            {"type": "input_scan", "risk": 0.1, "msg": "ok"},
            {"type": "policy_decision", "decision": "allow"},
            {"type": "output_filter", "redactions": 0},
        ]
    )
    assert _verify_chain(events) is True


def test_tampered_event_fails_verification():
    events = _build_chain(
        [
            {"type": "input_scan", "risk": 0.1, "msg": "ok"},
            {"type": "policy_decision", "decision": "allow"},
        ]
    )
    events[1]["decision"] = "block"  # tamper payload after hashing
    with pytest.raises(ValueError, match="tamper detected"):
        _verify_chain(events)


def test_truncated_log_detects_broken_continuity():
    events = _build_chain(
        [
            {"type": "a", "n": 1},
            {"type": "b", "n": 2},
            {"type": "c", "n": 3},
        ]
    )
    truncated = events[1:]  # missing first link/genesis continuity
    with pytest.raises(ValueError, match="broken link"):
        _verify_chain(truncated)
