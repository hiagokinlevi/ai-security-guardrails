from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _canonicalize_event(event: dict[str, Any]) -> str:
    """Return deterministic JSON for hashing."""
    return json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class SecurityEventLogger:
    """Append-only JSONL security logger with hash-chained integrity fields."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _last_event_hash(self) -> str:
        if not self.log_path.exists():
            return "0" * 64

        last_hash = "0" * 64
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                last_hash = obj.get("event_hash", "0" * 64)
        return last_hash

    def append_event(self, event_type: str, payload: dict[str, Any], **extra: Any) -> dict[str, Any]:
        base_event: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
            **extra,
        }

        prev_hash = self._last_event_hash()
        canonical = _canonicalize_event(base_event)
        event_hash = _sha256_hex(canonical + prev_hash)

        entry = {
            **base_event,
            "prev_hash": prev_hash,
            "event_hash": event_hash,
        }

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return entry


def verify_log_chain(log_path: str | Path) -> tuple[bool, int | None]:
    """
    Verify full chain integrity.
    Returns (is_valid, first_broken_line_number_1_indexed_or_None).
    """
    path = Path(log_path)
    if not path.exists():
        return True, None

    expected_prev = "0" * 64
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            prev_hash = entry.get("prev_hash")
            event_hash = entry.get("event_hash")

            if prev_hash != expected_prev:
                return False, idx

            to_hash = dict(entry)
            to_hash.pop("prev_hash", None)
            to_hash.pop("event_hash", None)

            canonical = _canonicalize_event(to_hash)
            recomputed = _sha256_hex(canonical + prev_hash)
            if recomputed != event_hash:
                return False, idx

            expected_prev = event_hash

    return True, None
