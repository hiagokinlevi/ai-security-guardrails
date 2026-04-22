from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from middleware.fastapi_middleware import get_request_id

logger = logging.getLogger("ai_security_guardrails.audit")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit_security_event(event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Emit a structured security/audit event.

    Ensures every event contains request_id for cross-log correlation.
    """
    event: dict[str, Any] = {
        "timestamp": _utc_now_iso(),
        "event_type": event_type,
        "request_id": get_request_id(),
    }
    if payload:
        event.update(payload)

    logger.info(json.dumps(event, separators=(",", ":"), sort_keys=True))
    return event
