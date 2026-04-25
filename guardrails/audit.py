from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: str
    action: str
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    policy_digest: Optional[str] = None


def canonical_policy_digest(policy: Dict[str, Any]) -> str:
    canonical = json.dumps(policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_audit_event(
    *,
    event_type: str,
    action: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
    policy_digest: Optional[str] = None,
) -> AuditEvent:
    return AuditEvent(
        event_type=event_type,
        action=action,
        message=message,
        metadata=metadata or {},
        policy_digest=policy_digest,
    )
