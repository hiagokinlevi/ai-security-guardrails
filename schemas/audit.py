from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AuditEventType(str, Enum):
    INPUT_VALIDATION = "input_validation"
    OUTPUT_VALIDATION = "output_validation"
    POLICY_DECISION = "policy_decision"
    STARTUP_SECURITY_CHECKS_PASSED = "startup_security_checks_passed"


class StartupSecurityChecksPassedEvent(BaseModel):
    event_type: AuditEventType = Field(default=AuditEventType.STARTUP_SECURITY_CHECKS_PASSED)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    check_names: list[str]
    policy_path: str
    policy_sha256: str
    process_uid: int
    process_gid: int
    metadata: dict[str, Any] = Field(default_factory=dict)
