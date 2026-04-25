from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


AUDIT_SCHEMA_VERSION = "1"


class AuditEvent(BaseModel):
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str
    payload: Dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class AuditConfig:
    expected_schema_version: str = AUDIT_SCHEMA_VERSION


class AuditSchemaVersionMismatchError(RuntimeError):
    pass


class AuditLogger:
    def __init__(self, *, config: Optional[AuditConfig] = None) -> None:
        self.config = config or AuditConfig()
        self._validate_schema_compatibility()

    def _validate_schema_compatibility(self) -> None:
        if self.config.expected_schema_version != AUDIT_SCHEMA_VERSION:
            raise AuditSchemaVersionMismatchError(
                "Audit schema version mismatch: "
                f"configured={self.config.expected_schema_version} "
                f"runtime={AUDIT_SCHEMA_VERSION}"
            )

    def emit(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> AuditEvent:
        return AuditEvent(
            event_type=event_type,
            payload=payload or {},
            schema_version=AUDIT_SCHEMA_VERSION,
        )
