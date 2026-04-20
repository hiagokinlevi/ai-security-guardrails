from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


_EVENT_TYPE_PROMPT_INJECTION = "prompt_injection_detected"
_EVENT_TYPE_TOOL_MISUSE = "tool_misuse_detected"
_EVENT_TYPE_PII_REDACTED = "pii_redacted"


@dataclass(frozen=True)
class SecurityEventLoggerConfig:
    """Configuration for :class:`SecurityEventLogger`.

    Attributes:
        log_file: Path to JSONL log file. Each event is one JSON object per line.
        ensure_parent_dir: Whether to create parent directories automatically.
    """

    log_file: Path = Path("security_events.jsonl")
    ensure_parent_dir: bool = True


class SecurityEventLogger:
    """Structured JSONL logger for security guardrail violations.

    This utility writes one compact JSON object per line to support easy ingestion by
    log processors and SIEM pipelines.
    """

    def __init__(self, config: SecurityEventLoggerConfig | None = None) -> None:
        self._config = config or SecurityEventLoggerConfig(
            log_file=Path(os.getenv("SECURITY_LOG_FILE", "security_events.jsonl"))
        )
        self._lock = threading.Lock()

        if self._config.ensure_parent_dir:
            self._config.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event_type: str,
        message: str,
        *,
        severity: str = "warning",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Log a generic security event.

        Returns the event payload for optional downstream use.
        """
        event: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "severity": severity,
            "message": message,
            "metadata": dict(metadata or {}),
        }
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))

        with self._lock:
            with self._config.log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        return event

    def log_prompt_injection_detected(
        self,
        *,
        message: str = "Prompt injection pattern detected.",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.log_event(
            _EVENT_TYPE_PROMPT_INJECTION,
            message,
            severity="high",
            metadata=metadata,
        )

    def log_tool_misuse(
        self,
        *,
        message: str = "Tool misuse attempt detected.",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.log_event(
            _EVENT_TYPE_TOOL_MISUSE,
            message,
            severity="high",
            metadata=metadata,
        )

    def log_pii_redacted(
        self,
        *,
        message: str = "PII was detected and redacted.",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.log_event(
            _EVENT_TYPE_PII_REDACTED,
            message,
            severity="info",
            metadata=metadata,
        )


def get_security_logger(log_file: str | Path | None = None) -> SecurityEventLogger:
    """Factory helper for creating a security event logger.

    If ``log_file`` is not provided, ``SECURITY_LOG_FILE`` env var is honored,
    then defaults to ``security_events.jsonl``.
    """
    if log_file is None:
        return SecurityEventLogger()

    return SecurityEventLogger(SecurityEventLoggerConfig(log_file=Path(log_file)))
