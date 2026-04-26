from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from guardrails.audit import AuditLogger
from guardrails.schemas import PolicySchema


class PolicyLoadError(RuntimeError):
    """Raised when policy loading or validation fails."""


def _summarize_error(err: Exception) -> str:
    msg = str(err).strip()
    if len(msg) > 500:
        return msg[:497] + "..."
    return msg


def load_policy_or_fail_closed(policy_path: str | Path, audit: AuditLogger) -> PolicySchema:
    """
    Load and validate policy. On failure, emit structured startup audit event,
    then raise PolicyLoadError for fail-closed termination by caller.
    """
    path = str(policy_path)
    try:
        with open(policy_path, "r", encoding="utf-8") as f:
            raw: Any = yaml.safe_load(f)
        return PolicySchema.model_validate(raw)
    except FileNotFoundError as e:
        audit.log_event(
            "startup_policy_load_failed",
            {
                "reason_code": "POLICY_FILE_NOT_FOUND",
                "policy_path": path,
                "validation_error_summary": _summarize_error(e),
            },
        )
        raise PolicyLoadError(f"Fail-closed: policy file not found at {path}") from e
    except (yaml.YAMLError, ValidationError, ValueError, TypeError) as e:
        audit.log_event(
            "startup_policy_load_failed",
            {
                "reason_code": "POLICY_SCHEMA_VALIDATION_FAILED",
                "policy_path": path,
                "validation_error_summary": _summarize_error(e),
            },
        )
        raise PolicyLoadError("Fail-closed: policy load/validation failed") from e
