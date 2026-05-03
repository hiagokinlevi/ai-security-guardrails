from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from guardrails.audit import emit_audit_event


class PolicyLoadError(RuntimeError):
    """Raised when policy loading fails."""


def _env_flag_true(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _policy_owner_override_enabled(config: dict[str, Any] | None = None) -> bool:
    # Guarded override: default false unless explicitly enabled by env/config.
    env_enabled = _env_flag_true(os.getenv("GUARDRAILS_ALLOW_POLICY_OWNER_MISMATCH"))
    if env_enabled:
        return True

    if not config:
        return False

    security_cfg = config.get("security") if isinstance(config, dict) else None
    if isinstance(security_cfg, dict):
        return bool(security_cfg.get("allow_policy_owner_mismatch", False))
    return False


def _validate_policy_owner(policy_path: Path, config: dict[str, Any] | None = None) -> None:
    # On unsupported platforms (e.g., no getuid/stat uid), skip this check.
    if not hasattr(os, "getuid"):
        return

    st = policy_path.stat()
    file_uid = getattr(st, "st_uid", None)
    process_uid = os.getuid()

    if file_uid is None:
        return

    if file_uid != process_uid and not _policy_owner_override_enabled(config):
        emit_audit_event(
            {
                "event_type": "policy_load_failure",
                "reason": "policy_owner_mismatch",
                "policy_path": str(policy_path),
                "policy_uid": file_uid,
                "process_uid": process_uid,
                "override_enabled": False,
            }
        )
        raise PolicyLoadError(
            f"Policy file owner UID ({file_uid}) does not match process UID ({process_uid})"
        )


def load_policy(policy_path: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(policy_path)

    if not path.exists():
        raise PolicyLoadError(f"Policy file not found: {policy_path}")

    _validate_policy_owner(path, config=config)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        raise PolicyLoadError(f"Failed to load policy: {exc}") from exc

    if not isinstance(data, dict):
        raise PolicyLoadError("Policy root must be a mapping/object")

    return data
