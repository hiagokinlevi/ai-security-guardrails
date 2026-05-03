from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import yaml


CONFIG_SECURITY_EXIT_CODE = 78
_OVERRIDE_ENV = "GUARDRAILS_ALLOW_INSECURE_POLICY_PERMS"


def _emit_audit(event: dict[str, Any]) -> None:
    # Structured audit event to stderr for startup/config security failures.
    sys.stderr.write(json.dumps(event, separators=(",", ":")) + "\n")


def _is_override_enabled() -> bool:
    raw = os.getenv(_OVERRIDE_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _validate_policy_file_permissions(policy_path: Path) -> None:
    mode = policy_path.stat().st_mode
    insecure_mask = stat.S_IWGRP | stat.S_IWOTH
    insecure_bits = mode & insecure_mask

    if insecure_bits and not _is_override_enabled():
        event = {
            "event": "policy_file_permission_rejected",
            "path": str(policy_path),
            "mode_octal": oct(stat.S_IMODE(mode)),
            "reason": "policy file is group-writable and/or world-writable",
            "override_env": _OVERRIDE_ENV,
        }
        _emit_audit(event)
        raise SystemExit(CONFIG_SECURITY_EXIT_CODE)


def load_policy(path: str | os.PathLike[str]) -> dict[str, Any]:
    policy_path = Path(path)
    _validate_policy_file_permissions(policy_path)

    with policy_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Policy file must contain a top-level mapping/object")

    return data
