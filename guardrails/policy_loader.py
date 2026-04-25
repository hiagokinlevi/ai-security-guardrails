from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class PolicyBootstrapError(RuntimeError):
    """Raised when policy bootstrap fails in a deny-by-default manner."""


AUDIT_EVENT_POLICY_DEFAULT_MISSING = "POLICY_DEFAULT_MISSING"


def load_policy(policy_path: str | Path) -> dict[str, Any]:
    """Load a YAML policy file.

    Startup is deny-by-default: if the configured policy file does not exist or
    is unreadable, initialization must fail explicitly.
    """
    path = Path(policy_path)

    if not path.exists() or not path.is_file():
        raise PolicyBootstrapError(
            f"[{AUDIT_EVENT_POLICY_DEFAULT_MISSING}] Default policy file is missing or invalid: {path}"
        )

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyBootstrapError(
            f"[{AUDIT_EVENT_POLICY_DEFAULT_MISSING}] Default policy file is unreadable: {path}"
        ) from exc

    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise PolicyBootstrapError(f"Invalid policy YAML: {path}") from exc

    if not isinstance(data, dict):
        raise PolicyBootstrapError(f"Policy root must be a mapping: {path}")

    return data
