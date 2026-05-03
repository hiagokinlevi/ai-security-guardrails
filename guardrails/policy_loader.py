from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from guardrails.audit import audit_event
from guardrails.errors import StartupConfigError


REASON_CODE_POLICY_PATH_SYMLINK = "startup_config_policy_path_symlink"


def _contains_symlink_in_parents(path: Path) -> bool:
    """Return True if any parent component of path is a symlink."""
    # Resolve piece-by-piece from cwd/anchor to avoid silently resolving links.
    parts = path.parts
    if not parts:
        return False

    current = Path(parts[0]) if path.is_absolute() else Path(parts[0])
    for part in parts[1:-1]:
        current = current / part
        if current.is_symlink():
            return True
    return False


def validate_policy_path_startup(policy_path: str, *, check_parent_symlinks: bool = True) -> None:
    """Fail-closed startup validation for policy path hardening.

    Reject symlinked policy files (and optionally symlinked parent dirs) to
    prevent policy substitution attacks.
    """
    p = Path(policy_path)

    direct_symlink = p.is_symlink()
    parent_symlink = check_parent_symlinks and _contains_symlink_in_parents(p)

    if direct_symlink or parent_symlink:
        detail: dict[str, Any] = {
            "reason_code": REASON_CODE_POLICY_PATH_SYMLINK,
            "policy_path": str(p),
            "direct_symlink": direct_symlink,
            "parent_symlink": parent_symlink,
        }
        audit_event("startup_validation_failed", detail)
        raise StartupConfigError(
            f"Startup validation failed: policy path must not use symlinks ({p})"
        )


def load_policy(policy_path: str) -> dict[str, Any]:
    # existing startup validation path hook
    validate_policy_path_startup(policy_path)

    # existing logic (kept minimal):
    import yaml

    with open(policy_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
