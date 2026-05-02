from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import yaml

from guardrails.audit import AuditLogger


SECURITY_STARTUP_ERROR_CODE = 78


def _format_mode(mode: int) -> str:
    return oct(stat.S_IMODE(mode))


def _is_insecure_mode(mode: int) -> tuple[bool, str | None]:
    perm = stat.S_IMODE(mode)
    if perm & stat.S_IWOTH:
        return True, "world_writable"
    if perm & stat.S_IWGRP:
        return True, "group_writable"
    return False, None


def _emit_insecure_permission_event(
    *,
    audit_logger: AuditLogger | None,
    path: Path,
    mode: int,
    reason: str,
    allow_override: bool,
    target: str,
) -> None:
    if not audit_logger:
        return
    audit_logger.log_event(
        {
            "event_type": "startup.policy_permission_check",
            "status": "rejected" if not allow_override else "allowed_with_override",
            "path": str(path),
            "mode": _format_mode(mode),
            "reason": reason,
            "target": target,
            "allow_insecure_policy_permissions": allow_override,
        }
    )


def _check_secure_permissions(
    policy_path: Path,
    *,
    strict: bool = True,
    allow_insecure_policy_permissions: bool = False,
    check_parent_directory: bool = True,
    audit_logger: AuditLogger | None = None,
) -> None:
    if not strict:
        return

    st = policy_path.stat()
    insecure, reason = _is_insecure_mode(st.st_mode)
    if insecure and reason:
        _emit_insecure_permission_event(
            audit_logger=audit_logger,
            path=policy_path,
            mode=st.st_mode,
            reason=reason,
            allow_override=allow_insecure_policy_permissions,
            target="policy_file",
        )
        if not allow_insecure_policy_permissions:
            raise SystemExit(SECURITY_STARTUP_ERROR_CODE)

    if check_parent_directory:
        parent = policy_path.parent
        pst = parent.stat()
        pinsecure, preason = _is_insecure_mode(pst.st_mode)
        if pinsecure and preason:
            _emit_insecure_permission_event(
                audit_logger=audit_logger,
                path=parent,
                mode=pst.st_mode,
                reason=preason,
                allow_override=allow_insecure_policy_permissions,
                target="parent_directory",
            )
            if not allow_insecure_policy_permissions:
                raise SystemExit(SECURITY_STARTUP_ERROR_CODE)


def load_policy(
    policy_path: str | os.PathLike[str],
    *,
    strict_permission_check: bool = True,
    allow_insecure_policy_permissions: bool = False,
    check_parent_directory_permissions: bool = True,
    audit_logger: AuditLogger | None = None,
) -> dict[str, Any]:
    path = Path(policy_path)
    _check_secure_permissions(
        path,
        strict=strict_permission_check,
        allow_insecure_policy_permissions=allow_insecure_policy_permissions,
        check_parent_directory=check_parent_directory_permissions,
        audit_logger=audit_logger,
    )

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("Policy must be a YAML mapping")
        return data
