from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditPathCheckResult:
    path_checked: str
    target_kind: str
    exists: bool
    owned_by_effective_user: bool
    group_writable: bool
    other_writable: bool
    passed: bool
    override_used: bool
    effective_uid: int
    effective_gid: int


def _emit_startup_audit_event(result: AuditPathCheckResult) -> None:
    event = {
        "event_type": "startup.audit_log_path_hardening",
        "path_checked": result.path_checked,
        "target_kind": result.target_kind,
        "exists": result.exists,
        "owned_by_effective_user": result.owned_by_effective_user,
        "group_writable": result.group_writable,
        "other_writable": result.other_writable,
        "passed": result.passed,
        "override_used": result.override_used,
        "effective_uid": result.effective_uid,
        "effective_gid": result.effective_gid,
    }
    print(json.dumps(event, sort_keys=True))


def validate_audit_log_path_hardening(config: Any) -> None:
    audit_log_path = Path(str(getattr(config, "audit_log_path")))
    allow_insecure = bool(getattr(config, "allow_insecure_audit_log_path", False))

    # Validate file if already present; otherwise validate parent directory.
    target = audit_log_path if audit_log_path.exists() else audit_log_path.parent
    target_kind = "file" if target == audit_log_path else "directory"

    st = target.stat()
    mode = stat.S_IMODE(st.st_mode)

    effective_uid = os.geteuid()
    effective_gid = os.getegid()

    owned_by_effective_user = st.st_uid == effective_uid
    group_writable = bool(mode & stat.S_IWGRP)
    other_writable = bool(mode & stat.S_IWOTH)

    passed = owned_by_effective_user and not group_writable and not other_writable

    result = AuditPathCheckResult(
        path_checked=str(target),
        target_kind=target_kind,
        exists=target.exists(),
        owned_by_effective_user=owned_by_effective_user,
        group_writable=group_writable,
        other_writable=other_writable,
        passed=passed,
        override_used=(not passed and allow_insecure),
        effective_uid=effective_uid,
        effective_gid=effective_gid,
    )
    _emit_startup_audit_event(result)

    if not passed and not allow_insecure:
        raise RuntimeError(
            "Startup hardening failed for audit log path: ownership/permissions are insecure. "
            "Set allow_insecure_audit_log_path=true only for explicit local/dev override."
        )
