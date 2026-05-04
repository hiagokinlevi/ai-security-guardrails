from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from schemas.audit import StartupSecurityChecksPassedEvent


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _emit_audit_event(event: StartupSecurityChecksPassedEvent) -> None:
    # Immutable structured output to stdout/stderr log collector.
    print(json.dumps(event.model_dump(mode="json", exclude_none=True), separators=(",", ":"), sort_keys=True))


def validate_startup_security(policy_path: str, checks_run: list[str], *, metadata: dict[str, Any] | None = None) -> None:
    """
    Called after all startup fail-closed checks have passed.
    Emits a single structured audit event proving checks were executed successfully.
    """
    path = Path(policy_path)
    policy_sha256 = _sha256_file(path)

    uid = os.getuid() if hasattr(os, "getuid") else -1
    gid = os.getgid() if hasattr(os, "getgid") else -1

    event = StartupSecurityChecksPassedEvent(
        check_names=checks_run,
        policy_path=str(path),
        policy_sha256=policy_sha256,
        process_uid=uid,
        process_gid=gid,
        metadata=metadata or {},
    )
    _emit_audit_event(event)
