from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from guardrails.startup_validation import validate_audit_log_path_hardening


@dataclass
class _Config:
    audit_log_path: str
    allow_insecure_audit_log_path: bool = False


def test_startup_audit_log_path_hardening_passes_for_secure_existing_file(tmp_path: Path) -> None:
    log_file = tmp_path / "audit.log"
    log_file.write_text("ok")
    os.chmod(log_file, 0o600)

    cfg = _Config(audit_log_path=str(log_file), allow_insecure_audit_log_path=False)
    validate_audit_log_path_hardening(cfg)


def test_startup_audit_log_path_hardening_fails_for_group_writable_parent_when_file_missing(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "logs"
    parent.mkdir()
    os.chmod(parent, 0o770)
    log_file = parent / "audit.log"

    cfg = _Config(audit_log_path=str(log_file), allow_insecure_audit_log_path=False)
    with pytest.raises(RuntimeError, match="Startup hardening failed"):
        validate_audit_log_path_hardening(cfg)


def test_startup_audit_log_path_hardening_allows_override(tmp_path: Path) -> None:
    parent = tmp_path / "logs"
    parent.mkdir()
    os.chmod(parent, 0o777)
    log_file = parent / "audit.log"

    cfg = _Config(audit_log_path=str(log_file), allow_insecure_audit_log_path=True)
    validate_audit_log_path_hardening(cfg)
