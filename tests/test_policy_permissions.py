from __future__ import annotations

import os
from pathlib import Path

import pytest

from guardrails.policy import SECURITY_STARTUP_ERROR_CODE, load_policy


class DummyAuditLogger:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def log_event(self, event: dict) -> None:
        self.events.append(event)


def _write_policy(path: Path) -> None:
    path.write_text("action: allow\n", encoding="utf-8")


def test_rejects_world_writable_policy_file(tmp_path: Path) -> None:
    policy = tmp_path / "policy.yaml"
    _write_policy(policy)
    os.chmod(policy, 0o666)

    audit = DummyAuditLogger()
    with pytest.raises(SystemExit) as exc:
        load_policy(policy, audit_logger=audit)

    assert exc.value.code == SECURITY_STARTUP_ERROR_CODE
    assert audit.events
    assert audit.events[-1]["event_type"] == "startup.policy_permission_check"
    assert audit.events[-1]["target"] == "policy_file"


def test_allows_insecure_permissions_with_override(tmp_path: Path) -> None:
    policy = tmp_path / "policy.yaml"
    _write_policy(policy)
    os.chmod(policy, 0o666)

    audit = DummyAuditLogger()
    loaded = load_policy(
        policy,
        allow_insecure_policy_permissions=True,
        audit_logger=audit,
    )

    assert loaded["action"] == "allow"
    assert audit.events
    assert audit.events[-1]["status"] == "allowed_with_override"


def test_rejects_group_writable_parent_directory(tmp_path: Path) -> None:
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    policy = policy_dir / "policy.yaml"
    _write_policy(policy)

    os.chmod(policy, 0o644)
    os.chmod(policy_dir, 0o775)

    with pytest.raises(SystemExit) as exc:
        load_policy(policy)

    assert exc.value.code == SECURITY_STARTUP_ERROR_CODE
