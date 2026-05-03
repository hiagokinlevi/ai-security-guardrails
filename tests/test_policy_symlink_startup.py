from __future__ import annotations

from pathlib import Path

import pytest

from guardrails.errors import StartupConfigError
from guardrails.policy_loader import (
    REASON_CODE_POLICY_PATH_SYMLINK,
    validate_policy_path_startup,
)


def test_startup_rejects_symlinked_policy_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    real_policy = tmp_path / "policy.yaml"
    real_policy.write_text("version: 1\n", encoding="utf-8")

    link_policy = tmp_path / "policy-link.yaml"
    link_policy.symlink_to(real_policy)

    events = []

    def _capture(event_type, detail):
        events.append((event_type, detail))

    monkeypatch.setattr("guardrails.policy_loader.audit_event", _capture)

    with pytest.raises(StartupConfigError):
        validate_policy_path_startup(str(link_policy))

    assert events, "expected startup failure audit event"
    event_type, detail = events[0]
    assert event_type == "startup_validation_failed"
    assert detail["reason_code"] == REASON_CODE_POLICY_PATH_SYMLINK
    assert detail["direct_symlink"] is True
