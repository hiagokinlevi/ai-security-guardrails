from __future__ import annotations

from pathlib import Path

import pytest

from guardrails import policy_loader


def _write_policy(tmp_path: Path) -> Path:
    p = tmp_path / "policy.yaml"
    p.write_text("mode: enforce\n", encoding="utf-8")
    return p


def test_policy_owner_match_mismatch_and_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    policy_path = _write_policy(tmp_path)

    # Match case: should load.
    monkeypatch.setattr(policy_loader.os, "getuid", lambda: 1000)
    monkeypatch.setattr(policy_loader.Path, "stat", lambda self: type("S", (), {"st_uid": 1000})())
    loaded = policy_loader.load_policy(str(policy_path))
    assert loaded["mode"] == "enforce"

    # Mismatch case without override: should fail closed.
    monkeypatch.setattr(policy_loader.Path, "stat", lambda self: type("S", (), {"st_uid": 2000})())
    monkeypatch.delenv("GUARDRAILS_ALLOW_POLICY_OWNER_MISMATCH", raising=False)
    with pytest.raises(policy_loader.PolicyLoadError):
        policy_loader.load_policy(str(policy_path))

    # Mismatch with explicit override: should load.
    monkeypatch.setenv("GUARDRAILS_ALLOW_POLICY_OWNER_MISMATCH", "true")
    loaded_override = policy_loader.load_policy(str(policy_path))
    assert loaded_override["mode"] == "enforce"
