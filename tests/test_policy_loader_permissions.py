from __future__ import annotations

import os
from pathlib import Path

import pytest

from guardrails.policy_loader import CONFIG_SECURITY_EXIT_CODE, load_policy


def test_policy_permissions_reject_and_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text("rules: []\n", encoding="utf-8")

    # Insecure mode: group writable
    os.chmod(policy, 0o664)

    monkeypatch.delenv("GUARDRAILS_ALLOW_INSECURE_POLICY_PERMS", raising=False)
    with pytest.raises(SystemExit) as exc:
        load_policy(policy)
    assert exc.value.code == CONFIG_SECURITY_EXIT_CODE

    # Explicit non-prod override should allow startup/load
    monkeypatch.setenv("GUARDRAILS_ALLOW_INSECURE_POLICY_PERMS", "true")
    loaded = load_policy(policy)
    assert isinstance(loaded, dict)
    assert loaded.get("rules") == []
