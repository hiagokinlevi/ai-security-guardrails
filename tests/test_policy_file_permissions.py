import os
from pathlib import Path

import pytest

from security.policy_loader import (
    POLICY_PERMISSIONS_OVERRIDE_ENV,
    PolicySecurityError,
    validate_policy_file_permissions,
)


@pytest.mark.parametrize("mode", [0o600, 0o640])
def test_policy_permissions_allowed(tmp_path: Path, mode: int) -> None:
    p = tmp_path / "policy.yaml"
    p.write_text("rules: []\n", encoding="utf-8")
    os.chmod(p, mode)

    validate_policy_file_permissions(p)


def test_policy_permissions_rejected_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(POLICY_PERMISSIONS_OVERRIDE_ENV, raising=False)

    p = tmp_path / "policy.yaml"
    p.write_text("rules: []\n", encoding="utf-8")
    os.chmod(p, 0o644)

    with pytest.raises(PolicySecurityError) as exc:
        validate_policy_file_permissions(p)

    msg = str(exc.value)
    assert "0o644" in msg
    assert POLICY_PERMISSIONS_OVERRIDE_ENV in msg


def test_policy_permissions_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(POLICY_PERMISSIONS_OVERRIDE_ENV, "1")

    p = tmp_path / "policy.yaml"
    p.write_text("rules: []\n", encoding="utf-8")
    os.chmod(p, 0o666)

    validate_policy_file_permissions(p)
