from pathlib import Path

import pytest

from guardrails.policy_loader import (
    AUDIT_EVENT_POLICY_DEFAULT_MISSING,
    PolicyBootstrapError,
    load_policy,
)


def test_load_policy_fails_when_default_policy_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.yaml"

    with pytest.raises(PolicyBootstrapError) as exc_info:
        load_policy(missing)

    message = str(exc_info.value)
    assert AUDIT_EVENT_POLICY_DEFAULT_MISSING in message
    assert "missing or invalid" in message.lower()
