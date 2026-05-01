from __future__ import annotations

import pytest

from guardrails.policy_loader import POLICY_PARSE_EXIT_CODE, load_policy_or_exit


def test_malformed_policy_fails_closed_with_exit_code_78(tmp_path):
    policy_file = tmp_path / "bad_policy.yaml"
    policy_file.write_text("rules:\n  - action: explode\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        load_policy_or_exit(policy_file)

    assert exc.value.code == POLICY_PARSE_EXIT_CODE
