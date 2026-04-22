import pytest

from guardrails.policy_loader import PolicyLoadError, load_policy


def test_policy_loader_raises_on_malformed_yaml(tmp_path):
    policy_file = tmp_path / "bad_policy.yaml"
    policy_file.write_text("rules: [\n  - id: r1\n    action: block\n    pattern: '(abc'", encoding="utf-8")

    with pytest.raises(PolicyLoadError, match="Invalid YAML"):
        load_policy(policy_file)


def test_policy_loader_raises_on_missing_required_rule_keys(tmp_path):
    policy_file = tmp_path / "missing_keys_policy.yaml"
    policy_file.write_text(
        """
rules:
  - id: rule-1
    action: block
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(PolicyLoadError, match="missing required keys: pattern"):
        load_policy(policy_file)
