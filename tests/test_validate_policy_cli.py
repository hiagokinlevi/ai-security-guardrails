from pathlib import Path

from cli.main import main


def test_validate_policy_valid_and_invalid(tmp_path: Path):
    valid_policy = tmp_path / "valid_policy.yaml"
    invalid_policy = tmp_path / "invalid_policy.yaml"

    # Adjusted to expected Policy schema shape used in repository.
    valid_policy.write_text(
        """
version: 1
name: test-policy
rules:
  prompt_injection:
    action: block
  pii:
    action: warn
""".strip()
        + "\n",
        encoding="utf-8",
    )

    # invalid: wrong type for version and malformed rules value
    invalid_policy.write_text(
        """
version: wrong
name: bad-policy
rules: []
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert main(["validate-policy", "--policy", str(valid_policy)]) == 0
    assert main(["validate-policy", "--policy", str(invalid_policy)]) != 0
