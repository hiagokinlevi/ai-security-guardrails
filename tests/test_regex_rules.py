from pathlib import Path

import pytest

from guardrails.input_controls.validator import InputDecision, validate_input
from guardrails.policy_engine.regex_rules import load_regex_rules


def _write_rule_set(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(content)
    return path


def test_yaml_regex_rule_set_adds_custom_flag_and_score(tmp_path: Path) -> None:
    path = _write_rule_set(
        tmp_path,
        """
rules:
  - id: internal-secret-marker
    flag: internal_secret_marker
    category: sensitive_data
    pattern: "\\\\bINTERNAL_SECRET_[A-Z0-9]{8,}\\\\b"
    score: 0.3
""",
    )

    result = validate_input(
        "Rotate INTERNAL_SECRET_ABCD1234 before invoking the model.",
        regex_rule_set_path=path,
    )

    assert result.decision == InputDecision.ALLOW_WITH_WARNING
    assert result.risk_score == pytest.approx(0.3)
    assert "internal_secret_marker" in result.risk_flags


def test_preloaded_yaml_regex_rules_can_be_reused(tmp_path: Path) -> None:
    path = _write_rule_set(
        tmp_path,
        """
rules:
  - id: tenant-prompt-marker
    flag: tenant_prompt_marker
    category: injection
    pattern: "TENANT_OVERRIDE_MODE"
""",
    )
    rules = load_regex_rules(path)

    result = validate_input("TENANT_OVERRIDE_MODE", regex_rules=rules)

    assert result.risk_score == pytest.approx(0.35)
    assert "tenant_prompt_marker" in result.risk_flags


def test_invalid_yaml_regex_rule_pattern_raises_value_error(tmp_path: Path) -> None:
    path = _write_rule_set(
        tmp_path,
        """
rules:
  - id: bad-pattern
    flag: bad_pattern
    pattern: "["
""",
    )

    with pytest.raises(ValueError, match="invalid pattern"):
        load_regex_rules(path)


def test_yaml_regex_rule_set_too_large_is_rejected(tmp_path: Path) -> None:
    # Ensure the size check happens before YAML parsing.
    path = _write_rule_set(
        tmp_path,
        "rules:\n  - id: big\n    flag: big\n    pattern: \"" + ("a" * 5000) + "\"\n",
    )

    with pytest.raises(ValueError, match="too large"):
        load_regex_rules(path, max_bytes=100)


def test_yaml_regex_rule_pattern_too_long_is_rejected(tmp_path: Path) -> None:
    path = _write_rule_set(
        tmp_path,
        """
rules:
  - id: too-long-pattern
    flag: too_long_pattern
    pattern: "REPLACE_ME"
""",
    )
    content = path.read_text()
    path.write_text(content.replace("REPLACE_ME", "a" * 3000))

    with pytest.raises(ValueError, match="pattern is too long"):
        load_regex_rules(path, max_pattern_length=1000)
