"""
Tests for guardrails/policy_engine/engine.py
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from guardrails.policy_engine.engine import (
    PolicyAction,
    PolicyConfig,
    PolicyDecision,
    PolicyEngine,
    load_policy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_policy_file(policy_dict: dict, dir_path: str) -> Path:
    """Write a policy dict to a temporary YAML file and return its path."""
    path = Path(dir_path) / "test_policy.yaml"
    with open(path, "w") as f:
        yaml.dump(policy_dict, f)
    return path


DEFAULT_POLICY_DICT = {
    "name": "test",
    "version": "1.0",
    "description": "Test policy",
    "input": {
        "max_length": 5000,
        "risk_threshold": 0.7,
        "on_injection_signal": "send_to_review",
        "on_sensitive_data": "allow_with_warning",
        "on_block": "return_safe_error_message",
    },
    "output": {
        "risk_threshold": 0.8,
        "redact_pii": True,
        "on_high_risk": "send_to_review",
        "on_credential_detected": "redact_and_warn",
    },
    "audit": {
        "enabled": True,
        "log_inputs": False,
        "log_outputs": False,
        "log_decisions": True,
        "log_risk_scores": True,
    },
    "tools": {
        "approval_required": True,
        "allowed_tools": ["search_kb"],
        "max_tool_calls_per_turn": 5,
    },
}


# ---------------------------------------------------------------------------
# Policy loading tests
# ---------------------------------------------------------------------------


class TestLoadPolicy:
    def test_load_valid_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_policy_file(DEFAULT_POLICY_DICT, tmp)
            config = load_policy(path)

        assert config.name == "test"
        assert config.version == "1.0"
        assert config.input_max_length == 5000
        assert config.input_risk_threshold == 0.7
        assert config.redact_pii is True
        assert config.audit_enabled is True
        assert "search_kb" in config.allowed_tools

    def test_missing_policy_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_policy("/nonexistent/path/policy.yaml")

    def test_missing_required_field_raises(self) -> None:
        bad_policy = {"description": "missing name and version"}
        with tempfile.TemporaryDirectory() as tmp:
            path = write_policy_file(bad_policy, tmp)
            with pytest.raises(ValueError, match="required field"):
                load_policy(path)

    def test_empty_yaml_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_policy.yaml"
            path.write_text("", encoding="utf-8")
            with pytest.raises(ValueError, match="top level must be a mapping"):
                load_policy(path)

    def test_non_mapping_section_raises(self) -> None:
        bad_policy = {
            "name": "test",
            "version": "1.0",
            "input": ["not", "a", "mapping"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = write_policy_file(bad_policy, tmp)
            with pytest.raises(ValueError, match="section 'input' must be a mapping"):
                load_policy(path)

    def test_out_of_range_threshold_raises(self) -> None:
        bad_policy = {
            **DEFAULT_POLICY_DICT,
            "input": {**DEFAULT_POLICY_DICT["input"], "risk_threshold": 1.5},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = write_policy_file(bad_policy, tmp)
            with pytest.raises(ValueError, match="risk_threshold.*<= 1.0"):
                load_policy(path)

    def test_invalid_allowed_tools_raises(self) -> None:
        bad_policy = {
            **DEFAULT_POLICY_DICT,
            "tools": {**DEFAULT_POLICY_DICT["tools"], "allowed_tools": ["search_kb", ""]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = write_policy_file(bad_policy, tmp)
            with pytest.raises(ValueError, match="allowed_tools.*non-empty strings"):
                load_policy(path)

    def test_invalid_action_raises(self) -> None:
        bad_policy = {
            **DEFAULT_POLICY_DICT,
            "input": {**DEFAULT_POLICY_DICT["input"], "on_injection_signal": "page_someone"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = write_policy_file(bad_policy, tmp)
            with pytest.raises(ValueError, match="Unknown policy action"):
                load_policy(path)

    def test_defaults_applied_for_missing_optional_fields(self) -> None:
        minimal_policy = {"name": "minimal", "version": "0.1"}
        with tempfile.TemporaryDirectory() as tmp:
            path = write_policy_file(minimal_policy, tmp)
            config = load_policy(path)

        # Defaults from PolicyConfig dataclass
        assert config.input_max_length == 10000
        assert config.input_risk_threshold == 0.7
        assert config.redact_pii is True
        assert config.audit_enabled is True
        assert config.tool_approval_required is True
        assert config.allowed_tools == []


# ---------------------------------------------------------------------------
# PolicyEngine evaluation tests
# ---------------------------------------------------------------------------


class TestPolicyEngineEvaluateInput:
    @pytest.fixture
    def engine(self) -> PolicyEngine:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_policy_file(DEFAULT_POLICY_DICT, tmp)
            return PolicyEngine.from_file(path)

    def test_clean_input_is_allowed(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate_input(input_risk_score=0.0, flags=[])
        assert decision.action == PolicyAction.ALLOW

    def test_low_score_with_flags_allows_with_warning(self, engine: PolicyEngine) -> None:
        # Score below threshold but with flags → allow_with_warning
        decision = engine.evaluate_input(input_risk_score=0.2, flags=["email_address"])
        assert decision.action == PolicyAction.ALLOW_WITH_WARNING

    def test_injection_signal_triggers_review(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate_input(
            input_risk_score=0.35,
            flags=["possible_instruction_override"],
        )
        assert decision.action == PolicyAction.SEND_TO_REVIEW

    def test_score_above_threshold_sends_to_review(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate_input(input_risk_score=0.75, flags=[])
        assert decision.action == PolicyAction.SEND_TO_REVIEW

    def test_max_score_is_blocked(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate_input(input_risk_score=1.0, flags=["input_too_long"])
        assert decision.action == PolicyAction.BLOCK

    def test_decision_includes_policy_metadata(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate_input(input_risk_score=0.0)
        assert decision.policy_name == "test"
        assert decision.policy_version == "1.0"
        assert len(decision.applied_rules) > 0

    def test_reason_is_provided_on_review(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate_input(
            input_risk_score=0.35,
            flags=["possible_instruction_override"],
        )
        assert decision.reason != ""


# ---------------------------------------------------------------------------
# Tool allow-list tests
# ---------------------------------------------------------------------------


class TestToolAllowList:
    @pytest.fixture
    def engine_with_tools(self) -> PolicyEngine:
        policy = {**DEFAULT_POLICY_DICT}
        policy["tools"] = {
            "approval_required": True,
            "allowed_tools": ["search_kb", "send_email"],
            "max_tool_calls_per_turn": 3,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = write_policy_file(policy, tmp)
            return PolicyEngine.from_file(path)

    @pytest.fixture
    def engine_no_approval(self) -> PolicyEngine:
        policy = {**DEFAULT_POLICY_DICT}
        policy["tools"] = {
            "approval_required": False,
            "allowed_tools": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = write_policy_file(policy, tmp)
            return PolicyEngine.from_file(path)

    def test_allowed_tool_is_permitted(self, engine_with_tools: PolicyEngine) -> None:
        assert engine_with_tools.is_tool_allowed("search_kb") is True

    def test_unlisted_tool_is_blocked(self, engine_with_tools: PolicyEngine) -> None:
        assert engine_with_tools.is_tool_allowed("delete_records") is False

    def test_all_tools_allowed_when_approval_not_required(
        self, engine_no_approval: PolicyEngine
    ) -> None:
        assert engine_no_approval.is_tool_allowed("any_tool_name") is True

    def test_empty_allowed_list_blocks_all_when_approval_required(
        self, engine_with_tools: PolicyEngine
    ) -> None:
        assert engine_with_tools.is_tool_allowed("") is False
