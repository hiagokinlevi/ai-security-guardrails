"""
Policy Engine
==============
Loads YAML policy files and evaluates guardrail decisions against them.

The policy engine is the central authority for allow/warn/block decisions.
It consumes the risk scores from the input and output controls and translates
them into a final PolicyDecision. Policies are defined as YAML files to
support version control, code review, and environment-specific overrides.

Policy YAML schema:
  See policies/default_policy.yaml for a fully documented example.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml


class PolicyAction(str, Enum):
    """Actions the policy engine can take."""

    ALLOW = "allow"
    ALLOW_WITH_WARNING = "allow_with_warning"
    SEND_TO_REVIEW = "send_to_review"
    BLOCK = "block"


@dataclass
class PolicyDecision:
    """Outcome of a policy evaluation."""

    action: PolicyAction
    policy_name: str
    policy_version: str
    reason: str = ""
    applied_rules: list[str] = field(default_factory=list)


@dataclass
class PolicyConfig:
    """Parsed and validated policy configuration."""

    name: str
    version: str
    description: str

    # Input policy settings
    input_max_length: int = 10000
    input_risk_threshold: float = 0.7
    on_injection_signal: str = "send_to_review"
    on_sensitive_data: str = "allow_with_warning"
    on_block: str = "return_safe_error_message"

    # Output policy settings
    output_risk_threshold: float = 0.8
    redact_pii: bool = True
    on_high_risk_output: str = "send_to_review"
    on_credential_detected: str = "redact_and_warn"

    # Audit settings
    audit_enabled: bool = True
    log_inputs: bool = False    # Never log raw inputs by default
    log_outputs: bool = False   # Never log raw outputs by default
    log_decisions: bool = True
    log_risk_scores: bool = True

    # Tool policy settings
    tool_approval_required: bool = True
    allowed_tools: list[str] = field(default_factory=list)
    max_tool_calls_per_turn: int = 5


def _action_from_string(value: str) -> PolicyAction:
    """Convert a policy string value to a PolicyAction enum."""
    mapping: dict[str, PolicyAction] = {
        "allow": PolicyAction.ALLOW,
        "allow_with_warning": PolicyAction.ALLOW_WITH_WARNING,
        "send_to_review": PolicyAction.SEND_TO_REVIEW,
        "block": PolicyAction.BLOCK,
        # Aliases used in YAML files
        "redact_and_warn": PolicyAction.ALLOW_WITH_WARNING,
        "return_safe_error_message": PolicyAction.BLOCK,
    }
    action = mapping.get(value.lower())
    if action is None:
        raise ValueError(
            f"Unknown policy action '{value}'. "
            f"Valid values: {', '.join(mapping.keys())}"
        )
    return action


def load_policy(policy_path: str | Path) -> PolicyConfig:
    """
    Load and parse a YAML policy file into a PolicyConfig object.

    Args:
        policy_path: Path to the YAML policy file.

    Returns:
        Parsed PolicyConfig.

    Raises:
        FileNotFoundError: If the policy file does not exist.
        ValueError: If required fields are missing or have invalid values.
    """
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    with path.open("r") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    # Validate required top-level fields
    for required_field in ("name", "version"):
        if required_field not in raw:
            raise ValueError(f"Policy file missing required field: '{required_field}'")

    input_cfg = raw.get("input", {})
    output_cfg = raw.get("output", {})
    audit_cfg = raw.get("audit", {})
    tools_cfg = raw.get("tools", {})

    return PolicyConfig(
        name=raw["name"],
        version=str(raw["version"]),
        description=raw.get("description", ""),
        # Input
        input_max_length=input_cfg.get("max_length", 10000),
        input_risk_threshold=input_cfg.get("risk_threshold", 0.7),
        on_injection_signal=input_cfg.get("on_injection_signal", "send_to_review"),
        on_sensitive_data=input_cfg.get("on_sensitive_data", "allow_with_warning"),
        on_block=input_cfg.get("on_block", "return_safe_error_message"),
        # Output
        output_risk_threshold=output_cfg.get("risk_threshold", 0.8),
        redact_pii=output_cfg.get("redact_pii", True),
        on_high_risk_output=output_cfg.get("on_high_risk", "send_to_review"),
        on_credential_detected=output_cfg.get("on_credential_detected", "redact_and_warn"),
        # Audit
        audit_enabled=audit_cfg.get("enabled", True),
        log_inputs=audit_cfg.get("log_inputs", False),
        log_outputs=audit_cfg.get("log_outputs", False),
        log_decisions=audit_cfg.get("log_decisions", True),
        log_risk_scores=audit_cfg.get("log_risk_scores", True),
        # Tools
        tool_approval_required=tools_cfg.get("approval_required", True),
        allowed_tools=tools_cfg.get("allowed_tools", []),
        max_tool_calls_per_turn=tools_cfg.get("max_tool_calls_per_turn", 5),
    )


class PolicyEngine:
    """
    Evaluates guardrail decisions against a loaded policy.

    Usage:
        engine = PolicyEngine.from_file("policies/default_policy.yaml")
        decision = engine.evaluate_input(input_risk_score=0.85, flags=["possible_injection"])
    """

    def __init__(self, policy: PolicyConfig) -> None:
        self._policy = policy

    @classmethod
    def from_file(cls, policy_path: str | Path) -> "PolicyEngine":
        """Create a PolicyEngine from a YAML policy file."""
        return cls(load_policy(policy_path))

    @classmethod
    def from_env(cls) -> "PolicyEngine":
        """
        Create a PolicyEngine from the path specified in the POLICY_FILE
        environment variable, falling back to the default policy.
        """
        policy_path = os.getenv(
            "POLICY_FILE",
            Path(__file__).parent.parent.parent / "policies" / "default_policy.yaml",
        )
        return cls.from_file(policy_path)

    @property
    def policy(self) -> PolicyConfig:
        """Access the underlying policy configuration."""
        return self._policy

    def evaluate_input(
        self,
        input_risk_score: float,
        flags: Optional[list[str]] = None,
    ) -> PolicyDecision:
        """
        Evaluate an input validation result against the policy.

        Args:
            input_risk_score: Float between 0.0 and 1.0 from the input validator.
            flags: List of risk flag names detected during validation.

        Returns:
            PolicyDecision with the recommended action.
        """
        flags = flags or []
        applied_rules: list[str] = []

        # Hard block for inputs exceeding length (score == 1.0 from validator)
        if input_risk_score >= 1.0:
            return PolicyDecision(
                action=PolicyAction.BLOCK,
                policy_name=self._policy.name,
                policy_version=self._policy.version,
                reason="Input risk score is at maximum — blocked by policy.",
                applied_rules=["max_risk_score"],
            )

        # Check injection signals
        injection_flags = [f for f in flags if "injection" in f or "override" in f or "jailbreak" in f]
        if injection_flags:
            applied_rules.append("injection_signal_rule")
            action = _action_from_string(self._policy.on_injection_signal)
            if action in (PolicyAction.BLOCK, PolicyAction.SEND_TO_REVIEW):
                return PolicyDecision(
                    action=action,
                    policy_name=self._policy.name,
                    policy_version=self._policy.version,
                    reason=f"Injection signals detected: {injection_flags}",
                    applied_rules=applied_rules,
                )

        # Check sensitive data in input
        sensitive_flags = [f for f in flags if "credential" in f or "card" in f or "key" in f]
        if sensitive_flags:
            applied_rules.append("sensitive_data_rule")
            action = _action_from_string(self._policy.on_sensitive_data)
            if action == PolicyAction.BLOCK:
                return PolicyDecision(
                    action=action,
                    policy_name=self._policy.name,
                    policy_version=self._policy.version,
                    reason=f"Sensitive data detected in input: {sensitive_flags}",
                    applied_rules=applied_rules,
                )

        # General threshold check
        if input_risk_score >= self._policy.input_risk_threshold:
            applied_rules.append("risk_threshold_rule")
            return PolicyDecision(
                action=PolicyAction.SEND_TO_REVIEW,
                policy_name=self._policy.name,
                policy_version=self._policy.version,
                reason=f"Input risk score {input_risk_score} exceeds threshold "
                       f"{self._policy.input_risk_threshold}.",
                applied_rules=applied_rules,
            )

        # Input passes all checks
        action = PolicyAction.ALLOW_WITH_WARNING if flags else PolicyAction.ALLOW
        return PolicyDecision(
            action=action,
            policy_name=self._policy.name,
            policy_version=self._policy.version,
            reason="Input passed all policy checks.",
            applied_rules=applied_rules or ["default_allow"],
        )

    def is_tool_allowed(self, tool_name: str) -> bool:
        """
        Check whether an agent tool is allowed by the current policy.

        If tool_approval_required is False, all tools are allowed.
        Otherwise, only tools in the allowed_tools list are permitted.
        """
        if not self._policy.tool_approval_required:
            return True
        return tool_name in self._policy.allowed_tools
