from __future__ import annotations

import sys
from typing import Any

import yaml

from guardrails.audit import emit_audit_event
from guardrails.schemas.audit import PolicyDecisionReasonCode

POLICY_LOAD_VALIDATION_EXIT_CODE = 78


def _walk_reason_code_paths(node: Any, path: str = "$") -> list[tuple[str, str]]:
    """Return list of (path, reason_code) pairs for all configured reason codes."""
    found: list[tuple[str, str]] = []

    if isinstance(node, dict):
        if "policy_decision_reason_code" in node and isinstance(
            node["policy_decision_reason_code"], str
        ):
            found.append((f"{path}.policy_decision_reason_code", node["policy_decision_reason_code"]))

        for key, value in node.items():
            child_path = f"{path}.{key}"
            found.extend(_walk_reason_code_paths(value, child_path))

    elif isinstance(node, list):
        for idx, item in enumerate(node):
            found.extend(_walk_reason_code_paths(item, f"{path}[{idx}]"))

    return found


def _validate_reason_codes_or_exit(policy: dict[str, Any]) -> None:
    valid_codes = {member.value for member in PolicyDecisionReasonCode}
    invalid_paths: list[str] = []

    for path, code in _walk_reason_code_paths(policy):
        if code not in valid_codes:
            invalid_paths.append(f"{path}={code}")

    if invalid_paths:
        emit_audit_event(
            {
                "event_type": "policy_load_validation",
                "action": "block",
                "reason": "invalid_policy_decision_reason_code",
                "details": {
                    "invalid_reason_code_paths": invalid_paths,
                },
            }
        )
        raise SystemExit(POLICY_LOAD_VALIDATION_EXIT_CODE)


def load_policy(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        policy = yaml.safe_load(f) or {}

    _validate_reason_codes_or_exit(policy)
    return policy
