from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

POLICY_PARSE_EXIT_CODE = 78


def _fatal_policy_error(*, policy_path: str, error_type: str, message: str) -> None:
    logger.critical(
        "policy_startup_fatal",
        extra={
            "event": "policy_startup_fatal",
            "policy_path": policy_path,
            "error_type": error_type,
            "message": message,
            "exit_code": POLICY_PARSE_EXIT_CODE,
        },
    )
    raise SystemExit(POLICY_PARSE_EXIT_CODE)


def _validate_policy_shape(policy: dict[str, Any], policy_path: str) -> None:
    if not isinstance(policy, dict):
        _fatal_policy_error(
            policy_path=policy_path,
            error_type="policy_type_error",
            message="Policy root must be a mapping/object.",
        )

    rules = policy.get("rules")
    if rules is None:
        return

    if not isinstance(rules, list):
        _fatal_policy_error(
            policy_path=policy_path,
            error_type="rules_type_error",
            message="'rules' must be a list.",
        )

    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            _fatal_policy_error(
                policy_path=policy_path,
                error_type="rule_type_error",
                message=f"Rule at index {idx} must be an object.",
            )

        action = rule.get("action")
        if action is not None and action not in {"allow", "warn", "block"}:
            _fatal_policy_error(
                policy_path=policy_path,
                error_type="invalid_rule_action",
                message=f"Rule at index {idx} has invalid action '{action}'.",
            )


def load_policy_or_exit(policy_path: str | Path) -> dict[str, Any]:
    path = Path(policy_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        _fatal_policy_error(
            policy_path=str(path),
            error_type="policy_read_error",
            message=str(exc),
        )

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        _fatal_policy_error(
            policy_path=str(path),
            error_type="policy_yaml_parse_error",
            message=str(exc),
        )

    if parsed is None:
        parsed = {}

    _validate_policy_shape(parsed, str(path))
    return parsed
