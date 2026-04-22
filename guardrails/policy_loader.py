from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class PolicyLoadError(RuntimeError):
    """Raised when a policy file cannot be loaded or validated."""


def load_policy(policy_path: str | Path) -> dict[str, Any]:
    """Load and strictly validate a YAML policy file.

    Fail-closed behavior:
    - Missing file -> PolicyLoadError
    - Invalid YAML -> PolicyLoadError
    - Invalid schema / missing required keys -> PolicyLoadError
    """
    path = Path(policy_path)
    if not path.exists():
        raise PolicyLoadError(f"Policy file not found: {path}")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyLoadError(f"Unable to read policy file '{path}': {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise PolicyLoadError(
            f"Invalid YAML in policy file '{path}': {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise PolicyLoadError(
            f"Policy file '{path}' must contain a top-level mapping/object."
        )

    _validate_policy_schema(data, path)
    return data


def _validate_policy_schema(policy: dict[str, Any], path: Path) -> None:
    rules = policy.get("rules")
    if rules is None:
        raise PolicyLoadError(
            f"Policy file '{path}' is missing required key: 'rules'."
        )
    if not isinstance(rules, list):
        raise PolicyLoadError(
            f"Policy file '{path}' key 'rules' must be a list."
        )

    required_rule_keys = {"id", "action", "pattern"}
    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise PolicyLoadError(
                f"Policy file '{path}' rule at index {idx} must be an object/mapping."
            )

        missing = sorted(required_rule_keys - set(rule.keys()))
        if missing:
            raise PolicyLoadError(
                f"Policy file '{path}' rule at index {idx} is missing required keys: {', '.join(missing)}"
            )
