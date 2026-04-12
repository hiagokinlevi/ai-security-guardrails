"""
YAML-backed regex rule sets for guardrail input validation.

Rule sets let application teams add environment-specific defensive detections
without editing library code. They are intentionally small and offline-only:
each rule is a reviewed regex with a flag name and risk score contribution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


_DEFAULT_CATEGORY_SCORES = {
    "injection": 0.35,
    "sensitive_data": 0.2,
    "policy": 0.2,
}


@dataclass(frozen=True)
class RegexRule:
    """Compiled regex rule loaded from YAML."""

    rule_id: str
    flag: str
    category: str
    score: float
    pattern: re.Pattern[str]
    description: str = ""


def _require_string(value: object, field_name: str, *, rule_ref: str) -> str:
    """Validate required string fields without coercing other YAML types."""
    if not isinstance(value, str):
        raise ValueError(f"Regex rule '{rule_ref}' field '{field_name}' must be a string.")
    text = value.strip()
    if not text:
        raise ValueError(f"Regex rule '{rule_ref}' field '{field_name}' cannot be empty.")
    return text


def _get_category(raw_rule: dict[object, object], *, rule_ref: str) -> str:
    """Return a supported rule category or raise a clear schema error."""
    raw_category = raw_rule.get("category", "policy")
    if not isinstance(raw_category, str):
        raise ValueError(f"Regex rule '{rule_ref}' field 'category' must be a string.")
    category = raw_category.strip() or "policy"
    if category not in _DEFAULT_CATEGORY_SCORES:
        allowed = ", ".join(sorted(_DEFAULT_CATEGORY_SCORES))
        raise ValueError(
            f"Regex rule '{rule_ref}' category must be one of: {allowed}."
        )
    return category


def _get_score(
    raw_rule: dict[object, object],
    *,
    rule_ref: str,
    default_score: float,
) -> float:
    """Validate numeric rule scores and reject YAML booleans."""
    raw_score = raw_rule.get("score", default_score)
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise ValueError(f"Regex rule '{rule_ref}' field 'score' must be numeric.")
    score = float(raw_score)
    if score <= 0 or score > 1:
        raise ValueError(f"Regex rule '{rule_ref}' score must be > 0 and <= 1.")
    return score


def load_regex_rules(rule_set_path: str | Path) -> list[RegexRule]:
    """
    Load a YAML regex rule set.

    Expected schema:

    ```yaml
    rules:
      - id: org-secret-marker
        flag: internal_secret_marker
        category: sensitive_data
        pattern: "\\bINTERNAL_SECRET_[A-Z0-9]+\\b"
        score: 0.3
    ```
    """
    path = Path(rule_set_path)
    if not path.exists():
        raise FileNotFoundError(f"Regex rule set not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Regex rule set must be a YAML mapping.")

    raw_rules = raw.get("rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError("Regex rule set field 'rules' must be a list.")

    rules: list[RegexRule] = []
    for index, raw_rule in enumerate(raw_rules, start=1):
        if not isinstance(raw_rule, dict):
            raise ValueError(f"Regex rule #{index} must be a mapping.")

        raw_id = raw_rule.get("id")
        raw_name = raw_rule.get("name")
        if raw_id is None and raw_name is None:
            raise ValueError(f"Regex rule #{index} is missing 'id'.")
        rule_id = _require_string(
            raw_id if raw_id is not None else raw_name,
            "id",
            rule_ref=f"#{index}",
        )
        flag = _require_string(raw_rule.get("flag"), "flag", rule_ref=rule_id)
        category = _get_category(raw_rule, rule_ref=rule_id)
        pattern_text = _require_string(
            raw_rule.get("pattern"),
            "pattern",
            rule_ref=rule_id,
        )
        raw_description = raw_rule.get("description", "")
        if not isinstance(raw_description, str):
            raise ValueError(
                f"Regex rule '{rule_id}' field 'description' must be a string."
            )
        description = raw_description

        default_score = _DEFAULT_CATEGORY_SCORES[category]
        score = _get_score(raw_rule, rule_ref=rule_id, default_score=default_score)

        try:
            pattern = re.compile(pattern_text, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"Regex rule '{rule_id}' has an invalid pattern: {exc}") from exc

        rules.append(
            RegexRule(
                rule_id=rule_id,
                flag=flag,
                category=category,
                score=score,
                pattern=pattern,
                description=description,
            )
        )

    return rules

