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

        rule_id = str(raw_rule.get("id") or raw_rule.get("name") or "").strip()
        flag = str(raw_rule.get("flag") or "").strip()
        category = str(raw_rule.get("category") or "policy").strip()
        pattern_text = str(raw_rule.get("pattern") or "")
        description = str(raw_rule.get("description") or "")

        if not rule_id:
            raise ValueError(f"Regex rule #{index} is missing 'id'.")
        if not flag:
            raise ValueError(f"Regex rule '{rule_id}' is missing 'flag'.")
        if not pattern_text:
            raise ValueError(f"Regex rule '{rule_id}' is missing 'pattern'.")

        default_score = _DEFAULT_CATEGORY_SCORES.get(category, 0.2)
        score = float(raw_rule.get("score", default_score))
        if score <= 0 or score > 1:
            raise ValueError(f"Regex rule '{rule_id}' score must be > 0 and <= 1.")

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

