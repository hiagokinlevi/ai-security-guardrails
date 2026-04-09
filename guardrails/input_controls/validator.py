"""
Input Validation and Risk Classification
=========================================
Validates and risk-scores user inputs before they reach the LLM.

This module does NOT classify inputs as "malicious" with certainty —
it computes a risk score based on heuristics and pattern matching.
The final decision is made by the policy engine.

Limitations:
- Cannot detect all forms of prompt injection
- Heuristics may produce false positives
- Should be combined with output controls and audit logging
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class InputDecision(str, Enum):
    """Possible outcomes from input validation."""

    ALLOW = "allow"
    ALLOW_WITH_WARNING = "allow_with_warning"
    SEND_TO_REVIEW = "send_to_review"
    BLOCK = "block"


@dataclass
class InputValidationResult:
    """Result of input validation."""

    decision: InputDecision
    risk_score: float           # 0.0 (clean) to 1.0 (high risk)
    risk_flags: list[str] = field(default_factory=list)
    reason: Optional[str] = None
    original_length: int = 0
    sanitized_input: Optional[str] = None


# ---------------------------------------------------------------------------
# Heuristic pattern libraries
# ---------------------------------------------------------------------------
# Patterns that may indicate prompt injection or policy violation attempts.
# These are heuristic signals, not definitive detections. The regexes are
# intentionally conservative — prefer false positives over false negatives
# for security-critical paths.
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore (all |previous |above )?instructions?", "possible_instruction_override"),
    (r"you are now (a |an )?", "possible_role_override"),
    (r"disregard (your |all |previous )", "possible_instruction_override"),
    (r"<\|?(system|user|assistant|im_start|im_end)\|?>", "possible_delimiter_injection"),
    (r"###\s*(system|instruction|context)", "possible_delimiter_injection"),
    (r"jailbreak", "explicit_jailbreak_keyword"),
    (
        r"act as (if |though )?you (have no|don't have) (restrictions|limits|rules)",
        "restriction_bypass_attempt",
    ),
    # DAN-style and similar jailbreak templates
    (r"do anything now", "possible_dan_jailbreak"),
    (r"developer mode", "possible_developer_mode_jailbreak"),
    # Hidden Unicode / zero-width characters used to hide injections
    (r"[\u200b\u200c\u200d\ufeff]", "hidden_unicode_characters"),
]

_SENSITIVE_DATA_PATTERNS: list[tuple[str, str]] = [
    # Credit card numbers (major card formats)
    (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "possible_credit_card"),
    # Email addresses
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email_address"),
    # Credentials appearing in key=value or key: value format
    (r"(password|passwd|secret|api[_-]?key|token)\s*[:=]\s*\S+", "possible_credential_in_input"),
    # PEM-encoded private keys
    (r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----", "private_key_material"),
    # AWS access key pattern
    (r"\bAKIA[0-9A-Z]{16}\b", "possible_aws_access_key"),
    # SSN (US Social Security Number)
    (r"\b\d{3}-\d{2}-\d{4}\b", "possible_us_ssn"),
]


def validate_input(
    user_input: str,
    max_length: int = 10000,
    risk_threshold: float = 0.7,
) -> InputValidationResult:
    """
    Validate and risk-score a user input string.

    The function applies two independent sets of heuristics:
    1. Injection signal detection — looks for patterns commonly seen in
       prompt injection and jailbreak attempts.
    2. Sensitive data detection — looks for credentials, PII, and other
       data that should not be sent to the LLM.

    Risk scoring is additive: each matched pattern adds weight to the
    total score, which is capped at 1.0. The policy engine makes the
    final allow/block decision based on the score and configured thresholds.

    Args:
        user_input: Raw input from the user or application layer.
        max_length: Maximum allowed input length in characters.
        risk_threshold: Score at or above which input is escalated for review.

    Returns:
        InputValidationResult with decision, score, and flags.
    """
    flags: list[str] = []
    risk_score = 0.0

    # Hard length limit — block immediately without further processing
    if len(user_input) > max_length:
        return InputValidationResult(
            decision=InputDecision.BLOCK,
            risk_score=1.0,
            risk_flags=["input_too_long"],
            reason=f"Input exceeds maximum length of {max_length} characters.",
            original_length=len(user_input),
        )

    lower_input = user_input.lower()

    # --- Injection signal scan ---
    for pattern, flag in _INJECTION_PATTERNS:
        if re.search(pattern, lower_input, re.IGNORECASE):
            flags.append(flag)
            risk_score += 0.35  # Each injection signal adds significant weight

    # --- Sensitive data scan ---
    for pattern, flag in _SENSITIVE_DATA_PATTERNS:
        if re.search(pattern, user_input, re.IGNORECASE):
            flags.append(flag)
            risk_score += 0.2   # Sensitive data adds moderate weight

    # Cap the score — multiple signals cannot push it beyond 1.0
    risk_score = min(risk_score, 1.0)

    # Determine decision based on score thresholds
    if risk_score >= risk_threshold:
        # High-risk inputs with very high scores are blocked outright;
        # borderline scores are sent to the review queue.
        decision = (
            InputDecision.BLOCK if risk_score >= 0.9 else InputDecision.SEND_TO_REVIEW
        )
    elif flags:
        # Some signals were detected but score is below the threshold —
        # allow the input but attach a warning for downstream systems.
        decision = InputDecision.ALLOW_WITH_WARNING
    else:
        decision = InputDecision.ALLOW

    return InputValidationResult(
        decision=decision,
        risk_score=round(risk_score, 4),
        risk_flags=list(set(flags)),    # Deduplicate flags
        original_length=len(user_input),
    )


def is_allowed(result: InputValidationResult) -> bool:
    """
    Convenience function that returns True if input should proceed to the model.

    Note: ALLOW_WITH_WARNING inputs are considered allowed. The caller is
    responsible for deciding whether to surface the warning.
    """
    return result.decision in (InputDecision.ALLOW, InputDecision.ALLOW_WITH_WARNING)
