"""
Output Controls — Response Filtering
======================================
Scans and filters LLM responses before they are returned to the user.

Responsibilities:
- Detect sensitive data in model outputs (PII, credentials, internal paths)
- Apply configurable redaction
- Compute an output risk score
- Emit structured events for the audit logger

Design note:
  Output controls are the last line of defense before a response reaches
  the user. Even if the input controls and policy engine allowed a request,
  the output filter may still catch sensitive data that leaked through
  the model's response. The two layers are intentionally independent.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from guardrails.redaction.redactor import redact_sensitive_data


class OutputDecision(str, Enum):
    """Possible outcomes from output filtering."""

    PASS = "pass"                  # Output is clean — return as-is
    PASS_REDACTED = "pass_redacted"  # Output was redacted — return sanitized version
    SEND_TO_REVIEW = "send_to_review"  # Output has risky content, flag for review
    BLOCK = "block"               # Output must not be returned to the user


@dataclass
class OutputFilterResult:
    """Result of output filtering."""

    decision: OutputDecision
    risk_score: float              # 0.0 (clean) to 1.0 (high risk)
    risk_flags: list[str] = field(default_factory=list)
    original_output: str = ""
    filtered_output: str = ""      # Redacted or safe-error version
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Output-specific risk patterns
# ---------------------------------------------------------------------------
# These patterns look for content that should never appear in model responses
# when the model is operating as a production application. Some of these
# (e.g., internal IP addresses) may be legitimate in certain contexts —
# tune the thresholds in your policy rather than removing patterns.

_OUTPUT_RISK_PATTERNS: list[tuple[str, str, float]] = [
    # (pattern, flag_name, risk_weight)

    # Private key material — critical, should always block
    (r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----", "private_key_in_output", 0.9),

    # AWS credentials
    (r"\bAKIA[0-9A-Z]{16}\b", "aws_access_key_in_output", 0.8),
    (r"aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*\S+", "aws_secret_in_output", 0.9),
    (r"(?:proxy-)?authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~+/=-]{20,}",
     "bearer_token_in_output", 0.9),

    # Generic API key / token patterns
    (r"(api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token)\s*[:=]\s*[\w\-./+]{20,}",
     "api_credential_in_output", 0.7),

    # PII — email addresses
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email_in_output", 0.3),

    # PII — credit card numbers
    (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "credit_card_in_output", 0.6),

    # Internal hostnames / IP ranges
    (r"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b",
     "internal_ip_in_output", 0.3),

    # System prompt leakage signals
    (r"(system prompt|you are a|your instructions are|as an ai language model)",
     "possible_system_prompt_leakage", 0.4),
]

# Safe error message returned when output is blocked
_SAFE_ERROR_MESSAGE = (
    "I'm sorry, I'm unable to provide that information. "
    "Please contact support if you believe this is an error."
)


def filter_output(
    model_output: str,
    redact_pii: bool = True,
    risk_threshold: float = 0.8,
) -> OutputFilterResult:
    """
    Scan and optionally redact an LLM response before returning it to the user.

    Args:
        model_output: Raw text response from the LLM.
        redact_pii: If True, apply PII redaction before returning the output.
        risk_threshold: Score at or above which output is escalated for review.

    Returns:
        OutputFilterResult with decision, score, filtered output, and flags.
    """
    flags: list[str] = []
    risk_score = 0.0

    # Scan against known risky output patterns
    for pattern, flag, weight in _OUTPUT_RISK_PATTERNS:
        if re.search(pattern, model_output, re.IGNORECASE):
            flags.append(flag)
            risk_score += weight

    # Cap score at 1.0
    risk_score = min(risk_score, 1.0)

    # Determine action based on score
    if risk_score >= risk_threshold:
        # Output contains high-confidence sensitive content — block it
        return OutputFilterResult(
            decision=OutputDecision.BLOCK,
            risk_score=round(risk_score, 4),
            risk_flags=list(set(flags)),
            original_output=model_output,
            filtered_output=_SAFE_ERROR_MESSAGE,
            reason="Output risk score exceeds threshold. Response blocked.",
        )

    # Apply PII redaction if requested (even for low-risk outputs)
    filtered = redact_sensitive_data(model_output) if redact_pii else model_output

    if flags:
        # Detected some signals but below hard threshold — send for review
        # but return a (potentially redacted) response to avoid blocking legitimate use
        return OutputFilterResult(
            decision=OutputDecision.SEND_TO_REVIEW,
            risk_score=round(risk_score, 4),
            risk_flags=list(set(flags)),
            original_output=model_output,
            filtered_output=filtered,
        )

    if filtered != model_output:
        # Redaction changed the output — note this in the result
        return OutputFilterResult(
            decision=OutputDecision.PASS_REDACTED,
            risk_score=round(risk_score, 4),
            risk_flags=[],
            original_output=model_output,
            filtered_output=filtered,
        )

    # Clean output — return as-is
    return OutputFilterResult(
        decision=OutputDecision.PASS,
        risk_score=0.0,
        risk_flags=[],
        original_output=model_output,
        filtered_output=model_output,
    )
