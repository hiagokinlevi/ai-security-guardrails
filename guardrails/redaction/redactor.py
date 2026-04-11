"""
PII and Secret Redactor
========================
Applies regex-based redaction to text strings, replacing sensitive patterns
with clearly-marked placeholder tokens.

Design philosophy:
  Redaction is always lossy — it removes information. This module is
  intentionally conservative: it would rather over-redact than under-redact.
  If you find legitimate content being redacted, adjust the calling code's
  `redact_pii` flag rather than weakening these patterns.

  All replacement tokens use the format [REDACTED:<TYPE>] so that:
  1. Downstream systems know redaction occurred.
  2. The type of redacted content is visible for auditing.
  3. The token is clearly not valid data.
"""

import re
from dataclasses import dataclass, field


@dataclass
class RedactionResult:
    """Result of a redaction pass over a text string."""

    original_text: str
    redacted_text: str
    redaction_count: int = 0          # Number of individual replacements made
    redacted_types: list[str] = field(default_factory=list)  # Types of content redacted


# ---------------------------------------------------------------------------
# Redaction rule definitions
# ---------------------------------------------------------------------------
# Each rule is a tuple of (compiled_regex, replacement_token, rule_name).
# Rules are applied in order — later rules may match text already redacted
# by earlier rules, which is harmless since the token format is consistent.

_REDACTION_RULES: list[tuple[re.Pattern[str], str, str]] = [
    # Private key material — highest priority
    (
        re.compile(
            r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (RSA |EC |OPENSSH )?PRIVATE KEY-----",
            re.IGNORECASE,
        ),
        "[REDACTED:PRIVATE_KEY]",
        "private_key",
    ),
    # AWS access key ID
    (
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "[REDACTED:AWS_KEY_ID]",
        "aws_access_key_id",
    ),
    # Generic API key / secret in key=value or key: value format
    (
        re.compile(
            r"(api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token|bearer)\s*[:=]\s*([\w\-./+]{20,})",
            re.IGNORECASE,
        ),
        r"\1=[REDACTED:API_CREDENTIAL]",
        "api_credential",
    ),
    # Authorization headers carrying bearer tokens
    (
        re.compile(
            r"((?:proxy-)?authorization)\s*([:=])\s*bearer\s+([A-Za-z0-9._~+/=-]{20,})",
            re.IGNORECASE,
        ),
        r"\1\2 Bearer [REDACTED:BEARER_TOKEN]",
        "bearer_token",
    ),
    # Password / secret in assignment format
    (
        re.compile(
            r"(password|passwd|secret|client[_-]?secret)\s*[:=]\s*\S+",
            re.IGNORECASE,
        ),
        r"\1=[REDACTED:SECRET]",
        "password_or_secret",
    ),
    # Credit card numbers (Luhn-like 16-digit patterns)
    (
        re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
        "[REDACTED:CREDIT_CARD]",
        "credit_card",
    ),
    # US Social Security Numbers
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED:SSN]",
        "us_ssn",
    ),
    # Phone numbers (basic international format)
    (
        re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
        "[REDACTED:PHONE]",
        "phone_number",
    ),
    # Email addresses
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "[REDACTED:EMAIL]",
        "email",
    ),
    # IPv4 addresses (private ranges only — public IPs are often legitimate)
    (
        re.compile(
            r"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
            r"|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
            r"|192\.168\.\d{1,3}\.\d{1,3})\b"
        ),
        "[REDACTED:PRIVATE_IP]",
        "private_ip",
    ),
]


def redact_sensitive_data(text: str) -> str:
    """
    Apply all redaction rules to the given text and return the redacted string.

    This is the fast-path function for callers that only need the redacted text
    and do not need metadata about what was redacted. Use `redact_with_report`
    if you need the full RedactionResult.

    Args:
        text: Input string to redact.

    Returns:
        Redacted string with sensitive patterns replaced by placeholder tokens.
    """
    for pattern, replacement, _ in _REDACTION_RULES:
        text = pattern.sub(replacement, text)
    return text


def redact_with_report(text: str) -> RedactionResult:
    """
    Apply all redaction rules and return a full RedactionResult including
    counts and types of redacted content.

    Useful for audit logging when you want to know what was found.

    Args:
        text: Input string to redact.

    Returns:
        RedactionResult with the redacted text and metadata.
    """
    current_text = text
    total_count = 0
    redacted_types: list[str] = []

    for pattern, replacement, rule_name in _REDACTION_RULES:
        matches = pattern.findall(current_text)
        if matches:
            current_text = pattern.sub(replacement, current_text)
            total_count += len(matches)
            redacted_types.append(rule_name)

    return RedactionResult(
        original_text=text,
        redacted_text=current_text,
        redaction_count=total_count,
        redacted_types=redacted_types,
    )
