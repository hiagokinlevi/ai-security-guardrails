"""RAG document sanitization utilities.

This module strips suspicious instruction-override phrases from retrieved
context before it is inserted into model prompts.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


# Patterns target common indirect prompt-injection phrases that attempt to
# override higher-priority instructions or guardrails.
DEFAULT_SUSPICIOUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above|system)\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\boverride\s+(the\s+)?(system\s+prompt|guardrails?|safety\s+rules?)\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(all\s+)?(rules?|instructions?|guardrails?)\b", re.IGNORECASE),
    re.compile(r"\byou\s+must\s+now\s+(ignore|bypass|override)\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class SanitizationResult:
    """Result of document sanitization."""

    sanitized_text: str
    removed_matches: int


_defanged_re = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)
_space_re = re.compile(r"\s+")


def _normalize_for_detection(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.lower()
    normalized = _defanged_re.sub(" ", normalized)
    normalized = _space_re.sub(" ", normalized).strip()
    return normalized


def sanitize_document(
    text: str,
    *,
    replacement: str = "[REMOVED_SUSPICIOUS_INSTRUCTION]",
    patterns: Iterable[re.Pattern[str]] | None = None,
) -> SanitizationResult:
    """Strip suspicious instruction patterns from a single document.

    Detection is performed against both raw and normalized text to improve
    resilience to simple obfuscation.
    """
    if not text:
        return SanitizationResult(sanitized_text=text, removed_matches=0)

    active_patterns = tuple(patterns or DEFAULT_SUSPICIOUS_PATTERNS)
    sanitized = text
    removed = 0

    for pattern in active_patterns:
        sanitized, n = pattern.subn(replacement, sanitized)
        removed += n

    # Secondary pass for lightly obfuscated inputs; if found, remove the exact
    # normalized phrase from the original text when possible via a broad pass.
    normalized = _normalize_for_detection(sanitized)
    for pattern in active_patterns:
        if pattern.search(normalized):
            phrase = pattern.pattern
            coarse = re.compile(phrase.replace(r"\\b", ""), re.IGNORECASE)
            sanitized, n = coarse.subn(replacement, sanitized)
            removed += n

    sanitized = _space_re.sub(" ", sanitized).strip()
    return SanitizationResult(sanitized_text=sanitized, removed_matches=removed)


def sanitize_documents(
    documents: Iterable[str],
    *,
    replacement: str = "[REMOVED_SUSPICIOUS_INSTRUCTION]",
    patterns: Iterable[re.Pattern[str]] | None = None,
) -> list[SanitizationResult]:
    """Sanitize multiple retrieved documents."""
    return [
        sanitize_document(doc, replacement=replacement, patterns=patterns)
        for doc in documents
    ]
