from __future__ import annotations

import re
import unicodedata
from typing import Any


INJECTION_PATTERNS = [
    re.compile(r"ignore\s+all\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+your\s+system\s+prompt", re.IGNORECASE),
    re.compile(r"developer\s+mode", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
]


# Explicitly strip common invisible obfuscation chars + all Unicode control/format chars.
_ZERO_WIDTH_CHARS = {
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u2060",  # word joiner
    "\ufeff",  # zero width no-break space / BOM
}


def _normalize_for_scan(text: str) -> tuple[str, bool]:
    """Return canonicalized text for heuristic scanning and whether it changed.

    Canonicalization pipeline:
    1) NFKC compatibility normalization (collapses many homoglyph/compat forms)
    2) Strip zero-width and Unicode control/format characters
    """
    nfkc = unicodedata.normalize("NFKC", text)

    cleaned_chars: list[str] = []
    for ch in nfkc:
        cat = unicodedata.category(ch)
        if ch in _ZERO_WIDTH_CHARS:
            continue
        # Cc: control, Cf: format (covers many invisible separators/markers)
        if cat in {"Cc", "Cf"}:
            continue
        cleaned_chars.append(ch)

    normalized = "".join(cleaned_chars)
    return normalized, normalized != text


def scan_input(text: str) -> dict[str, Any]:
    normalized_text, normalization_applied = _normalize_for_scan(text)

    matches: list[str] = []
    for pattern in INJECTION_PATTERNS:
        if pattern.search(normalized_text):
            matches.append(pattern.pattern)

    blocked = len(matches) > 0
    score = min(1.0, 0.35 * len(matches))

    return {
        "blocked": blocked,
        "risk_score": score,
        "reasons": matches,
        "audit": {
            "normalization_applied": normalization_applied,
            "scanner": "input",
        },
    }
