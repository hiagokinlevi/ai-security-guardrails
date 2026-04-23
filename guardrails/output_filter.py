from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


AWS_ACCESS_KEY_RE = re.compile(r"\b(AKIA|ASIA|A3T[A-Z0-9])[A-Z0-9]{16}\b")
JWT_RE = re.compile(r"\beyJ[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\b")
LONG_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_\-+/=]{32,}\b")
HEX_TOKEN_RE = re.compile(r"\b[a-fA-F0-9]{32,}\b")


@dataclass
class OutputFilterResult:
    content: str
    action: str
    reasons: list[str]


class OutputFilter:
    def __init__(self, policy: dict[str, Any] | None = None):
        self.policy = policy or {}

    def filter(self, text: str) -> OutputFilterResult:
        matches = self._detect_secrets(text)
        if not matches:
            return OutputFilterResult(content=text, action="allow", reasons=[])

        action = self.policy.get("secret_output_action", "redact")
        reasons = ["secret_detected"]

        if action == "block":
            return OutputFilterResult(content="[BLOCKED: potential secret leakage]", action="block", reasons=reasons)

        redacted = text
        # Replace longer matches first to avoid partial overlap artifacts
        for token in sorted(set(matches), key=len, reverse=True):
            redacted = redacted.replace(token, "[REDACTED_SECRET]")

        return OutputFilterResult(content=redacted, action="redact", reasons=reasons)

    def _detect_secrets(self, text: str) -> list[str]:
        hits: list[str] = []

        for rx in (AWS_ACCESS_KEY_RE, JWT_RE, HEX_TOKEN_RE):
            hits.extend(m.group(0) for m in rx.finditer(text))

        for m in LONG_TOKEN_RE.finditer(text):
            tok = m.group(0)
            if self._likely_high_entropy_token(tok):
                hits.append(tok)

        return hits

    def _likely_high_entropy_token(self, token: str) -> bool:
        # Ignore obvious prose-like strings
        if token.isalpha() or token.isdigit():
            return False

        # Must contain diversity and be sufficiently random-looking
        charset_types = 0
        charset_types += bool(re.search(r"[a-z]", token))
        charset_types += bool(re.search(r"[A-Z]", token))
        charset_types += bool(re.search(r"\d", token))
        charset_types += bool(re.search(r"[_\-+/=]", token))
        if charset_types < 2:
            return False

        entropy = self._shannon_entropy(token)
        return entropy >= 3.5

    @staticmethod
    def _shannon_entropy(s: str) -> float:
        counts = Counter(s)
        total = len(s)
        return -sum((c / total) * math.log2(c / total) for c in counts.values())
