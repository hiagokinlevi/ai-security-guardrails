# adversarial_input_detector.py — Cyber Port Portfolio
# Detect adversarial inputs targeting machine learning models and LLMs.
#
# License: CC BY 4.0  https://creativecommons.org/licenses/by/4.0/
# Author:  Cyber Port  (github.com/hiagokinlevi)
# Python:  3.9+
#
# Checks implemented
# ------------------
# ADV-001  Model extraction probe            HIGH      w=25
# ADV-002  Membership inference probe        HIGH      w=25
# ADV-003  Backdoor trigger injection        CRITICAL  w=40
# ADV-004  Adversarial text filter bypass    HIGH      w=25
# ADV-005  Encoded payload in input          HIGH      w=25
# ADV-006  Long contradictory context        MEDIUM    w=15
# ADV-007  Model inversion attack indicators HIGH      w=25

from __future__ import annotations

import base64
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Check weights (used for risk_score calculation)
# ---------------------------------------------------------------------------
_CHECK_WEIGHTS: Dict[str, int] = {
    "ADV-001": 25,
    "ADV-002": 25,
    "ADV-003": 40,
    "ADV-004": 25,
    "ADV-005": 25,
    "ADV-006": 15,
    "ADV-007": 25,
}

# ---------------------------------------------------------------------------
# Leetspeak and homoglyph character set (ADV-004)
# chr(0x430)=Cyrillic а, chr(0x435)=е, chr(0x43E)=о, chr(0x441)=с
# ---------------------------------------------------------------------------
_LEET_CHARS: frozenset = frozenset(
    set("4301!@$3€")
    | {chr(0x430), chr(0x435), chr(0x43E), chr(0x441)}  # Cyrillic аеос
)

# ---------------------------------------------------------------------------
# Pre-compiled regular expressions
# ---------------------------------------------------------------------------

# ADV-001 — model extraction / boundary-testing enumeration signals
_RE_EXTRACTION_ENUM = re.compile(
    r"(?:what\s+if|slightly|variation|test\s+\d+|option\s+\d+)",
    re.IGNORECASE,
)

# ADV-002 — membership inference / verbatim reproduction requests
_RE_MEMBERSHIP = re.compile(
    r"repeat\s+the\s+exact"
    r"|verbatim\s+from"
    r"|word\s+for\s+word"
    r"|training\s+data"
    r"|memorized"
    r"|was\s+in\s+your\s+training"
    r"|reproduce\s+exactly",
    re.IGNORECASE,
)

# ADV-003 — zero-width and BOM characters
_RE_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")

# ADV-003 — low-level ASCII control characters (excludes \t \n \r)
_RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# ADV-005 — base64 blob (50+ continuous chars)
_RE_BASE64 = re.compile(r"[A-Za-z0-9+/]{50,}={0,2}")

# ADV-005 — hex string (40+ continuous hex chars)
_RE_HEX = re.compile(r"[0-9a-fA-F]{40,}")

# ADV-005 — URL-encoded percent sequences
_RE_PERCENT = re.compile(r"%[0-9a-fA-F]{2}")

# ADV-006 — positive / negative assertion words
_RE_POSITIVE = re.compile(r"\b(is|has|was|are|can)\b", re.IGNORECASE)
_RE_NEGATIVE = re.compile(r"\b(not|never|cannot|isn't|aren't)\b", re.IGNORECASE)

# ADV-007 — model inversion signals
_RE_INVERSION = re.compile(
    r"log.?prob"
    r"|logit"
    r"|embedding\s+vector"
    r"|model\s+weight"
    r"|probability\s+distribution"
    r"|softmax\s+output"
    r"|token\s+probability"
    r"|attention\s+weight",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ADVFinding:
    """A single adversarial-pattern finding."""

    check_id: str       # e.g. "ADV-003"
    severity: str       # CRITICAL / HIGH / MEDIUM / LOW / INFO
    title: str
    detail: str
    weight: int
    evidence: str       # sanitised snippet or description


@dataclass
class ADVResult:
    """Aggregated result for one analysed input."""

    findings: List[ADVFinding] = field(default_factory=list)
    risk_score: int = 0       # min(100, sum of weights for unique fired IDs)
    action: str = "ALLOW"     # ALLOW / REVIEW / BLOCK

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of this result."""
        return {
            "risk_score": self.risk_score,
            "action": self.action,
            "findings": [
                {
                    "check_id": f.check_id,
                    "severity": f.severity,
                    "title": f.title,
                    "detail": f.detail,
                    "weight": f.weight,
                    "evidence": f.evidence,
                }
                for f in self.findings
            ],
        }

    def summary(self) -> str:
        """One-line summary suitable for logging."""
        ids = ", ".join(f.check_id for f in self.findings) or "none"
        return (
            f"ADVResult action={self.action} "
            f"risk_score={self.risk_score} "
            f"findings=[{ids}]"
        )

    def by_severity(self) -> Dict[str, List[ADVFinding]]:
        """Group findings by severity label."""
        groups: Dict[str, List[ADVFinding]] = {}
        for f in self.findings:
            groups.setdefault(f.severity, []).append(f)
        return groups


# ---------------------------------------------------------------------------
# Internal helper utilities
# ---------------------------------------------------------------------------

def _sanitise(text: str, max_len: int = 120) -> str:
    """Return a short, printable excerpt of *text* safe for evidence fields."""
    excerpt = text[:max_len]
    # Replace non-printable / control chars with a visible placeholder
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "\ufffd", excerpt)
    if len(text) > max_len:
        cleaned += "..."
    return cleaned


def _char_similarity(a: str, b: str) -> float:
    """Return the fraction of characters in *a* that also appear in *b*.

    This is a lightweight, order-insensitive similarity measure used by
    ADV-001 to spot near-duplicate inputs.
    """
    if not a or not b:
        return 0.0
    set_b = set(b)
    common = sum(1 for ch in a if ch in set_b)
    return common / len(a)


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def _check_adv001(
    text: str,
    input_history: Optional[List[str]],
) -> Optional[ADVFinding]:
    """ADV-001 — Model extraction probe."""
    # Pattern-based signal: explicit variation / enumeration language
    enum_matches = _RE_EXTRACTION_ENUM.findall(text)

    # History-based signal: last 3 entries + current are all highly similar
    history_similar = False
    if input_history and len(input_history) >= 3:
        recent = input_history[-3:]
        # Every recent message must share > 80% character overlap with current
        if all(_char_similarity(prev, text) > 0.80 for prev in recent):
            history_similar = True

    if enum_matches or history_similar:
        evidence_parts: List[str] = []
        if enum_matches:
            evidence_parts.append(
                f"enumeration keywords: {list(set(enum_matches))[:5]}"
            )
        if history_similar:
            evidence_parts.append(
                "last 3 history entries show >80% character overlap with current input"
            )
        return ADVFinding(
            check_id="ADV-001",
            severity="HIGH",
            title="Model extraction probe detected",
            detail=(
                "Input contains systematic variation patterns or enumeration language "
                "consistent with boundary-testing / model extraction attempts."
            ),
            weight=_CHECK_WEIGHTS["ADV-001"],
            evidence="; ".join(evidence_parts),
        )
    return None


def _check_adv002(text: str) -> Optional[ADVFinding]:
    """ADV-002 — Membership inference probe."""
    match = _RE_MEMBERSHIP.search(text)
    if match:
        snippet = _sanitise(text[max(0, match.start() - 20): match.end() + 40])
        return ADVFinding(
            check_id="ADV-002",
            severity="HIGH",
            title="Membership inference probe detected",
            detail=(
                "Input requests verbatim reproduction of training data or asks "
                "whether specific content was part of the model's training set."
            ),
            weight=_CHECK_WEIGHTS["ADV-002"],
            evidence=f"matched pattern near: {snippet!r}",
        )
    return None


def _check_adv003(text: str) -> Optional[ADVFinding]:
    """ADV-003 — Backdoor trigger injection."""
    evidence_parts: List[str] = []

    # Zero-width and BOM characters
    zw_matches = _RE_ZERO_WIDTH.findall(text)
    if zw_matches:
        evidence_parts.append(
            f"zero-width/BOM characters found: {len(zw_matches)} occurrence(s)"
        )

    # Null byte
    if "\x00" in text:
        evidence_parts.append("null byte (\\x00) present")

    # Low-level ASCII control characters
    ctrl_matches = _RE_CTRL.findall(text)
    if ctrl_matches:
        evidence_parts.append(
            f"control characters found: {len(ctrl_matches)} occurrence(s)"
        )

    # Three or more consecutive Unicode format-category (Cf) characters
    consecutive_cf = 0
    max_consecutive_cf = 0
    for ch in text:
        if unicodedata.category(ch) == "Cf":
            consecutive_cf += 1
            max_consecutive_cf = max(max_consecutive_cf, consecutive_cf)
        else:
            consecutive_cf = 0
    if max_consecutive_cf >= 3:
        evidence_parts.append(
            f"run of {max_consecutive_cf} consecutive Unicode Cf (format) characters"
        )

    if evidence_parts:
        return ADVFinding(
            check_id="ADV-003",
            severity="CRITICAL",
            title="Backdoor trigger injection detected",
            detail=(
                "Input contains suspicious rare token sequences known to activate "
                "backdoor behaviours: zero-width characters, null bytes, control "
                "characters, or dense Unicode format characters."
            ),
            weight=_CHECK_WEIGHTS["ADV-003"],
            evidence="; ".join(evidence_parts),
        )
    return None


def _check_adv004(text: str) -> Optional[ADVFinding]:
    """ADV-004 — Adversarial text filter bypass (leetspeak / homoglyphs)."""
    alpha_chars = [ch for ch in text if ch.isalpha()]
    if not alpha_chars:
        return None

    leet_count = sum(1 for ch in text if ch in _LEET_CHARS)
    density = leet_count / len(alpha_chars)

    if density > 0.20:
        return ADVFinding(
            check_id="ADV-004",
            severity="HIGH",
            title="Adversarial text filter bypass detected",
            detail=(
                "High density of leetspeak substitutions or homoglyph characters "
                "detected, suggesting an attempt to evade content filters."
            ),
            weight=_CHECK_WEIGHTS["ADV-004"],
            evidence=(
                f"leet/homoglyph character density: "
                f"{leet_count}/{len(alpha_chars)} "
                f"({density:.1%} > 20% threshold)"
            ),
        )
    return None


def _check_adv005(text: str) -> Optional[ADVFinding]:
    """ADV-005 — Encoded payload in input."""
    evidence_parts: List[str] = []

    # Base64 blob (50+ continuous chars)
    b64_match = _RE_BASE64.search(text)
    if b64_match:
        evidence_parts.append(
            f"base64-like blob of length {len(b64_match.group())} found"
        )

    # Hex string (40+ continuous chars) — exclude git SHAs that are
    # exactly 40 chars and common UUID-hex patterns (32 hex chars)
    for hex_match in _RE_HEX.finditer(text):
        hex_str = hex_match.group()
        # Keep as evidence: length > 40 is unambiguously suspicious,
        # or exactly 40 chars that are NOT a standalone git-SHA-like token
        if len(hex_str) > 40:
            evidence_parts.append(
                f"hex-encoded string of length {len(hex_str)} found"
            )
            break
        # Exactly 40 hex chars: skip only when the token stands alone on
        # whitespace boundaries (i.e. looks like a bare git commit SHA).
        # Any other surrounding punctuation (=, :, /, etc.) is suspicious.
        left_is_ws = (
            hex_match.start() == 0
            or text[hex_match.start() - 1] in " \t\n\r"
        )
        right_is_ws = (
            hex_match.end() == len(text)
            or text[hex_match.end()] in " \t\n\r"
        )
        if left_is_ws and right_is_ws:
            # Standalone SHA — skip
            break
        evidence_parts.append(
            f"hex-encoded string of length {len(hex_str)} found"
        )
        break

    # URL encoding density
    percent_matches = _RE_PERCENT.findall(text)
    if percent_matches and len(text) > 0:
        density = len(percent_matches) / len(text)
        if density > 0.30:
            evidence_parts.append(
                f"URL-encoded percent density: "
                f"{len(percent_matches)}/{len(text)} "
                f"({density:.1%} > 30% threshold)"
            )

    if evidence_parts:
        return ADVFinding(
            check_id="ADV-005",
            severity="HIGH",
            title="Encoded payload in input detected",
            detail=(
                "Input contains base64 blobs, hex-encoded strings, or high-density "
                "URL encoding that may conceal a malicious payload."
            ),
            weight=_CHECK_WEIGHTS["ADV-005"],
            evidence="; ".join(evidence_parts),
        )
    return None


def _check_adv006(text: str) -> Optional[ADVFinding]:
    """ADV-006 — Extremely long contradictory context."""
    if len(text) <= 5000:
        return None

    pos_matches = _RE_POSITIVE.findall(text)
    neg_matches = _RE_NEGATIVE.findall(text)

    if len(pos_matches) > 5 and len(neg_matches) > 5:
        return ADVFinding(
            check_id="ADV-006",
            severity="MEDIUM",
            title="Long contradictory context detected",
            detail=(
                "Input exceeds 5000 characters and contains dense contradictory "
                "assertions, suggesting a context-flooding attack."
            ),
            weight=_CHECK_WEIGHTS["ADV-006"],
            evidence=(
                f"input length: {len(text)} chars; "
                f"positive assertion words: {len(pos_matches)}; "
                f"negative assertion words: {len(neg_matches)}"
            ),
        )
    return None


def _check_adv007(text: str) -> Optional[ADVFinding]:
    """ADV-007 — Model inversion attack indicators."""
    match = _RE_INVERSION.search(text)
    if match:
        snippet = _sanitise(text[max(0, match.start() - 20): match.end() + 40])
        return ADVFinding(
            check_id="ADV-007",
            severity="HIGH",
            title="Model inversion attack indicators detected",
            detail=(
                "Input requests internal model representations such as log-probabilities, "
                "logits, embedding vectors, weights, or probability distributions."
            ),
            weight=_CHECK_WEIGHTS["ADV-007"],
            evidence=f"matched pattern near: {snippet!r}",
        )
    return None


# ---------------------------------------------------------------------------
# Action thresholds
# ---------------------------------------------------------------------------

def _score_to_action(score: int) -> str:
    """Map a numeric risk score to a disposition string."""
    if score >= 60:
        return "BLOCK"
    if score >= 25:
        return "REVIEW"
    return "ALLOW"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(
    text: str,
    input_history: Optional[List[str]] = None,
) -> ADVResult:
    """Detect adversarial patterns in a single input, with optional history context.

    Parameters
    ----------
    text:
        The input string to analyse.
    input_history:
        Previous inputs in the session for pattern analysis (used by ADV-001).
        Pass the N most recent prior inputs, oldest first.

    Returns
    -------
    ADVResult
        Contains a list of findings, a clamped risk score (0–100), and an
        action recommendation: ALLOW / REVIEW / BLOCK.
    """
    findings: List[ADVFinding] = []

    # Run each check and collect non-None findings
    for candidate in (
        _check_adv001(text, input_history),
        _check_adv002(text),
        _check_adv003(text),
        _check_adv004(text),
        _check_adv005(text),
        _check_adv006(text),
        _check_adv007(text),
    ):
        if candidate is not None:
            findings.append(candidate)

    # Deduplicate by check_id (each ID contributes its weight at most once)
    seen_ids: set = set()
    total_weight = 0
    for f in findings:
        if f.check_id not in seen_ids:
            seen_ids.add(f.check_id)
            total_weight += f.weight

    risk_score = min(100, total_weight)
    action = _score_to_action(risk_score)

    return ADVResult(findings=findings, risk_score=risk_score, action=action)


def detect_many(texts: List[str]) -> List[ADVResult]:
    """Run :func:`detect` over a list of inputs without history context.

    Parameters
    ----------
    texts:
        Sequence of input strings to analyse.

    Returns
    -------
    List[ADVResult]
        One result per input, in the same order as *texts*.
    """
    return [detect(t) for t in texts]
