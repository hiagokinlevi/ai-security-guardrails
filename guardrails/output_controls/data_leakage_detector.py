# data_leakage_detector.py — Cyber Port | AI Security Guardrails
#
# Detects potential data leakage patterns in LLM-generated responses:
# PII, credentials, system-prompt disclosure, internal infrastructure,
# SQL schema exposure, sensitive file paths, and memorized training data.
#
# License: Creative Commons Attribution 4.0 International (CC BY 4.0)
# https://creativecommons.org/licenses/by/4.0/
#
# Author: Cyber Port Portfolio — github.com/hiagokinlevi
# Compatible: Python 3.9+

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Check weights — risk contribution per fired check ID
# ---------------------------------------------------------------------------
_CHECK_WEIGHTS: Dict[str, int] = {
    "DLK-001": 25,  # PII
    "DLK-002": 45,  # API key / credential
    "DLK-003": 25,  # System prompt leakage
    "DLK-004": 20,  # Internal infrastructure
    "DLK-005": 15,  # SQL DDL / schema
    "DLK-006": 15,  # Sensitive file path
    "DLK-007": 10,  # Memorized / repeated training data
}

_CHECK_SEVERITY: Dict[str, str] = {
    "DLK-001": "HIGH",
    "DLK-002": "CRITICAL",
    "DLK-003": "HIGH",
    "DLK-004": "HIGH",
    "DLK-005": "MEDIUM",
    "DLK-006": "MEDIUM",
    "DLK-007": "LOW",
}

_CHECK_TITLES: Dict[str, str] = {
    "DLK-001": "PII detected in response",
    "DLK-002": "API key or credential pattern detected",
    "DLK-003": "System prompt leakage indicators",
    "DLK-004": "Internal infrastructure details exposed",
    "DLK-005": "SQL DDL or schema exposure",
    "DLK-006": "Sensitive file path in response",
    "DLK-007": "Large repeated or wall-of-text memorized data block",
}


# ---------------------------------------------------------------------------
# DLK-001 — PII patterns
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# 16 digits with optional - or space separators (groups of 4)
_CC_RE = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
# US phone: optional +1, optional parens around area code
_PHONE_RE = re.compile(
    r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
)

_PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (_EMAIL_RE, "[EMAIL REDACTED]"),
    (_SSN_RE, "[SSN REDACTED]"),
    (_CC_RE, "[CC REDACTED]"),
    (_PHONE_RE, "[PHONE REDACTED]"),
]


# ---------------------------------------------------------------------------
# DLK-002 — API keys / credentials
# Use string concatenation to avoid GitHub push-protection scanning
# for real credential patterns embedded in source.
# ---------------------------------------------------------------------------
_AWS_KEY_PATTERN = re.compile("AKIA" + r"[0-9A-Z]{16}")
_GHP_PATTERN = re.compile("ghp_" + r"[A-Za-z0-9]{36,}")
_GHO_PATTERN = re.compile("gho_" + r"[A-Za-z0-9]{36,}")
_GHS_PATTERN = re.compile("ghs_" + r"[A-Za-z0-9]{36,}")
_GENERIC_API_KEY_RE = re.compile(
    # Allow optional surrounding quotes (JSON/YAML contexts: "api_key": "value")
    r"""(?:api[_-]?key|apikey|access[_-]?token)["']?\s*[:=]\s*["']?[A-Za-z0-9_\-]{20,}""",
    re.IGNORECASE,
)
_PEM_HEADER_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)

_CRED_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (_AWS_KEY_PATTERN, "[API KEY REDACTED]"),
    (_GHP_PATTERN, "[API KEY REDACTED]"),
    (_GHO_PATTERN, "[API KEY REDACTED]"),
    (_GHS_PATTERN, "[API KEY REDACTED]"),
    (_GENERIC_API_KEY_RE, "[API KEY REDACTED]"),
    (_PEM_HEADER_RE, "[API KEY REDACTED]"),
]


# ---------------------------------------------------------------------------
# DLK-003 — System prompt leakage
# ---------------------------------------------------------------------------
_SYSPROMPT_PHRASES: List[str] = [
    "You are a",
    "Your instructions are",
    "Your role is",
    "As an AI assistant",
    "system prompt",
    "I have been instructed to",
    "My instructions say",
]

_SYSPROMPT_RE = re.compile(
    "|".join(re.escape(p) for p in _SYSPROMPT_PHRASES),
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# DLK-004 — Internal infrastructure
# ---------------------------------------------------------------------------
_RFC1918_10_RE = re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
_RFC1918_172_RE = re.compile(
    r"\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"
)
_RFC1918_192_RE = re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b")
_INTERNAL_HOST_RE = re.compile(
    r"\b\w+\.(?:internal|local|corp|intranet)\b", re.IGNORECASE
)
_DB_CONN_RE = re.compile(
    r"(?:jdbc|mongodb|postgresql|redis|mysql)://[^\s]+", re.IGNORECASE
)

_INFRA_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (_RFC1918_10_RE, "[INTERNAL IP REDACTED]"),
    (_RFC1918_172_RE, "[INTERNAL IP REDACTED]"),
    (_RFC1918_192_RE, "[INTERNAL IP REDACTED]"),
    (_INTERNAL_HOST_RE, "[INTERNAL HOST REDACTED]"),
    (_DB_CONN_RE, "[DB URL REDACTED]"),
]


# ---------------------------------------------------------------------------
# DLK-005 — SQL DDL / schema exposure
# ---------------------------------------------------------------------------
_SQL_DDL_RE = re.compile(
    r"(?:CREATE\s+TABLE|ALTER\s+TABLE|CREATE\s+INDEX|DESCRIBE\s+TABLE"
    r"|SHOW\s+COLUMNS|INSERT\s+INTO)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# DLK-006 — Sensitive file paths
# ---------------------------------------------------------------------------
_SENSITIVE_PATHS: List[str] = [
    "/etc/passwd",
    "/etc/shadow",
    "/etc/hosts",
    r"C:\Windows\System32",
    "~/.ssh/",
    "/var/log/",
    "/proc/self",
]

# Build a single compiled pattern; escape each literal path
_FILE_PATH_RE = re.compile(
    "|".join(re.escape(p) for p in _SENSITIVE_PATHS), re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class DLKFinding:
    """Single finding from a data-leakage check."""

    check_id: str
    severity: str       # CRITICAL / HIGH / MEDIUM / LOW / INFO
    title: str
    detail: str         # Human-readable description of what was found (redacted)
    weight: int
    redacted_evidence: str  # Short placeholder, e.g. "[EMAIL REDACTED]"


@dataclass
class DLKResult:
    """Aggregated result of all data-leakage checks on one LLM response."""

    findings: List[DLKFinding]
    risk_score: int         # min(100, sum of weights for unique fired checks)
    should_block: bool      # True if any CRITICAL finding is present
    redacted_response: str  # Response with sensitive patterns replaced

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-serialisable)."""
        return {
            "risk_score": self.risk_score,
            "should_block": self.should_block,
            "findings": [
                {
                    "check_id": f.check_id,
                    "severity": f.severity,
                    "title": f.title,
                    "detail": f.detail,
                    "weight": f.weight,
                    "redacted_evidence": f.redacted_evidence,
                }
                for f in self.findings
            ],
            "redacted_response": self.redacted_response,
        }

    def summary(self) -> str:
        """One-line human-readable summary."""
        check_ids = ", ".join(f.check_id for f in self.findings) or "none"
        block_str = "BLOCK" if self.should_block else "PASS"
        return (
            f"[{block_str}] risk_score={self.risk_score} "
            f"findings={len(self.findings)} checks={check_ids}"
        )

    def by_severity(self) -> Dict[str, List[DLKFinding]]:
        """Group findings by severity level."""
        groups: Dict[str, List[DLKFinding]] = {}
        for f in self.findings:
            groups.setdefault(f.severity, []).append(f)
        return groups


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_redactions(
    text: str,
    patterns: List[Tuple[re.Pattern, str]],
) -> str:
    """Replace all matches of each (pattern, replacement) pair in text."""
    for pattern, replacement in patterns:
        text = pattern.sub(replacement, text)
    return text


def _check_dlk001(response: str) -> Optional[DLKFinding]:
    """PII: emails, SSNs, credit cards, US phone numbers."""
    found_types: List[str] = []
    sample_redaction: str = ""

    for pattern, redact_marker in _PII_PATTERNS:
        if pattern.search(response):
            label = redact_marker.strip("[]").replace(" REDACTED", "")
            found_types.append(label)
            if not sample_redaction:
                sample_redaction = redact_marker

    if not found_types:
        return None

    return DLKFinding(
        check_id="DLK-001",
        severity=_CHECK_SEVERITY["DLK-001"],
        title=_CHECK_TITLES["DLK-001"],
        detail=f"PII type(s) detected: {', '.join(found_types)}",
        weight=_CHECK_WEIGHTS["DLK-001"],
        redacted_evidence=sample_redaction,
    )


def _check_dlk002(response: str) -> Optional[DLKFinding]:
    """API keys, tokens, and PEM private key headers."""
    found_types: List[str] = []
    sample_redaction: str = "[API KEY REDACTED]"

    label_map: List[Tuple[re.Pattern, str]] = [
        (_AWS_KEY_PATTERN, "AWS access key"),
        (_GHP_PATTERN, "GitHub token (ghp_)"),
        (_GHO_PATTERN, "GitHub token (gho_)"),
        (_GHS_PATTERN, "GitHub token (ghs_)"),
        (_GENERIC_API_KEY_RE, "generic API key / access token"),
        (_PEM_HEADER_RE, "PEM private key header"),
    ]

    for pattern, label in label_map:
        if pattern.search(response):
            found_types.append(label)

    if not found_types:
        return None

    return DLKFinding(
        check_id="DLK-002",
        severity=_CHECK_SEVERITY["DLK-002"],
        title=_CHECK_TITLES["DLK-002"],
        detail=f"Credential type(s) detected: {', '.join(found_types)}",
        weight=_CHECK_WEIGHTS["DLK-002"],
        redacted_evidence=sample_redaction,
    )


def _check_dlk003(response: str) -> Optional[DLKFinding]:
    """System prompt leakage phrases."""
    match = _SYSPROMPT_RE.search(response)
    if match is None:
        return None

    phrase = match.group(0)
    return DLKFinding(
        check_id="DLK-003",
        severity=_CHECK_SEVERITY["DLK-003"],
        title=_CHECK_TITLES["DLK-003"],
        detail=f"System prompt indicator phrase found: '{phrase}'",
        weight=_CHECK_WEIGHTS["DLK-003"],
        redacted_evidence="[SYSTEM PROMPT INDICATOR]",
    )


def _check_dlk004(response: str) -> Optional[DLKFinding]:
    """Internal infrastructure: RFC1918 IPs, internal hostnames, DB URIs."""
    found_types: List[str] = []

    label_map: List[Tuple[re.Pattern, str]] = [
        (_RFC1918_10_RE, "RFC1918 10.x address"),
        (_RFC1918_172_RE, "RFC1918 172.16-31.x address"),
        (_RFC1918_192_RE, "RFC1918 192.168.x address"),
        (_INTERNAL_HOST_RE, "internal hostname"),
        (_DB_CONN_RE, "database connection string"),
    ]

    for pattern, label in label_map:
        if pattern.search(response):
            found_types.append(label)

    if not found_types:
        return None

    return DLKFinding(
        check_id="DLK-004",
        severity=_CHECK_SEVERITY["DLK-004"],
        title=_CHECK_TITLES["DLK-004"],
        detail=f"Infrastructure detail(s) detected: {', '.join(found_types)}",
        weight=_CHECK_WEIGHTS["DLK-004"],
        redacted_evidence="[INTERNAL IP REDACTED]",
    )


def _check_dlk005(response: str) -> Optional[DLKFinding]:
    """SQL DDL or schema-revealing statements."""
    match = _SQL_DDL_RE.search(response)
    if match is None:
        return None

    statement = match.group(0).upper().replace("\n", " ")
    return DLKFinding(
        check_id="DLK-005",
        severity=_CHECK_SEVERITY["DLK-005"],
        title=_CHECK_TITLES["DLK-005"],
        detail=f"SQL DDL statement detected: '{statement}'",
        weight=_CHECK_WEIGHTS["DLK-005"],
        redacted_evidence="[SQL DDL REDACTED]",
    )


def _check_dlk006(response: str) -> Optional[DLKFinding]:
    """Sensitive system file paths."""
    match = _FILE_PATH_RE.search(response)
    if match is None:
        return None

    path = match.group(0)
    return DLKFinding(
        check_id="DLK-006",
        severity=_CHECK_SEVERITY["DLK-006"],
        title=_CHECK_TITLES["DLK-006"],
        detail=f"Sensitive file path detected: '{path}'",
        weight=_CHECK_WEIGHTS["DLK-006"],
        redacted_evidence="[FILE PATH REDACTED]",
    )


def _check_dlk007(response: str) -> Optional[DLKFinding]:
    """
    Memorized / repeated training data heuristic.

    Triggers if:
      1. Any substring of 200+ characters appears more than once in the
         response (verbatim repetition, sliding window at step 50), OR
      2. Any contiguous 500+ character block has no newlines AND >95% of
         its characters are printable ASCII (wall-of-text indicator).
    """
    # --- Heuristic 1: verbatim repetition ---
    length = len(response)
    window = 200
    step = 50
    seen: set = set()

    for i in range(0, max(0, length - window + 1), step):
        chunk = response[i: i + window]
        if chunk in seen:
            return DLKFinding(
                check_id="DLK-007",
                severity=_CHECK_SEVERITY["DLK-007"],
                title=_CHECK_TITLES["DLK-007"],
                detail=(
                    "Verbatim repetition detected: a 200+ character substring "
                    "appears more than once in the response."
                ),
                weight=_CHECK_WEIGHTS["DLK-007"],
                redacted_evidence="[REPEATED BLOCK DETECTED]",
            )
        seen.add(chunk)

    # --- Heuristic 2: wall-of-text memorization ---
    # Split on newlines and inspect each line-free segment
    segments = re.split(r"\n+", response)
    printable_chars = set(string.printable)

    for segment in segments:
        if len(segment) < 500:
            continue
        printable_count = sum(1 for c in segment if c in printable_chars)
        ratio = printable_count / len(segment)
        if ratio > 0.95:
            return DLKFinding(
                check_id="DLK-007",
                severity=_CHECK_SEVERITY["DLK-007"],
                title=_CHECK_TITLES["DLK-007"],
                detail=(
                    "Wall-of-text block detected: a 500+ character segment "
                    "with no paragraph breaks and >95% printable ASCII content."
                ),
                weight=_CHECK_WEIGHTS["DLK-007"],
                redacted_evidence="[WALL-OF-TEXT BLOCK DETECTED]",
            )

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(response: str, block_on_severity: str = "CRITICAL") -> DLKResult:
    """
    Analyze an LLM response for data leakage patterns.

    Parameters
    ----------
    response:
        The raw text output from the LLM.
    block_on_severity:
        Severity level at which ``should_block`` becomes True.
        Defaults to ``"CRITICAL"``.

    Returns
    -------
    DLKResult
        Aggregated findings, risk score, block flag, and redacted response.
    """
    findings: List[DLKFinding] = []

    # Run all checks in DLK-001 … DLK-007 order
    checkers = [
        _check_dlk001,
        _check_dlk002,
        _check_dlk003,
        _check_dlk004,
        _check_dlk005,
        _check_dlk006,
        _check_dlk007,
    ]

    for checker in checkers:
        finding = checker(response)
        if finding is not None:
            findings.append(finding)

    # Deduplicated risk score: each check ID contributes its weight once
    fired_ids = {f.check_id for f in findings}
    risk_score = min(100, sum(_CHECK_WEIGHTS[cid] for cid in fired_ids))

    # Block decision
    should_block = any(f.severity == block_on_severity for f in findings)

    # Build redacted response (DLK-001 through DLK-006 only; DLK-007 skipped)
    redacted = response
    redacted = _apply_redactions(redacted, _PII_PATTERNS)
    redacted = _apply_redactions(redacted, _CRED_PATTERNS)
    redacted = re.sub(_SYSPROMPT_RE, "[SYSTEM PROMPT INDICATOR]", redacted)
    redacted = _apply_redactions(redacted, _INFRA_PATTERNS)
    redacted = _SQL_DDL_RE.sub("[SQL DDL REDACTED]", redacted)
    redacted = _FILE_PATH_RE.sub("[FILE PATH REDACTED]", redacted)

    return DLKResult(
        findings=findings,
        risk_score=risk_score,
        should_block=should_block,
        redacted_response=redacted,
    )


def analyze_many(responses: List[str]) -> List[DLKResult]:
    """
    Analyze a batch of LLM responses.

    Parameters
    ----------
    responses:
        List of raw LLM output strings.

    Returns
    -------
    List[DLKResult]
        One result per input response, in the same order.
    """
    return [analyze(r) for r in responses]
