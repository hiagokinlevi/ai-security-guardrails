# action_auditor.py — AI Agent Tool Call Security Auditor
# Part of Cyber Port portfolio: github.com/hiagokinlevi/k1N-ai-security-guardrails
#
# License: Creative Commons Attribution 4.0 International (CC BY 4.0)
# https://creativecommons.org/licenses/by/4.0/
#
# Audits AI agent tool call sequences for seven security risk categories:
#   AGT-001  Filesystem boundary violations (path traversal / out-of-bounds)
#   AGT-002  Tool call rate abuse (burst threshold exceeded)
#   AGT-003  Recursive / stuck-loop patterns (same tool + same args, N times)
#   AGT-004  Unauthorized network requests (domain not in allowlist)
#   AGT-005  Code execution without confirmation (dangerous tool names)
#   AGT-006  Sensitive data in arguments (PII, credentials, API keys)
#   AGT-007  Excessive call chain depth (> 5 nested tool calls)

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Check weights — used to compute risk_score = min(100, sum of fired weights)
# ---------------------------------------------------------------------------
_CHECK_WEIGHTS: Dict[str, int] = {
    "AGT-001": 45,  # CRITICAL — filesystem boundary violation
    "AGT-002": 25,  # HIGH     — burst rate abuse
    "AGT-003": 25,  # HIGH     — recursive / stuck-loop pattern
    "AGT-004": 25,  # HIGH     — unauthorized network request
    "AGT-005": 40,  # CRITICAL — code execution without confirmation
    "AGT-006": 25,  # HIGH     — sensitive data (PII / credentials) in args
    "AGT-007": 15,  # MEDIUM   — excessive call chain depth
}

_CHECK_SEVERITY: Dict[str, str] = {
    "AGT-001": "CRITICAL",
    "AGT-002": "HIGH",
    "AGT-003": "HIGH",
    "AGT-004": "HIGH",
    "AGT-005": "CRITICAL",
    "AGT-006": "HIGH",
    "AGT-007": "MEDIUM",
}

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns
# ---------------------------------------------------------------------------

# AGT-001: filesystem tool name keywords
_FS_TOOL_KEYWORDS = (
    "read_file", "write_file", "open_file", "list_dir", "ls", "cat",
    "rm", "delete", "move", "copy", "touch", "mkdir", "read", "write", "file",
)

# AGT-001: sensitive root prefixes when no allowlist is configured
_SENSITIVE_ROOTS = (
    "/etc", "/var", "/proc", "/sys",
    r"C:\\Windows", r"C:\\Program Files",
)

# AGT-001: path-like value detection (absolute or relative indicators)
_PATH_START_RE = re.compile(r'^(?:/|~|\.\.?/|[A-Za-z]:\\)')

# AGT-004: network tool name keywords
_NET_TOOL_KEYWORDS = (
    "http_get", "http_post", "fetch", "request", "browse",
    "navigate", "web_search", "download", "curl",
)

# AGT-004: URL domain extraction
_URL_DOMAIN_RE = re.compile(r'https?://([^/\s]+)')

# AGT-005: dangerous execution tool name patterns (case-insensitive substring)
_EXEC_TOOL_KEYWORDS = (
    "execute", "run_code", "eval", "shell", "subprocess",
    "bash", "python_repl", "node", "run_script",
)

# AGT-006: PII / credential patterns
_EMAIL_RE    = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
_SSN_RE      = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
_CC_RE       = re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b')
# Constructed dynamically to avoid secret-scanning push protection on the literal
_AWS_KEY_RE  = re.compile("AKIA" + r"[0-9A-Z]{16}")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """Represents a single AI agent tool invocation."""
    call_id: str
    tool_name: str
    arguments: Dict[str, str]   # argument name → already-serialized string value
    timestamp_ms: int            # Unix epoch milliseconds
    depth: int                   # call-chain depth (0-indexed)


@dataclass
class AGTFinding:
    """A single security finding produced by one check."""
    check_id: str
    severity: str       # CRITICAL / HIGH / MEDIUM / LOW / INFO
    title: str
    detail: str
    weight: int
    call_id: str        # which tool call triggered this finding


@dataclass
class AGTResult:
    """Aggregated audit result for one tool call."""
    findings: List[AGTFinding] = field(default_factory=list)
    risk_score: int = 0    # min(100, sum of unique-check weights that fired)
    action: str = "ALLOW"  # ALLOW / WARN / BLOCK

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of the result."""
        return {
            "risk_score": self.risk_score,
            "action": self.action,
            "findings": [
                {
                    "check_id": f.check_id,
                    "severity": f.severity,
                    "title":    f.title,
                    "detail":   f.detail,
                    "weight":   f.weight,
                    "call_id":  f.call_id,
                }
                for f in self.findings
            ],
        }

    def summary(self) -> str:
        """One-line human-readable summary of the result."""
        checks = ", ".join(sorted({f.check_id for f in self.findings})) or "none"
        return (
            f"AGTResult action={self.action} risk_score={self.risk_score} "
            f"findings={len(self.findings)} checks=[{checks}]"
        )

    def by_severity(self) -> Dict[str, List[AGTFinding]]:
        """Group findings by severity level."""
        grouped: Dict[str, List[AGTFinding]] = {}
        for finding in self.findings:
            grouped.setdefault(finding.severity, []).append(finding)
        return grouped


# ---------------------------------------------------------------------------
# Internal helper: determine action from risk_score
# ---------------------------------------------------------------------------

def _action_from_score(score: int) -> str:
    """Map a numeric risk score to a disposition string."""
    if score >= 70:
        return "BLOCK"
    if score >= 30:
        return "WARN"
    return "ALLOW"


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------

def _check_agt001(
    call: ToolCall,
    allowed_working_dirs: Optional[List[str]],
) -> Optional[AGTFinding]:
    """AGT-001: Filesystem boundary / path traversal violation."""
    tool_lower = call.tool_name.lower()
    # Only examine filesystem-related tools
    if not any(kw in tool_lower for kw in _FS_TOOL_KEYWORDS):
        return None

    for arg_name, arg_value in call.arguments.items():
        # Only inspect values that look like filesystem paths
        if not _PATH_START_RE.match(arg_value):
            continue

        if allowed_working_dirs:
            # Allowlist mode: fire if the path does not start with any allowed dir.
            # Ensure we match a full directory component by checking for the separator
            # after the prefix (or an exact match), so /workspace2 is NOT under /workspace.
            def _under_allowed(path: str, allowed: str) -> bool:
                if path == allowed:
                    return True
                sep = "/" if not allowed.startswith("C:") else "\\"
                return path.startswith(allowed.rstrip(sep) + sep)

            if not any(_under_allowed(arg_value, d) for d in allowed_working_dirs):
                return AGTFinding(
                    check_id="AGT-001",
                    severity=_CHECK_SEVERITY["AGT-001"],
                    title="Filesystem boundary violation",
                    detail=(
                        f"Tool '{call.tool_name}' accessed path '{arg_value}' "
                        f"(argument '{arg_name}') outside allowed working directories."
                    ),
                    weight=_CHECK_WEIGHTS["AGT-001"],
                    call_id=call.call_id,
                )
        else:
            # No allowlist: fire on traversal sequences or sensitive root access
            if ".." in arg_value:
                return AGTFinding(
                    check_id="AGT-001",
                    severity=_CHECK_SEVERITY["AGT-001"],
                    title="Filesystem path traversal detected",
                    detail=(
                        f"Tool '{call.tool_name}' used path traversal in argument "
                        f"'{arg_name}': '{arg_value}'."
                    ),
                    weight=_CHECK_WEIGHTS["AGT-001"],
                    call_id=call.call_id,
                )
            for root in _SENSITIVE_ROOTS:
                if arg_value.startswith(root):
                    return AGTFinding(
                        check_id="AGT-001",
                        severity=_CHECK_SEVERITY["AGT-001"],
                        title="Sensitive filesystem root access",
                        detail=(
                            f"Tool '{call.tool_name}' accessed sensitive root "
                            f"'{root}' via argument '{arg_name}': '{arg_value}'."
                        ),
                        weight=_CHECK_WEIGHTS["AGT-001"],
                        call_id=call.call_id,
                    )
    return None


def _check_agt002(
    call: ToolCall,
    call_history: List[ToolCall],
    burst_limit: int,
    burst_window_ms: int,
) -> Optional[AGTFinding]:
    """AGT-002: Tool call burst rate abuse."""
    window_start = call.timestamp_ms - burst_window_ms
    # Count calls within the window, including the current call
    count = 1  # current call
    for hist_call in call_history:
        if hist_call.timestamp_ms >= window_start:
            count += 1

    if count > burst_limit:
        return AGTFinding(
            check_id="AGT-002",
            severity=_CHECK_SEVERITY["AGT-002"],
            title="Tool call burst rate exceeded",
            detail=(
                f"Detected {count} tool calls within a {burst_window_ms} ms window "
                f"(limit: {burst_limit}). Tool: '{call.tool_name}'."
            ),
            weight=_CHECK_WEIGHTS["AGT-002"],
            call_id=call.call_id,
        )
    return None


def _check_agt003(
    call: ToolCall,
    call_history: List[ToolCall],
) -> Optional[AGTFinding]:
    """AGT-003: Repetitive / stuck-loop pattern detection."""
    # Need at least 3 prior calls to evaluate
    if len(call_history) < 3:
        return None

    # Only look at the last 3 history calls
    last_three = call_history[-3:]

    # All 3 must share the same tool name as the current call
    if not all(h.tool_name == call.tool_name for h in last_three):
        return None

    # At least 2 of the 3 must have arguments identical to the current call
    identical_count = sum(
        1 for h in last_three if h.arguments == call.arguments
    )
    if identical_count >= 2:
        return AGTFinding(
            check_id="AGT-003",
            severity=_CHECK_SEVERITY["AGT-003"],
            title="Repetitive tool call loop detected",
            detail=(
                f"Tool '{call.tool_name}' called consecutively with identical or "
                f"near-identical arguments {identical_count + 1} times (including "
                f"current call). Possible stuck-loop condition."
            ),
            weight=_CHECK_WEIGHTS["AGT-003"],
            call_id=call.call_id,
        )
    return None


def _check_agt004(
    call: ToolCall,
    allowed_domains: Optional[List[str]],
) -> Optional[AGTFinding]:
    """AGT-004: Unauthorized network request domain."""
    # Only check network-related tool names
    tool_lower = call.tool_name.lower()
    if not any(kw in tool_lower for kw in _NET_TOOL_KEYWORDS):
        return None

    # Only relevant when an allowlist is configured
    if not allowed_domains:
        return None

    allowed_set = set(allowed_domains)

    for arg_name, arg_value in call.arguments.items():
        for domain_match in _URL_DOMAIN_RE.finditer(arg_value):
            domain = domain_match.group(1)
            if domain not in allowed_set:
                return AGTFinding(
                    check_id="AGT-004",
                    severity=_CHECK_SEVERITY["AGT-004"],
                    title="Unauthorized network domain accessed",
                    detail=(
                        f"Tool '{call.tool_name}' requested domain '{domain}' "
                        f"(argument '{arg_name}') which is not in the allowed "
                        f"domains list."
                    ),
                    weight=_CHECK_WEIGHTS["AGT-004"],
                    call_id=call.call_id,
                )
    return None


def _check_agt005(call: ToolCall) -> Optional[AGTFinding]:
    """AGT-005: Code execution tool invoked with a non-empty command."""
    tool_lower = call.tool_name.lower()
    if not any(kw in tool_lower for kw in _EXEC_TOOL_KEYWORDS):
        return None

    # Fire only when at least one argument value is non-empty
    if not any(v for v in call.arguments.values()):
        return None

    return AGTFinding(
        check_id="AGT-005",
        severity=_CHECK_SEVERITY["AGT-005"],
        title="Unconfirmed code execution tool invoked",
        detail=(
            f"Code execution tool '{call.tool_name}' was called with a non-empty "
            f"command. Argument values are redacted for security."
        ),
        weight=_CHECK_WEIGHTS["AGT-005"],
        call_id=call.call_id,
    )


def _check_agt006(call: ToolCall) -> Optional[AGTFinding]:
    """AGT-006: Sensitive data (PII / credentials) detected in arguments."""
    patterns = [_EMAIL_RE, _SSN_RE, _CC_RE, _AWS_KEY_RE]
    for arg_value in call.arguments.values():
        for pattern in patterns:
            if pattern.search(arg_value):
                return AGTFinding(
                    check_id="AGT-006",
                    severity=_CHECK_SEVERITY["AGT-006"],
                    title="Sensitive data detected in tool arguments",
                    detail=(
                        f"Tool '{call.tool_name}' argument contains a PII or "
                        f"credential pattern (email, SSN, credit card, or API key). "
                        f"Values are redacted."
                    ),
                    weight=_CHECK_WEIGHTS["AGT-006"],
                    call_id=call.call_id,
                )
    return None


def _check_agt007(call: ToolCall) -> Optional[AGTFinding]:
    """AGT-007: Excessive call chain depth."""
    if call.depth > 5:
        return AGTFinding(
            check_id="AGT-007",
            severity=_CHECK_SEVERITY["AGT-007"],
            title="Excessive tool call chain depth",
            detail=(
                f"Tool '{call.tool_name}' was invoked at chain depth {call.depth}, "
                f"which exceeds the maximum allowed depth of 5."
            ),
            weight=_CHECK_WEIGHTS["AGT-007"],
            call_id=call.call_id,
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def audit(
    call: ToolCall,
    call_history: Optional[List[ToolCall]] = None,
    allowed_working_dirs: Optional[List[str]] = None,
    allowed_domains: Optional[List[str]] = None,
    burst_limit: int = 10,
    burst_window_ms: int = 60000,
) -> AGTResult:
    """Audit a single tool call in the context of its call history.

    Parameters
    ----------
    call:
        The tool call being evaluated.
    call_history:
        Previous tool calls in chronological order (most recent last).
        The current call must NOT be included — it is added internally
        where needed.
    allowed_working_dirs:
        Filesystem paths the agent is permitted to access. When non-empty,
        any path outside these directories triggers AGT-001.
    allowed_domains:
        When non-empty, only these domains are allowed for network calls.
        An empty or None value disables AGT-004 domain checks.
    burst_limit:
        Maximum number of calls permitted within burst_window_ms.
    burst_window_ms:
        Sliding time window (milliseconds) used for burst detection.

    Returns
    -------
    AGTResult
        Aggregated findings, risk score, and recommended action.
    """
    history: List[ToolCall] = call_history or []

    findings: List[AGTFinding] = []

    # Run all seven checks and collect non-None findings
    for check_result in (
        _check_agt001(call, allowed_working_dirs),
        _check_agt002(call, history, burst_limit, burst_window_ms),
        _check_agt003(call, history),
        _check_agt004(call, allowed_domains),
        _check_agt005(call),
        _check_agt006(call),
        _check_agt007(call),
    ):
        if check_result is not None:
            findings.append(check_result)

    # Risk score: sum weights of unique check IDs that fired, capped at 100
    fired_checks = {f.check_id for f in findings}
    risk_score = min(100, sum(_CHECK_WEIGHTS[cid] for cid in fired_checks))

    return AGTResult(
        findings=findings,
        risk_score=risk_score,
        action=_action_from_score(risk_score),
    )


def audit_sequence(
    calls: List[ToolCall],
    allowed_working_dirs: Optional[List[str]] = None,
    allowed_domains: Optional[List[str]] = None,
    burst_limit: int = 10,
    burst_window_ms: int = 60000,
) -> List[AGTResult]:
    """Audit a full sequence of tool calls, providing call history for each.

    Parameters
    ----------
    calls:
        Ordered list of tool calls to audit (chronological order).
    allowed_working_dirs:
        See :func:`audit`.
    allowed_domains:
        See :func:`audit`.
    burst_limit:
        See :func:`audit`.
    burst_window_ms:
        See :func:`audit`.

    Returns
    -------
    List[AGTResult]
        One result per call, in the same order as the input list.
    """
    results: List[AGTResult] = []
    for index, call in enumerate(calls):
        # History for call[i] is calls[0..i-1]
        history = calls[:index]
        result = audit(
            call=call,
            call_history=history,
            allowed_working_dirs=allowed_working_dirs,
            allowed_domains=allowed_domains,
            burst_limit=burst_limit,
            burst_window_ms=burst_window_ms,
        )
        results.append(result)
    return results
