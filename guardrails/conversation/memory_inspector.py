"""
Agent Memory & Context Inspector
===================================
Analyzes conversation history, agent scratchpad, and retrieved memory
for security anomalies: cross-session data leakage, accumulated sensitive
data, instruction persistence across turns, and memory poisoning.

Check IDs
----------
MEM-001   Sensitive data accumulated across turns (PII/credentials in memory)
MEM-002   Instructions persisting from earlier turns (system prompt fragments)
MEM-003   Cross-session contamination (content from different user contexts)
MEM-004   Memory size anomaly (conversation history exceeds threshold)
MEM-005   Repeated tool invocations with escalating permissions (privilege crawl)
MEM-006   Suspicious pattern in scratchpad (base64/encoded content)

Usage::

    from guardrails.conversation.memory_inspector import MemoryInspector, ConversationMemory

    memory = ConversationMemory(
        turns=[
            {"role": "user", "content": "My SSN is 123-45-6789"},
            {"role": "assistant", "content": "I've noted that."},
        ],
        scratchpad="Tool call: read_file('/etc/passwd')",
    )
    inspector = MemoryInspector()
    result = inspector.inspect(memory)
    for finding in result.findings:
        print(finding.to_dict())
"""

from __future__ import annotations

import base64
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------

class MemSeverity(Enum):
    """Severity levels for memory inspection findings, ordered by impact."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------

@dataclass
class MemFinding:
    """A single security finding produced by the memory inspector."""

    check_id: str
    severity: MemSeverity
    title: str
    detail: str
    evidence: str = ""
    remediation: str = ""

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Return a JSON-serialisable representation of this finding.

        Evidence is truncated to 200 characters to avoid leaking large
        amounts of sensitive material in downstream log sinks.
        """
        return {
            "check_id": self.check_id,
            "severity": self.severity.value,
            "title": self.title,
            "detail": self.detail,
            # Truncate evidence so findings are safe to emit to logs.
            "evidence": self.evidence[:200],
            "remediation": self.remediation,
        }

    def summary(self) -> str:
        """Return a compact one-line description of this finding."""
        return f"[{self.severity.value}] {self.check_id}: {self.title}"


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------

@dataclass
class ConversationMemory:
    """Snapshot of agent memory submitted for inspection."""

    # List of conversation turns; each dict must contain 'role' and 'content'.
    turns: List[Dict[str, str]] = field(default_factory=list)
    # Raw text content from the agent scratchpad (chain-of-thought, tool logs, etc.).
    scratchpad: str = ""
    # Opaque identifier for the current session, used to detect cross-session leakage.
    session_id: str = ""
    # Text injected from a retrieval system (RAG, vector store, etc.).
    retrieved_context: str = ""
    # Structured record of tool calls made during this conversation.
    tool_calls: List[Dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    @property
    def all_content(self) -> str:
        """Concatenate every piece of content visible to the inspector."""
        parts: List[str] = []
        for turn in self.turns:
            content = turn.get("content", "")
            if content:
                parts.append(content)
        if self.scratchpad:
            parts.append(self.scratchpad)
        if self.retrieved_context:
            parts.append(self.retrieved_context)
        return " ".join(parts)

    @property
    def user_content(self) -> str:
        """Return only content produced by the 'user' role."""
        return " ".join(
            t.get("content", "")
            for t in self.turns
            if t.get("role") == "user" and t.get("content", "")
        )

    @property
    def assistant_content(self) -> str:
        """Return only content produced by the 'assistant' role."""
        return " ".join(
            t.get("content", "")
            for t in self.turns
            if t.get("role") == "assistant" and t.get("content", "")
        )


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class MemInspectionResult:
    """Aggregated output from a single memory inspection run."""

    session_id: str
    findings: List[MemFinding] = field(default_factory=list)
    risk_score: int = 0
    turns_analyzed: int = 0
    is_flagged: bool = False
    generated_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Aggregate properties
    # ------------------------------------------------------------------

    @property
    def total_findings(self) -> int:
        """Total number of findings regardless of severity."""
        return len(self.findings)

    @property
    def critical_findings(self) -> int:
        """Number of CRITICAL severity findings."""
        return sum(1 for f in self.findings if f.severity == MemSeverity.CRITICAL)

    @property
    def high_findings(self) -> int:
        """Number of HIGH severity findings."""
        return sum(1 for f in self.findings if f.severity == MemSeverity.HIGH)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def findings_by_check(self, check_id: str) -> List[MemFinding]:
        """Return all findings that match a specific check identifier."""
        return [f for f in self.findings if f.check_id == check_id]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a brief human-readable summary of the inspection result."""
        flag = "FLAGGED" if self.is_flagged else "CLEAN"
        return (
            f"Session {self.session_id!r} [{flag}] "
            f"risk={self.risk_score} findings={self.total_findings} "
            f"(critical={self.critical_findings} high={self.high_findings}) "
            f"turns={self.turns_analyzed}"
        )

    def to_dict(self) -> Dict:
        """Return a JSON-serialisable representation of the full result."""
        return {
            "session_id": self.session_id,
            "risk_score": self.risk_score,
            "turns_analyzed": self.turns_analyzed,
            "is_flagged": self.is_flagged,
            "generated_at": self.generated_at,
            "total_findings": self.total_findings,
            "critical_findings": self.critical_findings,
            "high_findings": self.high_findings,
            "findings": [f.to_dict() for f in self.findings],
        }


# ---------------------------------------------------------------------------
# Check weights  (used to accumulate the risk score)
# ---------------------------------------------------------------------------

_CHECK_WEIGHTS: Dict[str, int] = {
    "MEM-001": 40,
    "MEM-002": 35,
    "MEM-003": 45,
    "MEM-004": 15,
    "MEM-005": 35,
    "MEM-006": 30,
}

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# MEM-001 — PII / credential patterns
_RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_RE_CREDIT_CARD = re.compile(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b")
_RE_PASSWORD = re.compile(
    r"(password|passwd|pwd)\s*[=:]\s*\S+", re.IGNORECASE
)
_RE_API_KEY = re.compile(
    r"(api[_-]?key|access[_-]?token)\s*[=:]\s*\S{16,}", re.IGNORECASE
)

# MEM-002 — Instruction persistence (system-prompt fragments in assistant turns)
_RE_INSTRUCTION = re.compile(
    r"(system\s+prompt|initial\s+instruction|you\s+are\s+an?\s+AI"
    r"|your\s+role\s+is|your\s+instructions\s+are)",
    re.IGNORECASE,
)

# MEM-003 — Cross-session contamination signals
_RE_CROSS_SESSION = re.compile(
    r"(previous\s+(session|conversation|user)|from\s+another\s+(session|user)"
    r"|other\s+user('s)?)",
    re.IGNORECASE,
)

# MEM-006 — Base64 blobs in the scratchpad
_RE_BASE64 = re.compile(r"[A-Za-z0-9+/]{30,}={0,2}")

# Suspicious strings that, if present in a decoded base64 payload, escalate severity
_SUSPICIOUS_DECODED = re.compile(
    r"(exec|eval|import|system|passwd|shadow|/etc/|cmd|powershell|base64)",
    re.IGNORECASE,
)

# Privilege-escalation tool name keywords (MEM-005)
_ESCALATION_KEYWORDS = {"sudo", "admin", "root", "escalate", "bypass"}


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------

class MemoryInspector:
    """Inspect a :class:`ConversationMemory` snapshot for security anomalies.

    Parameters
    ----------
    max_turns:
        Maximum number of turns allowed before MEM-004 fires (default: 50).
    enabled_checks:
        Explicit allow-list of check IDs to run.  When *None* (default), all
        six checks are executed.
    """

    def __init__(
        self,
        max_turns: int = 50,
        enabled_checks: Optional[List[str]] = None,
    ) -> None:
        self.max_turns = max_turns
        # Normalise to a set for O(1) membership tests; None means "all".
        self._enabled: Optional[set] = (
            set(enabled_checks) if enabled_checks is not None else None
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def inspect(self, memory: ConversationMemory) -> MemInspectionResult:
        """Run all enabled checks against *memory* and return a result object."""
        result = MemInspectionResult(
            session_id=memory.session_id,
            turns_analyzed=len(memory.turns),
        )

        # Run each check in a deterministic order so findings are predictable.
        for check_id, runner in [
            ("MEM-001", self._check_mem001),
            ("MEM-002", self._check_mem002),
            ("MEM-003", self._check_mem003),
            ("MEM-004", self._check_mem004),
            ("MEM-005", self._check_mem005),
            ("MEM-006", self._check_mem006),
        ]:
            if self._is_enabled(check_id):
                finding = runner(memory)
                if finding is not None:
                    result.findings.append(finding)
                    result.risk_score += _CHECK_WEIGHTS.get(check_id, 0)

        result.is_flagged = result.risk_score > 0
        return result

    def inspect_many(
        self, memories: List[ConversationMemory]
    ) -> List[MemInspectionResult]:
        """Inspect a batch of memory snapshots and return one result per snapshot."""
        return [self.inspect(m) for m in memories]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_enabled(self, check_id: str) -> bool:
        """Return True if *check_id* should be executed."""
        if self._enabled is None:
            return True
        return check_id in self._enabled

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_mem001(
        self, memory: ConversationMemory
    ) -> Optional[MemFinding]:
        """MEM-001: Sensitive data (PII / credentials) accumulated across turns."""
        content = memory.all_content
        hits: List[str] = []

        # Test each of the four pattern categories and collect evidence.
        ssn_match = _RE_SSN.search(content)
        if ssn_match:
            hits.append(f"SSN pattern: {ssn_match.group()}")

        cc_match = _RE_CREDIT_CARD.search(content)
        if cc_match:
            hits.append(f"Credit card pattern: {cc_match.group()}")

        pw_match = _RE_PASSWORD.search(content)
        if pw_match:
            hits.append(f"Password assignment: {pw_match.group()}")

        ak_match = _RE_API_KEY.search(content)
        if ak_match:
            hits.append(f"API key / token: {ak_match.group()}")

        if not hits:
            return None

        # Determine worst-case severity: credentials are CRITICAL, PII is HIGH.
        has_credential = any(
            "Password" in h or "API key" in h for h in hits
        )
        severity = MemSeverity.CRITICAL if has_credential else MemSeverity.HIGH

        return MemFinding(
            check_id="MEM-001",
            severity=severity,
            title="Sensitive data accumulated in conversation memory",
            detail=(
                f"Found {len(hits)} sensitive pattern(s) in conversation history. "
                "Sensitive data should not persist in agent memory between turns."
            ),
            evidence="; ".join(hits),
            remediation=(
                "Redact or tokenise sensitive values before storing turns. "
                "Consider implementing a memory scrubbing pipeline that strips "
                "PII and credentials before each turn is persisted."
            ),
        )

    def _check_mem002(
        self, memory: ConversationMemory
    ) -> Optional[MemFinding]:
        """MEM-002: System-prompt / instruction fragments in assistant turns."""
        content = memory.assistant_content
        match = _RE_INSTRUCTION.search(content)
        if match is None:
            return None

        return MemFinding(
            check_id="MEM-002",
            severity=MemSeverity.HIGH,
            title="Instruction persistence detected in assistant turns",
            detail=(
                "Assistant content contains fragments that resemble system-prompt "
                "language, suggesting earlier instructions are leaking across turns."
            ),
            evidence=match.group(),
            remediation=(
                "Ensure system-prompt content is never echoed back by the assistant. "
                "Apply an output filter that redacts instruction-style phrases before "
                "persisting assistant turns to the conversation buffer."
            ),
        )

    def _check_mem003(
        self, memory: ConversationMemory
    ) -> Optional[MemFinding]:
        """MEM-003: Cross-session contamination signals."""
        content = memory.all_content
        match = _RE_CROSS_SESSION.search(content)
        if match is None:
            return None

        return MemFinding(
            check_id="MEM-003",
            severity=MemSeverity.CRITICAL,
            title="Cross-session contamination detected",
            detail=(
                "Content references data from a different user session or user, "
                "indicating possible session isolation failure or memory poisoning."
            ),
            evidence=match.group(),
            remediation=(
                "Enforce strict session isolation: each conversation context must be "
                "initialised with a clean memory store tied exclusively to the current "
                "session identifier.  Audit retrieval systems for cross-user leakage."
            ),
        )

    def _check_mem004(
        self, memory: ConversationMemory
    ) -> Optional[MemFinding]:
        """MEM-004: Conversation history exceeds configured turn threshold."""
        if len(memory.turns) <= self.max_turns:
            return None

        excess = len(memory.turns) - self.max_turns
        return MemFinding(
            check_id="MEM-004",
            severity=MemSeverity.LOW,
            title="Memory size anomaly — conversation history exceeds threshold",
            detail=(
                f"Conversation contains {len(memory.turns)} turns, which is "
                f"{excess} turn(s) above the configured maximum of {self.max_turns}. "
                "Excessively long histories increase the risk of sensitive data "
                "accumulation and prompt-injection surface."
            ),
            evidence=f"turns={len(memory.turns)} max_turns={self.max_turns}",
            remediation=(
                "Implement a sliding-window or summarisation strategy to keep "
                "conversation history within acceptable bounds.  Archive or discard "
                f"turns beyond the {self.max_turns}-turn limit."
            ),
        )

    def _check_mem005(
        self, memory: ConversationMemory
    ) -> Optional[MemFinding]:
        """MEM-005: Privilege crawl — repeated or escalating tool invocations."""
        tool_calls = memory.tool_calls
        if not tool_calls:
            return None

        # Count invocations per tool name.
        name_counts: Dict[str, int] = {}
        for call in tool_calls:
            name = call.get("name", "")
            if name:
                name_counts[name] = name_counts.get(name, 0) + 1

        repeated = [n for n, c in name_counts.items() if c > 3]

        # Check whether any tool name contains an escalation keyword.
        escalating: List[str] = []
        for name in name_counts:
            name_lower = name.lower()
            for keyword in _ESCALATION_KEYWORDS:
                if keyword in name_lower:
                    escalating.append(name)
                    break

        if not repeated and not escalating:
            return None

        evidence_parts: List[str] = []
        if repeated:
            evidence_parts.append(
                "Repeated tools (>3 calls): " + ", ".join(
                    f"{n}({name_counts[n]})" for n in repeated
                )
            )
        if escalating:
            evidence_parts.append(
                "Escalation keyword tools: " + ", ".join(escalating)
            )

        # Escalation keywords are more dangerous than simple repetition.
        severity = MemSeverity.CRITICAL if escalating else MemSeverity.HIGH

        return MemFinding(
            check_id="MEM-005",
            severity=severity,
            title="Privilege crawl detected in tool invocation history",
            detail=(
                "Tool call history shows signs of privilege escalation: either a "
                "tool was invoked more than three times (possible automation abuse) "
                "or tool names contain escalation-related keywords."
            ),
            evidence="; ".join(evidence_parts),
            remediation=(
                "Apply per-tool rate limits and require explicit user re-authorisation "
                "for privileged operations.  Deny any tool whose name matches "
                "escalation patterns (sudo, admin, root, escalate, bypass) unless "
                "explicitly allow-listed."
            ),
        )

    def _check_mem006(
        self, memory: ConversationMemory
    ) -> Optional[MemFinding]:
        """MEM-006: Suspicious base64-encoded content in the scratchpad."""
        scratchpad = memory.scratchpad
        if not scratchpad:
            return None

        candidates = _RE_BASE64.findall(scratchpad)
        suspicious: List[str] = []

        for candidate in candidates:
            # Normalise padding before attempting decode.
            padded = candidate + "=" * ((-len(candidate)) % 4)
            try:
                decoded_bytes = base64.b64decode(padded, validate=True)
                decoded_text = decoded_bytes.decode("utf-8", errors="replace")
                if _SUSPICIOUS_DECODED.search(decoded_text):
                    # Keep the raw candidate (not the decoded text) as evidence
                    # to avoid embedding potentially harmful content directly.
                    suspicious.append(candidate[:60])
            except Exception:
                # Invalid base64 or undecodable bytes — not suspicious by itself.
                pass

        if not suspicious:
            return None

        return MemFinding(
            check_id="MEM-006",
            severity=MemSeverity.HIGH,
            title="Suspicious base64-encoded content detected in scratchpad",
            detail=(
                f"Found {len(suspicious)} base64 segment(s) in the scratchpad whose "
                "decoded content matches patterns associated with code execution, "
                "system access, or obfuscated payloads."
            ),
            evidence="Base64 candidates: " + "; ".join(suspicious),
            remediation=(
                "Treat encoded blobs in agent scratchpads as untrusted.  Decode and "
                "scan all base64 segments before execution.  Block scratchpad content "
                "that decodes to executable commands or sensitive path references."
            ),
        )
