"""
Agent Tool Call Policy Enforcer
================================
Policy engine for enforcing security controls on LLM agent tool calls.

When an AI agent has access to tools (web search, code execution, file access,
API calls), it must operate within defined security boundaries. This module
provides a policy engine that evaluates tool call requests against:

  - Tool allowlist/denylist (which tools are permitted)
  - Argument validation rules (what argument values are allowed)
  - Rate limiting per tool per session
  - Argument redaction (prevent sensitive data from reaching tool logs)
  - Audit logging of all tool call decisions

Policy decisions:
  - ALLOW:  Tool call is permitted and logged
  - DENY:   Tool call is blocked (reason included in response)
  - REDACT: Tool call is allowed but specified arguments are masked in logs

Usage:
    from guardrails.policy_engine.tool_policy import (
        ToolPolicy,
        ToolCallRequest,
        ToolPolicyEngine,
        PolicyDecision,
    )

    policy = ToolPolicy(
        tool_name="web_search",
        allowed=True,
        rate_limit=10,           # max 10 calls per session
        blocked_arg_patterns={"query": [r"(?i)internal.corp", r"(?i)password"]},
    )
    engine = ToolPolicyEngine(policies=[policy])

    request = ToolCallRequest(tool_name="web_search", arguments={"query": "news today"})
    result = engine.evaluate(request)
    print(result.decision)  # "ALLOW"
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------

class PolicyDecision:
    ALLOW  = "ALLOW"
    DENY   = "DENY"
    REDACT = "REDACT"   # allow but mask specified args in logs


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolPolicy:
    """
    Security policy for a single tool.

    Attributes:
        tool_name:              Tool identifier (must match ToolCallRequest.tool_name).
        allowed:                Whether the tool is permitted at all (default True).
        rate_limit:             Maximum number of calls allowed per session (None = unlimited).
        required_args:          Argument names that must be present.
        blocked_arg_patterns:   Per-argument regex patterns that block the call if matched.
                                Format: {"arg_name": ["pattern1", "pattern2"]}
        redact_arg_patterns:    Per-argument regex patterns for log redaction only.
                                Matched values are replaced with "****" in audit logs.
        max_arg_length:         Maximum length of any single argument value (None = unlimited).
        deny_reason:            Custom denial message shown when allowed=False.
        notes:                  Documentation notes for this policy.
    """
    tool_name:              str
    allowed:                bool = True
    rate_limit:             Optional[int] = None
    required_args:          list[str] = field(default_factory=list)
    blocked_arg_patterns:   dict[str, list[str]] = field(default_factory=dict)
    redact_arg_patterns:    dict[str, list[str]] = field(default_factory=dict)
    max_arg_length:         Optional[int] = None
    deny_reason:            str = "Tool call denied by policy"
    notes:                  str = ""


@dataclass
class ToolCallRequest:
    """
    A request for an agent to call a tool.

    Attributes:
        tool_name:   Name of the tool to invoke.
        arguments:   Dict of argument name → value.
        session_id:  Optional session identifier for rate limiting.
        request_id:  Optional unique ID for this call (for audit correlation).
    """
    tool_name:  str
    arguments:  dict[str, Any] = field(default_factory=dict)
    session_id: str = "default"
    request_id: str = ""


@dataclass
class ToolPolicyResult:
    """
    Result of evaluating a tool call request against policies.

    Attributes:
        decision:       "ALLOW" | "DENY" | "REDACT"
        tool_name:      Tool that was evaluated.
        reason:         Human-readable explanation of the decision.
        sanitized_args: Arguments with sensitive values redacted (for logging).
        matched_rule:   Description of the rule that triggered DENY/REDACT.
        evaluated_at:   Unix timestamp of the evaluation.
    """
    decision:       str
    tool_name:      str
    reason:         str
    sanitized_args: dict[str, Any] = field(default_factory=dict)
    matched_rule:   Optional[str] = None
    evaluated_at:   float = field(default_factory=time.time)

    @property
    def allowed(self) -> bool:
        return self.decision in (PolicyDecision.ALLOW, PolicyDecision.REDACT)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision":       self.decision,
            "tool_name":      self.tool_name,
            "reason":         self.reason,
            "sanitized_args": self.sanitized_args,
            "matched_rule":   self.matched_rule,
            "evaluated_at":   self.evaluated_at,
            "allowed":        self.allowed,
        }


@dataclass
class ToolAuditEntry:
    """An audit log entry for a tool call decision."""
    request_id:   str
    session_id:   str
    tool_name:    str
    decision:     str
    reason:       str
    matched_rule: Optional[str]
    evaluated_at: float


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------

class ToolPolicyEngine:
    """
    Evaluates tool call requests against a set of ToolPolicy objects.

    Evaluation order:
      1. Check if the tool is on the allowlist (has a policy with allowed=True).
         If no policy exists, the tool is DENIED by default (deny-by-default posture).
      2. Check rate limit for the session.
      3. Check required arguments are present.
      4. Check argument length limits.
      5. Check blocked argument patterns (deny if any pattern matches).
      6. Check redact patterns (redact matching values in logs, but ALLOW).

    Args:
        policies:          List of ToolPolicy objects.
        default_allow:     If True, tools without a policy are ALLOWED (not recommended).
                           Default is False (deny-by-default).
        audit_log:         If True, maintain an in-memory audit log.
        max_audit_entries: Maximum audit log entries to retain (default: 10000).
    """

    def __init__(
        self,
        policies: Optional[list[ToolPolicy]] = None,
        default_allow: bool = False,
        audit_log: bool = True,
        max_audit_entries: int = 10000,
    ) -> None:
        self._policies: dict[str, ToolPolicy] = {}
        self._rate_counts: dict[str, dict[str, int]] = {}   # session_id → {tool_name → count}
        self._default_allow = default_allow
        self._audit: list[ToolAuditEntry] = []
        self._audit_enabled = audit_log
        self._max_audit = max_audit_entries

        for p in (policies or []):
            self.add_policy(p)

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def add_policy(self, policy: ToolPolicy) -> None:
        """Register or replace a policy for a tool."""
        self._policies[policy.tool_name] = policy

    def remove_policy(self, tool_name: str) -> bool:
        """Remove a policy. Returns True if the policy existed."""
        return self._policies.pop(tool_name, None) is not None

    def get_policy(self, tool_name: str) -> Optional[ToolPolicy]:
        """Return the policy for a tool, or None if not registered."""
        return self._policies.get(tool_name)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, request: ToolCallRequest) -> ToolPolicyResult:
        """
        Evaluate a tool call request against all policies.

        Args:
            request: The tool call request to evaluate.

        Returns:
            ToolPolicyResult with decision, reason, and sanitized arguments.
        """
        tool = request.tool_name
        policy = self._policies.get(tool)

        # Step 1: Tool not in policy registry
        if policy is None:
            if self._default_allow:
                result = ToolPolicyResult(
                    decision=PolicyDecision.ALLOW,
                    tool_name=tool,
                    reason="No policy registered; default_allow=True",
                    sanitized_args=dict(request.arguments),
                )
            else:
                result = ToolPolicyResult(
                    decision=PolicyDecision.DENY,
                    tool_name=tool,
                    reason=f"Tool '{tool}' is not in the policy registry (deny-by-default)",
                    sanitized_args=dict(request.arguments),
                    matched_rule="default_deny",
                )
            self._audit_result(request, result)
            return result

        # Step 2: Explicitly denied tool
        if not policy.allowed:
            result = ToolPolicyResult(
                decision=PolicyDecision.DENY,
                tool_name=tool,
                reason=policy.deny_reason,
                sanitized_args=dict(request.arguments),
                matched_rule=f"policy.allowed=False for '{tool}'",
            )
            self._audit_result(request, result)
            return result

        # Step 3: Rate limit check
        if policy.rate_limit is not None:
            session_counts = self._rate_counts.setdefault(request.session_id, {})
            current = session_counts.get(tool, 0)
            if current >= policy.rate_limit:
                result = ToolPolicyResult(
                    decision=PolicyDecision.DENY,
                    tool_name=tool,
                    reason=(
                        f"Rate limit exceeded for tool '{tool}': "
                        f"{current}/{policy.rate_limit} calls in this session"
                    ),
                    sanitized_args=dict(request.arguments),
                    matched_rule=f"rate_limit={policy.rate_limit}",
                )
                self._audit_result(request, result)
                return result

        # Step 4: Required arguments check
        missing = [a for a in policy.required_args if a not in request.arguments]
        if missing:
            result = ToolPolicyResult(
                decision=PolicyDecision.DENY,
                tool_name=tool,
                reason=f"Required argument(s) missing: {missing}",
                sanitized_args=dict(request.arguments),
                matched_rule=f"required_args={policy.required_args}",
            )
            self._audit_result(request, result)
            return result

        # Step 5: Argument length check
        if policy.max_arg_length is not None:
            for arg, val in request.arguments.items():
                if isinstance(val, str) and len(val) > policy.max_arg_length:
                    result = ToolPolicyResult(
                        decision=PolicyDecision.DENY,
                        tool_name=tool,
                        reason=(
                            f"Argument '{arg}' exceeds max length "
                            f"({len(val)} > {policy.max_arg_length})"
                        ),
                        sanitized_args=dict(request.arguments),
                        matched_rule=f"max_arg_length={policy.max_arg_length}",
                    )
                    self._audit_result(request, result)
                    return result

        # Step 6: Blocked argument patterns
        for arg, patterns in policy.blocked_arg_patterns.items():
            val = str(request.arguments.get(arg, ""))
            for pattern in patterns:
                try:
                    if re.search(pattern, val, re.IGNORECASE):
                        result = ToolPolicyResult(
                            decision=PolicyDecision.DENY,
                            tool_name=tool,
                            reason=(
                                f"Argument '{arg}' matched blocked pattern in tool '{tool}'"
                            ),
                            sanitized_args=self._sanitize_args(request.arguments, policy),
                            matched_rule=f"blocked_arg_patterns[{arg!r}]={pattern!r}",
                        )
                        self._audit_result(request, result)
                        return result
                except re.error:
                    continue

        # Step 7: Build sanitized args (with redactions applied)
        sanitized = self._sanitize_args(request.arguments, policy)
        needs_redact = sanitized != request.arguments

        # All checks passed — increment rate counter and allow
        if policy.rate_limit is not None:
            self._rate_counts.setdefault(request.session_id, {})[tool] = (
                self._rate_counts[request.session_id].get(tool, 0) + 1
            )

        decision = PolicyDecision.REDACT if needs_redact else PolicyDecision.ALLOW
        result = ToolPolicyResult(
            decision=decision,
            tool_name=tool,
            reason=(
                "Argument(s) redacted in audit log" if needs_redact
                else f"Tool '{tool}' call permitted"
            ),
            sanitized_args=sanitized,
        )
        self._audit_result(request, result)
        return result

    def _sanitize_args(
        self, arguments: dict[str, Any], policy: ToolPolicy
    ) -> dict[str, Any]:
        """Apply redact_arg_patterns to produce a sanitized copy of arguments."""
        sanitized = dict(arguments)
        for arg, patterns in policy.redact_arg_patterns.items():
            if arg not in sanitized:
                continue
            val = str(sanitized[arg])
            for pattern in patterns:
                try:
                    if re.search(pattern, val, re.IGNORECASE):
                        sanitized[arg] = "****[REDACTED]"
                        break
                except re.error:
                    continue
        return sanitized

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def _audit_result(self, request: ToolCallRequest, result: ToolPolicyResult) -> None:
        if not self._audit_enabled:
            return
        if len(self._audit) >= self._max_audit:
            self._audit.pop(0)
        self._audit.append(ToolAuditEntry(
            request_id=request.request_id,
            session_id=request.session_id,
            tool_name=result.tool_name,
            decision=result.decision,
            reason=result.reason,
            matched_rule=result.matched_rule,
            evaluated_at=result.evaluated_at,
        ))

    def get_audit_log(
        self,
        session_id: Optional[str] = None,
        decision_filter: Optional[str] = None,
    ) -> list[ToolAuditEntry]:
        """
        Return audit log entries, optionally filtered.

        Args:
            session_id:       If set, only return entries for this session.
            decision_filter:  If set, only return entries with this decision
                              ("ALLOW", "DENY", or "REDACT").

        Returns:
            List of ToolAuditEntry objects.
        """
        entries = self._audit
        if session_id:
            entries = [e for e in entries if e.session_id == session_id]
        if decision_filter:
            entries = [e for e in entries if e.decision == decision_filter]
        return entries

    def clear_audit_log(self) -> None:
        """Clear the in-memory audit log."""
        self._audit.clear()

    def reset_rate_counts(self, session_id: Optional[str] = None) -> None:
        """
        Reset rate limit counters.

        If session_id is given, resets only that session's counters.
        Otherwise, resets all sessions.
        """
        if session_id:
            self._rate_counts.pop(session_id, None)
        else:
            self._rate_counts.clear()

    def rate_count(self, session_id: str, tool_name: str) -> int:
        """Return the current call count for a tool in a session."""
        return self._rate_counts.get(session_id, {}).get(tool_name, 0)
