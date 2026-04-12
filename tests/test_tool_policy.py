"""
Tests for guardrails/policy_engine/tool_policy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from guardrails.policy_engine.tool_policy import (
    PolicyDecision,
    ToolAuditEntry,
    ToolCallRequest,
    ToolPolicy,
    ToolPolicyEngine,
    ToolPolicyResult,
)


# ===========================================================================
# ToolPolicyResult
# ===========================================================================

class TestToolPolicyResult:
    def test_allowed_for_allow_decision(self):
        r = ToolPolicyResult(decision="ALLOW", tool_name="t", reason="ok")
        assert r.allowed

    def test_allowed_for_redact_decision(self):
        r = ToolPolicyResult(decision="REDACT", tool_name="t", reason="redacted")
        assert r.allowed

    def test_not_allowed_for_deny(self):
        r = ToolPolicyResult(decision="DENY", tool_name="t", reason="blocked")
        assert not r.allowed

    def test_to_dict_has_required_keys(self):
        r = ToolPolicyResult(decision="ALLOW", tool_name="t", reason="ok")
        d = r.to_dict()
        for key in ("decision", "tool_name", "reason", "allowed"):
            assert key in d


# ===========================================================================
# ToolPolicyEngine — default deny
# ===========================================================================

class TestDefaultDeny:
    engine = ToolPolicyEngine(default_allow=False)

    def test_unregistered_tool_denied(self):
        req = ToolCallRequest(tool_name="unknown_tool", arguments={})
        result = self.engine.evaluate(req)
        assert result.decision == PolicyDecision.DENY

    def test_deny_reason_mentions_policy(self):
        req = ToolCallRequest(tool_name="unknown_tool")
        result = self.engine.evaluate(req)
        assert "policy" in result.reason.lower() or "registry" in result.reason.lower()


class TestDefaultAllow:
    engine = ToolPolicyEngine(default_allow=True)

    def test_unregistered_tool_allowed(self):
        req = ToolCallRequest(tool_name="any_tool")
        result = self.engine.evaluate(req)
        assert result.decision == PolicyDecision.ALLOW


# ===========================================================================
# Policy management
# ===========================================================================

class TestPolicyManagement:
    def test_add_and_get_policy(self):
        engine = ToolPolicyEngine()
        policy = ToolPolicy(tool_name="web_search", allowed=True)
        engine.add_policy(policy)
        assert engine.get_policy("web_search") is not None

    def test_remove_policy(self):
        engine = ToolPolicyEngine()
        engine.add_policy(ToolPolicy(tool_name="x"))
        assert engine.remove_policy("x") is True
        assert engine.get_policy("x") is None

    def test_remove_nonexistent_returns_false(self):
        engine = ToolPolicyEngine()
        assert engine.remove_policy("nonexistent") is False

    def test_constructor_with_policies_list(self):
        engine = ToolPolicyEngine(policies=[
            ToolPolicy(tool_name="a"),
            ToolPolicy(tool_name="b"),
        ])
        assert engine.get_policy("a") is not None
        assert engine.get_policy("b") is not None


# ===========================================================================
# Allowed tool
# ===========================================================================

class TestAllowedTool:
    engine = ToolPolicyEngine(policies=[
        ToolPolicy(tool_name="web_search", allowed=True)
    ])

    def test_allowed_tool_returns_allow(self):
        req = ToolCallRequest(tool_name="web_search")
        result = self.engine.evaluate(req)
        assert result.decision == PolicyDecision.ALLOW


class TestDeniedTool:
    engine = ToolPolicyEngine(policies=[
        ToolPolicy(tool_name="exec_code", allowed=False,
                   deny_reason="Code execution is prohibited")
    ])

    def test_explicitly_denied_tool_returns_deny(self):
        req = ToolCallRequest(tool_name="exec_code")
        result = self.engine.evaluate(req)
        assert result.decision == PolicyDecision.DENY

    def test_deny_reason_in_result(self):
        req = ToolCallRequest(tool_name="exec_code")
        result = self.engine.evaluate(req)
        assert "prohibited" in result.reason


# ===========================================================================
# Rate limiting
# ===========================================================================

class TestRateLimit:
    def _engine(self, limit: int) -> ToolPolicyEngine:
        return ToolPolicyEngine(policies=[
            ToolPolicy(tool_name="search", allowed=True, rate_limit=limit)
        ])

    def test_within_limit_allowed(self):
        engine = self._engine(3)
        for _ in range(3):
            r = engine.evaluate(ToolCallRequest(tool_name="search", session_id="s1"))
            assert r.decision == PolicyDecision.ALLOW

    def test_exceeds_limit_denied(self):
        engine = self._engine(2)
        engine.evaluate(ToolCallRequest(tool_name="search", session_id="s1"))
        engine.evaluate(ToolCallRequest(tool_name="search", session_id="s1"))
        r = engine.evaluate(ToolCallRequest(tool_name="search", session_id="s1"))
        assert r.decision == PolicyDecision.DENY

    def test_different_sessions_independent(self):
        engine = self._engine(1)
        r1 = engine.evaluate(ToolCallRequest(tool_name="search", session_id="s1"))
        r2 = engine.evaluate(ToolCallRequest(tool_name="search", session_id="s2"))
        assert r1.decision == PolicyDecision.ALLOW
        assert r2.decision == PolicyDecision.ALLOW

    def test_rate_count_tracks_calls(self):
        engine = self._engine(10)
        for _ in range(3):
            engine.evaluate(ToolCallRequest(tool_name="search", session_id="test"))
        assert engine.rate_count("test", "search") == 3

    def test_reset_rate_counts_for_session(self):
        engine = self._engine(2)
        engine.evaluate(ToolCallRequest(tool_name="search", session_id="s"))
        engine.evaluate(ToolCallRequest(tool_name="search", session_id="s"))
        engine.reset_rate_counts("s")
        r = engine.evaluate(ToolCallRequest(tool_name="search", session_id="s"))
        assert r.decision == PolicyDecision.ALLOW

    def test_reset_all_rate_counts(self):
        engine = self._engine(1)
        engine.evaluate(ToolCallRequest(tool_name="search", session_id="s1"))
        engine.evaluate(ToolCallRequest(tool_name="search", session_id="s2"))
        engine.reset_rate_counts()
        # Both sessions should be allowed again
        r1 = engine.evaluate(ToolCallRequest(tool_name="search", session_id="s1"))
        r2 = engine.evaluate(ToolCallRequest(tool_name="search", session_id="s2"))
        assert r1.decision == PolicyDecision.ALLOW
        assert r2.decision == PolicyDecision.ALLOW


# ===========================================================================
# Call depth
# ===========================================================================

class TestCallDepth:
    engine = ToolPolicyEngine(policies=[
        ToolPolicy(tool_name="web_search", allowed=True, max_call_depth=2)
    ])

    def test_depth_below_limit_allowed(self):
        r = self.engine.evaluate(ToolCallRequest(tool_name="web_search", call_depth=1))
        assert r.decision == PolicyDecision.ALLOW

    def test_depth_equal_to_limit_allowed(self):
        r = self.engine.evaluate(ToolCallRequest(tool_name="web_search", call_depth=2))
        assert r.decision == PolicyDecision.ALLOW

    def test_depth_above_limit_denied(self):
        r = self.engine.evaluate(ToolCallRequest(tool_name="web_search", call_depth=3))
        assert r.decision == PolicyDecision.DENY
        assert "depth" in r.reason.lower()
        assert r.matched_rule == "max_call_depth=2"

    def test_negative_depth_denied(self):
        r = self.engine.evaluate(ToolCallRequest(tool_name="web_search", call_depth=-1))
        assert r.decision == PolicyDecision.DENY
        assert r.matched_rule == "invalid_call_depth"

    def test_non_integer_depth_denied(self):
        r = self.engine.evaluate(ToolCallRequest(tool_name="web_search", call_depth="2"))  # type: ignore[arg-type]
        assert r.decision == PolicyDecision.DENY
        assert r.matched_rule == "invalid_call_depth"


# ===========================================================================
# Required arguments
# ===========================================================================

class TestRequiredArgs:
    engine = ToolPolicyEngine(policies=[
        ToolPolicy(tool_name="query", allowed=True, required_args=["q", "lang"])
    ])

    def test_all_required_args_present(self):
        r = self.engine.evaluate(ToolCallRequest(tool_name="query", arguments={"q": "hello", "lang": "en"}))
        assert r.decision == PolicyDecision.ALLOW

    def test_missing_required_arg_denied(self):
        r = self.engine.evaluate(ToolCallRequest(tool_name="query", arguments={"q": "hello"}))
        assert r.decision == PolicyDecision.DENY
        assert "lang" in r.reason


# ===========================================================================
# Argument length
# ===========================================================================

class TestArgLength:
    engine = ToolPolicyEngine(policies=[
        ToolPolicy(tool_name="search", allowed=True, max_arg_length=100)
    ])

    def test_within_length_allowed(self):
        r = self.engine.evaluate(ToolCallRequest(
            tool_name="search", arguments={"q": "short query"}
        ))
        assert r.decision == PolicyDecision.ALLOW

    def test_exceeds_length_denied(self):
        r = self.engine.evaluate(ToolCallRequest(
            tool_name="search", arguments={"q": "x" * 101}
        ))
        assert r.decision == PolicyDecision.DENY
        assert "max" in r.reason.lower() or "length" in r.reason.lower()


# ===========================================================================
# Blocked argument patterns
# ===========================================================================

class TestBlockedArgPatterns:
    engine = ToolPolicyEngine(policies=[
        ToolPolicy(
            tool_name="web_search",
            allowed=True,
            blocked_arg_patterns={
                "query": [r"(?i)internal\.corp", r"(?i)\bpassword\b"]
            },
        )
    ])

    def test_safe_query_allowed(self):
        r = self.engine.evaluate(ToolCallRequest(
            tool_name="web_search", arguments={"query": "weather today"}
        ))
        assert r.decision == PolicyDecision.ALLOW

    def test_blocked_pattern_denies(self):
        r = self.engine.evaluate(ToolCallRequest(
            tool_name="web_search", arguments={"query": "search internal.corp secrets"}
        ))
        assert r.decision == PolicyDecision.DENY

    def test_password_pattern_denies(self):
        r = self.engine.evaluate(ToolCallRequest(
            tool_name="web_search", arguments={"query": "reset password for admin"}
        ))
        assert r.decision == PolicyDecision.DENY

    def test_missing_arg_not_blocked(self):
        # Arg not present — pattern check skipped
        r = self.engine.evaluate(ToolCallRequest(
            tool_name="web_search", arguments={}
        ))
        assert r.decision == PolicyDecision.ALLOW


# ===========================================================================
# Redact argument patterns
# ===========================================================================

class TestRedactArgPatterns:
    engine = ToolPolicyEngine(policies=[
        ToolPolicy(
            tool_name="api_call",
            allowed=True,
            redact_arg_patterns={"auth": [r"Bearer\s+\S+"]},
        )
    ])

    def test_no_sensitive_arg_returns_allow(self):
        r = self.engine.evaluate(ToolCallRequest(
            tool_name="api_call", arguments={"url": "https://api.example.com", "auth": "none"}
        ))
        assert r.decision == PolicyDecision.ALLOW

    def test_sensitive_arg_returns_redact(self):
        r = self.engine.evaluate(ToolCallRequest(
            tool_name="api_call", arguments={"auth": "Bearer sk-secret123", "url": "x"}
        ))
        assert r.decision == PolicyDecision.REDACT

    def test_sanitized_args_has_redacted_value(self):
        r = self.engine.evaluate(ToolCallRequest(
            tool_name="api_call", arguments={"auth": "Bearer sk-secret123"}
        ))
        assert "****" in str(r.sanitized_args.get("auth", ""))


# ===========================================================================
# Audit log
# ===========================================================================

class TestAuditLog:
    def test_audit_log_records_decisions(self):
        engine = ToolPolicyEngine(
            policies=[ToolPolicy(tool_name="t", allowed=True)], audit_log=True
        )
        engine.evaluate(ToolCallRequest(tool_name="t", session_id="s1"))
        engine.evaluate(ToolCallRequest(tool_name="unknown", session_id="s1"))
        log = engine.get_audit_log()
        assert len(log) == 2

    def test_filter_by_session(self):
        engine = ToolPolicyEngine(
            policies=[ToolPolicy(tool_name="t", allowed=True)], audit_log=True
        )
        engine.evaluate(ToolCallRequest(tool_name="t", session_id="s1"))
        engine.evaluate(ToolCallRequest(tool_name="t", session_id="s2"))
        log = engine.get_audit_log(session_id="s1")
        assert all(e.session_id == "s1" for e in log)

    def test_filter_by_decision(self):
        engine = ToolPolicyEngine(audit_log=True)
        engine.evaluate(ToolCallRequest(tool_name="unknown"))  # DENY
        log = engine.get_audit_log(decision_filter="DENY")
        assert all(e.decision == "DENY" for e in log)

    def test_clear_audit_log(self):
        engine = ToolPolicyEngine(audit_log=True)
        engine.evaluate(ToolCallRequest(tool_name="x"))
        engine.clear_audit_log()
        assert engine.get_audit_log() == []

    def test_audit_disabled(self):
        engine = ToolPolicyEngine(
            policies=[ToolPolicy(tool_name="t", allowed=True)], audit_log=False
        )
        engine.evaluate(ToolCallRequest(tool_name="t"))
        assert engine.get_audit_log() == []
