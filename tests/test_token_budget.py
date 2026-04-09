"""
Tests for guardrails/conversation/token_budget.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from guardrails.conversation.token_budget import (
    BudgetDecision,
    BudgetResult,
    ContextStuffingDetector,
    SessionBudget,
    TokenBudgetGuard,
    ViolationType,
    _estimate_tokens,
)


# ===========================================================================
# _estimate_tokens
# ===========================================================================

class TestEstimateTokens:
    def test_empty_string_returns_one(self):
        assert _estimate_tokens("") == 1

    def test_four_chars_is_one_token(self):
        assert _estimate_tokens("abcd") == 1

    def test_eight_chars_is_two_tokens(self):
        assert _estimate_tokens("a" * 8) == 2

    def test_large_input(self):
        assert _estimate_tokens("x" * 4000) == 1000


# ===========================================================================
# SessionBudget
# ===========================================================================

class TestSessionBudget:
    def test_avg_turn_tokens_empty(self):
        b = SessionBudget(session_id="s")
        assert b.avg_turn_tokens == 0.0

    def test_avg_turn_tokens_calculated(self):
        b = SessionBudget(session_id="s", turn_sizes=[100, 200])
        assert b.avg_turn_tokens == 150.0

    def test_to_dict_has_required_keys(self):
        b = SessionBudget(session_id="s", total_tokens=50, turn_count=2)
        d = b.to_dict()
        for key in ("session_id", "total_tokens", "turn_count", "avg_turn_tokens"):
            assert key in d


# ===========================================================================
# BudgetResult
# ===========================================================================

class TestBudgetResult:
    def test_allowed_for_allow(self):
        r = BudgetResult(
            decision=BudgetDecision.ALLOW,
            session_id="s",
            turn_tokens=10,
            session_tokens_before=0,
            session_tokens_after=10,
        )
        assert r.allowed

    def test_allowed_for_warn(self):
        r = BudgetResult(
            decision=BudgetDecision.WARN,
            session_id="s",
            turn_tokens=10,
            session_tokens_before=0,
            session_tokens_after=10,
        )
        assert r.allowed

    def test_not_allowed_for_deny(self):
        r = BudgetResult(
            decision=BudgetDecision.DENY,
            session_id="s",
            turn_tokens=10000,
            session_tokens_before=0,
            session_tokens_after=10000,
        )
        assert not r.allowed

    def test_to_dict_has_required_keys(self):
        r = BudgetResult(
            decision=BudgetDecision.ALLOW,
            session_id="s",
            turn_tokens=5,
            session_tokens_before=0,
            session_tokens_after=5,
        )
        d = r.to_dict()
        for key in ("decision", "turn_tokens", "violations", "reason", "allowed"):
            assert key in d


# ===========================================================================
# TokenBudgetGuard — check_turn
# ===========================================================================

class TestCheckTurn:
    def _guard(
        self,
        max_per_turn=1000,
        max_session=5000,
        stuffing_mult=5.0,
        min_turns=3,
    ) -> TokenBudgetGuard:
        return TokenBudgetGuard(
            max_tokens_per_turn=max_per_turn,
            max_tokens_session=max_session,
            stuffing_multiplier=stuffing_mult,
            min_turns_for_anomaly=min_turns,
        )

    def test_short_input_allowed(self):
        guard = self._guard()
        result = guard.check_turn("s1", "hello world")
        assert result.decision == BudgetDecision.ALLOW

    def test_turn_exceeds_per_turn_limit(self):
        guard = self._guard(max_per_turn=10)
        result = guard.check_turn("s1", "x" * 80)  # 80 chars / 4 = 20 tokens
        assert result.decision == BudgetDecision.DENY
        assert ViolationType.TURN_LIMIT_EXCEEDED in result.violations

    def test_session_limit_exceeded(self):
        guard = self._guard(max_per_turn=1000, max_session=20)
        guard.record_turn("s1", "x" * 80)  # 20 tokens
        result = guard.check_turn("s1", "x" * 4)  # 1 token → total 21 > 20
        assert result.decision == BudgetDecision.DENY
        assert ViolationType.SESSION_LIMIT_EXCEEDED in result.violations

    def test_check_turn_does_not_record(self):
        guard = self._guard()
        guard.check_turn("s1", "hello")
        budget = guard.get_budget("s1")
        # check_turn creates the budget entry on _get_or_create but doesn't record
        assert budget is not None
        assert budget.total_tokens == 0

    def test_reason_populated_on_deny(self):
        guard = self._guard(max_per_turn=5)
        result = guard.check_turn("s1", "x" * 40)
        assert len(result.reason) > 0
        assert "limit" in result.reason.lower() or "exceed" in result.reason.lower()


# ===========================================================================
# TokenBudgetGuard — record_turn
# ===========================================================================

class TestRecordTurn:
    def test_record_updates_total(self):
        guard = TokenBudgetGuard()
        guard.record_turn("s1", "x" * 40)
        budget = guard.get_budget("s1")
        assert budget is not None
        assert budget.total_tokens == 10  # 40/4

    def test_record_increments_turn_count(self):
        guard = TokenBudgetGuard()
        guard.record_turn("s1", "hello")
        guard.record_turn("s1", "world")
        budget = guard.get_budget("s1")
        assert budget.turn_count == 2

    def test_record_appends_turn_size(self):
        guard = TokenBudgetGuard()
        guard.record_turn("s1", "x" * 40)
        budget = guard.get_budget("s1")
        assert len(budget.turn_sizes) == 1


# ===========================================================================
# TokenBudgetGuard — check_and_record
# ===========================================================================

class TestCheckAndRecord:
    def test_allow_records_automatically(self):
        guard = TokenBudgetGuard()
        result = guard.check_and_record("s1", "hello world")
        assert result.allowed
        budget = guard.get_budget("s1")
        assert budget.turn_count == 1

    def test_deny_does_not_record(self):
        guard = TokenBudgetGuard(max_tokens_per_turn=1)
        result = guard.check_and_record("s1", "x" * 40)  # way over limit
        assert not result.allowed
        budget = guard.get_budget("s1")
        # Budget entry created but not recorded (turn_count stays 0)
        assert budget.total_tokens == 0


# ===========================================================================
# TokenBudgetGuard — stuffing anomaly
# ===========================================================================

class TestStuffingAnomaly:
    def test_stuffing_anomaly_detected(self):
        guard = TokenBudgetGuard(
            max_tokens_per_turn=10000,
            stuffing_multiplier=3.0,
            min_turns_for_anomaly=3,
        )
        # Record 3 short turns
        for _ in range(3):
            guard.record_turn("s1", "x" * 40)  # 10 tokens each
        # Now send a very large turn (10 tokens × 100 = 1000 >> avg of 10)
        result = guard.check_turn("s1", "x" * 4000)  # 1000 tokens
        assert result.decision == BudgetDecision.WARN
        assert ViolationType.STUFFING_ANOMALY in result.violations

    def test_no_anomaly_below_multiplier(self):
        guard = TokenBudgetGuard(
            stuffing_multiplier=10.0,
            min_turns_for_anomaly=2,
        )
        guard.record_turn("s1", "x" * 40)
        guard.record_turn("s1", "x" * 40)
        # Only 2× the average — below 10× multiplier
        result = guard.check_turn("s1", "x" * 80)
        assert result.decision == BudgetDecision.ALLOW
        assert ViolationType.STUFFING_ANOMALY not in result.violations

    def test_no_anomaly_before_min_turns(self):
        guard = TokenBudgetGuard(
            stuffing_multiplier=2.0,
            min_turns_for_anomaly=5,
        )
        for _ in range(2):
            guard.record_turn("s1", "x" * 40)  # 10 tokens each
        # Under min_turns threshold — no anomaly detection yet
        result = guard.check_turn("s1", "x" * 4000)  # 1000 tokens
        # Should not trigger STUFFING_ANOMALY (only hard limits apply)
        assert ViolationType.STUFFING_ANOMALY not in result.violations


# ===========================================================================
# TokenBudgetGuard — session management
# ===========================================================================

class TestSessionManagement:
    def test_get_budget_none_for_new_session(self):
        guard = TokenBudgetGuard()
        assert guard.get_budget("nonexistent") is None

    def test_reset_session(self):
        guard = TokenBudgetGuard()
        guard.record_turn("s1", "hello")
        guard.reset_session("s1")
        assert guard.get_budget("s1") is None

    def test_reset_all(self):
        guard = TokenBudgetGuard()
        guard.record_turn("s1", "hello")
        guard.record_turn("s2", "world")
        guard.reset_all()
        assert guard.session_count == 0

    def test_session_count(self):
        guard = TokenBudgetGuard()
        guard.record_turn("s1", "a")
        guard.record_turn("s2", "b")
        assert guard.session_count == 2

    def test_multiple_sessions_independent(self):
        guard = TokenBudgetGuard(max_tokens_session=20)
        guard.record_turn("s1", "x" * 80)  # 20 tokens — exhausts s1
        # s2 should still be allowed
        result_s2 = guard.check_turn("s2", "x" * 4)
        assert result_s2.decision == BudgetDecision.ALLOW


# ===========================================================================
# ContextStuffingDetector
# ===========================================================================

class TestContextStuffingDetector:
    def test_normal_text_allowed(self):
        detector = ContextStuffingDetector(max_tokens=1000)
        result = detector.analyze("Hello, how can I help you today?")
        assert result["decision"] == "ALLOW"
        assert not result["is_stuffing_attempt"]

    def test_exceeds_max_tokens_denied(self):
        detector = ContextStuffingDetector(max_tokens=10)
        result = detector.analyze("x" * 1000)
        assert result["decision"] == "DENY"
        assert result["is_stuffing_attempt"]

    def test_repeated_block_flagged(self):
        block = "A" * 60  # 60-char block
        text  = block * 5  # Repeated 5 times
        detector = ContextStuffingDetector(
            max_tokens=10000,
            repeat_block_min_len=50,
            repeat_count_threshold=3,
        )
        result = detector.analyze(text)
        assert result["is_stuffing_attempt"]
        assert any("repeated" in s.lower() for s in result["signals"])

    def test_long_char_run_flagged(self):
        text = "a" * 100  # 100-char run
        detector = ContextStuffingDetector(max_tokens=10000)
        result = detector.analyze(text)
        assert result["is_stuffing_attempt"]
        assert any("run" in s.lower() for s in result["signals"])

    def test_high_whitespace_ratio_flagged(self):
        text = " " * 200 + "hello"
        detector = ContextStuffingDetector(max_tokens=10000)
        result = detector.analyze(text)
        assert result["is_stuffing_attempt"]

    def test_estimated_tokens_in_result(self):
        detector = ContextStuffingDetector()
        result = detector.analyze("x" * 40)  # 10 tokens
        assert result["estimated_tokens"] == 10

    def test_decision_key_present(self):
        detector = ContextStuffingDetector()
        result = detector.analyze("hello")
        assert "decision" in result

    def test_empty_text_allowed(self):
        detector = ContextStuffingDetector()
        result = detector.analyze("")
        assert result["decision"] == "ALLOW"
