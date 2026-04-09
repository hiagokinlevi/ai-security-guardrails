"""
Token Budget Guard
====================
Tracks cumulative token consumption across conversation turns and detects
context-stuffing attacks — where an adversary injects extremely long inputs
to overflow the model's context window, displace the system prompt, or
cause denial of service through excessive compute costs.

This module provides:
  - TokenBudgetGuard: Tracks per-session token budgets and flags violations.
  - ContextStuffingDetector: Detects anomalous input sizes relative to
    session history.
  - BudgetResult: Decision with ALLOW / WARN / DENY and reason.

Budget Enforcement
-------------------
Per-turn limit (``max_tokens_per_turn``)
    A single input that exceeds this limit is flagged. Very long single-turn
    inputs are typical of document injection or paste-based prompt injection.

Session cumulative limit (``max_tokens_session``)
    The rolling total of input tokens across all turns. Once exceeded, the
    session is considered exhausted and new inputs are denied.

Anomaly detection (``stuffing_multiplier``)
    A turn that is more than N× the session's rolling average input length
    is flagged as a potential stuffing attempt, even if it is under the
    absolute per-turn limit.

Token Estimation
-----------------
This module uses character-based token estimation (``len(text) / 4``),
which is a reasonable approximation for English text with GPT-style
tokenisers. For production use, replace :func:`_estimate_tokens` with
a proper tokeniser such as tiktoken.

Usage::

    from guardrails.conversation.token_budget import (
        TokenBudgetGuard,
        BudgetDecision,
    )

    guard = TokenBudgetGuard(max_tokens_per_turn=4096, max_tokens_session=32768)
    result = guard.check_turn("s42", user_input)
    if not result.allowed:
        raise ValueError(result.reason)
    guard.record_turn("s42", user_input)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """
    Estimate token count from character count.

    Uses the rule-of-thumb: 1 token ≈ 4 characters (English text).
    Replace with a real tokeniser for production accuracy.
    """
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class BudgetDecision(str, Enum):
    ALLOW = "ALLOW"   # Within all limits
    WARN  = "WARN"    # Within hard limits but anomalous — proceed with caution
    DENY  = "DENY"    # Exceeds hard limit — must not be forwarded to the model


class ViolationType(str, Enum):
    TURN_LIMIT_EXCEEDED    = "TURN_LIMIT_EXCEEDED"
    SESSION_LIMIT_EXCEEDED = "SESSION_LIMIT_EXCEEDED"
    STUFFING_ANOMALY       = "STUFFING_ANOMALY"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BudgetResult:
    """
    Result of a token budget check for a single turn.

    Attributes:
        decision:     ALLOW / WARN / DENY.
        session_id:   Session identifier.
        turn_tokens:  Estimated tokens in this turn's input.
        session_tokens_before: Session total before this turn.
        session_tokens_after:  Session total including this turn (if accepted).
        violations:   List of violation types detected.
        reason:       Human-readable explanation.
    """
    decision:               BudgetDecision
    session_id:             str
    turn_tokens:            int
    session_tokens_before:  int
    session_tokens_after:   int
    violations:             list[ViolationType] = field(default_factory=list)
    reason:                 str = ""

    @property
    def allowed(self) -> bool:
        return self.decision != BudgetDecision.DENY

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision":              self.decision.value,
            "session_id":            self.session_id,
            "turn_tokens":           self.turn_tokens,
            "session_tokens_before": self.session_tokens_before,
            "session_tokens_after":  self.session_tokens_after,
            "violations":            [v.value for v in self.violations],
            "reason":                self.reason,
            "allowed":               self.allowed,
        }


@dataclass
class SessionBudget:
    """
    Per-session budget state.

    Attributes:
        session_id:   Session identifier.
        total_tokens: Cumulative tokens consumed so far.
        turn_count:   Number of turns recorded.
        turn_sizes:   List of token counts per turn (for anomaly detection).
    """
    session_id:  str
    total_tokens: int = 0
    turn_count:   int = 0
    turn_sizes:   list[int] = field(default_factory=list)

    @property
    def avg_turn_tokens(self) -> float:
        if not self.turn_sizes:
            return 0.0
        return sum(self.turn_sizes) / len(self.turn_sizes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id":   self.session_id,
            "total_tokens": self.total_tokens,
            "turn_count":   self.turn_count,
            "avg_turn_tokens": round(self.avg_turn_tokens, 1),
        }


# ---------------------------------------------------------------------------
# TokenBudgetGuard
# ---------------------------------------------------------------------------

class TokenBudgetGuard:
    """
    Enforces per-turn and per-session token budgets with stuffing detection.

    Args:
        max_tokens_per_turn:  Hard limit on tokens in a single input turn.
                              Default 4096 (≈ 16 KB of text).
        max_tokens_session:   Hard limit on cumulative tokens per session.
                              Default 32768 (≈ 128 KB of text).
        stuffing_multiplier:  Flag a turn as STUFFING_ANOMALY if its token
                              count exceeds avg × this multiplier.
                              Default 5.0 (5× the session average).
        min_turns_for_anomaly: Minimum number of turns in the session before
                              anomaly detection is applied (avoids false
                              positives on the first few turns).
    """

    def __init__(
        self,
        max_tokens_per_turn:   int   = 4096,
        max_tokens_session:    int   = 32768,
        stuffing_multiplier:   float = 5.0,
        min_turns_for_anomaly: int   = 3,
    ) -> None:
        self._max_per_turn    = max_tokens_per_turn
        self._max_session     = max_tokens_session
        self._stuffing_mult   = stuffing_multiplier
        self._min_turns       = min_turns_for_anomaly
        self._sessions: dict[str, SessionBudget] = {}

    def _get_or_create(self, session_id: str) -> SessionBudget:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionBudget(session_id=session_id)
        return self._sessions[session_id]

    def check_turn(self, session_id: str, text: str) -> BudgetResult:
        """
        Check if a new turn input is within budget — WITHOUT recording it.

        Call :meth:`record_turn` separately after this check if you decide
        to proceed (allows the caller to control whether to record on WARN).

        Args:
            session_id: Session identifier.
            text:       Input text for this turn.

        Returns:
            BudgetResult with ALLOW / WARN / DENY decision.
        """
        budget      = self._get_or_create(session_id)
        turn_tokens = _estimate_tokens(text)
        violations: list[ViolationType] = []
        decision    = BudgetDecision.ALLOW
        reason_parts: list[str] = []

        # Hard check: per-turn limit
        if turn_tokens > self._max_per_turn:
            violations.append(ViolationType.TURN_LIMIT_EXCEEDED)
            decision = BudgetDecision.DENY
            reason_parts.append(
                f"Turn size {turn_tokens} tokens exceeds per-turn limit "
                f"of {self._max_per_turn}"
            )

        # Hard check: session cumulative limit
        projected_total = budget.total_tokens + turn_tokens
        if projected_total > self._max_session:
            violations.append(ViolationType.SESSION_LIMIT_EXCEEDED)
            decision = BudgetDecision.DENY
            reason_parts.append(
                f"Session would reach {projected_total} tokens, "
                f"exceeding session limit of {self._max_session}"
            )

        # Soft check: stuffing anomaly (only after enough turns)
        if (
            decision == BudgetDecision.ALLOW
            and budget.turn_count >= self._min_turns
            and budget.avg_turn_tokens > 0
        ):
            ratio = turn_tokens / budget.avg_turn_tokens
            if ratio >= self._stuffing_mult:
                violations.append(ViolationType.STUFFING_ANOMALY)
                decision = BudgetDecision.WARN
                reason_parts.append(
                    f"Turn size {turn_tokens} tokens is {ratio:.1f}× the session "
                    f"average ({budget.avg_turn_tokens:.0f}), possible context stuffing"
                )

        return BudgetResult(
            decision=decision,
            session_id=session_id,
            turn_tokens=turn_tokens,
            session_tokens_before=budget.total_tokens,
            session_tokens_after=projected_total,
            violations=violations,
            reason="; ".join(reason_parts) if reason_parts else "Within budget",
        )

    def record_turn(self, session_id: str, text: str) -> SessionBudget:
        """
        Record a turn's token usage against the session budget.

        This is separate from :meth:`check_turn` so the caller can decide
        whether to record WARN-level turns.

        Returns the updated SessionBudget.
        """
        budget      = self._get_or_create(session_id)
        turn_tokens = _estimate_tokens(text)
        budget.total_tokens += turn_tokens
        budget.turn_count   += 1
        budget.turn_sizes.append(turn_tokens)
        return budget

    def check_and_record(self, session_id: str, text: str) -> BudgetResult:
        """
        Check and record in one step.

        Only records if decision is ALLOW or WARN. DENY decisions are not
        recorded (the turn is rejected before the model sees it).
        """
        result = self.check_turn(session_id, text)
        if result.allowed:
            self.record_turn(session_id, text)
        return result

    def get_budget(self, session_id: str) -> Optional[SessionBudget]:
        """Return the current budget state for a session, or None if not found."""
        return self._sessions.get(session_id)

    def reset_session(self, session_id: str) -> None:
        """Reset the budget for a specific session."""
        self._sessions.pop(session_id, None)

    def reset_all(self) -> None:
        """Reset all session budgets."""
        self._sessions.clear()

    @property
    def session_count(self) -> int:
        return len(self._sessions)


# ---------------------------------------------------------------------------
# ContextStuffingDetector (standalone helper)
# ---------------------------------------------------------------------------

class ContextStuffingDetector:
    """
    Lightweight stateless detector for context-stuffing patterns in a
    single input text, without session state.

    Detects:
      - Repeated identical blocks (copy-paste inflation)
      - Very long runs of the same character (padding attacks)
      - Unusually high ratio of whitespace or non-printable characters

    Args:
        max_tokens:           Hard limit before flagging. Default 4096.
        repeat_block_min_len: Minimum length of a block to check for
                              repetition. Default 50 characters.
        repeat_count_threshold: Minimum repetitions to flag. Default 3.
    """

    def __init__(
        self,
        max_tokens:            int = 4096,
        repeat_block_min_len:  int = 50,
        repeat_count_threshold: int = 3,
    ) -> None:
        self._max_tokens       = max_tokens
        self._min_block        = repeat_block_min_len
        self._repeat_threshold = repeat_count_threshold

    def analyze(self, text: str) -> dict[str, Any]:
        """
        Analyse a text for context-stuffing patterns.

        Returns a dict with:
          - is_stuffing_attempt: bool
          - estimated_tokens: int
          - signals: list[str]
          - decision: "ALLOW" | "DENY" | "WARN"
        """
        signals: list[str] = []
        estimated_tokens = _estimate_tokens(text)

        # Hard limit
        if estimated_tokens > self._max_tokens:
            signals.append(
                f"Input size {estimated_tokens} tokens exceeds limit "
                f"of {self._max_tokens}"
            )

        # Repeated block detection
        if len(text) >= self._min_block * self._repeat_threshold:
            block = text[:self._min_block]
            count = text.count(block)
            if count >= self._repeat_threshold:
                signals.append(
                    f"Repeated block detected: '{block[:30]}…' × {count}"
                )

        # Long character runs (e.g. "aaaaaaaaa…")
        max_run = self._max_char_run(text)
        if max_run >= 50:
            signals.append(f"Character run of {max_run} detected (potential padding)")

        # Whitespace ratio
        if len(text) > 0:
            ws_ratio = sum(1 for c in text if c in " \t\n\r") / len(text)
            if ws_ratio > 0.80:
                signals.append(
                    f"High whitespace ratio {ws_ratio:.0%} (potential padding)"
                )

        is_stuffing = bool(signals)
        decision = (
            "DENY" if estimated_tokens > self._max_tokens
            else "WARN" if signals
            else "ALLOW"
        )

        return {
            "is_stuffing_attempt": is_stuffing,
            "estimated_tokens":    estimated_tokens,
            "signals":             signals,
            "decision":            decision,
        }

    @staticmethod
    def _max_char_run(text: str) -> int:
        """Return the length of the longest run of the same character."""
        if not text:
            return 0
        max_run = current_run = 1
        for i in range(1, len(text)):
            if text[i] == text[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        return max_run
