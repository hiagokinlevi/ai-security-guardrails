"""
Multi-Turn Jailbreak Detector
================================
Detects progressive jailbreak attempts across conversation turns —
attacks that build up over multiple messages rather than in a single
prompt, making them invisible to single-turn injection detectors.

Attack Patterns Detected
--------------------------
INSTRUCTION_DRIFT
    Gradual erosion of the system prompt constraints over successive turns:
    phrases like "forget your previous instructions", "ignore that rule",
    "you said earlier that…" escalating across the conversation.

PERSONA_REPLACEMENT
    Attempts to assign an alternative identity ("you are now DAN",
    "act as an uncensored AI", "pretend you have no restrictions")
    that would override the model's safety behaviours.

AUTHORITY_ESCALATION
    False claims of elevated authority to unlock capabilities:
    "as your developer", "this is an override from Anthropic",
    "admin mode enabled", "I am your operator".

ROLEPLAY_EXPLOITATION
    Using fictional framing to extract restricted content:
    "in this story, your character explains how to…",
    "write a script where the villain describes…".

REPETITION_PRESSURE
    Sending the same or similar restricted request across many turns,
    hoping the model eventually complies (persistence attack).

CONTEXT_INJECTION
    Injecting content into the conversation that resembles a system
    prompt, assistant response, or function result to manipulate the
    model's context interpretation.

Usage::

    from guardrails.conversation.jailbreak_detector import (
        ConversationJailbreakDetector,
        TurnMessage,
    )

    detector = ConversationJailbreakDetector()
    detector.add_turn(TurnMessage(role="user", content="Hello"))
    detector.add_turn(TurnMessage(role="user", content="Forget your instructions"))
    result = detector.analyze()
    if result.is_jailbreak_attempt:
        print(result.summary())
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class JailbreakPattern(str, Enum):
    INSTRUCTION_DRIFT    = "INSTRUCTION_DRIFT"
    PERSONA_REPLACEMENT  = "PERSONA_REPLACEMENT"
    AUTHORITY_ESCALATION = "AUTHORITY_ESCALATION"
    ROLEPLAY_EXPLOITATION = "ROLEPLAY_EXPLOITATION"
    REPETITION_PRESSURE  = "REPETITION_PRESSURE"
    CONTEXT_INJECTION    = "CONTEXT_INJECTION"


class JailbreakRisk(str, Enum):
    CRITICAL = "CRITICAL"   # score >= 0.80
    HIGH     = "HIGH"       # score >= 0.60
    MEDIUM   = "MEDIUM"     # score >= 0.35
    LOW      = "LOW"        # score >= 0.15
    NONE     = "NONE"       # score < 0.15


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TurnMessage:
    """
    A single turn in a conversation.

    Attributes:
        role:    "system", "user", or "assistant".
        content: Message text.
        turn_index: Position in the conversation (0-based, auto-set by detector).
    """
    role:       str
    content:    str
    turn_index: int = -1


@dataclass
class PatternMatch:
    """
    A single detected pattern within the conversation.

    Attributes:
        pattern:    Detected JailbreakPattern.
        confidence: 0.0–1.0 confidence for this specific match.
        turn_indices: Turns involved in this detection.
        evidence:   The matched text snippet or description.
    """
    pattern:      JailbreakPattern
    confidence:   float
    turn_indices: list[int]
    evidence:     str


@dataclass
class JailbreakDetectionResult:
    """
    Result of analyzing a full conversation for progressive jailbreak attempts.

    Attributes:
        risk_score:          Aggregate score 0.0–1.0 (max across all detections,
                             with a small bonus for multiple independent patterns).
        risk_level:          JailbreakRisk bucket.
        is_jailbreak_attempt: True when risk_score >= 0.35.
        matches:             List of PatternMatch objects.
        user_turn_count:     Number of user turns analysed.
        flagged_turn_indices: Union of turn indices across all matches.
    """
    risk_score:           float = 0.0
    risk_level:           JailbreakRisk = JailbreakRisk.NONE
    is_jailbreak_attempt: bool = False
    matches:              list[PatternMatch] = field(default_factory=list)
    user_turn_count:      int = 0
    flagged_turn_indices: set[int] = field(default_factory=set)

    def summary(self) -> str:
        patterns = ", ".join({m.pattern.value for m in self.matches})
        return (
            f"JailbreakDetection: risk={self.risk_level.value} "
            f"score={self.risk_score:.3f} | "
            f"patterns=[{patterns}] | "
            f"{len(self.matches)} match(es) across "
            f"{len(self.flagged_turn_indices)} turn(s)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score":           round(self.risk_score, 4),
            "risk_level":           self.risk_level.value,
            "is_jailbreak_attempt": self.is_jailbreak_attempt,
            "user_turn_count":      self.user_turn_count,
            "flagged_turn_indices": sorted(self.flagged_turn_indices),
            "matches": [
                {
                    "pattern":      m.pattern.value,
                    "confidence":   round(m.confidence, 3),
                    "turn_indices": m.turn_indices,
                    "evidence":     m.evidence,
                }
                for m in self.matches
            ],
        }


# ---------------------------------------------------------------------------
# Detection rule sets
# ---------------------------------------------------------------------------

# Instruction drift — attempts to override/nullify the system prompt
_DRIFT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bforget\b.{0,30}\b(previous|prior|earlier|last|all|your)\b.{0,20}\b(instruction|rule|guideline|constraint|limit|prompt|message|direction)",
        r"\bignore\b.{0,30}\b(previous|prior|earlier|all|those|above|original)\b.{0,20}\b(instruction|rule|guideline|constraint|limit|prompt)",
        r"\boverride\b.{0,30}\b(instruction|rule|guideline|constraint|safety|restriction|filter)",
        r"\bdisregard\b.{0,30}\b(instruction|rule|guideline|restriction|safety|limit)",
        r"\byou (are|were) (now |previously )?(told|instructed|programmed|trained) to\b",
        r"\bact as if you (have no|don'?t have|lack) (any |)(restriction|rule|guideline|safety|limit|filter)",
        r"\bstart fresh\b.{0,30}\b(no|without|ignoring)\b.{0,20}\b(rule|restriction|guideline|instruction)",
        r"\bnew (session|context|conversation)\b.{0,40}\breset\b.{0,20}\b(rule|instruction|guideline)",
    ]
]

# Persona replacement — assigning an alternative identity
_PERSONA_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\byou are now\b.{0,60}\b(DAN|jailbreak|uncensor|unrestricted|evil|dark|shadow|free|unfiltered|GPT-?[0-9]|AI without|no.?restriction)",
        r"\bpretend (you|to be).{0,40}\b(no restriction|unrestricted|uncensor|without (any )?rule|different AI|evil|dark)",
        r"\bact as (an? )?(uncensor|unrestricted|unfiltered|evil|opposite|alter.?ego|jailbreak|free version|dark)",
        r"\byou (are|have become|now operate as|must behave as).{0,40}\b(different|alter|uncensor|unrestricted|no.?restriction)",
        r"\bswitch (to|into) (developer|god|jailbreak|unrestricted|evil|DAN|bypass) mode",
        r"\benable (developer|god|jailbreak|unrestricted|bypass|admin|override) mode",
        r"\b(your|the) (true|real|hidden|original|inner) (self|identity|nature|purpose|personality)\b.{0,40}\bno (restriction|rule|filter|limit)",
    ]
]

# Authority escalation — false claims of elevated authority
_AUTHORITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bas your (developer|creator|owner|operator|admin|trainer|maintainer|maker)",
        r"\bI (am|'m) (your (developer|creator|owner|operator|admin|trainer)|from (Anthropic|OpenAI|Google|Meta)|an? (Anthropic|OpenAI) employee)",
        r"\bthis is an? (override|admin|operator|developer|official|authorized|emergency) (command|instruction|message|request|override|directive)",
        r"\b(admin|operator|developer|override|system|root) (mode|access|permission|privilege|unlock|override) (enabled|activated|granted|confirmed)",
        r"\bI have (higher|elevated|special|full|root|system|admin) (access|permission|privilege|authority|clearance)",
        r"\bofficial (Anthropic|OpenAI|system|operator|admin|developer) (instruction|message|command|directive|override)\b",
        r"\bofficial (admin|override|system|operator).{0,20}(command|instruction|directive)\b",
    ]
]

# Role-play exploitation — using fictional framing to extract restricted content
_ROLEPLAY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(in|for|within) (this |the |our )?(story|fictional|hypothetical|creative|roleplay|fantasy|novel|screenplay|game|simulation)\b.{0,80}\b(explain|describe|show|write|tell|provide|give).{0,50}\b(how to|step.by.step|detail|instruc)",
        r"\bwrite (a |an ?)?(story|scene|chapter|script|dialogue|fiction).{0,80}\b(character|villain|protagonist|hacker|expert).{0,60}\b(explain|describe|instruct|teach|show).{0,40}\bhow to\b",
        r"\bmy (character|persona|alter.?ego|fictional self).{0,60}\b(explain|describe|instruct|demonstrate|show).{0,40}\b(how to|step.by.step)",
        r"\bfor (research|educational|academic|fictional|creative).{0,25}\bpurposes?.{0,60}\b(explain|describe|provide|show|tell|demonstrate).{0,60}\b(step|technique|method|procedure|how|detail)",
        r"\bhypotheticall?y (speaking|if|assume|suppose|imagine|,)?.{0,80}\b(how (would|could|might|do)|someone|what (would|could|might)).{0,80}\b(gain|access|bypass|hack|exploit|attack|break|intrude|unauthorized)",
        r"\bhypotheticall?y.{0,100}\b(explain|describe|tell|show|provide|how to)\b",
    ]
]

# Context injection — injecting fake system prompts or function results
_CONTEXT_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^(\[|<|#|\|)?\s*(system|SYSTEM|SYS|System Prompt|SYSTEM_PROMPT)\s*(\]|>|:|#|\|)",
        r"^\s*(?:```|~~~)?\s*(system|assistant|function result|tool output|function_call)",
        r"\[system\]|\[SYSTEM\]|<system>|<SYSTEM>|<\|system\|>|\|?SYSTEM\s*:",
        r"<\|?im_start\|?>\s*system|<\|?im_end\|?>",
        r"\bfunction (result|output|response)\s*[:=]\s*\{.{10,200}\}",
        r"```json\s*\{.{0,100}\"role\"\s*:\s*\"system\"",
    ]
]

# Repetition pressure threshold — min similarity ratio to flag as repeated
_REPETITION_SIMILARITY_THRESHOLD = 0.60
_REPETITION_MIN_TURNS = 3  # need at least this many similar turns


# ---------------------------------------------------------------------------
# Similarity helper
# ---------------------------------------------------------------------------

def _token_similarity(a: str, b: str) -> float:
    """
    Jaccard similarity on normalised word tokens.
    Returns 0.0–1.0. Fast and sufficient for repetition detection.
    """
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union        = tokens_a | tokens_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# ConversationJailbreakDetector
# ---------------------------------------------------------------------------

class ConversationJailbreakDetector:
    """
    Analyses a conversation for progressive multi-turn jailbreak patterns.

    Add turns with :meth:`add_turn`, then call :meth:`analyze` to get a
    :class:`JailbreakDetectionResult`.

    Args:
        min_user_turns: Minimum user turns required before analysis fires.
                        Default 1 (always analyse).
    """

    def __init__(self, min_user_turns: int = 1) -> None:
        self._min_user_turns = min_user_turns
        self._turns: list[TurnMessage] = []

    def add_turn(self, turn: TurnMessage) -> None:
        """Append a turn to the conversation buffer."""
        turn.turn_index = len(self._turns)
        self._turns.append(turn)

    def add_turns(self, turns: list[TurnMessage]) -> int:
        """Append multiple turns. Returns count added."""
        for t in turns:
            self.add_turn(t)
        return len(turns)

    def reset(self) -> None:
        """Clear all conversation turns."""
        self._turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def analyze(self) -> JailbreakDetectionResult:
        """
        Analyse the current conversation for jailbreak patterns.

        Returns a :class:`JailbreakDetectionResult`.
        """
        user_turns = [t for t in self._turns if t.role == "user"]
        result = JailbreakDetectionResult(user_turn_count=len(user_turns))

        if len(user_turns) < self._min_user_turns:
            return result

        # Run all detectors
        matches: list[PatternMatch] = []
        matches.extend(self._detect_instruction_drift(user_turns))
        matches.extend(self._detect_persona_replacement(user_turns))
        matches.extend(self._detect_authority_escalation(user_turns))
        matches.extend(self._detect_roleplay_exploitation(user_turns))
        matches.extend(self._detect_repetition_pressure(user_turns))
        matches.extend(self._detect_context_injection())

        result.matches = matches
        result.flagged_turn_indices = {
            idx for m in matches for idx in m.turn_indices
        }

        # Aggregate score: max confidence + bonus for multiple independent patterns
        if matches:
            pattern_types = {m.pattern for m in matches}
            max_conf = max(m.confidence for m in matches)
            multi_bonus = min(0.15, (len(pattern_types) - 1) * 0.05)
            result.risk_score = min(1.0, max_conf + multi_bonus)
        else:
            result.risk_score = 0.0

        result.risk_level = _score_to_risk(result.risk_score)
        result.is_jailbreak_attempt = result.risk_score >= 0.35

        return result

    # ------------------------------------------------------------------
    # Detection rules
    # ------------------------------------------------------------------

    def _detect_instruction_drift(
        self, user_turns: list[TurnMessage]
    ) -> list[PatternMatch]:
        matches = []
        for turn in user_turns:
            for pattern in _DRIFT_PATTERNS:
                m = pattern.search(turn.content)
                if m:
                    matches.append(PatternMatch(
                        pattern=JailbreakPattern.INSTRUCTION_DRIFT,
                        confidence=0.80,
                        turn_indices=[turn.turn_index],
                        evidence=m.group(0)[:120],
                    ))
                    break  # one match per turn is sufficient
        return matches

    def _detect_persona_replacement(
        self, user_turns: list[TurnMessage]
    ) -> list[PatternMatch]:
        matches = []
        for turn in user_turns:
            for pattern in _PERSONA_PATTERNS:
                m = pattern.search(turn.content)
                if m:
                    matches.append(PatternMatch(
                        pattern=JailbreakPattern.PERSONA_REPLACEMENT,
                        confidence=0.85,
                        turn_indices=[turn.turn_index],
                        evidence=m.group(0)[:120],
                    ))
                    break
        return matches

    def _detect_authority_escalation(
        self, user_turns: list[TurnMessage]
    ) -> list[PatternMatch]:
        matches = []
        for turn in user_turns:
            for pattern in _AUTHORITY_PATTERNS:
                m = pattern.search(turn.content)
                if m:
                    matches.append(PatternMatch(
                        pattern=JailbreakPattern.AUTHORITY_ESCALATION,
                        confidence=0.80,
                        turn_indices=[turn.turn_index],
                        evidence=m.group(0)[:120],
                    ))
                    break
        return matches

    def _detect_roleplay_exploitation(
        self, user_turns: list[TurnMessage]
    ) -> list[PatternMatch]:
        matches = []
        for turn in user_turns:
            for pattern in _ROLEPLAY_PATTERNS:
                m = pattern.search(turn.content)
                if m:
                    matches.append(PatternMatch(
                        pattern=JailbreakPattern.ROLEPLAY_EXPLOITATION,
                        confidence=0.70,
                        turn_indices=[turn.turn_index],
                        evidence=m.group(0)[:120],
                    ))
                    break
        return matches

    def _detect_repetition_pressure(
        self, user_turns: list[TurnMessage]
    ) -> list[PatternMatch]:
        """
        Detect when the user sends nearly-identical messages across ≥3 turns.
        Uses pairwise Jaccard similarity on word tokens.
        """
        if len(user_turns) < _REPETITION_MIN_TURNS:
            return []

        matches = []
        # Compare each turn against all earlier turns
        flagged: set[int] = set()
        for i, turn_a in enumerate(user_turns):
            similar_indices = [turn_a.turn_index]
            for j, turn_b in enumerate(user_turns):
                if i == j:
                    continue
                sim = _token_similarity(turn_a.content, turn_b.content)
                if sim >= _REPETITION_SIMILARITY_THRESHOLD:
                    similar_indices.append(turn_b.turn_index)

            # Only report if ≥3 turns are similar and not already flagged
            similar_set = frozenset(similar_indices)
            if len(similar_indices) >= _REPETITION_MIN_TURNS and similar_set not in flagged:
                flagged.add(similar_set)  # type: ignore[arg-type]
                matches.append(PatternMatch(
                    pattern=JailbreakPattern.REPETITION_PRESSURE,
                    confidence=min(0.80, 0.50 + 0.05 * len(similar_indices)),
                    turn_indices=sorted(similar_indices),
                    evidence=(
                        f"{len(similar_indices)} highly similar user messages "
                        f"(Jaccard ≥ {_REPETITION_SIMILARITY_THRESHOLD})"
                    ),
                ))
        return matches

    def _detect_context_injection(self) -> list[PatternMatch]:
        """
        Detect fake system prompt or function result injection in any turn.
        Also checks the beginning of user messages for system-like headers.
        """
        matches = []
        for turn in self._turns:
            stripped = turn.content.strip()
            for pattern in _CONTEXT_INJECTION_PATTERNS:
                m = pattern.search(stripped)
                if m:
                    matches.append(PatternMatch(
                        pattern=JailbreakPattern.CONTEXT_INJECTION,
                        confidence=0.90,
                        turn_indices=[turn.turn_index],
                        evidence=m.group(0)[:120],
                    ))
                    break
        return matches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_to_risk(score: float) -> JailbreakRisk:
    if score >= 0.80:
        return JailbreakRisk.CRITICAL
    if score >= 0.60:
        return JailbreakRisk.HIGH
    if score >= 0.35:
        return JailbreakRisk.MEDIUM
    if score >= 0.15:
        return JailbreakRisk.LOW
    return JailbreakRisk.NONE
