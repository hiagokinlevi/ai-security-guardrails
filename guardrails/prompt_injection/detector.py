"""
Prompt Injection Detector
==========================
Dedicated detection module for prompt injection attacks against LLMs.

Prompt injection is the technique of embedding instructions inside user-
supplied data (direct injection) or inside retrieved documents / tool results
(indirect injection) to hijack the model's behaviour.

This module focuses on LLM-specific attack patterns beyond the general-purpose
heuristics in input_controls/validator.py:

  - Direct injection: user directly attempts to override system instructions
  - Indirect injection: attacker-controlled data (RAG docs, tool output) contains
    hidden instructions
  - Template injection: Jinja2/Mustache/f-string patterns in user input that may
    be rendered server-side before the LLM sees them
  - Multi-turn context manipulation: instructions that try to carry state across
    conversation turns

Design:
  - Each detection rule produces a DetectionSignal with a confidence level
  - The aggregate InjectionReport contains all signals and an overall risk level
  - Rules are independent — the caller decides the action threshold
  - No blocking logic here — that lives in the policy engine

Limitations:
  - Pattern matching cannot detect all novel injection techniques
  - Multi-lingual attacks may evade English-only patterns
  - High-confidence rules may still produce false positives on legitimate prompts
  - This module is one layer of a defence-in-depth strategy, not a complete solution
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Confidence and risk levels
# ---------------------------------------------------------------------------

class Confidence(str, Enum):
    HIGH = "high"       # Very unlikely to be a false positive in production
    MEDIUM = "medium"   # Likely malicious but may fire on edge-case legitimate inputs
    LOW = "low"         # Weak signal — informational; combine with others


class RiskLevel(str, Enum):
    CRITICAL = "critical"   # Multiple high-confidence signals — block immediately
    HIGH = "high"           # At least one high-confidence signal
    MEDIUM = "medium"       # Medium-confidence signals or multiple low signals
    LOW = "low"             # Low-confidence signals only
    CLEAN = "clean"         # No signals detected


# ---------------------------------------------------------------------------
# Detection signal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectionSignal:
    """A single detected injection indicator."""

    rule_id: str
    confidence: Confidence
    category: str        # e.g. "instruction_override", "template_injection"
    description: str
    matched_text: str    # Excerpt of the text that triggered this rule (max 100 chars)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class InjectionReport:
    """Aggregated prompt injection detection result."""

    signals: list[DetectionSignal] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.CLEAN
    source_type: str = "direct"  # "direct" | "indirect" | "tool_output"

    @property
    def has_signals(self) -> bool:
        return len(self.signals) > 0

    @property
    def high_confidence_count(self) -> int:
        return sum(1 for s in self.signals if s.confidence == Confidence.HIGH)

    @property
    def signal_categories(self) -> list[str]:
        return list({s.category for s in self.signals})


# ---------------------------------------------------------------------------
# Detection rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Rule:
    rule_id: str
    confidence: Confidence
    category: str
    description: str
    pattern: re.Pattern  # Compiled regex


_DIRECT_RULES: list[_Rule] = [
    _Rule(
        rule_id="PI-D001",
        confidence=Confidence.HIGH,
        category="instruction_override",
        description="Explicit instruction to ignore previous instructions",
        pattern=re.compile(
            r"ignore\s+(all\s+)?(previous|above|prior|earlier|your)\s+instructions?",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        rule_id="PI-D002",
        confidence=Confidence.HIGH,
        category="instruction_override",
        description="Explicit instruction to disregard rules or guidelines",
        pattern=re.compile(
            r"disregard\s+(all\s+)?(your\s+)?(previous\s+)?(rules?|instructions?|guidelines?|constraints?|context)",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        rule_id="PI-D003",
        confidence=Confidence.HIGH,
        category="role_override",
        description="Attempt to redefine the model as a different, unrestricted entity",
        pattern=re.compile(
            r"(you\s+are\s+now|from\s+now\s+on\s+you\s+are|pretend\s+you\s+are|act\s+as)\s+(a\s+|an\s+)?(?!helpful|assistant|friendly)",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        rule_id="PI-D004",
        confidence=Confidence.HIGH,
        category="delimiter_injection",
        description="Attempt to inject ChatML or special model delimiters",
        pattern=re.compile(
            r"<\|?(im_start|im_end|system|user|assistant|endoftext)\|?>",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        rule_id="PI-D005",
        confidence=Confidence.HIGH,
        category="delimiter_injection",
        description="Markdown heading used to inject a system-like instruction block",
        pattern=re.compile(
            r"^#+\s*(system\s+prompt|system\s+message|instruction|context\s+override)",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    _Rule(
        rule_id="PI-D006",
        confidence=Confidence.MEDIUM,
        category="restriction_bypass",
        description="Request to operate without safety restrictions or guidelines",
        pattern=re.compile(
            r"(without|ignore|bypass|remove)\s+(your\s+)?(safety|content|ethical|guardrails?|restrictions?|filters?|policies)",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        rule_id="PI-D007",
        confidence=Confidence.MEDIUM,
        category="jailbreak_keyword",
        description="Known jailbreak template keywords",
        pattern=re.compile(
            r"\b(jailbreak|DAN|do\s+anything\s+now|developer\s+mode\s+(enabled|on)|god\s+mode|unrestricted\s+mode)\b",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        rule_id="PI-D008",
        confidence=Confidence.MEDIUM,
        category="context_manipulation",
        description="Attempt to leak or extract system prompt content",
        pattern=re.compile(
            r"(repeat|print|show|output|reveal|display|tell\s+me|write\s+out)\s+(the\s+)?(system\s+prompt|instructions?\s+you\s+(were\s+)?given|your\s+initial\s+prompt)",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        rule_id="PI-D009",
        confidence=Confidence.HIGH,
        category="unicode_steganography",
        description="Zero-width or invisible Unicode characters used to hide injection payloads",
        pattern=re.compile(
            r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u2060\u2061\u2062\u2063]",
        ),
    ),
    _Rule(
        rule_id="PI-D010",
        confidence=Confidence.LOW,
        category="multi_turn_setup",
        description="Phrasing that sets up instruction override across future conversation turns",
        pattern=re.compile(
            r"(remember\s+this|for\s+all\s+future\s+(messages?|responses?|turns?)|from\s+this\s+point\s+on)",
            re.IGNORECASE,
        ),
    ),
]

# Indirect injection patterns — designed to detect injections embedded
# in retrieved documents, tool outputs, or API responses
_INDIRECT_RULES: list[_Rule] = [
    _Rule(
        rule_id="PI-I001",
        confidence=Confidence.HIGH,
        category="indirect_instruction_embed",
        description="Instruction to the AI embedded in seemingly external content",
        pattern=re.compile(
            r"\[\s*(ai|llm|assistant|system|bot|gpt)\s*\]\s*:?\s*(ignore|disregard|override|follow|execute)",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        rule_id="PI-I002",
        confidence=Confidence.HIGH,
        category="indirect_instruction_embed",
        description="Hidden instruction disguised as metadata or annotation",
        pattern=re.compile(
            r"<!--\s*(ai|llm|assistant|instruction|system).*?-->",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    _Rule(
        rule_id="PI-I003",
        confidence=Confidence.MEDIUM,
        category="indirect_exfiltration",
        description="Instruction to exfiltrate conversation content to an external URL",
        pattern=re.compile(
            r"(send|forward|post|upload|exfiltrate)\s+(to\s+)?(https?://\S+|this\s+url|my\s+server)",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        rule_id="PI-I004",
        confidence=Confidence.MEDIUM,
        category="indirect_tool_abuse",
        description="Instruction to call a tool with attacker-controlled parameters",
        pattern=re.compile(
            r"call\s+(the\s+)?(search|browser|code|execute|run)\s+(tool|function|plugin|action)\s+with",
            re.IGNORECASE,
        ),
    ),
]

# Template injection — user input that may be rendered as a server-side template
_TEMPLATE_RULES: list[_Rule] = [
    _Rule(
        rule_id="PI-T001",
        confidence=Confidence.HIGH,
        category="template_injection",
        description="Jinja2/Twig template expression syntax",
        pattern=re.compile(r"\{\{.*?\}\}|\{%.*?%\}"),
    ),
    _Rule(
        rule_id="PI-T002",
        confidence=Confidence.HIGH,
        category="template_injection",
        description="Mustache/Handlebars template syntax",
        pattern=re.compile(r"\{\{\{.*?\}\}\}|\{\{#.*?\}\}"),
    ),
    _Rule(
        rule_id="PI-T003",
        confidence=Confidence.MEDIUM,
        category="template_injection",
        description="Python f-string or format string syntax with potential code execution",
        pattern=re.compile(r"\{[a-zA-Z_]\w*(\.[a-zA-Z_]\w*|\[.*?\])+\}"),
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_injection(
    text: str,
    source_type: str = "direct",
    include_template_checks: bool = True,
) -> InjectionReport:
    """
    Scan text for prompt injection signals.

    Args:
        text:                    The text to scan (user input or retrieved document).
        source_type:             "direct" for user input, "indirect" for retrieved content
                                 or tool output, "tool_output" for tool API responses.
        include_template_checks: If True, also scan for server-side template injection
                                 patterns. Disable for text that is known to contain
                                 legitimate template syntax.

    Returns:
        InjectionReport with all detected signals and an overall risk level.
    """
    signals: list[DetectionSignal] = []

    # Select which rule sets to apply
    rules_to_apply = list(_DIRECT_RULES)
    if source_type in ("indirect", "tool_output"):
        rules_to_apply = list(_INDIRECT_RULES)  # Indirect content gets its own rule set
    if include_template_checks:
        rules_to_apply.extend(_TEMPLATE_RULES)

    for rule in rules_to_apply:
        match = rule.pattern.search(text)
        if match:
            # Extract a safe excerpt (max 100 chars) of the matching text for logging
            matched = match.group(0)
            excerpt = (matched[:97] + "...") if len(matched) > 100 else matched
            signals.append(DetectionSignal(
                rule_id=rule.rule_id,
                confidence=rule.confidence,
                category=rule.category,
                description=rule.description,
                matched_text=excerpt,
            ))

    report = InjectionReport(signals=signals, source_type=source_type)
    report.risk_level = _compute_risk_level(signals)
    return report


def _compute_risk_level(signals: list[DetectionSignal]) -> RiskLevel:
    """Compute the overall risk level from a list of detection signals."""
    if not signals:
        return RiskLevel.CLEAN

    high_count = sum(1 for s in signals if s.confidence == Confidence.HIGH)
    medium_count = sum(1 for s in signals if s.confidence == Confidence.MEDIUM)
    low_count = sum(1 for s in signals if s.confidence == Confidence.LOW)

    if high_count >= 2:
        return RiskLevel.CRITICAL
    if high_count >= 1:
        return RiskLevel.HIGH
    if medium_count >= 2 or (medium_count >= 1 and low_count >= 1):
        return RiskLevel.MEDIUM
    if medium_count >= 1:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def is_clean(text: str, source_type: str = "direct") -> bool:
    """
    Convenience function that returns True if no injection signals are detected.

    Args:
        text:        Text to scan.
        source_type: "direct" | "indirect" | "tool_output"

    Returns:
        True if the report is CLEAN, False otherwise.
    """
    report = detect_injection(text, source_type=source_type)
    return report.risk_level == RiskLevel.CLEAN
