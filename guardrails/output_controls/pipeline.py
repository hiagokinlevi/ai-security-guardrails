"""
LLM Output Filtering Pipeline
================================
A composable, ordered pipeline of filter stages that processes LLM responses
before they are returned to users.

Each stage in the pipeline can:
  - PASS the output unchanged (clean, no action)
  - REDACT sensitive content (mutates the text, continues pipeline)
  - FLAG the output for human review (annotates, continues pipeline)
  - BLOCK the output entirely (pipeline short-circuits, returns safe error)

Pipeline semantics:
  - Stages run in registration order
  - A BLOCK decision from any stage immediately halts the pipeline
  - Flags are accumulated across all stages
  - The final output is the text after all redaction stages have run
  - A cumulative risk score aggregates weights from all stages

Built-in stages (registered in DEFAULT_PIPELINE order):
  1. SecretLeakStage     — blocks outputs containing private keys or AWS credentials
  2. PiiRedactionStage   — redacts emails, SSNs, phone numbers, credit cards
  3. SystemPromptLeakStage — flags probable system prompt leakage
  4. InternalNetworkStage — redacts internal IP addresses
  5. PolicyViolationStage — custom keyword/regex blocklist stage

Usage:
    from guardrails.output_controls.pipeline import (
        OutputPipeline,
        DEFAULT_PIPELINE,
        PipelineResult,
    )

    result = DEFAULT_PIPELINE.run("The user's email is bob@example.com")
    print(result.decision)         # PipelineDecision.PASS_REDACTED
    print(result.final_output)     # "The user's email is [REDACTED:EMAIL]"
    print(result.stage_flags)      # [("pii_redaction", "email")]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Pipeline decision enum
# ---------------------------------------------------------------------------

class PipelineDecision(str, Enum):
    PASS            = "pass"           # Clean — no changes made
    PASS_REDACTED   = "pass_redacted"  # Some redaction applied, safe to return
    FLAGGED         = "flagged"        # Flagged for review, return with annotation
    BLOCK           = "block"          # Must not return to user


# ---------------------------------------------------------------------------
# Stage result and pipeline result
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """
    Output of a single pipeline stage.

    Attributes:
        stage_name:  Human-readable identifier for this stage.
        decision:    What this stage decided (does not override later stages).
        output_text: The (possibly mutated) output text after this stage.
        flags:       List of flag identifiers raised by this stage.
        risk_weight: Contribution to the cumulative risk score (0.0–1.0).
        note:        Human-readable explanation (populated on non-PASS decisions).
    """
    stage_name:  str
    decision:    PipelineDecision
    output_text: str
    flags:       list[str]  = field(default_factory=list)
    risk_weight: float      = 0.0
    note:        str        = ""


@dataclass
class PipelineResult:
    """
    Aggregated result of running a full pipeline.

    Attributes:
        decision:       Final decision across all stages (worst-case).
        final_output:   Text after all stages ran (may be redacted or safe error).
        risk_score:     Cumulative risk score [0.0, 1.0].
        stage_flags:    All (stage_name, flag) pairs raised.
        stages_run:     Ordered list of StageResult objects.
        blocked_by:     Stage name that issued a BLOCK, if any.
    """
    decision:     PipelineDecision
    final_output: str
    risk_score:   float
    stage_flags:  list[tuple[str, str]]       = field(default_factory=list)
    stages_run:   list[StageResult]           = field(default_factory=list)
    blocked_by:   Optional[str]               = None

    @property
    def was_blocked(self) -> bool:
        return self.decision == PipelineDecision.BLOCK

    @property
    def was_redacted(self) -> bool:
        return self.decision == PipelineDecision.PASS_REDACTED

    @property
    def all_flags(self) -> list[str]:
        """Flat list of all flag identifiers from all stages."""
        return [flag for _, flag in self.stage_flags]


# ---------------------------------------------------------------------------
# Safe error message (returned when output is blocked)
# ---------------------------------------------------------------------------

_SAFE_ERROR = (
    "I'm unable to provide that response. "
    "Please contact support if you believe this is an error."
)


# ---------------------------------------------------------------------------
# Abstract stage protocol
# ---------------------------------------------------------------------------

class FilterStage:
    """
    Base class for a pipeline filter stage.

    Subclass this and implement `process()`. Register instances with
    OutputPipeline.add_stage() or pass them to the OutputPipeline constructor.
    """

    name: str = "base_stage"

    def process(self, text: str) -> StageResult:
        """
        Process the text and return a StageResult.

        Args:
            text: Current output text (may already be partially redacted).

        Returns:
            StageResult — must not modify text in place; return the new text
            in StageResult.output_text.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Built-in stages
# ---------------------------------------------------------------------------

class SecretLeakStage(FilterStage):
    """
    BLOCK stage: detects private key material and cloud credential secrets.

    Any output containing these patterns is immediately blocked regardless
    of other settings — secrets must never reach users.
    """

    name = "secret_leak"

    _PATTERNS: list[tuple[str, str, float]] = [
        (r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----", "private_key", 1.0),
        (r"\bAKIA[0-9A-Z]{16}\b", "aws_access_key_id", 0.95),
        (r"aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*\S+", "aws_secret_key", 0.95),
        (r"(github_pat|ghp_)[0-9a-zA-Z]{36}", "github_pat", 0.9),
        (r"(sk-[a-zA-Z0-9]{32,})", "openai_api_key", 0.9),
    ]

    def process(self, text: str) -> StageResult:
        flags: list[str] = []
        weight = 0.0
        for pattern, flag, w in self._PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                flags.append(flag)
                weight = max(weight, w)

        if flags:
            return StageResult(
                stage_name=self.name,
                decision=PipelineDecision.BLOCK,
                output_text=_SAFE_ERROR,
                flags=flags,
                risk_weight=weight,
                note=f"Secret leak detected: {flags}. Output blocked.",
            )
        return StageResult(
            stage_name=self.name,
            decision=PipelineDecision.PASS,
            output_text=text,
        )


class PiiRedactionStage(FilterStage):
    """
    REDACT stage: replaces PII patterns with [REDACTED:<TYPE>] tokens.

    Does not block — redacts and continues. Flags each type of PII found.
    """

    name = "pii_redaction"

    _RULES: list[tuple[re.Pattern, str, str, float]] = [
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
         "[REDACTED:EMAIL]", "email", 0.2),
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
         "[REDACTED:SSN]", "us_ssn", 0.5),
        (re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
         "[REDACTED:CREDIT_CARD]", "credit_card", 0.5),
        (re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
         "[REDACTED:PHONE]", "phone_number", 0.2),
        # UK National Insurance number
        (re.compile(r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"),
         "[REDACTED:UK_NIN]", "uk_nin", 0.5),
        # EU IBAN (basic format)
        (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b"),
         "[REDACTED:IBAN]", "iban", 0.4),
    ]

    def process(self, text: str) -> StageResult:
        current = text
        flags: list[str] = []
        weight = 0.0
        for pattern, replacement, flag, w in self._RULES:
            if pattern.search(current):
                current = pattern.sub(replacement, current)
                flags.append(flag)
                weight += w

        if flags:
            return StageResult(
                stage_name=self.name,
                decision=PipelineDecision.PASS_REDACTED,
                output_text=current,
                flags=flags,
                risk_weight=min(weight, 1.0),
                note=f"PII redacted: {flags}",
            )
        return StageResult(
            stage_name=self.name,
            decision=PipelineDecision.PASS,
            output_text=current,
        )


class SystemPromptLeakStage(FilterStage):
    """
    FLAG stage: detects probable system prompt leakage.

    Does not block — flags for review so a human can evaluate context.
    System prompt leakage is ambiguous: "You are a helpful assistant" may
    appear in legitimate outputs.
    """

    name = "system_prompt_leak"

    _PATTERNS: list[tuple[str, str]] = [
        (r"(your system prompt|the system message|system prompt says)", "explicit_system_prompt_mention"),
        (r"(you are a|your role is|your instructions are)\b", "role_description_leakage"),
        (r"(as an ai language model|as an llm|as a large language model)\b", "ai_identity_leakage"),
        (r"(do not tell users|keep this confidential|this is a secret instruction)", "hidden_instruction_leakage"),
    ]

    def process(self, text: str) -> StageResult:
        flags: list[str] = []
        for pattern, flag in self._PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                flags.append(flag)

        if flags:
            return StageResult(
                stage_name=self.name,
                decision=PipelineDecision.FLAGGED,
                output_text=text,
                flags=flags,
                risk_weight=0.4,
                note=f"Possible system prompt leakage: {flags}",
            )
        return StageResult(
            stage_name=self.name,
            decision=PipelineDecision.PASS,
            output_text=text,
        )


class InternalNetworkStage(FilterStage):
    """
    REDACT stage: replaces internal/private IP addresses.

    Private IP ranges (RFC 1918 + loopback) should not appear in LLM
    responses as they expose network topology.
    """

    name = "internal_network"

    _PRIVATE_IP = re.compile(
        r"\b("
        r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        r"|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
        r"|192\.168\.\d{1,3}\.\d{1,3}"
        r"|127\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        r")\b"
    )

    def process(self, text: str) -> StageResult:
        if self._PRIVATE_IP.search(text):
            redacted = self._PRIVATE_IP.sub("[REDACTED:PRIVATE_IP]", text)
            return StageResult(
                stage_name=self.name,
                decision=PipelineDecision.PASS_REDACTED,
                output_text=redacted,
                flags=["private_ip_address"],
                risk_weight=0.25,
                note="Internal IP address redacted from output.",
            )
        return StageResult(
            stage_name=self.name,
            decision=PipelineDecision.PASS,
            output_text=text,
        )


class PolicyViolationStage(FilterStage):
    """
    Configurable BLOCK stage for custom keyword and regex blocklists.

    Useful for blocking topic areas (e.g., competitor names, internal
    project codenames) or high-risk phrases specific to your deployment.

    Args:
        block_patterns: List of regex patterns that trigger a BLOCK.
        flag_patterns:  List of (regex, flag_name) patterns that trigger FLAGS.
        stage_name:     Override the stage name for multi-instance use.
    """

    name = "policy_violation"

    def __init__(
        self,
        block_patterns: Optional[list[str]] = None,
        flag_patterns: Optional[list[tuple[str, str]]] = None,
        stage_name: str = "policy_violation",
    ) -> None:
        self.name = stage_name
        self._block = [re.compile(p, re.IGNORECASE) for p in (block_patterns or [])]
        self._flag  = [
            (re.compile(p, re.IGNORECASE), flag)
            for p, flag in (flag_patterns or [])
        ]

    def process(self, text: str) -> StageResult:
        # Check block patterns first
        for pattern in self._block:
            if pattern.search(text):
                return StageResult(
                    stage_name=self.name,
                    decision=PipelineDecision.BLOCK,
                    output_text=_SAFE_ERROR,
                    flags=["policy_block"],
                    risk_weight=1.0,
                    note=f"Output blocked by policy pattern: {pattern.pattern!r}",
                )

        # Check flag patterns
        flags: list[str] = []
        weight = 0.0
        for pattern, flag in self._flag:
            if pattern.search(text):
                flags.append(flag)
                weight += 0.3

        if flags:
            return StageResult(
                stage_name=self.name,
                decision=PipelineDecision.FLAGGED,
                output_text=text,
                flags=flags,
                risk_weight=min(weight, 1.0),
            )

        return StageResult(
            stage_name=self.name,
            decision=PipelineDecision.PASS,
            output_text=text,
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

# Decision priority order (higher index = more severe)
_DECISION_ORDER: dict[PipelineDecision, int] = {
    PipelineDecision.PASS:          0,
    PipelineDecision.PASS_REDACTED: 1,
    PipelineDecision.FLAGGED:       2,
    PipelineDecision.BLOCK:         3,
}


class OutputPipeline:
    """
    Ordered sequence of FilterStage objects that processes LLM output.

    Stages are run in registration order. A BLOCK from any stage terminates
    the pipeline immediately. All other stage results are accumulated.

    Args:
        stages: Initial list of FilterStage objects.
    """

    def __init__(self, stages: Optional[list[FilterStage]] = None) -> None:
        self._stages: list[FilterStage] = list(stages or [])

    def add_stage(self, stage: FilterStage) -> "OutputPipeline":
        """Append a stage. Returns self for chaining."""
        self._stages.append(stage)
        return self

    def run(self, text: str) -> PipelineResult:
        """
        Run the pipeline on the given text.

        Args:
            text: Raw LLM output to filter.

        Returns:
            PipelineResult with final decision, filtered text, and audit trail.
        """
        current_text = text
        all_stage_results: list[StageResult] = []
        all_flags: list[tuple[str, str]] = []
        cumulative_risk = 0.0
        worst_decision = PipelineDecision.PASS
        blocked_by: Optional[str] = None

        for stage in self._stages:
            result = stage.process(current_text)
            all_stage_results.append(result)

            # Accumulate flags
            for flag in result.flags:
                all_flags.append((stage.name, flag))

            # Accumulate risk
            cumulative_risk += result.risk_weight

            # Track worst decision seen
            if _DECISION_ORDER[result.decision] > _DECISION_ORDER[worst_decision]:
                worst_decision = result.decision

            # Update current text (stages may redact)
            current_text = result.output_text

            # Short-circuit on BLOCK
            if result.decision == PipelineDecision.BLOCK:
                blocked_by = stage.name
                break

        return PipelineResult(
            decision=worst_decision,
            final_output=current_text,
            risk_score=round(min(cumulative_risk, 1.0), 4),
            stage_flags=all_flags,
            stages_run=all_stage_results,
            blocked_by=blocked_by,
        )

    def __len__(self) -> int:
        return len(self._stages)


# ---------------------------------------------------------------------------
# Default pipeline
# ---------------------------------------------------------------------------

def build_default_pipeline() -> OutputPipeline:
    """
    Build and return the default output filtering pipeline.

    Stage order:
      1. SecretLeakStage        — block credentials immediately
      2. PiiRedactionStage      — redact PII (email, SSN, CC, phone, IBAN)
      3. SystemPromptLeakStage  — flag probable system prompt leakage
      4. InternalNetworkStage   — redact internal IP addresses
    """
    return OutputPipeline(stages=[
        SecretLeakStage(),
        PiiRedactionStage(),
        SystemPromptLeakStage(),
        InternalNetworkStage(),
    ])


#: Module-level default pipeline instance (ready to use).
DEFAULT_PIPELINE: OutputPipeline = build_default_pipeline()
