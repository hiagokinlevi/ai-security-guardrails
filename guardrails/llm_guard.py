from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class GuardrailBlockedError(RuntimeError):
    """Raised when guardrails block an LLM request or response."""


@dataclass
class GuardedResult:
    """Structured result returned by LLMGuard wrapper."""

    output: str
    input_blocked: bool = False
    output_blocked: bool = False
    input_reason: str | None = None
    output_reason: str | None = None


class LLMGuard:
    """
    Minimal provider-agnostic LLM guardrails wrapper.

    - Pre-check: prompt injection scan
    - Post-checks: PII redaction + output filtering

    The wrapper accepts callables to keep integration simple and framework-neutral.
    """

    def __init__(
        self,
        scan_prompt_injection: Callable[[str], Any],
        redact_pii: Callable[[str], str],
        filter_output: Callable[[str], Any],
        *,
        fail_closed: bool = True,
    ) -> None:
        self._scan_prompt_injection = scan_prompt_injection
        self._redact_pii = redact_pii
        self._filter_output = filter_output
        self._fail_closed = fail_closed

    @staticmethod
    def _is_blocked(result: Any) -> bool:
        if isinstance(result, bool):
            return result
        if isinstance(result, dict):
            for key in ("blocked", "is_blocked", "block"):
                if key in result:
                    return bool(result[key])
            decision = result.get("decision")
            if isinstance(decision, str):
                return decision.lower() == "block"
        decision = getattr(result, "decision", None)
        if isinstance(decision, str):
            return decision.lower() == "block"
        for attr in ("blocked", "is_blocked", "block"):
            if hasattr(result, attr):
                return bool(getattr(result, attr))
        return False

    @staticmethod
    def _reason(result: Any, default: str) -> str:
        if isinstance(result, dict):
            return str(result.get("reason") or result.get("message") or default)
        for attr in ("reason", "message"):
            val = getattr(result, attr, None)
            if val:
                return str(val)
        return default

    def __call__(self, llm_call: Callable[[str], str], prompt: str) -> GuardedResult:
        try:
            pre = self._scan_prompt_injection(prompt)
            if self._is_blocked(pre):
                reason = self._reason(pre, "prompt blocked by injection scan")
                raise GuardrailBlockedError(reason)
        except GuardrailBlockedError:
            raise
        except Exception as exc:  # defensive: scanner failure
            if self._fail_closed:
                raise GuardrailBlockedError(f"pre-check failed: {exc}") from exc

        raw_output = llm_call(prompt)
        if not isinstance(raw_output, str):
            raw_output = str(raw_output)

        redacted_output = self._redact_pii(raw_output)

        try:
            post = self._filter_output(redacted_output)
            if self._is_blocked(post):
                reason = self._reason(post, "output blocked by policy filter")
                raise GuardrailBlockedError(reason)
        except GuardrailBlockedError:
            raise
        except Exception as exc:  # defensive: filter failure
            if self._fail_closed:
                raise GuardrailBlockedError(f"post-check failed: {exc}") from exc

        return GuardedResult(output=redacted_output)


__all__ = ["LLMGuard", "GuardedResult", "GuardrailBlockedError"]
