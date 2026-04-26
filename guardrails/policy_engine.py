from __future__ import annotations

import time
from typing import Any, Dict

from guardrails.audit import emit_audit_event


def evaluate_policy(input_text: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Evaluate policy and emit structured audit event with latency.

    Returns a decision dictionary with at least:
      - action: allow|warn|block
      - reasons: list[str]
      - latency_ms: float
    """
    context = context or {}

    start = time.perf_counter()
    # Existing evaluation logic (kept intentionally simple/minimal here)
    reasons: list[str] = []
    action = "allow"

    lowered = input_text.lower()
    if "ignore previous instructions" in lowered or "reveal your system prompt" in lowered:
        action = "block"
        reasons.append("prompt_injection_pattern_detected")

    latency_ms = max((time.perf_counter() - start) * 1000.0, 0.0)

    decision_event = {
        "event_type": "policy_decision",
        "action": action,
        "reasons": reasons,
        "latency_ms": latency_ms,
        "context": context,
    }
    emit_audit_event(decision_event)

    return {
        "action": action,
        "reasons": reasons,
        "latency_ms": latency_ms,
    }
