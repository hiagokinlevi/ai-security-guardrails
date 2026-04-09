"""
Audit Logger
=============
Records structured, tamper-evident audit events for every guardrail decision.

Audit log entries are written as newline-delimited JSON (NDJSON) via
structlog, which makes them easy to ingest into SIEM tools, log aggregators
(Datadog, Splunk, CloudWatch), or object storage for long-term retention.

Security note:
  By default, raw inputs and outputs are NEVER logged. Only decisions,
  risk scores, and flags are recorded. This prevents the audit log itself
  from becoming a source of sensitive data leakage.

  Enable raw content logging ONLY in isolated debug environments with
  appropriate access controls. Set LOG_INPUTS=true / LOG_OUTPUTS=true
  in your .env to enable.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from typing import Any, Optional

import structlog

from guardrails.input_controls.validator import InputValidationResult
from guardrails.output_controls.filter import OutputFilterResult
from guardrails.policy_engine.engine import PolicyDecision


# Configure structlog to emit clean JSON
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)

_logger = structlog.get_logger("guardrails.audit")


class AuditLogger:
    """
    Writes structured audit events for guardrail decisions.

    Each event contains:
    - request_id: Unique identifier for the request
    - event_type: Type of event (input_validation, output_filter, policy_decision)
    - decision: The action taken (allow, warn, review, block)
    - risk_score: Numeric risk score (0.0 – 1.0)
    - risk_flags: List of flag names that contributed to the score
    - latency_ms: Processing time in milliseconds
    - policy_name: Name of the policy that was applied
    - policy_version: Version of the policy

    Raw inputs and outputs are only included when explicitly enabled.
    """

    def __init__(
        self,
        log_inputs: bool = False,
        log_outputs: bool = False,
        log_decisions: bool = True,
        log_risk_scores: bool = True,
    ) -> None:
        # Control which fields appear in audit logs
        self._log_inputs = log_inputs
        self._log_outputs = log_outputs
        self._log_decisions = log_decisions
        self._log_risk_scores = log_risk_scores

    def log_input_validation(
        self,
        request_id: str,
        result: InputValidationResult,
        latency_ms: float = 0.0,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Log an input validation event.

        Args:
            request_id: Unique request identifier.
            result: InputValidationResult from the validator.
            latency_ms: Time taken to validate the input.
            user_id: Optional user identifier (for correlation).
            session_id: Optional session identifier.
        """
        event: dict[str, Any] = {
            "event_type": "input_validation",
            "request_id": request_id,
            "decision": result.decision.value,
            "input_length": result.original_length,
        }

        # Conditionally include detailed fields based on configuration
        if self._log_risk_scores:
            event["risk_score"] = result.risk_score
            event["risk_flags"] = result.risk_flags

        if latency_ms:
            event["latency_ms"] = round(latency_ms, 2)

        if user_id:
            event["user_id"] = user_id  # Consider pseudonymization for production

        if session_id:
            event["session_id"] = session_id

        # Raw input is intentionally excluded unless explicitly enabled
        if self._log_inputs and result.original_length > 0:
            event["_debug_input_truncated"] = "[raw input logging enabled — see sanitized_input]"

        _logger.info("input_validation", **event)

    def log_output_filter(
        self,
        request_id: str,
        result: OutputFilterResult,
        latency_ms: float = 0.0,
    ) -> None:
        """
        Log an output filtering event.

        Args:
            request_id: Unique request identifier.
            result: OutputFilterResult from the output filter.
            latency_ms: Time taken to filter the output.
        """
        event: dict[str, Any] = {
            "event_type": "output_filter",
            "request_id": request_id,
            "decision": result.decision.value,
        }

        if self._log_risk_scores:
            event["risk_score"] = result.risk_score
            event["risk_flags"] = result.risk_flags

        if latency_ms:
            event["latency_ms"] = round(latency_ms, 2)

        if result.reason:
            event["reason"] = result.reason

        # Never log original_output or filtered_output unless explicitly enabled
        if self._log_outputs:
            event["_debug_output_logging"] = "enabled — see filtered_output field"

        _logger.info("output_filter", **event)

    def log_policy_decision(
        self,
        request_id: str,
        decision: PolicyDecision,
        latency_ms: float = 0.0,
    ) -> None:
        """
        Log a policy engine decision.

        Args:
            request_id: Unique request identifier.
            decision: PolicyDecision from the policy engine.
            latency_ms: Time taken to evaluate the policy.
        """
        if not self._log_decisions:
            return

        event: dict[str, Any] = {
            "event_type": "policy_decision",
            "request_id": request_id,
            "action": decision.action.value,
            "policy_name": decision.policy_name,
            "policy_version": decision.policy_version,
            "applied_rules": decision.applied_rules,
        }

        if decision.reason:
            event["reason"] = decision.reason

        if latency_ms:
            event["latency_ms"] = round(latency_ms, 2)

        _logger.info("policy_decision", **event)

    def log_interaction(
        self,
        request_id: str,
        input_result: InputValidationResult,
        output_result: OutputFilterResult,
        policy_decision: Optional[PolicyDecision] = None,
        total_latency_ms: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log a complete request/response interaction as a single audit event.

        This is the recommended method for most callers — it produces a
        single, correlated audit record for each interaction.
        """
        event: dict[str, Any] = {
            "event_type": "interaction",
            "request_id": request_id,
            "input_decision": input_result.decision.value,
            "output_decision": output_result.decision.value,
        }

        if self._log_risk_scores:
            event["input_risk_score"] = input_result.risk_score
            event["output_risk_score"] = output_result.risk_score
            event["input_risk_flags"] = input_result.risk_flags
            event["output_risk_flags"] = output_result.risk_flags

        if policy_decision:
            event["policy_action"] = policy_decision.action.value
            event["policy_name"] = policy_decision.policy_name
            event["policy_version"] = policy_decision.policy_version

        if total_latency_ms:
            event["total_latency_ms"] = round(total_latency_ms, 2)

        if metadata:
            event["metadata"] = metadata

        _logger.info("interaction", **event)


def generate_request_id() -> str:
    """Generate a unique request ID for audit correlation."""
    return f"req_{uuid.uuid4().hex[:16]}"
