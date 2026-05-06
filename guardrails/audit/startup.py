from __future__ import annotations

from typing import Any

from guardrails.policy.loader import build_policy_decision_reason_code_inventory


def emit_startup_security_audit_event(
    *,
    logger: Any,
    app_version: str,
    environment: str,
    allowed_policy_decision_reason_codes: list[str],
) -> None:
    inventory = build_policy_decision_reason_code_inventory(
        allowed_policy_decision_reason_codes
    )

    payload = {
        "event": "startup_security_audit",
        "app_version": app_version,
        "environment": environment,
        "policy_decision_reason_code_inventory": list(inventory.reason_codes),
        "policy_decision_reason_code_inventory_sha256": inventory.sha256,
    }

    logger.info(payload)
