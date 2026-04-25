import pytest

from guardrails.audit import (
    AUDIT_SCHEMA_VERSION,
    AuditConfig,
    AuditLogger,
    AuditSchemaVersionMismatchError,
)


def test_startup_schema_version_match_succeeds_and_emits_required_field() -> None:
    logger = AuditLogger(config=AuditConfig(expected_schema_version=AUDIT_SCHEMA_VERSION))

    event = logger.emit("policy_decision", {"decision": "allow"})

    assert event.schema_version == AUDIT_SCHEMA_VERSION


def test_startup_schema_version_mismatch_fails_closed() -> None:
    with pytest.raises(AuditSchemaVersionMismatchError):
        AuditLogger(config=AuditConfig(expected_schema_version="999"))
