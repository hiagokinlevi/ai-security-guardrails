import uuid

from guardrails.audit import AuditLogger, set_correlation_id


def test_same_correlation_id_across_request_lifecycle_events() -> None:
    inbound = str(uuid.uuid4())
    set_correlation_id(inbound)

    logger = AuditLogger()

    e1 = logger.emit(
        event_type="input_validation",
        action="validate_input",
        decision="allow",
        reason="ok",
    )
    e2 = logger.emit(
        event_type="policy_decision",
        action="evaluate_policy",
        decision="warn",
        reason="suspicious_pattern",
    )
    e3 = logger.emit(
        event_type="output_filtering",
        action="filter_output",
        decision="allow",
        reason="redaction_applied",
    )

    assert e1.correlation_id == inbound
    assert e2.correlation_id == inbound
    assert e3.correlation_id == inbound
    assert {evt.correlation_id for evt in logger.events} == {inbound}
