from cli.verifier import verify_audit_events


def test_verify_sequence_id_valid_ordering():
    events = [
        {"sequence_id": 1, "type": "input"},
        {"sequence_id": 2, "type": "policy_decision"},
        {"sequence_id": 3, "type": "output"},
    ]
    ok, msg = verify_audit_events(events)
    assert ok is True
    assert msg == "ok"


def test_verify_sequence_id_duplicate_fails():
    events = [
        {"sequence_id": 10, "type": "input"},
        {"sequence_id": 11, "type": "policy_decision"},
        {"sequence_id": 11, "type": "output"},
    ]
    ok, msg = verify_audit_events(events)
    assert ok is False
    assert "out of order or duplicate" in msg


def test_verify_sequence_id_gap_fails():
    events = [
        {"sequence_id": 21, "type": "input"},
        {"sequence_id": 22, "type": "policy_decision"},
        {"sequence_id": 24, "type": "output"},
    ]
    ok, msg = verify_audit_events(events)
    assert ok is False
    assert "gap" in msg
