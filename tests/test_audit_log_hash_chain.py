import json

from security.logger import SecurityEventLogger, verify_log_chain


def test_hash_chain_integrity(tmp_path):
    log_file = tmp_path / "security_events.jsonl"
    logger = SecurityEventLogger(log_file)

    first = logger.append_event("input_scan", {"risk_score": 0.2, "decision": "allow"})
    second = logger.append_event("output_scan", {"risk_score": 0.8, "decision": "block"})

    assert len(first["prev_hash"]) == 64
    assert len(first["event_hash"]) == 64
    assert second["prev_hash"] == first["event_hash"]

    ok, broken_at = verify_log_chain(log_file)
    assert ok is True
    assert broken_at is None


def test_hash_chain_tamper_detection(tmp_path):
    log_file = tmp_path / "security_events.jsonl"
    logger = SecurityEventLogger(log_file)

    logger.append_event("input_scan", {"risk_score": 0.1, "decision": "allow"})
    logger.append_event("policy_eval", {"rule": "pii", "decision": "warn"})
    logger.append_event("output_scan", {"risk_score": 0.9, "decision": "block"})

    lines = log_file.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["payload"]["decision"] = "block"  # historical tamper
    lines[0] = json.dumps(tampered, ensure_ascii=False)
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, broken_at = verify_log_chain(log_file)
    assert ok is False
    assert broken_at == 1
