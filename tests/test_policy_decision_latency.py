from guardrails import policy_engine


def test_policy_decision_emits_non_negative_latency(monkeypatch):
    captured = {}

    def _capture(event):
        captured.update(event)

    monkeypatch.setattr(policy_engine, "emit_audit_event", _capture)

    result = policy_engine.evaluate_policy("hello world")

    assert "latency_ms" in result
    assert isinstance(result["latency_ms"], (int, float))
    assert result["latency_ms"] >= 0

    assert "latency_ms" in captured
    assert isinstance(captured["latency_ms"], (int, float))
    assert captured["latency_ms"] >= 0
