from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from guardrails.schemas import DecisionCode
from middleware.fastapi_guardrails import GuardrailsMiddleware


class _AuditCollector:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event):
        self.events.append(event)


def _build_app(max_request_bytes: int, collector: _AuditCollector) -> TestClient:
    app = FastAPI()
    app.add_middleware(
        GuardrailsMiddleware,
        max_request_bytes=max_request_bytes,
        audit_logger=collector,
    )

    @app.post("/echo")
    async def echo(payload: dict):
        return payload

    return TestClient(app)


def test_oversized_request_is_blocked_and_emits_input_too_large_decision():
    collector = _AuditCollector()
    client = _build_app(max_request_bytes=64, collector=collector)

    # Guaranteed >64 bytes body
    payload = {"text": "x" * 512}
    response = client.post("/echo", content=json.dumps(payload), headers={"content-type": "application/json"})

    assert response.status_code in (400, 413)
    assert collector.events, "expected an audit event for blocked oversized request"

    event = collector.events[-1]
    # Support pydantic model or dict payloads
    data = event.model_dump() if hasattr(event, "model_dump") else dict(event)

    assert data["decision"] == DecisionCode.INPUT_TOO_LARGE or data["decision"] == DecisionCode.INPUT_TOO_LARGE.value
    assert "too large" in data["reason"].lower()
