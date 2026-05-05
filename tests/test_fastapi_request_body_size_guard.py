import json
import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from middleware.fastapi import RequestBodySizeGuardMiddleware


def _build_app(limit: int) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestBodySizeGuardMiddleware, max_request_body_bytes=limit)

    @app.post("/ingest")
    async def ingest(payload: dict):
        return {"ok": True, "payload": payload}

    return app


def test_request_body_size_guard_allows_and_blocks(caplog):
    app = _build_app(limit=100)
    client = TestClient(app)

    with caplog.at_level(logging.INFO):
        allowed = client.post("/ingest", json={"text": "a" * 10}, headers={"x-correlation-id": "cid-allow"})
        blocked = client.post("/ingest", json={"text": "b" * 1000}, headers={"x-correlation-id": "cid-block"})

    assert allowed.status_code == 200
    assert blocked.status_code == 413
    assert blocked.json()["reason_code"] == "request_body_too_large"

    events = []
    for rec in caplog.records:
        try:
            payload = json.loads(rec.getMessage())
        except Exception:
            continue
        if payload.get("event_type") == "request_body_size_guard":
            events.append(payload)

    allow_events = [e for e in events if e.get("correlation_id") == "cid-allow"]
    block_events = [e for e in events if e.get("correlation_id") == "cid-block"]

    assert allow_events
    assert block_events
    assert allow_events[-1]["decision_reason_code"] == "request_body_within_limit"
    assert block_events[-1]["decision_reason_code"] == "request_body_too_large"
    assert block_events[-1]["decision"] == "block"
