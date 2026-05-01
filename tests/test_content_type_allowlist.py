from fastapi import FastAPI
from fastapi.testclient import TestClient

from middleware.fastapi_guardrails import GuardrailsMiddleware


def build_app(policy=None):
    app = FastAPI()
    app.add_middleware(GuardrailsMiddleware, policy=policy or {})

    @app.post("/llm/ingress")
    async def ingress(payload: dict):
        return {"ok": True, "echo": payload}

    return app


def test_accepts_application_json_content_type():
    app = build_app({"ingress": {"content_type_allowlist": ["application/json"]}})
    client = TestClient(app)

    r = client.post("/llm/ingress", json={"prompt": "hello"})

    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_rejects_unsupported_content_type_with_reason_code():
    app = build_app({"ingress": {"content_type_allowlist": ["application/json"]}})
    client = TestClient(app)

    r = client.post(
        "/llm/ingress",
        data="prompt=hello",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert r.status_code == 415
    body = r.json()
    assert body["error"] == "unsupported_media_type"
    assert body["reason_code"] == "unsupported_content_type"


def test_default_policy_is_json_only_when_allowlist_missing():
    app = build_app({})
    client = TestClient(app)

    r = client.post(
        "/llm/ingress",
        data="plain text",
        headers={"content-type": "text/plain"},
    )

    assert r.status_code == 415
    assert r.json()["reason_code"] == "unsupported_content_type"
