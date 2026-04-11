from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any


def _install_starlette_stubs() -> None:
    if "starlette" in sys.modules:
        return

    starlette = types.ModuleType("starlette")
    middleware = types.ModuleType("starlette.middleware")
    middleware_base = types.ModuleType("starlette.middleware.base")
    requests = types.ModuleType("starlette.requests")
    responses = types.ModuleType("starlette.responses")
    types_mod = types.ModuleType("starlette.types")

    class BaseHTTPMiddleware:
        def __init__(self, app: object) -> None:
            self.app = app

    class Response:
        def __init__(
            self,
            content: bytes | str = b"",
            status_code: int = 200,
            headers: dict[str, str] | None = None,
            media_type: str | None = None,
        ) -> None:
            self.body = content if isinstance(content, bytes) else content.encode("utf-8")
            self.status_code = status_code
            self.headers = headers or {}
            self.media_type = media_type
            self.body_iterator = self._iterate_body()

        async def _iterate_body(self):  # type: ignore[no-untyped-def]
            yield self.body

    class JSONResponse(Response):
        def __init__(
            self,
            content: Any,
            status_code: int = 200,
            headers: dict[str, str] | None = None,
        ) -> None:
            super().__init__(
                content=json.dumps(content).encode("utf-8"),
                status_code=status_code,
                headers=headers,
                media_type="application/json",
            )

    class Request:
        def __init__(self, path: str, body: bytes, headers: dict[str, str] | None = None) -> None:
            self.url = types.SimpleNamespace(path=path)
            self._body = body
            self.headers = headers or {}

        async def body(self) -> bytes:
            return self._body

    middleware_base.BaseHTTPMiddleware = BaseHTTPMiddleware
    requests.Request = Request
    responses.JSONResponse = JSONResponse
    responses.Response = Response
    types_mod.ASGIApp = object

    sys.modules["starlette"] = starlette
    sys.modules["starlette.middleware"] = middleware
    sys.modules["starlette.middleware.base"] = middleware_base
    sys.modules["starlette.requests"] = requests
    sys.modules["starlette.responses"] = responses
    sys.modules["starlette.types"] = types_mod


def _install_structlog_stub() -> None:
    if "structlog" in sys.modules and hasattr(sys.modules["structlog"], "configure"):
        return

    structlog = types.ModuleType("structlog")

    class _Logger:
        def info(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _TimeStamper:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {}

    class _JSONRenderer:
        def __call__(self, *_args: object, **_kwargs: object) -> str:
            return ""

    structlog.configure = lambda **_kwargs: None
    structlog.get_logger = lambda *_args, **_kwargs: _Logger()
    structlog.stdlib = types.SimpleNamespace(add_log_level=lambda *_args, **_kwargs: {})
    structlog.processors = types.SimpleNamespace(
        TimeStamper=_TimeStamper,
        JSONRenderer=_JSONRenderer,
    )
    sys.modules["structlog"] = structlog


_install_starlette_stubs()
_install_structlog_stub()
sys.modules.pop("middleware.fastapi_middleware", None)
fastapi_middleware = importlib.import_module("middleware.fastapi_middleware")

GuardrailsMiddleware = fastapi_middleware.GuardrailsMiddleware
Request = sys.modules["starlette.requests"].Request
Response = sys.modules["starlette.responses"].Response

POLICY_PATH = Path(__file__).resolve().parent.parent / "policies" / "default_policy.yaml"


async def _allow_request(_: object) -> Response:
    return Response(content=b'{"message":"processed"}', status_code=200)


def test_skip_header_is_ignored_by_default() -> None:
    middleware = GuardrailsMiddleware(app=object(), policy_path=str(POLICY_PATH))
    request = Request(
        path="/chat",
        body=json.dumps({"message": "A" * 10001}).encode("utf-8"),
        headers={"x-guardrails-skip": "true"},
    )

    response = asyncio.run(middleware.dispatch(request, _allow_request))

    assert response.status_code == 400
    payload = json.loads(response.body.decode("utf-8"))
    assert payload["error"] == "Your message could not be processed."


def test_skip_header_requires_matching_secret() -> None:
    middleware = GuardrailsMiddleware(
        app=object(),
        policy_path=str(POLICY_PATH),
        skip_header_secret="lane-only-secret",
    )
    request = Request(
        path="/chat",
        body=json.dumps({"message": "A" * 10001}).encode("utf-8"),
        headers={"x-guardrails-skip": "wrong-secret"},
    )

    response = asyncio.run(middleware.dispatch(request, _allow_request))

    assert response.status_code == 400
    payload = json.loads(response.body.decode("utf-8"))
    assert payload["error"] == "Your message could not be processed."


def test_skip_header_bypasses_guardrails_when_secret_matches() -> None:
    middleware = GuardrailsMiddleware(
        app=object(),
        policy_path=str(POLICY_PATH),
        skip_header_secret="lane-only-secret",
    )
    request = Request(
        path="/chat",
        body=json.dumps({"message": "A" * 10001}).encode("utf-8"),
        headers={"x-guardrails-skip": "lane-only-secret"},
    )

    response = asyncio.run(middleware.dispatch(request, _allow_request))

    assert response.status_code == 200
    assert json.loads(response.body.decode("utf-8"))["message"] == "processed"
