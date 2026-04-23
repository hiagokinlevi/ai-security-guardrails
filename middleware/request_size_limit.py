from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


DEFAULT_MAX_REQUEST_BODY_BYTES = 1_048_576  # 1 MiB


def _get_max_body_bytes() -> int:
    raw = os.getenv("MAX_REQUEST_BODY_BYTES", str(DEFAULT_MAX_REQUEST_BODY_BYTES)).strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_REQUEST_BODY_BYTES
    return value if value > 0 else DEFAULT_MAX_REQUEST_BODY_BYTES


def _emit_payload_too_large_audit_event(request: Request, max_bytes: int, actual_bytes: int) -> None:
    event: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "security.block",
        "reason": "payload_too_large",
        "path": str(request.url.path),
        "method": request.method,
        "max_bytes": max_bytes,
        "actual_bytes": actual_bytes,
        "client_ip": request.client.host if request.client else None,
    }

    logger = getattr(request.app.state, "audit_logger", None)
    if logger is not None and hasattr(logger, "info"):
        logger.info(event)
        return

    print(json.dumps(event, separators=(",", ":"), default=str))


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Blocks requests with payloads larger than configured byte threshold."""

    def __init__(self, app, max_body_bytes: int | None = None) -> None:
        super().__init__(app)
        self.max_body_bytes = max_body_bytes or _get_max_body_bytes()

    async def dispatch(self, request: Request, call_next):
        content_length_header = request.headers.get("content-length")
        if content_length_header is not None:
            try:
                content_length = int(content_length_header)
            except ValueError:
                content_length = None
            if content_length is not None and content_length > self.max_body_bytes:
                _emit_payload_too_large_audit_event(request, self.max_body_bytes, content_length)
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "payload_too_large",
                            "message": "Request payload exceeds allowed maximum size.",
                            "max_bytes": self.max_body_bytes,
                            "actual_bytes": content_length,
                        }
                    },
                )

        body = await request.body()
        body_size = len(body)
        if body_size > self.max_body_bytes:
            _emit_payload_too_large_audit_event(request, self.max_body_bytes, body_size)
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "payload_too_large",
                        "message": "Request payload exceeds allowed maximum size.",
                        "max_bytes": self.max_body_bytes,
                        "actual_bytes": body_size,
                    }
                },
            )

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(request.scope, receive=receive)
        return await call_next(request)
