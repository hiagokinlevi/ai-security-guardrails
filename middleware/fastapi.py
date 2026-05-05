from __future__ import annotations

import json
import logging
import os
from typing import Any, Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


DEFAULT_MAX_REQUEST_BODY_BYTES = 1_048_576  # 1 MiB secure default


class RequestBodySizeGuardMiddleware(BaseHTTPMiddleware):
    """Fail closed when incoming request body exceeds configured raw-byte threshold."""

    def __init__(self, app: Any, max_request_body_bytes: int | None = None) -> None:
        super().__init__(app)
        self.max_request_body_bytes = (
            max_request_body_bytes
            if isinstance(max_request_body_bytes, int) and max_request_body_bytes > 0
            else _env_int("MAX_REQUEST_BODY_BYTES", DEFAULT_MAX_REQUEST_BODY_BYTES)
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Any]],
    ) -> Any:
        correlation_id = (
            request.headers.get("x-correlation-id")
            or request.headers.get("x-request-id")
            or "unknown"
        )

        content_length_header = request.headers.get("content-length")
        content_length: int | None = None
        if content_length_header is not None:
            try:
                content_length = int(content_length_header)
            except ValueError:
                content_length = None

        if content_length is not None and content_length > self.max_request_body_bytes:
            logger.info(
                json.dumps(
                    {
                        "event_type": "request_body_size_guard",
                        "correlation_id": correlation_id,
                        "content_length": content_length,
                        "max_request_body_bytes": self.max_request_body_bytes,
                        "decision": "block",
                        "decision_reason_code": "request_body_too_large",
                    }
                )
            )
            return JSONResponse(
                status_code=413,
                content={
                    "detail": "Request body too large",
                    "reason_code": "request_body_too_large",
                },
            )

        body = await request.body()
        body_size = len(body)
        if body_size > self.max_request_body_bytes:
            logger.info(
                json.dumps(
                    {
                        "event_type": "request_body_size_guard",
                        "correlation_id": correlation_id,
                        "content_length": body_size,
                        "max_request_body_bytes": self.max_request_body_bytes,
                        "decision": "block",
                        "decision_reason_code": "request_body_too_large",
                    }
                )
            )
            return JSONResponse(
                status_code=413,
                content={
                    "detail": "Request body too large",
                    "reason_code": "request_body_too_large",
                },
            )

        logger.info(
            json.dumps(
                {
                    "event_type": "request_body_size_guard",
                    "correlation_id": correlation_id,
                    "content_length": content_length if content_length is not None else body_size,
                    "max_request_body_bytes": self.max_request_body_bytes,
                    "decision": "allow",
                    "decision_reason_code": "request_body_within_limit",
                }
            )
        )

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(request.scope, receive)
        return await call_next(request)
