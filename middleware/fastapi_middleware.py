from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Context-local request id for downstream logging/helpers
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    """Return current request_id from context."""
    return request_id_ctx.get()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Inject and propagate request id for each request lifecycle."""

    header_name = "X-Request-ID"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        incoming_request_id = request.headers.get(self.header_name)
        request_id = incoming_request_id or str(uuid.uuid4())

        # Store in FastAPI request state for app-level consumers
        request.state.request_id = request_id

        # Store in context var for utilities invoked deeper in stack
        token = request_id_ctx.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)

        # Echo back for client/server correlation
        response.headers[self.header_name] = request_id
        return response
