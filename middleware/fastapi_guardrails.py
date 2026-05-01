from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from guardrails.audit import emit_audit_event


REASON_UNSUPPORTED_CONTENT_TYPE = "unsupported_content_type"
DEFAULT_ALLOWED_CONTENT_TYPES = ["application/json"]


class GuardrailsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, policy: dict[str, Any] | None = None):
        super().__init__(app)
        self.policy = policy or {}
        self.allowed_content_types = self._load_allowed_content_types(self.policy)

    @staticmethod
    def _load_allowed_content_types(policy: dict[str, Any]) -> set[str]:
        ingress = policy.get("ingress", {}) if isinstance(policy, dict) else {}
        allowlist = ingress.get("content_type_allowlist", DEFAULT_ALLOWED_CONTENT_TYPES)
        if not isinstance(allowlist, list) or not allowlist:
            allowlist = DEFAULT_ALLOWED_CONTENT_TYPES
        normalized = {str(v).strip().lower() for v in allowlist if str(v).strip()}
        return normalized or set(DEFAULT_ALLOWED_CONTENT_TYPES)

    async def dispatch(self, request: Request, call_next):
        # Enforce before body parsing (fail-closed)
        content_type_header = (request.headers.get("content-type") or "").strip().lower()
        media_type = content_type_header.split(";", 1)[0].strip() if content_type_header else ""

        if media_type not in self.allowed_content_types:
            emit_audit_event(
                {
                    "event_type": "ingress_rejected",
                    "decision": "block",
                    "reason_code": REASON_UNSUPPORTED_CONTENT_TYPE,
                    "details": {
                        "received_content_type": media_type or None,
                        "allowed_content_types": sorted(self.allowed_content_types),
                        "path": str(request.url.path),
                        "method": request.method,
                    },
                }
            )
            return JSONResponse(
                status_code=415,
                content={
                    "error": "unsupported_media_type",
                    "reason_code": REASON_UNSUPPORTED_CONTENT_TYPE,
                },
            )

        return await call_next(request)
