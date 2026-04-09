"""
FastAPI Guardrails Middleware
==============================
ASGI middleware that wraps all LLM-bound requests with input validation
and output filtering.

Usage:
    from fastapi import FastAPI
    from middleware.fastapi_middleware import GuardrailsMiddleware

    app = FastAPI()
    app.add_middleware(
        GuardrailsMiddleware,
        policy_path="policies/default_policy.yaml",
    )

How it works:
    1. On each request, the middleware checks for an "x-guardrails-skip"
       header. Requests with this header bypass guardrails (useful for
       internal health-check endpoints). Only use this for non-LLM routes.
    2. For routes matching the monitored path prefix (/chat by default),
       the middleware reads the request body, validates the input, and
       either blocks the request or allows it to proceed.
    3. After the inner handler runs, the middleware intercepts the response
       and applies output filtering before returning it to the client.
    4. All decisions are logged to the structured audit logger.

Limitations:
    - Streaming responses (SSE / WebSocket) are not yet supported.
      For streaming, apply guardrails at the application layer.
    - Only JSON request bodies with a "message" or "content" field
      are inspected. Extend _extract_user_message() for other formats.
"""

from __future__ import annotations

import json
import time
from typing import Awaitable, Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from guardrails.audit.logger import AuditLogger, generate_request_id
from guardrails.input_controls.validator import InputDecision, validate_input
from guardrails.output_controls.filter import OutputDecision, filter_output
from guardrails.policy_engine.engine import PolicyEngine


class GuardrailsMiddleware(BaseHTTPMiddleware):
    """
    Starlette/FastAPI ASGI middleware that applies security guardrails
    to all matching routes.
    """

    def __init__(
        self,
        app: ASGIApp,
        policy_path: str = "policies/default_policy.yaml",
        monitored_prefix: str = "/chat",  # Only inspect requests under this path
        log_inputs: bool = False,
        log_outputs: bool = False,
    ) -> None:
        super().__init__(app)
        # Load the policy — fail fast if the file is missing or invalid
        self._engine = PolicyEngine.from_file(policy_path)
        self._monitored_prefix = monitored_prefix
        self._audit = AuditLogger(
            log_inputs=log_inputs,
            log_outputs=log_outputs,
            log_decisions=self._engine.policy.log_decisions,
            log_risk_scores=self._engine.policy.log_risk_scores,
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """
        Main middleware dispatch method.

        Called for every incoming HTTP request. Routes not matching the
        monitored prefix pass through without inspection.
        """
        # Skip guardrails for non-monitored routes (e.g., health checks, docs)
        if not request.url.path.startswith(self._monitored_prefix):
            return await call_next(request)

        # Allow explicit bypass via header (for internal/trusted callers only)
        if request.headers.get("x-guardrails-skip") == "true":
            return await call_next(request)

        request_id = generate_request_id()
        start_time = time.monotonic()

        # --- Input validation ---
        user_message = await self._extract_user_message(request)
        if user_message is None:
            # Cannot extract input — let the request proceed unvalidated
            # Log this as a gap so it can be reviewed
            return await call_next(request)

        policy = self._engine.policy
        input_result = validate_input(
            user_message,
            max_length=policy.input_max_length,
            risk_threshold=policy.input_risk_threshold,
        )
        policy_decision = self._engine.evaluate_input(
            input_risk_score=input_result.risk_score,
            flags=input_result.risk_flags,
        )

        input_latency_ms = (time.monotonic() - start_time) * 1000
        self._audit.log_input_validation(
            request_id=request_id,
            result=input_result,
            latency_ms=input_latency_ms,
        )
        self._audit.log_policy_decision(
            request_id=request_id,
            decision=policy_decision,
            latency_ms=input_latency_ms,
        )

        # Block the request if the policy requires it
        if input_result.decision == InputDecision.BLOCK:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Your message could not be processed.",
                    "request_id": request_id,
                },
                headers={"x-request-id": request_id},
            )

        # --- Process request ---
        response = await call_next(request)

        # --- Output filtering ---
        # Read the response body for filtering
        # Note: this buffers the full response in memory — not suitable for large streaming responses
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        try:
            response_data = json.loads(response_body)
            model_text = self._extract_model_output(response_data)
        except (json.JSONDecodeError, KeyError):
            # Cannot parse the response — return it as-is
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        output_result = filter_output(
            model_text,
            redact_pii=policy.redact_pii,
            risk_threshold=policy.output_risk_threshold,
        )

        output_latency_ms = (time.monotonic() - start_time) * 1000
        self._audit.log_output_filter(
            request_id=request_id,
            result=output_result,
            latency_ms=output_latency_ms,
        )

        # If output is blocked, return a safe error
        if output_result.decision == OutputDecision.BLOCK:
            return JSONResponse(
                status_code=200,  # Keep 200 to avoid leaking block reason to client
                content={
                    "message": output_result.filtered_output,
                    "request_id": request_id,
                },
                headers={"x-request-id": request_id},
            )

        # Replace model output with filtered version in response
        self._inject_filtered_output(response_data, output_result.filtered_output)
        filtered_body = json.dumps(response_data).encode()

        return Response(
            content=filtered_body,
            status_code=response.status_code,
            headers={
                **dict(response.headers),
                "x-request-id": request_id,
                "content-length": str(len(filtered_body)),
            },
            media_type="application/json",
        )

    @staticmethod
    async def _extract_user_message(request: Request) -> Optional[str]:
        """
        Extract the user's message text from the request body.

        Supports:
        - {"message": "..."} — simple chatbot format
        - {"messages": [{"role": "user", "content": "..."}]} — OpenAI format
        """
        try:
            body = await request.body()
            data = json.loads(body)
        except (json.JSONDecodeError, Exception):
            return None

        # Simple message field
        if "message" in data:
            return str(data["message"])

        # OpenAI messages array — extract the last user message
        if "messages" in data:
            messages = data["messages"]
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    return str(msg.get("content", ""))

        return None

    @staticmethod
    def _extract_model_output(response_data: dict) -> str:
        """
        Extract the model's text output from the response JSON.

        Supports:
        - {"message": "..."} — simple format
        - OpenAI chat completion format
        """
        if "message" in response_data:
            return str(response_data["message"])

        # OpenAI completion format
        choices = response_data.get("choices", [])
        if choices:
            return str(choices[0].get("message", {}).get("content", ""))

        return ""

    @staticmethod
    def _inject_filtered_output(response_data: dict, filtered_text: str) -> None:
        """Replace the model output in the response dict with filtered text (in-place)."""
        if "message" in response_data:
            response_data["message"] = filtered_text
        elif "choices" in response_data and response_data["choices"]:
            response_data["choices"][0]["message"]["content"] = filtered_text
