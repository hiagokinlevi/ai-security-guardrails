from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping
import logging


@dataclass(frozen=True)
class ToolValidationResult:
    """Result of validating an agent tool invocation."""

    allowed: bool
    tool_name: str
    reason: str


def _normalize_allowlist(allowlist: Iterable[str]) -> set[str]:
    return {item.strip() for item in allowlist if isinstance(item, str) and item.strip()}


def _extract_tool_name(tool_call: Any) -> str:
    """Best-effort extraction of a tool name from common call shapes.

    Supports:
    - plain string: "search"
    - dict payloads: {"name": "search"}, {"tool": "search"},
      {"function": {"name": "search"}}
    - objects with attributes: .name, .tool, .function.name
    """
    if isinstance(tool_call, str):
        return tool_call.strip()

    if isinstance(tool_call, Mapping):
        if isinstance(tool_call.get("name"), str):
            return tool_call["name"].strip()
        if isinstance(tool_call.get("tool"), str):
            return tool_call["tool"].strip()
        fn = tool_call.get("function")
        if isinstance(fn, Mapping) and isinstance(fn.get("name"), str):
            return fn["name"].strip()

    name = getattr(tool_call, "name", None)
    if isinstance(name, str):
        return name.strip()

    tool = getattr(tool_call, "tool", None)
    if isinstance(tool, str):
        return tool.strip()

    function = getattr(tool_call, "function", None)
    if function is not None:
        fn_name = getattr(function, "name", None)
        if isinstance(fn_name, str):
            return fn_name.strip()

    return ""


def validate_tool_call(
    tool_call: Any,
    allowlist: Iterable[str],
    *,
    logger: logging.Logger | None = None,
    on_unauthorized: Callable[[ToolValidationResult], None] | None = None,
) -> ToolValidationResult:
    """Validate that a tool invocation is within an allowed set.

    Args:
        tool_call: Tool call payload/object/string from an agent framework.
        allowlist: Iterable of allowed tool names.
        logger: Optional logger used to emit warning events for unauthorized calls.
        on_unauthorized: Optional callback invoked when a call is rejected.

    Returns:
        ToolValidationResult indicating whether the invocation is allowed.
    """
    normalized_allowlist = _normalize_allowlist(allowlist)
    tool_name = _extract_tool_name(tool_call)

    if not tool_name:
        result = ToolValidationResult(
            allowed=False,
            tool_name="",
            reason="Unable to determine tool name from tool call payload",
        )
        if logger:
            logger.warning("Tool call rejected: %s", result.reason)
        if on_unauthorized:
            on_unauthorized(result)
        return result

    if tool_name in normalized_allowlist:
        return ToolValidationResult(
            allowed=True,
            tool_name=tool_name,
            reason="Tool is in allowlist",
        )

    result = ToolValidationResult(
        allowed=False,
        tool_name=tool_name,
        reason=f"Tool '{tool_name}' is not in configured allowlist",
    )
    if logger:
        logger.warning("Unauthorized tool invocation rejected", extra={"tool_name": tool_name})
    if on_unauthorized:
        on_unauthorized(result)
    return result
