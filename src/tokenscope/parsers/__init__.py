"""Provider parsers. Add a provider by adding a module with ``parse_call``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tokenscope.models import Call
from tokenscope.parsers import anthropic_messages, openai_chat
from tokenscope.parsers.base import TraceParseError

ParseFn = Callable[[Any, int], Call]

PARSERS: dict[str, ParseFn] = {
    "openai": openai_chat.parse_call,
    "anthropic": anthropic_messages.parse_call,
}


def detect_format(payload: Any) -> str:
    """Best-effort guess from one call payload. Prefer passing --format explicitly."""
    if not isinstance(payload, dict):
        raise TraceParseError("cannot detect format: payload is not an object")
    if "system" in payload:
        return "anthropic"
    for tool in payload.get("tools") or []:
        if isinstance(tool, dict) and "input_schema" in tool:
            return "anthropic"
    for msg in payload.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("tool_use", "tool_result"):
                    return "anthropic"
    return "openai"


__all__ = ["PARSERS", "TraceParseError", "detect_format"]
