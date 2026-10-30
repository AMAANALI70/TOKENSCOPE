"""Parser for Anthropic Messages API style requests.

Expected line shape::

    {"model": "...", "system": "..." | [blocks], "tools": [...],
     "messages": [...], "response": {"content": [...], "usage": {...}}}
"""

from __future__ import annotations

from typing import Any

from tokenscope.models import (
    Call,
    Category,
    ContentBlock,
    Message,
    Segment,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
    UsageProvenance,
)
from tokenscope.parsers.base import TraceParseError, canonical_json, text_of


def parse_call(payload: Any, index: int) -> Call:
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        raise TraceParseError(f"call {index}: expected an object with a 'messages' list")

    segments: list[Segment] = []
    structured_tools: list[ToolDefinition] = []

    system = text_of(payload.get("system"))
    if system:
        segments.append(Segment(Category.SYSTEM, system))

    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        params = tool.get("input_schema")
        parameters = params if isinstance(params, dict) else {}
        structured_tools.append(
            ToolDefinition(
                name=name or "unknown",
                description=tool.get("description"),
                parameters=parameters,
                raw_schema=tool,
            )
        )
        segments.append(Segment(Category.TOOL_SCHEMA, canonical_json(tool), tool_name=name))

    tool_use_id_to_name: dict[str, str] = {}
    structured_messages: list[Message] = []

    for i, msg in enumerate(payload["messages"]):
        if not isinstance(msg, dict):
            raise TraceParseError(f"call {index}: message {i} is not an object")
        role = msg.get("role")
        if role not in ("user", "assistant"):
            raise TraceParseError(f"call {index}: message {i} has unsupported role {role!r}")
        text_category = Category.USER if role == "user" else Category.ASSISTANT

        content = msg.get("content")
        msg_blocks: list[ContentBlock] = []
        msg_tool_calls: list[ToolCall] = []
        tool_call_id: str | None = None
        tool_res_obj: ToolResult | None = None

        if isinstance(content, str):
            segments.append(Segment(text_category, content, None, i))
            structured_messages.append(Message(role=role, content=content))
            continue
        if not isinstance(content, list):
            raise TraceParseError(f"call {index}: message {i} has invalid content")

        for block in content:
            if not isinstance(block, dict):
                msg_blocks.append(ContentBlock(type="raw", text=str(block), data=block))
                continue
            kind = block.get("type", "unknown")
            if kind == "text":
                txt = block.get("text", "")
                segments.append(Segment(text_category, txt, None, i))
                msg_blocks.append(ContentBlock(type="text", text=txt, data=block))
            elif kind == "thinking":
                thk = block.get("thinking", "")
                segments.append(Segment(Category.ASSISTANT, thk, None, i))
                msg_blocks.append(ContentBlock(type="thinking", text=thk, data=block))
            elif kind == "tool_use":
                name = block.get("name") or "unknown"
                t_id = block.get("id")
                if t_id:
                    tool_use_id_to_name[t_id] = name
                raw_input = block.get("input", {})
                args_str = (
                    canonical_json(raw_input)
                    if isinstance(raw_input, (dict, list))
                    else str(raw_input)
                )
                msg_tool_calls.append(
                    ToolCall(
                        id=t_id,
                        name=name,
                        arguments=args_str,
                        parsed_arguments=raw_input if isinstance(raw_input, dict) else None,
                    )
                )
                body = f"{name}({args_str})"
                segments.append(Segment(Category.TOOL_CALL, body, name, i))
                msg_blocks.append(ContentBlock(type="tool_use", text=body, data=block))
            elif kind == "tool_result":
                res_id = block.get("tool_use_id")
                tool_call_id = res_id
                name = tool_use_id_to_name.get(res_id or "")
                res_text = text_of(block.get("content"))
                segments.append(Segment(Category.TOOL_RESULT, res_text, name, i))
                tool_res_obj = ToolResult(
                    tool_call_id=res_id,
                    name=name,
                    content=res_text,
                    is_error=bool(block.get("is_error", False)),
                    raw_content=block.get("content"),
                )
                msg_blocks.append(
                    ContentBlock(
                        type="tool_result",
                        text=res_text,
                        data=tool_res_obj,
                    )
                )
            else:
                # Generic / unknown / multimodal content block (e.g. image, document, custom)
                msg_blocks.append(ContentBlock(type=str(kind), text=block.get("text"), data=block))

        structured_messages.append(
            Message(
                role=role,
                content=msg_blocks,
                tool_calls=msg_tool_calls,
                tool_call_id=tool_call_id,
                tool_result=tool_res_obj,
            )
        )

    output: list[Segment] = []
    response_messages: list[Message] = []
    usage: Usage | None = None
    response = payload.get("response")
    call_id = payload.get("call_id") or payload.get("id")
    execution_id = payload.get("execution_id") or payload.get("session_id")

    if isinstance(response, dict):
        if not call_id:
            call_id = response.get("id")
        resp_blocks: list[ContentBlock] = []
        resp_tool_calls: list[ToolCall] = []

        for block in response.get("content") or []:
            if not isinstance(block, dict):
                resp_blocks.append(ContentBlock(type="raw", text=str(block), data=block))
                continue
            b_type = block.get("type", "unknown")
            if b_type == "text":
                txt = block.get("text", "")
                output.append(Segment(Category.OUTPUT, txt))
                resp_blocks.append(ContentBlock(type="text", text=txt, data=block))
            elif b_type == "tool_use":
                name = block.get("name") or "unknown"
                t_id = block.get("id")
                raw_input = block.get("input", {})
                args_str = (
                    canonical_json(raw_input)
                    if isinstance(raw_input, (dict, list))
                    else str(raw_input)
                )
                resp_tool_calls.append(
                    ToolCall(
                        id=t_id,
                        name=name,
                        arguments=args_str,
                        parsed_arguments=raw_input if isinstance(raw_input, dict) else None,
                    )
                )
                body = f"{name}({args_str})"
                output.append(Segment(Category.OUTPUT, body, name))
                resp_blocks.append(ContentBlock(type="tool_use", text=body, data=block))
            else:
                resp_blocks.append(
                    ContentBlock(type=str(b_type), text=block.get("text"), data=block)
                )

        response_messages.append(
            Message(
                role="assistant",
                content=resp_blocks,
                tool_calls=resp_tool_calls,
            )
        )

        u = response.get("usage") if isinstance(response, dict) else payload.get("usage")
        if isinstance(u, dict) and "input_tokens" in u:
            try:
                uncached = int(u["input_tokens"])
                cache_read = int(u.get("cache_read_input_tokens", 0) or 0)
                cache_write = int(u.get("cache_creation_input_tokens", 0) or 0)
                output_toks = int(u.get("output_tokens", 0))
            except (ValueError, TypeError) as exc:
                raise TraceParseError(f"call {index}: invalid non-integer usage: {exc}") from exc

            if uncached < 0 or cache_read < 0 or cache_write < 0 or output_toks < 0:
                raise TraceParseError(f"call {index}: negative token counts in usage")

            details_dict: dict[str, int] = {}
            raw_details = u.get("cache_creation")
            if isinstance(raw_details, dict):
                for k, v in raw_details.items():
                    if isinstance(v, (int, float)):
                        details_dict[k] = int(v)

            total_input = uncached + cache_read + cache_write
            try:
                usage = Usage(
                    input_tokens=total_input,
                    output_tokens=output_toks,
                    cached_input_tokens=cache_read,
                    uncached_input_tokens=uncached,
                    cache_read_input_tokens=cache_read,
                    cache_creation_input_tokens=cache_write,
                    cache_creation_details=details_dict,
                    provenance=UsageProvenance.PROVIDER,
                )
            except ValueError as exc:
                raise TraceParseError(f"call {index}: inconsistent usage tokens: {exc}") from exc

    metadata: dict[str, Any] = {}
    if system:
        metadata["system"] = system
    if isinstance(response, dict):
        for k in ("stop_reason", "stop_sequence", "type"):
            if k in response:
                metadata[k] = response[k]

    return Call(
        index=index,
        model=payload.get("model"),
        input_segments=segments,
        output_segments=output,
        usage=usage,
        call_id=call_id,
        execution_id=execution_id,
        messages=structured_messages,
        tools=structured_tools,
        response_messages=response_messages,
        metadata=metadata,
        provider="anthropic",
    )
