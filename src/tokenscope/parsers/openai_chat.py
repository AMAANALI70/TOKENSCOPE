"""Parser for OpenAI Chat Completions style requests.

Expected line shape (one LLM call per line in the JSONL trace)::

    {"model": "...", "messages": [...], "tools": [...],
     "response": {"choices": [{"message": {...}}], "usage": {...}}}

``response`` is optional. Without it we can still attribute input tokens.
"""

from __future__ import annotations

import json
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

    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool) if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name") if isinstance(fn, dict) else None
        segments.append(Segment(Category.TOOL_SCHEMA, canonical_json(tool), tool_name=name))
        raw_params = fn.get("parameters") if isinstance(fn, dict) else None
        parameters = raw_params if isinstance(raw_params, dict) else {}
        structured_tools.append(
            ToolDefinition(
                name=name or "unknown",
                description=fn.get("description") if isinstance(fn, dict) else None,
                parameters=parameters,
                raw_schema=tool,
            )
        )

    call_id_to_name: dict[str, str] = {}
    structured_messages: list[Message] = []

    for i, msg in enumerate(payload["messages"]):
        if not isinstance(msg, dict):
            raise TraceParseError(f"call {index}: message {i} is not an object")
        role = msg.get("role")

        msg_tool_calls: list[ToolCall] = []
        tool_result_obj: ToolResult | None = None
        raw_content = msg.get("content")

        if role in ("system", "developer"):
            segments.append(Segment(Category.SYSTEM, text_of(raw_content), None, i))
        elif role == "user":
            segments.append(Segment(Category.USER, text_of(raw_content), None, i))
        elif role == "assistant":
            text = text_of(raw_content)
            if text:
                segments.append(Segment(Category.ASSISTANT, text, None, i))
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
                name = fn.get("name") or "unknown"
                tc_id = tc.get("id")
                if tc_id:
                    call_id_to_name[tc_id] = name
                raw_args = fn.get("arguments", "")
                parsed_args = None
                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args)
                    except Exception:
                        parsed_args = None
                    args_str = raw_args
                elif isinstance(raw_args, dict):
                    parsed_args = raw_args
                    args_str = canonical_json(raw_args)
                else:
                    args_str = str(raw_args)

                msg_tool_calls.append(
                    ToolCall(
                        id=tc_id,
                        name=name,
                        arguments=args_str,
                        parsed_arguments=parsed_args,
                    )
                )
                body = f"{name}({args_str})"
                segments.append(Segment(Category.TOOL_CALL, body, name, i))
        elif role in ("tool", "function"):
            name = msg.get("name") or call_id_to_name.get(msg.get("tool_call_id", ""))
            res_content = text_of(raw_content)
            segments.append(Segment(Category.TOOL_RESULT, res_content, name, i))
            tool_result_obj = ToolResult(
                tool_call_id=msg.get("tool_call_id"),
                name=name,
                content=res_content,
                is_error=False,
                raw_content=raw_content,
            )
        else:
            raise TraceParseError(f"call {index}: message {i} has unsupported role {role!r}")

        # Preserve structured content blocks and multimodal metadata
        msg_content: str | list[ContentBlock] | None
        if isinstance(raw_content, list):
            blocks: list[ContentBlock] = []
            for part in raw_content:
                if isinstance(part, str):
                    blocks.append(ContentBlock(type="text", text=part, data=part))
                elif isinstance(part, dict):
                    p_type = str(part.get("type", "unknown"))
                    blocks.append(ContentBlock(type=p_type, text=part.get("text"), data=part))
                else:
                    blocks.append(ContentBlock(type="raw", text=str(part), data=part))
            msg_content = blocks
        elif isinstance(raw_content, str):
            msg_content = raw_content
        elif raw_content is None:
            msg_content = None
        else:
            msg_content = text_of(raw_content)

        structured_messages.append(
            Message(
                role=str(role),
                content=msg_content,
                name=msg.get("name"),
                tool_calls=msg_tool_calls,
                tool_call_id=msg.get("tool_call_id"),
                tool_result=tool_result_obj,
            )
        )

    output: list[Segment] = []
    response_messages: list[Message] = []
    usage: Usage | None = None
    response = payload.get("response")
    usage_dict = payload.get("usage")
    call_id = payload.get("call_id") or payload.get("id")
    execution_id = payload.get("execution_id") or payload.get("session_id")

    if isinstance(response, dict):
        if not call_id:
            call_id = response.get("id")
        usage_dict = response.get("usage", usage_dict)
        choices = response.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            text = text_of(message.get("content"))
            resp_tool_calls: list[ToolCall] = []
            if text:
                output.append(Segment(Category.OUTPUT, text))
            for tc in message.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
                name = fn.get("name") or "unknown"
                raw_args = fn.get("arguments", "")
                parsed_args = None
                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args)
                    except Exception:
                        parsed_args = None
                    args_str = raw_args
                elif isinstance(raw_args, dict):
                    parsed_args = raw_args
                    args_str = canonical_json(raw_args)
                else:
                    args_str = str(raw_args)

                resp_tool_calls.append(
                    ToolCall(
                        id=tc.get("id"),
                        name=name,
                        arguments=args_str,
                        parsed_arguments=parsed_args,
                    )
                )
                output.append(Segment(Category.OUTPUT, f"{name}({args_str})", name))
            response_messages.append(
                Message(
                    role=message.get("role", "assistant"),
                    content=text,
                    tool_calls=resp_tool_calls,
                )
            )

    if isinstance(usage_dict, dict) and "prompt_tokens" in usage_dict:
        try:
            prompt_tokens = int(usage_dict["prompt_tokens"])
            completion_tokens = int(usage_dict.get("completion_tokens", 0))
            details = usage_dict.get("prompt_tokens_details") or {}
            cached_tokens = int(details.get("cached_tokens", 0) or 0)
        except (ValueError, TypeError) as exc:
            raise TraceParseError(f"call {index}: invalid non-integer usage: {exc}") from exc

        if prompt_tokens < 0 or completion_tokens < 0 or cached_tokens < 0:
            raise TraceParseError(f"call {index}: negative token counts in usage")
        if cached_tokens > prompt_tokens:
            raise TraceParseError(
                f"call {index}: cached_tokens ({cached_tokens}) > prompt_tokens ({prompt_tokens})"
            )

        uncached_tokens = prompt_tokens - cached_tokens
        try:
            usage = Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cached_input_tokens=cached_tokens,
                uncached_input_tokens=uncached_tokens,
                cache_read_input_tokens=cached_tokens,
                cache_creation_input_tokens=0,
                provenance=UsageProvenance.PROVIDER,
            )
        except ValueError as exc:
            raise TraceParseError(f"call {index}: inconsistent usage tokens: {exc}") from exc

    metadata: dict[str, Any] = {}
    if isinstance(response, dict):
        for k in ("created", "system_fingerprint", "service_tier"):
            if k in response:
                metadata[k] = response[k]
        choices = response.get("choices") or []
        if choices and isinstance(choices[0], dict) and "finish_reason" in choices[0]:
            metadata["finish_reason"] = choices[0]["finish_reason"]

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
        provider="openai",
    )
