from tokenscope.loader import load_trace
from tokenscope.models import ContentBlock, Message, ToolCall, ToolDefinition


def test_openai_structured_representations(write_jsonl):
    payload = {
        "id": "chatcmpl-test-123",
        "model": "gpt-4o",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up key in DB",
                    "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
                },
            }
        ],
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Find key: abc"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"key": "abc"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "val_123"},
        ],
        "response": {
            "id": "chatcmpl-test-123",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Key found: val_123"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        },
    }
    trace = load_trace(write_jsonl([payload]), "openai")
    call = trace.calls[0]

    assert call.call_id == "chatcmpl-test-123"
    assert call.metadata.get("finish_reason") == "stop"

    # Structured tool definitions
    assert len(call.tools) == 1
    tool_def = call.tools[0]
    assert isinstance(tool_def, ToolDefinition)
    assert tool_def.name == "lookup"
    assert tool_def.description == "Look up key in DB"
    assert "properties" in tool_def.parameters

    # Structured messages
    assert len(call.messages) == 4
    msg0, msg1, msg2, msg3 = call.messages
    assert isinstance(msg0, Message)
    assert msg0.role == "system"
    assert msg0.content == "You are helpful."

    assert msg1.role == "user"
    assert msg1.content == "Find key: abc"

    assert msg2.role == "assistant"
    assert len(msg2.tool_calls) == 1
    tc = msg2.tool_calls[0]
    assert isinstance(tc, ToolCall)
    assert tc.id == "call_1"
    assert tc.name == "lookup"
    assert tc.parsed_arguments == {"key": "abc"}

    assert msg3.role == "tool"
    assert msg3.tool_call_id == "call_1"
    assert msg3.content == "val_123"

    # Response messages
    assert len(call.response_messages) == 1
    assert call.response_messages[0].content == "Key found: val_123"


def test_anthropic_structured_representations(write_jsonl):
    payload = {
        "id": "msg_anthropic_456",
        "model": "claude-3-5-sonnet",
        "system": "You are concise.",
        "tools": [
            {
                "name": "calculator",
                "description": "Evaluate math expression",
                "input_schema": {"type": "object", "properties": {"expr": {"type": "string"}}},
            }
        ],
        "messages": [
            {"role": "user", "content": "2 + 2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me calculate."},
                    {
                        "type": "tool_use",
                        "id": "t_calc",
                        "name": "calculator",
                        "input": {"expr": "2+2"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t_calc", "content": "4"}],
            },
        ],
        "response": {
            "id": "msg_anthropic_456",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "4"}],
            "usage": {"input_tokens": 30, "output_tokens": 2},
        },
    }
    trace = load_trace(write_jsonl([payload]), "anthropic")
    call = trace.calls[0]

    assert call.call_id == "msg_anthropic_456"
    assert call.metadata.get("system") == "You are concise."
    assert call.metadata.get("stop_reason") == "end_turn"

    # Structured tool definitions
    assert len(call.tools) == 1
    t_def = call.tools[0]
    assert t_def.name == "calculator"
    assert "properties" in t_def.parameters

    # Structured messages
    assert len(call.messages) == 3
    asst_msg = call.messages[1]
    assert asst_msg.role == "assistant"
    assert len(asst_msg.tool_calls) == 1
    assert asst_msg.tool_calls[0].name == "calculator"
    assert asst_msg.tool_calls[0].parsed_arguments == {"expr": "2+2"}

    assert isinstance(asst_msg.content, list)
    assert any(isinstance(b, ContentBlock) and b.type == "thinking" for b in asst_msg.content)

    user_tool_res = call.messages[2]
    assert user_tool_res.tool_call_id == "t_calc"


def test_trace_execution_identifiers(write_jsonl):
    payload = {
        "trace_id": "run-xyz-789",
        "execution_id": "exec-abc-123",
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    }
    trace = load_trace(write_jsonl([payload]), "openai")
    assert trace.trace_id == "run-xyz-789"
    assert trace.execution_id == "exec-abc-123"
