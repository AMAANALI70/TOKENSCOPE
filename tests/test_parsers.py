import pytest

from tokenscope.loader import load_trace
from tokenscope.models import Category
from tokenscope.parsers import TraceParseError, detect_format


def _by_cat(call, cat):
    return [s for s in call.input_segments if s.category is cat]


def test_openai_roles_and_tool_name_resolution(write_jsonl):
    payload = {
        "model": "gpt-4o",
        "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"a"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "file body"},
        ],
        "response": {
            "choices": [{"message": {"content": "done"}}],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 10},
            },
        },
    }
    trace = load_trace(write_jsonl([payload]), "openai")
    call = trace.calls[0]
    assert _by_cat(call, Category.SYSTEM)[0].text == "sys"
    assert _by_cat(call, Category.TOOL_SCHEMA)[0].tool_name == "read_file"
    assert _by_cat(call, Category.TOOL_CALL)[0].tool_name == "read_file"
    # tool result gets its tool name via tool_call_id
    assert _by_cat(call, Category.TOOL_RESULT)[0].tool_name == "read_file"
    assert call.output_segments[0].text == "done"
    assert call.usage.input_tokens == 50 and call.usage.cached_input_tokens == 10


def test_anthropic_blocks_and_usage_normalisation(write_jsonl):
    payload = {
        "model": "claude",
        "system": [{"type": "text", "text": "be brief"}],
        "tools": [{"name": "grep", "description": "d", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": "find it"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "t1", "name": "grep", "input": {"q": "x"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "match"}],
                    }
                ],
            },
        ],
        "response": {
            "content": [{"type": "text", "text": "found"}],
            "usage": {
                "input_tokens": 20,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 5,
                "output_tokens": 3,
            },
        },
    }
    trace = load_trace(write_jsonl([payload]))  # auto-detect
    assert trace.source_format == "anthropic"
    call = trace.calls[0]
    assert _by_cat(call, Category.SYSTEM)[0].text == "be brief"
    assert _by_cat(call, Category.TOOL_RESULT)[0].tool_name == "grep"
    assert _by_cat(call, Category.TOOL_RESULT)[0].text == "match"
    # total prompt = uncached + cache read + cache write
    assert call.usage.input_tokens == 125
    assert call.usage.cached_input_tokens == 100


def test_detect_format():
    assert detect_format({"system": "x", "messages": []}) == "anthropic"
    assert detect_format({"messages": [{"role": "user", "content": "x"}]}) == "openai"


def test_invalid_json_reports_line_number(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"messages": []}\n{not json}\n', encoding="utf-8")
    with pytest.raises(TraceParseError) as exc:
        load_trace(p, "openai")
    assert ":2:" in str(exc.value)


def test_unsupported_role_is_an_error(write_jsonl):
    with pytest.raises(TraceParseError):
        load_trace(write_jsonl([{"messages": [{"role": "alien", "content": "x"}]}]), "openai")


def test_empty_trace_is_an_error(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(TraceParseError):
        load_trace(p)
