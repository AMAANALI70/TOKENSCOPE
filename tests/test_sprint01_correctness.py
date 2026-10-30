import json

import pytest

from tokenscope.attribution import attribute
from tokenscope.loader import load_trace
from tokenscope.models import (
    Call,
    Category,
    Segment,
    Trace,
    Usage,
    UsageProvenance,
)
from tokenscope.parsers.base import TraceParseError
from tokenscope.pricing import DEFAULT_PRICING, ModelPricing, PricingRegistry
from tokenscope.report import to_dict
from tokenscope.tokenizers import HeuristicCounter

C = HeuristicCounter()


# ==============================================================================
# Priority 1: Pricing Correctness
# ==============================================================================


def test_pricing_registry_longest_prefix_and_exact_match():
    reg = PricingRegistry()
    p_4o = ModelPricing(input_cost_per_mtok=2.50)
    p_4o_mini = ModelPricing(input_cost_per_mtok=0.15)
    reg.register("gpt-4o", p_4o)
    reg.register("gpt-4o-mini", p_4o_mini)

    # Exact matches
    assert reg.get("gpt-4o") == p_4o
    assert reg.get("gpt-4o-mini") == p_4o_mini

    # Prefix matches: gpt-4o-mini-2024-07-18 must match gpt-4o-mini (len 11), NOT gpt-4o (len 6)
    assert reg.get("gpt-4o-mini-2024-07-18") == p_4o_mini
    assert reg.get("gpt-4o-2024-08-06") == p_4o


def test_pricing_registry_provider_specific_priority():
    reg = PricingRegistry()
    generic_claude = ModelPricing(input_cost_per_mtok=5.0)
    bedrock_claude = ModelPricing(input_cost_per_mtok=6.0)

    reg.register("claude-3-5-sonnet", generic_claude)
    reg.register("bedrock/claude-3-5-sonnet", bedrock_claude)

    # Provider-specific exact match takes priority
    assert reg.get("claude-3-5-sonnet", provider="bedrock") == bedrock_claude
    assert reg.get("claude-3-5-sonnet", provider="anthropic") == generic_claude
    assert reg.get("claude-3-5-sonnet") == generic_claude

    # Provider-specific prefix match
    assert reg.get("claude-3-5-sonnet-20241022", provider="bedrock") == bedrock_claude


def test_different_models_in_same_trace_receive_their_own_pricing():
    # Call 0 uses gpt-4o-mini (100k input @ $0.15/Mtok)
    # Call 1 uses gpt-4o (100k input @ $2.50/Mtok)
    u0 = Usage(input_tokens=100_000, output_tokens=0, uncached_input_tokens=100_000)
    u1 = Usage(input_tokens=100_000, output_tokens=0, uncached_input_tokens=100_000)

    call0 = Call(0, "gpt-4o-mini", [Segment(Category.USER, "a" * 40)], usage=u0)
    call1 = Call(1, "gpt-4o", [Segment(Category.USER, "b" * 40)], usage=u1)

    trace = Trace([call0, call1], "openai")
    attr = attribute(trace, C)

    data = to_dict(attr, pricing=DEFAULT_PRICING)
    cost_info = data["cost"]
    assert cost_info["status"] == "reported"

    # Expected: (100k * 0.15 / 1e6) + (100k * 2.50 / 1e6) = 0.015 + 0.250 = 0.265
    expected_usd = 0.015 + 0.250
    assert abs(cost_info["total_usd"] - expected_usd) < 1e-6


# ==============================================================================
# Priority 2: Usage Provenance and Cost Accounting
# ==============================================================================


def test_usage_provenance_respected_in_trace_attribution():
    u_provider = Usage(input_tokens=100, output_tokens=10, provenance=UsageProvenance.PROVIDER)
    u_estimated = Usage(input_tokens=200, output_tokens=20, provenance=UsageProvenance.ESTIMATED)

    call0 = Call(0, "gpt-4o", [Segment(Category.USER, "x" * 40)], usage=u_provider)
    call1 = Call(1, "gpt-4o", [Segment(Category.USER, "y" * 80)], usage=u_estimated)

    attr = attribute(Trace([call0, call1], "openai"), C)

    assert attr.has_reported_usage is True
    assert attr.all_calls_have_reported_usage is False
    # Only provider-reported tokens are summed in reported totals
    assert attr.reported_input_total == 100
    assert attr.reported_output_total == 10

    # Calibration factor only considers provider-reported calls (call0 has 10 est, 100 reported)
    assert abs(attr.calibration_factor - 10.0) < 1e-6


def test_only_estimated_usage_is_not_classified_as_reported():
    u_estimated = Usage(input_tokens=200, output_tokens=20, provenance=UsageProvenance.ESTIMATED)
    call = Call(0, "gpt-4o", [Segment(Category.USER, "y" * 80)], usage=u_estimated)
    attr = attribute(Trace([call], "openai"), C)

    assert attr.has_reported_usage is False
    assert attr.all_calls_have_reported_usage is False
    assert attr.reported_input_total is None
    assert attr.calibration_factor is None


def test_partially_reported_trace_cost_accounting():
    # Call 0: provider usage (100 tokens @ $2.50/Mtok = 0.00025)
    u0 = Usage(input_tokens=100, output_tokens=0, uncached_input_tokens=100)
    call0 = Call(0, "gpt-4o", [Segment(Category.USER, "a" * 40)], usage=u0)

    # Call 1: no provider usage. Estimated input: 400 chars = 100 tokens @ $2.50/Mtok = 0.00025
    call1 = Call(1, "gpt-4o", [Segment(Category.USER, "b" * 400)], usage=None)

    attr = attribute(Trace([call0, call1], "openai"), C)
    pricing = ModelPricing(input_cost_per_mtok=2.50)

    data = to_dict(attr, pricing=pricing)
    cost_info = data["cost"]

    assert cost_info["status"] == "partially_reported"
    assert cost_info["reported_calls"] == 1
    assert cost_info["estimated_calls"] == 1
    # Total cost = 0.00025 (reported) + 0.00025 (estimated) = 0.00050
    assert abs(cost_info["total_usd"] - 0.00050) < 1e-7


def test_cost_status_distinctions():
    pricing = ModelPricing(input_cost_per_mtok=2.0)
    seg = Segment(Category.USER, "a" * 40)
    u = Usage(input_tokens=10, output_tokens=0)

    # Fully reported
    attr_rep = attribute(Trace([Call(0, "m", [seg], usage=u)], "openai"), C)
    assert to_dict(attr_rep, pricing=pricing)["cost"]["status"] == "reported"

    # Estimated
    attr_est = attribute(Trace([Call(0, "m", [seg], usage=None)], "openai"), C)
    assert to_dict(attr_est, pricing=pricing)["cost"]["status"] == "estimated"

    # Unknown
    assert to_dict(attr_est, pricing=None)["cost"]["status"] == "unknown"


# ==============================================================================
# Priority 3: Provider Usage Semantics & Invariants
# ==============================================================================


def test_usage_invariant_validation():
    # Valid invariant: 100 == 70 + 20 + 10
    u = Usage(
        input_tokens=100,
        output_tokens=5,
        uncached_input_tokens=70,
        cache_read_input_tokens=20,
        cache_creation_input_tokens=10,
    )
    assert u.total_input_tokens == 100

    # Inconsistent total != uncached + read + create
    with pytest.raises(ValueError, match="Inconsistent token counts"):
        Usage(
            input_tokens=100,
            output_tokens=5,
            uncached_input_tokens=50,  # 50 + 20 + 10 != 100
            cache_read_input_tokens=20,
            cache_creation_input_tokens=10,
        )


def test_negative_usage_counts_raise():
    with pytest.raises(ValueError, match="negative"):
        Usage(input_tokens=-10, output_tokens=5)

    with pytest.raises(ValueError, match="negative"):
        Usage(input_tokens=10, output_tokens=-5)


def test_openai_parser_rejects_inconsistent_and_negative_usage(write_jsonl):
    # cached_tokens exceeds prompt_tokens
    payload_bad_cache = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "response": {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 50, "prompt_tokens_details": {"cached_tokens": 80}},
        },
    }
    with pytest.raises(TraceParseError, match="cached_tokens"):
        load_trace(write_jsonl([payload_bad_cache]), "openai")

    # Negative tokens
    payload_neg = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "response": {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": -5, "completion_tokens": 1},
        },
    }
    with pytest.raises(TraceParseError, match="negative"):
        load_trace(write_jsonl([payload_neg]), "openai")


def test_anthropic_duration_specific_cache_creation_breakdown(write_jsonl):
    payload = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "hi"}],
        "response": {
            "content": [{"type": "text", "text": "hello"}],
            "usage": {
                "input_tokens": 20,
                "output_tokens": 5,
                "cache_creation_input_tokens": 150,
                "cache_read_input_tokens": 50,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 50,
                    "ephemeral_1h_input_tokens": 100,
                },
            },
        },
    }
    trace = load_trace(write_jsonl([payload]), "anthropic")
    usage = trace.calls[0].usage
    assert usage is not None
    assert usage.cache_creation_input_tokens == 150
    assert usage.cache_creation_details == {
        "ephemeral_5m_input_tokens": 50,
        "ephemeral_1h_input_tokens": 100,
    }
    assert usage.total_input_tokens == 20 + 50 + 150


def test_usage_missing_and_zero_fields():
    # All zero usage is completely valid and consistent
    u_zero = Usage(input_tokens=0, output_tokens=0)
    assert u_zero.total_input_tokens == 0
    assert u_zero.uncached_input_tokens == 0
    assert u_zero.cache_read_input_tokens == 0

    # Partially available fields default properly
    u_partial = Usage(input_tokens=50, output_tokens=10)
    assert u_partial.uncached_input_tokens == 50
    assert u_partial.cache_read_input_tokens == 0


# ==============================================================================
# Priority 4: Structured Content Preservation
# ==============================================================================


def test_anthropic_preserves_unknown_content_blocks(write_jsonl):
    payload = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect image"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg"}},
                    {"type": "custom_extension", "data": "xyz"},
                ],
            }
        ],
        "response": {
            "content": [
                {"type": "text", "text": "done"},
                {"type": "redacted_thinking", "data": "hidden"},
            ],
        },
    }
    trace = load_trace(write_jsonl([payload]), "anthropic")
    call = trace.calls[0]

    # User message blocks
    user_blocks = call.messages[0].content
    assert isinstance(user_blocks, list)
    assert len(user_blocks) == 3
    assert user_blocks[0].type == "text"
    assert user_blocks[1].type == "image"
    assert user_blocks[1].data["source"]["media_type"] == "image/jpeg"
    assert user_blocks[2].type == "custom_extension"
    assert user_blocks[2].data["data"] == "xyz"

    # Response blocks
    resp_blocks = call.response_messages[0].content
    assert isinstance(resp_blocks, list)
    assert len(resp_blocks) == 2
    assert resp_blocks[0].type == "text"
    assert resp_blocks[1].type == "redacted_thinking"


def test_openai_preserves_multimodal_content_blocks(write_jsonl):
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
                ],
            }
        ],
    }
    trace = load_trace(write_jsonl([payload]), "openai")
    call = trace.calls[0]

    msg = call.messages[0]
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2
    assert msg.content[0].type == "text"
    assert msg.content[1].type == "image_url"
    assert msg.content[1].data["image_url"]["url"] == "https://example.com/cat.jpg"


def test_preserves_original_tool_arguments_when_parsing_fails(write_jsonl):
    # Malformed JSON in tool call arguments
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc_broken",
                        "type": "function",
                        "function": {"name": "query", "arguments": "{malformed json:"},
                    }
                ],
            }
        ],
    }
    trace = load_trace(write_jsonl([payload]), "openai")
    call = trace.calls[0]
    tc = call.messages[0].tool_calls[0]

    # Original arguments string preserved
    assert tc.arguments == "{malformed json:"
    # Parsed arguments is None because JSON parse failed
    assert tc.parsed_arguments is None


def test_tool_call_and_tool_result_relationship_preserved(write_jsonl):
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "fetch_user", "arguments": '{"id":1}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": '{"name":"Alice"}',
            },
        ],
    }
    trace = load_trace(write_jsonl([payload]), "openai")
    call = trace.calls[0]

    asst_msg = call.messages[0]
    tool_msg = call.messages[1]

    assert len(asst_msg.tool_calls) == 1
    assert asst_msg.tool_calls[0].id == "call_abc"

    assert tool_msg.tool_call_id == "call_abc"
    assert tool_msg.tool_result is not None
    assert tool_msg.tool_result.tool_call_id == "call_abc"
    assert tool_msg.tool_result.name == "fetch_user"
    assert tool_msg.tool_result.content == '{"name":"Alice"}'


# ==============================================================================
# Priority 5: Metadata
# ==============================================================================


def test_call_execution_id_populated_from_payload(write_jsonl):
    payload = {
        "model": "gpt-4o",
        "execution_id": "exec-run-42",
        "messages": [{"role": "user", "content": "test"}],
    }
    trace = load_trace(write_jsonl([payload]), "openai")
    assert trace.calls[0].execution_id == "exec-run-42"


def test_trace_envelope_metadata_validation(tmp_path):
    # Valid envelope with metadata dict
    p = tmp_path / "trace_envelope.json"
    p.write_text(
        json.dumps({
            "trace_id": "tr-100",
            "execution_id": "ex-200",
            "metadata": {"environment": "staging", "user_id": 123},
            "calls": [
                {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
            ],
        }),
        encoding="utf-8",
    )
    trace = load_trace(p, "openai")
    assert trace.trace_id == "tr-100"
    assert trace.execution_id == "ex-200"
    assert trace.metadata == {"environment": "staging", "user_id": 123}

    # Missing metadata should not break parsing
    p_no_meta = tmp_path / "no_meta.json"
    p_no_meta.write_text(
        json.dumps({
            "calls": [
                {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
            ],
        }),
        encoding="utf-8",
    )
    trace_no_meta = load_trace(p_no_meta, "openai")
    assert trace_no_meta.metadata == {}

    # Invalid metadata (not a dict) must raise TraceParseError
    p_bad_meta = tmp_path / "bad_meta.json"
    p_bad_meta.write_text(
        json.dumps({
            "metadata": "invalid_string",
            "calls": [
                {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
            ],
        }),
        encoding="utf-8",
    )
    with pytest.raises(TraceParseError, match="must be an object"):
        load_trace(p_bad_meta, "openai")
