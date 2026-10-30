import json

from typer.testing import CliRunner

from tokenscope.attribution import attribute
from tokenscope.cli import app
from tokenscope.loader import load_trace
from tokenscope.models import Call, Category, Segment, Trace, Usage
from tokenscope.pricing import ModelPricing, PricingRegistry
from tokenscope.report import to_dict
from tokenscope.tokenizers import HeuristicCounter

C = HeuristicCounter()
runner = CliRunner()


# ==============================================================================
# Issue 1: Duration-Specific Cache Pricing
# ==============================================================================


def test_duration_specific_cache_pricing_calculation():
    # Claude 3.5 Sonnet: 5m @ $3.75/Mtok, 1h @ $6.00/Mtok
    pricing = ModelPricing(
        input_cost_per_mtok=3.00,
        output_cost_per_mtok=15.00,
        cache_read_cost_per_mtok=0.30,
        cache_creation_cost_per_mtok=3.75,
        cache_creation_duration_rates={
            "ephemeral_5m_input_tokens": 3.75,
            "ephemeral_1h_input_tokens": 6.00,
        },
    )

    usage = Usage(
        input_tokens=250_000,
        output_tokens=10_000,
        uncached_input_tokens=50_000,
        cache_read_input_tokens=50_000,
        cache_creation_input_tokens=150_000,
        cache_creation_details={
            "ephemeral_5m_input_tokens": 100_000,
            "ephemeral_1h_input_tokens": 50_000,
        },
    )

    breakdown = pricing.breakdown_usage(usage)
    assert breakdown.is_approximate is False
    assert breakdown.is_provider_reported is True

    # 5m: 100k / 1e6 * 3.75 = 0.375
    # 1h: 50k / 1e6 * 6.00 = 0.300
    # Expected cache creation cost = 0.675
    assert abs(breakdown.cache_creation_details_cost["ephemeral_5m_input_tokens"] - 0.375) < 1e-6
    assert abs(breakdown.cache_creation_details_cost["ephemeral_1h_input_tokens"] - 0.300) < 1e-6
    assert abs(breakdown.cache_creation_cost - 0.675) < 1e-6

    # Uncached: 50k * 3.00 / 1e6 = 0.150
    # Read: 50k * 0.30 / 1e6 = 0.015
    # Output: 10k * 15.00 / 1e6 = 0.150
    # Total = 0.150 + 0.015 + 0.675 + 0.150 = 0.990
    assert abs(breakdown.total_cost - 0.990) < 1e-6


def test_duration_specific_cache_pricing_fallback_is_approximate():
    # Only 5m rate is configured, 1h rate is missing
    pricing = ModelPricing(
        input_cost_per_mtok=3.00,
        output_cost_per_mtok=15.00,
        cache_read_cost_per_mtok=0.30,
        cache_creation_cost_per_mtok=3.75,
        cache_creation_duration_rates={
            "ephemeral_5m_input_tokens": 3.75,
        },
    )

    usage = Usage(
        input_tokens=150_000,
        output_tokens=0,
        uncached_input_tokens=50_000,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=100_000,
        cache_creation_details={
            "ephemeral_1h_input_tokens": 100_000,  # missing duration rate!
        },
    )

    breakdown = pricing.breakdown_usage(usage)
    # Clearly identified as approximate rather than silently applying wrong rate
    assert breakdown.is_approximate is True
    assert len(breakdown.notes) > 0
    assert "ephemeral_1h_input_tokens" in breakdown.notes[0]

    call = Call(0, "claude-3-5-sonnet", [Segment(Category.USER, "x" * 40)], usage=usage)
    attr = attribute(Trace([call], "anthropic"), C)
    data = to_dict(attr, pricing=pricing)
    assert data["cost"]["is_approximate"] is True
    assert len(data["cost"]["approximate_notes"]) > 0


def test_aggregate_cache_creation_compatibility():
    # When cache_creation_details is empty, uses aggregate cache_creation_cost_per_mtok
    pricing = ModelPricing(cache_creation_cost_per_mtok=4.00)
    usage = Usage(input_tokens=100_000, output_tokens=0, cache_creation_input_tokens=100_000)
    b = pricing.breakdown_usage(usage)
    assert b.is_approximate is False
    assert abs(b.cache_creation_cost - 0.400) < 1e-6


# ==============================================================================
# Issue 2: CLI Multi-Model Pricing
# ==============================================================================


def test_cli_multi_model_pricing_preserves_per_call_models(tmp_path):
    p = tmp_path / "mixed_trace.jsonl"
    call0 = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "response": {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 100_000, "completion_tokens": 0},
        },
    }
    call1 = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "response": {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 100_000, "completion_tokens": 0},
        },
    }
    p.write_text(f"{json.dumps(call0)}\n{json.dumps(call1)}\n", encoding="utf-8")

    # 1. Without CLI overrides: Call 0 gets 0.15/Mtok, Call 1 gets 2.50/Mtok
    res = runner.invoke(app, ["audit", str(p), "--json"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    # Expected total = 100k * 0.15/1e6 + 100k * 2.50/1e6 = 0.015 + 0.250 = 0.265
    assert abs(data["cost"]["total_usd"] - 0.265) < 1e-6

    # 2. With --model gpt-4o as fallback: per-call pricing MUST still be preserved for gpt-4o-mini!
    res_fallback = runner.invoke(app, ["audit", str(p), "--model", "gpt-4o", "--json"])
    assert res_fallback.exit_code == 0
    data_fallback = json.loads(res_fallback.output)
    assert abs(data_fallback["cost"]["total_usd"] - 0.265) < 1e-6

    # 3. With --model gpt-4o --price-per-mtok 3.0: overrides gpt-4o while preserving gpt-4o-mini
    res_custom = runner.invoke(
        app, ["audit", str(p), "--model", "gpt-4o", "--price-per-mtok", "3.0", "--json"]
    )
    assert res_custom.exit_code == 0
    data_custom = json.loads(res_custom.output)
    # Expected total = (100k * 0.15/1e6) + (100k * 3.00/1e6) = 0.015 + 0.300 = 0.315
    assert abs(data_custom["cost"]["total_usd"] - 0.315) < 1e-6


# ==============================================================================
# Issue 3: Provider-Specific Pricing
# ==============================================================================


def test_provider_specific_pricing_precedence():
    reg = PricingRegistry()
    # Generic gpt-4o @ $2.50
    reg.register("gpt-4o", ModelPricing(input_cost_per_mtok=2.50))
    # Azure gpt-4o @ discounted $2.00
    reg.register("azure/gpt-4o", ModelPricing(input_cost_per_mtok=2.00))

    u = Usage(input_tokens=100_000, output_tokens=0)

    # Call on Azure provider
    call_azure = Call(
        0,
        "gpt-4o",
        [Segment(Category.USER, "x" * 40)],
        usage=u,
        provider="azure",
    )
    attr_azure = attribute(Trace([call_azure], "azure"), C)
    data_azure = to_dict(attr_azure, pricing_registry=reg)
    # Must use azure/gpt-4o rate: 100k * 2.00 / 1e6 = 0.200
    assert abs(data_azure["cost"]["total_usd"] - 0.200) < 1e-6

    # Call on OpenAI provider
    call_openai = Call(
        0,
        "gpt-4o",
        [Segment(Category.USER, "x" * 40)],
        usage=u,
        provider="openai",
    )
    attr_openai = attribute(Trace([call_openai], "openai"), C)
    data_openai = to_dict(attr_openai, pricing_registry=reg)
    # Must use generic gpt-4o rate: 100k * 2.50 / 1e6 = 0.250
    assert abs(data_openai["cost"]["total_usd"] - 0.250) < 1e-6


def test_provider_passed_from_parser_to_pricing():
    # When loading from openai parser, call.provider is "openai"
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "response": {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 100_000, "completion_tokens": 0},
        },
    }
    trace = load_trace(write_payload(payload), "openai")
    assert trace.calls[0].provider == "openai"


def write_payload(payload: dict) -> str:
    import tempfile

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
        return f.name
