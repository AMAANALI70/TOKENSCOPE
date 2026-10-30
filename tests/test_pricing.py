from tokenscope.models import Usage
from tokenscope.pricing import DEFAULT_PRICING, ModelPricing, PricingRegistry, cost


def test_basic_cost_helper():
    assert cost(1_000_000, 2.50) == 2.50
    assert cost(500_000, 10.00) == 5.00


def test_model_pricing_calculate_cost():
    pricing = ModelPricing(
        input_cost_per_mtok=3.00,
        output_cost_per_mtok=15.00,
        cache_read_cost_per_mtok=0.30,
        cache_creation_cost_per_mtok=3.75,
    )
    # 100k uncached, 50k cached read, 10k cache create, 20k output
    total = pricing.calculate_cost(
        uncached_input=100_000,
        output=20_000,
        cache_read=50_000,
        cache_creation=10_000,
    )
    expected = (
        (100_000 / 1_000_000 * 3.00)
        + (50_000 / 1_000_000 * 0.30)
        + (10_000 / 1_000_000 * 3.75)
        + (20_000 / 1_000_000 * 15.00)
    )
    assert abs(total - expected) < 1e-6
    # Verify cached tokens are discounted and not billed at normal rate
    assert pricing.cache_read_cost_per_mtok < pricing.input_cost_per_mtok


def test_model_pricing_breakdown_usage():
    pricing = ModelPricing(
        input_cost_per_mtok=2.50,
        output_cost_per_mtok=10.00,
        cache_read_cost_per_mtok=1.25,
        cache_creation_cost_per_mtok=0.00,
    )
    usage = Usage(
        input_tokens=100_000,
        output_tokens=10_000,
        uncached_input_tokens=60_000,
        cache_read_input_tokens=40_000,
        cache_creation_input_tokens=0,
    )
    breakdown = pricing.breakdown_usage(usage)
    assert breakdown.is_provider_reported is True
    assert abs(breakdown.uncached_input_cost - (60_000 / 1e6 * 2.50)) < 1e-6
    assert abs(breakdown.cache_read_cost - (40_000 / 1e6 * 1.25)) < 1e-6
    assert breakdown.cache_creation_cost == 0.0
    assert abs(breakdown.output_cost - (10_000 / 1e6 * 10.00)) < 1e-6
    assert abs(breakdown.total_cost - (0.15 + 0.05 + 0.10)) < 1e-6


def test_model_pricing_breakdown_estimated():
    pricing = ModelPricing(input_cost_per_mtok=2.50, output_cost_per_mtok=10.00)
    b = pricing.breakdown_estimated(estimated_input=100_000, estimated_output=10_000)
    assert b.is_provider_reported is False
    assert b.cache_read_cost == 0.0
    assert b.cache_creation_cost == 0.0
    assert abs(b.total_cost - (0.25 + 0.10)) < 1e-6


def test_pricing_registry():
    registry = PricingRegistry()
    custom = ModelPricing(input_cost_per_mtok=1.0, output_cost_per_mtok=2.0)
    registry.register("my-custom-model", custom)

    assert registry.get("my-custom-model") == custom
    # Case-insensitive
    assert registry.get("MY-CUSTOM-MODEL") == custom
    # Prefix match
    assert registry.get("my-custom-model-v2") == custom
    # Unknown model
    assert registry.get("non-existent") is None


def test_default_pricing_registry():
    gpt4o = DEFAULT_PRICING.get("gpt-4o")
    assert gpt4o is not None
    assert gpt4o.input_cost_per_mtok == 2.50
    assert gpt4o.cache_read_cost_per_mtok == 1.25

    claude = DEFAULT_PRICING.get("claude-3-5-sonnet")
    assert claude is not None
    assert claude.input_cost_per_mtok == 3.00
    assert claude.cache_read_cost_per_mtok == 0.30
    assert claude.cache_creation_cost_per_mtok == 3.75
