import pytest

from tokenscope.models import Usage
from tokenscope.pricing import ModelPricing

# ==============================================================================
# Sprint 0.4: Cache Creation Accounting Completeness Regression Tests
# ==============================================================================


def test_fully_accounted_duration_details_is_exact():
    """1. If duration-specific details fully cover cache creation tokens and all

    rates exist, report exact cost (is_approximate=False).
    """
    pricing = ModelPricing(
        input_cost_per_mtok=3.00,
        output_cost_per_mtok=15.00,
        cache_read_cost_per_mtok=0.30,
        cache_creation_cost_per_mtok=3.75,
        cache_creation_duration_rates={
            "5m": 3.75,
            "1h": 6.00,
        },
    )

    usage = Usage(
        input_tokens=250_000,
        output_tokens=10_000,
        uncached_input_tokens=50_000,
        cache_read_input_tokens=50_000,
        cache_creation_input_tokens=150_000,
        cache_creation_details={
            "5m": 100_000,
            "1h": 50_000,
        },
    )

    breakdown = pricing.breakdown_usage(usage)
    assert breakdown.is_approximate is False
    assert breakdown.has_unresolved_cost is False
    assert len(breakdown.notes) == 0

    # 100k @ 3.75 = 0.375, 50k @ 6.00 = 0.300 -> 0.675
    assert abs(breakdown.cache_creation_cost - 0.675) < 1e-6
    assert abs(breakdown.cache_creation_details_cost["5m"] - 0.375) < 1e-6
    assert abs(breakdown.cache_creation_details_cost["1h"] - 0.300) < 1e-6


def test_missing_duration_details_on_model_with_duration_rates_is_approximate():
    """2. If duration-specific details are missing but the model has duration-specific

    pricing rates, flag the aggregate-rate calculation as approximate.
    """
    pricing = ModelPricing(
        input_cost_per_mtok=3.00,
        output_cost_per_mtok=15.00,
        cache_creation_cost_per_mtok=3.75,
        cache_creation_duration_rates={
            "5m": 3.75,
            "1h": 6.00,
        },
    )

    # cache_creation_details is omitted / None
    usage = Usage(
        input_tokens=150_000,
        output_tokens=0,
        uncached_input_tokens=50_000,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=100_000,
    )

    breakdown = pricing.breakdown_usage(usage)
    # MUST be flagged as approximate because assumed TTL is not proven
    assert breakdown.is_approximate is True
    assert breakdown.has_unresolved_cost is False
    assert len(breakdown.notes) == 1
    assert "details missing" in breakdown.notes[0].lower()
    assert "$3.75/Mtok" in breakdown.notes[0]

    # Applies aggregate rate: 100k * 3.75 / 1e6 = 0.375
    assert abs(breakdown.cache_creation_cost - 0.375) < 1e-6


def test_partially_accounted_duration_details_prices_remainder_as_approximate():
    """3. If duration details cover only part of the total cache creation tokens,

    price known categories individually and flag the unclassified remainder as approximate.
    """
    pricing = ModelPricing(
        input_cost_per_mtok=3.00,
        output_cost_per_mtok=15.00,
        cache_creation_cost_per_mtok=3.75,
        cache_creation_duration_rates={
            "5m": 3.75,
            "1h": 6.00,
        },
    )

    # 150k cache creation, but only 100k accounted in details
    usage = Usage(
        input_tokens=150_000,
        output_tokens=0,
        uncached_input_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=150_000,
        cache_creation_details={
            "5m": 100_000,
        },
    )

    breakdown = pricing.breakdown_usage(usage)
    assert breakdown.is_approximate is True
    assert breakdown.has_unresolved_cost is False
    assert len(breakdown.notes) == 1
    assert "50,000" in breakdown.notes[0]
    assert "unclassified" in breakdown.notes[0].lower()

    # 5m: 100k @ 3.75 = 0.375
    # remainder 'other': 50k @ 3.75 = 0.1875
    # Total = 0.5625
    assert abs(breakdown.cache_creation_details_cost["5m"] - 0.375) < 1e-6
    assert abs(breakdown.cache_creation_details_cost["other"] - 0.1875) < 1e-6
    assert abs(breakdown.cache_creation_cost - 0.5625) < 1e-6


def test_unknown_duration_category_preserves_explanatory_note_and_approximate():
    """4. If an unknown duration is encountered, preserve the existing explanatory

    note and approximation status.
    """
    pricing = ModelPricing(
        input_cost_per_mtok=3.00,
        output_cost_per_mtok=15.00,
        cache_creation_cost_per_mtok=3.75,
        cache_creation_duration_rates={
            "5m": 3.75,
        },
    )

    usage = Usage(
        input_tokens=100_000,
        output_tokens=0,
        uncached_input_tokens=50_000,
        cache_creation_input_tokens=50_000,
        cache_creation_details={
            "unknown_tier_24h": 50_000,
        },
    )

    breakdown = pricing.breakdown_usage(usage)
    assert breakdown.is_approximate is True
    assert breakdown.has_unresolved_cost is False
    assert len(breakdown.notes) == 1
    assert "unknown_tier_24h" in breakdown.notes[0]
    assert "$3.75/Mtok" in breakdown.notes[0]
    expected_cost = 50_000 / 1e6 * 3.75
    assert abs(breakdown.cache_creation_cost - expected_cost) < 1e-6


def test_model_with_no_duration_specific_pricing_configuration_is_exact():
    """5. Models with no duration-specific pricing configuration report exact cost

    when using the configured aggregate rate.
    """
    pricing = ModelPricing(
        input_cost_per_mtok=2.50,
        output_cost_per_mtok=10.00,
        cache_creation_cost_per_mtok=4.00,
        # No duration-specific configuration
        cache_creation_duration_rates={},
    )

    usage = Usage(
        input_tokens=100_000,
        output_tokens=0,
        cache_creation_input_tokens=100_000,
    )

    breakdown = pricing.breakdown_usage(usage)
    assert breakdown.is_approximate is False
    assert breakdown.has_unresolved_cost is False
    assert len(breakdown.notes) == 0
    assert abs(breakdown.cache_creation_cost - 0.400) < 1e-6


def test_no_defensible_cache_creation_rate_represents_unresolved_cost_explicitly():
    """6. If no defensible cache creation rate exists, represent the unresolved

    cost explicitly rather than silently reporting zero or a fabricated exact cost.
    """
    # Model has duration rates for 5m, but cache_creation_cost_per_mtok is 0.0 (no aggregate rate)
    pricing = ModelPricing(
        input_cost_per_mtok=3.00,
        output_cost_per_mtok=15.00,
        cache_creation_cost_per_mtok=0.00,  # Not defensible as fallback when 5m is 3.75!
        cache_creation_duration_rates={"5m": 3.75},
    )

    usage = Usage(
        input_tokens=100_000,
        output_tokens=0,
        cache_creation_input_tokens=100_000,
        cache_creation_details={"unknown_dur": 100_000},
    )

    breakdown = pricing.breakdown_usage(usage)
    # Must NOT silently report 0.0 as exact cost
    assert breakdown.is_approximate is True
    assert breakdown.has_unresolved_cost is True
    assert breakdown.unresolved_cache_creation_tokens == 100_000
    assert len(breakdown.notes) == 1
    assert "No defensible cache creation rate exists" in breakdown.notes[0]

    # calculate_cost must warn the caller rather than silently returning fabricated cost
    with pytest.warns(UserWarning, match=r"Calculated cost is approximate.*No defensible"):
        pricing.calculate_cost(
            uncached_input=0,
            output=0,
            cache_creation=100_000,
            cache_creation_details={"unknown_dur": 100_000},
        )
