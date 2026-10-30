import warnings

import pytest

from tokenscope.models import Usage
from tokenscope.pricing import ModelPricing

# ==============================================================================
# Sprint 0.5: Final Cache Accounting Edge Cases Regression Tests
# ==============================================================================


def test_positive_cache_creation_with_no_configured_rate_and_zero_input_rate():
    """1. Positive cache creation tokens with no configured rate and zero ordinary

    input rate must mark the cost unresolved and not silently return zero.
    """
    # Neither cache_creation_cost_per_mtok nor duration rates are configured.
    # ordinary input rate is explicitly 0.0.
    pricing = ModelPricing(input_cost_per_mtok=0.0, output_cost_per_mtok=0.0)

    usage = Usage(
        input_tokens=10_000,
        output_tokens=0,
        cache_creation_input_tokens=10_000,
    )

    breakdown = pricing.breakdown_usage(usage)
    assert breakdown.has_unresolved_cost is True
    assert breakdown.is_approximate is True
    assert breakdown.unresolved_cache_creation_tokens == 10_000
    assert len(breakdown.notes) == 1
    assert "no cache creation pricing configured" in breakdown.notes[0].lower()

    # calculate_cost must warn caller and not silently claim 0.0 is exact
    with pytest.warns(UserWarning, match=r"Calculated cost is approximate.*No cache creation"):
        cost = pricing.calculate_cost(uncached_input=0, output=0, cache_creation=10_000)
    assert cost == 0.0  # unresolved contribution is 0, but flagged with warning


def test_explicitly_configured_zero_cache_creation_rate():
    """2. Explicitly configured zero cache creation rate is legitimate zero-cost

    pricing and must report exact 0.0 cost without approximation or warning.
    """
    # Explicitly configured 0.0 cache creation rate
    pricing = ModelPricing(
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
        cache_creation_cost_per_mtok=0.0,
    )

    usage = Usage(
        input_tokens=10_000,
        output_tokens=0,
        cache_creation_input_tokens=10_000,
    )

    breakdown = pricing.breakdown_usage(usage)
    assert breakdown.has_unresolved_cost is False
    assert breakdown.is_approximate is False
    assert breakdown.is_inconsistent is False
    assert breakdown.cache_creation_cost == 0.0
    assert len(breakdown.notes) == 0

    # calculate_cost must not raise any approximate warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cost = pricing.calculate_cost(uncached_input=0, output=0, cache_creation=10_000)
    assert cost == 0.0


def test_duration_details_exactly_matching_aggregate_usage():
    """3. Duration details exactly matching aggregate usage must calculate exact

    cost without approximation.
    """
    pricing = ModelPricing(
        cache_creation_duration_rates={"5m": 3.75, "1h": 6.00},
        cache_creation_cost_per_mtok=3.75,
    )

    usage = Usage(
        input_tokens=150_000,
        output_tokens=0,
        cache_creation_input_tokens=150_000,
        cache_creation_details={"5m": 100_000, "1h": 50_000},
    )

    breakdown = pricing.breakdown_usage(usage)
    assert breakdown.is_inconsistent is False
    assert breakdown.has_unresolved_cost is False
    assert breakdown.is_approximate is False
    assert abs(breakdown.cache_creation_cost - 0.675) < 1e-6
    assert abs(breakdown.cache_creation_details_cost["5m"] - 0.375) < 1e-6
    assert abs(breakdown.cache_creation_details_cost["1h"] - 0.300) < 1e-6
    assert len(breakdown.notes) == 0


def test_duration_details_exceeding_aggregate_usage():
    """4. If duration-specific count exceeds aggregate cache creation tokens:

    - Do not calculate exact cache creation cost (c_cost = 0.0)
    - Flag usage as inconsistent
    - Preserve original counts for diagnostics
    - Mark affected cost as unresolved
    - Add explanatory note
    """
    pricing = ModelPricing(
        cache_creation_duration_rates={"5m": 3.75},
        cache_creation_cost_per_mtok=3.75,
    )

    # 100k aggregate, but 120k in details
    usage = Usage(
        input_tokens=100_000,
        output_tokens=0,
        cache_creation_input_tokens=100_000,
        cache_creation_details={"5m": 120_000},
    )

    assert usage.is_inconsistent is True
    # Original counts preserved for diagnostics
    assert usage.cache_creation_details == {"5m": 120_000}

    breakdown = pricing.breakdown_usage(usage)
    assert breakdown.is_inconsistent is True
    assert breakdown.has_unresolved_cost is True
    assert breakdown.is_approximate is True
    assert breakdown.cache_creation_cost == 0.0
    assert breakdown.unresolved_cache_creation_tokens == 100_000
    # Original count preserved in breakdown diagnostics
    assert abs(breakdown.cache_creation_details_cost["5m"] - (120_000 / 1e6 * 3.75)) < 1e-6
    assert len(breakdown.notes) == 1
    assert "inconsistent cache creation breakdown" in breakdown.notes[0].lower()
    assert "exceeds aggregate" in breakdown.notes[0].lower()


def test_multiple_duration_categories_with_inconsistent_totals():
    """5. Multiple duration categories with inconsistent totals exceeding aggregate

    must preserve all diagnostic counts, flag inconsistent, and mark unresolved.
    """
    pricing = ModelPricing(
        cache_creation_duration_rates={"5m": 3.75, "1h": 6.00},
        cache_creation_cost_per_mtok=3.75,
    )

    # Aggregate: 100k, Details sum: 70k + 60k = 130k (> 100k)
    usage = Usage(
        input_tokens=100_000,
        output_tokens=0,
        cache_creation_input_tokens=100_000,
        cache_creation_details={"5m": 70_000, "1h": 60_000},
    )

    assert usage.is_inconsistent is True

    breakdown = pricing.breakdown_usage(usage)
    assert breakdown.is_inconsistent is True
    assert breakdown.has_unresolved_cost is True
    assert breakdown.is_approximate is True
    assert breakdown.cache_creation_cost == 0.0
    assert breakdown.unresolved_cache_creation_tokens == 100_000
    # Diagnostics preserve both categories
    assert abs(breakdown.cache_creation_details_cost["5m"] - (70_000 / 1e6 * 3.75)) < 1e-6
    assert abs(breakdown.cache_creation_details_cost["1h"] - (60_000 / 1e6 * 6.00)) < 1e-6
    assert "130,000" in breakdown.notes[0]
    assert "100,000" in breakdown.notes[0]

    # Calling calculate_cost directly with inconsistent details also validates
    with pytest.warns(UserWarning, match=r"Inconsistent cache creation breakdown"):
        cost = pricing.calculate_cost(
            uncached_input=0,
            output=0,
            cache_creation=100_000,
            cache_creation_details={"5m": 70_000, "1h": 60_000},
        )
    assert cost == 0.0
