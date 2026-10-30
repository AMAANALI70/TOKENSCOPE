import json

import pytest
from typer.testing import CliRunner

from tokenscope.cli import app
from tokenscope.models import Usage
from tokenscope.pricing import (
    CostBreakdown,
    ModelPricing,
    normalize_duration_key,
)

runner = CliRunner()


# ==============================================================================
# Priority 1: Duration Matching Tests
# ==============================================================================


def test_canonical_duration_normalization():
    # 5m aliases
    for alias in (
        "5m",
        "5min",
        "5-minute",
        "5_minute",
        "ephemeral_5m",
        "ephemeral_5m_input_tokens",
        "cache_creation_5m",
        "cache_creation_ephemeral_5m",
    ):
        assert normalize_duration_key(alias) == "5m"
        assert normalize_duration_key(alias.upper()) == "5m"

    # 1h aliases
    for alias in (
        "1h",
        "1hr",
        "1-hour",
        "1_hour",
        "60m",
        "ephemeral_1h",
        "ephemeral_1h_input_tokens",
        "cache_creation_1h",
        "cache_creation_ephemeral_1h",
    ):
        assert normalize_duration_key(alias) == "1h"
        assert normalize_duration_key(alias.upper()) == "1h"

    # Misleading values must NEVER normalize to 5m or 1h
    for misleading in ("15m", "25m", "ephemeral_15m_input_tokens", "1h30m", "2h", "10h"):
        assert normalize_duration_key(misleading) is None

    # Unknown or empty durations
    for unknown in ("unknown", "daily", "ephemeral", ""):
        assert normalize_duration_key(unknown) is None


def test_duration_matching_no_substring_confusion():
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

    # Valid 5m lookups
    assert pricing.get_duration_cache_creation_rate("5m") == 3.75
    assert pricing.get_duration_cache_creation_rate("ephemeral_5m_input_tokens") == 3.75
    assert pricing.get_duration_cache_creation_rate("5min") == 3.75

    # Valid 1h lookups
    assert pricing.get_duration_cache_creation_rate("1h") == 6.00
    assert pricing.get_duration_cache_creation_rate("ephemeral_1h_input_tokens") == 6.00
    assert pricing.get_duration_cache_creation_rate("60m") == 6.00

    # Misleading values containing '5m' or '1h' must return None (NO substring matching!)
    assert pricing.get_duration_cache_creation_rate("15m") is None
    assert pricing.get_duration_cache_creation_rate("25m") is None
    assert pricing.get_duration_cache_creation_rate("ephemeral_15m_input_tokens") is None
    assert pricing.get_duration_cache_creation_rate("1h30m") is None
    assert pricing.get_duration_cache_creation_rate("2h") is None

    # Unknown durations
    assert pricing.get_duration_cache_creation_rate("unknown_duration") is None
    assert pricing.get_duration_cache_creation_rate("") is None


def test_duration_matching_with_explicit_custom_key():
    # If a model explicitly defines a custom rate (e.g. 15m), exact match returns it
    pricing = ModelPricing(
        cache_creation_duration_rates={
            "5m": 3.75,
            "15m": 4.50,
        }
    )
    assert pricing.get_duration_cache_creation_rate("5m") == 3.75
    assert pricing.get_duration_cache_creation_rate("15m") == 4.50
    assert pricing.get_duration_cache_creation_rate("1h") is None


# ==============================================================================
# Priority 2: Consistent Pricing Calculations Tests
# ==============================================================================


def test_calculate_cost_and_breakdown_usage_consistency():
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

    uncached = 100_000
    cache_read = 50_000
    cache_creation = 150_000
    output = 20_000
    details = {
        "ephemeral_5m_input_tokens": 100_000,
        "ephemeral_1h_input_tokens": 50_000,
    }

    # 1. Via calculate_cost
    calc_total = pricing.calculate_cost(
        uncached_input=uncached,
        output=output,
        cache_read=cache_read,
        cache_creation=cache_creation,
        cache_creation_details=details,
    )

    # 2. Via breakdown_usage
    usage = Usage(
        input_tokens=uncached + cache_read + cache_creation,
        output_tokens=output,
        uncached_input_tokens=uncached,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
        cache_creation_details=details,
    )
    breakdown = pricing.breakdown_usage(usage)

    # 3. Via compute_cost_breakdown
    direct_breakdown = pricing.compute_cost_breakdown(
        uncached_input=uncached,
        output=output,
        cache_read=cache_read,
        cache_creation=cache_creation,
        cache_creation_details=details,
    )

    # All three must produce identical total cost
    assert abs(calc_total - breakdown.total_cost) < 1e-9
    assert abs(calc_total - direct_breakdown.total_cost) < 1e-9
    assert breakdown.is_approximate is False
    assert direct_breakdown.is_approximate is False


def test_calculate_cost_warns_on_approximate_duration():
    pricing = ModelPricing(
        input_cost_per_mtok=3.00,
        output_cost_per_mtok=15.00,
        cache_read_cost_per_mtok=0.30,
        cache_creation_cost_per_mtok=3.75,
        cache_creation_duration_rates={
            "5m": 3.75,
        },
    )

    # Missing duration rate for '15m'
    with pytest.warns(UserWarning, match=r"Calculated cost is approximate.*15m"):
        cost_val = pricing.calculate_cost(
            uncached_input=10_000,
            output=1_000,
            cache_creation=50_000,
            cache_creation_details={"15m": 50_000},
        )
    assert cost_val > 0


def test_compute_cost_breakdown_identifies_approximate_notes():
    pricing = ModelPricing(
        input_cost_per_mtok=3.00,
        output_cost_per_mtok=15.00,
        cache_creation_cost_per_mtok=4.00,
        cache_creation_duration_rates={"5m": 3.75},
    )

    breakdown: CostBreakdown = pricing.compute_cost_breakdown(
        uncached_input=10_000,
        output=1_000,
        cache_creation=50_000,
        cache_creation_details={"unknown_dur": 50_000},
    )

    assert breakdown.is_approximate is True
    assert len(breakdown.notes) == 1
    assert "unknown_dur" in breakdown.notes[0]
    assert "$4.0/Mtok" in breakdown.notes[0]


# ==============================================================================
# Priority 3: CLI Pricing Semantics Tests
# ==============================================================================


def test_cli_custom_rate_without_model_is_fallback_not_universal_override(tmp_path):
    p = tmp_path / "mixed_trace.jsonl"
    call_mini = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "response": {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 100_000, "completion_tokens": 0},
        },
    }
    call_4o = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "response": {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 100_000, "completion_tokens": 0},
        },
    }
    # Unmodeled call without model identifier
    call_unmodeled = {
        "messages": [{"role": "user", "content": "hi"}],
        "response": {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 100_000, "completion_tokens": 0},
        },
    }
    p.write_text(
        f"{json.dumps(call_mini)}\n{json.dumps(call_4o)}\n{json.dumps(call_unmodeled)}\n",
        encoding="utf-8",
    )

    # Run with --price-per-mtok 10.0 (custom fallback rate)
    res = runner.invoke(app, ["audit", str(p), "--price-per-mtok", "10.0", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.output)

    # gpt-4o-mini rate in registry: 0.15/Mtok -> 100k * 0.15 / 1e6 = 0.015
    # gpt-4o rate in registry: 2.50/Mtok -> 100k * 2.50 / 1e6 = 0.250
    # unmodeled call rate from fallback: 10.00/Mtok -> 100k * 10.00 / 1e6 = 1.000
    # Total must be 0.015 + 0.250 + 1.000 = 1.265
    # If 10.0 was mistakenly applied universally, total would be 3 * 1.000 = 3.000!
    assert abs(data["cost"]["total_usd"] - 1.265) < 1e-6


def test_cli_model_override_only_targets_specified_model(tmp_path):
    p = tmp_path / "mixed_trace.jsonl"
    call_mini = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "response": {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 100_000, "completion_tokens": 0},
        },
    }
    call_4o = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "response": {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 100_000, "completion_tokens": 0},
        },
    }
    p.write_text(f"{json.dumps(call_mini)}\n{json.dumps(call_4o)}\n", encoding="utf-8")

    # Run with --model gpt-4o --price-per-mtok 5.0 (explicit override of gpt-4o only)
    res = runner.invoke(
        app, ["audit", str(p), "--model", "gpt-4o", "--price-per-mtok", "5.0", "--json"]
    )
    assert res.exit_code == 0
    data = json.loads(res.output)

    # gpt-4o-mini must STILL use $0.15/Mtok = 0.015
    # gpt-4o uses overridden $5.00/Mtok = 0.500
    # Total = 0.515
    assert abs(data["cost"]["total_usd"] - 0.515) < 1e-6


def test_cli_help_documents_pricing_semantics():
    res = runner.invoke(app, ["audit", "--help"])
    assert res.exit_code == 0
    # Help text must clarify fallback pricing vs explicit model override
    assert "fallback pricing" in res.output.lower()
    assert "override" in res.output.lower()
