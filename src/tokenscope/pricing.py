"""Provider and model-aware pricing abstraction.

Accounts for:
- Normal (uncached) input tokens.
- Cache-read input tokens (discounted prompt caching).
- Cache-creation input tokens (cache write / caching setup).
- Duration-specific cache creation rates (e.g. Anthropic 5m vs 1h TTL).
- Output / completion tokens.

Strictly separated from attribution logic to keep token accounting pure.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenscope.models import Usage


# Canonical duration aliases mapping common representations to normalized duration keys
_CANONICAL_DURATION_ALIASES: dict[str, str] = {
    # 5-minute aliases
    "5m": "5m",
    "5min": "5m",
    "5-minute": "5m",
    "5_minute": "5m",
    "ephemeral_5m": "5m",
    "ephemeral_5m_input_tokens": "5m",
    "cache_creation_5m": "5m",
    "cache_creation_ephemeral_5m": "5m",
    # 1-hour aliases
    "1h": "1h",
    "1hr": "1h",
    "1-hour": "1h",
    "1_hour": "1h",
    "60m": "1h",
    "ephemeral_1h": "1h",
    "ephemeral_1h_input_tokens": "1h",
    "cache_creation_1h": "1h",
    "cache_creation_ephemeral_1h": "1h",
}


def normalize_duration_key(key: str) -> str | None:
    """Normalize a duration string or field name to its canonical identifier.

    Returns canonical string (e.g. '5m', '1h') if recognized, else None.
    Arbitrary substring matching is strictly avoided.
    """
    return _CANONICAL_DURATION_ALIASES.get(key.strip().lower())


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Itemized cost components in USD."""

    uncached_input_cost: float
    cache_read_cost: float
    cache_creation_cost: float
    output_cost: float
    total_cost: float
    is_provider_reported: bool
    currency: str = "USD"
    is_approximate: bool = False
    cache_creation_details_cost: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    has_unresolved_cost: bool = False
    unresolved_cache_creation_tokens: int = 0
    is_inconsistent: bool = False


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Token pricing rates in USD per 1,000,000 tokens (Mtok).

    Normal input, cached input read, cache write, and output each have their own rate.
    Duration-specific cache creation rates can be specified via cache_creation_duration_rates.
    """

    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    cache_read_cost_per_mtok: float = 0.0
    cache_creation_cost_per_mtok: float | None = None
    cache_creation_duration_rates: dict[str, float] = field(default_factory=dict)

    def get_duration_cache_creation_rate(self, duration_key: str) -> float | None:
        """Lookup rate for a duration category using explicit normalization and aliases.

        Matches exact duration keys or canonical aliases (e.g. '5m', 'ephemeral_5m_input_tokens').
        Arbitrary substring matches (e.g. '15m' matching '5m') are strictly rejected.
        """
        if not self.cache_creation_duration_rates:
            return None

        rates_lower = {k.lower().strip(): v for k, v in self.cache_creation_duration_rates.items()}
        raw_key = duration_key.lower().strip()

        # 1. Exact match against registered keys
        if raw_key in rates_lower:
            return rates_lower[raw_key]

        # 2. Canonical alias match
        canonical = normalize_duration_key(raw_key)
        if canonical is not None:
            if canonical in rates_lower:
                return rates_lower[canonical]
            for reg_k, reg_v in rates_lower.items():
                if normalize_duration_key(reg_k) == canonical:
                    return reg_v

        return None

    def compute_cost_breakdown(
        self,
        uncached_input: int,
        output: int,
        cache_read: int = 0,
        cache_creation: int = 0,
        cache_creation_details: dict[str, int] | None = None,
        is_provider_reported: bool = True,
        is_inconsistent: bool = False,
    ) -> CostBreakdown:
        """Itemize cost components sharing the single unified calculation engine."""
        uncached_cost = (uncached_input / 1_000_000) * self.input_cost_per_mtok
        cache_read_cost = (cache_read / 1_000_000) * self.cache_read_cost_per_mtok
        output_cost = (output / 1_000_000) * self.output_cost_per_mtok

        c_cost = 0.0
        details_costs: dict[str, float] = {}
        is_approximate = False
        notes: list[str] = []
        has_unresolved_cost = False
        unresolved_tokens = 0

        has_configured_aggregate = self.cache_creation_cost_per_mtok is not None
        has_defensible_aggregate = has_configured_aggregate and (
            (
                self.cache_creation_cost_per_mtok is not None
                and self.cache_creation_cost_per_mtok > 0.0
            )
            or not self.cache_creation_duration_rates
        )

        if cache_creation_details:
            details_total = sum(cache_creation_details.values())
            if details_total > cache_creation or is_inconsistent:
                is_inconsistent = True
                has_unresolved_cost = True
                is_approximate = True
                unresolved_tokens = cache_creation
                notes.append(
                    f"Inconsistent cache creation breakdown: details sum ({details_total:,}) "
                    f"exceeds aggregate tokens ({cache_creation:,}). "
                    "Cost is invalid and marked unresolved."
                )
                for dur_key, count in cache_creation_details.items():
                    dur_rate = self.get_duration_cache_creation_rate(dur_key)
                    details_costs[dur_key] = (
                        (count / 1_000_000) * dur_rate if dur_rate is not None else 0.0
                    )
                c_cost = 0.0
            else:
                details_tokens_accounted = 0
                for dur_key, count in cache_creation_details.items():
                    dur_rate = self.get_duration_cache_creation_rate(dur_key)
                    if dur_rate is not None:
                        cat_cost = (count / 1_000_000) * dur_rate
                        details_costs[dur_key] = cat_cost
                        c_cost += cat_cost
                        details_tokens_accounted += count
                    else:
                        is_approximate = True
                        if has_defensible_aggregate:
                            rate_val = self.cache_creation_cost_per_mtok or 0.0
                            notes.append(
                                f"Duration rate for {dur_key!r} unavailable; "
                                f"used aggregate rate ${rate_val}/Mtok."
                            )
                            cat_cost = (count / 1_000_000) * rate_val
                            details_costs[dur_key] = cat_cost
                            c_cost += cat_cost
                        else:
                            has_unresolved_cost = True
                            unresolved_tokens += count
                            notes.append(
                                f"No defensible cache creation rate exists for {dur_key!r} "
                                f"({count:,} tokens); cost is unresolved."
                            )
                            details_costs[dur_key] = 0.0
                        details_tokens_accounted += count

                rem_tokens = max(0, cache_creation - details_tokens_accounted)
                if rem_tokens > 0:
                    is_approximate = True
                    if has_defensible_aggregate:
                        rate_val = self.cache_creation_cost_per_mtok or 0.0
                        rem_cost = (rem_tokens / 1_000_000) * rate_val
                        c_cost += rem_cost
                        details_costs["other"] = rem_cost
                        notes.append(
                            f"{rem_tokens:,} cache creation tokens unclassified by duration; "
                            f"approximated using rate ${rate_val}/Mtok."
                        )
                    else:
                        has_unresolved_cost = True
                        unresolved_tokens += rem_tokens
                        notes.append(
                            f"No defensible cache creation rate exists for {rem_tokens:,} "
                            "unclassified cache creation tokens; cost is unresolved."
                        )
                        details_costs["other"] = 0.0
        else:
            if cache_creation > 0:
                if self.cache_creation_duration_rates:
                    is_approximate = True
                    if has_defensible_aggregate:
                        rate_val = self.cache_creation_cost_per_mtok or 0.0
                        c_cost = (cache_creation / 1_000_000) * rate_val
                        notes.append(
                            "Cache creation duration details missing; "
                            f"approximated using rate ${rate_val}/Mtok."
                        )
                    else:
                        has_unresolved_cost = True
                        unresolved_tokens += cache_creation
                        notes.append(
                            "Cache creation duration details missing and no aggregate rate "
                            f"exists ({cache_creation:,} tokens); cost is unresolved."
                        )
                else:
                    if has_configured_aggregate:
                        rate_val = self.cache_creation_cost_per_mtok or 0.0
                        c_cost = (cache_creation / 1_000_000) * rate_val
                    else:
                        has_unresolved_cost = True
                        is_approximate = True
                        unresolved_tokens += cache_creation
                        notes.append(
                            f"No cache creation pricing configured for {cache_creation:,} "
                            "cache creation tokens; cost is unresolved."
                        )

        total = uncached_cost + cache_read_cost + c_cost + output_cost

        return CostBreakdown(
            uncached_input_cost=uncached_cost,
            cache_read_cost=cache_read_cost,
            cache_creation_cost=c_cost,
            output_cost=output_cost,
            total_cost=total,
            is_provider_reported=is_provider_reported,
            is_approximate=is_approximate,
            cache_creation_details_cost=details_costs,
            notes=notes,
            has_unresolved_cost=has_unresolved_cost,
            unresolved_cache_creation_tokens=unresolved_tokens,
            is_inconsistent=is_inconsistent,
        )

    def calculate_cost(
        self,
        uncached_input: int,
        output: int,
        cache_read: int = 0,
        cache_creation: int = 0,
        cache_creation_details: dict[str, int] | None = None,
    ) -> float:
        """Compute total dollar cost for given token counts."""
        breakdown = self.compute_cost_breakdown(
            uncached_input=uncached_input,
            output=output,
            cache_read=cache_read,
            cache_creation=cache_creation,
            cache_creation_details=cache_creation_details,
            is_provider_reported=False,
        )
        if breakdown.is_approximate:
            warnings.warn(
                f"Calculated cost is approximate: {'; '.join(breakdown.notes)}",
                UserWarning,
                stacklevel=2,
            )
        return breakdown.total_cost

    def breakdown_usage(self, usage: Usage) -> CostBreakdown:
        """Itemize cost based on provider-reported usage."""
        uncached = (
            usage.uncached_input_tokens
            if usage.uncached_input_tokens is not None
            else max(
                0,
                usage.input_tokens
                - (usage.cache_read_input_tokens or 0)
                - usage.cache_creation_input_tokens,
            )
        )
        return self.compute_cost_breakdown(
            uncached_input=uncached,
            output=usage.output_tokens,
            cache_read=usage.cache_read_input_tokens or 0,
            cache_creation=usage.cache_creation_input_tokens,
            cache_creation_details=usage.cache_creation_details,
            is_provider_reported=True,
            is_inconsistent=usage.is_inconsistent,
        )

    def breakdown_estimated(self, estimated_input: int, estimated_output: int) -> CostBreakdown:
        """Estimate cost without cache discount knowledge."""
        u_cost = (estimated_input / 1_000_000) * self.input_cost_per_mtok
        o_cost = (estimated_output / 1_000_000) * self.output_cost_per_mtok
        total = u_cost + o_cost
        return CostBreakdown(
            uncached_input_cost=u_cost,
            cache_read_cost=0.0,
            cache_creation_cost=0.0,
            output_cost=o_cost,
            total_cost=total,
            is_provider_reported=False,
        )


class PricingRegistry:
    """Registry mapping model names/patterns to their pricing schedules."""

    def __init__(self, rates: dict[str, ModelPricing] | None = None) -> None:
        self._rates: dict[str, ModelPricing] = {}
        if rates:
            for k, v in rates.items():
                self.register(k, v)

    def register(self, model_key: str, pricing: ModelPricing) -> None:
        self._rates[model_key.lower()] = pricing

    def get(self, model: str | None, provider: str | None = None) -> ModelPricing | None:
        if not model:
            return None
        key = model.lower()

        # 1. Exact matches: provider-specific exact match first, then generic exact match
        if provider:
            namespaced = f"{provider.lower()}/{key}"
            if namespaced in self._rates:
                return self._rates[namespaced]
        if key in self._rates:
            return self._rates[key]

        # 2. Prefix matches: candidate collection with longest prefix and provider priority
        candidates: list[tuple[int, bool, ModelPricing]] = []
        for reg_key, pricing in self._rates.items():
            if "/" in reg_key:
                pfx_provider, pfx_model = reg_key.split("/", 1)
                if provider and pfx_provider == provider.lower() and key.startswith(pfx_model):
                    candidates.append((len(pfx_model), True, pricing))
            elif key.startswith(reg_key):
                candidates.append((len(reg_key), False, pricing))

        if candidates:
            # Provider-specific prefix matches win over generic, sorted by longest prefix
            candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
            return candidates[0][2]

        return None


# Reference pricing for standard tier models (USD per 1M tokens)
# Users can override or register custom models in the registry.
DEFAULT_PRICING = PricingRegistry(
    {
        # OpenAI GPT-4o family
        "gpt-4o": ModelPricing(
            input_cost_per_mtok=2.50,
            output_cost_per_mtok=10.00,
            cache_read_cost_per_mtok=1.25,
            cache_creation_cost_per_mtok=0.00,
        ),
        "gpt-4o-mini": ModelPricing(
            input_cost_per_mtok=0.15,
            output_cost_per_mtok=0.60,
            cache_read_cost_per_mtok=0.075,
            cache_creation_cost_per_mtok=0.00,
        ),
        "o1": ModelPricing(
            input_cost_per_mtok=15.00,
            output_cost_per_mtok=60.00,
            cache_read_cost_per_mtok=7.50,
            cache_creation_cost_per_mtok=0.00,
        ),
        "o1-mini": ModelPricing(
            input_cost_per_mtok=3.00,
            output_cost_per_mtok=12.00,
            cache_read_cost_per_mtok=1.50,
            cache_creation_cost_per_mtok=0.00,
        ),
        # Anthropic Claude 3.5 family (with duration-specific cache rates: 5m @ 1.25x, 1h @ 2.0x)
        "claude-3-5-sonnet": ModelPricing(
            input_cost_per_mtok=3.00,
            output_cost_per_mtok=15.00,
            cache_read_cost_per_mtok=0.30,
            cache_creation_cost_per_mtok=3.75,
            cache_creation_duration_rates={
                "ephemeral_5m_input_tokens": 3.75,
                "ephemeral_1h_input_tokens": 6.00,
                "5m": 3.75,
                "1h": 6.00,
            },
        ),
        "claude-3-5-haiku": ModelPricing(
            input_cost_per_mtok=1.00,
            output_cost_per_mtok=5.00,
            cache_read_cost_per_mtok=0.10,
            cache_creation_cost_per_mtok=1.25,
            cache_creation_duration_rates={
                "ephemeral_5m_input_tokens": 1.25,
                "ephemeral_1h_input_tokens": 2.00,
                "5m": 1.25,
                "1h": 2.00,
            },
        ),
        "claude-3-opus": ModelPricing(
            input_cost_per_mtok=15.00,
            output_cost_per_mtok=75.00,
            cache_read_cost_per_mtok=1.50,
            cache_creation_cost_per_mtok=18.75,
            cache_creation_duration_rates={
                "ephemeral_5m_input_tokens": 18.75,
                "ephemeral_1h_input_tokens": 30.00,
                "5m": 18.75,
                "1h": 30.00,
            },
        ),
    }
)


def cost(tokens: int, price_per_mtok: float) -> float:
    """Compute basic token cost. Preserved for backward compatibility."""
    return tokens / 1_000_000 * price_per_mtok
