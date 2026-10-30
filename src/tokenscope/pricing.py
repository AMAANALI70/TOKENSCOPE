"""Provider and model-aware pricing abstraction.

Accounts for:
- Normal (uncached) input tokens.
- Cache-read input tokens (discounted prompt caching).
- Cache-creation input tokens (cache write / caching setup).
- Output / completion tokens.

Strictly separated from attribution logic to keep token accounting pure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenscope.models import Usage


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


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Token pricing rates in USD per 1,000,000 tokens (Mtok).

    Normal input, cached input read, cache write, and output each have their own rate.
    """

    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    cache_read_cost_per_mtok: float = 0.0
    cache_creation_cost_per_mtok: float = 0.0

    def calculate_cost(
        self,
        uncached_input: int,
        output: int,
        cache_read: int = 0,
        cache_creation: int = 0,
    ) -> float:
        """Compute total dollar cost for given token counts."""
        uncached_cost = (uncached_input / 1_000_000) * self.input_cost_per_mtok
        cache_read_cost = (cache_read / 1_000_000) * self.cache_read_cost_per_mtok
        cache_create_cost = (cache_creation / 1_000_000) * self.cache_creation_cost_per_mtok
        output_cost = (output / 1_000_000) * self.output_cost_per_mtok
        return uncached_cost + cache_read_cost + cache_create_cost + output_cost

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
        read = usage.cache_read_input_tokens or 0
        create = usage.cache_creation_input_tokens or 0
        out = usage.output_tokens

        u_cost = (uncached / 1_000_000) * self.input_cost_per_mtok
        r_cost = (read / 1_000_000) * self.cache_read_cost_per_mtok
        c_cost = (create / 1_000_000) * self.cache_creation_cost_per_mtok
        o_cost = (out / 1_000_000) * self.output_cost_per_mtok
        total = u_cost + r_cost + c_cost + o_cost

        return CostBreakdown(
            uncached_input_cost=u_cost,
            cache_read_cost=r_cost,
            cache_creation_cost=c_cost,
            output_cost=o_cost,
            total_cost=total,
            is_provider_reported=True,
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
DEFAULT_PRICING = PricingRegistry({
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
    # Anthropic Claude 3.5 family
    "claude-3-5-sonnet": ModelPricing(
        input_cost_per_mtok=3.00,
        output_cost_per_mtok=15.00,
        cache_read_cost_per_mtok=0.30,
        cache_creation_cost_per_mtok=3.75,
    ),
    "claude-3-5-haiku": ModelPricing(
        input_cost_per_mtok=1.00,
        output_cost_per_mtok=5.00,
        cache_read_cost_per_mtok=0.10,
        cache_creation_cost_per_mtok=1.25,
    ),
    "claude-3-opus": ModelPricing(
        input_cost_per_mtok=15.00,
        output_cost_per_mtok=75.00,
        cache_read_cost_per_mtok=1.50,
        cache_creation_cost_per_mtok=18.75,
    ),
})


def cost(tokens: int, price_per_mtok: float) -> float:
    """Compute basic token cost. Preserved for backward compatibility."""
    return tokens / 1_000_000 * price_per_mtok
