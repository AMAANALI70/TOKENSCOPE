"""Turn a TraceAttribution into JSON or a terminal report."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from tokenscope.attribution import TraceAttribution
from tokenscope.models import Category, UsageProvenance
from tokenscope.pricing import CostBreakdown, ModelPricing, PricingRegistry, cost


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:5.1f}%" if whole else "  n/a"


def to_dict(
    attr: TraceAttribution,
    price_per_mtok: float | None = None,
    usd_inr: float | None = None,
    pricing: ModelPricing | PricingRegistry | None = None,
    pricing_registry: PricingRegistry | None = None,
) -> dict[str, Any]:
    billed = attr.billed_by_category
    unique = attr.unique_by_category

    registry = pricing_registry or (pricing if isinstance(pricing, PricingRegistry) else None)
    fallback_pricing = pricing if isinstance(pricing, ModelPricing) else None
    if fallback_pricing is None and price_per_mtok is not None:
        fallback_pricing = ModelPricing(input_cost_per_mtok=price_per_mtok)

    cost_dict: dict[str, Any]
    call_breakdowns: list[CostBreakdown] = []
    reported_calls_count = 0
    estimated_calls_count = 0
    has_unknown_pricing = False

    if not attr.calls:
        cost_dict = {"status": "unknown"}
    else:
        for c in attr.calls:
            # Resolve model-specific pricing per call
            call_pricing: ModelPricing | None = None
            if registry is not None and c.model:
                call_pricing = registry.get(c.model)
            if call_pricing is None:
                call_pricing = fallback_pricing

            if call_pricing is None:
                has_unknown_pricing = True
                break

            # Handle usage provenance per call
            if c.usage is not None and c.usage.provenance == UsageProvenance.PROVIDER:
                call_breakdowns.append(call_pricing.breakdown_usage(c.usage))
                reported_calls_count += 1
            else:
                call_breakdowns.append(
                    call_pricing.breakdown_estimated(c.estimated_input, c.output_tokens)
                )
                estimated_calls_count += 1

        if has_unknown_pricing or not call_breakdowns:
            cost_dict = {"status": "unknown"}
        else:
            uncached_usd = sum(b.uncached_input_cost for b in call_breakdowns)
            read_usd = sum(b.cache_read_cost for b in call_breakdowns)
            create_usd = sum(b.cache_creation_cost for b in call_breakdowns)
            output_usd = sum(b.output_cost for b in call_breakdowns)
            total_usd = sum(b.total_cost for b in call_breakdowns)

            if reported_calls_count == len(attr.calls):
                status = "reported"
                note = None
            elif reported_calls_count > 0:
                status = "partially_reported"
                note = (
                    f"Partially reported: {reported_calls_count}/{len(attr.calls)} calls used "
                    "provider usage, remaining calls used local estimates"
                )
            else:
                status = "estimated"
                note = "Estimated tokens without provider cache discount accounting"

            cost_dict = {
                "status": status,
                "uncached_input_usd": uncached_usd,
                "cache_read_usd": read_usd,
                "cache_creation_usd": create_usd,
                "output_usd": output_usd,
                "total_usd": total_usd,
                "reported_calls": reported_calls_count,
                "estimated_calls": estimated_calls_count,
            }
            if note:
                cost_dict["note"] = note
            if usd_inr is not None:
                cost_dict["total_inr"] = total_usd * usd_inr

    out: dict[str, Any] = {
        "tokenizer": attr.tokenizer,
        "calls": len(attr.calls),
        "trace_id": attr.trace_id,
        "execution_id": attr.execution_id,
        "billed_input_tokens": attr.billed_input_total,
        "unique_input_tokens": attr.unique_input_total,
        "output_tokens_estimated": attr.output_total,
        "resend_multiplier": attr.resend_multiplier,
        "calibration_factor": attr.calibration_factor,
        "estimated": {
            "billed_input_tokens": attr.billed_input_total,
            "unique_input_tokens": attr.unique_input_total,
            "output_tokens": attr.output_total,
            "resend_multiplier": attr.resend_multiplier,
        },
        "reported": (
            {
                "total_input_tokens": attr.reported_input_total,
                "uncached_input_tokens": attr.reported_uncached_input_total,
                "cache_read_input_tokens": attr.reported_cache_read_input_total,
                "cache_creation_input_tokens": attr.reported_cache_creation_input_total,
                "output_tokens": attr.reported_output_total,
                "calibration_factor": attr.calibration_factor,
            }
            if attr.has_reported_usage
            else None
        ),
        "cost": cost_dict,
        "by_category": {
            c.value: {"billed": billed[c], "unique": unique[c]} for c in billed
        },
        "by_tool": [
            {"category": cat.value, "tool": tool, "billed": n}
            for (cat, tool), n in sorted(attr.by_tool.items(), key=lambda kv: -kv[1])
        ],
    }

    if price_per_mtok is not None:
        usd = cost(attr.billed_input_total, price_per_mtok)
        out["input_cost_usd"] = usd
        if usd_inr is not None:
            out["input_cost_inr"] = usd * usd_inr
    return out


def render(
    attr: TraceAttribution,
    console: Console,
    price_per_mtok: float | None = None,
    usd_inr: float | None = None,
    top_tools: int = 10,
    pricing: ModelPricing | PricingRegistry | None = None,
    pricing_registry: PricingRegistry | None = None,
) -> None:
    billed = attr.billed_by_category
    unique = attr.unique_by_category
    total_b, total_u = attr.billed_input_total, attr.unique_input_total

    table = Table(title="Where the input tokens went (locally estimated)", title_justify="left")
    table.add_column("Category")
    table.add_column("Billed", justify="right")
    table.add_column("Share", justify="right")
    table.add_column("Unique", justify="right")
    for cat in sorted(billed, key=lambda c: -billed[c]):
        if billed[cat] == 0:
            continue
        table.add_row(
            cat.value,
            f"{billed[cat]:,}",
            _pct(billed[cat], total_b),
            f"{unique[cat]:,}",
        )
    table.add_row("TOTAL", f"{total_b:,}", "100.0%", f"{total_u:,}", style="bold")
    console.print(table)

    mult = attr.resend_multiplier
    console.print(f"Calls: {len(attr.calls)}   Output tokens (est.): {attr.output_total:,}")
    if mult is not None:
        console.print(
            f"Resend multiplier: [bold]{mult:.1f}x[/bold] "
            "(billed / unique: how many times the same content was paid for)"
        )

    factor = attr.calibration_factor
    if factor is not None:
        console.print(
            f"Tokenizer: {attr.tokenizer}. Provider-reported / estimated = {factor:.2f} "
            "(1.00 = exact match)."
        )
    else:
        console.print(
            f"Tokenizer: {attr.tokenizer}. No provider usage in trace; counts are estimates."
        )

    if attr.has_reported_usage:
        rep_table = Table(title="Provider-reported usage breakdown", title_justify="left")
        rep_table.add_column("Metric")
        rep_table.add_column("Tokens", justify="right")
        if attr.reported_input_total is not None:
            rep_table.add_row("Total Prompt Tokens", f"{attr.reported_input_total:,}")
        if attr.reported_uncached_input_total is not None:
            rep_table.add_row("Uncached Prompt Tokens", f"{attr.reported_uncached_input_total:,}")
        if attr.reported_cache_read_input_total is not None:
            rep_table.add_row("Cache-Read Tokens", f"{attr.reported_cache_read_input_total:,}")
        if attr.reported_cache_creation_input_total is not None:
            rep_table.add_row(
                "Cache-Creation Tokens", f"{attr.reported_cache_creation_input_total:,}"
            )
        if attr.reported_output_total is not None:
            rep_table.add_row("Output Tokens", f"{attr.reported_output_total:,}")
        console.print(rep_table)

    tools = sorted(attr.by_tool.items(), key=lambda kv: -kv[1])[:top_tools]
    if tools:
        t = Table(title="Most expensive tools (billed tokens)", title_justify="left")
        t.add_column("Tool")
        t.add_column("Kind")
        t.add_column("Billed", justify="right")
        for (cat, tool), n in tools:
            kind = {
                Category.TOOL_SCHEMA: "schema",
                Category.TOOL_CALL: "call args",
                Category.TOOL_RESULT: "result",
            }.get(cat, cat.value)
            t.add_row(tool, kind, f"{n:,}")
        console.print(t)

    # Cost reporting
    rep_dict = to_dict(
        attr,
        price_per_mtok=price_per_mtok,
        usd_inr=usd_inr,
        pricing=pricing,
        pricing_registry=pricing_registry,
    )
    cost_info = rep_dict.get("cost", {})
    status = cost_info.get("status")

    if status == "reported":
        total_usd = cost_info["total_usd"]
        line = f"Provider-reported cost: [bold green]${total_usd:,.4f}[/bold green]"
        if usd_inr is not None and "total_inr" in cost_info:
            line += f"  (₹{cost_info['total_inr']:,.2f} at ₹{usd_inr}/USD)"
        console.print(line)
        console.print(
            f"  [dim]Uncached: ${cost_info['uncached_input_usd']:,.4f} | "
            f"Cache-read: ${cost_info['cache_read_usd']:,.4f} | "
            f"Cache-create: ${cost_info['cache_creation_usd']:,.4f} | "
            f"Output: ${cost_info['output_usd']:,.4f}[/dim]"
        )
    elif status == "partially_reported":
        total_usd = cost_info["total_usd"]
        rep_n = cost_info.get("reported_calls", 0)
        est_n = cost_info.get("estimated_calls", 0)
        line = (
            f"Partially reported cost: [bold yellow]${total_usd:,.4f}[/bold yellow] "
            f"[dim]({rep_n}/{len(attr.calls)} calls reported, {est_n} estimated)[/dim]"
        )
        if usd_inr is not None and "total_inr" in cost_info:
            line += f"  (₹{cost_info['total_inr']:,.2f} at ₹{usd_inr}/USD)"
        console.print(line)
    elif status == "estimated":
        total_usd = cost_info["total_usd"]
        line = f"Estimated cost: [bold yellow]${total_usd:,.4f}[/bold yellow]"
        if usd_inr is not None and "total_inr" in cost_info:
            line += f"  (₹{cost_info['total_inr']:,.2f} at ₹{usd_inr}/USD)"
        console.print(line + "  [dim](cache discounts unknown)[/dim]")
    else:
        console.print("Cost: [dim]unknown (no pricing configured)[/dim]")
