"""Turn a TraceAttribution into JSON or a terminal report."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from tokenscope.attribution import TraceAttribution
from tokenscope.models import Category
from tokenscope.pricing import CostBreakdown, ModelPricing, cost


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:5.1f}%" if whole else "  n/a"


def to_dict(
    attr: TraceAttribution,
    price_per_mtok: float | None = None,
    usd_inr: float | None = None,
    pricing: ModelPricing | None = None,
) -> dict[str, Any]:
    billed = attr.billed_by_category
    unique = attr.unique_by_category

    effective_pricing = pricing
    if effective_pricing is None and price_per_mtok is not None:
        effective_pricing = ModelPricing(input_cost_per_mtok=price_per_mtok)

    cost_dict: dict[str, Any]
    if effective_pricing is not None:
        if attr.has_reported_usage:
            call_costs: list[CostBreakdown] = []
            for c in attr.calls:
                if c.usage:
                    call_costs.append(effective_pricing.breakdown_usage(c.usage))
            uncached_usd = sum(b.uncached_input_cost for b in call_costs)
            read_usd = sum(b.cache_read_cost for b in call_costs)
            create_usd = sum(b.cache_creation_cost for b in call_costs)
            output_usd = sum(b.output_cost for b in call_costs)
            total_usd = sum(b.total_cost for b in call_costs)
            status = "reported" if attr.all_calls_have_reported_usage else "partially_reported"
            cost_dict = {
                "status": status,
                "uncached_input_usd": uncached_usd,
                "cache_read_usd": read_usd,
                "cache_creation_usd": create_usd,
                "output_usd": output_usd,
                "total_usd": total_usd,
            }
        else:
            est_b = effective_pricing.breakdown_estimated(
                attr.billed_input_total,
                attr.output_total,
            )
            cost_dict = {
                "status": "estimated",
                "uncached_input_usd": est_b.uncached_input_cost,
                "cache_read_usd": None,
                "cache_creation_usd": None,
                "output_usd": est_b.output_cost,
                "total_usd": est_b.total_cost,
                "note": "Estimated tokens without provider cache discount accounting",
            }
        if usd_inr is not None and "total_usd" in cost_dict and cost_dict["total_usd"] is not None:
            cost_dict["total_inr"] = cost_dict["total_usd"] * usd_inr
    else:
        cost_dict = {"status": "unknown"}

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
    pricing: ModelPricing | None = None,
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
    effective_pricing = pricing
    if effective_pricing is None and price_per_mtok is not None:
        effective_pricing = ModelPricing(input_cost_per_mtok=price_per_mtok)

    if effective_pricing is not None:
        rep_dict = to_dict(attr, price_per_mtok, usd_inr, pricing=effective_pricing)
        cost_info = rep_dict.get("cost", {})
        status = cost_info.get("status")

        if status in ("reported", "partially_reported"):
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
        elif status == "estimated":
            total_usd = cost_info["total_usd"]
            line = f"Estimated cost: [bold yellow]${total_usd:,.4f}[/bold yellow]"
            if usd_inr is not None and "total_inr" in cost_info:
                line += f"  (₹{cost_info['total_inr']:,.2f} at ₹{usd_inr}/USD)"
            console.print(line + "  [dim](cache discounts unknown)[/dim]")
    else:
        console.print("Cost: [dim]unknown (no pricing configured)[/dim]")
