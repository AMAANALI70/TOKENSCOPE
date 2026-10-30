"""Command line interface: ``tokenscope audit trace.jsonl``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from tokenscope import __version__
from tokenscope.attribution import attribute
from tokenscope.loader import load_trace
from tokenscope.parsers import TraceParseError
from tokenscope.pricing import DEFAULT_PRICING, ModelPricing
from tokenscope.report import render, to_dict
from tokenscope.tokenizers import TokenizerUnavailableError, get_counter

app = typer.Typer(add_completion=False, help="Find where your LLM agent's tokens go.")


def _version(value: bool) -> None:
    if value:
        typer.echo(f"tokenscope {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    """TokenScope: profile, optimize, verify."""


@app.command()
def audit(
    path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="JSONL/JSON agent trace."),
    ],
    fmt: Annotated[str, typer.Option("--format", help="auto | openai | anthropic")] = "auto",
    tokenizer: Annotated[str, typer.Option(help="auto | tiktoken | heuristic")] = "auto",
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    price_per_mtok: Annotated[
        float | None,
        typer.Option(help="Input price in USD per 1M tokens, to show cost."),
    ] = None,
    usd_inr: Annotated[
        float | None,
        typer.Option(help="USD to INR rate, to also show cost in rupees."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model name for automatic pricing lookup (e.g. gpt-4o)."),
    ] = None,
    output_price_per_mtok: Annotated[
        float | None,
        typer.Option(help="Output price in USD per 1M tokens."),
    ] = None,
    cache_read_price_per_mtok: Annotated[
        float | None,
        typer.Option(help="Cache-read price in USD per 1M tokens."),
    ] = None,
    cache_creation_price_per_mtok: Annotated[
        float | None,
        typer.Option(help="Cache-creation price in USD per 1M tokens."),
    ] = None,
) -> None:
    """Attribute every input token in a trace to a category and tool."""
    try:
        trace = load_trace(path, fmt)
        counter = get_counter(tokenizer)
    except (TraceParseError, TokenizerUnavailableError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    pricing: ModelPricing | None = None
    target_model = model or (trace.calls[0].model if trace.calls and trace.calls[0].model else None)

    if model:
        pricing = DEFAULT_PRICING.get(model, trace.source_format)

    if (
        price_per_mtok is not None
        or output_price_per_mtok is not None
        or cache_read_price_per_mtok is not None
        or cache_creation_price_per_mtok is not None
    ):
        base = pricing or (DEFAULT_PRICING.get(target_model) if target_model else None)
        pricing = ModelPricing(
            input_cost_per_mtok=(
                price_per_mtok
                if price_per_mtok is not None
                else (base.input_cost_per_mtok if base else 0.0)
            ),
            output_cost_per_mtok=(
                output_price_per_mtok
                if output_price_per_mtok is not None
                else (base.output_cost_per_mtok if base else 0.0)
            ),
            cache_read_cost_per_mtok=(
                cache_read_price_per_mtok
                if cache_read_price_per_mtok is not None
                else (base.cache_read_cost_per_mtok if base else 0.0)
            ),
            cache_creation_cost_per_mtok=(
                cache_creation_price_per_mtok
                if cache_creation_price_per_mtok is not None
                else (base.cache_creation_cost_per_mtok if base else 0.0)
            ),
        )

    attr = attribute(trace, counter)
    if as_json:
        typer.echo(
            json.dumps(
                to_dict(attr, price_per_mtok=price_per_mtok, usd_inr=usd_inr, pricing=pricing),
                indent=2,
            )
        )
    else:
        render(
            attr,
            Console(),
            price_per_mtok=price_per_mtok,
            usd_inr=usd_inr,
            pricing=pricing,
        )


if __name__ == "__main__":
    app()
