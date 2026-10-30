import io

from rich.console import Console

from tokenscope.attribution import attribute
from tokenscope.models import Call, Category, Segment, Trace, Usage
from tokenscope.pricing import ModelPricing
from tokenscope.report import render, to_dict
from tokenscope.tokenizers import HeuristicCounter

C = HeuristicCounter()


def test_reported_vs_estimated_separation():
    # 40 chars = 10 estimated tokens
    seg1 = Segment(Category.USER, "a" * 40)
    # 80 chars = 20 estimated tokens
    seg2 = Segment(Category.USER, "b" * 80)

    u1 = Usage(
        input_tokens=150,
        output_tokens=15,
        uncached_input_tokens=90,
        cache_read_input_tokens=50,
        cache_creation_input_tokens=10,
    )
    u2 = Usage(
        input_tokens=200,
        output_tokens=25,
        uncached_input_tokens=160,
        cache_read_input_tokens=40,
        cache_creation_input_tokens=0,
    )

    calls = [
        Call(0, "gpt-4o", [seg1], usage=u1),
        Call(1, "gpt-4o", [seg2], usage=u2),
    ]
    trace = Trace(calls, "openai", trace_id="tr-1", execution_id="ex-1")
    attr = attribute(trace, C)

    # Locally estimated metrics
    assert attr.billed_input_total == 10 + 20
    assert attr.unique_input_total == 10 + 20
    assert attr.output_total == 0

    # Provider-reported metrics
    assert attr.has_reported_usage is True
    assert attr.all_calls_have_reported_usage is True
    assert attr.reported_input_total == 150 + 200
    assert attr.reported_uncached_input_total == 90 + 160
    assert attr.reported_cache_read_input_total == 50 + 40
    assert attr.reported_cache_creation_input_total == 10 + 0
    assert attr.reported_output_total == 15 + 25

    # Check TraceAttribution execution identifiers
    assert attr.trace_id == "tr-1"
    assert attr.execution_id == "ex-1"


def test_mixed_reported_usage():
    seg = Segment(Category.USER, "a" * 40)
    u = Usage(input_tokens=100, output_tokens=10)
    calls = [
        Call(0, "gpt-4o", [seg], usage=u),
        Call(1, "gpt-4o", [seg], usage=None),
    ]
    attr = attribute(Trace(calls, "openai"), C)
    assert attr.has_reported_usage is True
    assert attr.all_calls_have_reported_usage is False
    assert attr.reported_input_total == 100
    assert attr.reported_output_total == 10


def test_no_reported_usage():
    seg = Segment(Category.USER, "a" * 40)
    calls = [Call(0, "gpt-4o", [seg])]
    attr = attribute(Trace(calls, "openai"), C)
    assert attr.has_reported_usage is False
    assert attr.all_calls_have_reported_usage is False
    assert attr.reported_input_total is None
    assert attr.reported_uncached_input_total is None
    assert attr.reported_cache_read_input_total is None
    assert attr.reported_cache_creation_input_total is None
    assert attr.reported_output_total is None


def test_report_to_dict_distinguishes_reported_and_estimated():
    seg = Segment(Category.USER, "a" * 40)  # 10 estimated tokens
    usage = Usage(
        input_tokens=100,
        output_tokens=10,
        uncached_input_tokens=70,
        cache_read_input_tokens=30,
        cache_creation_input_tokens=0,
    )
    attr = attribute(Trace([Call(0, "gpt-4o", [seg], usage=usage)], "openai"), C)

    pricing = ModelPricing(
        input_cost_per_mtok=2.50,
        output_cost_per_mtok=10.00,
        cache_read_cost_per_mtok=1.25,
    )

    data = to_dict(attr, pricing=pricing, usd_inr=85.0)

    # Top-level backward compatible fields
    assert data["calls"] == 1
    assert data["billed_input_tokens"] == 10

    # Structured sections
    assert "estimated" in data
    assert data["estimated"]["billed_input_tokens"] == 10

    assert "reported" in data
    assert data["reported"]["total_input_tokens"] == 100
    assert data["reported"]["uncached_input_tokens"] == 70
    assert data["reported"]["cache_read_input_tokens"] == 30

    assert "cost" in data
    assert data["cost"]["status"] == "reported"
    # Uncached: 70 * 2.50 / 1e6; Cached: 30 * 1.25 / 1e6; Output: 10 * 10 / 1e6
    expected_usd = (70 * 2.50 + 30 * 1.25 + 10 * 10.00) / 1_000_000
    assert abs(data["cost"]["total_usd"] - expected_usd) < 1e-7
    assert "total_inr" in data["cost"]


def test_report_cost_unknown_when_no_pricing():
    seg = Segment(Category.USER, "a" * 40)
    attr = attribute(Trace([Call(0, "gpt-4o", [seg])], "openai"), C)
    data = to_dict(attr)
    assert data["cost"]["status"] == "unknown"


def test_render_does_not_crash():
    buf = io.StringIO()
    console = Console(file=buf)
    seg = Segment(Category.USER, "a" * 40)
    usage = Usage(input_tokens=100, output_tokens=10, cache_read_input_tokens=20)
    attr = attribute(Trace([Call(0, "gpt-4o", [seg], usage=usage)], "openai"), C)

    pricing = ModelPricing(input_cost_per_mtok=2.50, output_cost_per_mtok=10.00)
    render(attr, console, pricing=pricing, usd_inr=85.0)

    # Without usage and without pricing
    attr_no_usage = attribute(Trace([Call(0, "gpt-4o", [seg])], "openai"), C)
    render(attr_no_usage, console)
    assert len(buf.getvalue()) > 0
