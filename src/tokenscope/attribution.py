"""Token attribution: the core of TokenScope.

Given a trace, answer: *which categories (and which tools) consumed how many tokens?*

Key idea: in an agent loop the full history is re-sent on every call, so what you
are billed for is very different from what is new. We therefore track two views:

* ``billed``: every input token on every call (what the provider charges for,
  ignoring cache discounts). Sum over calls.
* ``unique``: tokens counted only the first time a segment appears in the trace.
  This approximates the "information" in the run.

``resend_multiplier = billed / unique`` says how many times, on average, the
same content was paid for. High values mean context management is the lever.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field

from tokenscope.models import (
    INPUT_CATEGORIES,
    Category,
    Segment,
    Trace,
    Usage,
    UsageProvenance,
)
from tokenscope.tokenizers import TokenCounter


def fingerprint(segment: Segment) -> str:
    """Stable identity of a segment's content, used to detect re-sent text."""
    h = hashlib.sha256()
    h.update(segment.category.value.encode())
    h.update(b"\x00")
    h.update(segment.text.encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


@dataclass(slots=True)
class CallAttribution:
    call_index: int
    by_category: dict[Category, int]  # billed input tokens for this call
    unique_by_category: dict[Category, int]  # of those, tokens first seen in this call
    output_tokens: int  # estimated tokens generated in this call
    reported_input: int | None  # provider-reported prompt tokens, if present
    usage: Usage | None = None  # full normalized provider usage
    call_id: str | None = None
    model: str | None = None
    provider: str | None = None

    @property
    def estimated_input(self) -> int:
        return sum(self.by_category.values())


@dataclass(slots=True)
class TraceAttribution:
    tokenizer: str
    calls: list[CallAttribution]
    # (category, tool_name) -> billed tokens. Answers "which tool is expensive?"
    by_tool: dict[tuple[Category, str], int] = field(default_factory=dict)
    trace_id: str | None = None
    execution_id: str | None = None
    source_format: str | None = None

    @property
    def billed_by_category(self) -> dict[Category, int]:
        totals: dict[Category, int] = {c: 0 for c in INPUT_CATEGORIES}
        for call in self.calls:
            for cat, n in call.by_category.items():
                totals[cat] += n
        return totals

    @property
    def unique_by_category(self) -> dict[Category, int]:
        totals: dict[Category, int] = {c: 0 for c in INPUT_CATEGORIES}
        for call in self.calls:
            for cat, n in call.unique_by_category.items():
                totals[cat] += n
        return totals

    @property
    def billed_input_total(self) -> int:
        return sum(self.billed_by_category.values())

    @property
    def unique_input_total(self) -> int:
        return sum(self.unique_by_category.values())

    @property
    def output_total(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def resend_multiplier(self) -> float | None:
        unique = self.unique_input_total
        return self.billed_input_total / unique if unique else None

    @property
    def calibration_factor(self) -> float | None:
        """reported / estimated over calls that carry provider usage.

        ~1.0 means our tokenizer matches the provider's. Far from 1.0 means the
        absolute numbers are off (shares between categories are usually still fine).
        """
        reported = est = 0
        for c in self.calls:
            if c.usage is not None and c.usage.provenance == UsageProvenance.PROVIDER:
                reported += c.usage.input_tokens
                est += c.estimated_input
            elif c.usage is None and c.reported_input is not None:
                reported += c.reported_input
                est += c.estimated_input
        return reported / est if est else None

    @property
    def has_reported_usage(self) -> bool:
        """True if any call in this trace contains provider-reported usage."""
        return any(
            c.usage is not None and c.usage.provenance == UsageProvenance.PROVIDER
            for c in self.calls
        )

    @property
    def all_calls_have_reported_usage(self) -> bool:
        """True if every call in this trace contains provider-reported usage."""
        return bool(self.calls) and all(
            c.usage is not None and c.usage.provenance == UsageProvenance.PROVIDER
            for c in self.calls
        )

    @property
    def reported_input_total(self) -> int | None:
        """Sum of provider-reported total prompt tokens across calls with provider usage."""
        if not self.has_reported_usage:
            return None
        return sum(
            c.usage.input_tokens
            for c in self.calls
            if c.usage is not None and c.usage.provenance == UsageProvenance.PROVIDER
        )

    @property
    def reported_uncached_input_total(self) -> int | None:
        """Sum of provider-reported uncached prompt tokens."""
        if not self.has_reported_usage:
            return None
        return sum(
            c.usage.uncached_input_tokens
            for c in self.calls
            if (
                c.usage is not None
                and c.usage.provenance == UsageProvenance.PROVIDER
                and c.usage.uncached_input_tokens is not None
            )
        )

    @property
    def reported_cache_read_input_total(self) -> int | None:
        """Sum of provider-reported cache-read prompt tokens."""
        if not self.has_reported_usage:
            return None
        return sum(
            c.usage.cache_read_input_tokens
            for c in self.calls
            if (
                c.usage is not None
                and c.usage.provenance == UsageProvenance.PROVIDER
                and c.usage.cache_read_input_tokens is not None
            )
        )

    @property
    def reported_cache_creation_input_total(self) -> int | None:
        """Sum of provider-reported cache-creation prompt tokens."""
        if not self.has_reported_usage:
            return None
        return sum(
            c.usage.cache_creation_input_tokens
            for c in self.calls
            if c.usage is not None and c.usage.provenance == UsageProvenance.PROVIDER
        )

    @property
    def reported_output_total(self) -> int | None:
        """Sum of provider-reported output/completion tokens."""
        if not self.has_reported_usage:
            return None
        return sum(
            c.usage.output_tokens
            for c in self.calls
            if c.usage is not None and c.usage.provenance == UsageProvenance.PROVIDER
        )


def attribute(trace: Trace, counter: TokenCounter) -> TraceAttribution:
    """Attribute every input segment of every call to a category (and tool)."""
    seen: set[str] = set()
    token_cache: dict[str, int] = {}
    by_tool: dict[tuple[Category, str], int] = defaultdict(int)
    calls: list[CallAttribution] = []

    for call in trace.calls:
        billed: dict[Category, int] = {c: 0 for c in INPUT_CATEGORIES}
        unique: dict[Category, int] = {c: 0 for c in INPUT_CATEGORIES}

        for seg in call.input_segments:
            fp = fingerprint(seg)
            if fp not in token_cache:
                token_cache[fp] = counter.count(seg.text)
            n = token_cache[fp]

            billed[seg.category] += n
            if fp not in seen:
                seen.add(fp)
                unique[seg.category] += n
            if seg.tool_name:
                by_tool[(seg.category, seg.tool_name)] += n

        output_tokens = sum(counter.count(s.text) for s in call.output_segments)
        reported_input_val = (
            call.usage.input_tokens
            if (call.usage and call.usage.provenance == UsageProvenance.PROVIDER)
            else None
        )
        call_provider = getattr(call, "provider", None) or trace.source_format
        calls.append(
            CallAttribution(
                call_index=call.index,
                by_category=billed,
                unique_by_category=unique,
                output_tokens=output_tokens,
                reported_input=reported_input_val,
                usage=call.usage,
                call_id=call.call_id,
                model=call.model,
                provider=call_provider,
            )
        )

    return TraceAttribution(
        tokenizer=counter.name,
        calls=calls,
        by_tool=dict(by_tool),
        trace_id=trace.trace_id,
        execution_id=trace.execution_id,
        source_format=trace.source_format,
    )
