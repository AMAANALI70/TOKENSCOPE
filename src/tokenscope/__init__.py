"""TokenScope: find where your LLM agent's tokens go."""

from tokenscope.attribution import CallAttribution, TraceAttribution, attribute
from tokenscope.loader import load_trace
from tokenscope.models import (
    Call,
    Category,
    ContentBlock,
    Message,
    Segment,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Trace,
    Usage,
    UsageProvenance,
)
from tokenscope.pricing import (
    DEFAULT_PRICING,
    CostBreakdown,
    ModelPricing,
    PricingRegistry,
)
from tokenscope.tokenizers import TokenCounter, get_counter

__version__ = "0.1.0"

__all__ = [
    "Call",
    "CallAttribution",
    "Category",
    "ContentBlock",
    "CostBreakdown",
    "DEFAULT_PRICING",
    "Message",
    "ModelPricing",
    "PricingRegistry",
    "Segment",
    "TokenCounter",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "Trace",
    "TraceAttribution",
    "Usage",
    "UsageProvenance",
    "attribute",
    "get_counter",
    "load_trace",
]
