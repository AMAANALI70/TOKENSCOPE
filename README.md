# TokenScope

**Find where your LLM agent's tokens go. Profile, optimize, verify.**

Agent costs are dominated by *re-sent context*: tool schemas, tool outputs and history
that get paid for again on every call. TokenScope reads your agent traces and tells you
exactly which part of the context is burning tokens, per category and per tool.

> Status: v0.1 (alpha). Profiling works today. Detectors, optimizers and the
> verification harness are on the [roadmap](docs/ROADMAP.md).

## Quick start

```bash
pip install -e ".[tiktoken]"        # tiktoken is optional; falls back to chars/4
tokenscope audit examples/sample_openai_trace.jsonl
tokenscope audit trace.jsonl --price-per-mtok 2.50 --usd-inr 85   # cost view in USD and INR
tokenscope audit trace.jsonl --json                                # machine-readable
```

Example (synthetic sample trace, heuristic tokenizer):

```
category      billed   share   unique
tool_result    3,474   83.0%    1,158
tool_schema      568   13.6%      142
system            48    1.1%       12
user              48    1.1%       12
tool_call         48    1.1%       16
TOTAL          4,186  100.0%    1,340

Resend multiplier: 3.1x   (billed / unique)
```

Reading it: 83% of input tokens are tool results, and the same content was paid for
3.1 times on average. That points at output trimming and context pruning, not at
shortening the system prompt.

## Trace format

One LLM call per line (JSONL), in OpenAI Chat Completions or Anthropic Messages shape,
optionally with the `response` (and its `usage`) attached. See
[docs/DESIGN.md](docs/DESIGN.md#trace-format).

## Pricing and Cost Accounting

TokenScope features a model-aware pricing engine that accounts for:
- Normal (uncached) input tokens
- Cache-read input tokens (discounted prompt caching)
- Cache-creation input tokens (cache writes / setup, including duration-specific 5m and 1h TTL rates)
- Output tokens

### Multi-Model Resolution
In multi-model traces (e.g. mixed GPT-4o and GPT-4o Mini calls), TokenScope resolves rates per call using its built-in `DEFAULT_PRICING` registry with exact-match and longest-prefix resolution.

### Fallback Pricing vs. Explicit Model Overrides
The CLI provides flexible pricing flags:
- **Fallback pricing (`--price-per-mtok` without `--model`)**:
  Specifies fallback pricing applied to calls lacking model information or unrecognized custom models. Known models in the trace continue to resolve their own registry rates and are not universally clobbered.
  ```bash
  # Sets fallback rate ($2.50) for unmodeled calls while known models retain their registry prices
  tokenscope audit trace.jsonl --price-per-mtok 2.50
  ```

- **Explicit model override (`--model <name>` with `--price-per-mtok`)**:
  Overrides the pricing schedule for `<name>` in the registry for that run. Other models in the trace remain unaffected.
  ```bash
  # Overrides gpt-4o input rate to $3.00/Mtok, but preserves gpt-4o-mini at $0.15/Mtok
  tokenscope audit trace.jsonl --model gpt-4o --price-per-mtok 3.00
  ```

- **Duration-specific cache creation pricing**:
  Models like Claude 3.5 Sonnet support duration-specific cache creation rates (e.g. 5m and 1h TTLs). If a trace contains an unconfigured duration tier, TokenScope falls back to the aggregate cache creation rate and flags the cost as approximate.


## Python API

```python
from tokenscope import load_trace, attribute, get_counter

attr = attribute(load_trace("trace.jsonl"), get_counter("auto"))
print(attr.billed_by_category, attr.resend_multiplier)
```

## Docs

- [Design and approach](docs/DESIGN.md)
- [Roadmap](docs/ROADMAP.md)
- [Pitfalls and engineering checklist](docs/PITFALLS.md)

## Privacy

Traces contain prompts, code and possibly secrets. TokenScope runs fully offline and
never uploads anything. Do not commit real traces (`traces/` is git-ignored).

## License

MIT
