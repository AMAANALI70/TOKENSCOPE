# Roadmap

~1-2 hours/day. Ship something usable by the end of week 2.

| Version | Weeks | Deliverable | Done when |
|---|---|---|---|
| v0.1 | 1-2 | Parsers, attribution, CLI report, JSON output | `tokenscope audit` works on a real trace of yours; CI green; PyPI test release |
| v0.2 | 3-4 | Detectors: repeated reads, oversized tool outputs, unused tool schemas, cache-hostile layout, retry loops | Each detector has a test and a clear, actionable finding message |
| v0.3 | 5-6 | Optimizers: tool-output trimming (logs/JSON/HTML), context dedup, cache-aware reordering | Optimizers are pure functions `Trace -> Trace` with tests |
| v0.4 | 7-8 | Replay + verification harness, baseline comparison, cost-vs-success chart | 20+ tasks, variance reported, one baseline compared |
| v0.5 | 9-10 | Docs, write-up, demo, launch | Blog post, Show HN draft, README GIF |

## Backlog (do NOT start before v0.4)
- Live proxy mode (FastAPI/httpx) that logs traces automatically
- More formats (Gemini, LangChain / LlamaIndex callbacks, OpenTelemetry GenAI spans)
- HTML report
- Cache-discount-aware cost model
- Multimodal token accounting

## First week checklist
- [ ] Create GitHub repo, push this scaffold, confirm CI passes
- [ ] Run `tokenscope audit` on a real trace from an agent you built
- [ ] Add one more parser fixture from a real run (scrub secrets first)
- [ ] Write a 3-line build-log post: what you shipped, one surprising number
