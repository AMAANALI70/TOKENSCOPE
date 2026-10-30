# Pitfalls and engineering checklist

## Correctness
- **Tokenizer mismatch.** tiktoken is wrong for Claude/Gemini/Llama. Always surface the
  calibration factor and never present absolute numbers as exact unless it is ~1.0.
- **Cache discounts.** "Billed" here is pre-discount. Cached input is cheaper
  (provider-specific). Do not claim dollar savings without accounting for cache hits.
- **Compression can destroy caching.** Rewriting an earlier part of the prompt changes
  the prefix and invalidates the cache, which can cost more than it saves. Any optimizer
  that edits history must be evaluated for cache impact.
- **Don't compress structured data blindly.** Dropping tokens from JSON/code can break
  parsing. Trimming must be structure-aware (truncate arrays, drop fields, keep head/tail).
- **Multimodal content is ignored** in v0.1 and undercounts such traces.
- **Message framing overhead** is not modelled (small, shows as drift).

## Evaluation honesty
- Savings without a quality check are marketing. Every optimizer needs a cost-vs-success result.
- LLM behavior is noisy. Repeat runs, report variance, fix temperature, pin model versions.
- Report negative results. They are the most credible part of the project.
- Never quote a figure from a vendor blog as your own; measure it.

## Security and privacy
- Traces contain prompts, source code, PII and secrets. Keep everything local by default.
- `traces/` is git-ignored. Scrub fixtures before committing and use synthetic data in tests/examples.
- Never log raw segment text in error messages.

## Engineering hygiene
- Keep the neutral model the only thing downstream code touches.
- Pure functions for detectors and optimizers (`Trace -> findings` / `Trace -> Trace`): easy to test.
- Type hints everywhere; `mypy --strict` stays green.
- Pin nothing needlessly; use lower bounds in `pyproject.toml`.
- Small PRs to yourself, conventional commits, a CHANGELOG from v0.1.
- Performance: fingerprint caching is enough for traces up to ~10^5 segments. Profile before optimizing.

## Scope discipline
- New idea? Put it in the backlog. Finish the current version first.
- Ship v0.1 early to real users; their traces will break your assumptions fast.
- Resist adding ML. Deterministic, explainable rules are the differentiator here.

## Launch checklist (v0.5)
- [ ] README top: one chart, one command, one result
- [ ] `pip install` works in a clean venv
- [ ] Demo GIF/video under 60s
- [ ] One blog post with a real finding and a cost-vs-success chart
- [ ] Show HN / r/LocalLLaMA / LinkedIn / X posts; tag maintainers of related projects
- [ ] One PR or issue to a related open-source project
