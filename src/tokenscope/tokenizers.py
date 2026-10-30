"""Token counting.

Counting is pluggable because no single tokenizer is correct for every model.
``tiktoken`` is exact for OpenAI models and an approximation for others; the
heuristic needs no dependencies and no network. Attribution reports "drift"
against provider-reported usage so you always know how far off an estimate is.
"""

from __future__ import annotations

import math
from typing import Protocol


class TokenizerUnavailableError(RuntimeError):
    """Requested tokenizer cannot be loaded (missing package or offline)."""


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class HeuristicCounter:
    """~4 characters per token. Dependency-free and deterministic."""

    name = "heuristic:chars/4"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))


class TiktokenCounter:
    """Exact for OpenAI-family models, approximate for others."""

    def __init__(self, encoding: str = "o200k_base") -> None:
        try:
            import tiktoken

            self._enc = tiktoken.get_encoding(encoding)
        except Exception as exc:  # missing package, or encoding download failed
            raise TokenizerUnavailableError(
                f"tiktoken encoding {encoding!r} unavailable: {exc}"
            ) from exc
        self.name = f"tiktoken:{encoding}"

    def count(self, text: str) -> int:
        if not text:
            return 0
        # disallowed_special=() so literal "<|endoftext|>" in logs doesn't raise.
        return len(self._enc.encode(text, disallowed_special=()))


def get_counter(kind: str = "auto") -> TokenCounter:
    """``auto`` tries tiktoken and silently falls back to the heuristic."""
    if kind == "heuristic":
        return HeuristicCounter()
    if kind == "tiktoken":
        return TiktokenCounter()
    if kind == "auto":
        try:
            return TiktokenCounter()
        except TokenizerUnavailableError:
            return HeuristicCounter()
    raise ValueError(f"unknown tokenizer {kind!r}; expected auto, tiktoken or heuristic")
