"""Helpers shared by all provider parsers."""

from __future__ import annotations

import json
from typing import Any


class TraceParseError(ValueError):
    """The trace is malformed or in a format we don't understand."""


def canonical_json(value: Any) -> str:
    """Stable, compact JSON so identical objects produce identical text."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def text_of(content: Any) -> str:
    """Flatten message content (str | list of parts | None) into plain text.

    Non-text parts (images, audio, files) are ignored in v0.1. That undercounts
    multimodal traces; the drift metric will reveal it.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and part.get("type") in (
                    None,
                    "text",
                    "input_text",
                    "output_text",
                ):
                    parts.append(text)
        return "\n".join(parts)
    return canonical_json(content)
