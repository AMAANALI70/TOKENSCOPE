"""Load a trace file (JSONL, JSON array, or trace envelope) into the neutral model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tokenscope.models import Trace
from tokenscope.parsers import PARSERS, TraceParseError, detect_format


def _read_payloads(path: Path) -> list[Any]:
    raw = path.read_text(encoding="utf-8")
    if raw.lstrip().startswith("["):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TraceParseError(f"{path}: invalid JSON: {exc}") from exc
        return list(data)

    payloads: list[Any] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payloads.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise TraceParseError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return payloads


def load_trace(path: str | Path, fmt: str = "auto") -> Trace:
    path_obj = Path(path)
    raw = path_obj.read_text(encoding="utf-8")
    trace_id: str | None = None
    execution_id: str | None = None
    metadata: dict[str, Any] = {}

    stripped = raw.lstrip()
    payloads: list[Any] = []
    if stripped.startswith("{"):
        try:
            full_json = json.loads(raw)
            if isinstance(full_json, dict) and "calls" in full_json:
                trace_id = full_json.get("trace_id") or full_json.get("id")
                execution_id = full_json.get("execution_id")
                metadata = full_json.get("metadata", {})
                payloads = full_json["calls"]
            else:
                payloads = _read_payloads(path_obj)
        except json.JSONDecodeError:
            payloads = _read_payloads(path_obj)
    else:
        payloads = _read_payloads(path_obj)

    if not payloads:
        raise TraceParseError(f"{path}: trace is empty")
    if fmt == "auto":
        fmt = detect_format(payloads[0])
    if fmt not in PARSERS:
        raise TraceParseError(f"unknown format {fmt!r}; expected one of {sorted(PARSERS)}")

    parse = PARSERS[fmt]
    calls = [parse(p, i) for i, p in enumerate(payloads)]

    if not trace_id and isinstance(payloads[0], dict):
        trace_id = payloads[0].get("trace_id") or payloads[0].get("run_id")
    if not execution_id and isinstance(payloads[0], dict):
        execution_id = payloads[0].get("execution_id") or payloads[0].get("session_id")

    return Trace(
        calls=calls,
        source_format=fmt,
        trace_id=trace_id,
        execution_id=execution_id,
        metadata=metadata,
    )
