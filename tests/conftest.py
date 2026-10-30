import json
from pathlib import Path

import pytest


@pytest.fixture
def write_jsonl(tmp_path: Path):
    def _write(payloads: list[dict], name: str = "trace.jsonl") -> Path:
        p = tmp_path / name
        p.write_text("\n".join(json.dumps(x) for x in payloads) + "\n", encoding="utf-8")
        return p

    return _write
