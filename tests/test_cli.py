import json

import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner  # noqa: E402

from tokenscope.cli import app  # noqa: E402

runner = CliRunner()


def test_audit_json_output():
    result = runner.invoke(
        app, ["audit", "examples/sample_openai_trace.jsonl", "--json", "--tokenizer", "heuristic"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["calls"] == 4
    assert data["billed_input_tokens"] > data["unique_input_tokens"]


def test_audit_table_output_runs():
    result = runner.invoke(
        app, ["audit", "examples/sample_openai_trace.jsonl", "--tokenizer", "heuristic"]
    )
    assert result.exit_code == 0, result.output
    assert "Resend multiplier" in result.output


def test_bad_format_exits_2():
    result = runner.invoke(app, ["audit", "examples/sample_openai_trace.jsonl", "--format", "nope"])
    assert result.exit_code == 2


def test_audit_with_model_pricing():
    result = runner.invoke(
        app,
        [
            "audit",
            "examples/sample_openai_trace.jsonl",
            "--json",
            "--model",
            "gpt-4o",
            "--tokenizer",
            "heuristic",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "cost" in data
    assert data["cost"]["status"] in ("reported", "estimated")
    assert data["cost"]["total_usd"] > 0


def test_audit_with_custom_pricing_options():
    result = runner.invoke(
        app,
        [
            "audit",
            "examples/sample_openai_trace.jsonl",
            "--json",
            "--price-per-mtok",
            "3.0",
            "--cache-read-price-per-mtok",
            "1.5",
            "--usd-inr",
            "84.0",
            "--tokenizer",
            "heuristic",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "cost" in data
    assert data["cost"]["total_usd"] > 0
    assert "total_inr" in data["cost"]
    assert abs(data["cost"]["total_inr"] - data["cost"]["total_usd"] * 84.0) < 1e-4
