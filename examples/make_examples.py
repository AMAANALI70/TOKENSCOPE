"""Generate small, synthetic sample traces. Run: python examples/make_examples.py"""

import json
from pathlib import Path

HERE = Path(__file__).parent

BIG_LOG = "\n".join(
    f"2026-01-01 12:00:{i:02d} INFO worker-{i % 3} processed batch {i} ok" for i in range(60)
)
FILE_BODY = "def add(a, b):\n    return a + b\n" * 40

TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_logs",
            "description": "Fetch recent service logs.",
            "parameters": {
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email. Rarely needed.",
            "parameters": {
                "type": "object",
                "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
            },
        },
    },
]


def openai_trace() -> list[dict]:
    system = {"role": "system", "content": "You are a careful debugging agent. Be concise."}
    user = {"role": "user", "content": "Why is the worker slow? Check logs and utils.py."}
    a1 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "get_logs", "arguments": '{"service":"worker"}'},
            }
        ],
    }
    t1 = {"role": "tool", "tool_call_id": "c1", "content": BIG_LOG}
    a2 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c2",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"utils.py"}'},
            }
        ],
    }
    t2 = {"role": "tool", "tool_call_id": "c2", "content": FILE_BODY}
    a3 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c3",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"utils.py"}'},
            }
        ],
    }
    t3 = {"role": "tool", "tool_call_id": "c3", "content": FILE_BODY}  # same file read twice: waste

    histories = [
        [system, user],
        [system, user, a1, t1],
        [system, user, a1, t1, a2, t2],
        [system, user, a1, t1, a2, t2, a3, t3],
    ]
    calls = []
    for msgs in histories:
        calls.append({"model": "gpt-4o", "tools": TOOLS_OPENAI, "messages": msgs})
    return calls


if __name__ == "__main__":
    path = HERE / "sample_openai_trace.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in openai_trace()) + "\n", encoding="utf-8")
    print(f"wrote {path}")
