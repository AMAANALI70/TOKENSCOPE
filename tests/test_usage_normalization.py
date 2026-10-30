from tokenscope.loader import load_trace
from tokenscope.models import Usage, UsageProvenance


def test_openai_usage_normalization_with_cache(write_jsonl):
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "response": {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 40},
            },
        },
    }
    trace = load_trace(write_jsonl([payload]), "openai")
    call = trace.calls[0]
    usage = call.usage
    assert usage is not None
    assert usage.input_tokens == 100
    assert usage.total_input_tokens == 100
    assert usage.output_tokens == 15
    assert usage.cached_input_tokens == 40
    assert usage.cache_read_input_tokens == 40
    assert usage.cache_creation_input_tokens == 0
    assert usage.uncached_input_tokens == 60
    assert usage.provenance == UsageProvenance.PROVIDER


def test_openai_usage_normalization_without_cache(write_jsonl):
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "response": {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {
                "prompt_tokens": 80,
                "completion_tokens": 20,
            },
        },
    }
    trace = load_trace(write_jsonl([payload]), "openai")
    call = trace.calls[0]
    usage = call.usage
    assert usage is not None
    assert usage.input_tokens == 80
    assert usage.total_input_tokens == 80
    assert usage.output_tokens == 20
    assert usage.cached_input_tokens == 0
    assert usage.cache_read_input_tokens == 0
    assert usage.cache_creation_input_tokens == 0
    assert usage.uncached_input_tokens == 80
    assert usage.provenance == UsageProvenance.PROVIDER


def test_anthropic_usage_normalization_with_cache(write_jsonl):
    payload = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "hello"}],
        "response": {
            "content": [{"type": "text", "text": "world"}],
            "usage": {
                "input_tokens": 50,
                "cache_read_input_tokens": 200,
                "cache_creation_input_tokens": 30,
                "output_tokens": 25,
            },
        },
    }
    trace = load_trace(write_jsonl([payload]), "anthropic")
    call = trace.calls[0]
    usage = call.usage
    assert usage is not None
    # Anthropic reports uncached as input_tokens; total prompt is sum of all three
    assert usage.input_tokens == 280
    assert usage.total_input_tokens == 280
    assert usage.uncached_input_tokens == 50
    assert usage.cache_read_input_tokens == 200
    assert usage.cached_input_tokens == 200
    assert usage.cache_creation_input_tokens == 30
    assert usage.output_tokens == 25
    assert usage.provenance == UsageProvenance.PROVIDER


def test_anthropic_usage_normalization_without_cache(write_jsonl):
    payload = {
        "model": "claude-3-5-haiku",
        "messages": [{"role": "user", "content": "hello"}],
        "response": {
            "content": [{"type": "text", "text": "world"}],
            "usage": {
                "input_tokens": 45,
                "output_tokens": 10,
            },
        },
    }
    trace = load_trace(write_jsonl([payload]), "anthropic")
    call = trace.calls[0]
    usage = call.usage
    assert usage is not None
    assert usage.input_tokens == 45
    assert usage.total_input_tokens == 45
    assert usage.uncached_input_tokens == 45
    assert usage.cache_read_input_tokens == 0
    assert usage.cache_creation_input_tokens == 0
    assert usage.output_tokens == 10
    assert usage.provenance == UsageProvenance.PROVIDER


def test_usage_dataclass_defaults():
    # Direct instantiation with only required arguments
    u = Usage(input_tokens=100, output_tokens=20)
    assert u.total_input_tokens == 100
    assert u.uncached_input_tokens == 100
    assert u.cache_read_input_tokens == 0
    assert u.cached_input_tokens == 0
    assert u.cache_creation_input_tokens == 0
    assert u.provenance == UsageProvenance.PROVIDER

    # Direct instantiation with cached_input_tokens alias
    u2 = Usage(input_tokens=100, output_tokens=20, cached_input_tokens=30)
    assert u2.cache_read_input_tokens == 30
    assert u2.uncached_input_tokens == 70

    # Custom provenance
    u3 = Usage(input_tokens=50, output_tokens=10, provenance=UsageProvenance.ESTIMATED)
    assert u3.provenance == UsageProvenance.ESTIMATED
