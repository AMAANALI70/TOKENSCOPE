from tokenscope.attribution import attribute
from tokenscope.loader import load_trace
from tokenscope.models import Call, Category, Segment, Trace, Usage
from tokenscope.tokenizers import HeuristicCounter

C = HeuristicCounter()


def _seg(cat, n_chars, tool=None, ch="x"):
    return Segment(cat, ch * n_chars, tool_name=tool)


def test_category_totals_equal_estimated_input():
    call = Call(0, "m", [_seg(Category.SYSTEM, 40), _seg(Category.USER, 80)])
    attr = attribute(Trace([call], "openai"), C)
    assert attr.calls[0].estimated_input == 10 + 20
    assert attr.billed_by_category[Category.SYSTEM] == 10
    assert attr.billed_by_category[Category.USER] == 20


def test_resent_history_is_billed_every_time_but_unique_once():
    sys_seg = _seg(Category.SYSTEM, 400)  # 100 tokens
    result = _seg(Category.TOOL_RESULT, 800, "t", ch="r")  # 200 tokens
    new = _seg(Category.USER, 40, ch="n")  # 10 tokens
    calls = [
        Call(0, "m", [sys_seg]),
        Call(1, "m", [sys_seg, result]),
        Call(2, "m", [sys_seg, result, new]),
    ]
    attr = attribute(Trace(calls, "openai"), C)
    assert attr.billed_input_total == 100 + 300 + 310
    assert attr.unique_input_total == 100 + 200 + 10
    assert abs(attr.resend_multiplier - 710 / 310) < 1e-9


def test_duplicate_within_one_call_counts_unique_once():
    same = _seg(Category.TOOL_RESULT, 400, "read_file")
    attr = attribute(Trace([Call(0, "m", [same, same])], "openai"), C)
    assert attr.billed_input_total == 200
    assert attr.unique_input_total == 100


def test_per_tool_attribution_sums_across_calls():
    schema = _seg(Category.TOOL_SCHEMA, 400, "read_file", ch="s")  # 100
    calls = [Call(0, "m", [schema]), Call(1, "m", [schema])]
    attr = attribute(Trace(calls, "openai"), C)
    assert attr.by_tool[(Category.TOOL_SCHEMA, "read_file")] == 200


def test_calibration_factor_uses_only_calls_with_usage():
    seg = _seg(Category.USER, 400)  # est 100
    calls = [
        Call(0, "m", [seg], usage=Usage(input_tokens=120, output_tokens=5)),
        Call(1, "m", [seg]),  # no usage: must be ignored
    ]
    attr = attribute(Trace(calls, "openai"), C)
    assert abs(attr.calibration_factor - 1.2) < 1e-9


def test_no_usage_means_no_calibration():
    attr = attribute(Trace([Call(0, "m", [_seg(Category.USER, 40)])], "openai"), C)
    assert attr.calibration_factor is None


def test_output_tokens_estimated():
    call = Call(
        0, "m", [_seg(Category.USER, 4)], output_segments=[Segment(Category.OUTPUT, "y" * 40)]
    )
    assert attribute(Trace([call], "openai"), C).output_total == 10


def test_sample_trace_shows_resend_effect():
    trace = load_trace("examples/sample_openai_trace.jsonl")
    attr = attribute(trace, C)
    assert len(attr.calls) == 4
    assert attr.resend_multiplier > 2  # history + schemas resent each call
    # the file was read twice, so its result text is only unique once
    assert (
        attr.billed_by_category[Category.TOOL_RESULT]
        > attr.unique_by_category[Category.TOOL_RESULT]
    )
