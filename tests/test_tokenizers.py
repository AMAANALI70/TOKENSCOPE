import pytest

from tokenscope.tokenizers import HeuristicCounter, get_counter


def test_heuristic_empty_is_zero():
    assert HeuristicCounter().count("") == 0


def test_heuristic_nonempty_at_least_one():
    assert HeuristicCounter().count("a") == 1


def test_heuristic_roughly_four_chars_per_token():
    assert HeuristicCounter().count("x" * 400) == 100


def test_get_counter_heuristic():
    assert get_counter("heuristic").name.startswith("heuristic")


def test_get_counter_auto_never_raises():
    assert get_counter("auto").count("hello world") > 0


def test_get_counter_unknown_kind():
    with pytest.raises(ValueError):
        get_counter("nope")
