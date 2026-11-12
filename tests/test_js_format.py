import math
from typing import Any, List

import pytest

from statsig_ai.wrappers.js_format import (
    js_json_stringify,
    js_number_to_string,
    js_string,
    utf16_length,
    utf16_slice,
)

# Expected values are what Node prints for the same input.


@pytest.mark.parametrize(
    "value, expected",
    [
        (1.0, "1"),
        (0.5, "0.5"),
        (0.1, "0.1"),
        (-2.5, "-2.5"),
        (0.30000000000000004, "0.30000000000000004"),
        (0.000001, "0.000001"),
        (1e-7, "1e-7"),
        (1.5e-7, "1.5e-7"),
        (1e20, "100000000000000000000"),
        (1e21, "1e+21"),
        (123456789.125, "123456789.125"),
        (-0.0, "0"),
        (math.nan, "NaN"),
        (math.inf, "Infinity"),
        (-math.inf, "-Infinity"),
        (42, "42"),
        (2**53 + 1, "9007199254740992"),
        (10**400, "Infinity"),
    ],
)
def test_numbers_print_as_in_javascript(value, expected):
    assert js_number_to_string(value) == expected


def test_primitives_print_as_javascript_string_does():
    assert js_string(True) == "true"
    assert js_string(False) == "false"
    assert js_string(None) == "null"
    assert js_string(1.0) == "1"
    assert js_string("text") == "text"


def test_json_is_compact_and_keeps_non_ascii_text():
    value = {"a": [1.0, math.nan, math.inf, -0.0, 1e21], "s": "café 日本語 😀", "n": None, "t": True}

    assert js_json_stringify(value) == (
        '{"a":[1,null,null,0,1e+21],"s":"café 日本語 😀","n":null,"t":true}'
    )


def test_nested_json_is_compact():
    messages = [
        {"role": "user", "content": "Héllo 👋🏽"},
        {"role": "assistant", "content": [{"type": "text", "text": "日本語"}]},
    ]

    assert js_json_stringify(messages) == (
        '[{"role":"user","content":"Héllo 👋🏽"},'
        '{"role":"assistant","content":[{"type":"text","text":"日本語"}]}]'
    )


def test_json_escapes_the_characters_javascript_escapes():
    assert (
        js_json_stringify(["\x00\x1f\x7f", "tab\t", 'q"\\', "/", "a\ud800b"])
        == '["\\u0000\\u001f\x7f","tab\\t","q\\"\\\\","/","a\\ud800b"]'
    )


def test_json_lists_integer_keys_first_as_a_javascript_object_does():
    value = {"b": 1, "10": 2, "2": 3, "01": 4, "a": 5}

    assert js_json_stringify(value) == '{"2":3,"10":2,"b":1,"01":4,"a":5}'


def test_json_writes_tuples_as_arrays_and_non_string_keys_as_strings():
    assert js_json_stringify((1, "a")) == '[1,"a"]'
    assert js_json_stringify({True: 1, None: 2, 1.5: 3, "k": 4}) == (
        '{"true":1,"null":2,"1.5":3,"k":4}'
    )


def test_json_raises_for_a_circular_reference():
    value: List[Any] = []
    value.append(value)

    with pytest.raises(ValueError):
        js_json_stringify(value)


def test_json_raises_for_values_json_cannot_hold():
    with pytest.raises(TypeError):
        js_json_stringify({"when": object()})


def test_shared_values_are_not_circular():
    shared = {"x": 1}

    assert js_json_stringify([shared, shared]) == '[{"x":1},{"x":1}]'


def test_utf16_length_counts_characters_outside_the_bmp_twice():
    assert utf16_length("café") == 4
    assert utf16_length("日本語") == 3
    assert utf16_length("a😀b") == 4


def test_utf16_slice_cuts_like_javascript():
    assert utf16_slice("abcdef", 3) == "abc"
    assert utf16_slice("a😀b", 3) == "a😀"
    assert utf16_slice("a😀b", 1) == "a"
    # JavaScript keeps half of the emoji's surrogate pair, which UTF-8 encodes as U+FFFD.
    assert utf16_slice("a😀b", 2) == "a\ufffd"
