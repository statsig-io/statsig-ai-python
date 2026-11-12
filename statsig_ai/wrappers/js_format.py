"""Writes values the way the Node SDK does, so gen_ai events from both SDKs match.

`js_json_stringify` is JavaScript's `JSON.stringify`, `js_string` is `String()`, and
`utf16_length` and `utf16_slice` are a JavaScript string's `length` and `slice()`.
"""

from decimal import Decimal
import json
import math
import re
from typing import Any, Dict, List, Set, Tuple, Union

# Integers up to this size are exact in a JavaScript number.
_MAX_SAFE_INTEGER = 2**53
# Keys a JavaScript object lists first, in ascending order: array indexes.
_MAX_ARRAY_INDEX = 2**32 - 2
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def js_string(value: Any) -> str:
    """`String(value)` for a primitive: `true`, `null`, and numbers as JavaScript prints them."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return js_number_to_string(value)
    return str(value)


def js_number_to_string(value: Union[int, float]) -> str:
    """How JavaScript prints a number: `1` for 1.0, `1e-7`, `1e+21`, `NaN`, `Infinity`."""
    if isinstance(value, int):
        if abs(value) <= _MAX_SAFE_INTEGER:
            return str(value)
        try:
            value = float(value)
        except OverflowError:
            return "Infinity" if value > 0 else "-Infinity"
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value == 0:
        return "0"

    # repr() gives the shortest digits that read back as the same float, as JavaScript
    # does. Without an exponent, it differs from JavaScript only in the trailing ".0".
    text = repr(value)
    if "e" not in text:
        return text[:-2] if text.endswith(".0") else text

    sign = "-" if value < 0 else ""
    parsed = Decimal(text.lstrip("-")).normalize().as_tuple()
    digits = "".join(str(d) for d in parsed.digits)
    k = len(digits)
    n = int(parsed.exponent) + k  # The value is 0.<digits> * 10**n.

    # ECMAScript Number::toString
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * -n + digits
    exponent = n - 1
    mantissa = digits if k == 1 else digits[0] + "." + digits[1:]
    return f"{sign}{mantissa}e{'+' if exponent >= 0 else '-'}{abs(exponent)}"


def js_json_stringify(value: Any) -> str:
    """`JSON.stringify(value)`.

    The output is compact, keeps non-ASCII text as is, prints numbers as JavaScript
    does (`NaN` and `Infinity` as `null`), and lists a dict's integer keys first, as
    a JavaScript object does. Raises ValueError for a circular reference, as
    `JSON.stringify` throws, and TypeError for a value JSON can't hold.
    """
    return _stringify(value, set())


def _stringify(value: Any, parents: Set[int]) -> str:
    if value is None or isinstance(value, bool):
        return js_string(value)
    if isinstance(value, (int, float)):
        number = js_number_to_string(value)
        return "null" if number in ("NaN", "Infinity", "-Infinity") else number
    if isinstance(value, str):
        return _quote(value)
    if not isinstance(value, (list, tuple, dict)):
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    if id(value) in parents:
        raise ValueError("Circular reference detected")
    parents.add(id(value))
    # Loops rather than comprehensions, so each level of nesting uses one stack frame.
    parts: List[str] = []
    try:
        if isinstance(value, dict):
            for key, item in _object_entries(value):
                parts.append(_quote(key) + ":" + _stringify(item, parents))
            return "{" + ",".join(parts) + "}"
        for item in value:
            parts.append(_stringify(item, parents))
        return "[" + ",".join(parts) + "]"
    finally:
        parents.discard(id(value))


def _quote(text: str) -> str:
    # JSON.stringify escapes the same characters as json.dumps, and also lone surrogates.
    quoted = json.dumps(text, ensure_ascii=False)
    return _LONE_SURROGATE.sub(lambda m: f"\\u{ord(m.group()):04x}", quoted)


def _object_entries(obj: Dict[Any, Any]) -> List[Tuple[str, Any]]:
    entries: Dict[str, Any] = {}
    for key, item in obj.items():
        if not isinstance(key, (str, int, float, bool)) and key is not None:
            raise TypeError(f"keys must be str, int, float, bool or None, not {type(key).__name__}")
        # A JavaScript object's keys are strings. Setting a key again keeps its place.
        entries[js_string(key)] = item
    index_keys = sorted((k for k in entries if _is_array_index(k)), key=int)
    other_keys = [k for k in entries if not _is_array_index(k)]
    return [(k, entries[k]) for k in index_keys + other_keys]


def _is_array_index(key: str) -> bool:
    return (
        key.isascii()
        and key.isdigit()
        and (key == "0" or not key.startswith("0"))
        and int(key) <= _MAX_ARRAY_INDEX
    )


def utf16_length(text: str) -> int:
    """A JavaScript string's `length`: characters outside the BMP count as two."""
    return len(text) + sum(1 for ch in text if ord(ch) > 0xFFFF)


def utf16_slice(text: str, max_units: int) -> str:
    """A JavaScript string's `slice(0, max_units)`.

    Where JavaScript would cut a surrogate pair in half, this writes U+FFFD in place of
    the lone high surrogate, which is what encoding Node's string as UTF-8 sends. Python
    can't send a lone surrogate.
    """
    if utf16_length(text) == len(text):
        return text[:max_units]
    units = 0
    for i, ch in enumerate(text):
        width = 2 if ord(ch) > 0xFFFF else 1
        if units + width > max_units:
            return text[:i] + ("\ufffd" if units < max_units else "")
        units += width
    return text
