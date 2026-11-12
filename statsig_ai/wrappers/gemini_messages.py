"""Converts google-genai requests and responses to the OpenTelemetry GenAI message format.

Ported from the Node SDK's gemini.ts. google-genai is optional, so nothing here imports it:
fields are read by attribute from SDK objects, or by key from dicts.
"""

from enum import Enum
from typing import Any, List, Mapping, Optional, Sequence

from .gen_ai_messages import (
    GenAIBlobPart,
    GenAIInputMessage,
    GenAIMessagePart,
    GenAIOutputMessage,
    GenAIToolCallPart,
    GenAIToolCallResponsePart,
    GenAIUriPart,
    text_part,
)

_CONTENT_FILTER_REASONS = ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII")


def to_gemini_input_messages(contents: Any) -> List[GenAIInputMessage]:
    """Converts `contents`: a string, a part, a Content, or a list of them.

    Each Content is a message. Strings and parts outside a Content are sent as a user message,
    and a list inside the list is a user message of its own, as the SDK sends them.
    """
    items = contents if isinstance(contents, (list, tuple)) else [contents]
    messages: List[GenAIInputMessage] = []
    loose_parts: List[Any] = []
    for item in items:
        if not _is_content(item) and not isinstance(item, (list, tuple)):
            loose_parts.append(item)
            continue
        if loose_parts:
            messages.append(_to_input_message("user", loose_parts))
            loose_parts = []
        if isinstance(item, (list, tuple)):
            messages.append(_to_input_message("user", item))
        else:
            messages.append(
                _to_input_message(get_field(item, "role"), as_list(get_field(item, "parts")))
            )
    if loose_parts or not messages:
        messages.append(_to_input_message("user", loose_parts))
    return messages


def to_gemini_output_messages(candidates: Sequence[Any]) -> List[GenAIOutputMessage]:
    """Converts each response candidate to an assistant message."""
    messages: List[GenAIOutputMessage] = []
    for candidate in candidates:
        parts = to_gemini_parts(as_list(get_field(get_field(candidate, "content"), "parts")))
        messages.append(
            {
                "role": "assistant",
                "parts": parts,
                "finish_reason": to_gemini_finish_reason(
                    get_field(candidate, "finish_reason"), parts
                ),
            }
        )
    return messages


def to_gemini_parts(parts: Sequence[Any]) -> List[GenAIMessagePart]:
    """Converts text, thought, function call, function response, inline data and file parts.
    Other parts, such as executable code, are left out.
    """
    converted: List[GenAIMessagePart] = []
    for part in parts:
        message_part = _to_part(part)
        if message_part is not None:
            converted.append(message_part)
    return converted


def to_gemini_finish_reason(finish_reason: Any, parts: Sequence[GenAIMessagePart]) -> str:
    if any(part["type"] == "tool_call" for part in parts):
        return "tool_call"
    reason = enum_value(finish_reason)
    if reason is None or reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    if reason in _CONTENT_FILTER_REASONS:
        return "content_filter"
    return str(reason).lower()


def get_field(value: Any, name: str) -> Any:
    """Reads a field of a google-genai object, or of a dict, which the SDK accepts with its
    snake_case field names or the API's camelCase ones. Returns None if it isn't there.
    """
    if isinstance(value, Mapping):
        found = value.get(name)
        return found if found is not None else value.get(_camel_case(name))
    return getattr(value, name, None)


def enum_value(value: Any) -> Any:
    """The SDK's enums, such as FinishReason, are str enums. Reads their plain value."""
    return value.value if isinstance(value, Enum) else value


def _to_input_message(role: Any, parts: Sequence[Any]) -> GenAIInputMessage:
    converted = to_gemini_parts(parts)
    if converted and all(part["type"] == "tool_call_response" for part in converted):
        return {"role": "tool", "parts": converted}
    return {"role": "assistant" if role == "model" else "user", "parts": converted}


def _to_part(part: Any) -> Optional[GenAIMessagePart]:
    if isinstance(part, str):
        return text_part(part)

    text = get_field(part, "text")
    if isinstance(text, str):
        if get_field(part, "thought"):
            return {"type": "reasoning", "content": text}
        return text_part(text)

    function_call = get_field(part, "function_call")
    if function_call is not None:
        name = get_field(function_call, "name")
        tool_call: GenAIToolCallPart = {
            "type": "tool_call",
            "id": _first_not_none(get_field(function_call, "id"), name),
            "name": name,
            "arguments": get_field(function_call, "args"),
        }
        return _without_unset(tool_call, "id", "arguments")

    function_response = get_field(part, "function_response")
    if function_response is not None:
        tool_response: GenAIToolCallResponsePart = {
            "type": "tool_call_response",
            "id": _first_not_none(
                get_field(function_response, "id"), get_field(function_response, "name")
            ),
            "response": get_field(function_response, "response"),
        }
        return _without_unset(tool_response, "id")

    inline_data = get_field(part, "inline_data")
    if inline_data is not None:
        mime_type = get_field(inline_data, "mime_type")
        blob: GenAIBlobPart = {
            "type": "blob",
            "modality": _to_modality(mime_type),
            "mime_type": mime_type,
        }
        return _without_unset(blob, "mime_type")

    file_data = get_field(part, "file_data")
    if file_data is not None:
        mime_type = get_field(file_data, "mime_type")
        uri: GenAIUriPart = {
            "type": "uri",
            "modality": _to_modality(mime_type),
            "uri": get_field(file_data, "file_uri"),
            "mime_type": mime_type,
        }
        return _without_unset(uri, "mime_type")
    return None


def _is_content(item: Any) -> bool:
    if isinstance(item, Mapping):
        return "parts" in item
    return not isinstance(item, (str, list, tuple)) and hasattr(item, "parts")


def _to_modality(mime_type: Any) -> str:
    return mime_type.split("/")[0] if isinstance(mime_type, str) else "unknown"


def _without_unset(part: Any, *keys: str) -> Any:
    # Like JSON.stringify in the Node SDK, leaves out optional fields that aren't set.
    for key in keys:
        if part[key] is None:
            del part[key]
    return part


def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def as_list(value: Any) -> Sequence[Any]:
    return value if isinstance(value, (list, tuple)) else ()


def _camel_case(name: str) -> str:
    first, *rest = name.split("_")
    return first + "".join(word.capitalize() for word in rest)
