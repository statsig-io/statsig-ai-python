"""Converts Anthropic Messages API requests and responses to the OpenTelemetry GenAI message
format, and rebuilds a streamed message from its events.

Ported from the Node SDK's anthropic.ts.
"""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .gen_ai_messages import (
    GenAIBlobPart,
    GenAIInputMessage,
    GenAIMessagePart,
    GenAIOutputMessage,
    GenAIReasoningPart,
    GenAIToolCallPart,
    GenAIToolCallResponsePart,
    GenAIUriPart,
    parse_tool_arguments,
    text_part,
)

_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_call",
    "refusal": "content_filter",
}


def to_anthropic_input_messages(messages: Any) -> List[GenAIInputMessage]:
    """Converts the `messages` request parameter. A user message that only holds tool results
    becomes a `tool` message."""
    result: List[GenAIInputMessage] = []
    for item in _list(messages):
        message = as_dict(item)
        parts = to_anthropic_parts(message.get("content"))
        if parts and all(part["type"] == "tool_call_response" for part in parts):
            result.append({"role": "tool", "parts": parts})
        elif message.get("role") == "assistant":
            result.append({"role": "assistant", "parts": parts})
        else:
            result.append({"role": "user", "parts": parts})
    return result


def to_anthropic_output_messages(message: Any) -> List[GenAIOutputMessage]:
    fields = as_dict(message)
    return [
        {
            "role": "assistant",
            "parts": to_anthropic_parts(fields.get("content")),
            "finish_reason": to_anthropic_finish_reason(fields.get("stop_reason")),
        }
    ]


def to_anthropic_parts(content: Any) -> List[GenAIMessagePart]:
    """Converts message content, the `system` parameter or a tool result's content: a string or
    a list of content blocks."""
    if isinstance(content, str):
        return [text_part(content)]
    parts: List[GenAIMessagePart] = []
    for item in _list(content):
        block = as_dict(item)
        block_type = block.get("type")
        if block_type == "text":
            parts.append(text_part(block.get("text")))
        elif block_type == "thinking":
            reasoning: GenAIReasoningPart = {"type": "reasoning", "content": block.get("thinking")}
            parts.append(reasoning)
        elif block_type in ("tool_use", "server_tool_use"):
            parts.append(_tool_call_part(block))
        elif block_type == "tool_result":
            parts.append(_tool_call_response_part(block))
        elif block_type in ("image", "document"):
            parts.append(_media_part(block_type, as_dict(block.get("source"))))
    return parts


def to_anthropic_finish_reason(stop_reason: Any) -> str:
    if stop_reason is None:
        return "stop"
    return _FINISH_REASONS.get(stop_reason, stop_reason)


def aggregate_anthropic_stream_events(events: Iterable[Any]) -> Dict[str, Any]:
    """Rebuilds the final message from the events of a streamed call, so it records the same
    output, tool calls and usage as a call that isn't streamed."""
    message: Dict[str, Any] = {"usage": {}}
    blocks: Dict[Any, Dict[str, Any]] = {}
    tool_input_json: Dict[Any, str] = {}

    for item in events:
        event = as_dict(item)
        event_type = event.get("type")
        if event_type == "message_start":
            start = as_dict(event.get("message"))
            message["id"] = start.get("id")
            message["model"] = start.get("model")
            message["usage"] = _given_values(start.get("usage"))
        elif event_type == "content_block_start":
            blocks[event.get("index")] = dict(as_dict(event.get("content_block")))
        elif event_type == "content_block_delta":
            _apply_delta(blocks, tool_input_json, event)
        elif event_type == "message_delta":
            stop_reason = as_dict(event.get("delta")).get("stop_reason")
            if stop_reason is not None:
                message["stop_reason"] = stop_reason
            # Later events only update the token counts they carry.
            message["usage"].update(_given_values(event.get("usage")))

    for index, arguments in tool_input_json.items():
        # A tool call without arguments streams no JSON, so it keeps its starting `{}`.
        if index in blocks and arguments:
            blocks[index]["input"] = parse_tool_arguments(arguments)
    message["content"] = [blocks[index] for index in sorted(blocks, key=_index_order)]
    return message


def _apply_delta(
    blocks: Dict[Any, Dict[str, Any]], tool_input_json: Dict[Any, str], event: Mapping[str, Any]
) -> None:
    index = event.get("index")
    block = blocks.get(index)
    if block is None:
        return
    delta = as_dict(event.get("delta"))
    delta_type = delta.get("type")
    if delta_type == "text_delta":
        block["text"] = (block.get("text") or "") + (delta.get("text") or "")
    elif delta_type == "thinking_delta":
        block["thinking"] = (block.get("thinking") or "") + (delta.get("thinking") or "")
    elif delta_type == "input_json_delta":
        tool_input_json[index] = tool_input_json.get(index, "") + (delta.get("partial_json") or "")


def _tool_call_part(block: Any) -> GenAIToolCallPart:
    part: GenAIToolCallPart = {
        "type": "tool_call",
        "id": block.get("id"),
        "name": block.get("name"),
        "arguments": block.get("input"),
    }
    # The Node SDK leaves out missing fields, as JSON.stringify drops undefined values.
    if part["id"] is None:
        del part["id"]
    if part["arguments"] is None:
        del part["arguments"]
    return part


def _tool_call_response_part(block: Any) -> GenAIToolCallResponsePart:
    content = block.get("content")
    if isinstance(content, str):
        response = content
    else:
        response = "".join(_part_text(part) for part in to_anthropic_parts(content))
    part: GenAIToolCallResponsePart = {
        "type": "tool_call_response",
        "id": block.get("tool_use_id"),
        "response": response,
    }
    if part["id"] is None:
        del part["id"]
    return part


def _media_part(modality: str, source: Any) -> GenAIMessagePart:
    if source.get("type") == "url":
        uri: GenAIUriPart = {"type": "uri", "modality": modality, "uri": source.get("url")}
        return uri
    blob: GenAIBlobPart = {"type": "blob", "modality": modality}
    media_type = source.get("media_type")
    if media_type is not None:
        blob["mime_type"] = media_type
    return blob


def _part_text(part: GenAIMessagePart) -> str:
    content = part.get("content")
    return content if isinstance(content, str) else ""


def _given_values(value: Any) -> Dict[str, Any]:
    return {key: item for key, item in as_dict(value).items() if item is not None}


def _index_order(index: Any) -> Any:
    return (0, index) if isinstance(index, int) else (1, str(index))


def _list(value: Any) -> Sequence[Any]:
    # Only lists and tuples: reading another iterable here would use it up before the request.
    return value if isinstance(value, (list, tuple)) else ()


def as_dict(value: Any) -> Any:
    """Reads a dict or an Anthropic SDK object as a dict. Anything else reads as an empty dict,
    the way optional chaining reads it in the Node SDK.

    Typed Any, like the Node SDK's Record<string, any>: Anthropic payload fields are untyped.
    """
    if isinstance(value, Mapping):
        return value
    to_dict: Optional[Any] = getattr(value, "to_dict", None) or getattr(value, "model_dump", None)
    if callable(to_dict):
        try:
            fields = to_dict()
        except Exception:
            return {}
        if isinstance(fields, Mapping):
            return fields
    return {}
