"""Converts OpenAI requests and responses to the OpenTelemetry GenAI message format.

Ported from the Node SDK's openai-messages.ts.
"""

from typing import Any, List, Mapping, NamedTuple, Optional, Sequence

from .gen_ai_messages import (
    GenAIBlobPart,
    GenAIInputMessage,
    GenAIMessagePart,
    GenAIOutputMessage,
    GenAIReasoningPart,
    GenAIToolCallPart,
    GenAIToolCallResponsePart,
    parse_tool_arguments,
    text_part,
)

_SYSTEM_ROLES = ("system", "developer")


class OpenAIRequestMessages(NamedTuple):
    input_messages: List[GenAIInputMessage]
    system_instructions: List[GenAIMessagePart]


def to_openai_request_messages(params: Mapping[str, Any]) -> OpenAIRequestMessages:
    """Converts a Chat Completions, Responses, completions or embeddings request.

    System and developer messages (and Responses `instructions`) become system instructions
    instead of input messages.
    """
    instructions = params.get("instructions")
    system_instructions: List[GenAIMessagePart] = (
        [text_part(instructions)] if isinstance(instructions, str) else []
    )
    input_messages: List[GenAIInputMessage] = []

    items = _first_not_none(params, "messages", "input", "prompt")
    for item in items if isinstance(items, (list, tuple)) else [items]:
        if isinstance(item, str):
            input_messages.append({"role": "user", "parts": [text_part(item)]})
            continue
        fields = _as_dict(item)
        if fields.get("role") in _SYSTEM_ROLES:
            system_instructions.extend(_to_content_parts(fields.get("content")))
            continue
        message = _to_input_message(fields)
        if message is not None:
            input_messages.append(message)
    return OpenAIRequestMessages(input_messages, system_instructions)


def _to_input_message(item: Mapping[str, Any]) -> Optional[GenAIInputMessage]:
    item_type = item.get("type")
    if item_type == "function_call":
        return {"role": "assistant", "parts": [_to_function_call_part(item)]}
    if item_type == "function_call_output":
        return {
            "role": "tool",
            "parts": [_tool_call_response_part(item.get("call_id"), item.get("output"))],
        }
    if item_type == "reasoning":
        return {"role": "assistant", "parts": _to_reasoning_parts(item)}

    role = item.get("role")
    if role in ("tool", "function"):
        tool_call_id = item.get("tool_call_id")
        call_id = tool_call_id if tool_call_id is not None else item.get("name")
        return {"role": "tool", "parts": [_tool_call_response_part(call_id, item.get("content"))]}
    if role == "assistant":
        parts = _to_content_parts(item.get("content"))
        parts.extend(_to_chat_tool_call_part(call) for call in _list(item.get("tool_calls")))
        return {"role": "assistant", "parts": parts}
    if role == "user":
        return {"role": "user", "parts": _to_content_parts(item.get("content"))}
    return None


def to_openai_output_messages(response: Any) -> Optional[List[GenAIOutputMessage]]:
    """Converts Chat Completions `choices` (including text completions) or a Responses `output`
    list. Returns None for responses without messages, such as embeddings.
    """
    fields = _as_dict(response)
    choices = fields.get("choices")
    if isinstance(choices, (list, tuple)):
        return [_to_choice_message(_as_dict(choice)) for choice in choices]

    output = fields.get("output")
    if isinstance(output, (list, tuple)):
        parts: List[GenAIMessagePart] = []
        for item in output:
            parts.extend(_to_output_item_parts(_as_dict(item)))
        return [
            {
                "role": "assistant",
                "parts": parts,
                "finish_reason": _to_responses_finish_reason(fields, parts),
            }
        ]
    return None


def _to_choice_message(choice: Mapping[str, Any]) -> GenAIOutputMessage:
    message = _as_dict(choice.get("message"))
    parts: List[GenAIMessagePart] = []
    text = choice.get("text")
    if isinstance(text, str):
        parts.append(text_part(text))
    parts.extend(_to_content_parts(message.get("content")))
    refusal = message.get("refusal")
    if refusal:
        parts.append(text_part(refusal))
    parts.extend(_to_chat_tool_call_part(call) for call in _list(message.get("tool_calls")))
    return {
        "role": "assistant",
        "parts": parts,
        "finish_reason": _to_finish_reason(choice.get("finish_reason")),
    }


def _to_output_item_parts(item: Mapping[str, Any]) -> List[GenAIMessagePart]:
    item_type = item.get("type")
    if item_type == "message":
        return _to_content_parts(item.get("content"))
    if item_type == "function_call":
        return [_to_function_call_part(item)]
    if item_type == "reasoning":
        return _to_reasoning_parts(item)
    return []


def _to_content_parts(content: Any) -> List[GenAIMessagePart]:
    if isinstance(content, str):
        return [text_part(content)]
    parts: List[GenAIMessagePart] = []
    for item in _list(content):
        part = _as_dict(item)
        part_type = part.get("type")
        if part_type in ("text", "input_text", "output_text"):
            parts.append(text_part(part.get("text")))
        elif part_type == "refusal":
            parts.append(text_part(part.get("refusal")))
        elif part_type == "image_url":
            image_url = _as_dict(part.get("image_url")).get("url")
            parts.append({"type": "uri", "modality": "image", "uri": image_url})
        elif part_type == "input_image":
            image_url = part.get("image_url")
            if image_url:
                parts.append({"type": "uri", "modality": "image", "uri": image_url})
            else:
                parts.append({"type": "blob", "modality": "image"})
        elif part_type == "input_audio":
            audio: GenAIBlobPart = {"type": "blob", "modality": "audio"}
            audio_format = _as_dict(part.get("input_audio")).get("format")
            if audio_format is not None:
                audio["mime_type"] = audio_format
            parts.append(audio)
        elif part_type in ("file", "input_file"):
            parts.append({"type": "blob", "modality": "file"})
    return parts


def _to_chat_tool_call_part(tool_call: Any) -> GenAIToolCallPart:
    call = _as_dict(tool_call)
    function = _as_dict(call.get("function"))
    return _tool_call_part(call.get("id"), function.get("name"), function.get("arguments"))


def _to_function_call_part(item: Mapping[str, Any]) -> GenAIToolCallPart:
    return _tool_call_part(item.get("call_id"), item.get("name"), item.get("arguments"))


def _tool_call_part(call_id: Any, name: Any, arguments: Any) -> GenAIToolCallPart:
    part: GenAIToolCallPart = {
        "type": "tool_call",
        "id": call_id,
        "name": name,
        "arguments": parse_tool_arguments(arguments if arguments is not None else ""),
    }
    if call_id is None:
        del part["id"]
    return part


def _tool_call_response_part(call_id: Any, response: Any) -> GenAIToolCallResponsePart:
    part: GenAIToolCallResponsePart = {
        "type": "tool_call_response",
        "id": call_id,
        "response": response,
    }
    if call_id is None:
        del part["id"]
    return part


def _to_reasoning_parts(item: Mapping[str, Any]) -> List[GenAIMessagePart]:
    parts: List[GenAIMessagePart] = []
    for summary in _list(item.get("summary")):
        reasoning: GenAIReasoningPart = {
            "type": "reasoning",
            "content": _as_dict(summary).get("text"),
        }
        parts.append(reasoning)
    return parts


def _to_finish_reason(finish_reason: Any) -> str:
    if finish_reason in ("tool_calls", "function_call"):
        return "tool_call"
    if finish_reason is None:
        return "stop"
    return finish_reason


def _to_responses_finish_reason(response: Mapping[str, Any], parts: List[GenAIMessagePart]) -> str:
    if response.get("status") == "incomplete":
        reason = _as_dict(response.get("incomplete_details")).get("reason")
        return "content_filter" if reason == "content_filter" else "length"
    return "tool_call" if any(part["type"] == "tool_call" for part in parts) else "stop"


def _first_not_none(params: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = params.get(key)
        if value is not None:
            return value
    return []


def _list(value: Any) -> Sequence[Any]:
    return value if isinstance(value, (list, tuple)) else ()


def _as_dict(value: Any) -> Any:
    """Reads a dict or an OpenAI SDK object as a dict. Anything else reads as an empty dict, the
    way optional chaining reads it in the Node SDK.

    Typed Any, like the Node SDK's Record<string, any>: OpenAI payload fields are untyped, and a
    malformed payload can have None where the GenAI format expects a string.
    """
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None) or getattr(value, "model_dump", None)
    if callable(to_dict):
        try:
            fields = to_dict()
        except Exception:
            return {}
        if isinstance(fields, Mapping):
            return fields
    return {}
