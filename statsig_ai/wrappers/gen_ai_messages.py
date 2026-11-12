"""OpenTelemetry GenAI message format.

The wrappers write gen_ai.input.messages, gen_ai.output.messages and gen_ai.system_instructions
in this format, so traces from every provider share one shape. Ported from the Node SDK's
gen-ai-messages.ts.
https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
"""

import json
from typing import Any, List, Literal, Union

from typing_extensions import NotRequired, TypedDict


class GenAITextPart(TypedDict):
    type: Literal["text"]
    content: str


class GenAIReasoningPart(TypedDict):
    type: Literal["reasoning"]
    content: str


class GenAIToolCallPart(TypedDict):
    type: Literal["tool_call"]
    id: NotRequired[str]
    name: str
    arguments: NotRequired[Any]


class GenAIToolCallResponsePart(TypedDict):
    type: Literal["tool_call_response"]
    id: NotRequired[str]
    response: Any


class GenAIBlobPart(TypedDict):
    type: Literal["blob"]
    modality: str
    mime_type: NotRequired[str]


class GenAIUriPart(TypedDict):
    type: Literal["uri"]
    modality: str
    uri: str
    mime_type: NotRequired[str]


GenAIMessagePart = Union[
    GenAITextPart,
    GenAIReasoningPart,
    GenAIToolCallPart,
    GenAIToolCallResponsePart,
    GenAIBlobPart,
    GenAIUriPart,
]

GenAIRole = Literal["system", "user", "assistant", "tool"]


class GenAIInputMessage(TypedDict):
    role: GenAIRole
    parts: List[GenAIMessagePart]


class GenAIOutputMessage(TypedDict):
    role: Literal["assistant"]
    parts: List[GenAIMessagePart]
    finish_reason: str


def text_part(content: str) -> GenAITextPart:
    return {"type": "text", "content": content}


def parse_tool_arguments(arguments: str) -> Any:
    """Parses tool call arguments as JSON, or returns them unchanged if they aren't valid JSON."""
    try:
        # Like JSON.parse, reject NaN and Infinity, which json.dumps can't write back as JSON.
        return json.loads(arguments, parse_constant=_reject_constant)
    except (TypeError, ValueError):
        return arguments


def _reject_constant(constant: str) -> Any:
    raise ValueError(f"Invalid JSON constant: {constant}")
