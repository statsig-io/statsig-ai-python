from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .span_telemetry import MAX_JSON_CHARS


@dataclass
class GenAICaptureOptions:
    capture_all: bool = False
    capture_input_messages: bool = False
    capture_output_messages: bool = False
    capture_system_instructions: bool = False
    capture_tool_definitions: bool = False


class WrapOpenAIOptions:
    """Options for wrap_openai().

    `gen_ai_capture_options` picks the messages and tool definitions to record. None are recorded
    by default. `custom_attributes` are set on every span the wrapper creates, and on its
    `statsig::gen_ai` event. JSON attributes, such as messages, are cut to `max_json_chars`
    characters (40,000 by default) and the full length is recorded as `<attribute>_len`.
    """

    def __init__(
        self,
        gen_ai_capture_options: Optional[GenAICaptureOptions] = None,
        *,
        custom_attributes: Optional[Mapping[str, Any]] = None,
        max_json_chars: int = MAX_JSON_CHARS,
    ):
        self.gen_ai_capture_options = gen_ai_capture_options or GenAICaptureOptions()
        self.custom_attributes = dict(custom_attributes or {})
        self.max_json_chars = max_json_chars
