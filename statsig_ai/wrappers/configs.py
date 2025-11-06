from dataclasses import dataclass


@dataclass
class GenAICaptureOptions:
    capture_all: bool = False
    capture_input_messages: bool = False
    capture_output_messages: bool = False
    capture_system_instructions: bool = False
    capture_tool_definitions: bool = False


class WrapOpenAIOptions:
    def __init__(self, gen_ai_capture_options: GenAICaptureOptions = GenAICaptureOptions()):
        self.gen_ai_capture_options = gen_ai_capture_options
