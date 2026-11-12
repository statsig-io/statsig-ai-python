import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest
from openai.types.chat import ChatCompletion
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from statsig_ai import GenAICaptureOptions, WrapOpenAIOptions, wrap_openai
from statsig_ai.otel.singleton import OtelSingleton
from statsig_ai.wrappers import span_telemetry

from mock_openai import DefaultMockResponses, MockOpenAI

INPUT = "gen_ai.input.messages"
OUTPUT = "gen_ai.output.messages"
SYSTEM = "gen_ai.system_instructions"
MESSAGE_ATTRS = [INPUT, OUTPUT, SYSTEM]


class FakeStatsig:
    def __init__(self) -> None:
        self.events: List[Dict[str, str]] = []

    def log_event(self, _user, _event_name, _value, metadata) -> None:
        self.events.append(metadata)


class Tracing:
    def __init__(self, exporter: InMemorySpanExporter) -> None:
        self._exporter = exporter
        self.statsig = FakeStatsig()

    def only_span_and_event(self):
        spans = self._exporter.get_finished_spans()
        assert len(spans) == 1
        assert len(self.statsig.events) == 1
        return dict(spans[0].attributes or {}), self.statsig.events[0]

    def messages(self, key: str) -> Any:
        """Reads a message attribute from the span, and checks the gen_ai event has the same."""
        attrs, metadata = self.only_span_and_event()
        assert metadata[key] == attrs[key]
        return json.loads(attrs[key])


@pytest.fixture
def tracing(monkeypatch: pytest.MonkeyPatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    original_instance = OtelSingleton._instance
    OtelSingleton._instance = OtelSingleton(provider)
    result = Tracing(exporter)
    monkeypatch.setattr(span_telemetry, "get_statsig_instance_for_logging", lambda: result.statsig)

    yield result

    OtelSingleton._instance = original_instance
    provider.shutdown()


def wrapped(mock: MockOpenAI, capture: Optional[GenAICaptureOptions] = None):
    return wrap_openai(mock, WrapOpenAIOptions(capture or GenAICaptureOptions(capture_all=True)))


TOOL_CALL_COMPLETION = {
    "id": "chatcmpl_1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-test",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "get_gate", "arguments": '{"id":"checkout"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
}

TOOL_CALL_MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Is checkout on?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "list_gates", "arguments": "{}"},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": '["checkout"]'},
]

EXPECTED_TOOL_CALL_INPUT = [
    {"role": "user", "parts": [{"type": "text", "content": "Is checkout on?"}]},
    {
        "role": "assistant",
        "parts": [{"type": "tool_call", "id": "call_1", "name": "list_gates", "arguments": {}}],
    },
    {
        "role": "tool",
        "parts": [{"type": "tool_call_response", "id": "call_1", "response": '["checkout"]'}],
    },
]

EXPECTED_TOOL_CALL_OUTPUT = [
    {
        "role": "assistant",
        "parts": [
            {
                "type": "tool_call",
                "id": "call_2",
                "name": "get_gate",
                "arguments": {"id": "checkout"},
            }
        ],
        "finish_reason": "tool_call",
    }
]


def test_chat_completion_tool_calls_are_recorded_in_the_gen_ai_format(tracing):
    mock = MockOpenAI(chat_completion=TOOL_CALL_COMPLETION)

    wrapped(mock).chat.completions.create(model="gpt-test", messages=TOOL_CALL_MESSAGES)

    assert len(mock.chat.completions.calls) == 1
    assert tracing.messages(SYSTEM) == [{"type": "text", "content": "You are helpful."}]
    assert tracing.messages(INPUT) == EXPECTED_TOOL_CALL_INPUT
    assert tracing.messages(OUTPUT) == EXPECTED_TOOL_CALL_OUTPUT


def test_chat_completion_sdk_objects_are_recorded_in_the_gen_ai_format(tracing):
    completion = ChatCompletion.model_validate(TOOL_CALL_COMPLETION)

    wrapped(MockOpenAI(chat_completion=completion)).chat.completions.create(
        model="gpt-test", messages=TOOL_CALL_MESSAGES
    )

    assert tracing.messages(OUTPUT) == EXPECTED_TOOL_CALL_OUTPUT


def test_async_chat_completion_is_recorded_in_the_gen_ai_format(tracing):
    class AsyncCompletions:
        async def create(self, *args, **kwargs):
            return TOOL_CALL_COMPLETION

    mock = MockOpenAI()
    mock.chat.completions = AsyncCompletions()  # type: ignore[assignment]
    client = wrapped(mock)

    asyncio.run(client.chat.completions.create(model="gpt-test", messages=TOOL_CALL_MESSAGES))

    assert tracing.messages(INPUT) == EXPECTED_TOOL_CALL_INPUT
    assert tracing.messages(OUTPUT) == EXPECTED_TOOL_CALL_OUTPUT


def test_responses_instructions_input_items_and_output_are_recorded(tracing):
    mock = MockOpenAI(
        response={
            "id": "resp_1",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Gate is listed."}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Checkout is on."}],
                },
            ],
            "usage": {"input_tokens": 20, "output_tokens": 8},
        }
    )

    wrapped(mock).responses.create(
        model="gpt-test",
        instructions="Be brief.",
        input=[
            {"role": "user", "content": [{"type": "input_text", "text": "Is checkout on?"}]},
            {"type": "function_call", "call_id": "call_1", "name": "list_gates", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": '["checkout"]'},
        ],
    )

    assert len(mock.responses.calls) == 1
    assert tracing.messages(SYSTEM) == [{"type": "text", "content": "Be brief."}]
    assert tracing.messages(INPUT) == EXPECTED_TOOL_CALL_INPUT
    assert tracing.messages(OUTPUT) == [
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "Gate is listed."},
                {"type": "text", "content": "Checkout is on."},
            ],
            "finish_reason": "stop",
        }
    ]


def test_text_completions_are_recorded_in_the_gen_ai_format(tracing):
    wrapped(MockOpenAI()).completions.create(model="gpt-3.5-turbo-instruct", prompt="Say hi")

    assert tracing.messages(INPUT) == [
        {"role": "user", "parts": [{"type": "text", "content": "Say hi"}]}
    ]
    text = DefaultMockResponses.text_completion["choices"][0]["text"]  # type: ignore[index]
    assert tracing.messages(OUTPUT) == [
        {"role": "assistant", "parts": [{"type": "text", "content": text}], "finish_reason": "stop"}
    ]


def test_embeddings_record_input_but_no_output_messages(tracing):
    wrapped(MockOpenAI()).embeddings.create(model="text-embedding-3-small", input="Embed this")

    attrs, metadata = tracing.only_span_and_event()
    assert tracing.messages(INPUT) == [
        {"role": "user", "parts": [{"type": "text", "content": "Embed this"}]}
    ]
    assert OUTPUT not in attrs and OUTPUT not in metadata
    assert SYSTEM not in attrs and SYSTEM not in metadata


def test_system_instructions_are_left_out_when_the_request_has_none(tracing):
    wrapped(MockOpenAI()).chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": "hi"}]
    )

    attrs, metadata = tracing.only_span_and_event()
    assert SYSTEM not in attrs and SYSTEM not in metadata
    assert tracing.messages(INPUT) == [
        {"role": "user", "parts": [{"type": "text", "content": "hi"}]}
    ]


@pytest.mark.parametrize(
    "capture, expected",
    [
        (GenAICaptureOptions(), []),
        (GenAICaptureOptions(capture_input_messages=True), [INPUT]),
        (GenAICaptureOptions(capture_output_messages=True), [OUTPUT]),
        (GenAICaptureOptions(capture_system_instructions=True), [SYSTEM]),
        (GenAICaptureOptions(capture_tool_definitions=True), []),
        (GenAICaptureOptions(capture_all=True), MESSAGE_ATTRS),
    ],
)
def test_capture_options_gate_each_message_attribute(tracing, capture, expected):
    wrapped(MockOpenAI(), capture).chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ],
    )

    attrs, metadata = tracing.only_span_and_event()
    assert [key for key in MESSAGE_ATTRS if key in attrs] == expected
    assert [key for key in MESSAGE_ATTRS if key in metadata] == expected
