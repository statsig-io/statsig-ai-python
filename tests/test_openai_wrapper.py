"""wrap_openai() parity with the Node SDK's wrapOpenAI: Responses streaming and parsing, wrapper
options, image attributes, messages on failed and streamed calls, and the active span."""

import asyncio
import inspect
import json
from typing import Any, Callable, Dict, List, Optional

import pytest
from openai.types.chat import ChatCompletionChunk
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode, format_span_id

from statsig_ai import GenAICaptureOptions, WrapOpenAIOptions, wrap_openai
from statsig_ai.otel.singleton import OtelSingleton
from statsig_ai.wrappers import span_telemetry

from mock_openai import AsyncMockOpenAI, MockOpenAI

INPUT = "gen_ai.input.messages"
OUTPUT = "gen_ai.output.messages"
SYSTEM = "gen_ai.system_instructions"

TOOLS = [
    {
        "type": "function",
        "name": "get_gate",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
    }
]

RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "model": "gpt-test-2025",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Checking checkout.", "annotations": []}],
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_9",
            "name": "get_gate",
            "arguments": '{"id":"checkout"}',
            "status": "completed",
        },
    ],
    "usage": {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
}

RESPONSES_REQUEST: Dict[str, Any] = {
    "model": "gpt-test",
    "instructions": "Be brief.",
    "input": [
        {"role": "developer", "content": "Answer in English."},
        {"role": "user", "content": "Is checkout on?"},
    ],
    "tools": TOOLS,
}

CHAT_MESSAGES = [
    {"role": "developer", "content": "Answer in English."},
    {"role": "user", "content": "Is checkout on?"},
]

EXPECTED_INPUT = [{"role": "user", "parts": [{"type": "text", "content": "Is checkout on?"}]}]
EXPECTED_RESPONSES_SYSTEM = [
    {"type": "text", "content": "Be brief."},
    {"type": "text", "content": "Answer in English."},
]
EXPECTED_CHAT_SYSTEM = [{"type": "text", "content": "Answer in English."}]

EXPECTED_RESPONSE_OUTPUT = [
    {
        "role": "assistant",
        "parts": [
            {"type": "text", "content": "Checking checkout."},
            {
                "type": "tool_call",
                "id": "call_9",
                "name": "get_gate",
                "arguments": {"id": "checkout"},
            },
        ],
        "finish_reason": "tool_call",
    }
]


def _chunk(fields: Dict[str, Any]) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {"id": "chatcmpl_2", "object": "chat.completion.chunk", "created": 1, **fields}
    )


CHAT_TOOL_CHUNKS = [
    _chunk(
        {
            "model": "gpt-test-2025",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Let me check. "}}],
        }
    ),
    _chunk(
        {
            "model": "gpt-test-2025",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_3",
                                "type": "function",
                                "function": {"name": "get_gate", "arguments": '{"id":'},
                            }
                        ]
                    },
                }
            ],
        }
    ),
    _chunk(
        {
            "model": "gpt-test-2025",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": '"checkout"}'}}]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    ),
    _chunk(
        {
            "model": "gpt-test-2025",
            "choices": [],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
        }
    ),
]

EXPECTED_CHAT_STREAM_OUTPUT = [
    {
        "role": "assistant",
        "parts": [
            {"type": "text", "content": "Let me check. "},
            {
                "type": "tool_call",
                "id": "call_3",
                "name": "get_gate",
                "arguments": {"id": "checkout"},
            },
        ],
        "finish_reason": "tool_call",
    }
]

IMAGE = {
    "created": 1,
    "data": [{"b64_json": "aGk="}],
    "usage": {
        "input_tokens": 12,
        "output_tokens": 4160,
        "total_tokens": 4172,
        "input_tokens_details": {"image_tokens": 0, "text_tokens": 12},
    },
}


class StartAttributesRecorder(SpanProcessor):
    def __init__(self) -> None:
        self.by_span_id: Dict[int, Dict[str, Any]] = {}

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        self.by_span_id[span.context.span_id] = dict(span.attributes or {})


class FakeStatsig:
    def __init__(self) -> None:
        self.events: List[Dict[str, str]] = []

    def log_event(self, _user, _event_name, _value, metadata) -> None:
        self.events.append(metadata)


class Tracing:
    def __init__(self, exporter: InMemorySpanExporter, starts: StartAttributesRecorder) -> None:
        self._exporter = exporter
        self.starts = starts
        self.statsig = FakeStatsig()

    def spans(self) -> List[ReadableSpan]:
        return list(self._exporter.get_finished_spans())

    def model_spans(self) -> List[ReadableSpan]:
        return [s for s in self.spans() if (s.attributes or {}).get("gen_ai.provider.name")]

    def model_span(self) -> ReadableSpan:
        [span] = self.model_spans()
        return span

    def attrs(self, span: ReadableSpan) -> Dict[str, Any]:
        return dict(span.attributes or {})

    def event(self, span: ReadableSpan) -> Dict[str, str]:
        span_id = format_span_id(span.context.span_id)
        [metadata] = [e for e in self.statsig.events if e["span.span_id"] == span_id]
        return metadata

    def json(self, span: ReadableSpan, key: str) -> Any:
        """Reads a JSON attribute from the span, and checks its gen_ai event has the same."""
        value = self.attrs(span)[key]
        assert self.event(span)[key] == value
        return json.loads(value)


@pytest.fixture
def tracing(monkeypatch: pytest.MonkeyPatch):
    exporter = InMemorySpanExporter()
    starts = StartAttributesRecorder()
    provider = TracerProvider()
    provider.add_span_processor(starts)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    original_instance = OtelSingleton._instance
    OtelSingleton._instance = OtelSingleton(provider)
    result = Tracing(exporter, starts)
    monkeypatch.setattr(span_telemetry, "get_statsig_instance_for_logging", lambda: result.statsig)

    yield result

    OtelSingleton._instance = original_instance
    provider.shutdown()


def wrapped(mock: Any, capture: Optional[GenAICaptureOptions] = None, **options: Any):
    return wrap_openai(
        mock, WrapOpenAIOptions(capture or GenAICaptureOptions(capture_all=True), **options)
    )


def client_for(call: Callable[..., Any], **mock_args: Any):
    """A wrapped mock client: async if `call` is a coroutine function."""
    mock_type = AsyncMockOpenAI if inspect.iscoroutinefunction(call) else MockOpenAI
    return wrapped(mock_type(**mock_args))


def run(call: Callable[..., Any], *args: Any) -> Any:
    result = call(*args)
    return asyncio.run(result) if inspect.iscoroutine(result) else result


# ---------- Calls, as made with a sync or an async client ----------


def chat(client):
    return client.chat.completions.create(model="gpt-test", messages=CHAT_MESSAGES)


async def chat_async(client):
    return await client.chat.completions.create(model="gpt-test", messages=CHAT_MESSAGES)


def chat_stream(client):
    return list(
        client.chat.completions.create(model="gpt-test", messages=CHAT_MESSAGES, stream=True)
    )


async def chat_stream_async(client):
    stream = await client.chat.completions.create(
        model="gpt-test", messages=CHAT_MESSAGES, stream=True
    )
    return [chunk async for chunk in stream]


def responses_create_stream(client):
    return list(client.responses.create(**RESPONSES_REQUEST, stream=True))


async def responses_create_stream_async(client):
    stream = await client.responses.create(**RESPONSES_REQUEST, stream=True)
    return [event async for event in stream]


def responses_stream(client):
    with client.responses.stream(**RESPONSES_REQUEST) as stream:
        return "".join(e.delta for e in stream if e.type == "response.output_text.delta")


async def responses_stream_async(client):
    async with client.responses.stream(**RESPONSES_REQUEST) as stream:
        return "".join([e.delta async for e in stream if e.type == "response.output_text.delta"])


def responses_stream_final(client):
    with client.responses.stream(**RESPONSES_REQUEST) as stream:
        return stream.get_final_response().output_text


async def responses_stream_final_async(client):
    async with client.responses.stream(**RESPONSES_REQUEST) as stream:
        return (await stream.get_final_response()).output_text


def responses_parse(client):
    return client.responses.parse(**RESPONSES_REQUEST)


async def responses_parse_async(client):
    return await client.responses.parse(**RESPONSES_REQUEST)


def generate_image(client):
    return client.images.generate(
        model="gpt-image-1",
        prompt="A cat",
        size="1024x1024",
        quality="high",
        output_format="webp",
        output_compression=80,
    )


async def generate_image_async(client):
    return await generate_image(client)


# ---------- responses.stream() and responses.parse() ----------


@pytest.mark.parametrize(
    "read, expected",
    [
        (responses_stream, "Checking checkout."),
        (responses_stream_async, "Checking checkout."),
        (responses_stream_final, "Checking checkout."),
        (responses_stream_final_async, "Checking checkout."),
    ],
)
def test_responses_stream_records_the_final_response(tracing, read, expected):
    client = client_for(read, response=RESPONSE)

    assert run(read, client) == expected

    span = tracing.model_span()
    attrs = tracing.attrs(span)
    assert span.name == "generate_content gpt-test"
    assert span.status.status_code == StatusCode.OK
    assert attrs["gen_ai.operation.name"] == "generate_content"
    assert attrs["gen_ai.operation.source_name"] == "openai.responses.stream"
    assert attrs["gen_ai.request.model"] == "gpt-test"
    assert attrs["gen_ai.request.stream"] is True
    assert attrs["gen_ai.response.id"] == "resp_1"
    assert attrs["gen_ai.response.model"] == "gpt-test-2025"
    assert attrs["gen_ai.usage.input_tokens"] == 20
    assert attrs["gen_ai.usage.output_tokens"] == 8
    assert attrs["statsig.gen_ai.usage.total_tokens"] == 28
    assert "statsig.gen_ai.server.time_to_first_token_ms" in attrs
    assert tracing.json(span, SYSTEM) == EXPECTED_RESPONSES_SYSTEM
    assert tracing.json(span, INPUT) == EXPECTED_INPUT
    assert tracing.json(span, OUTPUT) == EXPECTED_RESPONSE_OUTPUT
    assert tracing.json(span, "gen_ai.tool.definitions") == TOOLS
    assert "choices" not in attrs
    assert tracing.event(span)["gen_ai.usage.output_tokens"] == "8"


def test_responses_stream_span_ends_when_the_stream_has_been_read(tracing):
    mock = MockOpenAI(response=RESPONSE)
    client = wrapped(mock)

    manager = client.responses.stream(**RESPONSES_REQUEST)
    assert not mock.responses.calls
    with manager as stream:
        assert tracing.spans() == []
        events = list(stream)
        # The span ends with the stream, before the manager exits.
        assert len(tracing.model_spans()) == 1
        assert stream.get_final_response().id == "resp_1"

    assert events[-1].type == "response.completed"
    assert len(mock.responses.calls) == 1
    assert mock.responses.streams[0].closed
    assert len(tracing.statsig.events) == 1


def test_responses_stream_closed_early_ends_the_span_without_the_output(tracing):
    mock = MockOpenAI(response=RESPONSE)

    with wrapped(mock).responses.stream(**RESPONSES_REQUEST) as stream:
        assert next(stream).type == "response.created"

    span = tracing.model_span()
    assert span.status.status_code == StatusCode.UNSET
    assert tracing.json(span, INPUT) == EXPECTED_INPUT
    assert OUTPUT not in tracing.attrs(span)
    assert mock.responses.streams[0].closed


@pytest.mark.parametrize("read", [responses_stream, responses_stream_async])
def test_responses_stream_request_failure_fails_the_span(tracing, read):
    client = client_for(read, error=RuntimeError("bad model"))

    with pytest.raises(RuntimeError, match="bad model"):
        run(read, client)

    span = tracing.model_span()
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "bad model"
    assert tracing.json(span, SYSTEM) == EXPECTED_RESPONSES_SYSTEM
    assert tracing.json(span, INPUT) == EXPECTED_INPUT
    assert tracing.event(span)["exception.type"] == "RuntimeError"


@pytest.mark.parametrize("parse", [responses_parse, responses_parse_async])
def test_responses_parse_is_traced(tracing, parse):
    client = client_for(parse, response=RESPONSE)

    assert run(parse, client).output_text == "Checking checkout."

    span = tracing.model_span()
    attrs = tracing.attrs(span)
    assert span.name == "generate_content gpt-test"
    assert span.status.status_code == StatusCode.OK
    assert attrs["gen_ai.operation.source_name"] == "openai.responses.parse"
    assert attrs["gen_ai.response.id"] == "resp_1"
    assert attrs["gen_ai.usage.input_tokens"] == 20
    assert "gen_ai.request.stream" not in attrs
    assert tracing.json(span, SYSTEM) == EXPECTED_RESPONSES_SYSTEM
    assert tracing.json(span, INPUT) == EXPECTED_INPUT
    assert tracing.json(span, OUTPUT) == EXPECTED_RESPONSE_OUTPUT


# ---------- custom_attributes and max_json_chars ----------


def test_custom_attributes_are_set_on_every_span_and_event(tracing):
    client = wrapped(
        MockOpenAI(response=RESPONSE),
        custom_attributes={"app.team": "growth", "app.retries": 2, "gen_ai.provider.name": "x"},
    )

    chat(client)
    responses_stream(client)
    generate_image(client)

    spans = tracing.model_spans()
    assert len(spans) == 3
    for span in spans:
        attrs = tracing.attrs(span)
        assert attrs["app.team"] == "growth"
        assert attrs["app.retries"] == 2
        # The wrapper's own attributes win, as in the Node SDK.
        assert attrs["gen_ai.provider.name"] == "openai"
        started = tracing.starts.by_span_id[span.context.span_id]
        assert started["app.team"] == "growth"
        event = tracing.event(span)
        assert event["app.team"] == "growth"
        assert event["app.retries"] == "2"


def test_max_json_chars_caps_json_attributes(tracing):
    content = "x" * 100
    client = wrapped(
        MockOpenAI(), GenAICaptureOptions(capture_input_messages=True), max_json_chars=50
    )

    client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": content}])

    full = json.dumps(
        [{"role": "user", "parts": [{"type": "text", "content": content}]}], separators=(",", ":")
    )
    span = tracing.model_span()
    attrs = tracing.attrs(span)
    assert attrs[INPUT] == full[:50] + "…(truncated)"
    assert attrs[f"{INPUT}_len"] == len(full)
    assert tracing.event(span)[INPUT] == attrs[INPUT]


def test_json_attributes_are_capped_at_40000_chars_by_default(tracing):
    assert WrapOpenAIOptions().max_json_chars == 40_000
    content = "x" * 50_000

    wrapped(MockOpenAI()).chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": content}]
    )

    full = json.dumps(
        [{"role": "user", "parts": [{"type": "text", "content": content}]}], separators=(",", ":")
    )
    attrs = tracing.attrs(tracing.model_span())
    assert attrs[INPUT] == full[:40_000] + "…(truncated)"
    assert attrs[f"{INPUT}_len"] == len(full)


# ---------- Images ----------


@pytest.mark.parametrize("generate", [generate_image, generate_image_async])
def test_image_generation_records_image_attributes_and_usage(tracing, generate):
    client = client_for(generate, image=IMAGE)

    assert run(generate, client) == IMAGE

    span = tracing.model_span()
    attrs = tracing.attrs(span)
    assert span.name == "generate_content gpt-image-1"
    assert attrs["gen_ai.operation.source_name"] == "openai.images.generate"
    assert attrs["statsig.gen_ai.images.size"] == "1024x1024"
    assert attrs["statsig.gen_ai.images.quality"] == "high"
    assert attrs["statsig.gen_ai.images.output_format"] == "webp"
    assert attrs["statsig.gen_ai.images.output_compression"] == 80
    assert attrs["gen_ai.usage.input_tokens"] == 12
    assert attrs["gen_ai.usage.output_tokens"] == 4160
    assert attrs["statsig.gen_ai.usage.total_tokens"] == 4172
    assert tracing.event(span)["statsig.gen_ai.images.size"] == "1024x1024"
    assert tracing.json(span, INPUT) == [
        {"role": "user", "parts": [{"type": "text", "content": "A cat"}]}
    ]
    assert OUTPUT not in attrs


def test_image_attributes_are_left_out_of_other_calls(tracing):
    chat(wrapped(MockOpenAI()))

    attrs = tracing.attrs(tracing.model_span())
    assert not [key for key in attrs if key.startswith("statsig.gen_ai.images.")]


def test_embeddings_record_encoding_formats_as_a_list(tracing):
    wrapped(MockOpenAI()).embeddings.create(
        model="text-embedding-3-small", input="hi", encoding_format="float", dimensions=256
    )

    attrs = tracing.attrs(tracing.model_span())
    assert json.loads(attrs["gen_ai.request.encoding_formats"]) == ["float"]
    assert attrs["gen_ai.embeddings.dimension.count"] == 256


# ---------- Messages on failed and streamed calls ----------


@pytest.mark.parametrize("call", [chat, chat_async, chat_stream, chat_stream_async])
def test_failed_calls_record_input_messages_and_system_instructions(tracing, call):
    client = client_for(call, error=RuntimeError("rate limited"))

    with pytest.raises(RuntimeError, match="rate limited"):
        run(call, client)

    span = tracing.model_span()
    assert span.status.status_code == StatusCode.ERROR
    assert tracing.json(span, INPUT) == EXPECTED_INPUT
    assert tracing.json(span, SYSTEM) == EXPECTED_CHAT_SYSTEM
    assert OUTPUT not in tracing.attrs(span)
    assert tracing.event(span)["span.status_code"] == "ERROR"


@pytest.mark.parametrize("read", [chat_stream, chat_stream_async])
def test_chat_streams_record_input_output_tool_calls_and_usage(tracing, read):
    client = client_for(read, chat_chunks=CHAT_TOOL_CHUNKS)

    assert len(run(read, client)) == len(CHAT_TOOL_CHUNKS)

    span = tracing.model_span()
    attrs = tracing.attrs(span)
    assert span.status.status_code == StatusCode.OK
    assert attrs["gen_ai.request.stream"] is True
    assert attrs["gen_ai.response.id"] == "chatcmpl_2"
    assert attrs["gen_ai.response.model"] == "gpt-test-2025"
    assert attrs["gen_ai.usage.input_tokens"] == 12
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert json.loads(attrs["gen_ai.response.finish_reasons"]) == ["tool_calls"]
    assert "statsig.gen_ai.server.time_to_first_token_ms" in attrs
    assert tracing.json(span, INPUT) == EXPECTED_INPUT
    assert tracing.json(span, SYSTEM) == EXPECTED_CHAT_SYSTEM
    assert tracing.json(span, OUTPUT) == EXPECTED_CHAT_STREAM_OUTPUT


def test_chat_stream_of_text_records_one_text_part(tracing):
    chat_stream(wrapped(MockOpenAI()))

    assert tracing.json(tracing.model_span(), OUTPUT) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "This is a mock stream"}],
            "finish_reason": "stop",
        }
    ]


def test_completions_stream_records_the_streamed_text(tracing):
    stream = wrapped(MockOpenAI()).completions.create(
        model="gpt-3.5-turbo-instruct", prompt="Say hi", stream=True
    )
    list(stream)

    span = tracing.model_span()
    assert tracing.attrs(span)["gen_ai.usage.output_tokens"] == 2
    assert tracing.json(span, OUTPUT) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "Hi there"}],
            "finish_reason": "stop",
        }
    ]


@pytest.mark.parametrize("read", [responses_create_stream, responses_create_stream_async])
def test_responses_create_streams_record_output_messages_and_usage(tracing, read):
    client = client_for(read, response=RESPONSE)

    events = run(read, client)

    assert events[-1].type == "response.completed"
    span = tracing.model_span()
    attrs = tracing.attrs(span)
    assert attrs["gen_ai.operation.source_name"] == "openai.responses.create"
    assert attrs["gen_ai.request.stream"] is True
    assert attrs["gen_ai.response.id"] == "resp_1"
    assert attrs["gen_ai.usage.output_tokens"] == 8
    assert tracing.json(span, INPUT) == EXPECTED_INPUT
    assert tracing.json(span, SYSTEM) == EXPECTED_RESPONSES_SYSTEM
    assert tracing.json(span, OUTPUT) == EXPECTED_RESPONSE_OUTPUT
    assert "choices" not in attrs


@pytest.mark.parametrize(
    "capture",
    [GenAICaptureOptions(), GenAICaptureOptions(capture_input_messages=True)],
    ids=["no capture", "input capture"],
)
def test_streamed_output_is_left_out_when_output_capture_is_off(tracing, capture):
    client = wrapped(MockOpenAI(response=RESPONSE), capture)

    responses_create_stream(client)
    responses_stream(client)
    chat_stream(client)

    spans = tracing.model_spans()
    assert len(spans) == 3
    for span in spans:
        attrs = tracing.attrs(span)
        event = tracing.event(span)
        for key in [OUTPUT, "choices"]:
            assert key not in attrs and key not in event
        assert "gen_ai.usage.output_tokens" in attrs


def test_streamed_output_is_recorded_with_output_capture_only(tracing):
    client = wrapped(
        MockOpenAI(response=RESPONSE), GenAICaptureOptions(capture_output_messages=True)
    )

    responses_create_stream(client)

    attrs = tracing.attrs(tracing.model_span())
    assert json.loads(attrs[OUTPUT]) == EXPECTED_RESPONSE_OUTPUT
    assert INPUT not in attrs and "choices" not in attrs


def test_stream_ends_its_span_once_read_or_closed(tracing):
    mock = MockOpenAI()
    client = wrapped(mock)

    stream = client.chat.completions.create(model="gpt-4", messages=CHAT_MESSAGES, stream=True)
    assert tracing.spans() == []
    with stream:
        next(stream)
    # The span ends when the stream is closed, without the output it didn't read.
    [span] = tracing.model_spans()
    assert span.status.status_code == StatusCode.UNSET
    assert OUTPUT not in tracing.attrs(span)
    assert mock.chat.completions.streams[0].closed
    assert stream.response is mock.chat.completions.streams[0]


def test_stream_failure_fails_the_span(tracing):
    client = wrapped(MockOpenAI(stream_error=ConnectionError("connection reset")))

    with pytest.raises(ConnectionError):
        chat_stream(client)

    span = tracing.model_span()
    assert span.status.status_code == StatusCode.ERROR
    assert tracing.json(span, INPUT) == EXPECTED_INPUT
    assert OUTPUT not in tracing.attrs(span)


# ---------- The active span ----------


def start_http_span() -> None:
    """Stands in for HTTP client instrumentation, which starts a span when a request is sent."""
    OtelSingleton.get_tracer_provider_static().get_tracer("http").start_span("http.request").end()


@pytest.mark.parametrize(
    "call",
    [
        chat,
        chat_async,
        chat_stream,
        chat_stream_async,
        responses_create_stream,
        responses_create_stream_async,
        responses_stream,
        responses_stream_async,
        responses_parse,
        responses_parse_async,
        generate_image,
        generate_image_async,
    ],
)
def test_spans_started_during_a_call_are_children_of_its_span(tracing, call):
    client = client_for(call, response=RESPONSE, on_request=start_http_span)

    run(call, client)

    model = tracing.model_span()
    [http] = [s for s in tracing.spans() if s.name == "http.request"]
    assert http.parent is not None
    assert http.parent.span_id == model.context.span_id
    assert model.parent is None
    assert trace.get_current_span() is trace.INVALID_SPAN


# ---------- System instructions ----------


def test_developer_messages_and_instructions_are_system_instructions(tracing):
    client = wrapped(MockOpenAI(response=RESPONSE))

    chat(client)
    client.responses.create(**RESPONSES_REQUEST)
    responses_stream(client)

    chat_span, *responses_spans = tracing.model_spans()
    assert tracing.json(chat_span, SYSTEM) == EXPECTED_CHAT_SYSTEM
    assert tracing.json(chat_span, INPUT) == EXPECTED_INPUT
    assert len(responses_spans) == 2
    for span in responses_spans:
        assert tracing.json(span, SYSTEM) == EXPECTED_RESPONSES_SYSTEM
        assert tracing.json(span, INPUT) == EXPECTED_INPUT
