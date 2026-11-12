import asyncio
import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import pytest
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from statsig_ai import (
    GenAICaptureOptions,
    Prompt,
    PromptVersion,
    StatsigContext,
    StatsigUser,
    WrapOpenAIOptions,
    start_workflow,
    statsig_context,
    statsig_environment,
    wrap_anthropic,
)
from statsig_ai.otel.conventions import (
    STATSIG_ATTR_CUSTOM_IDS,
    STATSIG_ATTR_ENVIRONMENT,
    STATSIG_ATTR_LLM_PROMPT_NAME,
    STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES,
    STATSIG_ATTR_LLM_PROMPT_VERSION,
    STATSIG_ATTR_LLM_PROMPT_VERSION_ID,
    STATSIG_ATTR_SPAN_LLM_ROOT,
    STATSIG_ATTR_SPAN_TYPE,
    STATSIG_ATTR_USER_ID,
)
from statsig_ai.otel.singleton import OtelSingleton
from statsig_ai.wrappers import span_telemetry
from statsig_ai.wrappers.anthropic_messages import (
    aggregate_anthropic_stream_events,
    to_anthropic_finish_reason,
    to_anthropic_input_messages,
)

from mock_anthropic import MockAnthropic, MockAsyncAnthropic, SDKObject, stream_events_for

INPUT = "gen_ai.input.messages"
OUTPUT = "gen_ai.output.messages"
SYSTEM = "gen_ai.system_instructions"
TOOLS = "gen_ai.tool.definitions"
CAPTURED = [INPUT, OUTPUT, SYSTEM, TOOLS]
CAPTURE_ALL = GenAICaptureOptions(capture_all=True)

RESPONSE: Dict[str, Any] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-test-20250101",
    "content": [
        {"type": "thinking", "thinking": "Check the gate.", "signature": "sig"},
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "toolu_2", "name": "get_gate", "input": {"id": "checkout"}},
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 10,
        "cache_read_input_tokens": 4,
        "cache_creation_input_tokens": 2,
        "output_tokens": 6,
    },
}
EXPECTED_OUTPUT = [
    {
        "role": "assistant",
        "parts": [
            {"type": "reasoning", "content": "Check the gate."},
            {"type": "text", "content": "Let me check."},
            {
                "type": "tool_call",
                "id": "toolu_2",
                "name": "get_gate",
                "arguments": {"id": "checkout"},
            },
        ],
        "finish_reason": "tool_call",
    }
]
# Cache reads and writes count as input tokens; reads are also reported as cached tokens.
EXPECTED_USAGE = {
    "gen_ai.usage.input_tokens": 16,
    "gen_ai.usage.output_tokens": 6,
    "statsig.gen_ai.usage.total_tokens": 22,
    "statsig.gen_ai.usage.input_cached_tokens": 4,
}
TOOL_DEFINITIONS = [{"name": "get_gate", "input_schema": {"type": "object"}}]


def make_request(**overrides: Any) -> Dict[str, Any]:
    return {
        "model": "claude-test",
        "max_tokens": 256,
        "temperature": 0.2,
        "top_k": 5,
        "stop_sequences": ["END"],
        "system": "You are helpful.",
        "tools": TOOL_DEFINITIONS,
        "messages": [
            {"role": "user", "content": "Is checkout on?"},
            # Callers pass a response's content blocks back as they are, as SDK objects.
            {
                "role": "assistant",
                "content": [
                    SDKObject(
                        {"type": "tool_use", "id": "toolu_1", "name": "list_gates", "input": {}}
                    )
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": '["checkout"]'}],
                    }
                ],
            },
        ],
        **overrides,
    }


EXPECTED_INPUT = [
    {"role": "user", "parts": [{"type": "text", "content": "Is checkout on?"}]},
    {
        "role": "assistant",
        "parts": [{"type": "tool_call", "id": "toolu_1", "name": "list_gates", "arguments": {}}],
    },
    {
        "role": "tool",
        "parts": [{"type": "tool_call_response", "id": "toolu_1", "response": '["checkout"]'}],
    },
]


class StartAttributesRecorder(SpanProcessor):
    """The attributes each span started with, in the order they started."""

    def __init__(self) -> None:
        self.spans: List[Dict[str, Any]] = []

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        self.spans.append(dict(span.attributes or {}))


class FakeStatsig:
    def __init__(self) -> None:
        self.events: List[Tuple[str, str, Dict[str, str]]] = []

    def log_event(self, _user, event_name, value, metadata) -> None:
        self.events.append((event_name, value, metadata))


class Tracing:
    def __init__(self, exporter: InMemorySpanExporter, started: StartAttributesRecorder) -> None:
        self.exporter = exporter
        self.started = started
        self.statsig = FakeStatsig()

    def spans(self) -> List[ReadableSpan]:
        return list(self.exporter.get_finished_spans())

    def span(self) -> ReadableSpan:
        [span] = self.spans()
        return span

    def attrs(self) -> Dict[str, Any]:
        return dict(self.span().attributes or {})

    def json(self, key: str) -> Any:
        return json.loads(self.attrs()[key])

    def event(self) -> Dict[str, str]:
        [(_, _, metadata)] = self.statsig.events
        return metadata

    def clear(self) -> None:
        self.exporter.clear()
        self.statsig.events.clear()


@pytest.fixture
def tracing(monkeypatch: pytest.MonkeyPatch):
    exporter = InMemorySpanExporter()
    started = StartAttributesRecorder()
    provider = TracerProvider()
    provider.add_span_processor(started)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    original_instance = OtelSingleton._instance
    OtelSingleton._instance = OtelSingleton(provider)
    result = Tracing(exporter, started)
    monkeypatch.setattr(span_telemetry, "get_statsig_instance_for_logging", lambda: result.statsig)

    yield result

    OtelSingleton._instance = original_instance
    provider.shutdown()


def sync_client(capture: Optional[GenAICaptureOptions] = CAPTURE_ALL):
    mock = MockAnthropic(RESPONSE, stream_events_for(RESPONSE))
    return mock, wrap_anthropic(mock, WrapOpenAIOptions(capture))


def async_client(capture: Optional[GenAICaptureOptions] = CAPTURE_ALL):
    mock = MockAsyncAnthropic(RESPONSE, stream_events_for(RESPONSE))
    return mock, wrap_anthropic(mock, WrapOpenAIOptions(capture))


def response_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["gen_ai.response.id", "gen_ai.response.model", "gen_ai.response.finish_reasons"]
    return {key: attrs.get(key) for key in [*keys, *EXPECTED_USAGE, OUTPUT]}


# ---------- messages.create ----------


def test_create_records_the_request_response_and_usage(tracing):
    mock, client = sync_client()

    message = client.messages.create(**make_request())

    assert message.id == "msg_1"
    assert mock.messages.requests == [make_request()]
    span = tracing.span()
    assert span.name == "chat claude-test"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.OK
    attrs = tracing.attrs()
    assert {k: v for k, v in attrs.items() if k in EXPECTED_USAGE} == EXPECTED_USAGE
    assert attrs["gen_ai.provider.name"] == "anthropic"
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.operation.source_name"] == "anthropic.messages.create"
    assert attrs["gen_ai.request.model"] == "claude-test"
    assert attrs["gen_ai.request.max_tokens"] == 256
    assert attrs["gen_ai.request.temperature"] == 0.2
    assert attrs["gen_ai.request.top_k"] == 5
    assert json.loads(attrs["gen_ai.request.stop_sequences"]) == ["END"]
    assert "gen_ai.request.stream" not in attrs
    assert attrs["gen_ai.response.id"] == "msg_1"
    assert attrs["gen_ai.response.model"] == "claude-test-20250101"
    assert json.loads(attrs["gen_ai.response.finish_reasons"]) == ["tool_use"]
    assert attrs["statsig.gen_ai.server.time_to_first_token_ms"] >= 0
    assert tracing.json(INPUT) == EXPECTED_INPUT
    assert tracing.json(SYSTEM) == [{"type": "text", "content": "You are helpful."}]
    assert tracing.json(OUTPUT) == EXPECTED_OUTPUT
    assert tracing.json(TOOLS) == TOOL_DEFINITIONS


def test_logs_a_statsig_gen_ai_event_with_the_span_attributes(tracing):
    _, client = sync_client()

    client.messages.create(**make_request())

    span = tracing.span()
    [(event_name, value, metadata)] = tracing.statsig.events
    assert event_name == "statsig::gen_ai"
    assert value == "chat claude-test"
    assert metadata["gen_ai.provider.name"] == "anthropic"
    assert metadata["span.trace_id"] == format(span.context.trace_id, "032x")
    assert metadata["span.span_id"] == format(span.context.span_id, "016x")
    assert metadata["span.status_code"] == "OK"
    for key, attr in (span.attributes or {}).items():
        assert metadata[key] == str(attr), key


def test_system_instructions_accept_text_blocks_and_are_left_out_without_system(tracing):
    _, client = sync_client()
    system = [
        {"type": "text", "text": "You are helpful."},
        {"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}},
    ]

    client.messages.create(**make_request(system=system))
    assert tracing.json(SYSTEM) == [
        {"type": "text", "content": "You are helpful."},
        {"type": "text", "content": "Be brief."},
    ]

    tracing.clear()
    request = make_request()
    del request["system"]
    client.messages.create(**request)
    assert SYSTEM not in tracing.attrs()


@pytest.mark.parametrize(
    "capture, recorded",
    [
        (None, []),
        (GenAICaptureOptions(capture_input_messages=True), [INPUT]),
        (GenAICaptureOptions(capture_output_messages=True), [OUTPUT]),
        (GenAICaptureOptions(capture_system_instructions=True), [SYSTEM]),
        (GenAICaptureOptions(capture_tool_definitions=True), [TOOLS]),
        (CAPTURE_ALL, CAPTURED),
    ],
)
def test_messages_are_only_recorded_when_captured(tracing, capture, recorded):
    _, client = sync_client(capture)

    client.messages.create(**make_request())

    attrs = tracing.attrs()
    assert [key for key in CAPTURED if key in attrs] == recorded
    assert [key for key in CAPTURED if key in tracing.event()] == recorded
    assert attrs["gen_ai.usage.input_tokens"] == 16


def test_sdk_placeholders_for_missing_parameters_are_left_out(tracing):
    class NotGiven:
        def __bool__(self) -> bool:
            return False

    _, client = sync_client()

    client.messages.create(**make_request(temperature=NotGiven(), system=NotGiven()))

    attrs = tracing.attrs()
    assert "gen_ai.request.temperature" not in attrs
    assert SYSTEM not in attrs


def test_async_create_records_the_call(tracing):
    _, client = async_client()

    message = asyncio.run(client.messages.create(**make_request()))

    assert message.id == "msg_1"
    span = tracing.span()
    assert span.status.status_code == StatusCode.OK
    assert span.attributes["gen_ai.operation.source_name"] == "anthropic.messages.create"
    assert tracing.json(INPUT) == EXPECTED_INPUT
    assert tracing.json(OUTPUT) == EXPECTED_OUTPUT
    assert {k: v for k, v in tracing.attrs().items() if k in EXPECTED_USAGE} == EXPECTED_USAGE


@pytest.mark.parametrize("is_async", [False, True])
def test_errors_are_recorded_on_the_span_and_raised(tracing, is_async):
    mock, client = async_client() if is_async else sync_client()
    mock.messages.error = RuntimeError("overloaded")

    with pytest.raises(RuntimeError, match="overloaded"):
        if is_async:
            asyncio.run(client.messages.create(**make_request()))
        else:
            client.messages.create(**make_request())

    span = tracing.span()
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "overloaded"
    assert tracing.event()["exception.type"] == "RuntimeError"
    [exception] = span.events
    assert exception.name == "exception"
    assert tracing.event()["span.status_code"] == "ERROR"
    # The request is still recorded.
    assert tracing.json(INPUT) == EXPECTED_INPUT


# ---------- messages.create(stream=True) ----------


def test_create_stream_records_the_same_as_a_call_that_is_not_streamed(tracing):
    _, client = sync_client()
    client.messages.create(**make_request())
    not_streamed = response_attrs(tracing.attrs())
    tracing.clear()

    stream = client.messages.create(**make_request(stream=True))
    events = iter(stream)
    first = next(events)
    assert first.type == "message_start"
    assert tracing.spans() == []

    received = [first, *events]

    assert len(received) == len(stream_events_for(RESPONSE))
    span = tracing.span()
    assert span.status.status_code == StatusCode.OK
    attrs = tracing.attrs()
    assert attrs["gen_ai.request.stream"] is True
    assert attrs["statsig.gen_ai.server.time_to_first_token_ms"] >= 0
    assert response_attrs(attrs) == not_streamed
    assert tracing.json(OUTPUT) == EXPECTED_OUTPUT
    assert tracing.event()["gen_ai.request.stream"] == "true"


def test_async_create_stream_records_the_message(tracing):
    _, client = async_client()

    async def read() -> int:
        stream = await client.messages.create(**make_request(stream=True))
        assert tracing.spans() == []
        return len([event async for event in stream])

    assert asyncio.run(read()) == len(stream_events_for(RESPONSE))
    assert tracing.span().status.status_code == StatusCode.OK
    assert tracing.json(OUTPUT) == EXPECTED_OUTPUT
    assert {k: v for k, v in tracing.attrs().items() if k in EXPECTED_USAGE} == EXPECTED_USAGE


def test_create_stream_passes_other_attributes_through_and_closes_the_stream(tracing):
    mock, client = sync_client()

    with client.messages.create(**make_request(stream=True)) as stream:
        assert stream.response.status_code == 200
        next(iter(stream))

    [raw_stream] = mock.messages.raw_streams
    assert raw_stream.closed
    # Like the Node SDK, a stream closed before its message ends the span without a status.
    span = tracing.span()
    assert span.status.status_code == StatusCode.UNSET
    assert OUTPUT not in tracing.attrs()


def test_create_stream_closed_after_the_message_stops_records_it(tracing):
    _, client = sync_client()

    stream = client.messages.create(**make_request(stream=True))
    for event in stream:
        if event.type == "message_stop":
            break
    stream.close()

    assert tracing.span().status.status_code == StatusCode.OK
    assert tracing.json(OUTPUT) == EXPECTED_OUTPUT


@pytest.mark.parametrize("is_async", [False, True])
def test_create_stream_errors_are_recorded_and_raised(tracing, is_async):
    mock, client = async_client() if is_async else sync_client()
    mock.messages.fail_stream_after = 3

    async def read_async() -> None:
        stream = await client.messages.create(**make_request(stream=True))
        async for _ in stream:
            pass

    with pytest.raises(ConnectionError, match="stream interrupted"):
        if is_async:
            asyncio.run(read_async())
        else:
            for _ in client.messages.create(**make_request(stream=True)):
                pass

    span = tracing.span()
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "stream interrupted"
    assert tracing.event()["exception.type"] == "ConnectionError"


# ---------- messages.stream ----------


@pytest.mark.parametrize(
    "read",
    [
        pytest.param(lambda stream: stream.get_final_message(), id="get_final_message"),
        pytest.param(lambda stream: "".join(stream.text_stream), id="text_stream"),
        pytest.param(list, id="iteration"),
    ],
)
def test_messages_stream_span_ends_once_the_stream_is_read(tracing, read):
    mock, client = sync_client()

    with client.messages.stream(**make_request()) as stream:
        assert tracing.spans() == []
        read(stream)
        # The span ends as soon as the stream is read, before the block exits.
        span = tracing.span()

    assert mock.messages.requests == [{**make_request(), "stream": True}]
    assert span.status.status_code == StatusCode.OK
    attrs = tracing.attrs()
    assert attrs["gen_ai.operation.source_name"] == "anthropic.messages.stream"
    assert attrs["gen_ai.request.stream"] is True
    assert attrs["gen_ai.response.id"] == "msg_1"
    assert {k: v for k, v in attrs.items() if k in EXPECTED_USAGE} == EXPECTED_USAGE
    assert tracing.json(INPUT) == EXPECTED_INPUT
    assert tracing.json(OUTPUT) == EXPECTED_OUTPUT


def test_messages_stream_closed_before_its_message_is_aborted(tracing):
    mock, client = sync_client()

    with client.messages.stream(**make_request()) as stream:
        next(iter(stream))

    [raw_stream] = mock.messages.raw_streams
    assert raw_stream.closed
    span = tracing.span()
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "Request was aborted."


def test_messages_stream_request_errors_are_recorded_and_raised(tracing):
    mock, client = sync_client()
    mock.messages.error = RuntimeError("bad model")

    with pytest.raises(RuntimeError, match="bad model"):
        with client.messages.stream(**make_request()):
            pass

    span = tracing.span()
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "bad model"
    assert span.attributes["gen_ai.operation.source_name"] == "anthropic.messages.stream"


def test_async_messages_stream(tracing):
    mock, client = async_client()

    async def run() -> None:
        async with client.messages.stream(**make_request()) as stream:
            assert tracing.spans() == []
            message = await stream.get_final_message()
            assert message.id == "msg_1"
            assert tracing.span().status.status_code == StatusCode.OK
        assert tracing.json(OUTPUT) == EXPECTED_OUTPUT
        tracing.clear()

        async with client.messages.stream(**make_request()) as stream:
            async for _ in stream:
                break
        assert tracing.span().status.description == "Request was aborted."
        tracing.clear()

        mock.messages.error = RuntimeError("bad model")
        with pytest.raises(RuntimeError, match="bad model"):
            async with client.messages.stream(**make_request()):
                pass
        assert tracing.span().status.description == "bad model"

    asyncio.run(run())
    assert all(raw_stream.closed for raw_stream in mock.messages.raw_streams)


# ---------- Span context ----------


def test_model_span_is_the_llm_root_outside_a_workflow_and_a_child_inside_one(tracing):
    _, client = sync_client()

    client.messages.create(**make_request())
    with start_workflow("support_turn"):
        client.messages.create(**make_request())

    root, inner, workflow = tracing.spans()
    assert root.parent is None
    assert root.attributes[STATSIG_ATTR_SPAN_LLM_ROOT] == "1"
    assert tracing.started.spans[0][STATSIG_ATTR_SPAN_LLM_ROOT] == "1"
    assert inner.parent.span_id == workflow.context.span_id
    assert STATSIG_ATTR_SPAN_LLM_ROOT not in inner.attributes
    root_event, inner_event, _ = [metadata for _, _, metadata in tracing.statsig.events]
    assert root_event[STATSIG_ATTR_SPAN_LLM_ROOT] == "1"
    assert STATSIG_ATTR_SPAN_LLM_ROOT not in inner_event


@pytest.mark.parametrize("is_async", [False, True])
def test_spans_started_during_the_call_are_children_of_the_model_span(tracing, is_async):
    mock, client = async_client() if is_async else sync_client()
    tracer = OtelSingleton.get_tracer_provider_static().get_tracer("test")
    mock.messages.on_call = lambda: tracer.start_span("http POST").end()

    if is_async:
        asyncio.run(client.messages.create(**make_request()))
    else:
        client.messages.create(**make_request())

    http, model = tracing.spans()
    assert http.parent.span_id == model.context.span_id


def test_model_span_and_event_get_the_statsig_context_and_environment(tracing):
    _, client = sync_client()
    user = StatsigUser(user_id="user-123", custom_ids={"org": "acme"})

    with statsig_environment("eval"):
        with statsig_context(StatsigContext(user=user, activity_id="activity-1")):
            client.messages.create(**make_request())

    attrs = tracing.attrs()
    assert attrs[STATSIG_ATTR_SPAN_TYPE] == "gen_ai"
    assert attrs[STATSIG_ATTR_USER_ID] == "user-123"
    assert json.loads(attrs[STATSIG_ATTR_CUSTOM_IDS]) == {
        "org": "acme",
        "statsig.activity_id": "activity-1",
    }
    assert attrs[STATSIG_ATTR_ENVIRONMENT] == "eval"
    [started] = tracing.started.spans
    for key in [STATSIG_ATTR_SPAN_TYPE, STATSIG_ATTR_USER_ID, STATSIG_ATTR_ENVIRONMENT]:
        assert started[key] == attrs[key]
        assert tracing.event()[key] == attrs[key]


class FakeDynamicConfig:
    def __init__(self, name: str, values: Dict[str, Any]) -> None:
        self.name = name
        self._values = values

    def get_string(self, key: str, fallback: str) -> str:
        return self._values.get(key, fallback)

    def get_array(self, key: str, fallback: list) -> list:
        return self._values.get(key, fallback)


@pytest.mark.parametrize(
    "capture, records_variables",
    [(GenAICaptureOptions(capture_input_messages=True), True), (None, False)],
)
def test_calls_inside_a_prompt_scope_are_tagged_with_the_version(
    tracing, capture, records_variables
):
    config = FakeDynamicConfig(
        "prompt_config:version-a",
        {"name": "v2", "prompts": [{"role": "user", "content": "Answer: {{input}}"}]},
    )
    prompt = Prompt("support_agent", PromptVersion(config, False), [])  # type: ignore[arg-type]
    _, client = sync_client(capture)

    with prompt.with_live() as version:
        client.messages.create(
            model="claude-test",
            max_tokens=16,
            messages=version.get_prompt_messages({"input": "Hi"}),
        )

    attrs = tracing.attrs()
    assert attrs[STATSIG_ATTR_LLM_PROMPT_NAME] == "support_agent"
    assert attrs[STATSIG_ATTR_LLM_PROMPT_VERSION] == "v2"
    assert attrs[STATSIG_ATTR_LLM_PROMPT_VERSION_ID] == "version-a"
    if records_variables:
        assert json.loads(attrs[STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES]) == {"input": "Hi"}
    else:
        assert STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES not in attrs


# ---------- Wrapping ----------


def test_other_client_attributes_pass_through(tracing):
    _, client = sync_client()

    assert client.api_key == "test-key"
    assert client.messages.count_tokens(model="claude-test", messages=[]) == {"input_tokens": 3}
    assert tracing.spans() == []


def test_unsupported_objects_are_returned_unchanged(caplog):
    not_a_client = SDKObject({"chat": {}})

    with caplog.at_level(logging.WARNING):
        assert wrap_anthropic(not_a_client) is not_a_client

    assert "not wrapping" in caplog.text


def test_the_sdk_imports_without_anthropic_installed():
    code = (
        "import sys; sys.modules['anthropic'] = None; "
        "from statsig_ai import wrap_anthropic; "
        "assert 'anthropic' not in [m for m in sys.modules if sys.modules[m] is not None]"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)


# ---------- Message conversion ----------


def test_input_content_blocks_are_converted():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "x"},
                },
                {"type": "document", "source": {"type": "content", "content": []}},
                {"type": "text", "text": "What's this?"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "data": "secret"},
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"},
                {"type": "tool_result", "tool_use_id": "toolu_2"},
            ],
        },
        # Tool results next to other content stay a user message.
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_3", "content": "ok"},
                {"type": "text", "text": "Thanks"},
            ],
        },
    ]

    assert to_anthropic_input_messages(messages) == [
        {
            "role": "user",
            "parts": [
                {"type": "uri", "modality": "image", "uri": "https://example.com/a.png"},
                {"type": "blob", "modality": "image", "mime_type": "image/png"},
                {"type": "blob", "modality": "document"},
                {"type": "text", "content": "What's this?"},
            ],
        },
        {
            "role": "assistant",
            "parts": [
                {"type": "tool_call", "id": "srvtoolu_1", "name": "web_search", "arguments": {}}
            ],
        },
        {
            "role": "tool",
            "parts": [
                {"type": "tool_call_response", "id": "toolu_1", "response": "42"},
                {"type": "tool_call_response", "id": "toolu_2", "response": ""},
            ],
        },
        {
            "role": "user",
            "parts": [
                {"type": "tool_call_response", "id": "toolu_3", "response": "ok"},
                {"type": "text", "content": "Thanks"},
            ],
        },
    ]


@pytest.mark.parametrize(
    "stop_reason, finish_reason",
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_call"),
        ("refusal", "content_filter"),
        ("pause_turn", "pause_turn"),
        (None, "stop"),
    ],
)
def test_stop_reasons_map_to_finish_reasons(stop_reason, finish_reason):
    assert to_anthropic_finish_reason(stop_reason) == finish_reason


def test_a_streamed_tool_call_without_arguments_keeps_its_empty_input():
    message = aggregate_anthropic_stream_events(
        [
            {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 3}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "now", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": ""},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 2},
            },
        ]
    )

    assert message == {
        "id": "msg_1",
        "model": None,
        "usage": {"input_tokens": 3, "output_tokens": 2},
        "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "now", "input": {}}],
    }
