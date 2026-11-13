import asyncio
import inspect
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import pytest
from opentelemetry import context as otel_context
from opentelemetry import trace
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
    wrap_gemini,
)
from statsig_ai.otel.conventions import (
    STATSIG_ATTR_ACTIVITY_ID,
    STATSIG_ATTR_CUSTOM_IDS,
    STATSIG_ATTR_ENVIRONMENT,
    STATSIG_ATTR_LLM_PROMPT_NAME,
    STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES,
    STATSIG_ATTR_LLM_PROMPT_VERSION,
    STATSIG_ATTR_LLM_PROMPT_VERSION_ID,
    STATSIG_ATTR_SPAN_LLM_ROOT,
    STATSIG_ATTR_SPAN_TYPE,
    STATSIG_ATTR_USER_ID,
    STATSIG_CTX_KEY_LLM_ROOT,
)
from statsig_ai.otel.singleton import OtelSingleton
from statsig_ai.wrappers import span_telemetry

from mock_gemini import (
    Behavior,
    Blob,
    Candidate,
    Content,
    FileData,
    FinishReason,
    FunctionCall,
    FunctionResponse,
    GenerateContentResponse,
    MockGeminiClient,
    Part,
    UsageMetadata,
    text_response,
)

INPUT = "gen_ai.input.messages"
OUTPUT = "gen_ai.output.messages"
SYSTEM = "gen_ai.system_instructions"
TOOLS = "gen_ai.tool.definitions"
CAPTURED_ATTRS = [INPUT, OUTPUT, SYSTEM, TOOLS]
CONTEXT_KEYS = [STATSIG_ATTR_USER_ID, STATSIG_ATTR_CUSTOM_IDS, STATSIG_ATTR_ENVIRONMENT]
CAPTURE_ALL = GenAICaptureOptions(capture_all=True)


class StartAttributesRecorder(SpanProcessor):
    def __init__(self) -> None:
        self.by_span: Dict[str, Dict[str, Any]] = {}

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        self.by_span[span.name] = dict(span.attributes or {})


class FakeStatsig:
    def __init__(self) -> None:
        self.logged: List[Tuple[str, str, Dict[str, str]]] = []

    def log_event(self, _user, event_name, value, metadata) -> None:
        self.logged.append((event_name, value, metadata))

    @property
    def events(self) -> List[Dict[str, str]]:
        return [metadata for _, _, metadata in self.logged]


class Tracing:
    def __init__(self, exporter: InMemorySpanExporter, started: StartAttributesRecorder) -> None:
        self.exporter = exporter
        self.started = started
        self.statsig = FakeStatsig()

    def spans(self) -> List[ReadableSpan]:
        return list(self.exporter.get_finished_spans())

    def only_span(self) -> ReadableSpan:
        spans = self.spans()
        assert len(spans) == 1
        assert len(self.statsig.events) == 1
        return spans[0]

    def attrs(self) -> Dict[str, Any]:
        return dict(self.only_span().attributes or {})

    def json(self, key: str) -> Any:
        """Reads a JSON attribute from the span, and checks the gen_ai event has the same."""
        attrs = self.attrs()
        assert self.statsig.events[0][key] == attrs[key]
        return json.loads(attrs[key])


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


def wrapped(
    behavior: Behavior, capture: Optional[GenAICaptureOptions] = CAPTURE_ALL
) -> Tuple[Any, MockGeminiClient]:
    client = MockGeminiClient(behavior)
    return wrap_gemini(client, WrapOpenAIOptions(capture)), client


class FakeDynamicConfig:
    def __init__(self, name: str, values: Dict[str, Any]) -> None:
        self.name = name
        self._values = values

    def get_string(self, key: str, fallback: str) -> str:
        return self._values.get(key, fallback)

    def get_array(self, key: str, fallback: list) -> list:
        return self._values.get(key, fallback)


def make_version(version_id: str, name: str) -> PromptVersion:
    config = FakeDynamicConfig(
        f"prompt_config:{version_id}",
        {"name": name, "prompts": [{"role": "user", "content": "Answer: {{input}}"}]},
    )
    return PromptVersion(config, False)  # type: ignore[arg-type]


def make_prompt() -> Prompt:
    return Prompt(
        "support_agent", make_version("version-2", "v2"), [], StatsigUser(user_id="user-1")
    )


FUNCTION_CALL_RESPONSE = GenerateContentResponse(
    response_id="resp_1",
    model_version="gemini-test-001",
    candidates=[
        Candidate(
            content=Content(
                role="model",
                parts=[Part(function_call=FunctionCall(name="get_gate", args={"id": "checkout"}))],
            ),
            finish_reason=FinishReason.STOP,
        )
    ],
    usage_metadata=UsageMetadata(
        prompt_token_count=9, candidates_token_count=3, thoughts_token_count=2
    ),
)

FUNCTION_CALL_CONTENTS = [
    Content(role="user", parts=[Part(text="Is checkout on?")]),
    Content(role="model", parts=[Part(function_call=FunctionCall(name="list_gates", args={}))]),
    Content(
        role="user",
        parts=[
            Part(
                function_response=FunctionResponse(
                    name="list_gates", response={"gates": ["checkout"]}
                )
            )
        ],
    ),
]

EXPECTED_FUNCTION_CALL_INPUT = [
    {"role": "user", "parts": [{"type": "text", "content": "Is checkout on?"}]},
    {
        "role": "assistant",
        "parts": [{"type": "tool_call", "id": "list_gates", "name": "list_gates", "arguments": {}}],
    },
    {
        "role": "tool",
        "parts": [
            {"type": "tool_call_response", "id": "list_gates", "response": {"gates": ["checkout"]}}
        ],
    },
]

EXPECTED_FUNCTION_CALL_OUTPUT = [
    {
        "role": "assistant",
        "parts": [
            {
                "type": "tool_call",
                "id": "get_gate",
                "name": "get_gate",
                "arguments": {"id": "checkout"},
            }
        ],
        "finish_reason": "tool_call",
    }
]

STREAM_CHUNKS = [
    GenerateContentResponse(
        response_id="resp_2",
        model_version="gemini-test-001",
        candidates=[Candidate(content=Content(role="model", parts=[Part(text="Checkout ")]))],
    ),
    GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(role="model", parts=[Part(text="is on.")]),
                finish_reason=FinishReason.STOP,
            )
        ],
        usage_metadata=UsageMetadata(prompt_token_count=4, candidates_token_count=3),
    ),
]

EXPECTED_STREAM_OUTPUT = [
    {
        "role": "assistant",
        "parts": [{"type": "text", "content": "Checkout is on."}],
        "finish_reason": "stop",
    }
]


# generate_content


def test_generate_content_records_a_function_calling_turn_in_the_gen_ai_format(tracing):
    client, mock = wrapped(Behavior(response=FUNCTION_CALL_RESPONSE))
    config = {"system_instruction": "Be brief.", "temperature": 0.1, "max_output_tokens": 64}

    with make_prompt().with_live():
        response = client.models.generate_content(
            model="gemini-test", contents=FUNCTION_CALL_CONTENTS, config=config
        )

    assert response is FUNCTION_CALL_RESPONSE
    assert mock.behavior.calls == [
        {"model": "gemini-test", "contents": FUNCTION_CALL_CONTENTS, "config": config}
    ]
    span = tracing.only_span()
    assert span.name == "generate_content gemini-test"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.OK
    assert tracing.attrs() == {
        **tracing.attrs(),
        "gen_ai.provider.name": "gcp.gen_ai",
        "gen_ai.operation.name": "generate_content",
        "gen_ai.operation.source_name": "google.models.generate_content",
        "gen_ai.request.model": "gemini-test",
        "gen_ai.request.temperature": 0.1,
        "gen_ai.request.max_tokens": 64,
        "gen_ai.response.id": "resp_1",
        "gen_ai.response.model": "gemini-test-001",
        "gen_ai.usage.input_tokens": 9,
        "gen_ai.usage.output_tokens": 5,
        "statsig.gen_ai.usage.total_tokens": 14,
        "statsig.gen_ai.usage.output_reasoning_tokens": 2,
        STATSIG_ATTR_LLM_PROMPT_VERSION_ID: "version-2",
    }
    assert "gen_ai.request.stream" not in tracing.attrs()
    assert "statsig.gen_ai.usage.input_cached_tokens" not in tracing.attrs()
    assert tracing.json("gen_ai.response.finish_reasons") == ["STOP"]
    assert tracing.json(SYSTEM) == [{"type": "text", "content": "Be brief."}]
    assert tracing.json(INPUT) == EXPECTED_FUNCTION_CALL_INPUT
    assert tracing.json(OUTPUT) == EXPECTED_FUNCTION_CALL_OUTPUT


def test_cached_content_tokens_are_recorded_as_cached_input_tokens(tracing):
    response = text_response("ok")
    response.usage_metadata = UsageMetadata(
        prompt_token_count=120,
        candidates_token_count=8,
        cached_content_token_count=100,
        total_token_count=128,
    )
    client, _ = wrapped(Behavior(response=response))

    client.models.generate_content(model="gemini-test", contents="Hi")

    attrs = tracing.attrs()
    assert attrs["gen_ai.usage.input_tokens"] == 120
    assert attrs["gen_ai.usage.output_tokens"] == 8
    assert attrs["statsig.gen_ai.usage.total_tokens"] == 128
    assert attrs["statsig.gen_ai.usage.input_cached_tokens"] == 100
    assert "statsig.gen_ai.usage.output_reasoning_tokens" not in attrs
    [metadata] = tracing.statsig.events
    assert metadata["statsig.gen_ai.usage.input_cached_tokens"] == "100"


def test_a_response_without_usage_records_no_token_counts(tracing):
    response = text_response("ok")
    response.usage_metadata = None
    client, _ = wrapped(Behavior(response=response))

    client.models.generate_content(model="gemini-test", contents="Hi")

    assert not [key for key in tracing.attrs() if "usage" in key]


def test_the_request_config_is_recorded(tracing):
    client, _ = wrapped(Behavior(response=text_response("ok")))

    client.models.generate_content(
        model="gemini-test",
        contents="Hi",
        config={"topP": 0.9, "top_k": 40, "stop_sequences": ["END"], "seed": 7},
    )

    attrs = tracing.attrs()
    assert attrs["gen_ai.request.top_p"] == 0.9
    assert attrs["gen_ai.request.top_k"] == 40
    assert attrs["gen_ai.request.seed"] == 7
    assert json.loads(attrs["gen_ai.request.stop_sequences"]) == ["END"]


def test_a_call_without_a_model_is_named_unknown(tracing):
    mock = MockGeminiClient()
    mock.models.generate_content = lambda **kwargs: text_response("ok")  # type: ignore

    wrap_gemini(mock).models.generate_content(contents="Hi")

    assert tracing.only_span().name == "generate_content unknown"


# generate_content_stream


def test_generate_content_stream_merges_chunks_and_ends_the_span_once_consumed(tracing):
    client, _ = wrapped(Behavior(chunks=STREAM_CHUNKS))

    stream = client.models.generate_content_stream(model="gemini-test", contents="Is checkout on?")
    first = next(stream)
    assert first is STREAM_CHUNKS[0]
    assert tracing.spans() == []
    assert list(stream) == STREAM_CHUNKS[1:]

    span = tracing.only_span()
    assert span.name == "generate_content gemini-test"
    assert span.status.status_code == StatusCode.OK
    attrs = tracing.attrs()
    assert attrs == {
        **attrs,
        "gen_ai.operation.source_name": "google.models.generate_content_stream",
        "gen_ai.request.stream": True,
        "gen_ai.response.id": "resp_2",
        "gen_ai.response.model": "gemini-test-001",
        "gen_ai.usage.input_tokens": 4,
        "gen_ai.usage.output_tokens": 3,
    }
    assert "statsig.gen_ai.server.time_to_first_token_ms" in attrs
    assert tracing.json("gen_ai.response.finish_reasons") == ["STOP"]
    assert tracing.json(INPUT) == [
        {"role": "user", "parts": [{"type": "text", "content": "Is checkout on?"}]}
    ]
    assert tracing.json(OUTPUT) == EXPECTED_STREAM_OUTPUT


def test_a_stream_keeps_thoughts_text_and_function_calls_apart_for_each_candidate(tracing):
    chunks = [
        GenerateContentResponse(
            candidates=[
                Candidate(content=Content(parts=[Part(text="Let me ", thought=True)])),
                Candidate(content=Content(parts=[Part(text="Second")])),
            ]
        ),
        GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(
                        parts=[Part(text="check.", thought=True), Part(text="Checking")]
                    )
                ),
                Candidate(
                    content=Content(parts=[Part(text=" answer")]),
                    finish_reason=FinishReason.MAX_TOKENS,
                ),
            ]
        ),
        GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(
                        parts=[
                            Part(text="..."),
                            Part(function_call=FunctionCall(id="call_1", name="get_gate")),
                            Part(text="Done"),
                        ]
                    ),
                    finish_reason=FinishReason.STOP,
                )
            ]
        ),
    ]
    client, _ = wrapped(Behavior(chunks=chunks))

    for _ in client.models.generate_content_stream(model="gemini-test", contents="Hi"):
        pass

    assert tracing.json("gen_ai.response.finish_reasons") == ["STOP", "MAX_TOKENS"]
    assert tracing.json(OUTPUT) == [
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "Let me check."},
                {"type": "text", "content": "Checking..."},
                {"type": "tool_call", "id": "call_1", "name": "get_gate"},
                {"type": "text", "content": "Done"},
            ],
            "finish_reason": "tool_call",
        },
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "Second answer"}],
            "finish_reason": "length",
        },
    ]


def test_a_stream_the_caller_never_iterates_sends_no_request_and_records_no_span(tracing):
    client, mock = wrapped(Behavior(chunks=STREAM_CHUNKS))

    client.models.generate_content_stream(model="gemini-test", contents="Hi")

    assert mock.behavior.calls == []
    assert tracing.spans() == []


# Async


def test_async_generate_content_ends_the_span_once_awaited(tracing):
    client, mock = wrapped(Behavior(response=FUNCTION_CALL_RESPONSE))

    assert inspect.iscoroutinefunction(client.aio.models.generate_content)
    pending = client.aio.models.generate_content(
        model="gemini-test", contents=FUNCTION_CALL_CONTENTS
    )
    assert tracing.spans() == []
    response = asyncio.run(pending)

    assert response is FUNCTION_CALL_RESPONSE
    assert len(mock.behavior.calls) == 1
    span = tracing.only_span()
    assert span.name == "generate_content gemini-test"
    assert span.status.status_code == StatusCode.OK
    assert tracing.attrs()["gen_ai.operation.source_name"] == "google.models.generate_content"
    assert tracing.attrs()["statsig.gen_ai.usage.output_reasoning_tokens"] == 2
    assert tracing.json(INPUT) == EXPECTED_FUNCTION_CALL_INPUT
    assert tracing.json(OUTPUT) == EXPECTED_FUNCTION_CALL_OUTPUT


def test_async_generate_content_stream_merges_chunks_and_ends_the_span_once_consumed(tracing):
    client, _ = wrapped(Behavior(chunks=STREAM_CHUNKS))
    received: List[Any] = []

    async def consume() -> None:
        stream = await client.aio.models.generate_content_stream(
            model="gemini-test", contents="Is checkout on?"
        )
        async for chunk in stream:
            received.append(chunk)
            assert tracing.spans() == []

    asyncio.run(consume())

    assert received == STREAM_CHUNKS
    span = tracing.only_span()
    assert span.status.status_code == StatusCode.OK
    attrs = tracing.attrs()
    assert attrs["gen_ai.operation.source_name"] == "google.models.generate_content_stream"
    assert attrs["gen_ai.request.stream"] is True
    assert attrs["gen_ai.response.id"] == "resp_2"
    assert attrs["gen_ai.usage.output_tokens"] == 3
    assert tracing.json(OUTPUT) == EXPECTED_STREAM_OUTPUT


# Capture options


def test_messages_are_not_recorded_by_default(tracing):
    client, _ = wrapped(Behavior(response=FUNCTION_CALL_RESPONSE), capture=None)

    client.models.generate_content(
        model="gemini-test",
        contents=FUNCTION_CALL_CONTENTS,
        config={"system_instruction": "Be brief.", "tools": [{"function_declarations": []}]},
    )

    attrs = tracing.attrs()
    [metadata] = tracing.statsig.events
    for key in CAPTURED_ATTRS:
        assert key not in attrs
        assert key not in metadata
    assert attrs["gen_ai.usage.input_tokens"] == 9


@pytest.mark.parametrize(
    "capture, recorded",
    [
        (GenAICaptureOptions(capture_input_messages=True), INPUT),
        (GenAICaptureOptions(capture_output_messages=True), OUTPUT),
        (GenAICaptureOptions(capture_system_instructions=True), SYSTEM),
        (GenAICaptureOptions(capture_tool_definitions=True), TOOLS),
    ],
    ids=["input", "output", "system", "tools"],
)
def test_each_capture_option_records_only_its_attribute(tracing, capture, recorded):
    client, _ = wrapped(Behavior(response=FUNCTION_CALL_RESPONSE), capture=capture)

    client.models.generate_content(
        model="gemini-test",
        contents=FUNCTION_CALL_CONTENTS,
        config={"system_instruction": "Be brief.", "tools": [{"function_declarations": []}]},
    )

    attrs = tracing.attrs()
    assert [key for key in CAPTURED_ATTRS if key in attrs] == [recorded]


def test_system_instructions_are_left_out_when_the_request_has_none(tracing):
    client, _ = wrapped(Behavior(response=text_response("ok")))

    client.models.generate_content(model="gemini-test", contents="Hi", config={"temperature": 0})

    assert SYSTEM not in tracing.attrs()
    assert TOOLS not in tracing.attrs()


def get_weather(city: str) -> str:
    return f"Sunny in {city}"


class ToolObject:
    """Like the SDK's pydantic Tool."""

    def model_dump(self, mode: str, exclude_none: bool) -> Dict[str, Any]:
        assert (mode, exclude_none) == ("json", True)
        return {"function_declarations": [{"name": "get_gate"}]}


def test_tool_definitions_are_recorded_as_json(tracing):
    client, _ = wrapped(Behavior(response=text_response("ok")))

    client.models.generate_content(
        model="gemini-test",
        contents="Hi",
        config={"tools": [ToolObject(), get_weather, {"google_search": {}}]},
    )

    assert tracing.json(TOOLS) == [
        {"function_declarations": [{"name": "get_gate"}]},
        "get_weather",
        {"google_search": {}},
    ]


@pytest.mark.parametrize(
    "capture, includes_variables",
    [
        (GenAICaptureOptions(capture_input_messages=True), True),
        (GenAICaptureOptions(capture_all=True), True),
        (GenAICaptureOptions(), False),
    ],
    ids=["capture_input_messages", "capture_all", "off"],
)
def test_the_span_is_tagged_with_the_active_prompt_version(tracing, capture, includes_variables):
    client, _ = wrapped(Behavior(response=text_response("ok")), capture=capture)

    with make_prompt().with_version(make_version("version-3", "v3")) as version:
        [message] = version.get_prompt_messages({"input": "Hi"})
        client.models.generate_content(
            model="gemini-test",
            contents=[{"role": "user", "parts": [{"text": message["content"]}]}],
        )

    attrs = tracing.attrs()
    assert attrs[STATSIG_ATTR_LLM_PROMPT_NAME] == "support_agent"
    assert attrs[STATSIG_ATTR_LLM_PROMPT_VERSION] == "v3"
    assert attrs[STATSIG_ATTR_LLM_PROMPT_VERSION_ID] == "version-3"
    if includes_variables:
        assert tracing.json(STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES) == {"input": "Hi"}
    else:
        assert STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES not in attrs


# custom_attributes and max_json_chars


def test_custom_attributes_are_set_on_every_span_and_event(tracing):
    client = wrap_gemini(
        MockGeminiClient(Behavior(response=text_response("ok"), chunks=STREAM_CHUNKS)),
        WrapOpenAIOptions(
            custom_attributes={"app.team": "growth", "app.retries": 2, "gen_ai.provider.name": "x"}
        ),
    )

    client.models.generate_content(model="gemini-call", contents="Hi")
    list(client.models.generate_content_stream(model="gemini-stream", contents="Hi"))
    asyncio.run(client.aio.models.generate_content(model="gemini-async", contents="Hi"))

    spans = tracing.spans()
    assert len(spans) == 3
    for span, event in zip(spans, tracing.statsig.events):
        attrs = dict(span.attributes or {})
        assert attrs["app.team"] == "growth"
        assert attrs["app.retries"] == 2
        # The wrapper's own attributes win, as in the Node SDK.
        assert attrs["gen_ai.provider.name"] == "gcp.gen_ai"
        assert tracing.started.by_span[span.name]["app.team"] == "growth"
        assert event["app.team"] == "growth"
        assert event["app.retries"] == "2"


def test_max_json_chars_caps_json_attributes(tracing):
    content = "x" * 100
    client = wrap_gemini(
        MockGeminiClient(Behavior(response=text_response("ok"))),
        WrapOpenAIOptions(GenAICaptureOptions(capture_input_messages=True), max_json_chars=50),
    )

    client.models.generate_content(model="gemini-test", contents=content)

    full = json.dumps(
        [{"role": "user", "parts": [{"type": "text", "content": content}]}], separators=(",", ":")
    )
    attrs = tracing.attrs()
    assert attrs[INPUT] == full[:50] + "…(truncated)"
    assert attrs[f"{INPUT}_len"] == len(full)
    assert tracing.statsig.events[0][INPUT] == attrs[INPUT]


# Errors


class GeminiAPIError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def assert_failed(tracing: Tracing, error: Exception) -> None:
    span = tracing.only_span()
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == str(error)
    [exception_event] = [event for event in span.events if event.name == "exception"]
    assert str((exception_event.attributes or {})["exception.type"]).endswith("GeminiAPIError")
    [metadata] = tracing.statsig.events
    assert metadata["span.status_code"] == "ERROR"
    assert metadata["exception.type"] == type(error).__name__
    assert metadata["exception.message"] == str(error)


def test_a_failed_call_records_the_error_and_raises_it_again(tracing):
    error = GeminiAPIError("429 RESOURCE_EXHAUSTED")
    client, _ = wrapped(Behavior(error=error))

    with pytest.raises(GeminiAPIError) as raised:
        client.models.generate_content(model="gemini-test", contents="Hi")

    assert raised.value is error
    assert_failed(tracing, error)
    assert "gen_ai.request.model" in tracing.attrs()


def test_a_failed_async_call_records_the_error_and_raises_it_again(tracing):
    error = GeminiAPIError("500 INTERNAL")
    client, _ = wrapped(Behavior(error=error))

    with pytest.raises(GeminiAPIError) as raised:
        asyncio.run(client.aio.models.generate_content(model="gemini-test", contents="Hi"))

    assert raised.value is error
    assert_failed(tracing, error)


def test_a_stream_that_fails_records_the_error_and_raises_it_again(tracing):
    error = GeminiAPIError("stream reset")
    client, _ = wrapped(Behavior(chunks=STREAM_CHUNKS[:1], stream_error=error))
    received: List[Any] = []

    with pytest.raises(GeminiAPIError) as raised:
        for chunk in client.models.generate_content_stream(model="gemini-test", contents="Hi"):
            received.append(chunk)

    assert raised.value is error
    assert received == STREAM_CHUNKS[:1]
    assert_failed(tracing, error)
    assert OUTPUT not in tracing.attrs()


def test_an_async_stream_that_fails_records_the_error_and_raises_it_again(tracing):
    error = GeminiAPIError("stream reset")
    client, _ = wrapped(Behavior(chunks=STREAM_CHUNKS[:1], stream_error=error))

    async def consume() -> None:
        stream = await client.aio.models.generate_content_stream(model="gemini-test", contents="Hi")
        async for _ in stream:
            pass

    with pytest.raises(GeminiAPIError):
        asyncio.run(consume())

    assert_failed(tracing, error)


def test_a_response_that_cannot_be_read_still_returns_and_ends_the_span(tracing, caplog):
    class Unreadable:
        @property
        def candidates(self):
            raise RuntimeError("boom")

    response = Unreadable()
    client, _ = wrapped(Behavior(response=response))

    with caplog.at_level(logging.WARNING):
        assert client.models.generate_content(model="gemini-test", contents="Hi") is response

    assert tracing.only_span().status.status_code == StatusCode.OK
    assert "Failed to record Gen AI response attributes" in caplog.text


# Spans, context and the statsig::gen_ai event


def test_the_span_is_a_child_of_the_active_span_and_parents_spans_the_call_starts(tracing):
    tracer = trace.get_tracer("test", tracer_provider=OtelSingleton.get_tracer_provider_static())
    inner = Behavior(response=text_response("ok"))
    inner.on_call = lambda: tracer.start_span("http.request").end()
    client, _ = wrapped(inner)

    with start_workflow("support_turn") as workflow:
        client.models.generate_content(model="gemini-test", contents="Hi")
        asyncio.run(client.aio.models.generate_content(model="gemini-test", contents="Hi"))
        assert trace.get_current_span() is workflow

    spans = tracing.spans()
    by_name: Dict[str, List[ReadableSpan]] = {}
    for span in spans:
        by_name.setdefault(span.name, []).append(span)
    [root] = by_name["invoke_workflow"]
    model_spans = by_name["generate_content gemini-test"]
    requests = by_name["http.request"]
    assert len(model_spans) == 2
    for model_span, request in zip(model_spans, requests):
        assert model_span.parent is not None
        assert model_span.parent.span_id == root.context.span_id
        assert model_span.context.trace_id == root.context.trace_id
        assert request.parent is not None
        assert request.parent.span_id == model_span.context.span_id
        # The workflow span is the root of the turn.
        assert STATSIG_ATTR_SPAN_LLM_ROOT not in (model_span.attributes or {})


def test_model_span_and_event_get_the_statsig_context_and_environment(tracing):
    client, _ = wrapped(Behavior(response=text_response("ok")))
    user = StatsigUser(user_id="user-123", custom_ids={"org": "acme"})

    with statsig_environment("eval"):
        with statsig_context(StatsigContext(user=user, activity_id="activity-1")):
            client.models.generate_content(model="gemini-test", contents="Hi")

    span = tracing.only_span()
    attrs = dict(span.attributes or {})
    assert attrs[STATSIG_ATTR_SPAN_TYPE] == "gen_ai"
    assert attrs[STATSIG_ATTR_USER_ID] == "user-123"
    assert json.loads(attrs[STATSIG_ATTR_CUSTOM_IDS]) == {
        "org": "acme",
        STATSIG_ATTR_ACTIVITY_ID: "activity-1",
    }
    assert attrs[STATSIG_ATTR_ENVIRONMENT] == "eval"

    # The span starts with them, so span processors see them in on_start.
    started = tracing.started.by_span[span.name]
    [metadata] = tracing.statsig.events
    for key in [STATSIG_ATTR_SPAN_TYPE, *CONTEXT_KEYS]:
        assert started[key] == attrs[key]
        assert metadata[key] == attrs[key]


def test_model_span_and_event_skip_context_attributes_outside_any_scope(tracing):
    client, _ = wrapped(Behavior(response=text_response("ok")))

    client.models.generate_content(model="gemini-test", contents="Hi")

    attrs = tracing.attrs()
    [metadata] = tracing.statsig.events
    for key in CONTEXT_KEYS:
        assert key not in attrs
        assert key not in metadata


def test_the_model_span_is_the_llm_root_unless_the_context_has_a_root(tracing):
    client, _ = wrapped(Behavior(response=text_response("ok")))

    client.models.generate_content(model="gemini-root", contents="Hi")
    token = otel_context.attach(otel_context.set_value(STATSIG_CTX_KEY_LLM_ROOT, "invoke_workflow"))
    try:
        client.models.generate_content(model="gemini-inner", contents="Hi")
    finally:
        otel_context.detach(token)

    root, inner = tracing.spans()
    assert (root.attributes or {})[STATSIG_ATTR_SPAN_LLM_ROOT] == "1"
    assert tracing.started.by_span[root.name][STATSIG_ATTR_SPAN_LLM_ROOT] == "1"
    assert STATSIG_ATTR_SPAN_LLM_ROOT not in (inner.attributes or {})

    root_event, inner_event = tracing.statsig.events
    assert root_event[STATSIG_ATTR_SPAN_LLM_ROOT] == "1"
    assert STATSIG_ATTR_SPAN_LLM_ROOT not in inner_event


def test_each_call_logs_a_gen_ai_event_with_the_span_attributes(tracing):
    client, _ = wrapped(Behavior(response=FUNCTION_CALL_RESPONSE))

    client.models.generate_content(model="gemini-test", contents=FUNCTION_CALL_CONTENTS)

    span = tracing.only_span()
    [(event_name, value, metadata)] = tracing.statsig.logged
    assert event_name == "statsig::gen_ai"
    assert value == "generate_content gemini-test"
    for key, attr in (span.attributes or {}).items():
        expected = json.dumps(attr) if isinstance(attr, (list, tuple)) else str(attr)
        assert metadata[key] == expected, key
    assert metadata["gen_ai.provider.name"] == "gcp.gen_ai"
    assert metadata["span.trace_id"] == format(span.context.trace_id, "032x")
    assert metadata["span.span_id"] == format(span.context.span_id, "016x")
    assert metadata["span.status_code"] == "OK"


# The wrapped client


def test_other_client_attributes_are_passed_through(tracing):
    client, mock = wrapped(Behavior(response=text_response("ok")))

    assert client.files == "files-api"
    assert client.aio.files == "async-files-api"
    assert client.models.get(model="gemini-test") == "model:gemini-test"
    assert client.behavior is mock.behavior
    assert tracing.spans() == []


def test_a_client_without_aio_is_wrapped(tracing):
    mock = MockGeminiClient(Behavior(response=text_response("ok")))
    del mock.aio

    client = wrap_gemini(mock)
    client.models.generate_content(model="gemini-test", contents="Hi")

    assert not hasattr(client, "aio")
    assert tracing.only_span().name == "generate_content gemini-test"


@pytest.mark.parametrize("unsupported", [object(), None, {"models": {}}])
def test_an_unsupported_client_is_returned_unchanged_with_a_warning(unsupported, caplog):
    with caplog.at_level(logging.WARNING):
        assert wrap_gemini(unsupported) is unsupported

    assert "not wrapping" in caplog.text


# Real google-genai types, when the SDK is installed. The other tests don't need it.


def test_the_sdk_types_are_read_like_the_mocks(tracing):
    types = pytest.importorskip("google.genai.types")
    response = types.GenerateContentResponse(
        response_id="resp_1",
        model_version="gemini-test-001",
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(text="Checking.", thought=True),
                        types.Part(
                            function_call=types.FunctionCall(
                                name="get_gate", args={"id": "checkout"}
                            )
                        ),
                    ],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=20, candidates_token_count=3, cached_content_token_count=12
        ),
    )
    config = types.GenerateContentConfig(
        system_instruction=types.Content(parts=[types.Part(text="Be brief.")]),
        temperature=0.2,
        tools=[
            types.Tool(
                function_declarations=[types.FunctionDeclaration(name="get_gate", description="d")]
            )
        ],
    )
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text="Is checkout on?")]),
        types.Content(
            role="user",
            parts=[types.Part.from_function_response(name="list_gates", response={"gates": []})],
        ),
        types.Part.from_bytes(data=b"\x89PNG", mime_type="image/png"),
    ]
    client, _ = wrapped(Behavior(response=response))

    assert (
        client.models.generate_content(model="gemini-test", contents=contents, config=config)
        is response
    )

    attrs = tracing.attrs()
    assert attrs["gen_ai.request.temperature"] == 0.2
    assert attrs["statsig.gen_ai.usage.input_cached_tokens"] == 12
    assert tracing.json("gen_ai.response.finish_reasons") == ["STOP"]
    assert tracing.json(SYSTEM) == [{"type": "text", "content": "Be brief."}]
    assert tracing.json(TOOLS) == [
        {"function_declarations": [{"name": "get_gate", "description": "d"}]}
    ]
    assert tracing.json(INPUT) == [
        {"role": "user", "parts": [{"type": "text", "content": "Is checkout on?"}]},
        {
            "role": "tool",
            "parts": [
                {"type": "tool_call_response", "id": "list_gates", "response": {"gates": []}}
            ],
        },
        {
            "role": "user",
            "parts": [{"type": "blob", "modality": "image", "mime_type": "image/png"}],
        },
    ]
    assert tracing.json(OUTPUT) == [
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "Checking."},
                {
                    "type": "tool_call",
                    "id": "get_gate",
                    "name": "get_gate",
                    "arguments": {"id": "checkout"},
                },
            ],
            "finish_reason": "tool_call",
        }
    ]
