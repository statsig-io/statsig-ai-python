"""Runs wrap_anthropic on the real Anthropic SDK, with a mocked HTTP transport.

Tracing messages.stream() relies on the SDK's MessageStream reading its events from its
`_raw_stream`, so these tests catch an SDK release that changes that. Skipped when `anthropic`
isn't installed.
"""

import asyncio
import json
from typing import Any, Dict, List

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from statsig_ai import GenAICaptureOptions, WrapOpenAIOptions, wrap_anthropic
from statsig_ai.otel.singleton import OtelSingleton
from statsig_ai.wrappers import span_telemetry

anthropic = pytest.importorskip("anthropic")
try:
    import httpx2 as httpx  # The HTTP client of anthropic 2.x.
except ImportError:
    import httpx  # type: ignore[no-redef]

EVENTS: List[Dict[str, Any]] = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 7, "cache_read_input_tokens": 3, "output_tokens": 1},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Check"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ing."}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_gate", "input": {}},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": '{"id": "chec'},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": 'kout"}'},
    },
    {"type": "content_block_stop", "index": 1},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 9},
    },
    {"type": "message_stop"},
]
MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-test",
    "content": [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "toolu_1", "name": "get_gate", "input": {"id": "checkout"}},
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 7, "cache_read_input_tokens": 3, "output_tokens": 9},
}
EXPECTED_OUTPUT = [
    {
        "role": "assistant",
        "parts": [
            {"type": "text", "content": "Checking."},
            {
                "type": "tool_call",
                "id": "toolu_1",
                "name": "get_gate",
                "arguments": {"id": "checkout"},
            },
        ],
        "finish_reason": "tool_call",
    }
]
PARAMS: Dict[str, Any] = {
    "model": "claude-test",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "Is checkout on?"}],
}


def respond(request: Any) -> Any:
    body = json.loads(request.content)
    if body["model"] == "bad-model":
        error = {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
        return httpx.Response(400, json=error)
    if body.get("stream"):
        sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in EVENTS)
        return httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )
    return httpx.Response(200, json=MESSAGE)


OPTIONS = WrapOpenAIOptions(GenAICaptureOptions(capture_all=True))


def sync_client() -> Any:
    http_client = httpx.Client(transport=httpx.MockTransport(respond))
    return wrap_anthropic(
        anthropic.Anthropic(api_key="test-key", max_retries=0, http_client=http_client), OPTIONS
    )


def async_client() -> Any:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    return wrap_anthropic(
        anthropic.AsyncAnthropic(api_key="test-key", max_retries=0, http_client=http_client),
        OPTIONS,
    )


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    original_instance = OtelSingleton._instance
    OtelSingleton._instance = OtelSingleton(provider)
    monkeypatch.setattr(span_telemetry, "get_statsig_instance_for_logging", lambda: None)

    yield exporter

    OtelSingleton._instance = original_instance
    provider.shutdown()


def only_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    [span] = exporter.get_finished_spans()
    exporter.clear()
    return span


def assert_records_the_message(span: ReadableSpan, source_name: str) -> None:
    attrs = span.attributes or {}
    assert span.status.status_code == StatusCode.OK
    assert attrs["gen_ai.operation.source_name"] == source_name
    assert attrs["gen_ai.response.id"] == "msg_1"
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 9
    assert attrs["statsig.gen_ai.usage.input_cached_tokens"] == 3
    assert json.loads(str(attrs["gen_ai.output.messages"])) == EXPECTED_OUTPUT


def test_sync_client(exporter):
    client = sync_client()

    assert client.messages.create(**PARAMS).id == "msg_1"
    assert_records_the_message(only_span(exporter), "anthropic.messages.create")

    with client.messages.create(**PARAMS, stream=True) as stream:
        assert stream.response.status_code == 200
        assert len(list(stream)) == len(EVENTS)
    assert_records_the_message(only_span(exporter), "anthropic.messages.create")

    with client.messages.stream(**PARAMS) as message_stream:
        assert message_stream.get_final_message().content[1].input == {"id": "checkout"}
    assert_records_the_message(only_span(exporter), "anthropic.messages.stream")

    with client.messages.stream(**PARAMS) as message_stream:
        assert "".join(message_stream.text_stream) == "Checking."
    assert_records_the_message(only_span(exporter), "anthropic.messages.stream")

    with client.messages.stream(**PARAMS) as message_stream:
        next(iter(message_stream))
    assert only_span(exporter).status.description == "Request was aborted."

    with pytest.raises(anthropic.BadRequestError):
        with client.messages.stream(**{**PARAMS, "model": "bad-model"}):
            pass
    assert only_span(exporter).status.status_code == StatusCode.ERROR


def test_async_client(exporter):
    client = async_client()

    async def run() -> None:
        assert (await client.messages.create(**PARAMS)).id == "msg_1"
        assert_records_the_message(only_span(exporter), "anthropic.messages.create")

        stream = await client.messages.create(**PARAMS, stream=True)
        assert len([event async for event in stream]) == len(EVENTS)
        assert_records_the_message(only_span(exporter), "anthropic.messages.create")

        async with client.messages.stream(**PARAMS) as message_stream:
            message = await message_stream.get_final_message()
            assert message.content[0].text == "Checking."
        assert_records_the_message(only_span(exporter), "anthropic.messages.stream")

        with pytest.raises(anthropic.BadRequestError):
            await client.messages.create(**{**PARAMS, "model": "bad-model"})
        assert only_span(exporter).status.status_code == StatusCode.ERROR

    asyncio.run(run())
