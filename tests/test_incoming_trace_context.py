from typing import Dict, Optional

import pytest
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
from werkzeug.datastructures import Headers

from statsig_ai import (
    IncomingHeaders,
    incoming_trace_context,
    statsig_environment,
    wrap_openai,
)
from statsig_ai.otel.conventions import STATSIG_ATTR_ENVIRONMENT
from statsig_ai.otel.environment_context import get_environment_span_attrs_from_context
from statsig_ai.otel.singleton import OtelSingleton

from mock_openai import MockOpenAI

TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
SPAN_ID = "b7ad6b7169203331"
TRACEPARENT = f"00-{TRACE_ID}-{SPAN_ID}-01"


class GetOnlyHeaders:
    """Only offers a case-insensitive `.get`, like some framework header objects."""

    def __init__(self, headers: Dict[str, str]):
        self._headers = {k.lower(): v for k, v in headers.items()}

    def get(self, key: str) -> Optional[str]:
        return self._headers.get(key.lower())


@pytest.fixture
def tracing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    original_instance = OtelSingleton._instance
    OtelSingleton._instance = OtelSingleton(provider)

    def finished_spans() -> Dict[str, ReadableSpan]:
        return {span.name: span for span in exporter.get_finished_spans()}

    yield provider.get_tracer("statsig-test"), finished_spans

    OtelSingleton._instance = original_instance
    provider.shutdown()


def trace_id(span: ReadableSpan) -> str:
    return trace.format_trace_id(span.context.trace_id)


def parent_span_id(span: ReadableSpan) -> Optional[str]:
    return None if span.parent is None else trace.format_span_id(span.parent.span_id)


def run_tool(tracer, headers: IncomingHeaders) -> None:
    with incoming_trace_context(headers):
        tracer.start_span("execute_tool").end()


def test_nests_workflow_tool_and_model_call_spans_under_the_incoming_span(tracing):
    tracer, finished_spans = tracing
    openai = wrap_openai(MockOpenAI())

    with incoming_trace_context({"traceparent": TRACEPARENT, "tracestate": "statsig=eval"}):
        with tracer.start_as_current_span("invoke_workflow"):
            tracer.start_span("execute_tool").end()
            openai.chat.completions.create(
                model="gpt-4", messages=[{"role": "user", "content": "hi"}]
            )

    spans = finished_spans()
    assert set(spans) == {"invoke_workflow", "execute_tool", "chat gpt-4"}
    for span in spans.values():
        assert trace_id(span) == TRACE_ID

    workflow = spans["invoke_workflow"]
    assert parent_span_id(workflow) == SPAN_ID
    assert workflow.parent is not None and workflow.parent.is_remote
    assert workflow.context.trace_state.to_header() == "statsig=eval"

    workflow_span_id = trace.format_span_id(workflow.context.span_id)
    assert parent_span_id(spans["execute_tool"]) == workflow_span_id
    assert parent_span_id(spans["chat gpt-4"]) == workflow_span_id


def test_reads_mixed_case_keys_and_list_values(tracing):
    tracer, finished_spans = tracing

    run_tool(
        tracer,
        {
            "Traceparent": [TRACEPARENT, f"00-{'1' * 32}-{'2' * 16}-01"],
            "TraceState": ["a=1", "b=2"],
        },
    )

    span = finished_spans()["execute_tool"]
    assert trace_id(span) == TRACE_ID
    assert parent_span_id(span) == SPAN_ID
    assert span.context.trace_state.to_header() == "a=1,b=2"


@pytest.mark.parametrize(
    "headers",
    [
        GetOnlyHeaders({"TraceParent": TRACEPARENT}),
        Headers([("Traceparent", TRACEPARENT)]),
    ],
    ids=["get-only object", "werkzeug Headers"],
)
def test_reads_a_headers_object_with_get(tracing, headers):
    tracer, finished_spans = tracing

    run_tool(tracer, headers)

    span = finished_spans()["execute_tool"]
    assert trace_id(span) == TRACE_ID
    assert parent_span_id(span) == SPAN_ID


@pytest.mark.parametrize(
    "headers",
    [
        {"content-type": "application/json"},
        GetOnlyHeaders({}),
        {"traceparent": "not-a-traceparent"},
        {"traceparent": f"00-{'0' * 32}-{SPAN_ID}-01"},
        {"traceparent": f"ff-{TRACE_ID}-{SPAN_ID}-01"},
    ],
    ids=[
        "no trace headers",
        "an empty headers object",
        "a malformed traceparent",
        "an all-zero trace ID",
        "an unsupported version",
    ],
)
def test_runs_the_block_in_the_current_context_given(tracing, headers):
    tracer, finished_spans = tracing
    before = otel_context.get_current()

    with incoming_trace_context(headers) as ctx:
        assert ctx is before
        tracer.start_span("execute_tool").end()

    span = finished_spans()["execute_tool"]
    assert span.parent is None
    assert trace_id(span) != TRACE_ID


def test_keeps_an_active_span_that_is_already_in_the_incoming_trace(tracing):
    tracer, finished_spans = tracing
    remote = SpanContext(
        trace_id=int(TRACE_ID, 16),
        span_id=int(SPAN_ID, 16),
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    server = tracer.start_span(
        "POST /agent", context=trace.set_span_in_context(NonRecordingSpan(remote))
    )

    with trace.use_span(server, end_on_exit=True):
        run_tool(tracer, {"traceparent": TRACEPARENT})

    tool = finished_spans()["execute_tool"]
    assert parent_span_id(tool) == trace.format_span_id(server.get_span_context().span_id)


def test_replaces_an_active_span_from_another_trace_and_keeps_other_context_values(tracing):
    tracer, finished_spans = tracing

    with statsig_environment("eval"), tracer.start_as_current_span("unrelated"):
        with incoming_trace_context({"traceparent": TRACEPARENT}):
            assert get_environment_span_attrs_from_context() == {STATSIG_ATTR_ENVIRONMENT: "eval"}
            tracer.start_span("execute_tool").end()

    tool = finished_spans()["execute_tool"]
    assert trace_id(tool) == TRACE_ID
    assert parent_span_id(tool) == SPAN_ID


def test_detaches_the_incoming_context_when_the_block_raises(tracing):
    tracer, finished_spans = tracing

    with pytest.raises(ValueError):
        with incoming_trace_context({"traceparent": TRACEPARENT}):
            raise ValueError("boom")

    assert not trace.get_current_span().get_span_context().is_valid
    tracer.start_span("after").end()
    assert finished_spans()["after"].parent is None
