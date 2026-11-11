from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import format_span_id, format_trace_id

from statsig_ai.wrappers.span_telemetry import SpanTelemetry


def test_gen_ai_event_logs_hex_trace_and_span_ids():
    span = TracerProvider().get_tracer("test").start_span("chat gpt-4.1")
    telemetry = SpanTelemetry(span, "chat gpt-4.1", "openai")
    span_context = span.get_span_context()

    # Hex, like the Node SDK and eval grades, so gen_ai events join with grade trace_id.
    assert telemetry.metadata["span.trace_id"] == format_trace_id(span_context.trace_id)
    assert telemetry.metadata["span.span_id"] == format_span_id(span_context.span_id)
    assert len(telemetry.metadata["span.trace_id"]) == 32
    assert len(telemetry.metadata["span.span_id"]) == 16
    span.end()
