import time
from typing import Any, Dict, List, Optional

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import format_span_id, format_trace_id
from statsig_python_core import StatsigUser

from statsig_ai import StatsigContext, statsig_context
from statsig_ai.otel.conventions import STATSIG_ATTR_ACTIVITY_ID
from statsig_ai.otel.otel import InitializeTracingOptions, initialize_tracing
from statsig_ai.otel.resources import StatsigResourceAttributes
from statsig_ai.wrappers import span_telemetry
from statsig_ai.wrappers.span_telemetry import SpanTelemetry

RESOURCE_ENV_VARS = ["OTEL_SERVICE_NAME", "SERVICE_NAME", "PYTHON_ENV"]


class FakeStatsig:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_event(self, user, event_name, value, metadata) -> None:
        self.events.append(
            {"user": user, "event_name": event_name, "value": value, "metadata": metadata}
        )


class FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def perf_counter(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def reset_singleton_and_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STATSIG_SDK_KEY", "test-sdk-key")
    for name in RESOURCE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    from statsig_ai.otel import singleton as singleton_module

    original_instance = getattr(singleton_module.OtelSingleton, "_instance", None)
    singleton_module.OtelSingleton._instance = None

    yield

    singleton_module.OtelSingleton._instance = original_instance


@pytest.fixture
def statsig(monkeypatch: pytest.MonkeyPatch) -> FakeStatsig:
    fake = FakeStatsig()
    monkeypatch.setattr(span_telemetry, "get_statsig_instance_for_logging", lambda: fake)
    return fake


class Spans:
    def __init__(self) -> None:
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.tracer = self.provider.get_tracer("test")

    def start(
        self,
        span_name: str = "chat gpt-4.1",
        resources: Optional[StatsigResourceAttributes] = None,
        max_json_chars: int = span_telemetry.MAX_JSON_CHARS,
    ) -> SpanTelemetry:
        span = self.tracer.start_span(span_name)
        return SpanTelemetry(
            span, span_name, "openai", max_json_chars=max_json_chars, resources=resources
        )

    def attrs(self) -> Dict[str, Any]:
        [span] = self.exporter.get_finished_spans()
        return dict(span.attributes or {})


@pytest.fixture
def spans():
    result = Spans()
    yield result
    result.provider.shutdown()


def event_user(statsig: FakeStatsig) -> StatsigUser:
    [event] = statsig.events
    return event["user"]


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


def test_event_metadata_has_span_name_and_duration(spans, statsig, monkeypatch):
    clock = FakeClock(100.0)
    monkeypatch.setattr(span_telemetry, "time", clock)

    telemetry = spans.start("chat gpt-4.1")
    clock.now = 100.042
    telemetry.end()

    [event] = statsig.events
    assert event["event_name"] == "statsig::gen_ai"
    assert event["value"] == "chat gpt-4.1"
    assert event["metadata"]["span.name"] == "chat gpt-4.1"
    # Whole milliseconds as a string, as the Node SDK writes it.
    assert event["metadata"]["duration"] == "42"

    # Like the Node SDK, neither is written to the span, and duration is only on the event.
    attrs = spans.attrs()
    assert "span.name" not in attrs
    assert "duration" not in attrs
    assert "duration" not in telemetry.metadata


def test_duration_counts_the_time_until_the_span_ends(spans, statsig):
    telemetry = spans.start()
    time.sleep(0.02)
    telemetry.end()

    duration = statsig.events[0]["metadata"]["duration"]
    assert duration.isdigit()
    assert int(duration) >= 20


def test_short_json_value_records_its_length(spans, statsig):
    messages = [{"role": "user", "content": "hi"}]
    telemetry = spans.start()
    telemetry.set_attributes({"gen_ai.input.messages": messages})
    telemetry.end()

    # Compact, as Node's JSON.stringify writes it.
    full_json = '[{"role":"user","content":"hi"}]'
    attrs = spans.attrs()
    assert attrs["gen_ai.input.messages"] == full_json
    assert attrs["gen_ai.input.messages_len"] == 32

    metadata = statsig.events[0]["metadata"]
    assert metadata["gen_ai.input.messages"] == full_json
    assert metadata["gen_ai.input.messages_len"] == "32"


def test_truncated_json_value_records_its_full_length(spans, statsig):
    value = {"input": "x" * 100}
    telemetry = spans.start(max_json_chars=20)
    telemetry.set_attributes({"payload": value})
    telemetry.end()

    truncated = '{"input":"xxxxxxxxxx…(truncated)'
    attrs = spans.attrs()
    assert attrs["payload"] == truncated
    assert attrs["payload_len"] == 112

    metadata = statsig.events[0]["metadata"]
    assert metadata["payload"] == truncated
    assert metadata["payload_len"] == "112"


def test_non_ascii_json_is_written_as_is_and_its_length_counts_utf16_units(spans, statsig):
    telemetry = spans.start()
    telemetry.set_attributes({"gen_ai.input.messages": [{"content": "café 日本語 😀"}]})
    telemetry.end()

    full_json = '[{"content":"café 日本語 😀"}]'
    # Like a JavaScript string's length, the emoji counts as two: 26 characters, length 27.
    assert len(full_json) == 26
    attrs = spans.attrs()
    assert attrs["gen_ai.input.messages"] == full_json
    assert attrs["gen_ai.input.messages_len"] == 27

    metadata = statsig.events[0]["metadata"]
    assert metadata["gen_ai.input.messages"] == full_json
    assert metadata["gen_ai.input.messages_len"] == "27"


def test_nested_json_is_compact_and_records_its_length(spans, statsig):
    messages = [
        {"role": "user", "content": "Héllo 👋🏽"},
        {"role": "assistant", "content": [{"type": "text", "text": "日本語"}]},
    ]
    telemetry = spans.start()
    telemetry.set_attributes({"gen_ai.input.messages": messages})
    telemetry.end()

    # Node writes the same text, and its length is 102.
    full_json = (
        '[{"role":"user","content":"Héllo 👋🏽"},'
        '{"role":"assistant","content":[{"type":"text","text":"日本語"}]}]'
    )
    attrs = spans.attrs()
    assert attrs["gen_ai.input.messages"] == full_json
    assert attrs["gen_ai.input.messages_len"] == 102

    metadata = statsig.events[0]["metadata"]
    assert metadata["gen_ai.input.messages"] == full_json
    assert metadata["gen_ai.input.messages_len"] == "102"


def test_truncation_counts_utf16_units(spans, statsig):
    telemetry = spans.start(max_json_chars=4)
    telemetry.set_attributes({"emoji": ["😀😀"], "split": ["a😀"]})
    telemetry.end()

    metadata = statsig.events[0]["metadata"]
    # ["😀😀"] is 8 units long; the first 4 are `["` and one emoji.
    assert metadata["emoji"] == '["😀…(truncated)'
    assert metadata["emoji_len"] == "8"
    # Where Node's slice() cuts an emoji in half, it sends U+FFFD.
    assert metadata["split"] == '["a\ufffd…(truncated)'
    assert metadata["split_len"] == "7"


def test_tuples_are_written_as_json_arrays(spans, statsig):
    telemetry = spans.start()
    telemetry.set_attributes({"gen_ai.response.finish_reasons": ("stop", "length")})
    telemetry.end()

    assert spans.attrs()["gen_ai.response.finish_reasons"] == '["stop","length"]'
    assert statsig.events[0]["metadata"]["gen_ai.response.finish_reasons_len"] == "17"


def test_values_json_cannot_write_are_unserializable(spans, statsig):
    circular: List[Any] = []
    circular.append(circular)
    telemetry = spans.start()
    telemetry.set_attributes({"circular": circular, "object": {"when": object()}})
    telemetry.end()

    metadata = statsig.events[0]["metadata"]
    for key in ["circular", "object"]:
        assert metadata[key] == "[[unserializable]]"
        assert f"{key}_len" not in metadata


def test_plain_values_do_not_record_a_length(spans, statsig):
    telemetry = spans.start()
    telemetry.set_attributes({"gen_ai.request.model": "gpt-4.1"})
    telemetry.end()

    assert "gen_ai.request.model_len" not in spans.attrs()
    assert "gen_ai.request.model_len" not in statsig.events[0]["metadata"]


def test_boolean_and_number_metadata_is_written_as_javascript_prints_it(spans, statsig):
    attributes = {
        "gen_ai.request.stream": True,
        "gen_ai.request.store": False,
        "gen_ai.request.temperature": 1.0,
        "gen_ai.request.top_p": 0.5,
        "gen_ai.usage.input_tokens": 12,
        "tiny": 1e-7,
        "huge": 1e21,
    }
    telemetry = spans.start()
    telemetry.set_attributes(attributes)
    telemetry.end()

    metadata = statsig.events[0]["metadata"]
    assert metadata["gen_ai.request.stream"] == "true"
    assert metadata["gen_ai.request.store"] == "false"
    assert metadata["gen_ai.request.temperature"] == "1"
    assert metadata["gen_ai.request.top_p"] == "0.5"
    assert metadata["gen_ai.usage.input_tokens"] == "12"
    assert metadata["tiny"] == "1e-7"
    assert metadata["huge"] == "1e+21"

    # The span keeps the typed values.
    attrs = spans.attrs()
    for key, value in attributes.items():
        assert attrs[key] == value


class ApiError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"Error code: 500 - {message}")
        self.message = message


def test_record_exception_writes_type_and_message_to_metadata_only(spans, statsig):
    telemetry = spans.start()
    telemetry.record_exception(ValueError("bad news"))
    telemetry.end()

    [span] = spans.exporter.get_finished_spans()
    [exception] = span.events
    assert exception.name == "exception"
    assert (exception.attributes or {})["exception.type"] == "ValueError"
    assert (exception.attributes or {})["exception.message"] == "bad news"
    assert "exception.type" not in (span.attributes or {})
    assert "exception.message" not in (span.attributes or {})

    metadata = statsig.events[0]["metadata"]
    assert metadata["exception.type"] == "ValueError"
    assert metadata["exception.message"] == "bad news"


def test_record_exception_prefers_the_error_message_attribute(spans, statsig):
    telemetry = spans.start()
    telemetry.record_exception(ApiError("overloaded"))
    telemetry.end()

    metadata = statsig.events[0]["metadata"]
    assert metadata["exception.type"] == "ApiError"
    assert metadata["exception.message"] == "overloaded"


def test_record_exception_skips_an_empty_message(spans, statsig):
    telemetry = spans.start()
    telemetry.record_exception(KeyboardInterrupt())
    telemetry.end()

    metadata = statsig.events[0]["metadata"]
    assert metadata["exception.type"] == "KeyboardInterrupt"
    assert "exception.message" not in metadata


def test_record_exception_records_a_string_as_the_message(spans, statsig):
    telemetry = spans.start()
    telemetry.record_exception("boom")
    telemetry.end()

    [span] = spans.exporter.get_finished_spans()
    [exception] = span.events
    # What OpenTelemetry JS records for a string.
    assert dict(exception.attributes or {}) == {"exception.message": "boom"}
    assert "exception.message" not in (span.attributes or {})

    metadata = statsig.events[0]["metadata"]
    assert metadata["exception.type"] == "str"
    assert metadata["exception.message"] == "boom"


def test_fail_uses_the_error_text_when_its_message_is_none(spans, statsig):
    error = ApiError("ignored")
    error.message = None  # type: ignore[assignment]
    telemetry = spans.start()
    telemetry.fail(error)
    telemetry.end()

    metadata = statsig.events[0]["metadata"]
    assert metadata["span.status_code"] == "ERROR"
    assert metadata["span.status_message"] == "Error code: 500 - ignored"
    assert metadata["exception.message"] == "Error code: 500 - ignored"


def test_event_user_gets_resource_values_from_initialize_tracing(spans, statsig):
    result = initialize_tracing(
        InitializeTracingOptions(
            service_name="checkout",
            version="1.2.3",
            environment="staging",
            skip_global_context_manager_setup=True,
        )
    )
    try:
        spans.start().end()
    finally:
        result.provider.shutdown()

    user = event_user(statsig)
    assert user.user_id == "statsig-ai-openai-wrapper"
    assert user.custom_ids == {
        "service": "checkout",
        "version": "1.2.3",
        "environment": "staging",
    }


def test_event_user_omits_resource_values_initialize_tracing_did_not_set(spans, statsig):
    result = initialize_tracing(
        InitializeTracingOptions(service_name="checkout", skip_global_context_manager_setup=True)
    )
    try:
        spans.start().end()
    finally:
        result.provider.shutdown()

    assert event_user(statsig).custom_ids == {"service": "checkout"}


def test_event_user_reads_environment_variables_without_initialize_tracing(
    spans, statsig, monkeypatch
):
    monkeypatch.setenv("SERVICE_NAME", "env-service")
    monkeypatch.setenv("PYTHON_ENV", "ci")

    spans.start().end()

    user = event_user(statsig)
    assert user.user_id == "statsig-ai-openai-wrapper"
    assert user.custom_ids == {"service": "env-service", "environment": "ci"}


def test_event_user_has_no_custom_ids_without_initialize_tracing_or_environment(
    spans, statsig
):
    spans.start().end()

    user = event_user(statsig)
    assert user.user_id == "statsig-ai-openai-wrapper"
    assert not user.custom_ids


def test_event_user_gets_activity_id_from_statsig_context(spans, statsig):
    resources = StatsigResourceAttributes(service="checkout", environment="production")
    user = StatsigUser(user_id="user-123", custom_ids={"org": "acme"})

    with statsig_context(StatsigContext(user=user, activity_id="activity-1")):
        spans.start(resources=resources).end()

    # The context's user stays on the span attributes. The event user only takes its activity ID.
    assert event_user(statsig).custom_ids == {
        "service": "checkout",
        "environment": "production",
        STATSIG_ATTR_ACTIVITY_ID: "activity-1",
    }


def test_event_user_has_no_activity_id_without_one_in_statsig_context(spans, statsig):
    resources = StatsigResourceAttributes(service="checkout")

    with statsig_context(StatsigContext(user=StatsigUser(user_id="user-123"))):
        spans.start(resources=resources).end()
    spans.start(resources=resources).end()

    assert [event["user"].custom_ids for event in statsig.events] == [
        {"service": "checkout"},
        {"service": "checkout"},
    ]


def test_event_user_is_built_for_each_event(spans, statsig):
    with statsig_context(StatsigContext(activity_id="activity-1")):
        spans.start().end()
    with statsig_context(StatsigContext(activity_id="activity-2")):
        spans.start().end()

    assert [event["user"].custom_ids for event in statsig.events] == [
        {STATSIG_ATTR_ACTIVITY_ID: "activity-1"},
        {STATSIG_ATTR_ACTIVITY_ID: "activity-2"},
    ]


def test_error_status_writes_only_nodes_status_keys(spans, statsig):
    from opentelemetry.trace import StatusCode

    telemetry = spans.start()
    telemetry.set_status({"code": StatusCode.ERROR, "message": "boom"})
    telemetry.end()

    metadata = statsig.events[0]["metadata"]
    assert metadata["span.status_code"] == "ERROR"
    assert metadata["span.status_message"] == "boom"
    # Node's setStatus doesn't write an error.type key.
    assert "error.type" not in metadata
