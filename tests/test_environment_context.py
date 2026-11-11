import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from statsig_ai import (
    StatsigContext,
    StatsigSpanProcessor,
    StatsigUser,
    statsig_context,
    statsig_environment,
    statsig_user_context,
)
from statsig_ai.otel.conventions import STATSIG_ATTR_ENVIRONMENT, STATSIG_ATTR_USER_ID
from statsig_ai.otel.environment_context import get_environment_span_attrs_from_context


@pytest.fixture
def tracing():
    exporter = InMemorySpanExporter()
    processor = StatsigSpanProcessor(exporter)
    provider = TracerProvider()
    provider.add_span_processor(processor)

    def finished_spans():
        processor.force_flush()
        return {span.name: span.attributes or {} for span in exporter.get_finished_spans()}

    def environment_by_span():
        return {
            name: attrs.get(STATSIG_ATTR_ENVIRONMENT) for name, attrs in finished_spans().items()
        }

    yield provider.get_tracer("statsig-test-environment"), environment_by_span, finished_spans
    provider.shutdown()


def test_tags_spans_started_inside_statsig_environment(tracing):
    tracer, environment_by_span, _ = tracing

    with statsig_environment("eval"):
        tracer.start_span("eval-span").end()
        with statsig_context(StatsigContext(activity_id="activity")):
            tracer.start_span("nested-context-span").end()
    tracer.start_span("outside-span").end()

    assert environment_by_span() == {
        "eval-span": "eval",
        "nested-context-span": "eval",
        "outside-span": None,
    }


def test_inner_scope_overrides_the_outer_environment(tracing):
    tracer, environment_by_span, _ = tracing

    with statsig_environment("production"):
        with statsig_environment("eval"):
            tracer.start_span("inner-span").end()
        tracer.start_span("outer-span").end()

    assert environment_by_span() == {"inner-span": "eval", "outer-span": "production"}


def test_empty_environment_adds_no_attribute(tracing):
    tracer, environment_by_span, _ = tracing

    with statsig_environment("") as ctx:
        assert get_environment_span_attrs_from_context(ctx) is None
        tracer.start_span("empty-span").end()

    assert environment_by_span() == {"empty-span": None}


def test_restores_the_environment_when_the_block_raises():
    with pytest.raises(ValueError):
        with statsig_environment("eval"):
            raise ValueError("boom")

    assert get_environment_span_attrs_from_context() is None


def test_combines_with_the_user_context(tracing):
    tracer, _, finished_spans = tracing

    with statsig_environment("eval"), statsig_user_context(StatsigUser(user_id="u1")):
        tracer.start_span("both").end()

    attrs = finished_spans()["both"]
    assert attrs[STATSIG_ATTR_ENVIRONMENT] == "eval"
    assert attrs[STATSIG_ATTR_USER_ID] == "u1"
