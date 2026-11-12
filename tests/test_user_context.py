import json

import pytest
from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from statsig_ai import StatsigUser, StatsigSpanProcessor, statsig_user_context
from statsig_ai.otel.conventions import STATSIG_ATTR_CUSTOM_IDS, STATSIG_ATTR_USER_ID
from statsig_ai.otel.user_context import (
    UserContextItem,
    get_user_from_otel_context,
    get_user_span_attrs_from_context,
    set_user_to_otel_context,
)


@pytest.fixture
def tracing():
    exporter = InMemorySpanExporter()
    processor = StatsigSpanProcessor(exporter)
    provider = TracerProvider()
    provider.add_span_processor(processor)

    def finished_spans():
        processor.force_flush()
        return {span.name: span for span in exporter.get_finished_spans()}

    yield provider.get_tracer("statsig-test"), finished_spans
    provider.shutdown()


def test_sets_the_user_context_inside_the_block_and_restores_it_afterwards():
    user = StatsigUser(user_id="user-context-test", custom_ids={"team": "red"})

    assert get_user_from_otel_context() is None

    with statsig_user_context(user) as ctx:
        assert ctx is otel_context.get_current()
        assert get_user_from_otel_context() == UserContextItem(
            user_id="user-context-test", custom_ids={"team": "red"}
        )
        assert get_user_span_attrs_from_context() == {
            STATSIG_ATTR_USER_ID: "user-context-test",
            STATSIG_ATTR_CUSTOM_IDS: '{"team":"red"}',
        }

    assert get_user_from_otel_context() is None


def test_restores_the_context_when_the_block_raises():
    with pytest.raises(ValueError):
        with statsig_user_context(StatsigUser(user_id="raises")):
            raise ValueError("boom")

    assert get_user_from_otel_context() is None


def test_falls_back_to_unknown_user_id_and_empty_custom_ids():
    ctx = set_user_to_otel_context(otel_context.get_current(), StatsigUser(user_id=""))

    assert get_user_from_otel_context(ctx) == UserContextItem(user_id="unknown", custom_ids={})
    assert get_user_span_attrs_from_context(ctx) == {STATSIG_ATTR_USER_ID: "unknown"}


def test_processor_adds_user_attributes_from_the_parent_context(tracing):
    tracer, finished_spans = tracing
    user = StatsigUser(user_id="span-user", custom_ids={"org": "acme", "team": "red"})

    with statsig_user_context(user):
        tracer.start_span("inside").end()
    tracer.start_span("outside").end()

    spans = finished_spans()
    inside = spans["inside"].attributes or {}
    assert inside[STATSIG_ATTR_USER_ID] == "span-user"
    # StatsigUser doesn't keep the custom ID order, so compare parsed JSON.
    custom_ids = str(inside[STATSIG_ATTR_CUSTOM_IDS])
    assert json.loads(custom_ids) == {"org": "acme", "team": "red"}
    assert " " not in custom_ids

    outside = spans["outside"].attributes or {}
    assert STATSIG_ATTR_USER_ID not in outside
    assert STATSIG_ATTR_CUSTOM_IDS not in outside


def test_processor_reads_an_explicit_parent_context(tracing):
    tracer, finished_spans = tracing
    ctx = set_user_to_otel_context(otel_context.get_current(), StatsigUser(user_id="explicit"))

    tracer.start_span("explicit-parent", context=ctx).end()

    attrs = finished_spans()["explicit-parent"].attributes or {}
    assert attrs[STATSIG_ATTR_USER_ID] == "explicit"


def test_processor_omits_custom_ids_when_empty(tracing):
    tracer, finished_spans = tracing

    with statsig_user_context(StatsigUser(user_id="no-custom-ids")):
        tracer.start_span("no-custom-ids").end()

    attrs = finished_spans()["no-custom-ids"].attributes or {}
    assert attrs[STATSIG_ATTR_USER_ID] == "no-custom-ids"
    assert STATSIG_ATTR_CUSTOM_IDS not in attrs


def test_non_ascii_custom_ids_are_written_as_is():
    ctx = set_user_to_otel_context(otel_context.get_current(), StatsigUser("u", custom_ids={"team": "café"}))
    attrs = get_user_span_attrs_from_context(ctx)
    assert attrs is not None
    # Like Node's JSON.stringify, non-ASCII text isn't escaped.
    assert attrs[STATSIG_ATTR_CUSTOM_IDS] == '{"team":"café"}'
