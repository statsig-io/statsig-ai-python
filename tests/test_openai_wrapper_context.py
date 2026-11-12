import json
from typing import Any, Dict, List, Optional

import pytest
from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from statsig_ai import (
    StatsigContext,
    StatsigUser,
    statsig_context,
    statsig_environment,
    wrap_openai,
)
from statsig_ai.otel.conventions import (
    STATSIG_ATTR_ACTIVITY_ID,
    STATSIG_ATTR_CUSTOM_IDS,
    STATSIG_ATTR_ENVIRONMENT,
    STATSIG_ATTR_SPAN_LLM_ROOT,
    STATSIG_ATTR_SPAN_TYPE,
    STATSIG_ATTR_USER_ID,
    STATSIG_CTX_KEY_LLM_ROOT,
)
from statsig_ai.otel.singleton import OtelSingleton
from statsig_ai.wrappers import span_telemetry

from mock_openai import MockOpenAI

CONTEXT_KEYS = [STATSIG_ATTR_USER_ID, STATSIG_ATTR_CUSTOM_IDS, STATSIG_ATTR_ENVIRONMENT]


class StartAttributesRecorder(SpanProcessor):
    def __init__(self) -> None:
        self.by_span: Dict[str, Dict[str, Any]] = {}

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        self.by_span[span.name] = dict(span.attributes or {})


class FakeStatsig:
    def __init__(self) -> None:
        self.events: List[Dict[str, str]] = []
        self.users: List[StatsigUser] = []

    def log_event(self, user, _event_name, _value, metadata) -> None:
        self.events.append(metadata)
        self.users.append(user)


@pytest.fixture
def traced_openai(monkeypatch: pytest.MonkeyPatch):
    exporter = InMemorySpanExporter()
    start_recorder = StartAttributesRecorder()
    provider = TracerProvider()
    provider.add_span_processor(start_recorder)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    original_instance = OtelSingleton._instance
    OtelSingleton._instance = OtelSingleton(provider)
    statsig = FakeStatsig()
    monkeypatch.setattr(span_telemetry, "get_statsig_instance_for_logging", lambda: statsig)

    yield wrap_openai(MockOpenAI()), exporter, start_recorder, statsig

    OtelSingleton._instance = original_instance
    provider.shutdown()


def _chat(client) -> None:
    client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": "Say hi"}]
    )


def test_model_span_and_event_get_context_and_environment(traced_openai):
    client, exporter, start_recorder, statsig = traced_openai
    user = StatsigUser(user_id="user-123", custom_ids={"org": "acme"})

    with statsig_environment("eval"):
        with statsig_context(StatsigContext(user=user, activity_id="activity-1")):
            _chat(client)

    [span] = exporter.get_finished_spans()
    attrs = span.attributes or {}
    assert attrs[STATSIG_ATTR_SPAN_TYPE] == "gen_ai"
    assert attrs[STATSIG_ATTR_USER_ID] == "user-123"
    assert json.loads(str(attrs[STATSIG_ATTR_CUSTOM_IDS])) == {
        "org": "acme",
        STATSIG_ATTR_ACTIVITY_ID: "activity-1",
    }
    assert attrs[STATSIG_ATTR_ENVIRONMENT] == "eval"

    # The span starts with them, so span processors see them in on_start.
    started = start_recorder.by_span[span.name]
    for key in [STATSIG_ATTR_SPAN_TYPE, *CONTEXT_KEYS]:
        assert started[key] == attrs[key]

    [metadata] = statsig.events
    for key in [STATSIG_ATTR_SPAN_TYPE, *CONTEXT_KEYS]:
        assert metadata[key] == attrs[key]

    # Like the Node SDK, the event user keeps its placeholder ID and takes only the activity ID.
    [event_user] = statsig.users
    assert event_user.user_id == "statsig-ai-openai-wrapper"
    assert event_user.custom_ids == {STATSIG_ATTR_ACTIVITY_ID: "activity-1"}


def test_model_span_and_event_skip_context_attributes_outside_any_scope(traced_openai):
    client, exporter, _, statsig = traced_openai

    _chat(client)

    [span] = exporter.get_finished_spans()
    attrs = span.attributes or {}
    [metadata] = statsig.events
    assert attrs[STATSIG_ATTR_SPAN_TYPE] == "gen_ai"
    assert metadata[STATSIG_ATTR_SPAN_TYPE] == "gen_ai"
    for key in CONTEXT_KEYS:
        assert key not in attrs
        assert key not in metadata
    [event_user] = statsig.users
    assert not event_user.custom_ids


def test_model_span_is_the_llm_root_unless_the_context_has_a_root(traced_openai):
    client, exporter, start_recorder, statsig = traced_openai

    _chat(client)
    token = otel_context.attach(otel_context.set_value(STATSIG_CTX_KEY_LLM_ROOT, "invoke_workflow"))
    try:
        client.chat.completions.create(model="gpt-4o", messages=[])
    finally:
        otel_context.detach(token)

    root, inner = exporter.get_finished_spans()
    assert (root.attributes or {})[STATSIG_ATTR_SPAN_LLM_ROOT] == "1"
    assert start_recorder.by_span[root.name][STATSIG_ATTR_SPAN_LLM_ROOT] == "1"
    assert STATSIG_ATTR_SPAN_LLM_ROOT not in (inner.attributes or {})

    root_event, inner_event = statsig.events
    assert root_event[STATSIG_ATTR_SPAN_LLM_ROOT] == "1"
    assert STATSIG_ATTR_SPAN_LLM_ROOT not in inner_event
