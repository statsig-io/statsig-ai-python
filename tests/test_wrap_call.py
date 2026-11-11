import asyncio
import inspect
import json
from typing import Any, Dict, List, Optional

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from statsig_ai import (
    Prompt,
    PromptVersion,
    StatsigSpanProcessor,
    StatsigUser,
    start_workflow,
    statsig_environment,
    statsig_user_context,
    wrap,
    wrap_openai,
)
from statsig_ai.otel.conventions import (
    STATSIG_ATTR_ACTIVITY_ID,
    STATSIG_ATTR_CUSTOM_IDS,
    STATSIG_ATTR_ENVIRONMENT,
    STATSIG_ATTR_GEN_AI_SPAN_TYPE,
    STATSIG_ATTR_LLM_PROMPT_NAME,
    STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES,
    STATSIG_ATTR_LLM_PROMPT_VERSION_ID,
    STATSIG_ATTR_SPAN_LLM_ROOT,
    STATSIG_ATTR_SPAN_TYPE,
    STATSIG_ATTR_USER_ID,
    STATSIG_SPAN_LLM_ROOT_VALUE,
    StatsigGenAISpanType,
    StatsigSpanType,
)
from statsig_ai.otel.singleton import OtelSingleton
from statsig_ai.wrappers import span_telemetry

from mock_openai import DefaultMockResponses, MockOpenAI


class FakeStatsig:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_event(self, _user, event_name, value, metadata) -> None:
        self.events.append({"event_name": event_name, "value": value, "metadata": metadata})

    def by_value(self, value: str) -> List[Dict[str, str]]:
        return [e["metadata"] for e in self.events if e["value"] == value]


class Tracing:
    def __init__(self, processor: StatsigSpanProcessor, exporter: InMemorySpanExporter) -> None:
        self._processor = processor
        self._exporter = exporter
        self.statsig = FakeStatsig()

    def spans(self) -> List[ReadableSpan]:
        self._processor.force_flush()
        return list(self._exporter.get_finished_spans())

    def span(self, name: str) -> ReadableSpan:
        [span] = [s for s in self.spans() if s.name == name]
        return span

    def attrs(self, name: str) -> Dict[str, Any]:
        return dict(self.span(name).attributes or {})


@pytest.fixture
def tracing(monkeypatch: pytest.MonkeyPatch):
    exporter = InMemorySpanExporter()
    processor = StatsigSpanProcessor(exporter)
    provider = TracerProvider()
    provider.add_span_processor(processor)

    original_instance = OtelSingleton._instance
    OtelSingleton._instance = OtelSingleton(provider)
    result = Tracing(processor, exporter)
    monkeypatch.setattr(span_telemetry, "get_statsig_instance_for_logging", lambda: result.statsig)

    yield result

    OtelSingleton._instance = original_instance
    provider.shutdown()


class AsyncMockOpenAI:
    def __init__(self) -> None:
        self.chat = self.ChatNamespace()

    class ChatNamespace:
        class CompletionsNamespace:
            async def create(self, *args, **kwargs):
                await asyncio.sleep(0)
                return DefaultMockResponses.chat_completion

        def __init__(self) -> None:
            self.completions = self.CompletionsNamespace()


def ask(client, model: str):
    return client.chat.completions.create(model=model, messages=[])


def llm_root_by_span(tracing: Tracing) -> Dict[str, Optional[str]]:
    return {
        span.name: (span.attributes or {}).get(STATSIG_ATTR_SPAN_LLM_ROOT)
        for span in tracing.spans()
    }


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
    return PromptVersion(config, True)  # type: ignore[arg-type]


# Ported from the Node SDK's wrap-call tests.


def test_records_tool_metadata_and_user_attributes_for_sync_functions(tracing):
    user = StatsigUser(user_id="user-123", custom_ids={"team": "infra"})
    wrapped = wrap(
        lambda: "ok",
        type="tool",
        name="search",
        tool_type="retrieval",
        user=user,
        attributes={"custom.attr": "provided"},
    )

    assert wrapped() == "ok"

    [span] = tracing.spans()
    assert span.name == "execute_tool"
    assert span.status.status_code == StatusCode.OK
    attrs = dict(span.attributes or {})
    assert attrs[STATSIG_ATTR_SPAN_TYPE] == StatsigSpanType.GEN_AI
    assert attrs[STATSIG_ATTR_GEN_AI_SPAN_TYPE] == StatsigGenAISpanType.TOOL
    assert attrs["statsig.gen_ai.tool.name"] == "search"
    assert attrs["gen_ai.tool.type"] == "retrieval"
    assert attrs["custom.attr"] == "provided"
    assert attrs[STATSIG_ATTR_USER_ID] == "user-123"
    assert attrs[STATSIG_ATTR_CUSTOM_IDS] == json.dumps({"team": "infra"}, separators=(",", ":"))

    [event] = tracing.statsig.events
    assert event["event_name"] == "statsig::gen_ai"
    assert event["value"] == "execute_tool"
    metadata = event["metadata"]
    assert metadata["statsig.gen_ai.tool.name"] == "search"
    assert metadata["gen_ai.tool.type"] == "retrieval"
    assert metadata["custom.attr"] == "provided"
    assert metadata[STATSIG_ATTR_USER_ID] == "user-123"
    assert metadata["span.status_code"] == "OK"


@pytest.mark.asyncio
async def test_marks_spans_as_errors_when_wrapped_async_functions_raise(tracing):
    async def embed():
        await asyncio.sleep(0)
        raise ValueError("bad news")

    wrapped = wrap(embed, type="workflow", name="embedding")

    with pytest.raises(ValueError, match="bad news"):
        await wrapped()

    [span] = tracing.spans()
    assert span.name == "invoke_workflow"
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "bad news"
    attrs = dict(span.attributes or {})
    assert attrs["statsig.gen_ai.workflow.name"] == "embedding"
    assert attrs[STATSIG_ATTR_GEN_AI_SPAN_TYPE] == StatsigGenAISpanType.WORKFLOW
    assert [e.name for e in span.events] == ["exception"]

    [event] = tracing.statsig.events
    assert event["value"] == "invoke_workflow"
    assert event["metadata"]["statsig.gen_ai.workflow.name"] == "embedding"
    assert event["metadata"]["span.status_code"] == "ERROR"


def test_start_workflow_adds_llm_root_attribute_and_forwards_custom_attributes(tracing):
    with start_workflow("outer-workflow", attributes={"env.stage": "staging"}) as span:
        result = "done"

    assert result == "done"
    [finished] = tracing.spans()
    assert finished.name == "invoke_workflow"
    assert finished.context.span_id == span.get_span_context().span_id
    attrs = dict(finished.attributes or {})
    assert attrs["env.stage"] == "staging"
    assert attrs[STATSIG_ATTR_SPAN_LLM_ROOT] == STATSIG_SPAN_LLM_ROOT_VALUE
    assert attrs[STATSIG_ATTR_GEN_AI_SPAN_TYPE] == StatsigGenAISpanType.WORKFLOW
    assert attrs["statsig.gen_ai.workflow.name"] == "outer-workflow"

    [event] = tracing.statsig.events
    assert event["value"] == "invoke_workflow"
    assert event["metadata"]["statsig.gen_ai.workflow.name"] == "outer-workflow"
    assert event["metadata"]["env.stage"] == "staging"
    assert event["metadata"][STATSIG_ATTR_SPAN_LLM_ROOT] == STATSIG_SPAN_LLM_ROOT_VALUE


def test_marks_only_llm_calls_made_outside_a_workflow_as_the_llm_root(tracing):
    client = wrap_openai(MockOpenAI())

    ask(client, "standalone")

    @wrap(type="workflow", name="agent-turn")
    def agent_turn():
        ask(client, "first-round")
        wrap(lambda: ask(client, "in-tool"), type="tool", name="lookup")()
        ask(client, "final-round")

    agent_turn()

    assert llm_root_by_span(tracing) == {
        "chat standalone": STATSIG_SPAN_LLM_ROOT_VALUE,
        "chat first-round": None,
        "chat in-tool": None,
        "chat final-round": None,
        "execute_tool": None,
        "invoke_workflow": None,
    }


def test_tags_workflow_spans_with_the_active_version_and_the_variables_rendered_inside(tracing):
    version = make_version("version-3", "v3")
    prompt = Prompt("support_agent", version, [], StatsigUser(user_id="user-1"))

    with prompt.with_live():
        with start_workflow("agent-turn"):
            version.get_prompt_messages({"input": "Hi"})
            wrap(lambda: "ok", type="tool", name="lookup")()

    workflow = tracing.attrs("invoke_workflow")
    tool = tracing.attrs("execute_tool")
    assert workflow[STATSIG_ATTR_LLM_PROMPT_NAME] == "support_agent"
    assert workflow[STATSIG_ATTR_LLM_PROMPT_VERSION_ID] == "version-3"
    assert workflow[STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES] == json.dumps({"input": "Hi"})
    assert tool[STATSIG_ATTR_LLM_PROMPT_VERSION_ID] == "version-3"
    assert STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES not in tool

    [workflow_event] = tracing.statsig.by_value("invoke_workflow")
    assert workflow_event[STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES] == json.dumps(
        {"input": "Hi"}
    )
    [tool_event] = tracing.statsig.by_value("execute_tool")
    assert STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES not in tool_event


def test_sets_statsig_context_with_user_and_activity_id(tracing):
    user = StatsigUser(user_id="test-user-123", custom_ids={"sessionID": "session-456"})
    wrapped = wrap(
        lambda: "success", type="tool", name="test-tool", user=user, activity_id="activity-789"
    )

    assert wrapped() == "success"

    [span] = tracing.spans()
    assert span.name == "execute_tool"
    assert span.status.status_code == StatusCode.OK
    attrs = dict(span.attributes or {})
    assert attrs[STATSIG_ATTR_USER_ID] == "test-user-123"
    assert json.loads(attrs[STATSIG_ATTR_CUSTOM_IDS]) == {
        "sessionID": "session-456",
        STATSIG_ATTR_ACTIVITY_ID: "activity-789",
    }


def test_sets_statsig_context_with_only_activity_id(tracing):
    wrapped = wrap(lambda: "done", type="workflow", name="test-workflow", activity_id="activity-1")

    assert wrapped() == "done"

    attrs = tracing.attrs("invoke_workflow")
    assert STATSIG_ATTR_USER_ID not in attrs
    assert json.loads(attrs[STATSIG_ATTR_CUSTOM_IDS]) == {STATSIG_ATTR_ACTIVITY_ID: "activity-1"}


def test_sets_statsig_context_with_only_user(tracing):
    wrapped = wrap(
        lambda: "result", type="tool", name="test-tool", user=StatsigUser(user_id="user-only-456")
    )

    assert wrapped() == "result"

    attrs = tracing.attrs("execute_tool")
    assert attrs[STATSIG_ATTR_USER_ID] == "user-only-456"
    assert STATSIG_ATTR_ACTIVITY_ID not in attrs
    assert STATSIG_ATTR_CUSTOM_IDS not in attrs


@pytest.mark.asyncio
async def test_propagates_statsig_context_in_async_wrapped_functions(tracing):
    client = wrap_openai(AsyncMockOpenAI())

    async def run():
        await asyncio.sleep(0.01)
        await ask(client, "gpt-4")
        return "async-result"

    wrapped = wrap(
        run,
        type="workflow",
        name="async-workflow",
        user=StatsigUser(user_id="async-user-789"),
        activity_id="async-activity-999",
    )

    assert await wrapped() == "async-result"

    expected_custom_ids = {STATSIG_ATTR_ACTIVITY_ID: "async-activity-999"}
    for name in ["invoke_workflow", "chat gpt-4"]:
        attrs = tracing.attrs(name)
        assert attrs[STATSIG_ATTR_USER_ID] == "async-user-789"
        assert json.loads(attrs[STATSIG_ATTR_CUSTOM_IDS]) == expected_custom_ids
    assert tracing.span("invoke_workflow").status.status_code == StatusCode.OK


# Python forms: decorators, async functions and context managers.


def test_decorator_form_records_each_call_and_keeps_the_function_metadata(tracing):
    @wrap(type="tool", name="search", tool_type="retrieval")
    def search(query: str, *, limit: int = 3) -> List[str]:
        """Searches the index."""
        return [query] * limit

    assert search.__name__ == "search"
    assert search.__doc__ == "Searches the index."
    assert search("q", limit=2) == ["q", "q"]
    assert search("r") == ["r", "r", "r"]

    spans = tracing.spans()
    assert [s.name for s in spans] == ["execute_tool", "execute_tool"]
    for span in spans:
        assert (span.attributes or {})["statsig.gen_ai.tool.name"] == "search"
    assert spans[0].context.trace_id != spans[1].context.trace_id


@pytest.mark.asyncio
async def test_async_decorator_ends_the_span_after_the_await(tracing):
    @wrap(type="tool", name="fetch")
    async def fetch(url: str) -> str:
        await asyncio.sleep(0.02)
        return url.upper()

    assert inspect.iscoroutinefunction(fetch)
    assert fetch.__name__ == "fetch"
    assert await fetch("a") == "A"

    span = tracing.span("execute_tool")
    assert span.status.status_code == StatusCode.OK
    assert span.end_time is not None and span.start_time is not None
    assert span.end_time - span.start_time >= 20_000_000


@pytest.mark.asyncio
async def test_a_plain_function_returning_a_coroutine_keeps_the_span_open_until_awaited(tracing):
    client = wrap_openai(AsyncMockOpenAI())

    @wrap(type="workflow", name="turn")
    async def turn():
        return await wrap(lambda: ask(client, "in-tool"), type="tool", name="lookup")()

    result = await turn()

    assert result == DefaultMockResponses.chat_completion
    workflow = tracing.span("invoke_workflow")
    tool = tracing.span("execute_tool")
    model = tracing.span("chat in-tool")
    assert tool.parent is not None and tool.parent.span_id == workflow.context.span_id
    assert model.parent is not None and model.parent.span_id == tool.context.span_id
    assert tool.status.status_code == StatusCode.OK
    assert tool.end_time is not None and model.end_time is not None
    assert tool.end_time >= model.end_time


def test_sync_errors_are_recorded_and_reraised(tracing):
    @wrap(type="tool", name="explode")
    def explode():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        explode()

    span = tracing.span("execute_tool")
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "boom"
    [exception] = span.events
    assert exception.name == "exception"
    assert (exception.attributes or {})["exception.type"] == "RuntimeError"

    [event] = tracing.statsig.events
    assert event["metadata"]["span.status_code"] == "ERROR"
    assert event["metadata"]["exception.type"] == "RuntimeError"


def test_start_workflow_records_exceptions_and_lets_them_propagate(tracing):
    with pytest.raises(KeyError):
        with start_workflow("turn"):
            raise KeyError("missing")

    span = tracing.span("invoke_workflow")
    assert span.status.status_code == StatusCode.ERROR
    assert [e.name for e in span.events] == ["exception"]
    [event] = tracing.statsig.events
    assert event["metadata"]["span.status_code"] == "ERROR"


def test_start_workflow_sets_the_statsig_context(tracing):
    client = wrap_openai(MockOpenAI())
    user = StatsigUser(user_id="user-1", custom_ids={"org": "acme"})

    with start_workflow("turn", user=user, activity_id="activity-1"):
        ask(client, "gpt-4")

    expected_custom_ids = {"org": "acme", STATSIG_ATTR_ACTIVITY_ID: "activity-1"}
    for name in ["invoke_workflow", "chat gpt-4"]:
        attrs = tracing.attrs(name)
        assert attrs[STATSIG_ATTR_USER_ID] == "user-1"
        assert json.loads(attrs[STATSIG_ATTR_CUSTOM_IDS]) == expected_custom_ids


@pytest.mark.asyncio
async def test_start_workflow_works_inside_async_functions(tracing):
    client = wrap_openai(AsyncMockOpenAI())

    @wrap(type="tool", name="lookup")
    async def lookup(model: str):
        return await ask(client, model)

    with start_workflow("turn"):
        await asyncio.gather(lookup("a"), lookup("b"))

    workflow = tracing.span("invoke_workflow")
    tools = [s for s in tracing.spans() if s.name == "execute_tool"]
    assert len(tools) == 2
    for tool in tools:
        assert tool.parent is not None and tool.parent.span_id == workflow.context.span_id
    tool_ids = {tool.context.span_id for tool in tools}
    for name in ["chat a", "chat b"]:
        model = tracing.span(name)
        assert model.parent is not None and model.parent.span_id in tool_ids
    assert llm_root_by_span(tracing) == {
        "chat a": None,
        "chat b": None,
        "execute_tool": None,
        "invoke_workflow": STATSIG_SPAN_LLM_ROOT_VALUE,
    }


def test_wrap_spans_get_the_environment_and_user_context(tracing):
    with statsig_environment("staging"):
        with statsig_user_context(StatsigUser(user_id="ctx-user", custom_ids={"org": "acme"})):
            wrap(lambda: None, type="tool", name="lookup")()

    attrs = tracing.attrs("execute_tool")
    assert attrs[STATSIG_ATTR_ENVIRONMENT] == "staging"
    assert attrs[STATSIG_ATTR_USER_ID] == "ctx-user"
    [event] = tracing.statsig.events
    assert event["metadata"][STATSIG_ATTR_ENVIRONMENT] == "staging"
    assert event["metadata"][STATSIG_ATTR_USER_ID] == "ctx-user"
    assert json.loads(event["metadata"][STATSIG_ATTR_CUSTOM_IDS]) == {"org": "acme"}


def test_logs_one_gen_ai_event_per_call(tracing):
    client = wrap_openai(MockOpenAI())
    lookup = wrap(lambda: ask(client, "gpt-4"), type="tool", name="lookup")

    lookup()
    lookup()

    assert [(e["event_name"], e["value"]) for e in tracing.statsig.events] == [
        ("statsig::gen_ai", "chat gpt-4"),
        ("statsig::gen_ai", "execute_tool"),
        ("statsig::gen_ai", "chat gpt-4"),
        ("statsig::gen_ai", "execute_tool"),
    ]
    # Only model call events name a provider.
    [model_event, tool_event, *_] = tracing.statsig.events
    assert model_event["metadata"]["gen_ai.provider.name"] == "openai"
    assert "gen_ai.provider.name" not in tool_event["metadata"]


def test_model_calls_inside_a_workflow_are_its_children(tracing):
    client = wrap_openai(MockOpenAI())

    with start_workflow("outer"):
        wrap(lambda: ask(client, "in-workflow"), type="workflow", name="inner")()

    workflows = {
        (s.attributes or {})["statsig.gen_ai.workflow.name"]: s
        for s in tracing.spans()
        if s.name == "invoke_workflow"
    }
    outer, inner = workflows["outer"], workflows["inner"]
    model = tracing.span("chat in-workflow")
    assert inner.parent is not None and inner.parent.span_id == outer.context.span_id
    assert model.parent is not None and model.parent.span_id == inner.context.span_id
    assert model.context.trace_id == outer.context.trace_id
    assert outer.parent is None


# LLM root


def test_a_model_call_on_its_own_is_the_llm_root(tracing):
    ask(wrap_openai(MockOpenAI()), "gpt-4")

    assert llm_root_by_span(tracing) == {"chat gpt-4": STATSIG_SPAN_LLM_ROOT_VALUE}


def test_a_model_call_inside_a_workflow_is_not_the_llm_root(tracing):
    client = wrap_openai(MockOpenAI())

    wrap(lambda: ask(client, "gpt-4"), type="workflow", name="turn")()

    assert llm_root_by_span(tracing) == {"chat gpt-4": None, "invoke_workflow": None}


def test_a_model_call_inside_a_tool_alone_is_the_llm_root(tracing):
    client = wrap_openai(MockOpenAI())

    wrap(lambda: ask(client, "gpt-4"), type="tool", name="lookup")()

    assert llm_root_by_span(tracing) == {
        "chat gpt-4": STATSIG_SPAN_LLM_ROOT_VALUE,
        "execute_tool": None,
    }


def test_start_workflow_is_the_llm_root_and_model_calls_inside_it_are_not(tracing):
    client = wrap_openai(MockOpenAI())

    with start_workflow("turn"):
        ask(client, "first")
        wrap(lambda: ask(client, "in-tool"), type="tool", name="lookup")()

    assert llm_root_by_span(tracing) == {
        "chat first": None,
        "chat in-tool": None,
        "execute_tool": None,
        "invoke_workflow": STATSIG_SPAN_LLM_ROOT_VALUE,
    }


def test_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="workflow"):
        wrap(lambda: None, type="agent", name="x")  # type: ignore[call-overload]
