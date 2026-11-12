import json
from typing import Any, Dict, List, Optional

import pytest
from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from statsig_ai import (
    GenAICaptureOptions,
    Prompt,
    PromptVersion,
    StatsigSpanProcessor,
    StatsigUser,
    WrapOpenAIOptions,
    wrap_openai,
)
from statsig_ai.otel.conventions import (
    STATSIG_ATTR_CUSTOM_IDS,
    STATSIG_ATTR_LLM_PROMPT_NAME,
    STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES,
    STATSIG_ATTR_LLM_PROMPT_VERSION,
    STATSIG_ATTR_LLM_PROMPT_VERSION_ID,
    STATSIG_ATTR_USER_ID,
    STATSIG_CTX_KEY_ACTIVE_PROMPT,
    STATSIG_CTX_KEY_ACTIVE_PROMPT_VERSION,
    STATSIG_CTX_KEY_ACTIVE_PROMPT_VERSION_ID,
)
from statsig_ai.otel.prompt_context import (
    get_prompt_span_attrs_from_context,
    record_prompt_template_variables,
    set_prompt_version_to_context,
)
from statsig_ai.otel.singleton import OtelSingleton
from statsig_ai.otel.user_context import UserContextItem, get_user_from_otel_context
from statsig_ai.prompt import make_prompt
from statsig_ai.wrappers import span_telemetry

from mock_openai import MockOpenAI

PROMPT_ATTRS = [
    STATSIG_ATTR_LLM_PROMPT_NAME,
    STATSIG_ATTR_LLM_PROMPT_VERSION,
    STATSIG_ATTR_LLM_PROMPT_VERSION_ID,
    STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES,
]


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


class FakeStatsig:
    def __init__(self) -> None:
        self.events: List[Dict[str, str]] = []

    def log_event(self, _user, _event_name, _value, metadata) -> None:
        self.events.append(metadata)


class Tracing:
    def __init__(self, processor: StatsigSpanProcessor, exporter: InMemorySpanExporter) -> None:
        self._processor = processor
        self._exporter = exporter
        self.statsig = FakeStatsig()

    def spans(self) -> List[Dict[str, Any]]:
        self._processor.force_flush()
        return [dict(span.attributes or {}) for span in self._exporter.get_finished_spans()]


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


@pytest.fixture
def user() -> StatsigUser:
    return StatsigUser(user_id="prompt-user", custom_ids={"cohort": "beta"})


@pytest.fixture
def live() -> PromptVersion:
    return make_version("version-a", "VersionA")


@pytest.fixture
def prompt(user: StatsigUser, live: PromptVersion) -> Prompt:
    return Prompt("example-prompt", live, [], user)


def openai_client(capture: Optional[GenAICaptureOptions] = None):
    return wrap_openai(MockOpenAI(), WrapOpenAIOptions(capture))


def chat(client, version: Optional[PromptVersion] = None, text: str = "hello") -> None:
    messages = version.get_prompt_messages({"input": text}) if version else []
    client.chat.completions.create(model="gpt-4", messages=messages)


def prompt_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    return {key: attrs[key] for key in PROMPT_ATTRS if key in attrs}


EXPECTED_USER = UserContextItem(user_id="prompt-user", custom_ids={"cohort": "beta"})


# Context helpers, ported from the Node SDK's prompt-context tests.


def test_with_live_sets_prompt_and_user_in_the_context_and_restores_it(prompt, live):
    assert otel_context.get_value(STATSIG_CTX_KEY_ACTIVE_PROMPT) is None

    with prompt.with_live() as version:
        assert version is live
        assert otel_context.get_value(STATSIG_CTX_KEY_ACTIVE_PROMPT) == "example-prompt"
        assert otel_context.get_value(STATSIG_CTX_KEY_ACTIVE_PROMPT_VERSION) == "VersionA"
        assert otel_context.get_value(STATSIG_CTX_KEY_ACTIVE_PROMPT_VERSION_ID) == "version-a"
        assert get_user_from_otel_context() == EXPECTED_USER

    assert otel_context.get_value(STATSIG_CTX_KEY_ACTIVE_PROMPT) is None
    assert get_user_from_otel_context() is None


def test_with_context_sets_prompt_name_and_user_without_a_version(prompt):
    with prompt.with_context() as active:
        assert active is prompt
        assert otel_context.get_value(STATSIG_CTX_KEY_ACTIVE_PROMPT) == "example-prompt"
        assert otel_context.get_value(STATSIG_CTX_KEY_ACTIVE_PROMPT_VERSION) is None
        assert get_user_from_otel_context() == EXPECTED_USER
        assert get_prompt_span_attrs_from_context(include_template_variables=True) == {
            STATSIG_ATTR_LLM_PROMPT_NAME: "example-prompt"
        }

    assert otel_context.get_value(STATSIG_CTX_KEY_ACTIVE_PROMPT) is None
    assert get_user_from_otel_context() is None


def test_with_version_tags_the_version_and_records_its_template_variables(prompt):
    candidate = make_version("version-b", "VersionB")
    other = make_version("version-c", "VersionC")

    with prompt.with_version(candidate) as version:
        assert version is candidate
        assert get_prompt_span_attrs_from_context(include_template_variables=True) == {
            STATSIG_ATTR_LLM_PROMPT_NAME: "example-prompt",
            STATSIG_ATTR_LLM_PROMPT_VERSION: "VersionB",
            STATSIG_ATTR_LLM_PROMPT_VERSION_ID: "version-b",
        }

        candidate.get_prompt_messages({"input": "hello"})
        # Rendering a version the scope isn't for leaves the variables alone.
        other.get_prompt_messages({"input": "ignored"})

        attrs = get_prompt_span_attrs_from_context(include_template_variables=True)
        assert attrs[STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES] == {"input": "hello"}
        assert STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES not in get_prompt_span_attrs_from_context(
            include_template_variables=False
        )


def test_get_prompt_messages_outside_a_scope_renders_without_recording_anything():
    version = make_version("version-b", "VersionB")

    assert version.get_prompt_messages({"input": "hi"}) == [
        {"role": "user", "content": "Answer: hi"}
    ]
    assert get_prompt_span_attrs_from_context(include_template_variables=True) == {}


def test_records_a_copy_of_the_variables_and_keeps_empty_ones():
    ctx = set_prompt_version_to_context(otel_context.get_current(), "p", "V", "version-a")
    token = otel_context.attach(ctx)
    try:
        variables: Dict[str, Any] = {}
        record_prompt_template_variables("version-a", variables)
        variables["input"] = "added later"

        attrs = get_prompt_span_attrs_from_context(include_template_variables=True)
        assert attrs[STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES] == {}
    finally:
        otel_context.detach(token)


def test_reads_an_explicit_context():
    ctx = set_prompt_version_to_context(otel_context.get_current(), "p", "V", "version-a")

    assert get_prompt_span_attrs_from_context(ctx, include_template_variables=True) == {
        STATSIG_ATTR_LLM_PROMPT_NAME: "p",
        STATSIG_ATTR_LLM_PROMPT_VERSION: "V",
        STATSIG_ATTR_LLM_PROMPT_VERSION_ID: "version-a",
    }
    assert get_prompt_span_attrs_from_context(include_template_variables=True) == {}


def test_a_prompt_without_a_user_leaves_the_user_context_alone(live):
    with Prompt("no-user", live, []).with_live():
        assert otel_context.get_value(STATSIG_CTX_KEY_ACTIVE_PROMPT) == "no-user"
        assert get_user_from_otel_context() is None


def test_make_prompt_passes_the_user_to_the_prompt(user):
    class FakeParamStore:
        def get_string(self, key, fallback):
            return "prompt_config:live-id" if key == "live" else fallback

        def get_array(self, _key, fallback):
            return fallback

    class FakeStatsigClient:
        def get_dynamic_config(self, _user, config_id):
            return FakeDynamicConfig(config_id, {"name": "Live"})

    prompt = make_prompt(FakeStatsigClient(), "made", FakeParamStore(), user)  # type: ignore[arg-type]

    with prompt.with_live() as version:
        assert version.get_id() == "live-id"
        assert get_user_from_otel_context() == EXPECTED_USER


# Model spans


def test_model_spans_are_tagged_inside_with_live_and_not_outside(tracing, prompt):
    client = openai_client()

    with prompt.with_live():
        chat(client)
    chat(client)

    inside, outside = tracing.spans()
    expected = {
        STATSIG_ATTR_LLM_PROMPT_NAME: "example-prompt",
        STATSIG_ATTR_LLM_PROMPT_VERSION: "VersionA",
        STATSIG_ATTR_LLM_PROMPT_VERSION_ID: "version-a",
    }
    assert prompt_attrs(inside) == expected
    assert prompt_attrs(outside) == {}

    inside_event, outside_event = tracing.statsig.events
    assert prompt_attrs(inside_event) == expected
    assert prompt_attrs(outside_event) == {}


def test_with_version_tags_spans_with_the_version_and_its_id(tracing, prompt):
    candidate = make_version("version-b", "VersionB")

    with prompt.with_version(candidate):
        chat(openai_client())

    [span] = tracing.spans()
    assert span[STATSIG_ATTR_LLM_PROMPT_NAME] == "example-prompt"
    assert span[STATSIG_ATTR_LLM_PROMPT_VERSION] == "VersionB"
    assert span[STATSIG_ATTR_LLM_PROMPT_VERSION_ID] == "version-b"


@pytest.mark.parametrize(
    "capture",
    [GenAICaptureOptions(capture_input_messages=True), GenAICaptureOptions(capture_all=True)],
)
def test_template_variables_are_recorded_when_input_capture_is_on(tracing, prompt, capture):
    with prompt.with_live() as version:
        chat(openai_client(capture), version)

    [span] = tracing.spans()
    [event] = tracing.statsig.events
    variables = span[STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES]
    assert variables == '{"input":"hello"}'
    assert event[STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES] == variables
    assert span[f"{STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES}_len"] == len(variables)
    assert event[f"{STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES}_len"] == str(len(variables))


def test_long_template_variables_are_capped_like_other_json_attributes(tracing, prompt):
    with prompt.with_live() as version:
        chat(openai_client(GenAICaptureOptions(capture_input_messages=True)), version, "x" * 50_000)

    [span] = tracing.spans()
    [event] = tracing.statsig.events
    full_len = len('{"input":"' + "x" * 50_000 + '"}')
    variables = span[STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES]
    assert variables.endswith("…(truncated)")
    assert len(variables) == span_telemetry.MAX_JSON_CHARS + len("…(truncated)")
    assert span[f"{STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES}_len"] == full_len
    assert event[STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES] == variables
    assert event[f"{STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES}_len"] == str(full_len)


@pytest.mark.parametrize(
    "capture",
    [None, GenAICaptureOptions(capture_output_messages=True, capture_system_instructions=True)],
)
def test_template_variables_are_not_recorded_without_input_capture(tracing, prompt, capture):
    with prompt.with_live() as version:
        chat(openai_client(capture), version)

    [span] = tracing.spans()
    [event] = tracing.statsig.events
    assert span[STATSIG_ATTR_LLM_PROMPT_VERSION_ID] == "version-a"
    assert STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES not in span
    assert STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES not in event


def test_rendering_another_version_inside_a_scope_does_not_mislabel_spans(tracing, prompt):
    client = openai_client(GenAICaptureOptions(capture_all=True))
    other = make_version("version-c", "VersionC")

    with prompt.with_live():
        chat(client, other)

    [span] = tracing.spans()
    assert span[STATSIG_ATTR_LLM_PROMPT_VERSION] == "VersionA"
    assert span[STATSIG_ATTR_LLM_PROMPT_VERSION_ID] == "version-a"
    assert STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES not in span


def test_nested_scopes_tag_spans_with_the_innermost_version(tracing, prompt):
    client = openai_client(GenAICaptureOptions(capture_all=True))
    candidate = make_version("version-b", "VersionB")

    with prompt.with_live() as outer:
        outer.get_prompt_messages({"input": "outer"})
        with prompt.with_version(candidate) as inner:
            inner.get_prompt_messages({"input": "inner"})
            client.chat.completions.create(model="gpt-4", messages=[])
        client.chat.completions.create(model="gpt-4", messages=[])

    inner_span, outer_span = tracing.spans()
    assert inner_span[STATSIG_ATTR_LLM_PROMPT_VERSION_ID] == "version-b"
    assert json.loads(inner_span[STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES]) == {"input": "inner"}
    assert outer_span[STATSIG_ATTR_LLM_PROMPT_VERSION_ID] == "version-a"
    assert json.loads(outer_span[STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES]) == {"input": "outer"}


def test_with_version_tags_spans_with_the_prompt_user(tracing, prompt):
    with prompt.with_live():
        chat(openai_client())
    chat(openai_client())

    inside, outside = tracing.spans()
    assert inside[STATSIG_ATTR_USER_ID] == "prompt-user"
    assert json.loads(inside[STATSIG_ATTR_CUSTOM_IDS]) == {"cohort": "beta"}
    assert STATSIG_ATTR_USER_ID not in outside


def test_with_context_tags_spans_with_the_prompt_name_and_user(tracing, prompt):
    with prompt.with_context():
        chat(openai_client(GenAICaptureOptions(capture_all=True)))

    [span] = tracing.spans()
    assert prompt_attrs(span) == {STATSIG_ATTR_LLM_PROMPT_NAME: "example-prompt"}
    assert span[STATSIG_ATTR_USER_ID] == "prompt-user"
