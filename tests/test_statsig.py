from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import format_span_id, format_trace_id
from statsig_python_core import StatsigOptions, StatsigUser
from pytest_httpserver import HTTPServer
from statsig_ai import (
    AgentConfig,
    AgentVersion,
    OtelSingleton,
    PromptVersion,
    StatsigAI,
    StatsigCreateConfig,
)
from utils import get_test_data_resource
import pytest
from mock_scrapi import MockScrapi


@pytest.fixture
def statsig_setup(httpserver: HTTPServer):
    mock_scrapi = MockScrapi(httpserver)
    dcs_content = get_test_data_resource("eval_proj_dcs.json")

    mock_scrapi.stub(
        "/v2/download_config_specs/secret-key.json", response=dcs_content, method="GET"
    )
    mock_scrapi.stub("/v1/log_event", response='{"success": true}', method="POST")

    options = StatsigOptions(
        specs_url=mock_scrapi.url_for_endpoint("/v2/download_config_specs"),
        log_event_url=mock_scrapi.url_for_endpoint("/v1/log_event"),
    )

    yield options, mock_scrapi


def test_get_prompt(statsig_setup):
    options, _ = statsig_setup
    statsig_ai = StatsigAI(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=options)
    )
    statsig_ai.initialize()
    prompt = statsig_ai.get_prompt(StatsigUser("a-user"), "test_prompt")
    statsig_ai.shutdown()
    assert prompt.get_name() == "test_prompt"
    assert isinstance(prompt.get_live(), PromptVersion)
    assert isinstance(prompt.get_candidates(), list)
    assert all(isinstance(c, PromptVersion) for c in prompt.get_candidates())


def test_get_prompt_get_live(statsig_setup):
    options, _ = statsig_setup
    statsig_ai = StatsigAI(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=options)
    )
    statsig_ai.initialize()
    prompt = statsig_ai.get_prompt(StatsigUser("a-user"), "test_prompt")
    statsig_ai.shutdown()

    live_version = prompt.get_live()
    assert live_version is not None
    assert live_version.get_id() == "6KGzeo8TR9JTL7CZl7vccd"
    assert live_version.get_name() == "Version 1"
    assert live_version.get_temperature() == 1
    assert live_version.get_max_tokens() == 1000
    assert live_version.get_top_p() == 1
    assert live_version.get_frequency_penalty() == 0
    assert live_version.get_presence_penalty() == 0
    assert live_version.get_provider() == "openai"
    assert live_version.get_model() == "gpt-5"
    assert isinstance(live_version.get_workflow_body(), dict)
    assert live_version.get_eval_model() == "gpt-4o-mini"
    assert live_version.get_type() == "Live"
    assert live_version.get_prompt_name() == "test_prompt"


def test_get_prompt_get_live_objects_and_arrays(statsig_setup):
    options, _ = statsig_setup
    statsig_ai = StatsigAI(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=options)
    )
    statsig_ai.initialize()
    prompt = statsig_ai.get_prompt(StatsigUser("a-user"), "test_prompt")
    statsig_ai.shutdown()

    live_version = prompt.get_live()
    assert live_version.get_prompt_messages({}) == [{"role": "system", "content": ""}]
    assert live_version.get_workflow_body() == {}
    assert live_version.get_object("missing", {"a": 1}) == {"a": 1}
    assert live_version.get_array("missing", [1]) == [1]

    # Mutating a returned value must not leak into the config
    live_version.get_workflow_body()["mutated"] = True
    live_version.get_array("prompts", [])[0]["content"] = "mutated"
    assert live_version.get_workflow_body() == {}
    assert live_version.get_array("prompts", []) == [{"role": "system", "content": ""}]


def test_get_prompt_get_candidates(statsig_setup):
    options, _ = statsig_setup
    statsig_ai = StatsigAI(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=options)
    )
    statsig_ai.initialize()
    prompt = statsig_ai.get_prompt(StatsigUser("a-user"), "test_prompt")
    statsig_ai.shutdown()

    candidates = prompt.get_candidates()
    assert len(candidates) == 2
    assert candidates[0].get_id() == "7jszgFEAi1KRA2Tot6qikg"
    assert candidates[0].get_name() == "Version 2"
    assert candidates[1].get_id() == "7CKLvQvOwjj2vjx12gFO0Z"
    assert candidates[1].get_name() == "Version 3"


def test_log_eval_grade(statsig_setup):
    options, mock_scrapi = statsig_setup
    statsig_ai = StatsigAI(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=options)
    )
    statsig_ai.initialize()
    prompt_version = statsig_ai.get_prompt(StatsigUser("a-user"), "test_prompt").get_live()
    statsig_ai.log_eval_grade(
        StatsigUser("a-user"), prompt_version, 0.5, "test_grader", {"session_id": "1234567890"}
    )
    statsig_ai.flush_events()

    logged_events = mock_scrapi.get_logged_events()
    eval_event = list(filter(lambda e: e["eventName"] == "statsig::eval_result", logged_events))
    assert len(eval_event) == 1
    assert eval_event[0]["eventName"] == "statsig::eval_result"
    assert eval_event[0]["user"]["userID"] == "a-user"
    assert eval_event[0]["value"] == "test_prompt"
    metadata = eval_event[0]["metadata"]
    assert metadata["score"] == "0.5"
    assert metadata["grader_name"] == "test_grader"
    assert "grader_id" not in metadata
    assert metadata["ai_config_name"] == "test_prompt"
    assert metadata["version_name"] == "Version 1"
    assert metadata["version_id"] == "6KGzeo8TR9JTL7CZl7vccd"
    assert metadata["session_id"] == "1234567890"
    assert metadata["is_live"] == "true"
    assert metadata["source"] == "sdk"
    assert "trace_id" not in metadata
    assert "span_id" not in metadata
    assert "reasoning" not in metadata


def _log_eval_grade_and_get_event(
    statsig_setup, eval_data, user=None, use_candidate=False, span_name=None
):
    options, mock_scrapi = statsig_setup
    statsig_ai = StatsigAI(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=options)
    )
    statsig_ai.initialize()
    user = user or StatsigUser("a-user")
    prompt = statsig_ai.get_prompt(user, "test_prompt")
    prompt_version = prompt.get_candidates()[0] if use_candidate else prompt.get_live()

    span_context = None
    if span_name:
        tracer = TracerProvider().get_tracer("test")
        with tracer.start_as_current_span(span_name) as span:
            span_context = span.get_span_context()
            statsig_ai.log_eval_grade(user, prompt_version, 1.0, "test_grader", eval_data)
    else:
        statsig_ai.log_eval_grade(user, prompt_version, 1.0, "test_grader", eval_data)
    statsig_ai.shutdown()

    eval_events = [
        e for e in mock_scrapi.get_logged_events() if e["eventName"] == "statsig::eval_result"
    ]
    assert len(eval_events) == 1
    return eval_events[0], span_context


def test_log_eval_grade_candidate_is_not_live(statsig_setup):
    event, _ = _log_eval_grade_and_get_event(statsig_setup, {}, use_candidate=True)
    assert event["metadata"]["is_live"] == "false"
    assert event["metadata"]["version_name"] == "Version 2"
    assert event["metadata"]["session_id"] == ""


def test_log_eval_grade_uses_active_span(statsig_setup):
    event, span_context = _log_eval_grade_and_get_event(statsig_setup, {}, span_name="llm-call")
    metadata = event["metadata"]
    assert metadata["trace_id"] == format_trace_id(span_context.trace_id)
    assert metadata["span_id"] == format_span_id(span_context.span_id)
    assert len(metadata["trace_id"]) == 32
    assert len(metadata["span_id"]) == 16


def test_log_eval_grade_passed_ids_override_active_span(statsig_setup):
    event, _ = _log_eval_grade_and_get_event(
        statsig_setup,
        {"trace_id": "0af7651916cd43dd8448eb211c80319c", "span_id": "b7ad6b7169203331"},
        span_name="later-span",
    )
    assert event["metadata"]["trace_id"] == "0af7651916cd43dd8448eb211c80319c"
    assert event["metadata"]["span_id"] == "b7ad6b7169203331"


def test_log_eval_grade_reasoning_metadata_and_activity_id(statsig_setup):
    event, _ = _log_eval_grade_and_get_event(
        statsig_setup,
        {
            "session_id": "s-1",
            "activity_id": "activity-123",
            "reasoning": "Answered the question.",
            "metadata": {"judge_model": "gpt-4.1", "source": "custom"},
        },
        user=StatsigUser("a-user", custom_ids={"companyID": "c-1"}),
    )
    metadata = event["metadata"]
    assert metadata["reasoning"] == "Answered the question."
    assert metadata["judge_model"] == "gpt-4.1"
    # Caller metadata is merged last, as in the Node SDK.
    assert metadata["source"] == "custom"
    assert event["user"]["customIDs"] == {
        "companyID": "c-1",
        "statsig.activity_id": "activity-123",
    }


def _get_agent_config(statsig_setup, agent_config_name="test_agent"):
    options, _ = statsig_setup
    statsig_ai = StatsigAI(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=options)
    )
    statsig_ai.initialize()
    agent_config = statsig_ai.get_agent_config(StatsigUser("a-user"), agent_config_name)
    statsig_ai.shutdown()
    return agent_config


def test_get_agent_config(statsig_setup):
    agent_config = _get_agent_config(statsig_setup)
    assert isinstance(agent_config, AgentConfig)
    assert agent_config.get_name() == "test_agent"
    assert isinstance(agent_config.get_live(), AgentVersion)
    assert len(agent_config.get_candidates()) == 2
    assert all(isinstance(c, AgentVersion) for c in agent_config.get_candidates())


def test_get_agent_config_live_root(statsig_setup):
    root = _get_agent_config(statsig_setup).get_live().get_root()
    assert isinstance(root, PromptVersion)
    assert root.get_id() == "2nKw6Ty8VbR3mXc1Jq5Lh7"
    assert root.get_name() == "Agent Version 1"
    assert root.get_type() == "Live"
    assert root.get_prompt_name() == "test_agent"
    assert root.get_provider() == "openai"
    assert root.get_model() == "gpt-4.1"
    assert root.get_temperature() == 0.5
    assert root.get_max_tokens() == 2000
    assert root.get_top_p() == 1
    assert root.get_frequency_penalty() == 0
    assert root.get_presence_penalty() == 0
    assert root.get_eval_model() == "gpt-4o-mini"
    assert root.get_workflow_body() == {}
    assert root.get_prompt_messages({"input": "Where is my order?"}) == [
        {"role": "system", "content": "You are a support agent. Where is my order?"}
    ]
    assert root.is_live_for_user is True


def test_get_agent_config_candidate_roots(statsig_setup):
    candidates = _get_agent_config(statsig_setup).get_candidates()
    roots = [c.get_root() for c in candidates]
    assert [r.get_id() for r in roots] == ["6sPd0Hz4QfM9kWa2Ny8Ec3", "4tLj9Gv1XeB7rUo5Kc2Sm6"]
    assert [r.get_name() for r in roots] == ["Agent Version 2", "Agent Version 3"]
    assert [r.get_model() for r in roots] == ["gpt-5", "gpt-5-mini"]
    assert all(r.get_type() == "Candidate" for r in roots)
    assert all(r.get_prompt_name() == "test_agent" for r in roots)
    assert all(r.is_live_for_user is False for r in roots)


def test_get_agent_config_missing(statsig_setup):
    # As in the Node SDK, a missing agent config still returns an AgentConfig: the live
    # version's root is an empty prompt version, and there are no candidates.
    agent_config = _get_agent_config(statsig_setup, "missing_agent")
    assert agent_config.get_name() == "missing_agent"
    assert agent_config.get_candidates() == []
    root = agent_config.get_live().get_root()
    assert isinstance(root, PromptVersion)
    assert root.get_id() == ""
    assert root.get_name() == ""
    assert root.get_prompt_name() == ""
    assert root.get_model() == ""
    assert root.is_live_for_user is True


def test_get_agent_config_is_not_a_prompt(statsig_setup):
    # Agent configs live under `agent:<name>`, so a prompt's name doesn't resolve to one.
    agent_config = _get_agent_config(statsig_setup, "test_prompt")
    assert agent_config.get_live().get_root().get_id() == ""
    assert agent_config.get_candidates() == []


@pytest.mark.parametrize(
    "use_candidate, expected_is_live, expected_version_name, expected_version_id",
    [
        (False, "true", "Agent Version 1", "2nKw6Ty8VbR3mXc1Jq5Lh7"),
        (True, "false", "Agent Version 2", "6sPd0Hz4QfM9kWa2Ny8Ec3"),
    ],
)
def test_log_eval_grade_with_agent_root(
    statsig_setup, use_candidate, expected_is_live, expected_version_name, expected_version_id
):
    options, mock_scrapi = statsig_setup
    statsig_ai = StatsigAI(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=options)
    )
    statsig_ai.initialize()
    user = StatsigUser("a-user")
    agent_config = statsig_ai.get_agent_config(user, "test_agent")
    version = agent_config.get_candidates()[0] if use_candidate else agent_config.get_live()
    statsig_ai.log_eval_grade(
        user, version.get_root(), 0.75, "test_grader", {"session_id": "s-1"}
    )
    statsig_ai.shutdown()

    eval_events = [
        e for e in mock_scrapi.get_logged_events() if e["eventName"] == "statsig::eval_result"
    ]
    assert len(eval_events) == 1
    assert eval_events[0]["value"] == "test_agent"
    metadata = eval_events[0]["metadata"]
    assert metadata["is_live"] == expected_is_live
    assert metadata["version_name"] == expected_version_name
    assert metadata["version_id"] == expected_version_id
    assert metadata["ai_config_name"] == "test_agent"
    assert metadata["score"] == "0.75"
    assert metadata["grader_name"] == "test_grader"
    assert metadata["session_id"] == "s-1"


class _FlushRecordingProvider(TracerProvider):
    def __init__(self):
        super().__init__()
        self.flush_count = 0

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self.flush_count += 1
        return True


@pytest.fixture
def otel_singleton_provider():
    original_instance = OtelSingleton._instance
    OtelSingleton._instance = None
    provider = _FlushRecordingProvider()
    OtelSingleton.instantiate(provider)
    yield provider
    OtelSingleton._instance = original_instance


def test_flush_events_and_shutdown_flush_tracer_provider(statsig_setup, otel_singleton_provider):
    options, _ = statsig_setup
    statsig_ai = StatsigAI(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=options)
    )
    statsig_ai.initialize()
    statsig_ai.flush_events()
    assert otel_singleton_provider.flush_count == 1
    statsig_ai.shutdown()
    assert otel_singleton_provider.flush_count == 2
