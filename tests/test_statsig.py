from statsig_python_core import StatsigOptions, StatsigUser
from pytest_httpserver import HTTPServer
from statsig_ai import PromptVersion, StatsigAI, StatsigCreateConfig
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
    assert isinstance(prompt.get_candicates(), list)
    assert all(isinstance(c, PromptVersion) for c in prompt.get_candicates())


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


def test_get_prompt_get_candicates(statsig_setup):
    options, _ = statsig_setup
    statsig_ai = StatsigAI(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=options)
    )
    statsig_ai.initialize()
    prompt = statsig_ai.get_prompt(StatsigUser("a-user"), "test_prompt")
    statsig_ai.shutdown()

    candicates = prompt.get_candicates()
    assert len(candicates) == 2
    assert candicates[0].get_id() == "7jszgFEAi1KRA2Tot6qikg"
    assert candicates[0].get_name() == "Version 2"
    assert candicates[1].get_id() == "7CKLvQvOwjj2vjx12gFO0Z"
    assert candicates[1].get_name() == "Version 3"


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
    assert metadata["grader_id"] == "test_grader"
    assert metadata["ai_config_name"] == "test_prompt"
    assert metadata["version_name"] == "Version 1"
    assert metadata["version_id"] == "6KGzeo8TR9JTL7CZl7vccd"
    assert metadata["session_id"] == "1234567890"
