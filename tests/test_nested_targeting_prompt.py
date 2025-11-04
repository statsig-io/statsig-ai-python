from statsig_python_core import StatsigOptions, StatsigUser
from pytest_httpserver import HTTPServer
from statsig_ai import StatsigAI
from statsig_ai.statsig_ai_base import StatsigCreateConfig
from mock_scrapi import MockScrapi
from utils import get_test_data_resource
import pytest


@pytest.fixture
def statsig_setup(httpserver: HTTPServer):
    mock_scrapi = MockScrapi(httpserver)
    dcs_content = get_test_data_resource("eval_proj_dcs_targeting.json")

    mock_scrapi.stub(
        "/v2/download_config_specs/secret-key.json", response=dcs_content, method="GET"
    )
    mock_scrapi.stub("/v1/log_event", response='{"success": true}', method="POST")

    options = StatsigOptions(
        specs_url=mock_scrapi.url_for_endpoint("/v2/download_config_specs"),
        log_event_url=mock_scrapi.url_for_endpoint("/v1/log_event"),
    )

    yield options, mock_scrapi


def test_get_prompt_with_targeting(statsig_setup):
    options, _ = statsig_setup
    statsig_ai = StatsigAI(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=options)
    )
    statsig_ai.initialize()
    prompt = statsig_ai.get_prompt(StatsigUser("test-user"), "test-prompt-1")
    statsig_ai.shutdown()
    live_version = prompt.get_live()
    assert live_version is not None
    assert live_version.get_id() == "2QGncLi0YSYj9zavJ825qB"
    assert live_version.get_name() == "Version 1"
    assert live_version.get_temperature() == 0
    assert live_version.get_max_tokens() == 1000
    assert live_version.get_top_p() == 1
    assert live_version.get_frequency_penalty() == 0
    assert live_version.get_presence_penalty() == 0
    assert live_version.get_provider() == "openai"
    assert live_version.get_model() == "gpt-4o"
    assert live_version.get_prompt_name() == "test-prompt-2"


def test_get_prompt_with_nested_targeting(statsig_setup):
    options, _ = statsig_setup
    statsig_ai = StatsigAI(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=options)
    )
    statsig_ai.initialize()
    prompt = statsig_ai.get_prompt(StatsigUser("test-user-1"), "test-prompt-1")
    statsig_ai.shutdown()
    live_version = prompt.get_live()
    assert live_version is not None
    assert live_version.get_id() == "36qqmfh8wHcKzjmX2My2RQ"
    assert live_version.get_name() == "Version 1"
    assert live_version.get_temperature() == 2
    assert live_version.get_max_tokens() == 1000
    assert live_version.get_top_p() == 1
    assert live_version.get_frequency_penalty() == 0
    assert live_version.get_presence_penalty() == 0
    assert live_version.get_provider() == "openai"
    assert live_version.get_model() == "gpt-4.1"
    assert live_version.get_prompt_name() == "test-prompt-3"


def test_get_prompt_with_circular_targeting_without_crashing(statsig_setup, capsys):
    options, _ = statsig_setup
    statsig_ai = StatsigAI(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=options)
    )
    statsig_ai.initialize()
    prompt = statsig_ai.get_prompt(StatsigUser("test-user-circular"), "test-prompt-circular-a")
    statsig_ai.shutdown()
    captured = capsys.readouterr()
    assert (
        "[Statsig] Max targeting depth (300) reached while resolving prompt: test-prompt-circular-a."
        in captured.out
    )
    assert (
        'Possible circular reference starting from "prompt:test-prompt-circular-a".' in captured.out
    )
    live_version = prompt.get_live()
    assert live_version is not None
    assert live_version.get_name() == "Circular A Version"
    assert live_version.get_temperature() == 1
    assert live_version.get_max_tokens() == 500
    assert live_version.get_top_p() == 1
    assert live_version.get_prompt_name() == "test-prompt-circular-a"
