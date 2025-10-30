from statsig_python_core import StatsigOptions, StatsigUser
from pytest_httpserver import HTTPServer
import json
from statsig_ai import PromptVersion, StatsigAI
from statsig_ai.statsig_ai_base import StatsigCreateConfig
from utils import get_test_data_resource
import pytest


@pytest.fixture
def statsig_setup(httpserver: HTTPServer):
    dcs_content = get_test_data_resource("eval_proj_dcs.json")
    json_data = json.loads(dcs_content)

    httpserver.expect_request("/v2/download_config_specs/secret-key.json").respond_with_json(
        json_data
    )

    httpserver.expect_request("/v1/log_event").respond_with_json({"success": True})

    options = StatsigOptions(
        specs_url=httpserver.url_for("/v2/download_config_specs"),
        log_event_url=httpserver.url_for("/v1/log_event"),
    )

    yield options


def test_get_prompt(statsig_setup):
    options = statsig_setup
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
    options = statsig_setup
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
    options = statsig_setup
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
