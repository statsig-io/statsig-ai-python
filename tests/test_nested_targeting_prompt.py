from statsig_python_core import Statsig, StatsigOptions, StatsigUser
from pytest_httpserver import HTTPServer
import json
from statsig_ai import PromptVersion, StatsigAI
from statsig_ai.statsig_ai_base import StatsigCreateConfig
from utils import get_test_data_resource
import pytest


@pytest.fixture
def statsig_setup(httpserver: HTTPServer):
    dcs_content = get_test_data_resource("eval_proj_dcs_targeting.json")
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
