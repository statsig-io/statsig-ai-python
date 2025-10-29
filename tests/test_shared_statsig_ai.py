import pytest
from mock_scrapi import MockScrapi
from utils import get_test_data_resource
from pytest_httpserver import HTTPServer
from statsig_ai import StatsigAI, StatsigAIOptions
from statsig_python_core import Statsig, StatsigOptions, StatsigUser


@pytest.fixture
def statsig_setup(httpserver: HTTPServer):
    mock_scrapi = MockScrapi(httpserver)
    dcs_content = get_test_data_resource("eval_proj_dcs.json")
    mock_scrapi.stub(
        "/v2/download_config_specs/secret-key.json", response=dcs_content, method="GET"
    )
    mock_scrapi.stub("/v1/log_event", response='{"success": true}', method="POST")

    statsig_options = StatsigOptions()
    statsig_options.specs_url = mock_scrapi.url_for_endpoint(
        "/v2/download_config_specs"
    )
    statsig_options.log_event_url = mock_scrapi.url_for_endpoint("/v1/log_event")

    StatsigAI.remove_shared()

    yield statsig_options, mock_scrapi

    if StatsigAI.has_shared_instance():
        statsig = StatsigAI.shared()
        statsig.shutdown().wait()


def test_creating_shared_instance_with_create_config(statsig_setup):
    options, _ = statsig_setup

    statsig = StatsigAI.new_shared({"sdk_key": "secret-key"}, options)
    statsig.initialize().wait()
    assert statsig.get_statsig().check_gate(StatsigUser("my_user"), "test_public")


def test_getting_shared_instance_with_create_config(statsig_setup):
    options, _ = statsig_setup

    statsig = StatsigAI.new_shared({"sdk_key": "secret-key"}, options)
    shared_statsig = StatsigAI.shared()

    assert shared_statsig == statsig

    shared_statsig.initialize().wait()
    assert shared_statsig.get_statsig().check_gate(
        StatsigUser("my_user"), "test_public"
    )


def test_removing_shared_instance_with_create_config(statsig_setup):
    options, _ = statsig_setup

    statsig = StatsigAI.new_shared({"sdk_key": "secret-key"}, options)
    statsig.initialize().wait()
    StatsigAI.remove_shared()

    shared_statsig = StatsigAI.shared()
    assert not shared_statsig.get_statsig().check_gate(
        StatsigUser("my_user"), "test_public"
    )


def test_checking_if_shared_instance_exists_with_create_config():
    StatsigAI.new_shared({"sdk_key": "secret-key"})
    assert StatsigAI.has_shared_instance()

    StatsigAI.remove_shared()
    assert not StatsigAI.has_shared_instance()
