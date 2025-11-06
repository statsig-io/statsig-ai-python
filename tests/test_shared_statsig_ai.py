from typing import Tuple
import pytest
from mock_scrapi import MockScrapi
from statsig_ai.statsig_ai_base import StatsigAttachConfig, StatsigCreateConfig
from utils import get_test_data_resource
from pytest_httpserver import HTTPServer
from statsig_ai import StatsigAI
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
    statsig_options.specs_url = mock_scrapi.url_for_endpoint("/v2/download_config_specs")
    statsig_options.log_event_url = mock_scrapi.url_for_endpoint("/v1/log_event")

    StatsigAI.remove_shared()

    yield statsig_options, mock_scrapi

    if StatsigAI.has_shared():
        statsig_ai = StatsigAI.shared()
        statsig_ai.shutdown()


def test_creating_shared_instance_with_create_config(
    statsig_setup: Tuple[StatsigOptions, MockScrapi],
):
    statsig_options, _ = statsig_setup

    statsig_ai = StatsigAI.new_shared(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=statsig_options)
    )
    statsig_ai.initialize()
    assert statsig_ai.get_statsig().check_gate(StatsigUser("my_user"), "test_public")


def test_getting_shared_instance_with_create_config(statsig_setup):
    statsig_options, _ = statsig_setup

    statsig_ai = StatsigAI.new_shared(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=statsig_options)
    )
    shared_statsig_ai = StatsigAI.shared()

    assert shared_statsig_ai == statsig_ai

    shared_statsig_ai.initialize()
    gate = shared_statsig_ai.get_statsig().check_gate(StatsigUser("my_user"), "test_public")
    print("hello i am gate here", gate)

    assert shared_statsig_ai.get_statsig().check_gate(StatsigUser("my_user"), "test_public")


def test_removing_shared_instance_with_create_config(statsig_setup):
    statsig_options, _ = statsig_setup

    statsig_ai = StatsigAI.new_shared(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=statsig_options)
    )
    statsig_ai.initialize()
    StatsigAI.remove_shared()

    shared_statsig_ai = StatsigAI.shared()
    assert not shared_statsig_ai.get_statsig().check_gate(StatsigUser("my_user"), "test_public")


def test_getting_shared_instance_with_attach_config(statsig_setup):
    statsig_options, _ = statsig_setup
    statsig = Statsig(sdk_key="secret-key", options=statsig_options)
    StatsigAI.new_shared(statsig_source=StatsigAttachConfig(statsig=statsig))
    shared_statsig_ai = StatsigAI.shared()
    # attach config do not manage the statsig instance lifecycle, so the statsig instance is not initialized
    assert not shared_statsig_ai.get_statsig().check_gate(StatsigUser("my_user"), "test_public")
    statsig.initialize().wait()
    shared_statsig_ai.initialize()

    assert shared_statsig_ai.get_statsig() == statsig
    assert shared_statsig_ai.get_statsig().check_gate(StatsigUser("my_user"), "test_public")


def test_removing_shared_instance_with_attach_config(statsig_setup):
    statsig_options, _ = statsig_setup
    statsig = Statsig(sdk_key="secret-key", options=statsig_options)
    statsig.initialize().wait()
    StatsigAI.new_shared(statsig_source=StatsigAttachConfig(statsig=statsig))
    StatsigAI.shared().initialize()
    StatsigAI.remove_shared()
    assert not StatsigAI.shared().get_statsig().check_gate(StatsigUser("my_user"), "test_public")
