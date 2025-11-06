import pytest
import json
from pytest_httpserver import HTTPServer
from mock_scrapi import MockScrapi
from utils import get_test_data_resource
from statsig_ai import (
    StatsigOptions,
    StatsigAI,
    StatsigCreateConfig,
)
from statsig_ai.wrappers import wrap_openai, WrapOpenAIOptions, GenAICaptureOptions
from mock_openai import MockOpenAI


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


def test_wrap_openai(statsig_setup):
    oai_client = MockOpenAI()
    wrapped_client = wrap_openai(oai_client)
    assert wrapped_client is not None


def test_wrap_openai_chat_completions(statsig_setup):
    options, mock_scrapi = statsig_setup
    StatsigAI.new_shared(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=options)
    )
    StatsigAI.shared().initialize()
    oai_client = MockOpenAI()
    wrapped_client = wrap_openai(oai_client)
    response = wrapped_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello, world!"}],
        temperature=0.7,
        max_tokens=100,
    )
    StatsigAI.shared().shutdown()
    assert response["choices"][0]["message"]["content"] == "This is a mock chat response"
    gen_ai_events = [
        e for e in mock_scrapi.get_logged_events() if e["eventName"] == "statsig::gen_ai"
    ]
    assert len(gen_ai_events) == 1
    assert gen_ai_events[0]["eventName"] == "statsig::gen_ai"
    metadata = gen_ai_events[0]["metadata"]
    print(metadata)
    assert metadata["gen_ai.provider.name"] == "openai"
    assert metadata["gen_ai.operation.name"] == "chat"
    assert metadata["gen_ai.request.model"] == "gpt-4"
    assert metadata["gen_ai.response.model"] == "gpt-4"
    assert metadata["gen_ai.response.finish_reasons"] == json.dumps(["stop"])
    assert metadata["gen_ai.usage.input_tokens"] == "5"
    assert metadata["gen_ai.usage.output_tokens"] == "7"


def test_respects_capture_options(statsig_setup):
    options, mock_scrapi = statsig_setup
    StatsigAI.new_shared(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=options)
    )
    StatsigAI.shared().initialize()
    oai_client = MockOpenAI()
    wrapped_client = wrap_openai(
        oai_client,
        options=WrapOpenAIOptions(
            gen_ai_capture_options=GenAICaptureOptions(capture_input_messages=True)
        ),
    )
    response = wrapped_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello, world!"}],
        temperature=0.7,
        max_tokens=100,
    )
    StatsigAI.shared().shutdown()
    assert response["choices"][0]["message"]["content"] == "This is a mock chat response"
    gen_ai_events = [
        e for e in mock_scrapi.get_logged_events() if e["eventName"] == "statsig::gen_ai"
    ]
    assert len(gen_ai_events) == 1
    assert gen_ai_events[0]["eventName"] == "statsig::gen_ai"
    metadata = gen_ai_events[0]["metadata"]
    assert metadata["gen_ai.input.messages"] == json.dumps(
        [{"role": "user", "content": "Hello, world!"}]
    )
    assert metadata.get("gen_ai.output.messages") is None
