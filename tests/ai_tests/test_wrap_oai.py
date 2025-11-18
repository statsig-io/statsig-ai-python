import pytest
import json
import os
import sys
import asyncio
from typing import Dict, Any, Optional
from pytest_httpserver import HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mock_scrapi import MockScrapi
from utils import get_test_data_resource

from statsig_ai import (
    StatsigOptions,
    StatsigAI,
    StatsigCreateConfig,
    WrapOpenAIOptions,
    GenAICaptureOptions,
)
from statsig_ai import wrap_openai
from openai import OpenAI, AsyncOpenAI

TEST_MODEL = "gpt-4o-mini"
TEST_EMBEDDING_MODEL = "text-embedding-3-small"

OPERATION_REQUIRED_ATTRIBUTES = {
    "openai.chat.completions.create": {
        "id": True,
        "finish_reasons": True,
        "output_tokens": True,
        "otel_semantic_name": "chat",
    },
    "openai.completions.create": {
        "id": True,
        "finish_reasons": True,
        "output_tokens": True,
        "otel_semantic_name": "text_completion",
    },
    "openai.embeddings.create": {
        "id": False,
        "finish_reasons": False,
        "output_tokens": False,
        "otel_semantic_name": "embeddings",
    },
    "openai.responses.create": {
        "id": True,
        "finish_reasons": False,
        "output_tokens": True,
        "otel_semantic_name": "responses.create",
    },
}


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

    StatsigAI.new_shared(
        statsig_source=StatsigCreateConfig(sdk_key="secret-key", statsig_options=options),
    )

    yield mock_scrapi

    StatsigAI.shared().shutdown()
    StatsigAI.remove_shared()


def test_wrap_oai(statsig_setup):
    mock_scrapi = statsig_setup
    oai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    options = WrapOpenAIOptions()
    wrapped_client = wrap_openai(oai_client, options)
    assert wrapped_client is not None


class OpenAITestCase:
    def __init__(
        self,
        name: str,
        operation: str,
        client_type: str,  # "sync" or "async"
        is_stream: bool,
        api_method: str,  # e.g., "chat.completions.create"
        kwargs: Dict[str, Any],
        capture_options: Optional[GenAICaptureOptions] = None,
    ):
        self.name = name
        self.operation = operation
        self.client_type = client_type
        self.is_stream = is_stream
        self.api_method = api_method
        self.kwargs = kwargs
        self.capture_options = capture_options


TEST_CASES = [
    OpenAITestCase(
        name="chat_sync_regular",
        operation="openai.chat.completions.create",
        client_type="sync",
        is_stream=False,
        api_method="chat.completions.create",
        kwargs={
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "Say hello"}],
        },
    ),
    OpenAITestCase(
        name="chat_sync_stream",
        operation="openai.chat.completions.create",
        client_type="sync",
        is_stream=True,
        api_method="chat.completions.create",
        kwargs={
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ),
    OpenAITestCase(
        name="chat_async_regular",
        operation="openai.chat.completions.create",
        client_type="async",
        is_stream=False,
        api_method="chat.completions.create",
        kwargs={
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "Say hello"}],
        },
    ),
    OpenAITestCase(
        name="chat_async_stream",
        operation="openai.chat.completions.create",
        client_type="async",
        is_stream=True,
        api_method="chat.completions.create",
        kwargs={
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ),
    OpenAITestCase(
        name="embeddings_sync",
        operation="openai.embeddings.create",
        client_type="sync",
        is_stream=False,
        api_method="embeddings.create",
        kwargs={
            "model": TEST_EMBEDDING_MODEL,
            "input": "Hello world",
            "dimensions": 256,
            "encoding_format": "float",
        },
    ),
    OpenAITestCase(
        name="embeddings_async",
        operation="openai.embeddings.create",
        client_type="async",
        is_stream=False,
        api_method="embeddings.create",
        kwargs={
            "model": TEST_EMBEDDING_MODEL,
            "input": "Hello world",
            "dimensions": 512,
            "encoding_format": "float",
        },
    ),
    OpenAITestCase(
        name="chat_capture_input_messages",
        operation="openai.chat.completions.create",
        client_type="sync",
        is_stream=False,
        api_method="chat.completions.create",
        kwargs={
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "Test input capture"}],
        },
        capture_options=GenAICaptureOptions(capture_input_messages=True),
    ),
    OpenAITestCase(
        name="chat_capture_output_messages",
        operation="openai.chat.completions.create",
        client_type="sync",
        is_stream=False,
        api_method="chat.completions.create",
        kwargs={
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "Test output capture"}],
        },
        capture_options=GenAICaptureOptions(capture_output_messages=True),
    ),
    OpenAITestCase(
        name="chat_capture_all",
        operation="openai.chat.completions.create",
        client_type="sync",
        is_stream=False,
        api_method="chat.completions.create",
        kwargs={
            "model": TEST_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Test capture all - what's the temperature?"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_temperature",
                        "description": "Get the current temperature",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            },
                        },
                    },
                }
            ],
        },
        capture_options=GenAICaptureOptions(capture_all=True),
    ),
    OpenAITestCase(
        name="chat_capture_system_instructions",
        operation="openai.chat.completions.create",
        client_type="sync",
        is_stream=False,
        api_method="chat.completions.create",
        kwargs={
            "model": TEST_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Test system instruction capture"},
            ],
        },
        capture_options=GenAICaptureOptions(capture_system_instructions=True),
    ),
    OpenAITestCase(
        name="chat_capture_tool_definitions",
        operation="openai.chat.completions.create",
        client_type="sync",
        is_stream=False,
        api_method="chat.completions.create",
        kwargs={
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
        },
        capture_options=GenAICaptureOptions(capture_tool_definitions=True),
    ),
    OpenAITestCase(
        name="chat_no_capture_options",
        operation="openai.chat.completions.create",
        client_type="sync",
        is_stream=False,
        api_method="chat.completions.create",
        kwargs={
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "Test no capture"}],
        },
        capture_options=GenAICaptureOptions(),
    ),
]


def validate_base_attributes(metadata: Dict[str, str], operation: str):
    assert "gen_ai.provider.name" in metadata, "Missing provider name"
    assert metadata["gen_ai.provider.name"] == "openai"

    assert "gen_ai.operation.name" in metadata, "Missing operation name"
    requirements = OPERATION_REQUIRED_ATTRIBUTES.get(operation, {})
    expected_otel_name = requirements.get("otel_semantic_name", operation)
    assert (
        metadata["gen_ai.operation.name"] == expected_otel_name
    ), f"Expected operation name '{expected_otel_name}', got '{metadata['gen_ai.operation.name']}'"

    assert "gen_ai.operation.source_name" in metadata, "Missing operation source name"
    assert (
        metadata["gen_ai.operation.source_name"] == operation
    ), f"Expected source name '{operation}', got '{metadata['gen_ai.operation.source_name']}'"

    assert "gen_ai.request.model" in metadata, "Missing request model"

    assert "span.trace_id" in metadata, "Missing trace ID"
    assert "span.span_id" in metadata, "Missing span ID"
    assert "span.status_code" in metadata, "Missing status code"


def validate_response_attributes(
    metadata: Dict[str, str],
    operation: str,
):
    requirements = OPERATION_REQUIRED_ATTRIBUTES.get(operation, {})

    if requirements.get("id"):
        assert "gen_ai.response.id" in metadata, f"Missing response ID for {operation}"

    if requirements.get("finish_reasons"):
        assert (
            "gen_ai.response.finish_reasons" in metadata
        ), f"Missing finish reasons for {operation}"


def validate_duration_attributes(metadata: Dict[str, str], is_stream: bool):
    assert "span.status_code" in metadata
    if is_stream:
        assert (
            "statsig.gen_ai.server.time_to_first_token_ms" in metadata
        ), "Missing statsig.gen_ai.server.time_to_first_token_ms"
        # Validate it's a positive number
        ttft_ms = float(metadata["statsig.gen_ai.server.time_to_first_token_ms"])
        assert ttft_ms > 0, f"Time to first token should be positive, got {ttft_ms}"


def validate_optional_attributes(metadata: Dict[str, str], operation: str):
    if "gen_ai.usage.input_tokens" in metadata:
        assert isinstance(int(metadata["gen_ai.usage.input_tokens"]), int)

    if "gen_ai.usage.output_tokens" in metadata:
        assert isinstance(int(metadata["gen_ai.usage.output_tokens"]), int)


def validate_embeddings_attributes(metadata: Dict[str, str], kwargs: Dict[str, Any]):
    if "dimensions" in kwargs:
        assert (
            "gen_ai.embeddings.dimension.count" in metadata
        ), "Missing 'gen_ai.embeddings.dimension.count' for embeddings operation with dimensions"
        dimension_count = int(metadata["gen_ai.embeddings.dimension.count"])
        assert (
            dimension_count == kwargs["dimensions"]
        ), f"Expected dimension count {kwargs['dimensions']}, got {dimension_count}"

    if "encoding_format" in kwargs:
        assert (
            "gen_ai.request.encoding_formats" in metadata
        ), "Missing 'gen_ai.request.encoding_formats' for embeddings operation with encoding_format"
        encoding_format = metadata["gen_ai.request.encoding_formats"]
        assert (
            encoding_format == kwargs["encoding_format"]
        ), f"Expected encoding format '{kwargs['encoding_format']}', got '{encoding_format}'"


def validate_capture_options(
    metadata: Dict[str, str],
    capture_options: Optional[GenAICaptureOptions],
):
    if capture_options is None:
        return

    if capture_options.capture_input_messages or capture_options.capture_all:
        assert (
            "gen_ai.input.messages" in metadata
        ), "Missing 'gen_ai.input.messages' when capture_input_messages is enabled"
        input_messages = json.loads(metadata["gen_ai.input.messages"])
        assert isinstance(input_messages, list), "gen_ai.input.messages should be a list"

    if capture_options.capture_output_messages or capture_options.capture_all:
        assert (
            "gen_ai.output.messages" in metadata
        ), "Missing 'gen_ai.output.messages' when capture_output_messages is enabled"
        output_messages = json.loads(metadata["gen_ai.output.messages"])
        assert isinstance(output_messages, list), "gen_ai.output.messages should be a list"

    if capture_options.capture_system_instructions or capture_options.capture_all:
        if "gen_ai.system.instructions" in metadata:
            system_instructions = json.loads(metadata["gen_ai.system.instructions"])
            assert isinstance(
                system_instructions, list
            ), "gen_ai.system.instructions should be a list"

    if capture_options.capture_tool_definitions or capture_options.capture_all:
        assert (
            "gen_ai.tool.definitions" in metadata
        ), "Missing 'gen_ai.tool.definitions' when capture_tool_definitions is enabled"
        tool_definitions = json.loads(metadata["gen_ai.tool.definitions"])
        assert isinstance(tool_definitions, list), "gen_ai.tool.definitions should be a list"
        if len(tool_definitions) > 0:
            assert "type" in tool_definitions[0], "Tool should have a 'type' field"

    if not capture_options.capture_input_messages and not capture_options.capture_all:
        assert (
            "gen_ai.input.messages" not in metadata
        ), "gen_ai.input.messages should not be present when capture_input_messages is disabled"

    if not capture_options.capture_output_messages and not capture_options.capture_all:
        assert (
            "gen_ai.output.messages" not in metadata
        ), "gen_ai.output.messages should not be present when capture_output_messages is disabled"


def get_method_from_client(client, api_method: str):
    parts = api_method.split(".")
    obj = client
    for part in parts:
        obj = getattr(obj, part)
    return obj


@pytest.mark.asyncio
@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc.name)
async def test_openai_wrapper_matrix(statsig_setup, test_case: OpenAITestCase):
    """
    Test OpenAI wrapper with various operations, client types, and modes.

    This test:
    1. Creates the appropriate client (sync/async)
    2. Wraps it with telemetry tracking
    3. Executes the API call
    4. Validates that telemetry events are logged correctly
    5. Verifies operation-specific attributes are present
    """
    mock_scrapi = statsig_setup

    print(f"\n{'='*60}")
    print(f"Testing: {test_case.name}")
    print(f"  Operation: {test_case.operation}")
    print(f"  Client: {test_case.client_type}")
    print(f"  Streaming: {test_case.is_stream}")
    print(f"{'='*60}")

    mock_scrapi.reset()

    if test_case.client_type == "sync":
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    if test_case.capture_options:
        wrap_options = WrapOpenAIOptions(gen_ai_capture_options=test_case.capture_options)
    else:
        wrap_options = WrapOpenAIOptions()
    wrapped_client = wrap_openai(client, wrap_options)

    method = get_method_from_client(wrapped_client, test_case.api_method)

    if test_case.client_type == "async":
        result = await method(**test_case.kwargs)
    else:
        result = method(**test_case.kwargs)

    if test_case.is_stream:
        chunk_count = 0
        if test_case.client_type == "async":
            async for chunk in result:
                chunk_count += 1
        else:
            for chunk in result:
                chunk_count += 1
        print(f"  Consumed {chunk_count} stream chunks")

    await asyncio.sleep(0.5)
    StatsigAI.shared().flush_events()

    events = mock_scrapi.get_logged_events()
    gen_ai_events = [e for e in events if e.get("eventName") == "statsig::gen_ai"]

    assert len(gen_ai_events) > 0, (
        f"No gen_ai events logged for {test_case.name}. " f"Total events logged: {len(events)}"
    )

    event = gen_ai_events[-1]

    assert "eventName" in event, "Event missing 'eventName' field"
    assert (
        event["eventName"] == "statsig::gen_ai"
    ), f"Expected event name 'statsig::gen_ai', got '{event['eventName']}'"
    assert "value" in event, "Event missing 'value' field"
    assert "metadata" in event, "Event missing 'metadata' field"

    metadata = event["metadata"]

    print(f"\n  Validating attributes...")
    validate_base_attributes(metadata, test_case.operation)
    print(f"    ✓ Base attributes validated")

    validate_response_attributes(metadata, test_case.operation)
    print(f"    ✓ Response attributes validated")

    validate_duration_attributes(metadata, test_case.is_stream)
    print(f"    ✓ Duration attributes validated")

    validate_optional_attributes(metadata, test_case.operation)
    print(f"    ✓ Optional attributes validated")

    if "embeddings" in test_case.operation:
        validate_embeddings_attributes(metadata, test_case.kwargs)
        print(f"    ✓ Embeddings-specific attributes validated")

    if test_case.capture_options:
        validate_capture_options(metadata, test_case.capture_options)
        print(f"    ✓ Capture options validated")

    print(f"\n  ✅ {test_case.name} PASSED")
    print(f"  Metadata keys: {sorted(metadata.keys())}")
    print(f"{'='*60}\n")
