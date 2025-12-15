import gzip

import pytest
from pytest_httpserver import HTTPServer

from statsig_ai.otel.exporter import StatsigOTLPTraceExporterOptions
from statsig_ai.otel.otel import InitializeTracingOptions, initialize_tracing

from mock_scrapi import MockScrapi


@pytest.fixture(autouse=True)
def reset_otel_singleton():
    from statsig_ai.otel import singleton as singleton_module

    original_instance = getattr(singleton_module.OtelSingleton, "_instance", None)
    singleton_module.OtelSingleton._instance = None

    yield

    singleton_module.OtelSingleton._instance = original_instance


@pytest.fixture
def mock_scrapi(httpserver: HTTPServer) -> MockScrapi:
    return MockScrapi(httpserver)


def test_exporter_sends_gzipped_requests(mock_scrapi: MockScrapi):
    mock_scrapi.stub("/v1/traces", response="", status=200, method="POST")

    exporter_options = StatsigOTLPTraceExporterOptions(
        sdk_key="test-sdk-key",
        dsn=mock_scrapi.url_for_endpoint(""),
    )

    result = initialize_tracing(
        InitializeTracingOptions(
            exporter_options=exporter_options,
            skip_global_context_manager_setup=True,
        )
    )

    tracer = result.provider.get_tracer("test-tracer")
    with tracer.start_as_current_span("test-span") as span:
        span.set_attribute("test.attribute", "test-value")
        span.end()

    result.processor.force_flush()

    requests = mock_scrapi.get_requests_for_endpoint("/v1/traces")
    assert len(requests) == 1, f"Expected exactly one request, got {len(requests)}"

    request = requests[0]

    content_encoding = request.headers.get("Content-Encoding")
    assert (
        content_encoding == "gzip"
    ), f"Expected Content-Encoding to be 'gzip', got '{content_encoding}'"

    raw_body = request.get_data()
    assert len(raw_body) > 0, "Request body should not be empty"

    try:
        decompressed = gzip.decompress(raw_body)
        assert len(decompressed) > 0, "Decompressed data should not be empty"
    except gzip.BadGzipFile as e:
        pytest.fail(f"Failed to decompress request body as gzip: {e}")


def test_exporter_includes_statsig_api_key_header(mock_scrapi: MockScrapi):
    test_sdk_key = "secret-test-key-12345"

    mock_scrapi.stub("/v1/traces", response="", status=200, method="POST")

    exporter_options = StatsigOTLPTraceExporterOptions(
        sdk_key=test_sdk_key,
        dsn=mock_scrapi.url_for_endpoint(""),
    )

    result = initialize_tracing(
        InitializeTracingOptions(
            exporter_options=exporter_options,
            skip_global_context_manager_setup=True,
        )
    )

    tracer = result.provider.get_tracer("test-tracer")
    with tracer.start_as_current_span("test-span"):
        pass

    result.processor.force_flush()

    requests = mock_scrapi.get_requests_for_endpoint("/v1/traces")
    assert len(requests) == 1

    request = requests[0]

    assert (
        request.headers.get("Statsig-Api-Key") == test_sdk_key
    ), f"Expected 'statsig-api-key' header to be '{test_sdk_key}'"
