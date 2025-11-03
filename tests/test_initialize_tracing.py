import os
from typing import Callable, List

import pytest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from statsig_ai import InitializeTracingOptions, initialize_tracing


@pytest.fixture(autouse=True)
def set_env_and_reset_singleton(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STATSIG_SDK_KEY", "test-sdk-key")
    from statsig_ai.otel import singleton as singleton_module

    original_instance = getattr(singleton_module.OtelSingleton, "_instance", None)
    singleton_module.OtelSingleton._instance = None

    yield

    singleton_module.OtelSingleton._instance = original_instance


def test_initialize_tracing_components():
    result = initialize_tracing(
        InitializeTracingOptions(
            service_name="test-service",
            version="1.0.0",
            environment="test",
            skip_global_context_manager_setup=True,
        )
    )

    assert result.exporter is not None
    assert result.processor is not None
    assert result.provider is not None


def test_does_not_register_global_provider_by_default(monkeypatch: pytest.MonkeyPatch):
    calls: List[TracerProvider] = []

    def fake_set_tracer_provider(provider: TracerProvider):
        calls.append(provider)

    monkeypatch.setattr(trace, "set_tracer_provider", fake_set_tracer_provider)

    result = initialize_tracing(
        InitializeTracingOptions(
            service_name="test-service",
            version="1.0.0",
            environment="test",
            skip_global_context_manager_setup=True,
        )
    )

    assert result.provider is not None
    assert calls == []


def test_can_register_global_trace_provider(monkeypatch: pytest.MonkeyPatch):
    called_with: List[TracerProvider] = []

    def fake_set_tracer_provider(provider: TracerProvider):
        called_with.append(provider)

    monkeypatch.setattr(trace, "set_tracer_provider", fake_set_tracer_provider)

    initialize_tracing(
        InitializeTracingOptions(
            skip_global_context_manager_setup=True,
            enable_global_trace_provider_registration=True,
        )
    )

    assert len(called_with) == 1
    assert isinstance(called_with[0], TracerProvider)


def test_uses_provided_global_tracer_provider_for_singleton(monkeypatch: pytest.MonkeyPatch):
    from statsig_ai.otel.singleton import OtelSingleton

    custom_provider = TracerProvider()
    called = {"count": 0}

    def fake_set_tracer_provider(provider: TracerProvider):
        called["count"] += 1

    monkeypatch.setattr(trace, "set_tracer_provider", fake_set_tracer_provider)

    result = initialize_tracing(
        InitializeTracingOptions(
            global_trace_provider=custom_provider,
            skip_global_context_manager_setup=True,
        )
    )

    assert result.provider is not custom_provider
    assert OtelSingleton.get_instance().get_tracer_provider() is custom_provider
    assert called["count"] == 0


def test_populates_resource_attributes_from_options():
    result = initialize_tracing(
        InitializeTracingOptions(
            service_name="resource-service",
            version="2.0.0",
            environment="staging",
            skip_global_context_manager_setup=True,
        )
    )

    attrs = result.provider.resource.attributes  # type: ignore[attr-defined]

    assert attrs.get("service.name") == "resource-service"
    assert attrs.get("service.version") == "2.0.0"
    assert attrs.get("env") == "staging"


def test_falls_back_to_environment_variables_for_resource_attributes(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service-name")
    monkeypatch.setenv("PYTHON_ENV", "ci")

    result = initialize_tracing(
        InitializeTracingOptions(
            skip_global_context_manager_setup=True,
        )
    )

    attrs = result.provider.resource.attributes  # type: ignore[attr-defined]

    assert attrs.get("service.name") == "env-service-name"
    assert attrs.get("env") == "ci"
