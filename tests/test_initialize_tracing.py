from typing import List

import pytest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from statsig_ai.otel.otel import InitializeTracingOptions, initialize_tracing
from statsig_ai.otel.resources import StatsigResourceAttributes
from statsig_ai.otel.singleton import NoopOtelSingleton, OtelSingleton


@pytest.fixture(autouse=True)
def set_env_and_reset_singleton(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STATSIG_SDK_KEY", "test-sdk-key")
    for name in ["OTEL_SERVICE_NAME", "SERVICE_NAME", "PYTHON_ENV"]:
        monkeypatch.delenv(name, raising=False)
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
    assert attrs.get("deployment.environment.name") == "staging"
    # Like the Node SDK, the raw values are on the resource too.
    assert attrs.get("service") == "resource-service"
    assert attrs.get("version") == "2.0.0"
    assert attrs.get("environment") == "staging"


def test_stores_resource_values_on_the_singleton():
    initialize_tracing(
        InitializeTracingOptions(
            service_name="resource-service",
            version="2.0.0",
            environment="staging",
            skip_global_context_manager_setup=True,
        )
    )

    assert OtelSingleton.is_initialized()
    assert OtelSingleton.get_instance().get_resources() == StatsigResourceAttributes(
        service="resource-service", version="2.0.0", environment="staging"
    )


def test_stores_resource_values_with_a_provided_tracer_provider():
    initialize_tracing(
        InitializeTracingOptions(
            global_trace_provider=TracerProvider(),
            service_name="resource-service",
            skip_global_context_manager_setup=True,
        )
    )

    assert OtelSingleton.get_instance().get_resources() == StatsigResourceAttributes(
        service="resource-service"
    )


def test_uninitialized_singleton_has_no_resource_values():
    assert not OtelSingleton.is_initialized()
    assert NoopOtelSingleton.get_instance().get_resources() == StatsigResourceAttributes()


def test_omits_only_unset_resource_values():
    result = initialize_tracing(
        InitializeTracingOptions(version="", skip_global_context_manager_setup=True)
    )

    attrs = result.provider.resource.attributes  # type: ignore[attr-defined]

    assert attrs.get("version") == ""
    assert attrs.get("service.version") == ""
    assert "service" not in attrs
    assert "environment" not in attrs
    assert OtelSingleton.get_instance().get_resources() == StatsigResourceAttributes(version="")


def test_omits_deployment_environment_when_no_environment_is_set(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("PYTHON_ENV", raising=False)

    result = initialize_tracing(InitializeTracingOptions(skip_global_context_manager_setup=True))

    attrs = result.provider.resource.attributes  # type: ignore[attr-defined]

    assert "env" not in attrs
    assert "deployment.environment.name" not in attrs
    assert "environment" not in attrs


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
    assert attrs.get("service") == "env-service-name"
    assert attrs.get("env") == "ci"
    assert attrs.get("deployment.environment.name") == "ci"
    assert attrs.get("environment") == "ci"
    assert OtelSingleton.get_instance().get_resources() == StatsigResourceAttributes(
        service="env-service-name", environment="ci"
    )


def test_falls_back_to_service_name_after_otel_service_name(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SERVICE_NAME", "plain-service-name")

    result = initialize_tracing(InitializeTracingOptions(skip_global_context_manager_setup=True))

    attrs = result.provider.resource.attributes  # type: ignore[attr-defined]

    assert attrs.get("service.name") == "plain-service-name"
    assert attrs.get("service") == "plain-service-name"


def test_prefers_otel_service_name_over_service_name(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "otel-service-name")
    monkeypatch.setenv("SERVICE_NAME", "plain-service-name")

    result = initialize_tracing(InitializeTracingOptions(skip_global_context_manager_setup=True))

    attrs = result.provider.resource.attributes  # type: ignore[attr-defined]

    assert attrs.get("service.name") == "otel-service-name"
    assert attrs.get("service") == "otel-service-name"


def test_options_override_environment_variables(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service-name")
    monkeypatch.setenv("PYTHON_ENV", "ci")

    result = initialize_tracing(
        InitializeTracingOptions(
            service_name="option-service",
            environment="production",
            skip_global_context_manager_setup=True,
        )
    )

    attrs = result.provider.resource.attributes  # type: ignore[attr-defined]

    assert attrs.get("service") == "option-service"
    assert attrs.get("service.name") == "option-service"
    assert attrs.get("environment") == "production"
    assert attrs.get("env") == "production"
