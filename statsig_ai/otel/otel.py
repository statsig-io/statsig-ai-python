import logging
from typing import Optional
from dataclasses import dataclass

from opentelemetry import trace, context
from opentelemetry.context.context import Context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.sdk.resources import Resource

from .conventions import ATTR_DEPLOYMENT_ENVIRONMENT_NAME
from .exporter import StatsigOTLPTraceExporter, StatsigOTLPTraceExporterOptions
from .processor import StatsigSpanProcessor
from .resources import StatsigResourceAttributes, create_resource_attributes
from .singleton import OtelSingleton


class InitializeTracingOptions:
    def __init__(
        self,
        global_context_manager: Optional[Context] = None,
        skip_global_context_manager_setup: bool = False,
        enable_global_trace_provider_registration: bool = False,
        global_trace_provider: Optional[TracerProvider] = None,
        exporter_options: Optional[StatsigOTLPTraceExporterOptions] = None,
        service_name: Optional[str] = None,
        version: Optional[str] = None,
        environment: Optional[str] = None,
    ):
        self.global_context_manager = global_context_manager
        self.skip_global_context_manager_setup = skip_global_context_manager_setup
        self.enable_global_trace_provider_registration = enable_global_trace_provider_registration
        self.global_trace_provider = global_trace_provider
        self.exporter_options = exporter_options or StatsigOTLPTraceExporterOptions()
        self.service_name = service_name
        self.version = version
        self.environment = environment


@dataclass
class InitializeTracingResult:
    exporter: StatsigOTLPTraceExporter
    processor: StatsigSpanProcessor
    provider: TracerProvider


def initialize_tracing(
    options: Optional[InitializeTracingOptions] = None,
) -> InitializeTracingResult:
    options = options or InitializeTracingOptions()

    if not options.global_context_manager and not options.skip_global_context_manager_setup:
        try:
            context.attach(context.get_current())
        except Exception:
            logging.warning(
                "[Statsig] Could not automatically set up a global OTEL context manager.\n"
                "This may be expected if you or another library already set one.\n"
                "You can skip this setup by passing skip_global_context_manager_setup=True."
            )

    resources = create_resource_attributes(
        service_name=options.service_name,
        version=options.version,
        environment=options.environment,
    )
    trace_components = _create_trace_components(
        exporter_options=options.exporter_options,
        resources=resources,
    )

    tracer_provider = options.global_trace_provider or trace_components.provider

    OtelSingleton.instantiate(tracer_provider, resources)

    if options.enable_global_trace_provider_registration:
        trace.set_tracer_provider(tracer_provider)

    return trace_components


def _create_trace_components(
    exporter_options: StatsigOTLPTraceExporterOptions,
    resources: StatsigResourceAttributes,
) -> InitializeTracingResult:
    exporter = StatsigOTLPTraceExporter(
        options=exporter_options,
    )

    processor = StatsigSpanProcessor(exporter)

    # Like the Node SDK, the resource has the raw service, version and environment
    # values as well as the OpenTelemetry names for them.
    attributes = {
        "service": resources.service,
        "version": resources.version,
        "environment": resources.environment,
        "service.name": resources.service,
        "service.version": resources.version,
        ATTR_DEPLOYMENT_ENVIRONMENT_NAME: resources.environment,
        "env": resources.environment,
    }
    resource = Resource.create({k: v for k, v in attributes.items() if v is not None})

    provider = TracerProvider(
        resource=resource,
        sampler=ALWAYS_ON,
    )
    provider.add_span_processor(processor)

    return InitializeTracingResult(
        exporter=exporter,
        processor=processor,
        provider=provider,
    )
