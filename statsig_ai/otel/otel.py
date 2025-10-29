import os
from typing import Optional, Dict

from opentelemetry import trace, context
from opentelemetry.context.context import Context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.sdk.resources import Resource

from .exporter import StatsigOTLPTraceExporter
from .processor import StatsigSpanProcessor
from .singleton import OtelSingleton


class InitializeOptions:
    def __init__(
        self,
        global_context_manager: Optional[Context] = None,
        skip_global_context_manager_setup: bool = False,
        enable_global_trace_provider_registration: bool = False,
        global_trace_provider: Optional[TracerProvider] = None,
        exporter_options: Optional[Dict[str, str]] = None,
        service_name: Optional[str] = None,
        version: Optional[str] = None,
        environment: Optional[str] = None,
    ):
        self.global_context_manager = global_context_manager
        self.skip_global_context_manager_setup = skip_global_context_manager_setup
        self.enable_global_trace_provider_registration = (
            enable_global_trace_provider_registration
        )
        self.global_trace_provider = global_trace_provider
        self.exporter_options = exporter_options or {}
        self.service_name = service_name
        self.version = version
        self.environment = environment


def initialize_otel(options: Optional[InitializeOptions] = None):
    options = options or InitializeOptions()

    # --- Context Manager Setup ---
    if (
        not options.global_context_manager
        and not options.skip_global_context_manager_setup
    ):
        try:
            # In Python, the default context manager is thread-local.
            # For async frameworks, additional libraries (e.g., asyncio contextvars) can be used.
            context.attach(context.get_current())
        except Exception:
            print(
                "Could not automatically set up a global OTEL context manager.\n"
                "This may be expected if you or another library already set one.\n"
                "You can skip this setup by passing skip_global_context_manager_setup=True."
            )

    # --- Create trace components ---
    trace_components = _create_trace_components(
        exporter_options=options.exporter_options,
        resources={
            "service.name": options.service_name or os.getenv("OTEL_SERVICE_NAME"),
            "service.version": options.version,
            "env": options.environment or os.getenv("PYTHON_ENV"),
        },
    )

    tracer_provider = options.global_trace_provider or trace_components["provider"]

    # --- Singleton setup ---
    OtelSingleton.instantiate(tracer_provider)

    # --- Register provider globally ---
    if options.enable_global_trace_provider_registration:
        trace.set_tracer_provider(tracer_provider)

    return trace_components


def _create_trace_components(
    exporter_options: Dict[str, str],
    resources: Dict[str, Optional[str]],
):
    exporter = StatsigOTLPTraceExporter(
        sdk_key=exporter_options.get("sdkKey"),
        dsn=exporter_options.get("dsn"),
    )

    processor = StatsigSpanProcessor(exporter)

    resource = Resource.create({k: v for k, v in resources.items() if v is not None})

    provider = TracerProvider(
        resource=resource,
        sampler=ALWAYS_ON,
    )
    provider.add_span_processor(processor)

    return {
        "exporter": exporter,
        "processor": processor,
        "provider": provider,
    }
