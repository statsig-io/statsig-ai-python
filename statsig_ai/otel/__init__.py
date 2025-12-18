from .otel import (
    initialize_tracing,
    InitializeTracingOptions,
    InitializeTracingResult,
)
from .exporter import StatsigOTLPTraceExporter, StatsigOTLPTraceExporterOptions
from .processor import StatsigSpanProcessor
from .singleton import OtelSingleton, NoopOtelSingleton
from .statsig_context import (
    StatsigContext,
    statsig_context,
    get_statsig_context_from_otel_context,
    get_statsig_span_attrs_from_context,
    set_statsig_span_attrs_from_context,
    set_statsig_context_to_otel_context,
)
from .conventions import (
    STATSIG_ATTR_USER_ID,
    STATSIG_ATTR_CUSTOM_IDS,
    STATSIG_ATTR_ACTIVITY_ID,
    STATSIG_CTX_KEY,
)

__all__ = [
    "initialize_tracing",
    "InitializeTracingOptions",
    "InitializeTracingResult",
    "StatsigOTLPTraceExporter",
    "StatsigOTLPTraceExporterOptions",
    "StatsigSpanProcessor",
    "OtelSingleton",
    "NoopOtelSingleton",
    # Statsig Context
    "StatsigContext",
    "statsig_context",
    "get_statsig_context_from_otel_context",
    "get_statsig_span_attrs_from_context",
    "set_statsig_span_attrs_from_context",
    "set_statsig_context_to_otel_context",
    # Context conventions
    "STATSIG_ATTR_USER_ID",
    "STATSIG_ATTR_CUSTOM_IDS",
    "STATSIG_ATTR_ACTIVITY_ID",
    "STATSIG_CTX_KEY",
]
