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
from .user_context import statsig_user_context
from .environment_context import statsig_environment
from .incoming_trace_context import IncomingHeaders, incoming_trace_context
from .conventions import (
    STATSIG_ATTR_LLM_PROMPT_NAME,
    STATSIG_ATTR_LLM_PROMPT_VERSION,
    STATSIG_ATTR_LLM_PROMPT_VERSION_ID,
    STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES,
    STATSIG_ATTR_USER_ID,
    STATSIG_ATTR_CUSTOM_IDS,
    STATSIG_ATTR_ACTIVITY_ID,
    STATSIG_ATTR_ENVIRONMENT,
    ATTR_DEPLOYMENT_ENVIRONMENT_NAME,
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
    # User Context
    "statsig_user_context",
    # Environment
    "statsig_environment",
    # Incoming Trace Context
    "incoming_trace_context",
    "IncomingHeaders",
    # Context conventions
    "STATSIG_ATTR_LLM_PROMPT_NAME",
    "STATSIG_ATTR_LLM_PROMPT_VERSION",
    "STATSIG_ATTR_LLM_PROMPT_VERSION_ID",
    "STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES",
    "STATSIG_ATTR_USER_ID",
    "STATSIG_ATTR_CUSTOM_IDS",
    "STATSIG_ATTR_ACTIVITY_ID",
    "STATSIG_ATTR_ENVIRONMENT",
    "ATTR_DEPLOYMENT_ENVIRONMENT_NAME",
    "STATSIG_CTX_KEY",
]
