from collections.abc import Mapping as MappingABC
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Mapping, Optional, Protocol, Sequence, Union

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

_TRACE_HEADERS = ("traceparent", "tracestate")


class HeadersWithGet(Protocol):
    """A framework headers object, such as Werkzeug's, that looks names up with `.get`."""

    def get(self, key: str, /) -> Optional[str]: ...


# A plain mapping of header names to a value or a list of values, or a headers object.
IncomingHeaders = Union[Mapping[str, Union[str, Sequence[str]]], HeadersWithGet]

_propagator = TraceContextTextMapPropagator()


def _read_header(headers: IncomingHeaders, name: str) -> Optional[Union[str, List[str]]]:
    value: Any
    if isinstance(headers, MappingABC):
        key = next((k for k in headers if isinstance(k, str) and k.lower() == name), None)
        value = None if key is None else headers[key]
    else:
        value = headers.get(name)

    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        values = [v for v in value if isinstance(v, str)]
        return values or None
    return None


def set_incoming_trace_to_context(ctx: Context, headers: IncomingHeaders) -> Context:
    carrier: Dict[str, Union[str, List[str]]] = {}
    for name in _TRACE_HEADERS:
        value = _read_header(headers, name)
        if value is not None:
            carrier[name] = value

    incoming = trace.get_current_span(
        _propagator.extract(carrier, context=Context())
    ).get_span_context()
    if not incoming.is_valid:
        return ctx

    # OTel HTTP instrumentation may already have started a span in this trace.
    if trace.get_current_span(ctx).get_span_context().trace_id == incoming.trace_id:
        return ctx
    return trace.set_span_in_context(trace.NonRecordingSpan(incoming), ctx)


@contextmanager
def incoming_trace_context(headers: IncomingHeaders) -> Generator[Context, None, None]:
    """
    Runs the block in the trace named by the request's W3C `traceparent` and
    `tracestate` headers, so spans started inside it become children of the
    caller's span. Without a valid `traceparent`, the block runs unchanged.

    Accepts a mapping of header names (matched case-insensitively) to a value or a
    list of values, or a framework headers object with `.get(name)`, such as Flask's
    `request.headers` or FastAPI's.
    """
    new_ctx = set_incoming_trace_to_context(otel_context.get_current(), headers)
    token = otel_context.attach(new_ctx)
    try:
        yield new_ctx
    finally:
        otel_context.detach(token)
