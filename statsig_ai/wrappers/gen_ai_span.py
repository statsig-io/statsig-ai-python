"""Starts and finishes the span for one model call. Ported from the Node SDK's gen-ai-span.ts."""

from contextlib import contextmanager
import inspect
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterator, List, Mapping, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, StatusCode, Tracer

from ..otel.conventions import (
    STATSIG_ATTR_SPAN_LLM_ROOT,
    STATSIG_ATTR_SPAN_TYPE,
    STATSIG_CTX_KEY_LLM_ROOT,
    STATSIG_SPAN_LLM_ROOT_VALUE,
    StatsigSpanType,
)
from ..otel.environment_context import get_environment_span_attrs_from_context
from ..otel.prompt_context import get_prompt_span_attrs_from_context
from ..otel.statsig_context import get_statsig_span_attrs_from_context
from .configs import WrapOpenAIOptions
from .span_telemetry import SpanTelemetry

logger = logging.getLogger(__name__)

GetAttributes = Callable[[Any], Mapping[str, Any]]
GetStreamAttributes = Callable[[List[Any]], Mapping[str, Any]]
# Returns a provider's stream as a stream that ends the span once the caller has consumed it.
WrapStream = Callable[[Any, SpanTelemetry, GetStreamAttributes], Any]


def start_gen_ai_span(
    tracer: Tracer,
    span_name: str,
    provider_name: str,
    base_attrs: Mapping[str, Any],
    options: WrapOpenAIOptions,
) -> SpanTelemetry:
    """Starts a client span for one model call, tagged with the active Statsig prompt, prompt
    version, context and environment, so it can be graded and attributed.

    Template variables usually hold user input, so they follow the same capture option as the
    input messages. The call is the LLM root unless a workflow span has set the marker in the
    context.
    """
    capture_options = options.gen_ai_capture_options
    ctx = otel_context.get_current()
    statsig_attrs: Dict[str, Any] = {STATSIG_ATTR_SPAN_TYPE: StatsigSpanType.GEN_AI}
    llm_root = otel_context.get_value(STATSIG_CTX_KEY_LLM_ROOT, ctx)
    if llm_root is None or llm_root == "":
        ctx = otel_context.set_value(STATSIG_CTX_KEY_LLM_ROOT, span_name, ctx)
        statsig_attrs[STATSIG_ATTR_SPAN_LLM_ROOT] = STATSIG_SPAN_LLM_ROOT_VALUE

    statsig_attrs.update(
        get_prompt_span_attrs_from_context(
            ctx,
            include_template_variables=(
                capture_options.capture_all or capture_options.capture_input_messages
            ),
        )
    )
    statsig_attrs.update(get_statsig_span_attrs_from_context(ctx) or {})
    statsig_attrs.update(get_environment_span_attrs_from_context(ctx) or {})

    # As in the Node SDK, the wrapper's own attributes win over custom attributes.
    attributes = {**options.custom_attributes, **base_attrs, **statsig_attrs}
    # Messages and template variables are dicts and lists. SpanTelemetry writes them as capped
    # JSON below, so the span starts with the plain values only.
    start_attrs = {
        k: v for k, v in attributes.items() if v is not None and not isinstance(v, (dict, list))
    }
    span = tracer.start_span(span_name, context=ctx, kind=SpanKind.CLIENT, attributes=start_attrs)
    telemetry = SpanTelemetry(
        span, span_name, provider_name=provider_name, max_json_chars=options.max_json_chars
    )
    telemetry.set_attributes(attributes)
    return telemetry


def trace_gen_ai_call(
    telemetry: SpanTelemetry,
    call: Callable[[], Any],
    on_result: GetAttributes,
    on_stream: GetStreamAttributes,
    wrap_stream: Optional[WrapStream] = None,
) -> Any:
    """Runs a provider call inside its span, and ends the span with the attributes of its result.

    A coroutine is returned as a coroutine that ends the span once it's awaited. A stream (an
    iterator or async iterator) is returned as a stream that ends the span once the caller has
    consumed it: a generator, or what `wrap_stream` returns, for a provider whose stream object
    has more to it than iteration. If the call raises, the span records the exception and an
    ERROR status, and the exception is raised again.
    """
    try:
        with _activated(telemetry.span):
            result = call()
    except BaseException as e:
        telemetry.fail(e)
        telemetry.end()
        raise
    if inspect.iscoroutine(result):
        return _finish_after(telemetry, result, on_result, on_stream, wrap_stream)
    return _finish(telemetry, result, on_result, on_stream, wrap_stream)


async def _finish_after(
    telemetry: SpanTelemetry,
    pending: Awaitable[Any],
    on_result: GetAttributes,
    on_stream: GetStreamAttributes,
    wrap_stream: Optional[WrapStream],
) -> Any:
    try:
        with _activated(telemetry.span):
            result = await pending
    except BaseException as e:
        telemetry.fail(e)
        telemetry.end()
        raise
    return _finish(telemetry, result, on_result, on_stream, wrap_stream)


def _finish(
    telemetry: SpanTelemetry,
    result: Any,
    on_result: GetAttributes,
    on_stream: GetStreamAttributes,
    wrap_stream: Optional[WrapStream],
) -> Any:
    is_async_stream = hasattr(result, "__aiter__")
    # Checks for an iterator, not an iterable: pydantic responses are iterable over their fields.
    if wrap_stream is not None and (is_async_stream or isinstance(result, Iterator)):
        return wrap_stream(result, telemetry, on_stream)
    if is_async_stream:
        return _trace_async_stream(result, telemetry, on_stream)
    if isinstance(result, Iterator):
        return _trace_sync_stream(result, telemetry, on_stream)
    telemetry.record_time_to_first_token()
    telemetry.set_status({"code": StatusCode.OK})
    set_attributes_safely(telemetry, lambda: on_result(result))
    telemetry.end()
    return result


def _trace_sync_stream(
    stream: Iterator[Any], telemetry: SpanTelemetry, on_stream: GetStreamAttributes
) -> Iterator[Any]:
    items: List[Any] = []
    try:
        for item in stream:
            if not items:
                telemetry.record_time_to_first_token()
            items.append(item)
            yield item
        set_attributes_safely(telemetry, lambda: on_stream(items))
        telemetry.set_status({"code": StatusCode.OK})
    except Exception as e:
        telemetry.fail(e)
        raise
    finally:
        telemetry.end()


async def _trace_async_stream(
    stream: AsyncIterator[Any], telemetry: SpanTelemetry, on_stream: GetStreamAttributes
) -> AsyncIterator[Any]:
    items: List[Any] = []
    try:
        async for item in stream:
            if not items:
                telemetry.record_time_to_first_token()
            items.append(item)
            yield item
        set_attributes_safely(telemetry, lambda: on_stream(items))
        telemetry.set_status({"code": StatusCode.OK})
    except Exception as e:
        telemetry.fail(e)
        raise
    finally:
        telemetry.end()


def set_attributes_safely(
    telemetry: SpanTelemetry, get_attributes: Callable[[], Mapping[str, Any]]
) -> None:
    """Records response attributes, logging instead of raising if the response can't be read."""
    try:
        telemetry.set_attributes(dict(get_attributes()))
    except Exception as e:
        logger.warning("[Statsig] Failed to record Gen AI response attributes: %s", e)


@contextmanager
def _activated(span: Span) -> Iterator[None]:
    """Makes the model span current, so spans the provider call starts are its children."""
    token = otel_context.attach(trace.set_span_in_context(span))
    try:
        yield
    finally:
        otel_context.detach(token)
