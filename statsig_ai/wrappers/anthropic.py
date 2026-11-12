"""Traces Anthropic Messages API calls. Ported from the Node SDK's anthropic.ts.

The wrapper is duck-typed and never imports `anthropic`, so the SDK works without it installed.
"""

import inspect
import logging
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    TypeVar,
    cast,
)

from opentelemetry.trace import StatusCode, Tracer

from ..otel.singleton import OtelSingleton
from .anthropic_messages import (
    aggregate_anthropic_stream_events,
    as_dict,
    to_anthropic_input_messages,
    to_anthropic_output_messages,
    to_anthropic_parts,
)
from .configs import GenAICaptureOptions, WrapOpenAIOptions
from .gen_ai_span import (
    GetStreamAttributes,
    WrapStream,
    set_attributes_safely,
    start_gen_ai_span,
    trace_gen_ai_call,
)
from .span_telemetry import SpanTelemetry

logger = logging.getLogger(__name__)

T = TypeVar("T")
TraceCall = Callable[[Callable[[], Any], WrapStream], Any]

_PROVIDER_NAME = "anthropic"
# The span status of a messages.stream() stream closed before its final message, as in Node.
_ABORTED = "Request was aborted."


def wrap_anthropic(client: T, options: Optional[WrapOpenAIOptions] = None) -> T:
    """Records each `messages.create` (including `stream=True`) and `messages.stream` call of an
    `Anthropic` or `AsyncAnthropic` client as a span and a `statsig::gen_ai` event.

    Returns the client unchanged, with a warning, if it has no `messages.create`.
    """
    messages = getattr(client, "messages", None)
    if not callable(getattr(messages, "create", None)):
        logger.warning("[Statsig] Object does not have Anthropic APIs, not wrapping.")
        return client
    return cast(T, AnthropicWrapper(client, options or WrapOpenAIOptions()))


class _Proxy:
    """Passes the attributes it doesn't define through to the object it wraps."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def __getattr__(self, name: str) -> Any:
        if name == "_wrapped":
            raise AttributeError(name)
        return getattr(self._wrapped, name)


class AnthropicWrapper(_Proxy):
    def __init__(self, client: Any, options: WrapOpenAIOptions) -> None:
        super().__init__(client)
        self.messages = AnthropicMessagesWrapper(client.messages, options)


class AnthropicMessagesWrapper(_Proxy):
    def __init__(self, messages: Any, options: WrapOpenAIOptions) -> None:
        super().__init__(messages)
        self._options = options.gen_ai_capture_options

    def create(self, *args: Any, **kwargs: Any) -> Any:
        create = self._wrapped.create
        return self._trace(
            "anthropic.messages.create", kwargs, lambda: create(*args, **kwargs), _trace_stream
        )

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        stream = self._wrapped.stream
        params = {**kwargs, "stream": True}

        def trace_call(call: Callable[[], Any], wrap_stream: WrapStream) -> Any:
            return self._trace("anthropic.messages.stream", params, call, wrap_stream)

        try:
            manager = stream(*args, **kwargs)
        except BaseException as e:
            telemetry = self._start_span("anthropic.messages.stream", params)
            telemetry.fail(e)
            telemetry.end()
            raise
        return _TracedMessageStreamManager(manager, trace_call)

    def _trace(
        self,
        source_name: str,
        params: Mapping[str, Any],
        call: Callable[[], Any],
        wrap_stream: WrapStream,
    ) -> Any:
        return trace_gen_ai_call(
            self._start_span(source_name, params),
            call,
            on_result=lambda message: _response_attributes(message, self._options),
            on_stream=lambda events: _response_attributes(
                aggregate_anthropic_stream_events(events), self._options
            ),
            wrap_stream=wrap_stream,
        )

    def _start_span(self, source_name: str, params: Mapping[str, Any]) -> SpanTelemetry:
        attrs = _request_attributes(source_name, params, self._options)
        model = attrs.get("gen_ai.request.model")
        span_name = f"chat {model if model is not None else 'unknown'}"
        return start_gen_ai_span(_get_tracer(), span_name, _PROVIDER_NAME, attrs, self._options)


class _TracedMessageStreamManager(_Proxy):
    """Wraps the manager that `messages.stream()` returns, for `with` or `async with`.

    The span starts when entering the manager sends the request, and ends once the stream has
    been read to the end, however the caller reads it (iteration, `text_stream`,
    `get_final_message()`), or when the stream is closed.
    """

    def __init__(self, manager: Any, trace_call: TraceCall) -> None:
        super().__init__(manager)
        self._trace_call = trace_call
        self._recorder: Optional[_StreamRecorder] = None

    def __enter__(self) -> Any:
        enter = self._wrapped.__enter__
        return self._trace_call(enter, self._trace_message_stream)

    def __exit__(self, *exc_info: Any) -> Any:
        try:
            return self._wrapped.__exit__(*exc_info)
        finally:
            self._close_recorder()

    async def __aenter__(self) -> Any:
        enter = self._wrapped.__aenter__
        return await self._trace_call(enter, self._trace_message_stream)

    async def __aexit__(self, *exc_info: Any) -> Any:
        try:
            return await self._wrapped.__aexit__(*exc_info)
        finally:
            self._close_recorder()

    def _trace_message_stream(
        self, stream: Any, telemetry: SpanTelemetry, get_attributes: GetStreamAttributes
    ) -> Any:
        raw_stream = getattr(stream, "_raw_stream", None)
        if raw_stream is None:
            logger.warning(
                "[Statsig] Unsupported Anthropic message stream. Its span won't record the response."
            )
            self._recorder = _StreamRecorder(telemetry, get_attributes, abort_message=None)
            return stream
        self._recorder = _StreamRecorder(telemetry, get_attributes, abort_message=_ABORTED)
        # The SDK's MessageStream reads every event from its raw stream, whichever way the caller
        # reads it, and closes the raw stream when it's closed.
        stream._raw_stream = _traced(raw_stream, self._recorder)
        return stream

    def _close_recorder(self) -> None:
        if self._recorder is not None:
            self._recorder.close()


def _trace_stream(
    stream: Any, telemetry: SpanTelemetry, get_attributes: GetStreamAttributes
) -> Any:
    """Returns the stream of `messages.create(stream=True)`. Like the Node SDK, a stream closed
    before its message ends the span without a status."""
    return _traced(stream, _StreamRecorder(telemetry, get_attributes, abort_message=None))


def _traced(stream: Any, recorder: "_StreamRecorder") -> Any:
    if hasattr(stream, "__aiter__"):
        return _TracedAsyncStream(stream, recorder)
    return _TracedStream(stream, recorder)


class _StreamRecorder:
    """Collects a streamed call's events, and records the message they make on its span."""

    def __init__(
        self,
        telemetry: SpanTelemetry,
        get_attributes: GetStreamAttributes,
        abort_message: Optional[str],
    ) -> None:
        self._telemetry = telemetry
        self._get_attributes = get_attributes
        self._abort_message = abort_message
        self._events: List[Any] = []
        self._message_stopped = False

    def add(self, event: Any) -> None:
        if not self._events:
            self._telemetry.record_time_to_first_token()
        self._events.append(event)
        if _event_type(event) == "message_stop":
            self._message_stopped = True

    def finish(self) -> None:
        if self._telemetry.ended:
            return
        self._telemetry.set_status({"code": StatusCode.OK})
        set_attributes_safely(self._telemetry, lambda: self._get_attributes(self._events))
        self._telemetry.end()

    def fail(self, error: Exception) -> None:
        if self._telemetry.ended:
            return
        self._telemetry.fail(error)
        self._telemetry.end()

    def close(self) -> None:
        """Ends the span of a stream closed before it was read to the end."""
        if self._telemetry.ended:
            return
        if self._message_stopped:
            self.finish()
            return
        if self._abort_message is not None:
            self._telemetry.fail(RuntimeError(self._abort_message))
        self._telemetry.end()


class _TracedStream(_Proxy):
    """A sync stream of events, read through its recorder. Other attributes, such as `response`,
    are the stream's own."""

    def __init__(self, stream: Iterable[Any], recorder: _StreamRecorder) -> None:
        super().__init__(stream)
        self._recorder = recorder
        self._events = _traced_events(stream, recorder)

    def __iter__(self) -> Iterator[Any]:
        return self._events

    def __next__(self) -> Any:
        return next(self._events)

    def __enter__(self) -> "_TracedStream":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        self._events.close()
        self._recorder.close()
        close = getattr(self._wrapped, "close", None)
        if callable(close):
            close()


class _TracedAsyncStream(_Proxy):
    """An async stream of events, read through its recorder. Other attributes, such as
    `response`, are the stream's own."""

    def __init__(self, stream: AsyncIterable[Any], recorder: _StreamRecorder) -> None:
        super().__init__(stream)
        self._recorder = recorder
        self._events = _traced_async_events(stream, recorder)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._events

    async def __anext__(self) -> Any:
        return await self._events.__anext__()

    async def __aenter__(self) -> "_TracedAsyncStream":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._events.aclose()
        self._recorder.close()
        close = getattr(self._wrapped, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result


# Module functions rather than methods, so a dropped stream isn't kept alive by a reference cycle
# and its span ends as soon as it's garbage collected.
def _traced_events(stream: Iterable[Any], recorder: _StreamRecorder) -> Generator[Any, None, None]:
    try:
        for event in stream:
            recorder.add(event)
            yield event
    except Exception as e:
        recorder.fail(e)
        raise
    else:
        recorder.finish()
    finally:
        recorder.close()


async def _traced_async_events(
    stream: AsyncIterable[Any], recorder: _StreamRecorder
) -> AsyncGenerator[Any, None]:
    try:
        async for event in stream:
            recorder.add(event)
            yield event
    except Exception as e:
        recorder.fail(e)
        raise
    else:
        recorder.finish()
    finally:
        recorder.close()


def _request_attributes(
    source_name: str, params: Mapping[str, Any], options: GenAICaptureOptions
) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.operation.source_name": source_name,
        "gen_ai.provider.name": _PROVIDER_NAME,
        "gen_ai.request.model": _given(params.get("model")),
        "gen_ai.request.max_tokens": _given(params.get("max_tokens")),
        "gen_ai.request.temperature": _given(params.get("temperature")),
        "gen_ai.request.top_p": _given(params.get("top_p")),
        "gen_ai.request.top_k": _given(params.get("top_k")),
        "gen_ai.request.stop_sequences": _given(params.get("stop_sequences")),
    }
    if params.get("stream"):
        attrs["gen_ai.request.stream"] = True
    if options.capture_all or options.capture_input_messages:
        attrs["gen_ai.input.messages"] = to_anthropic_input_messages(params.get("messages"))
    system = _given(params.get("system"))
    if (options.capture_all or options.capture_system_instructions) and system is not None:
        attrs["gen_ai.system_instructions"] = to_anthropic_parts(system)
    if options.capture_all or options.capture_tool_definitions:
        attrs["gen_ai.tool.definitions"] = _given(params.get("tools"))
    return attrs


def _response_attributes(message: Any, options: GenAICaptureOptions) -> Dict[str, Any]:
    fields = as_dict(message)
    attrs: Dict[str, Any] = {
        "gen_ai.response.id": fields.get("id"),
        "gen_ai.response.model": fields.get("model"),
        **_usage_attributes(fields.get("usage")),
    }
    stop_reason = fields.get("stop_reason")
    if stop_reason:
        attrs["gen_ai.response.finish_reasons"] = [stop_reason]
    if options.capture_all or options.capture_output_messages:
        attrs["gen_ai.output.messages"] = to_anthropic_output_messages(fields)
    return attrs


def _usage_attributes(usage: Any) -> Dict[str, int]:
    fields = as_dict(usage)
    cached_tokens = _token_count(fields.get("cache_read_input_tokens"))
    # Anthropic reports cache reads and writes separately from input_tokens.
    input_tokens = (
        _token_count(fields.get("input_tokens"))
        + cached_tokens
        + _token_count(fields.get("cache_creation_input_tokens"))
    )
    output_tokens = _token_count(fields.get("output_tokens"))
    attrs: Dict[str, int] = {}
    if input_tokens > 0:
        attrs["gen_ai.usage.input_tokens"] = input_tokens
    if output_tokens > 0:
        attrs["gen_ai.usage.output_tokens"] = output_tokens
    if input_tokens + output_tokens > 0:
        attrs["statsig.gen_ai.usage.total_tokens"] = input_tokens + output_tokens
    if cached_tokens > 0:
        attrs["statsig.gen_ai.usage.input_cached_tokens"] = cached_tokens
    return attrs


def _token_count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _given(value: Any) -> Any:
    """Reads the SDK's NOT_GIVEN and omit placeholders as a missing parameter."""
    if value is None or type(value).__name__ in ("NotGiven", "Omit"):
        return None
    return list(value) if isinstance(value, tuple) else value


def _event_type(event: Any) -> Any:
    if isinstance(event, Mapping):
        return event.get("type")
    return getattr(event, "type", None)


def _get_tracer() -> Tracer:
    # Looked up on each call: a client is often wrapped before initialize_tracing() runs.
    return OtelSingleton.get_tracer_provider_static().get_tracer("statsig-anthropic-proxy")
