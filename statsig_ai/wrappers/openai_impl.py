"""Traces OpenAI client calls as GenAI spans.

Ported from the Node SDK's openai-impl.ts and gen-ai-span.ts.
"""

import functools
import inspect
import logging
from contextlib import contextmanager
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    Optional,
)

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind, StatusCode, Tracer

from statsig_ai.otel.conventions import (
    STATSIG_ATTR_SPAN_LLM_ROOT,
    STATSIG_ATTR_SPAN_TYPE,
    STATSIG_CTX_KEY_LLM_ROOT,
    STATSIG_SPAN_LLM_ROOT_VALUE,
    StatsigSpanType,
)
from statsig_ai.otel.environment_context import get_environment_span_attrs_from_context
from statsig_ai.otel.prompt_context import get_prompt_span_attrs_from_context
from statsig_ai.otel.statsig_context import get_statsig_span_attrs_from_context

from .genai_attribute_helper import extract_base_attributes, extract_response_attributes
from ..otel.singleton import OtelSingleton
from .configs import GenAICaptureOptions, WrapOpenAIOptions
from .span_telemetry import SpanTelemetry

logger = logging.getLogger(__name__)


def _get_tracer() -> Tracer:
    return OtelSingleton.get_tracer_provider_static().get_tracer("statsig-openai-proxy")


OTEL_OP_NAME_MAP = {
    "openai.chat.completions.create": "chat",
    "openai.completions.create": "text_completion",
    "openai.embeddings.create": "embeddings",
    "openai.images.generate": "generate_content",
    "openai.responses.create": "generate_content",
    "openai.responses.stream": "generate_content",
    "openai.responses.parse": "generate_content",
}


class BaseWrapper:
    def __init__(self, client: Any, options: WrapOpenAIOptions):
        self._client = client
        self._wrap_options = options
        self._options = options.gen_ai_capture_options

    def __getattr__(self, name):
        return getattr(self._client, name)

    def _start_span(self, operation_name: str, kwargs: Mapping[str, Any]) -> SpanTelemetry:
        model = kwargs.get("model")
        # Use OTEL semantic name if available, otherwise strip "openai." prefix
        otel_name = OTEL_OP_NAME_MAP.get(operation_name, operation_name.replace("openai.", ""))
        span_name = f"{otel_name} {model if model is not None else 'unknown'}"

        # Like the Node SDK's startGenAISpan, the span starts with the active prompt version,
        # statsig_context and statsig_environment attributes, and the statsig::gen_ai event
        # gets them too. Template variables usually hold user input, so they follow the same
        # capture option as the input messages.
        statsig_attrs: Dict[str, Any] = {STATSIG_ATTR_SPAN_TYPE: StatsigSpanType.GEN_AI}

        # Like the Node SDK, a model call is the LLM root unless a workflow span has set the
        # marker in the context. It sets the marker in the context its span starts in.
        ctx = otel_context.get_current()
        llm_root = otel_context.get_value(STATSIG_CTX_KEY_LLM_ROOT, ctx)
        if llm_root is None or llm_root == "":
            ctx = otel_context.set_value(STATSIG_CTX_KEY_LLM_ROOT, span_name, ctx)
            statsig_attrs[STATSIG_ATTR_SPAN_LLM_ROOT] = STATSIG_SPAN_LLM_ROOT_VALUE

        statsig_attrs.update(
            get_prompt_span_attrs_from_context(
                ctx,
                include_template_variables=(
                    self._options.capture_all or self._options.capture_input_messages
                ),
            )
        )
        statsig_attrs.update(get_statsig_span_attrs_from_context(ctx) or {})
        statsig_attrs.update(get_environment_span_attrs_from_context(ctx) or {})

        # As in the Node SDK, the wrapper's own attributes win over custom attributes.
        attrs: Dict[str, Any] = {
            **self._wrap_options.custom_attributes,
            **extract_base_attributes(operation_name, otel_name, kwargs, self._options),
            **statsig_attrs,
        }

        # Messages and template variables are lists and dicts. SpanTelemetry writes them as
        # capped JSON below, so the span starts with the plain values only.
        start_attrs = {
            k: v for k, v in attrs.items() if v is not None and not isinstance(v, (dict, list))
        }
        span = _get_tracer().start_span(
            span_name, context=ctx, kind=SpanKind.CLIENT, attributes=start_attrs
        )
        telemetry = SpanTelemetry(
            span,
            span_name,
            provider_name="openai",
            max_json_chars=self._wrap_options.max_json_chars,
        )
        telemetry.set_attributes(attrs)
        return telemetry

    def _wrap_call(self, fn_name: str, operation_name: str) -> Callable[..., Any]:
        original_fn = getattr(self._client, fn_name)

        def wrapper(*args, **kwargs):
            telemetry = self._start_span(operation_name, kwargs)
            # Like the Node SDK's traceGenAICall, the call runs with its span active, so spans
            # and logs made during the call, such as HTTP client spans, are its children.
            span_ctx = trace.set_span_in_context(telemetry.span)
            try:
                with _activated(span_ctx):
                    result = original_fn(*args, **kwargs)
            except BaseException as e:
                _end_failed(telemetry, e)
                raise

            if inspect.isawaitable(result):
                return self._finish_async_call(result, telemetry, span_ctx, kwargs)
            if kwargs.get("stream") and _is_sync_stream(result):
                telemetry.set_attributes({"gen_ai.request.stream": True})
                return _TracedStream(result, telemetry, self._options)
            _end_with_response(telemetry, result, self._options)
            return result

        return wrapper

    async def _finish_async_call(
        self,
        awaitable: Awaitable[Any],
        telemetry: SpanTelemetry,
        span_ctx: Context,
        kwargs: Mapping[str, Any],
    ) -> Any:
        try:
            # The request is sent when the coroutine is awaited, so the span is active again.
            with _activated(span_ctx):
                result = await awaitable
        except BaseException as e:
            _end_failed(telemetry, e)
            raise

        if kwargs.get("stream") and hasattr(result, "__aiter__"):
            telemetry.set_attributes({"gen_ai.request.stream": True})
            return _AsyncTracedStream(result, telemetry, self._options)
        _end_with_response(telemetry, result, self._options)
        return result

    def _wrap_stream_manager(self, fn_name: str, operation_name: str) -> Callable[..., Any]:
        """Wraps a method that returns a stream manager, such as `responses.stream()`."""
        original_fn = getattr(self._client, fn_name)

        def wrapper(*args, **kwargs):
            manager = original_fn(*args, **kwargs)
            start_span = functools.partial(self._start_span, operation_name, kwargs)
            if hasattr(manager, "__aenter__"):
                return _AsyncTracedStreamManager(manager, start_span, self._options)
            if hasattr(manager, "__enter__"):
                return _TracedStreamManager(manager, start_span, self._options)
            logger.warning(
                "[Statsig] %s did not return a stream manager. Not tracing it.", operation_name
            )
            return manager

        return wrapper


@contextmanager
def _activated(ctx: Context) -> Iterator[None]:
    token = otel_context.attach(ctx)
    try:
        yield
    finally:
        otel_context.detach(token)


def _end_failed(telemetry: SpanTelemetry, error: BaseException) -> None:
    telemetry.fail(error)
    telemetry.end()


def _end_with_response(telemetry: SpanTelemetry, result: Any, options: GenAICaptureOptions) -> None:
    telemetry.record_time_to_first_token()
    telemetry.set_status({"code": StatusCode.OK})
    _set_attributes_safely(telemetry, lambda: _response_attributes(result, options))
    telemetry.end()


def _response_attributes(result: Any, options: GenAICaptureOptions) -> Dict[str, Any]:
    response = _convert_result_to_dict(result)
    if response is None:
        logger.warning("[Statsig] Failed to convert result to dict. Type: %s", type(result))
        return {}
    return extract_response_attributes(response, options)


def _set_attributes_safely(
    telemetry: SpanTelemetry, get_attributes: Callable[[], Dict[str, Any]]
) -> None:
    try:
        telemetry.set_attributes(get_attributes())
    except Exception as e:
        logger.warning("[Statsig] Failed to record Gen AI response attributes: %s", e)


def _is_sync_stream(result: Any) -> bool:
    if isinstance(result, (Mapping, str, bytes)) or hasattr(result, "model_dump"):
        return False
    return hasattr(result, "__iter__")


def _convert_result_to_dict(result) -> Optional[dict]:
    if result is None:
        return None
    if isinstance(result, dict):
        return result
    # OpenAI response objects should have a to_dict method
    if hasattr(result, "to_dict") and callable(result.to_dict):
        try:
            return result.to_dict()
        except Exception:
            pass
    # try pydantic methods
    if hasattr(result, "model_dump") and callable(result.model_dump):
        try:
            return result.model_dump()
        except Exception:
            pass
    if hasattr(result, "dict") and callable(result.dict):
        try:
            return result.dict()
        except Exception:
            pass
    return None


def _get_field(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


class _StreamedChoice:
    """One choice of a Chat Completions or completions stream, rebuilt from its deltas."""

    def __init__(self, index: Any) -> None:
        self.index = index
        self.role: Optional[str] = None
        self.content: Optional[List[str]] = None
        self.refusal: Optional[List[str]] = None
        self.text: Optional[List[str]] = None
        self.tool_calls: Dict[Any, Dict[str, Any]] = {}
        self.finish_reason: Any = None

    def add(self, choice: Mapping[str, Any]) -> None:
        delta = choice.get("delta")
        if isinstance(delta, Mapping):
            self.role = delta.get("role") or self.role
            self.content = _append(self.content, delta.get("content"))
            self.refusal = _append(self.refusal, delta.get("refusal"))
            for tool_call in delta.get("tool_calls") or ():
                if isinstance(tool_call, Mapping):
                    self._add_tool_call(tool_call)
        # Completions (not chat) chunks stream text instead of a delta.
        self.text = _append(self.text, choice.get("text"))
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]

    def _add_tool_call(self, delta: Mapping[str, Any]) -> None:
        call = self.tool_calls.setdefault(
            delta.get("index") or 0, {"id": None, "name": None, "arguments": []}
        )
        function = delta.get("function")
        function = function if isinstance(function, Mapping) else {}
        if call["id"] is None:
            call["id"] = delta.get("id")
        if call["name"] is None:
            call["name"] = function.get("name")
        if isinstance(function.get("arguments"), str):
            call["arguments"].append(function["arguments"])

    def to_choice(self) -> Dict[str, Any]:
        message: Dict[str, Any] = {"role": self.role or "assistant"}
        if self.content is not None:
            message["content"] = "".join(self.content)
        if self.refusal is not None:
            message["refusal"] = "".join(self.refusal)
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": "".join(call["arguments"])},
                }
                for _, call in sorted(self.tool_calls.items())
            ]
        choice: Dict[str, Any] = {
            "index": self.index,
            "message": message,
            "finish_reason": self.finish_reason,
        }
        if self.text is not None:
            choice["text"] = "".join(self.text)
        return choice


def _append(parts: Optional[List[str]], delta: Any) -> Optional[List[str]]:
    if not isinstance(delta, str):
        return parts
    if parts is None:
        return [delta]
    parts.append(delta)
    return parts


class _StreamAggregator:
    """Rebuilds the response a stream amounts to.

    Chat Completions and completions responses are rebuilt from their chunks' deltas. A Responses
    API stream's lifecycle events (created, in progress, completed...) carry the response so far,
    so the last one is the response. Its other events are deltas of it, so they're skipped.
    """

    def __init__(self) -> None:
        self._response: Dict[str, Any] = {}
        self._choices: Dict[Any, _StreamedChoice] = {}
        self._lifecycle_response: Any = None
        self._failed = False

    def add(self, item: Any) -> None:
        if self._failed:
            return
        try:
            self._add(item)
        except Exception as e:
            self._failed = True
            logger.warning("[Statsig] Failed to read a streamed OpenAI chunk: %s", e)

    def _add(self, item: Any) -> None:
        lifecycle_response = _get_field(item, "response")
        if lifecycle_response is not None:
            self._lifecycle_response = lifecycle_response
            return
        event_type = _get_field(item, "type")
        if isinstance(event_type, str) and event_type.startswith("response."):
            return

        chunk = _convert_result_to_dict(item) or {}
        for key in ("id", "model", "service_tier", "system_fingerprint"):
            if self._response.get(key) is None and chunk.get(key) is not None:
                self._response[key] = chunk[key]
        if chunk.get("usage"):
            self._response["usage"] = chunk["usage"]
        for choice in chunk.get("choices") or ():
            if isinstance(choice, Mapping):
                index = choice.get("index") or 0
                self._choices.setdefault(index, _StreamedChoice(index)).add(choice)

    def response(self) -> Optional[Mapping[str, Any]]:
        if self._failed:
            return None
        if self._lifecycle_response is not None:
            return _convert_result_to_dict(self._lifecycle_response)
        response = dict(self._response)
        if self._choices:
            response["choices"] = [
                choice.to_choice() for _, choice in sorted(self._choices.items())
            ]
        return response


def _stream_attributes(stream: _StreamAggregator, options: GenAICaptureOptions) -> Dict[str, Any]:
    response = stream.response()
    return extract_response_attributes(response, options) if response is not None else {}


def _trace_items(
    source: Iterator[Any], telemetry: SpanTelemetry, options: GenAICaptureOptions
) -> Generator[Any, None, None]:
    """Yields a stream's items, and ends the span once the stream ends or is closed.

    Like the Node SDK's TelemetryStream, the response is recorded when the stream has been read
    to the end. A stream closed early ends the span without it.
    """
    stream = _StreamAggregator()
    first = True
    try:
        for item in source:
            if first:
                telemetry.record_time_to_first_token()
                first = False
            stream.add(item)
            yield item
        _set_attributes_safely(telemetry, lambda: _stream_attributes(stream, options))
        telemetry.set_status({"code": StatusCode.OK})
    except BaseException as e:
        if not isinstance(e, GeneratorExit):
            telemetry.fail(e)
        raise
    finally:
        telemetry.end()


async def _trace_async_items(
    source: AsyncIterator[Any], telemetry: SpanTelemetry, options: GenAICaptureOptions
) -> AsyncGenerator[Any, None]:
    """The async version of _trace_items."""
    stream = _StreamAggregator()
    first = True
    try:
        async for item in source:
            if first:
                telemetry.record_time_to_first_token()
                first = False
            stream.add(item)
            yield item
        _set_attributes_safely(telemetry, lambda: _stream_attributes(stream, options))
        telemetry.set_status({"code": StatusCode.OK})
    except BaseException as e:
        if not isinstance(e, GeneratorExit):
            telemetry.fail(e)
        raise
    finally:
        telemetry.end()


class _TracedStream:
    """A model call's stream, such as `create(stream=True)` returns.

    Iterating it iterates the stream. The span ends once the stream has been read to the end, or
    when the stream is closed. Other attributes and methods are the stream's.
    """

    def __init__(self, stream: Any, telemetry: SpanTelemetry, options: GenAICaptureOptions):
        self._stream = stream
        self._telemetry = telemetry
        self._items = _trace_items(iter(stream), telemetry, options)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __iter__(self) -> "_TracedStream":
        return self

    def __next__(self) -> Any:
        return next(self._items)

    def __enter__(self) -> "_TracedStream":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        try:
            close = getattr(self._stream, "close", None)
            if callable(close):
                close()
        finally:
            self._end_span()

    def _end_span(self) -> None:
        self._items.close()
        self._telemetry.end()


class _TracedResponseStream(_TracedStream):
    """The stream `responses.stream()` gives, read through the span's stream."""

    def until_done(self) -> "_TracedResponseStream":
        for _ in self:
            pass
        return self

    def get_final_response(self) -> Any:
        self.until_done()
        return self._stream.get_final_response()


class _TracedStreamManager:
    """The stream manager `responses.stream()` returns.

    The request is sent when the manager is entered, so the span starts then, and the request is
    sent with the span active. The span ends when the stream has been read to the end, or when
    the manager exits.
    """

    def __init__(
        self,
        manager: Any,
        start_span: Callable[[], SpanTelemetry],
        options: GenAICaptureOptions,
    ):
        self._manager = manager
        self._start_span = start_span
        self._options = options
        self._stream: Optional[_TracedResponseStream] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)

    def __enter__(self) -> _TracedResponseStream:
        telemetry = self._start_span()
        try:
            with _activated(trace.set_span_in_context(telemetry.span)):
                stream = self._manager.__enter__()
        except BaseException as e:
            _end_failed(telemetry, e)
            raise
        telemetry.set_attributes({"gen_ai.request.stream": True})
        self._stream = _TracedResponseStream(stream, telemetry, self._options)
        return self._stream

    def __exit__(self, *exc_info: Any) -> Any:
        try:
            return self._manager.__exit__(*exc_info)
        finally:
            if self._stream is not None:
                self._stream._end_span()


class _AsyncTracedStream:
    """The async version of _TracedStream."""

    def __init__(self, stream: Any, telemetry: SpanTelemetry, options: GenAICaptureOptions):
        self._stream = stream
        self._telemetry = telemetry
        self._items = _trace_async_items(stream.__aiter__(), telemetry, options)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __aiter__(self) -> "_AsyncTracedStream":
        return self

    async def __anext__(self) -> Any:
        return await self._items.__anext__()

    async def __aenter__(self) -> "_AsyncTracedStream":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def close(self) -> None:
        try:
            close = getattr(self._stream, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
        finally:
            await self._end_span()

    async def _end_span(self) -> None:
        await self._items.aclose()
        self._telemetry.end()


class _AsyncTracedResponseStream(_AsyncTracedStream):
    """The async version of _TracedResponseStream."""

    async def until_done(self) -> "_AsyncTracedResponseStream":
        async for _ in self:
            pass
        return self

    async def get_final_response(self) -> Any:
        await self.until_done()
        return await self._stream.get_final_response()


class _AsyncTracedStreamManager:
    """The async version of _TracedStreamManager."""

    def __init__(
        self,
        manager: Any,
        start_span: Callable[[], SpanTelemetry],
        options: GenAICaptureOptions,
    ):
        self._manager = manager
        self._start_span = start_span
        self._options = options
        self._stream: Optional[_AsyncTracedResponseStream] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)

    async def __aenter__(self) -> _AsyncTracedResponseStream:
        telemetry = self._start_span()
        try:
            with _activated(trace.set_span_in_context(telemetry.span)):
                stream = await self._manager.__aenter__()
        except BaseException as e:
            _end_failed(telemetry, e)
            raise
        telemetry.set_attributes({"gen_ai.request.stream": True})
        self._stream = _AsyncTracedResponseStream(stream, telemetry, self._options)
        return self._stream

    async def __aexit__(self, *exc_info: Any) -> Any:
        try:
            return await self._manager.__aexit__(*exc_info)
        finally:
            if self._stream is not None:
                await self._stream._end_span()


class OpenAIWrapper(BaseWrapper):
    def __init__(self, oai_client: Any, options: WrapOpenAIOptions):
        super().__init__(oai_client, options)

        # Chat Completions
        if hasattr(oai_client, "chat"):
            self.chat = ChatWrapper(oai_client.chat, options)

        # Legacy Completions
        if hasattr(oai_client, "completions"):
            self.completions = CompletionsWrapper(oai_client.completions, options)

        # Embeddings
        if hasattr(oai_client, "embeddings"):
            self.embeddings = EmbeddingsWrapper(oai_client.embeddings, options)

        # Images
        if hasattr(oai_client, "images"):
            self.images = ImagesWrapper(oai_client.images, options)

        # Responses
        if hasattr(oai_client, "responses"):
            self.responses = ResponsesWrapper(oai_client.responses, options)


class ChatWrapper(BaseWrapper):
    def __init__(self, chat, options: WrapOpenAIOptions):
        super().__init__(chat, options)
        self.completions = ChatCompletionsWrapper(chat.completions, options)


class ChatCompletionsWrapper(BaseWrapper):
    def __init__(self, completions, options: WrapOpenAIOptions):
        super().__init__(completions, options)
        self.create = self._wrap_call("create", "openai.chat.completions.create")


class CompletionsWrapper(BaseWrapper):
    def __init__(self, completions, options: WrapOpenAIOptions):
        super().__init__(completions, options)
        self.create = self._wrap_call("create", "openai.completions.create")


class EmbeddingsWrapper(BaseWrapper):
    def __init__(self, embeddings, options: WrapOpenAIOptions):
        super().__init__(embeddings, options)
        self.create = self._wrap_call("create", "openai.embeddings.create")


class ImagesWrapper(BaseWrapper):
    def __init__(self, images, options: WrapOpenAIOptions):
        super().__init__(images, options)
        self.generate = self._wrap_call("generate", "openai.images.generate")


class ResponsesWrapper(BaseWrapper):
    def __init__(self, responses, options: WrapOpenAIOptions):
        super().__init__(responses, options)
        self.create = self._wrap_call("create", "openai.responses.create")
        if hasattr(responses, "parse"):
            self.parse = self._wrap_call("parse", "openai.responses.parse")
        if hasattr(responses, "stream"):
            self.stream = self._wrap_stream_manager("stream", "openai.responses.stream")
