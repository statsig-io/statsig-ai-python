"""Traces google-genai calls. Ported from the Node SDK's gemini.ts.

google-genai is an optional dependency: nothing here imports it, and requests and responses are
read by attribute (or by key, for dicts).
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from opentelemetry.trace import Tracer

from ..otel.singleton import OtelSingleton
from .configs import GenAICaptureOptions, WrapOpenAIOptions
from .gemini_messages import (
    as_list,
    enum_value,
    get_field,
    to_gemini_input_messages,
    to_gemini_output_messages,
)
from .gen_ai_span import start_gen_ai_span, trace_gen_ai_call

logger = logging.getLogger(__name__)

PROVIDER_NAME = "gcp.gen_ai"
OPERATION_NAME = "generate_content"
GENERATE_CONTENT = "google.models.generate_content"
GENERATE_CONTENT_STREAM = "google.models.generate_content_stream"

WRAPPED_METHODS: Dict[str, str] = {
    "generate_content": GENERATE_CONTENT,
    "generate_content_stream": GENERATE_CONTENT_STREAM,
}


def wrap_gemini(client: Any, options: Optional[WrapOpenAIOptions] = None) -> Any:
    """Wraps a google-genai `Client` (`genai.Client(...)`).

    Each `client.models.generate_content` and `client.models.generate_content_stream` call, and
    their async versions on `client.aio.models`, is recorded as a span and a `statsig::gen_ai`
    event. A streamed call's span ends once the stream has been consumed. Other attributes of
    the client are passed through.

    Returns the client unchanged, with a warning, if it doesn't have `models.generate_content`.
    """
    if not _has_generate_content(client):
        logger.warning("[Statsig] Object does not have Google GenAI APIs, not wrapping.")
        return client
    return GeminiWrapper(client, options or WrapOpenAIOptions())


class GeminiWrapper:
    """A google-genai `Client`, or its `aio` client, whose `models` calls are traced."""

    def __init__(self, client: Any, options: WrapOpenAIOptions):
        self._client = client
        self.models = GeminiModelsWrapper(client.models, options.gen_ai_capture_options)
        aio = getattr(client, "aio", None)
        if _has_generate_content(aio):
            self.aio = GeminiWrapper(aio, options)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class GeminiModelsWrapper:
    def __init__(self, models: Any, capture_options: GenAICaptureOptions):
        self._models = models
        self._capture_options = capture_options
        for method_name, source_name in WRAPPED_METHODS.items():
            original = getattr(models, method_name, None)
            if callable(original):
                setattr(self, method_name, self._wrap(original, source_name))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models, name)

    def _wrap(self, original: Callable[..., Any], source_name: str) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(original):

            async def async_method(*args: Any, **kwargs: Any) -> Any:
                return await self._trace(original, source_name, args, kwargs)

            return async_method

        def method(*args: Any, **kwargs: Any) -> Any:
            return self._trace(original, source_name, args, kwargs)

        return method

    def _trace(
        self,
        original: Callable[..., Any],
        source_name: str,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        capture = self._capture_options
        model = kwargs.get("model")
        telemetry = start_gen_ai_span(
            _get_tracer(),
            f"{OPERATION_NAME} {model if model is not None else 'unknown'}",
            PROVIDER_NAME,
            extract_gemini_request_attributes(source_name, kwargs, capture),
            capture,
        )
        return trace_gen_ai_call(
            telemetry,
            lambda: original(*args, **kwargs),
            on_result=lambda response: extract_gemini_response_attributes(response, capture),
            on_stream=lambda chunks: extract_gemini_response_attributes(
                aggregate_gemini_stream_chunks(chunks), capture
            ),
        )


def _get_tracer() -> Tracer:
    # Looked up on each call, so a client wrapped before initialize_tracing() is traced.
    return OtelSingleton.get_tracer_provider_static().get_tracer("statsig-gemini-proxy")


def _has_generate_content(client: Any) -> bool:
    return callable(getattr(getattr(client, "models", None), "generate_content", None))


def extract_gemini_request_attributes(
    source_name: str, params: Mapping[str, Any], options: GenAICaptureOptions
) -> Dict[str, Any]:
    config = params.get("config")
    attrs: Dict[str, Any] = {
        "gen_ai.operation.name": OPERATION_NAME,
        "gen_ai.operation.source_name": source_name,
        "gen_ai.provider.name": PROVIDER_NAME,
        "gen_ai.request.model": params.get("model"),
        "gen_ai.request.max_tokens": get_field(config, "max_output_tokens"),
        "gen_ai.request.temperature": get_field(config, "temperature"),
        "gen_ai.request.top_p": get_field(config, "top_p"),
        "gen_ai.request.top_k": get_field(config, "top_k"),
        "gen_ai.request.stop_sequences": get_field(config, "stop_sequences"),
        "gen_ai.request.seed": get_field(config, "seed"),
    }
    if source_name == GENERATE_CONTENT_STREAM:
        attrs["gen_ai.request.stream"] = True
    if options.capture_all or options.capture_input_messages:
        attrs["gen_ai.input.messages"] = to_gemini_input_messages(params.get("contents"))
    system_instruction = get_field(config, "system_instruction")
    if (options.capture_all or options.capture_system_instructions) and (
        system_instruction is not None
    ):
        attrs["gen_ai.system_instructions"] = [
            part
            for message in to_gemini_input_messages(system_instruction)
            for part in message["parts"]
        ]
    if options.capture_all or options.capture_tool_definitions:
        attrs["gen_ai.tool.definitions"] = _to_json_value(get_field(config, "tools"))
    return attrs


def extract_gemini_response_attributes(
    response: Any, options: GenAICaptureOptions
) -> Dict[str, Any]:
    candidates = as_list(get_field(response, "candidates"))
    attrs: Dict[str, Any] = {
        "gen_ai.response.id": get_field(response, "response_id"),
        "gen_ai.response.model": get_field(response, "model_version"),
        **_extract_usage_attributes(get_field(response, "usage_metadata")),
    }
    finish_reasons = [
        enum_value(reason)
        for reason in (get_field(candidate, "finish_reason") for candidate in candidates)
        if reason
    ]
    if finish_reasons:
        attrs["gen_ai.response.finish_reasons"] = finish_reasons
    if options.capture_all or options.capture_output_messages:
        attrs["gen_ai.output.messages"] = to_gemini_output_messages(candidates)
    return attrs


def _extract_usage_attributes(usage: Any) -> Dict[str, Any]:
    input_tokens = _token_count(get_field(usage, "prompt_token_count"))
    # Thinking tokens are output tokens, as in the Node SDK.
    reasoning_tokens = _token_count(get_field(usage, "thoughts_token_count"))
    output_tokens = _token_count(get_field(usage, "candidates_token_count")) + reasoning_tokens
    cached_tokens = _token_count(get_field(usage, "cached_content_token_count"))
    attrs: Dict[str, Any] = {}
    if input_tokens > 0:
        attrs["gen_ai.usage.input_tokens"] = input_tokens
    if output_tokens > 0:
        attrs["gen_ai.usage.output_tokens"] = output_tokens
    if input_tokens + output_tokens > 0:
        attrs["statsig.gen_ai.usage.total_tokens"] = input_tokens + output_tokens
    if cached_tokens > 0:
        attrs["statsig.gen_ai.usage.input_cached_tokens"] = cached_tokens
    if reasoning_tokens > 0:
        attrs["statsig.gen_ai.usage.output_reasoning_tokens"] = reasoning_tokens
    return attrs


def aggregate_gemini_stream_chunks(chunks: Sequence[Any]) -> Dict[str, Any]:
    """Merges generate_content_stream chunks into one response, joining text that arrives in
    pieces for each candidate. Keeps the first response id and model version, and the last
    usage and finish reason.
    """
    candidates: List[Dict[str, Any]] = []
    response: Dict[str, Any] = {
        "response_id": None,
        "model_version": None,
        "usage_metadata": None,
        "candidates": candidates,
    }
    for chunk in chunks:
        for key in ("response_id", "model_version"):
            if response[key] is None:
                response[key] = get_field(chunk, key)
        usage = get_field(chunk, "usage_metadata")
        if usage is not None:
            response["usage_metadata"] = usage
        for index, candidate in enumerate(as_list(get_field(chunk, "candidates"))):
            if index == len(candidates):
                candidates.append({"content": {"role": "model", "parts": []}})
            merged = candidates[index]
            finish_reason = get_field(candidate, "finish_reason")
            if finish_reason:
                merged["finish_reason"] = finish_reason
            parts: List[Any] = merged["content"]["parts"]
            for part in as_list(get_field(get_field(candidate, "content"), "parts")):
                _merge_part(parts, part)
    return response


def _merge_part(parts: List[Any], part: Any) -> None:
    text = get_field(part, "text")
    if not isinstance(text, str):
        parts.append(part)
        return
    thought = bool(get_field(part, "thought"))
    last = parts[-1] if parts else None
    # Text parts are copied into dicts below, so the last one can be extended in place.
    if (
        isinstance(last, dict)
        and isinstance(last.get("text"), str)
        and bool(last.get("thought")) == thought
    ):
        last["text"] += text
    else:
        parts.append({"text": text, "thought": thought})


def _to_json_value(value: Any) -> Any:
    """Converts tool definitions, which can be SDK objects or Python functions, to JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json", exclude_none=True)
        except Exception:
            pass
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return value


def _token_count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
