import inspect
import logging
from typing import Any, Optional
from opentelemetry.trace import SpanKind, StatusCode, Tracer

from statsig_ai.otel.conventions import STATSIG_ATTR_SPAN_TYPE, StatsigSpanType

from .genai_attribute_helper import (
    extract_genai_attributes,
    extract_opt_in_attributes,
    extract_oai_usage_attributes,
)
from ..otel.singleton import OtelSingleton
from .configs import WrapOpenAIOptions
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
    def __init__(self, client: Any, operation_name: str, options: WrapOpenAIOptions):
        self._client = client
        self._operation_name = operation_name
        self._options = options.gen_ai_capture_options

    def __getattr__(self, name):
        return getattr(self._client, name)

    def _start_span_with_telemetry(self, model: str, kwargs: dict) -> SpanTelemetry:
        # Use OTEL semantic name if available, otherwise strip "openai." prefix
        otel_name = OTEL_OP_NAME_MAP.get(
            self._operation_name, self._operation_name.replace("openai.", "")
        )
        span_name = f"{otel_name} {model}"
        span = _get_tracer().start_span(span_name, kind=SpanKind.CLIENT)
        telemetry = SpanTelemetry(span, span_name, provider_name="openai")
        telemetry.set_attributes({STATSIG_ATTR_SPAN_TYPE: StatsigSpanType.GEN_AI})

        base_attrs = extract_genai_attributes(
            "openai", self._operation_name, model, kwargs, None, otel_operation_name=otel_name
        )
        telemetry.set_attributes(base_attrs)

        return telemetry

    def _get_stream_parser(self):
        if "chat.completions" in self._operation_name:
            return _parse_chat_completion_chunks
        if "responses" in self._operation_name:
            return _parse_responses_api_chunks
        return _parse_generic_chunks

    def _wrap_call(self, fn_name: str):
        original_fn = getattr(self._client, fn_name)
        stream_parser = self._get_stream_parser()

        def wrapper(*args, **kwargs):
            model = kwargs.get("model", "unknown")
            telemetry = self._start_span_with_telemetry(model, kwargs)
            should_end_telemetry = True

            try:
                result = original_fn(*args, **kwargs)

                # Async method: delegate to async handler to await and check if streaming
                if inspect.iscoroutine(result):
                    should_end_telemetry = False
                    return self._handle_async_result(
                        result, telemetry, model, kwargs, stream_parser
                    )

                # Sync streaming: wrap iterator to collect chunks for telemetry
                if kwargs.get("stream") and hasattr(result, "__iter__"):
                    should_end_telemetry = False
                    return self._wrap_sync_stream(result, telemetry, model, kwargs, stream_parser)

                # Sync non-streaming: extract telemetry immediately from complete response
                self._record_attributes(telemetry, model, kwargs, result)
                telemetry.record_time_to_first_token()
                telemetry.set_status({"code": StatusCode.OK})
                return result

            except Exception as e:
                telemetry.fail(e)
                raise
            finally:
                if should_end_telemetry:
                    telemetry.end()

        return wrapper

    async def _handle_async_result(
        self, coro, telemetry: SpanTelemetry, model: str, kwargs: dict, parser
    ):
        should_end_telemetry = True

        try:
            # Await the coroutine to get the actual result
            result = await coro

            # Async streaming: wrap async iterator to collect chunks for telemetry
            if kwargs.get("stream") and hasattr(result, "__aiter__"):
                should_end_telemetry = False
                return self._wrap_async_stream(result, telemetry, model, kwargs, parser)

            # Async non-streaming: extract telemetry immediately from complete response
            self._record_attributes(telemetry, model, kwargs, result)
            telemetry.set_status({"code": StatusCode.OK})
            return result

        except Exception as e:
            telemetry.fail(e)
            raise
        finally:
            if should_end_telemetry:
                telemetry.end()

    def _wrap_async_stream(
        self, stream, telemetry: SpanTelemetry, model: str, kwargs: dict, parser
    ):
        async def gen():
            all_chunks = []
            first = True
            try:
                async for chunk in stream:
                    if first:
                        telemetry.record_time_to_first_token()
                        first = False
                    all_chunks.append(chunk)
                    yield chunk

                attrs = parser(all_chunks, self._operation_name, model, kwargs)
                telemetry.set_attributes(attrs)
                telemetry.set_status({"code": StatusCode.OK})
            except Exception as e:
                telemetry.fail(e)
                raise
            finally:
                telemetry.end()

        return gen()

    def _wrap_sync_stream(self, stream, telemetry: SpanTelemetry, model: str, kwargs: dict, parser):
        def gen():
            all_chunks = []
            first = True
            try:
                for chunk in stream:
                    if first:
                        telemetry.record_time_to_first_token()
                        first = False
                    all_chunks.append(chunk)
                    yield chunk

                attrs = parser(all_chunks, self._operation_name, model, kwargs)
                telemetry.set_attributes(attrs)
                telemetry.set_status({"code": StatusCode.OK})
            except Exception as e:
                telemetry.fail(e)
                raise
            finally:
                telemetry.end()

        return gen()

    def _record_attributes(self, telemetry: SpanTelemetry, model: str, kwargs: dict, result: Any):
        result_dict = _convert_result_to_dict(result)
        if result_dict is None:
            logging.warning(
                "[Statsig] Failed to convert result to dict. Type: %s, Operation: %s",
                type(result),
                self._operation_name,
            )
            return

        otel_name = OTEL_OP_NAME_MAP.get(
            self._operation_name, self._operation_name.replace("openai.", "")
        )
        attrs = extract_genai_attributes(
            "openai",
            self._operation_name,
            model,
            kwargs,
            result_dict,
            otel_operation_name=otel_name,
        )
        opt_in = extract_opt_in_attributes(self._options, kwargs, result_dict)
        telemetry.set_attributes(attrs)
        telemetry.set_attributes(opt_in)


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


def _parse_chat_completion_chunks(items, _operation_name: str, _model: str, _kwargs: dict):
    attrs: dict[str, Any] = {
        "gen_ai.request.stream": True,
    }

    choice_deltas: dict[int, list[str]] = {}
    choice_roles: dict[int, str] = {}
    choice_finish_reasons: dict[int, str] = {}

    response_id = None
    response_model = None
    usage_data = None

    for item in items:
        parsed_item = _convert_result_to_dict(item) or {}

        if response_id is None:
            response_id = parsed_item.get("id")
            response_model = parsed_item.get("model")

        if "usage" in parsed_item:
            usage_data = parsed_item["usage"]

        choices = parsed_item.get("choices", [])
        for choice in choices:
            index = choice.get("index", 0)
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            if "role" in delta and index not in choice_roles:
                choice_roles[index] = delta["role"]

            if "content" in delta and delta["content"]:
                if index not in choice_deltas:
                    choice_deltas[index] = []
                choice_deltas[index].append(delta["content"])

            if finish_reason:
                choice_finish_reasons[index] = finish_reason

    if response_id:
        attrs["gen_ai.response.id"] = response_id
    if response_model:
        attrs["gen_ai.response.model"] = response_model

    if usage_data:
        attrs.update(extract_oai_usage_attributes(usage_data))

    result_choices = []
    for index in sorted(choice_deltas):
        content = "".join(choice_deltas[index])
        role = choice_roles.get(index, "assistant")
        finish_reason = choice_finish_reasons.get(index, "stop")

        result_choices.append(
            {
                "index": index,
                "message": {
                    "role": role,
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        )

    if choice_finish_reasons:
        attrs["gen_ai.response.finish_reasons"] = list(set(choice_finish_reasons.values()))

    return attrs


def _parse_responses_api_chunks(items, _operation_name: str, _model: str, _kwargs: dict):
    attrs: dict[str, Any] = {
        "gen_ai.request.stream": True,
    }

    delta_aggregator: dict[tuple[int, int], list[str]] = {}
    message_metadata: dict[int, dict[str, Any]] = {}

    first_response_data = None
    final_response = None

    for item in items:
        parsed_item = _convert_result_to_dict(item) or {}
        event_type = parsed_item.get("type")

        if first_response_data is None and event_type == "response.created":
            first_response_data = parsed_item.get("response", {})

        if event_type == "response.output_text.delta":
            output_index = parsed_item.get("output_index", 0)
            content_index = parsed_item.get("content_index", 0)
            delta = parsed_item.get("delta", "")

            key = (output_index, content_index)
            if key not in delta_aggregator:
                delta_aggregator[key] = []
            delta_aggregator[key].append(delta)

        elif event_type == "response.output_item.added":
            output_index = parsed_item.get("output_index", 0)
            item_data = parsed_item.get("item", {})
            message_metadata[output_index] = {
                "role": item_data.get("role", "assistant"),
                "id": item_data.get("id"),
            }

        elif event_type == "response.completed":
            final_response = parsed_item.get("response", {})

    if final_response:
        if response_id := final_response.get("id"):
            attrs["gen_ai.response.id"] = response_id
        if response_model := final_response.get("model"):
            attrs["gen_ai.response.model"] = response_model

        usage = final_response.get("usage", {})
        attrs.update(extract_oai_usage_attributes(usage))

        if status := final_response.get("status"):
            attrs["gen_ai.response.finish_reasons"] = [status]

    choices = []
    output_indices = sorted(set(idx for idx, _ in delta_aggregator))

    for output_idx in output_indices:
        content_parts = []
        content_indices = sorted(
            set(c_idx for o_idx, c_idx in delta_aggregator if o_idx == output_idx)
        )

        for content_idx in content_indices:
            key = (output_idx, content_idx)
            text = "".join(delta_aggregator[key])
            if text:
                content_parts.append(text)

        metadata = message_metadata.get(output_idx, {})

        choice = {
            "index": output_idx,
            "message": {
                "role": metadata.get("role", "assistant"),
                "content": "".join(content_parts),
            },
            "finish_reason": "completed" if final_response else "incomplete",
        }
        choices.append(choice)

    if choices:
        attrs["choices"] = choices

    return attrs


def _parse_generic_chunks(items, operation_name: str, model: str, kwargs: dict):
    last_chunk = None
    for item in items:
        chunk_dict = _convert_result_to_dict(item)
        if chunk_dict:
            last_chunk = chunk_dict

    if last_chunk:
        otel_name = OTEL_OP_NAME_MAP.get(operation_name, operation_name.replace("openai.", ""))
        attrs = extract_genai_attributes(
            "openai", operation_name, model, kwargs, last_chunk, otel_operation_name=otel_name
        )
        attrs["gen_ai.request.stream"] = True
        return attrs

    return {
        "gen_ai.request.stream": True,
    }


class OpenAIWrapper(BaseWrapper):
    def __init__(self, oai_client: Any, options: WrapOpenAIOptions):
        super().__init__(oai_client, "", options)

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
        super().__init__(chat, "openai.chat.completions.create", options)
        self.completions = ChatCompletionsWrapper(chat.completions, options)


class ChatCompletionsWrapper(BaseWrapper):
    def __init__(self, completions, options: WrapOpenAIOptions):
        super().__init__(completions, "openai.chat.completions.create", options)
        self.create = self._wrap_call("create")


class CompletionsWrapper(BaseWrapper):
    def __init__(self, completions, options: WrapOpenAIOptions):
        super().__init__(completions, "openai.completions.create", options)
        self.create = self._wrap_call("create")


class EmbeddingsWrapper(BaseWrapper):
    def __init__(self, embeddings, options: WrapOpenAIOptions):
        super().__init__(embeddings, "openai.embeddings.create", options)
        self.create = self._wrap_call("create")


class ImagesWrapper(BaseWrapper):
    def __init__(self, images, options: WrapOpenAIOptions):
        super().__init__(images, "openai.images.generate", options)
        self.generate = self._wrap_call("generate")


class ResponsesWrapper(BaseWrapper):
    def __init__(self, responses, options: WrapOpenAIOptions):
        super().__init__(responses, "openai.responses.create", options)
        self.create = self._wrap_call("create")
