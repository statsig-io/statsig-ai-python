import time
import logging
from typing import Any, Dict, Optional
from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode
from .span_telemetry import SpanTelemetry

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("statsig-openai-proxy")


class BaseWrapper:
    def __init__(self, client: Any):
        self._client = client

    def __getattr__(self, name):
        return getattr(self._client, name)


class OpenAIWrapper(BaseWrapper):
    def __init__(self, oai_client: Any):
        super().__init__(oai_client)

        # Chat Completions
        if hasattr(oai_client, "chat") and hasattr(oai_client.chat, "completions"):
            self.chat = type(
                "ChatNamespace",
                (),
                {"completions": ChatCompletionsWrapper(oai_client.chat.completions)},
            )()

        # Legacy Completions
        if hasattr(oai_client, "completions"):
            self.completions = CompletionsWrapper(oai_client.completions)

        # Embeddings
        if hasattr(oai_client, "embeddings"):
            embeddings_client = oai_client.embeddings
            class_name = type(embeddings_client).__name__.lower()
            if "async" in class_name:
                self.embeddings = EmbeddingsWrapper(embeddings_client)
            else:
                self.embeddings = EmbeddingsWrapper(embeddings_client)

        # Images
        if hasattr(oai_client, "images"):
            self.images = ImagesWrapper(oai_client.images)

        # Responses
        if hasattr(oai_client, "responses"):
            self.responses = ResponsesWrapper(oai_client.responses)


class ChatCompletionsWrapper(BaseWrapper):
    def create(self, *args, **kwargs):
        span_name = "openai.chat.completions.create"
        attrs = {
            "gen_ai.system": "openai",
            "gen_ai.operation.name": "chat.completions.create",
            "gen_ai.request.model": kwargs.get("model"),
        }

        span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes=attrs)
        telemetry = SpanTelemetry(span, span_name, 40_000, provider_name="openai")

        t0 = time.time()
        try:
            result = self._client.create(*args, **kwargs)

            first_choice = getattr(result, "choices", [{}])[0]
            out_text = first_choice.get("message", {}).get("content", "")
            telemetry.set_attributes(
                {
                    "gen_ai.response.model": getattr(result, "model", None),
                    "gen_ai.completion": out_text,
                    "gen_ai.response.finish_reason": first_choice.get("finish_reason"),
                    "gen_ai.metrics.time_to_first_token_ms": (time.time() - t0) * 1000,
                }
            )
            telemetry.set_usage(getattr(result, "usage", None))
            telemetry.set_status({"code": StatusCode.OK})
            return result
        except Exception as e:
            telemetry.fail(e)
            raise
        finally:
            telemetry.end()


class CompletionsWrapper(BaseWrapper):
    def create(self, *args, **kwargs):
        span_name = "openai.completions.create"
        attrs = {
            "gen_ai.system": "openai",
            "gen_ai.operation.name": "completions.create",
            "gen_ai.request.model": kwargs.get("model"),
        }

        span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes=attrs)
        telemetry = SpanTelemetry(span, span_name, 40_000, provider_name="openai")

        t0 = time.time()
        try:
            result = self._client.create(*args, **kwargs)
            first = getattr(result, "choices", [{}])[0]
            telemetry.set_attributes(
                {
                    "gen_ai.completion": first.get("text", ""),
                    "gen_ai.response.finish_reason": first.get("finish_reason"),
                    "gen_ai.metrics.time_to_first_token_ms": (time.time() - t0) * 1000,
                }
            )
            telemetry.set_usage(getattr(result, "usage", None))
            telemetry.set_status({"code": StatusCode.OK})
            return result
        except Exception as e:
            telemetry.fail(e)
            raise
        finally:
            telemetry.end()


class EmbeddingsWrapper(BaseWrapper):
    def create(self, *args, **kwargs):
        span_name = "openai.embeddings.create"
        attrs = {
            "gen_ai.system": "openai",
            "gen_ai.operation.name": "embeddings.create",
            "gen_ai.request.model": kwargs.get("model"),
            "gen_ai.request.encoding_format": kwargs.get("encoding_format", "float"),
        }

        span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes=attrs)
        telemetry = SpanTelemetry(span, span_name, 40_000, provider_name="openai")

        try:
            res = self._client.create(*args, **kwargs)
            telemetry.set_attributes(
                {
                    "gen_ai.response.model": getattr(res, "model", None),
                    "gen_ai.embeddings.count": len(res.data),
                    "gen_ai.embeddings.dimension": len(res.data[0].embedding) if res.data else None,
                }
            )
            telemetry.set_usage(getattr(res, "usage", None))
            telemetry.set_status({"code": StatusCode.OK})
            return res
        except Exception as e:
            telemetry.fail(e)
            raise
        finally:
            telemetry.end()


class ImagesWrapper(BaseWrapper):
    def generate(self, *args, **kwargs):
        span_name = "openai.images.generate"
        attrs = {
            "gen_ai.system": "openai",
            "gen_ai.operation.name": "images.generate",
            "gen_ai.request.model": kwargs.get("model"),
        }

        span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes=attrs)
        telemetry = SpanTelemetry(span, span_name, 40_000, provider_name="openai")

        try:
            res = self._client.generate(*args, **kwargs)
            telemetry.set_attributes(
                {
                    "gen_ai.response.created": getattr(res, "created", None),
                    "gen_ai.images.count": len(res.data),
                }
            )
            telemetry.set_status({"code": StatusCode.OK})
            return res
        except Exception as e:
            telemetry.fail(e)
            raise
        finally:
            telemetry.end()


class ResponsesWrapper(BaseWrapper):
    def create(self, *args, **kwargs):
        span_name = "openai.responses.create"
        attrs = {
            "gen_ai.system": "openai",
            "gen_ai.operation.name": "responses.create",
            "gen_ai.request.model": kwargs.get("model"),
        }

        span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes=attrs)
        telemetry = SpanTelemetry(span, span_name, 40_000, provider_name="openai")

        t0 = time.time()
        try:
            res = self._client.create(*args, **kwargs)
            out_text = getattr(res, "output_text", None) or getattr(res, "content", [{}])[0].get(
                "text", ""
            )
            telemetry.set_attributes(
                {
                    "gen_ai.completion": out_text,
                    "gen_ai.metrics.time_to_first_token_ms": (time.time() - t0) * 1000,
                }
            )
            telemetry.set_usage(getattr(res, "usage", None))
            telemetry.set_status({"code": StatusCode.OK})
            return res
        except Exception as e:
            telemetry.fail(e)
            raise
        finally:
            telemetry.end()
