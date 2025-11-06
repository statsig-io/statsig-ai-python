import time
import logging
from typing import Any
from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode
from .configs import WrapOpenAIOptions, GenAICaptureOptions
from .span_telemetry import SpanTelemetry
from .genai_attribute_helper import extract_genai_attributes, extract_opt_in_attributes

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("statsig-openai-proxy")


class BaseWrapper:
    def __init__(self, client: Any):
        self._client = client

    def __getattr__(self, name):
        return getattr(self._client, name)


class OpenAIWrapper(BaseWrapper):
    def __init__(self, oai_client: Any, options: WrapOpenAIOptions):
        super().__init__(oai_client)
        self._options = options.gen_ai_capture_options
        # Chat Completions
        if hasattr(oai_client, "chat") and hasattr(oai_client.chat, "completions"):
            self.chat = type(
                "ChatNamespace",
                (),
                {"completions": ChatCompletionsWrapper(oai_client.chat.completions, options)},
            )()

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


class ChatCompletionsWrapper(BaseWrapper):
    operation_name = "chat"
    _options: GenAICaptureOptions

    def __init__(self, client: Any, options: WrapOpenAIOptions):
        super().__init__(client)
        self._options = options.gen_ai_capture_options

    def create(self, *args, **kwargs):
        model = kwargs.get("model")
        span_name = f"{self.operation_name} {model}"
        span = tracer.start_span(span_name, kind=SpanKind.CLIENT)
        telemetry = SpanTelemetry(span, span_name, 40_000, provider_name="openai")
        try:
            res = self._client.create(*args, **kwargs)
            attributes = extract_genai_attributes("openai", self.operation_name, model, kwargs, res)
            opt_in_attributes = extract_opt_in_attributes(self._options, kwargs, res)
            telemetry.set_attributes(attributes)
            telemetry.set_attributes(opt_in_attributes)
            telemetry.set_status({"code": StatusCode.OK})
            return res
        except Exception as e:
            telemetry.fail(e)
            raise
        finally:
            telemetry.end()


class CompletionsWrapper(BaseWrapper):
    operation_name = "text_completion"
    _options: GenAICaptureOptions

    def __init__(self, client: Any, options: WrapOpenAIOptions):
        super().__init__(client)
        self._options = options.gen_ai_capture_options

    def create(self, *args, **kwargs):
        model = kwargs.get("model")
        span_name = f"{self.operation_name} {model}"
        span = tracer.start_span(span_name, kind=SpanKind.CLIENT)
        telemetry = SpanTelemetry(span, span_name, 40_000, provider_name="openai")
        try:
            res = self._client.create(*args, **kwargs)
            attributes = extract_genai_attributes("openai", self.operation_name, model, kwargs, res)
            opt_in_attributes = extract_opt_in_attributes(self._options, kwargs, res)
            telemetry.set_attributes(opt_in_attributes)
            telemetry.set_attributes(attributes)
            telemetry.set_status({"code": StatusCode.OK})
            return res
        except Exception as e:
            telemetry.fail(e)
            raise
        finally:
            telemetry.end()


class EmbeddingsWrapper(BaseWrapper):
    operation_name = "embeddings"
    _options: GenAICaptureOptions

    def __init__(self, client: Any, options: WrapOpenAIOptions):
        super().__init__(client)
        self._options = options.gen_ai_capture_options

    def create(self, *args, **kwargs):
        model = kwargs.get("model")
        span_name = f"{self.operation_name} {model}"
        span = tracer.start_span(span_name, kind=SpanKind.CLIENT)
        telemetry = SpanTelemetry(span, span_name, 40_000, provider_name="openai")
        try:
            res = self._client.create(*args, **kwargs)
            attributes = extract_genai_attributes("openai", self.operation_name, model, kwargs, res)
            opt_in_attributes = extract_opt_in_attributes(self._options, kwargs, res)
            telemetry.set_attributes(attributes)
            telemetry.set_attributes(opt_in_attributes)
            telemetry.set_status({"code": StatusCode.OK})
            return res
        except Exception as e:
            telemetry.fail(e)
            raise
        finally:
            telemetry.end()


class ImagesWrapper(BaseWrapper):
    operation_name = "images.generate"
    _options: GenAICaptureOptions

    def __init__(self, client: Any, options: WrapOpenAIOptions):
        super().__init__(client)
        self._options = options.gen_ai_capture_options

    def generate(self, *args, **kwargs):
        model = kwargs.get("model")
        span_name = f"{self.operation_name} {model}"
        span = tracer.start_span(span_name, kind=SpanKind.CLIENT)
        telemetry = SpanTelemetry(span, span_name, 40_000, provider_name="openai")
        try:
            res = self._client.generate(*args, **kwargs)
            attributes = extract_genai_attributes("openai", self.operation_name, model, kwargs, res)
            opt_in_attributes = extract_opt_in_attributes(self._options, kwargs, res)
            telemetry.set_attributes(attributes)
            telemetry.set_attributes(opt_in_attributes)
            telemetry.set_status({"code": StatusCode.OK})
            return res
        except Exception as e:
            telemetry.fail(e)
            raise
        finally:
            telemetry.end()


class ResponsesWrapper(BaseWrapper):
    operation_name = "responses.create"
    _options: GenAICaptureOptions

    def __init__(self, client: Any, options: WrapOpenAIOptions):
        super().__init__(client)
        self._options = options.gen_ai_capture_options

    def create(self, *args, **kwargs):
        model = kwargs.get("model")
        span_name = f"{self.operation_name} {model}"
        span = tracer.start_span(span_name, kind=SpanKind.CLIENT)
        telemetry = SpanTelemetry(span, span_name, 40_000, provider_name="openai")
        try:
            res = self._client.create(*args, **kwargs)
            attributes = extract_genai_attributes("openai", self.operation_name, model, kwargs, res)
            opt_in_attributes = extract_opt_in_attributes(self._options, kwargs, res)
            telemetry.set_attributes(attributes)
            telemetry.set_attributes(opt_in_attributes)
            telemetry.set_status({"code": StatusCode.OK})
            return res
        except Exception as e:
            telemetry.fail(e)
            raise
        finally:
            telemetry.end()
