from typing import Any, Dict, Optional
import logging
import time

from opentelemetry.context import Context
from opentelemetry.trace import Span, Status, StatusCode, format_span_id, format_trace_id
from statsig_python_core import Statsig, StatsigUser

from ..otel.conventions import STATSIG_ATTR_ACTIVITY_ID
from ..otel.resources import StatsigResourceAttributes, create_resource_attributes
from ..otel.singleton import OtelSingleton
from ..otel.statsig_context import get_statsig_context_from_otel_context
from .js_format import js_json_stringify, js_string, utf16_length, utf16_slice

GEN_AI_EVENT_NAME = "statsig::gen_ai"
GEN_AI_EVENT_USER_ID = "statsig-ai-openai-wrapper"

logger = logging.getLogger(__name__)

MAX_JSON_CHARS = 40_000


class SpanTelemetry:
    def __init__(
        self,
        span: Span,
        span_name: str,
        provider_name: Optional[str] = None,
        max_json_chars: int = MAX_JSON_CHARS,
        resources: Optional[StatsigResourceAttributes] = None,
    ):
        self.span = span
        self.span_name = span_name
        self.max_json_chars = max_json_chars
        self.resources = resources if resources is not None else get_resources_for_event_user()
        self.metadata: Dict[str, str] = {}
        self.ended = False
        self.start_time = time.perf_counter()

        self.metadata["span.name"] = span_name
        # Model call spans name their provider. wrap() spans have none.
        if provider_name is not None:
            self.metadata["gen_ai.provider.name"] = provider_name
        ctx = span.get_span_context()
        self.metadata["span.trace_id"] = format_trace_id(ctx.trace_id)
        self.metadata["span.span_id"] = format_span_id(ctx.span_id)

    def set_attributes(self, kv: Dict[str, Any]) -> None:
        for key, value in kv.items():
            if value is None:
                continue
            # Node writes every object, arrays included, as JSON. Tuples are Python's other array.
            if isinstance(value, (list, tuple, dict)):
                self._set_json_safe(key, value)
                continue

            self._set_attribute_on_span_and_metadata(key, value)

    def _set_json_safe(self, key: str, value: Any) -> None:
        # Written as Node's JSON.stringify writes it. Lengths and the cap count UTF-16
        # code units, as a JavaScript string's length does.
        try:
            json_str = js_json_stringify(value)
            json_len = utf16_length(json_str)
            is_truncated = json_len > self.max_json_chars
            self._set_attribute_on_span_and_metadata(
                key,
                (
                    utf16_slice(json_str, self.max_json_chars) + "…(truncated)"
                    if is_truncated
                    else json_str
                ),
            )
            # Like the Node SDK, the full length is recorded whether or not the value was cut.
            self._set_attribute_on_span_and_metadata(f"{key}_len", json_len)
        except Exception:
            # Like JSON.stringify throwing in Node: a circular reference, or here also a
            # value JSON can't hold.
            self._set_attribute_on_span_and_metadata(key, "[[unserializable]]")

    def _set_attribute_on_span_and_metadata(self, key: str, value: Any) -> None:
        self.span.set_attribute(key, value)
        self.metadata[key] = attribute_value_to_metadata(value)

    def set_status(self, status: Dict[str, Any]) -> None:
        code: StatusCode = status.get("code", StatusCode.UNSET)
        message: Optional[str] = status.get("message")

        self.span.set_status(Status(code, message))
        self.metadata["span.status_code"] = code.name
        self.metadata["span.status_code_value"] = str(code.value)
        if message:
            self.metadata["span.status_message"] = str(message)

    def record_time_to_first_token(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        self.set_attributes({"statsig.gen_ai.server.time_to_first_token_ms": round(elapsed * 1000)})

    def record_exception(self, error: Any) -> None:
        # Like the Node SDK, the span gets OpenTelemetry's `exception` event, and the
        # type and message go in the event metadata only, not in the span attributes.
        if isinstance(error, BaseException):
            self.span.record_exception(error)
        elif isinstance(error, str):
            # What OpenTelemetry JS records for a string. Python's record_exception()
            # only takes exceptions.
            self.span.add_event("exception", {"exception.message": error})

        # Node reads error.name, then the constructor's name. In Python both are the class name.
        exception_type = None if error is None else type(error).__name__
        message = getattr(error, "message", None)
        if message is None:
            if isinstance(error, str):
                message = error
            elif isinstance(error, BaseException):
                # A JavaScript Error always has a message. str() is Python's equivalent.
                message = str(error)

        if exception_type:
            self.metadata["exception.type"] = exception_type
        if message:
            self.metadata["exception.message"] = js_string(message)

    def fail(self, error: Any) -> None:
        self.record_exception(error)
        # Node: error?.message ?? String(error)
        message = getattr(error, "message", None)
        self.set_status(
            {"code": StatusCode.ERROR, "message": str(error) if message is None else message}
        )

    def end(self, ctx: Optional[Context] = None) -> None:
        """Ends the span and logs its statsig::gen_ai event.

        The event user gets the activity ID of the statsig_context in ``ctx``, or in the
        current context if ``ctx`` is None.
        """
        if self.ended:
            return
        self.ended = True
        self.span.end()
        self._log_span_event(self.span_name, dict(self.metadata), ctx)

    def _log_span_event(
        self, span_name: str, metadata: Dict[str, str], ctx: Optional[Context] = None
    ) -> None:
        statsig = get_statsig_instance_for_logging()
        if not statsig:
            logger.warning(
                "[Statsig] No shared global StatsigAI instance found. "
                "Call StatsigAI.new_shared() before invoking OpenAI methods to capture Gen AI telemetry."
            )
            return

        duration = round((time.perf_counter() - self.start_time) * 1000)
        event_metadata = {**metadata, "duration": str(duration)}

        try:
            statsig.log_event(
                self._get_event_user(ctx),
                GEN_AI_EVENT_NAME,
                span_name,
                event_metadata,
            )
        except Exception as e:
            logger.warning("[Statsig] Failed to log span event: %s", str(e))

    def _get_event_user(self, ctx: Optional[Context]) -> StatsigUser:
        # Like the Node SDK, every event has the same user ID, and its custom IDs carry
        # the tracer resource values and the activity ID of the active statsig_context.
        custom_ids: Dict[str, str] = {}
        if self.resources.service is not None:
            custom_ids["service"] = self.resources.service
        if self.resources.version is not None:
            custom_ids["version"] = self.resources.version
        if self.resources.environment is not None:
            custom_ids["environment"] = self.resources.environment

        statsig_ctx = get_statsig_context_from_otel_context(ctx)
        if statsig_ctx is not None and statsig_ctx.activity_id:
            custom_ids[STATSIG_ATTR_ACTIVITY_ID] = statsig_ctx.activity_id

        return StatsigUser(user_id=GEN_AI_EVENT_USER_ID, custom_ids=custom_ids)


def get_resources_for_event_user() -> StatsigResourceAttributes:
    # Like the Node SDK, the values initialize_tracing() put on the tracer resource,
    # or without it, the same defaults read from the environment.
    if OtelSingleton.is_initialized():
        return OtelSingleton.get_instance().get_resources()
    return create_resource_attributes()


def attribute_value_to_metadata(value: Any) -> str:
    # Node's attributeValueToMetadata: `true` and `false`, numbers as JavaScript prints
    # them, and arrays and objects as JSON.
    if value is None:
        return "null"
    if isinstance(value, (list, tuple, dict)):
        return safe_stringify(value)
    if isinstance(value, str):
        return value
    return js_string(value)


def safe_stringify(value: Any) -> str:
    try:
        return js_json_stringify(value)
    except Exception:
        return str(value)


def get_statsig_instance_for_logging() -> Optional[Statsig]:
    try:
        from ..statsig_ai import StatsigAI  # pylint: disable=import-outside-toplevel

        if hasattr(StatsigAI, "has_shared_instance") and StatsigAI.has_shared_instance():
            statsig_instance = StatsigAI.shared()
            return statsig_instance.get_statsig()
    except Exception as e:
        logger.warning("[Statsig] Unable to retrieve Statsig instance for span logging: %s", str(e))
    return None
