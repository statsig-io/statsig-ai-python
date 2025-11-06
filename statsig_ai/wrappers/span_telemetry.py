from typing import Any, Dict, Optional
import json
import logging
from opentelemetry.trace import Span, Status, StatusCode
from statsig_python_core import Statsig, StatsigUser

GEN_AI_EVENT_NAME = "statsig::gen_ai"
PLACEHOLDER_STATSIG_USER = StatsigUser(user_id="statsig-ai-openai-wrapper")

logger = logging.getLogger(__name__)


class SpanTelemetry:
    def __init__(
        self,
        span: Span,
        span_name: str,
        max_json_chars: int,
        provider_name: str,
    ):
        self.span = span
        self.span_name = span_name
        self.max_json_chars = max_json_chars
        self.metadata: Dict[str, str] = {}
        self.ended = False

        self.metadata["gen_ai.provider.name"] = provider_name
        ctx = span.get_span_context()
        self.metadata["span.trace_id"] = str(ctx.trace_id)
        self.metadata["span.span_id"] = str(ctx.span_id)

    def set_operation_name(self, operation_name: str) -> None:
        self.metadata["gen_ai.operation.name"] = operation_name

    def set_attributes(self, kv: Dict[str, Any]) -> None:
        for key, value in kv.items():
            if value is None:
                continue
            if isinstance(value, (list, dict)):
                self._set_json_safe(key, value)
                continue

            self.span.set_attribute(key, value)
            self.metadata[key] = attribute_value_to_metadata(value)

    def _set_json_safe(self, key: str, value: Any) -> None:
        try:
            json_str = json.dumps(value if value is not None else None)
            if len(json_str) > self.max_json_chars:
                truncated = json_str[: self.max_json_chars] + "…(truncated)"
                # Directly set attributes instead of calling set_attributes again
                self.span.set_attribute(key, truncated)
                self.span.set_attribute(f"{key}_truncated", True)
                self.span.set_attribute(f"{key}_len", len(json_str))

                self.metadata[key] = truncated
                self.metadata[f"{key}_truncated"] = "True"
                self.metadata[f"{key}_len"] = str(len(json_str))
            else:
                self.span.set_attribute(key, json_str)
                self.metadata[key] = json_str
        except Exception:
            self.span.set_attribute(key, "[[unserializable]]")
            self.metadata[key] = "[[unserializable]]"

    def set_status(self, status: Dict[str, Any]) -> None:
        code: StatusCode = status.get("code", StatusCode.UNSET)
        message: Optional[str] = status.get("message")

        self.span.set_status(Status(code, message))
        if code == StatusCode.ERROR:
            self.metadata["error.type"] = type(message).__name__ if message else "Error"
        self.metadata["span.status_code"] = code.name
        self.metadata["span.status_code_value"] = str(code.value)
        if message:
            self.metadata["span.status_message"] = str(message)

    def record_exception(self, error: Any) -> None:
        self.span.record_exception(error)
        type_name = getattr(error, "__class__", None)
        if type_name:
            self.metadata["exception.type"] = type_name.__name__
        if hasattr(error, "message"):
            self.metadata["exception.message"] = str(error.message)
        elif isinstance(error, str):
            self.metadata["exception.message"] = error

    def fail(self, error: Any) -> None:
        self.record_exception(error)
        msg = getattr(error, "message", str(error))
        self.set_status({"code": StatusCode.ERROR, "message": msg})

    def end(self) -> None:
        if self.ended:
            return
        self.ended = True
        self.span.end()
        self._log_span_event(self.span_name, dict(self.metadata))

    def _log_span_event(self, span_name: str, metadata: Dict[str, str]) -> None:
        statsig = get_statsig_instance_for_logging()
        if not statsig:
            logger.warning(
                "[Statsig] No shared global StatsigAI instance found. "
                "Call StatsigAI.new_shared() before invoking OpenAI methods to capture Gen AI telemetry."
            )
            return

        try:
            statsig.log_event(
                PLACEHOLDER_STATSIG_USER,
                GEN_AI_EVENT_NAME,
                span_name,
                metadata,
            )
        except Exception as e:
            logger.warning("[Statsig] Failed to log span event: %s", str(e))


def attribute_value_to_metadata(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        return safe_stringify(value)
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return str(value)


def safe_stringify(value: Any) -> str:
    try:
        return json.dumps(value)
    except Exception:
        return str(value)


def get_statsig_instance_for_logging() -> Optional[Statsig]:
    try:
        from ..statsig_ai import StatsigAI

        if hasattr(StatsigAI, "has_shared") and StatsigAI.has_shared():
            statsig_instance = StatsigAI.shared()
            return statsig_instance.get_statsig()
    except Exception as e:
        logger.warning("[Statsig] Unable to retrieve Statsig instance for span logging: %s", str(e))
    return None
