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
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_description: Optional[str] = None,
        request_model: Optional[str] = None,
    ):
        self.span = span
        self.span_name = span_name
        self.max_json_chars = max_json_chars
        self.metadata: Dict[str, str] = {}
        self.ended = False

        self.metadata["gen_ai.provider.name"] = provider_name
        if agent_id is not None:
            self.metadata["gen_ai.agent.id"] = agent_id
        if agent_name is not None:
            self.metadata["gen_ai.agent.name"] = agent_name
        if agent_description is not None:
            self.metadata["gen_ai.agent.description"] = agent_description
        if request_model is not None:
            self.metadata["gen_ai.request.model"] = request_model

        ctx = span.get_span_context()
        self.metadata["span.trace_id"] = ctx.trace_id
        self.metadata["span.span_id"] = ctx.span_id

    def set_operation_name(self, operation_name: str) -> None:
        self.metadata["gen_ai.operation.name"] = operation_name
        if self.metadata.get("gen_ai.agent.name"):
            self.span_name = f"{operation_name} {self.metadata['gen_ai.agent.name']}"
        else:
            self.span_name = operation_name

    def set_attributes(self, kv: Dict[str, Any]) -> None:
        for key, value in kv.items():
            if value is None:
                continue
            self.span.set_attribute(key, value)
            self.metadata[key] = attribute_value_to_metadata(value)

    def set_json(self, key: str, value: Any) -> None:
        try:
            json_str = json.dumps(value if value is not None else None)
            if len(json_str) > self.max_json_chars:
                truncated = json_str[: self.max_json_chars] + "…(truncated)"
                self.set_attributes({key: truncated})
                self.set_attributes({f"{key}_truncated": True})
                self.set_attributes({f"{key}_len": len(json_str)})
            else:
                self.set_attributes({key: json_str})
        except Exception:
            self.set_attributes({key: "[[unserializable]]"})

    def set_usage(self, usage: Optional[Dict[str, Any]]) -> None:
        if not usage:
            return
        self.set_attributes(usage_attrs(usage))

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


def usage_attrs(usage: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "gen_ai.usage.input_tokens": usage.get("prompt_tokens"),
        "gen_ai.usage.output_tokens": usage.get("completion_tokens"),
        "gen_ai.usage.total_tokens": usage.get("total_tokens"),
    }


def get_statsig_instance_for_logging() -> Optional[Statsig]:
    try:
        from ..statsig_ai import StatsigAI

        if hasattr(StatsigAI, "has_shared") and StatsigAI.has_shared():
            statsig_instance = StatsigAI.shared()
            return statsig_instance.get_statsig()
    except Exception as e:
        logger.warning("[Statsig] Unable to retrieve Statsig instance for span logging: %s", str(e))
    return None
