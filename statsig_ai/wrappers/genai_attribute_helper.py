from typing import Any, Dict, Optional
from .configs import GenAICaptureOptions


def extract_genai_attributes(
    provider_name: str,
    operation_name: str,
    model: Any,
    kwargs: Dict[str, Any],
    response: Optional[Any] = None,
    otel_operation_name: Optional[str] = None,
) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {
        "gen_ai.provider.name": provider_name,
        "gen_ai.operation.name": otel_operation_name if otel_operation_name else operation_name,
        "gen_ai.operation.source_name": operation_name,
        "gen_ai.request.model": model,
    }

    # ---------- Request attributes ----------
    attrs["gen_ai.request.max_tokens"] = kwargs.get("max_tokens") or kwargs.get(
        "max_completion_tokens"
    )
    attrs["gen_ai.request.temperature"] = kwargs.get("temperature")
    attrs["gen_ai.request.top_p"] = kwargs.get("top_p")
    attrs["gen_ai.request.top_k"] = kwargs.get("top_k")
    attrs["gen_ai.request.frequency_penalty"] = kwargs.get("frequency_penalty")
    attrs["gen_ai.request.presence_penalty"] = kwargs.get("presence_penalty")
    attrs["gen_ai.request.stop_sequences"] = kwargs.get("stop") or kwargs.get("stop_sequences")
    n = kwargs.get("n", 0)
    attrs["gen_ai.request.choice.count"] = n if n and n != 1 else None
    attrs["gen_ai.request.seed"] = kwargs.get("seed")
    attrs["gen_ai.conversation.id"] = kwargs.get("conversation_id")
    response_format = kwargs.get("response_format", {})
    attrs["gen_ai.output.type"] = response_format.get("type")

    # ---------- Response attributes ----------
    response = response or {}
    attrs["gen_ai.response.id"] = response.get("id", None)
    attrs["gen_ai.response.model"] = response.get("model", None)
    choices = response.get("choices", [])
    finish_reasons = [choice.get("finish_reason", "") for choice in choices]
    attrs["gen_ai.response.finish_reasons"] = finish_reasons

    # ---------- Usage attributes ----------
    usage = response.get("usage", {})
    attrs["gen_ai.usage.input_tokens"] = usage.get("prompt_tokens")
    attrs["gen_ai.usage.output_tokens"] = usage.get("completion_tokens")

    # ---------- Embeddings attributes ----------
    attrs.update(extract_embeddings_attributes(kwargs))

    # ---------- Provider-specific attributes ----------
    if provider_name == "openai":
        attrs.update(extract_openai_attributes(kwargs, response))

    # Missing attributes:
    # - service.address
    # - service.port

    return {k: v for k, v in attrs.items() if v is not None}


def extract_embeddings_attributes(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    attrs["gen_ai.embeddings.dimension.count"] = kwargs.get("dimensions")
    attrs["gen_ai.request.encoding_formats"] = kwargs.get("encoding_format")
    return attrs


def extract_opt_in_attributes(
    gen_ai_capture_options: GenAICaptureOptions,
    kwargs: Dict[str, Any],
    response: Any,
) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    if gen_ai_capture_options.capture_input_messages or gen_ai_capture_options.capture_all:
        attrs["gen_ai.input.messages"] = kwargs.get("messages")
    if gen_ai_capture_options.capture_output_messages or gen_ai_capture_options.capture_all:
        attrs["gen_ai.output.messages"] = response.get("choices", [])
    if gen_ai_capture_options.capture_system_instructions or gen_ai_capture_options.capture_all:
        msgs = kwargs.get("messages") or []
        system_instructions = [msg for msg in msgs if msg.get("role") == "system"]
        attrs["gen_ai.system.instructions"] = system_instructions
    if gen_ai_capture_options.capture_tool_definitions or gen_ai_capture_options.capture_all:
        attrs["gen_ai.tool.definitions"] = kwargs.get("tools")
    return attrs


def extract_openai_attributes(kwargs: Dict[str, Any], response: Any) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    request_service_tier = kwargs.get("service_tier", "auto")
    attrs["openai.request.service_tier"] = (
        request_service_tier if request_service_tier != "auto" else None
    )
    attrs["openai.response.service_tier"] = response.get("service_tier")
    attrs["openai.response.system_fingerprint"] = response.get("system_fingerprint")
    return attrs
