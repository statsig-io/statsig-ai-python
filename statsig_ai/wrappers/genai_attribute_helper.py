from typing import Any, Dict, Optional
from .configs import GenAICaptureOptions


def extract_oai_usage_attributes(usage: Dict[str, Any]) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}

    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
    if isinstance(input_tokens, int):
        attrs["gen_ai.usage.input_tokens"] = input_tokens

    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
    if isinstance(output_tokens, int):
        attrs["gen_ai.usage.output_tokens"] = output_tokens

    if isinstance(input_tokens, int) or isinstance(output_tokens, int):
        attrs["statsig.gen_ai.usage.total_tokens"] = (input_tokens or 0) + (output_tokens or 0)

    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details")
    if input_details:
        if isinstance(input_details.get("cached_tokens"), int):
            attrs["statsig.gen_ai.usage.input_cached_tokens"] = input_details["cached_tokens"]
        if isinstance(input_details.get("audio_tokens"), int):
            attrs["statsig.gen_ai.usage.input_audio_tokens"] = input_details["audio_tokens"]

    output_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details")
    if output_details:
        if isinstance(output_details.get("reasoning_tokens"), int):
            attrs["statsig.gen_ai.usage.output_reasoning_tokens"] = output_details[
                "reasoning_tokens"
            ]

    return {k: v for k, v in attrs.items() if v != 0}


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
    response_format = kwargs.get("response_format", {})
    attrs["gen_ai.output.type"] = response_format.get("type")

    # ---------- Response attributes ----------
    response = response or {}
    attrs["gen_ai.response.id"] = response.get("id", None)
    attrs["gen_ai.response.model"] = response.get("model", None)
    attrs["gen_ai.conversation.id"] = response.get("conversation_id", None)
    output_messages = response.get("choices") or response.get("output") or []
    finish_reasons = [msg.get("finish_reason", None) for msg in output_messages]
    attrs["gen_ai.response.finish_reasons"] = finish_reasons

    # ---------- Usage attributes ----------
    usage = response.get("usage", {})
    attrs.update(extract_oai_usage_attributes(usage))

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


def extract_images_attributes(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    attrs["statsig.gen_ai.images.output_compression"] = kwargs.get("output_compression")
    attrs["statsig.gen_ai.images.output_format"] = kwargs.get("output_format")
    attrs["statsig.gen_ai.images.quality"] = kwargs.get("quality")
    attrs["statsig.gen_ai.images.size"] = kwargs.get("size")
    return attrs


def extract_opt_in_attributes(
    gen_ai_capture_options: GenAICaptureOptions,
    kwargs: Dict[str, Any],
    response: Any,
) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    if gen_ai_capture_options.capture_input_messages or gen_ai_capture_options.capture_all:
        attrs["gen_ai.input.messages"] = kwargs.get("messages") or kwargs.get("input") or []
    if gen_ai_capture_options.capture_output_messages or gen_ai_capture_options.capture_all:
        attrs["gen_ai.output.messages"] = response.get("choices") or response.get("output") or []
    if gen_ai_capture_options.capture_system_instructions or gen_ai_capture_options.capture_all:
        msgs = kwargs.get("messages") or kwargs.get("input") or []
        system_instructions = [msg for msg in msgs if msg.get("role") == "system"]
        attrs["gen_ai.system_instructions"] = system_instructions
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
