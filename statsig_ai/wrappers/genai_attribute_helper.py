from typing import Any, Dict, Mapping
from .configs import GenAICaptureOptions
from .openai_messages import to_openai_output_messages, to_openai_request_messages


def extract_oai_usage_attributes(usage: Mapping[str, Any]) -> Dict[str, Any]:
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


def extract_base_attributes(
    operation_name: str,
    otel_operation_name: str,
    kwargs: Mapping[str, Any],
    options: GenAICaptureOptions,
) -> Dict[str, Any]:
    """Request attributes, set when the span starts.

    Like the Node SDK, they include the captured input messages, system instructions and tool
    definitions, so failed calls and streams record them too.
    """
    attrs: Dict[str, Any] = {
        "gen_ai.provider.name": "openai",
        "gen_ai.operation.name": otel_operation_name,
        "gen_ai.operation.source_name": operation_name,
        "gen_ai.request.model": kwargs.get("model"),
    }

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
    response_format = kwargs.get("response_format")
    if isinstance(response_format, Mapping):
        attrs["gen_ai.output.type"] = response_format.get("type")

    request_service_tier = kwargs.get("service_tier", "auto")
    if request_service_tier != "auto":
        attrs["openai.request.service_tier"] = request_service_tier

    capture_all = options.capture_all
    capture_input = capture_all or options.capture_input_messages
    capture_system = capture_all or options.capture_system_instructions
    if capture_input or capture_system:
        # Messages are written in the OpenTelemetry GenAI message format, like the Node SDK.
        input_messages, system_instructions = to_openai_request_messages(kwargs)
        if capture_input:
            attrs["gen_ai.input.messages"] = input_messages
        if capture_system and system_instructions:
            attrs["gen_ai.system_instructions"] = system_instructions
    if capture_all or options.capture_tool_definitions:
        attrs["gen_ai.tool.definitions"] = kwargs.get("tools")

    attrs.update(extract_embeddings_attributes(kwargs))
    attrs.update(extract_images_attributes(kwargs))

    return {k: v for k, v in attrs.items() if v is not None}


def extract_response_attributes(
    response: Mapping[str, Any], options: GenAICaptureOptions
) -> Dict[str, Any]:
    """Attributes of a response, or of the response a stream amounts to."""
    attrs: Dict[str, Any] = {
        "gen_ai.response.id": response.get("id"),
        "gen_ai.response.model": response.get("model"),
        "gen_ai.conversation.id": response.get("conversation_id"),
        "openai.response.service_tier": response.get("service_tier"),
        "openai.response.system_fingerprint": response.get("system_fingerprint"),
    }

    outputs = response.get("choices")
    if outputs is None:
        outputs = response.get("output")
    if isinstance(outputs, (list, tuple)) and outputs:
        attrs["gen_ai.response.finish_reasons"] = [
            output.get("finish_reason") if isinstance(output, Mapping) else None
            for output in outputs
        ]

    if options.capture_all or options.capture_output_messages:
        attrs["gen_ai.output.messages"] = to_openai_output_messages(response)

    usage = response.get("usage")
    if isinstance(usage, Mapping):
        attrs.update(extract_oai_usage_attributes(usage))

    return {k: v for k, v in attrs.items() if v is not None}


def extract_embeddings_attributes(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    encoding_format = kwargs.get("encoding_format")
    return {
        "gen_ai.embeddings.dimension.count": kwargs.get("dimensions"),
        # A list, like the Node SDK and the semantic conventions.
        "gen_ai.request.encoding_formats": (
            [encoding_format] if encoding_format is not None else None
        ),
    }


def extract_images_attributes(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "statsig.gen_ai.images.output_compression": kwargs.get("output_compression"),
        "statsig.gen_ai.images.output_format": kwargs.get("output_format"),
        "statsig.gen_ai.images.quality": kwargs.get("quality"),
        "statsig.gen_ai.images.size": kwargs.get("size"),
    }
