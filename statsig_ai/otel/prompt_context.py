from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from opentelemetry import context as otel_context
from opentelemetry.context import Context

from .conventions import (
    STATSIG_ATTR_LLM_PROMPT_NAME,
    STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES,
    STATSIG_ATTR_LLM_PROMPT_VERSION,
    STATSIG_ATTR_LLM_PROMPT_VERSION_ID,
    STATSIG_CTX_KEY_ACTIVE_PROMPT,
    STATSIG_CTX_KEY_ACTIVE_PROMPT_TEMPLATE,
    STATSIG_CTX_KEY_ACTIVE_PROMPT_VERSION,
    STATSIG_CTX_KEY_ACTIVE_PROMPT_VERSION_ID,
)


# The template variables aren't known when a version scope starts; they arrive
# later when get_prompt_messages() renders the prompt inside it. Contexts are
# immutable, so the scope holds this object and rendering fills it in.
@dataclass
class ActivePromptTemplate:
    version_id: str
    variables: Optional[Dict[str, Any]] = None


def set_prompt_version_to_context(
    ctx: Context,
    prompt_name: str,
    version_name: str,
    version_id: str,
) -> Context:
    template = ActivePromptTemplate(version_id=version_id)
    ctx = otel_context.set_value(STATSIG_CTX_KEY_ACTIVE_PROMPT, prompt_name, ctx)
    ctx = otel_context.set_value(STATSIG_CTX_KEY_ACTIVE_PROMPT_VERSION, version_name, ctx)
    ctx = otel_context.set_value(STATSIG_CTX_KEY_ACTIVE_PROMPT_VERSION_ID, version_id, ctx)
    return otel_context.set_value(STATSIG_CTX_KEY_ACTIVE_PROMPT_TEMPLATE, template, ctx)


def record_prompt_template_variables(
    version_id: str,
    variables: Optional[Mapping[str, Any]],
) -> None:
    """
    Called when a version renders its prompt. Only records the variables if that
    version is the one the active scope is for, so rendering a different version
    inside the scope doesn't mislabel its spans.
    """
    template = _get_active_prompt_template(otel_context.get_current())
    if template is not None and template.version_id == version_id:
        template.variables = dict(variables or {})


def get_prompt_span_attrs_from_context(
    ctx: Optional[Context] = None,
    *,
    include_template_variables: bool,
) -> Dict[str, Any]:
    """
    The template variables are returned as a dict, like the Node SDK's jsonAttribute.
    SpanTelemetry serializes them to JSON and caps them at max_json_chars, so they're
    left out of the attributes a span starts with.
    """
    attrs: Dict[str, Any] = {}
    context_values = [
        (STATSIG_CTX_KEY_ACTIVE_PROMPT, STATSIG_ATTR_LLM_PROMPT_NAME),
        (STATSIG_CTX_KEY_ACTIVE_PROMPT_VERSION, STATSIG_ATTR_LLM_PROMPT_VERSION),
        (STATSIG_CTX_KEY_ACTIVE_PROMPT_VERSION_ID, STATSIG_ATTR_LLM_PROMPT_VERSION_ID),
    ]
    for ctx_key, attr in context_values:
        value = otel_context.get_value(ctx_key, ctx)
        if isinstance(value, str) and value:
            attrs[attr] = value

    template = _get_active_prompt_template(ctx)
    if include_template_variables and template is not None and template.variables is not None:
        attrs[STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES] = dict(template.variables)

    return attrs


def _get_active_prompt_template(ctx: Optional[Context]) -> Optional[ActivePromptTemplate]:
    template = otel_context.get_value(STATSIG_CTX_KEY_ACTIVE_PROMPT_TEMPLATE, ctx)
    if not isinstance(template, ActivePromptTemplate):
        return None
    return template
