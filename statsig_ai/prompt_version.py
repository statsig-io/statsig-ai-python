import copy
from typing import Any, Dict, List, TypedDict

from statsig_python_core import DynamicConfig

from ._vendor import combustache
from .otel.prompt_context import record_prompt_template_variables


PromptParams = Dict[str, Any]


class PromptMessage(TypedDict):
    role: str
    content: str


class PromptVersion:
    def __init__(self, prompt_variant: DynamicConfig, is_live_for_user: bool = False):
        self.name: str = prompt_variant.get_string("name", "")
        self._type: str = prompt_variant.get_string("type", "")
        self._ai_config_name: str = prompt_variant.get_string("aiConfigName", "")

        parts = prompt_variant.name.split(":")
        self._id: str = parts[1] if len(parts) > 1 else ""

        self._prompt_variant = prompt_variant
        self.is_live_for_user = is_live_for_user

    def get_name(self) -> str:
        return self.name

    def get_id(self) -> str:
        return self._id

    def get_type(self) -> str:
        return self._type

    def get_prompt_name(self) -> str:
        return self._ai_config_name

    def get_temperature(self, fallback: float = 0) -> float:
        return self._prompt_variant.get_float("temperature", fallback)

    def get_max_tokens(self, fallback: int = 0) -> int:
        return self._prompt_variant.get_integer("maxTokens", fallback)

    def get_top_p(self, fallback: float = 0) -> float:
        return self._prompt_variant.get_float("topP", fallback)

    def get_frequency_penalty(self, fallback: float = 0) -> float:
        return self._prompt_variant.get_float("frequencyPenalty", fallback)

    def get_presence_penalty(self, fallback: float = 0) -> float:
        return self._prompt_variant.get_float("presencePenalty", fallback)

    def get_provider(self, fallback: str = "") -> str:
        return self._prompt_variant.get_string("provider", fallback)

    def get_model(self, fallback: str = "") -> str:
        return self._prompt_variant.get_string("model", fallback)

    # pylint: disable=dangerous-default-value
    def get_workflow_body(self, fallback: Dict[str, Any] = {}) -> Dict[str, Any]:
        return self.get_object("workflowBody", fallback)

    def get_eval_model(self, fallback: str = "") -> str:
        return self._prompt_variant.get_string("evalModel", fallback)

    def get_string(self, key: str, fallback: str) -> str:
        return self._prompt_variant.get_string(key, fallback)

    def get_integer(self, key: str, fallback: int) -> int:
        return self._prompt_variant.get_integer(key, fallback)

    def get_float(self, key: str, fallback: float) -> float:
        return self._prompt_variant.get_float(key, fallback)

    def get_boolean(self, key: str, fallback: bool) -> bool:
        return self._prompt_variant.get_bool(key, fallback)

    def get_array(self, key: str, fallback: list) -> list:
        # Core returns its own stored value (or the fallback itself), so copy it
        # to keep callers from mutating the config or a shared default.
        return copy.deepcopy(self._prompt_variant.get_array(key, fallback))

    def get_object(self, key: str, fallback: dict) -> dict:
        return copy.deepcopy(self._prompt_variant.get_object(key, fallback))

    def get_prompt_messages(self, params: PromptParams) -> List[PromptMessage]:
        prompts = self._prompt_variant.get_array(
            "prompts", [{"role": "system", "content": ""}]
        )
        record_prompt_template_variables(self._id, params)

        return [
            PromptMessage(role=p["role"], content=_render_mustache(p["content"], params))
            for p in prompts
        ]


def _render_mustache(template: str, params: PromptParams) -> str:
    """
    Renders a prompt template like the Node SDK's mustache.js call: values are
    inserted as-is, without HTML escaping, and missing ones render as "".
    """
    return combustache.render(
        template, params, stringify=_to_js_string, escape=lambda text: text
    )


def _to_js_string(value: Any) -> str:
    """Converts a value to text the way mustache.js does, with JavaScript's String()."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    # JavaScript has one number type, so 1.0 prints as "1". From 1e21 it uses exponents.
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e21:
        return str(int(value))
    if isinstance(value, (list, tuple)):
        return ",".join(_to_js_string(item) for item in value)
    return str(value)
