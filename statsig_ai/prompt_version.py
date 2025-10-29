import re
from typing import Any, Dict, List, Optional, TypedDict
from statsig_python_core import DynamicConfig


PromptParams = Dict[str, Any]


class PromptMessage(TypedDict):
    role: str
    content: str


class PromptVersion:
    def __init__(self, prompt_variant: DynamicConfig):
        self.name: str = prompt_variant.get_value("name", "")
        self._type: str = prompt_variant.get_value("type", "")
        self._ai_config_name: str = prompt_variant.get_value("aiConfigName", "")

        parts = prompt_variant.name.split(":")
        self._id: str = parts[1] if len(parts) > 1 else ""

        self._prompt_variant = prompt_variant
        self._temperature: Optional[float] = prompt_variant.get_value(
            "temperature", None
        )
        self._max_tokens: Optional[int] = prompt_variant.get_value("maxTokens", None)
        self._top_p: Optional[float] = prompt_variant.get_value("topP", None)
        self._frequency_penalty: Optional[float] = prompt_variant.get_value(
            "frequencyPenalty", None
        )
        self._presence_penalty: Optional[float] = prompt_variant.get_value(
            "presencePenalty", None
        )
        self._provider: Optional[str] = prompt_variant.get_value("provider", None)
        self._model: Optional[str] = prompt_variant.get_value("model", None)
        self._workflow_body: Optional[Dict[str, Any]] = prompt_variant.get_value(
            "workflowBody", None
        )
        self._eval_model: Optional[str] = prompt_variant.get_value("evalModel", None)

    def get_name(self) -> str:
        return self.name

    def get_id(self) -> str:
        return self._id

    def get_type(self) -> str:
        return self._type

    def get_prompt_name(self) -> str:
        return self._ai_config_name

    def get_temperature(self, fallback: Optional[float] = 0) -> float:
        return self._temperature if self._temperature is not None else fallback

    def get_max_tokens(self, fallback: Optional[int] = 0) -> int:
        return self._max_tokens if self._max_tokens is not None else fallback

    def get_top_p(self, fallback: Optional[float] = 0) -> float:
        return self._top_p if self._top_p is not None else fallback

    def get_frequency_penalty(self, fallback: Optional[float] = 0) -> float:
        return (
            self._frequency_penalty if self._frequency_penalty is not None else fallback
        )

    def get_presence_penalty(self, fallback: Optional[float] = 0) -> float:
        return (
            self._presence_penalty if self._presence_penalty is not None else fallback
        )

    def get_provider(self, fallback: str = "") -> str:
        return self._provider if self._provider is not None else fallback

    def get_model(self, fallback: str = "") -> str:
        return self._model if self._model is not None else fallback

    def get_workflow_body(
        self, fallback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return (
            self._workflow_body if self._workflow_body is not None else (fallback or {})
        )

    def get_eval_model(self, fallback: str = "") -> str:
        return self._eval_model if self._eval_model is not None else fallback

    def get_value(self, key: str, fallback: Any = None) -> Any:
        return self._prompt_variant.get_value(key, fallback)

    def get_prompt_messages(self, params: PromptParams) -> List[PromptMessage]:
        prompts = self._prompt_variant.get_value(
            "prompts", [{"role": "system", "content": ""}]
        )
        regex = re.compile(r"{{\s*([^}]+)\s*}}")

        def replace_placeholders(content: str) -> str:
            def replacer(match):
                path = match.group(1).strip()
                value = resolve_path(params, path)
                return str(value) if value is not None else f"{{{{{path}}}}}"

            return regex.sub(replacer, content)

        return [
            PromptMessage(p["role"], replace_placeholders(p["content"]))
            for p in prompts
        ]


def resolve_path(obj: Any, path: str) -> Any:
    """Safely resolves nested object paths, supporting array indices like `a.b[0].c`."""
    parts = path.replace("[", ".").replace("]", "").split(".")
    current = obj
    for key in parts:
        if current is None or key == "":
            return None
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list):
            try:
                idx = int(key)
                current = current[idx]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current
