import json
import re
from typing import Any, Dict, List, Optional, TypedDict
from statsig_python_core import DynamicConfig


PromptParams = Dict[str, Any]


class PromptMessage(TypedDict):
    role: str
    content: str


class PromptVersion:
    def __init__(self, prompt_variant: DynamicConfig):
        self.name: str = prompt_variant.get_string("name", "")
        self._type: str = prompt_variant.get_string("type", "")
        self._ai_config_name: str = prompt_variant.get_string("aiConfigName", "")

        parts = prompt_variant.name.split(":")
        self._id: str = parts[1] if len(parts) > 1 else ""

        self._prompt_variant = prompt_variant
        raw_value: Dict[str, Any] = (
            prompt_variant.value if isinstance(prompt_variant.value, dict) else {}
        )

        # Only capture values when explicitly present in the config; otherwise store None
        self._temperature: Optional[float] = (
            prompt_variant.get_float("temperature", 0) if "temperature" in raw_value else None
        )
        self._max_tokens: Optional[int] = (
            prompt_variant.get_integer("maxTokens", 0) if "maxTokens" in raw_value else None
        )
        self._top_p: Optional[float] = (
            prompt_variant.get_float("topP", 0) if "topP" in raw_value else None
        )
        self._frequency_penalty: Optional[float] = (
            prompt_variant.get_float("frequencyPenalty", 0)
            if "frequencyPenalty" in raw_value
            else None
        )
        self._presence_penalty: Optional[float] = (
            prompt_variant.get_float("presencePenalty", 0)
            if "presencePenalty" in raw_value
            else None
        )
        self._provider: Optional[str] = (
            prompt_variant.get_string("provider", "") if "provider" in raw_value else None
        )
        self._model: Optional[str] = (
            prompt_variant.get_string("model", "") if "model" in raw_value else None
        )
        if "workflowBody" in raw_value:
            workflow_body_json: str = prompt_variant.get_object_json("workflowBody", "{}")
            self._workflow_body: Optional[Dict[str, Any]] = json.loads(workflow_body_json)
        else:
            self._workflow_body = None
        self._eval_model: Optional[str] = (
            prompt_variant.get_string("evalModel", "") if "evalModel" in raw_value else None
        )

    def get_name(self) -> str:
        return self.name

    def get_id(self) -> str:
        return self._id

    def get_type(self) -> str:
        return self._type

    def get_prompt_name(self) -> str:
        return self._ai_config_name

    def get_temperature(self, fallback: float = 0) -> float:
        return self._temperature if self._temperature is not None else fallback

    def get_max_tokens(self, fallback: int = 0) -> int:
        return self._max_tokens if self._max_tokens is not None else fallback

    def get_top_p(self, fallback: float = 0) -> float:
        return self._top_p if self._top_p is not None else fallback

    def get_frequency_penalty(self, fallback: float = 0) -> float:
        return self._frequency_penalty if self._frequency_penalty is not None else fallback

    def get_presence_penalty(self, fallback: float = 0) -> float:
        return self._presence_penalty if self._presence_penalty is not None else fallback

    def get_provider(self, fallback: str = "") -> str:
        return self._provider if self._provider is not None else fallback

    def get_model(self, fallback: str = "") -> str:
        return self._model if self._model is not None else fallback

    def get_workflow_body(self, fallback: Dict[str, Any] = {}) -> Dict[str, Any]:
        return self._workflow_body if self._workflow_body is not None else (fallback or {})

    def get_eval_model(self, fallback: str = "") -> str:
        return self._eval_model if self._eval_model is not None else fallback

    def get_string(self, key: str, fallback: str = "") -> str:
        return self._prompt_variant.get_string(key, fallback)

    def get_integer(self, key: str, fallback: int = 0) -> int:
        return self._prompt_variant.get_integer(key, fallback)

    def get_float(self, key: str, fallback: float = 0) -> float:
        return self._prompt_variant.get_float(key, fallback)

    def get_boolean(self, key: str, fallback: bool = False) -> bool:
        return self._prompt_variant.get_bool(key, fallback)

    def get_array(self, key: str, fallback: list = []) -> list:
        array_json = self._prompt_variant.get_array_json(key, json.dumps(fallback))
        return json.loads(array_json)

    def get_object(self, key: str, fallback: dict = {}) -> dict:
        object_json = self._prompt_variant.get_object_json(key, json.dumps(fallback))
        return json.loads(object_json)

    def get_prompt_messages(self, params: PromptParams) -> List[PromptMessage]:
        prompts_array_json = self._prompt_variant.get_array_json(
            "prompts", json.dumps([{"role": "system", "content": ""}])
        )
        prompts = json.loads(prompts_array_json)
        regex = re.compile(r"{{\s*([^}]+)\s*}}")

        def replace_placeholders(content: str) -> str:
            def replacer(match):
                path = match.group(1).strip()
                value = resolve_path(params, path)
                return str(value) if value is not None else f"{{{{{path}}}}}"

            return regex.sub(replacer, content)

        return [
            PromptMessage(role=p["role"], content=replace_placeholders(p["content"]))
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
