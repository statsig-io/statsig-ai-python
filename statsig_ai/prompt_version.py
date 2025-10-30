import json
import re
from typing import Any, Dict, List, TypedDict
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

    def get_workflow_body(self, fallback: Dict[str, Any]) -> Dict[str, Any]:
        return self._prompt_variant.get_object_json("workflowBody", json.dumps(fallback))

    def get_eval_model(self, fallback: str = "") -> str:
        return self._prompt_variant.get_string("evalModel", fallback)

    def get_string(self, key: str, fallback: str = "") -> str:
        return self._prompt_variant.get_string(key, fallback)

    def get_integer(self, key: str, fallback: int = 0) -> int:
        return self._prompt_variant.get_integer(key, fallback)

    def get_float(self, key: str, fallback: float = 0) -> float:
        return self._prompt_variant.get_float(key, fallback)

    def get_boolean(self, key: str, fallback: bool = False) -> bool:
        return self._prompt_variant.get_bool(key, fallback)

    def get_array(self, key: str, fallback: list) -> list:
        array_json = self._prompt_variant.get_array_json(key, json.dumps(fallback))
        return json.loads(array_json)

    def get_object(self, key: str, fallback: dict) -> dict:
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
