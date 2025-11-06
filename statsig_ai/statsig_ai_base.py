from typing import Any, Optional, Union

from statsig_python_core import Statsig, StatsigOptions
from .prompt import make_prompt, PromptEvaluationOptions, Prompt
from .ai_eval_grade_data import AIEvalGradeData
from .prompt_version import PromptVersion


class StatsigCreateConfig:
    def __init__(self, sdk_key: str, statsig_options: Optional[StatsigOptions] = None):
        self.sdk_key = sdk_key
        self.statsig_options = statsig_options


class StatsigAttachConfig:
    def __init__(self, statsig: Statsig):
        self.statsig = statsig


StatsigSourceConfig = Union[StatsigCreateConfig, StatsigAttachConfig]


class StatsigAIInstance:
    _statsig: Statsig
    _owns_statsig_instance: bool

    def __init__(
        self,
        statsig_source: StatsigSourceConfig,
    ):
        if isinstance(statsig_source, StatsigAttachConfig):
            self._statsig = statsig_source.statsig
            self._owns_statsig_instance = False
        else:
            self._statsig = Statsig(statsig_source.sdk_key, statsig_source.statsig_options)
            self._owns_statsig_instance = True

    def initialize(self) -> None:
        if self._owns_statsig_instance:
            self._statsig.initialize().wait()

    def flush_events(self) -> None:
        if self._owns_statsig_instance:
            self._statsig.flush_events().wait()

    def shutdown(self) -> None:
        if self._owns_statsig_instance:
            self._statsig.shutdown().wait()

    def get_statsig(self) -> Statsig:
        return self._statsig

    def get_prompt(
        self,
        user: Any,
        prompt_name: str,
        _options: Optional[PromptEvaluationOptions] = None,
    ) -> Prompt:
        MAX_DEPTH = 300
        depth = 0

        base_param_store_name = f"prompt:{prompt_name}"
        current_param_store_name = base_param_store_name
        next_param_store_name = (
            self._statsig.get_parameter_store(user, current_param_store_name).get_string(
                "prompt_targeting_rules", ""
            )
            or ""
        )

        while (
            next_param_store_name != ""
            and next_param_store_name != current_param_store_name
            and depth < MAX_DEPTH
        ):
            next_param_store = self._statsig.get_parameter_store(user, next_param_store_name)
            possible_next_param_store_name = (
                next_param_store.get_string("prompt_targeting_rules", "") or ""
            )
            if possible_next_param_store_name in [next_param_store_name, ""]:
                current_param_store_name = next_param_store_name
                break
            current_param_store_name = next_param_store_name
            next_param_store_name = possible_next_param_store_name
            depth += 1

        if depth >= MAX_DEPTH:
            current_param_store_name = base_param_store_name
            print(
                f"[Statsig] Max targeting depth ({MAX_DEPTH}) reached while resolving prompt: {prompt_name}. "
                + f'Possible circular reference starting from "{base_param_store_name}".'
            )
        final_param_store = self._statsig.get_parameter_store(user, current_param_store_name)
        prompt_name = current_param_store_name.split(":")[1]

        return make_prompt(self._statsig, prompt_name, final_param_store, user)

    def log_eval_grade(
        self,
        user: Any,
        prompt_version: PromptVersion,
        score: float,
        grader_name: str,
        eval_data: AIEvalGradeData,
    ) -> None:
        if score is None or score < 0 or score > 1:
            print(
                f"[Statsig] AI eval result score is out of bounds: {score} is not between 0 and 1, skipping log event"
            )
            return

        self._statsig.log_event(
            user,
            "statsig::eval_result",
            prompt_version.get_prompt_name(),
            {
                "score": str(score),
                "session_id": eval_data.get("session_id", ""),
                "version_name": prompt_version.get_name(),
                "version_id": prompt_version.get_id(),
                "grader_id": grader_name,
                "ai_config_name": prompt_version.get_prompt_name(),
            },
        )
