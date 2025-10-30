from typing import List, TypedDict
from statsig_python_core import Statsig, StatsigUser, ParameterStore
from .prompt_version import PromptVersion


class PromptEvaluationOptions(TypedDict):
    pass


class Prompt:
    def __init__(self, name: str, live: PromptVersion, candicates: List[PromptVersion]):
        self.name = name
        self.live = live
        self.candicates = candicates

    def get_name(self) -> str:
        return self.name

    def get_live(self) -> PromptVersion:
        return self.live

    def get_candicates(self) -> List[PromptVersion]:
        return self.candicates


def make_prompt(
    statsig: Statsig, name: str, paramStore: ParameterStore, user: StatsigUser
) -> Prompt:
    live_config_id = paramStore.get_string("live", "") or ""
    candidates_config_ids = paramStore.get_array("candidates", []) or []
    live_config = statsig.get_dynamic_config(user, live_config_id)
    candidates_configs = []
    for config_id in candidates_config_ids:
        config = statsig.get_dynamic_config(user, config_id)
        candidates_configs.append(config)
    return Prompt(
        name,
        PromptVersion(live_config),
        [PromptVersion(config) for config in candidates_configs],
    )
