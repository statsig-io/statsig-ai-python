from typing import List

from statsig_python_core import DynamicConfig, Statsig, StatsigUser

from .agent_version import AgentVersion


class AgentConfig:
    def __init__(self, name: str, live: AgentVersion, candidates: List[AgentVersion]):
        self.name = name
        self._live = live
        self._candidates = candidates

    def get_name(self) -> str:
        return self.name

    def get_live(self) -> AgentVersion:
        return self._live

    def get_candidates(self) -> List[AgentVersion]:
        return self._candidates


def make_agent_config(statsig: Statsig, name: str, param_store, user: StatsigUser) -> AgentConfig:
    live_config_id = param_store.get_string("live", "") or ""
    candidates_config_ids = param_store.get_array("candidates", []) or []

    def get_root_config(version_config_id: str) -> DynamicConfig:
        version_config = statsig.get_dynamic_config(user, version_config_id)
        return statsig.get_dynamic_config(user, version_config.get_string("root", ""))

    return AgentConfig(
        name,
        AgentVersion(get_root_config(live_config_id), True),
        [AgentVersion(get_root_config(config_id), False) for config_id in candidates_config_ids],
    )
