from statsig_python_core import DynamicConfig

from ..prompt_version import PromptVersion


class AgentVersion:
    def __init__(self, root_config: DynamicConfig, is_live_for_user: bool):
        self._root_prompt = PromptVersion(root_config, is_live_for_user)

    def get_root(self) -> PromptVersion:
        return self._root_prompt
