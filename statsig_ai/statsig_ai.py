from typing import Optional, Union
from .statsig_ai_base import StatsigAIInstance
from .statsig_ai_options import StatsigAIOptions
from .statsig_ai_base import StatsigCreateConfig, StatsigAttachConfig


class StatsigAI(StatsigAIInstance):
    _statsig_ai_shared_instance: Optional[StatsigAIInstance] = None

    @classmethod
    def shared(cls) -> StatsigAIInstance:
        if not cls.has_shared_instance():
            return create_statsig_ai_error_instance(
                "StatsigAI.shared() called, but no instance has been set with StatsigAI.new_shared(...)"
            )
        return cls._statsig_ai_shared_instance

    @classmethod
    def new_shared(
        cls,
        statsig_source: Union[StatsigCreateConfig, StatsigAttachConfig],
        ai_options: Optional[StatsigAIOptions] = None,
    ) -> StatsigAIInstance:
        if cls.has_shared_instance():
            return create_statsig_ai_error_instance(
                "StatsigAI shared instance already exists. "
                "Call StatsigAI.remove_shared() before creating a new instance."
            )

        cls._statsig_ai_shared_instance = StatsigAIInstance(
            statsig_source=statsig_source, ai_options=ai_options
        )
        return cls._statsig_ai_shared_instance

    @classmethod
    def remove_shared(cls) -> None:
        cls._statsig_ai_shared_instance = None

    @classmethod
    def has_shared_instance(cls) -> bool:
        return cls._statsig_ai_shared_instance is not None


def create_statsig_ai_error_instance(message: str) -> StatsigAIInstance:
    print("Error: ", message)
    return StatsigAIInstance(
        statsig_source=StatsigCreateConfig(sdk_key="__STATSIG_ERROR_SDK_KEY__"),
        ai_options=None,
    )
