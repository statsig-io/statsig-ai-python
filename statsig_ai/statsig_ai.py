from typing import Optional, Union, overload
from statsig_python_core import Statsig, StatsigOptions
from .statsig_ai_base import StatsigAIInstance
from .statsig_ai_options import StatsigAIOptions
from .statsig_ai_base import StatsigCreateConfig, StatsigAttachConfig


class StatsigAI(StatsigAIInstance):
    _statsig_ai_shared_instance: Optional[StatsigAIInstance] = None

    # ----------------------------
    #    Overloaded Constructors
    # ----------------------------

    @overload
    def __new__(
        cls,
        statsig_source: StatsigCreateConfig,
        ai_options: Optional[StatsigAIOptions] = None,
    ) -> StatsigAIInstance: ...
    @overload
    def __new__(
        cls,
        statsig_source: StatsigAttachConfig,
        ai_options: Optional[StatsigAIOptions] = None,
    ) -> StatsigAIInstance: ...

    def __new__(
        cls,
        statsig_source: Union[StatsigCreateConfig, StatsigAttachConfig],
        ai_options: Optional[StatsigAIOptions] = None,
    ) -> StatsigAIInstance:
        instance = super().__new__(cls, statsig_source, ai_options)
        return instance

    # ----------------------------
    #       Shared Instance
    # ----------------------------

    @classmethod
    def shared(cls) -> StatsigAIInstance:
        if (
            not StatsigAI.has_shared_instance()
            or cls._statsig_ai_shared_instance is None
        ):
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
        if StatsigAI.has_shared_instance():
            return create_statsig_ai_error_instance(
                "StatsigAI shared instance already exists. Call StatsigAI.remove_shared() before creating a new instance."
            )

        cls._statsig_ai_shared_instance = super().__new__(
            cls, statsig_source, ai_options
        )
        return cls._statsig_ai_shared_instance

    @classmethod
    def remove_shared(cls) -> None:
        cls._statsig_ai_shared_instance = None

    @classmethod
    def has_shared_instance(cls) -> bool:
        return (
            hasattr(cls, "_statsig_ai_shared_instance")
            and cls._statsig_ai_shared_instance is not None
        )


def create_statsig_ai_error_instance(message: str) -> StatsigAIInstance:
    print("Error: ", message)
    return StatsigAIInstance.__new__(
        StatsigAIInstance, {"sdk_key": "__STATSIG_ERROR_SDK_KEY__"}, None
    )
