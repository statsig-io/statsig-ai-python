from typing import Optional


class StatsigAIOptions:
    """
    An object of properties for initializing the AI SDK with additional parameters
    """

    def __init__(
        self, enable_default_otel: Optional[bool], otel_service_name: Optional[str]
    ):
        self.enable_default_otel = enable_default_otel
        self.otel_service_name = otel_service_name
