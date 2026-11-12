import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StatsigResourceAttributes:
    """The service, version and environment that initialize_tracing() puts on the tracer resource.

    The statsig::gen_ai event also sends them as custom IDs on its user.
    """

    service: Optional[str] = None
    version: Optional[str] = None
    environment: Optional[str] = None


def create_resource_attributes(
    service_name: Optional[str] = None,
    version: Optional[str] = None,
    environment: Optional[str] = None,
) -> StatsigResourceAttributes:
    # Like the Node SDK, an unset or empty option falls back to the environment
    # variables. The environment comes from PYTHON_ENV, where Node reads NODE_ENV.
    return StatsigResourceAttributes(
        service=service_name or os.getenv("OTEL_SERVICE_NAME") or os.getenv("SERVICE_NAME"),
        version=version,
        environment=environment or os.getenv("PYTHON_ENV"),
    )
