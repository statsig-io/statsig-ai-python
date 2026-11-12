from typing import Optional

from opentelemetry import trace
from opentelemetry.trace import TracerProvider

from .resources import StatsigResourceAttributes


class OtelSingleton:
    _instance = None

    def __init__(
        self,
        tracer_provider: TracerProvider,
        resources: Optional[StatsigResourceAttributes] = None,
    ):
        self._tracer_provider = tracer_provider
        self._resources = resources if resources is not None else StatsigResourceAttributes()

    @classmethod
    def instantiate(
        cls,
        tracer_provider: TracerProvider,
        resources: Optional[StatsigResourceAttributes] = None,
    ):
        """
        Create or return an existing OtelSingleton instance.
        """
        if cls._instance is not None:
            print(
                "[Statsig otel] OtelSingleton instance already exists. "
                "Returning the existing instance."
            )
            return cls._instance
        cls._instance = cls(tracer_provider, resources)
        return cls._instance

    @classmethod
    def is_initialized(cls) -> bool:
        """Whether initialize_tracing() has set up tracing."""
        return cls._instance is not None

    @classmethod
    def get_instance(cls):
        if cls._instance is not None:
            return cls._instance
        return NoopOtelSingleton.get_instance()

    def get_tracer_provider(self) -> TracerProvider:
        return self._tracer_provider

    def get_resources(self) -> StatsigResourceAttributes:
        """The service, version and environment initialize_tracing() put on the tracer resource."""
        return self._resources

    @classmethod
    def get_tracer_provider_static(cls) -> TracerProvider:
        return cls.get_instance().get_tracer_provider()

    @classmethod
    def flush_instance(cls) -> None:
        # Without initialize_tracing(), wrapped calls go to the global provider, so flush that.
        # Read it directly to skip NoopOtelSingleton's warning on every flush.
        if cls._instance is not None:
            tracer_provider = cls._instance.get_tracer_provider()
        else:
            tracer_provider = trace.get_tracer_provider()
        force_flush = getattr(tracer_provider, "force_flush", None)
        if callable(force_flush):
            force_flush()


class NoopOtelSingleton(OtelSingleton):
    def __init__(self):
        super().__init__(trace.get_tracer_provider())

    @classmethod
    def get_instance(cls):
        print(
            "[Statsig otel] NoopOtelSingleton instance is being used. "
            "OtelSingleton has not been properly instantiated."
        )
        return cls()
