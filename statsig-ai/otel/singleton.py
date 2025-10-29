from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider


class OtelSingleton:
    _instance = None

    def __init__(self, tracer_provider: TracerProvider):
        self._tracer_provider = tracer_provider

    @classmethod
    def instantiate(cls, tracer_provider: TracerProvider):
        """
        Create or return an existing OtelSingleton instance.
        """
        if cls._instance is not None:
            print(
                "[Statsig otel] OtelSingleton instance already exists. "
                "Returning the existing instance."
            )
            return cls._instance
        cls._instance = cls(tracer_provider)
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is not None:
            return cls._instance
        return NoopOtelSingleton.get_instance()

    @classmethod
    def __reset(cls):
        cls._instance = None

    def get_tracer_provider(self) -> TracerProvider:
        return self._tracer_provider

    @classmethod
    def get_tracer_provider_static(cls) -> TracerProvider:
        return cls.get_instance().get_tracer_provider()


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
