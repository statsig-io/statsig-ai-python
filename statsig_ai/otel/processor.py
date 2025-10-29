from opentelemetry.sdk.trace.export import BatchSpanProcessor


class StatsigSpanProcessor(BatchSpanProcessor):
    """
    A Statsig wrapper around the OpenTelemetry BatchSpanProcessor.

    Currently identical to BatchSpanProcessor, but provides a named subclass
    for future customization.
    """

    pass
