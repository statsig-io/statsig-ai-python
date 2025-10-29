import os
from typing import Optional

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


class StatsigOTLPTraceExporter(OTLPSpanExporter):
    """
    A custom OTLP Trace Exporter for Statsig.
    Adds Statsig authentication and endpoint configuration.

    Args:
        sdk_key (str | None): Optional Statsig SDK key. If not provided, will
            attempt to read from the STATSIG_SDK_KEY environment variable.
        dsn (str | None): Optional base DSN for custom endpoint configuration.
            Defaults to 'https://api.statsig.com/otlp'. '/v1/traces' will be appended automatically.
    """

    def __init__(self, sdk_key: Optional[str] = None, dsn: Optional[str] = None):
        sdk_key = sdk_key or os.getenv("STATSIG_SDK_KEY")
        if not sdk_key:
            raise ValueError("Statsig SDK Key is required for StatsigOTLPTraceExporter")

        dsn = dsn or "https://api.statsig.com/otlp"
        endpoint = f"{dsn.rstrip('/')}/v1/traces"

        super().__init__(
            endpoint=endpoint,
            headers={"statsig-api-key": sdk_key},
        )
