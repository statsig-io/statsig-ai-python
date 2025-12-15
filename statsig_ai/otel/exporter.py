import os
from typing import Optional

from opentelemetry.exporter.otlp.proto.http.trace_exporter import Compression, OTLPSpanExporter


class StatsigOTLPTraceExporterOptions:
    def __init__(self, sdk_key: Optional[str] = None, dsn: Optional[str] = None):
        self.sdk_key = sdk_key
        self.dsn = dsn


class StatsigOTLPTraceExporter(OTLPSpanExporter):
    def __init__(self, options: StatsigOTLPTraceExporterOptions, debug: bool = False):
        self._debug = debug
        sdk_key = options.sdk_key or os.getenv("STATSIG_SDK_KEY")
        if not sdk_key:
            raise ValueError("Statsig SDK Key is required for StatsigOTLPTraceExporter")

        dsn = options.dsn or "https://api.statsig.com/otlp"

        super().__init__(
            endpoint=f"{dsn.rstrip('/')}/v1/traces",
            headers={"statsig-api-key": sdk_key},
            compression=Compression.Gzip,
        )
