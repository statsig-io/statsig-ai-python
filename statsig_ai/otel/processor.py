from typing import Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .environment_context import get_environment_span_attrs_from_context
from .user_context import get_user_span_attrs_from_context


class StatsigSpanProcessor(BatchSpanProcessor):
    """
    A Statsig wrapper around the OpenTelemetry BatchSpanProcessor.

    Tags each span with the Statsig user from `statsig_user_context` and the
    environment from `statsig_environment` when it starts.
    """

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        user_attrs = get_user_span_attrs_from_context(parent_context)
        if user_attrs:
            span.set_attributes(user_attrs)

        environment_attrs = get_environment_span_attrs_from_context(parent_context)
        if environment_attrs:
            span.set_attributes(environment_attrs)

        super().on_start(span, parent_context=parent_context)
