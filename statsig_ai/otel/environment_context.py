from contextlib import contextmanager
from typing import Dict, Generator, Optional

from opentelemetry import context as otel_context
from opentelemetry.context import Context

from .conventions import STATSIG_ATTR_ENVIRONMENT, STATSIG_CTX_KEY_ACTIVE_ENVIRONMENT


def get_environment_span_attrs_from_context(
    ctx: Optional[Context] = None,
) -> Optional[Dict[str, str]]:
    environment = otel_context.get_value(STATSIG_CTX_KEY_ACTIVE_ENVIRONMENT, ctx)
    if not isinstance(environment, str) or not environment:
        return None
    return {STATSIG_ATTR_ENVIRONMENT: environment}


@contextmanager
def statsig_environment(environment: str) -> Generator[Context, None, None]:
    """
    Tags every span started inside the block with `statsig.environment`, overriding
    the tracer's environment resource for those spans only. Spans started inside
    `statsig_environment("")` get no `statsig.environment`.

    Use `production` for live traffic. `eval` and `online_eval_judge` are reserved for
    spans Statsig generates (`Eval()` runs and online eval judges).
    """
    new_ctx = otel_context.set_value(STATSIG_CTX_KEY_ACTIVE_ENVIRONMENT, environment)
    token = otel_context.attach(new_ctx)
    try:
        yield new_ctx
    finally:
        otel_context.detach(token)
