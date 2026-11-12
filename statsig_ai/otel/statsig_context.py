from contextlib import contextmanager
from dataclasses import dataclass
import json
from typing import Dict, Generator, Optional

from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry.trace import Span
from statsig_python_core import StatsigUser

from .conventions import (
    STATSIG_ATTR_ACTIVITY_ID,
    STATSIG_ATTR_CUSTOM_IDS,
    STATSIG_ATTR_USER_ID,
    STATSIG_CTX_KEY,
)


@dataclass
class StatsigContext:
    user: Optional[StatsigUser] = None
    activity_id: Optional[str] = None


def set_statsig_context_to_otel_context(
    ctx: Context,
    to_set_statsig_context: StatsigContext,
) -> Context:
    return otel_context.set_value(STATSIG_CTX_KEY, to_set_statsig_context, ctx)


def get_statsig_context_from_otel_context(
    ctx: Optional[Context] = None,
) -> Optional[StatsigContext]:
    if ctx is None:
        ctx = otel_context.get_current()

    statsig_ctx = otel_context.get_value(STATSIG_CTX_KEY, ctx)
    if statsig_ctx is None or not isinstance(statsig_ctx, StatsigContext):
        return None

    return statsig_ctx


def get_statsig_span_attrs_from_context(
    ctx: Optional[Context] = None,
) -> Optional[Dict[str, str]]:
    statsig_ctx = get_statsig_context_from_otel_context(ctx)
    if statsig_ctx is None:
        return None

    attrs: Dict[str, str] = {}

    if statsig_ctx.user is not None:
        attrs[STATSIG_ATTR_USER_ID] = getattr(statsig_ctx.user, "user_id", None) or "unknown"

    custom_ids: Dict[str, str] = {}
    if statsig_ctx.user is not None:
        custom_ids = dict(getattr(statsig_ctx.user, "custom_ids", None) or {})

    if statsig_ctx.activity_id is not None:
        custom_ids[STATSIG_ATTR_ACTIVITY_ID] = statsig_ctx.activity_id

    if custom_ids:
        attrs[STATSIG_ATTR_CUSTOM_IDS] = json.dumps(custom_ids, separators=(",", ":"), ensure_ascii=False)

    return attrs if attrs else None


def set_statsig_span_attrs_from_context(
    span: Span,
    ctx: Optional[Context] = None,
) -> None:

    attrs = get_statsig_span_attrs_from_context(ctx)
    if attrs is None:
        return

    for key, value in attrs.items():
        span.set_attribute(key, value)


@contextmanager
def statsig_context(
    statsig_ctx: StatsigContext,
) -> Generator[Context, None, None]:
    ctx = otel_context.get_current()
    new_ctx = set_statsig_context_to_otel_context(ctx, statsig_ctx)
    token = otel_context.attach(new_ctx)
    try:
        yield new_ctx
    finally:
        otel_context.detach(token)
