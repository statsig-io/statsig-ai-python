from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from typing import Dict, Generator, Optional

from opentelemetry import context as otel_context
from opentelemetry.context import Context
from statsig_python_core import StatsigUser

from .conventions import (
    STATSIG_ATTR_CUSTOM_IDS,
    STATSIG_ATTR_USER_ID,
    STATSIG_CTX_KEY_ACTIVE_USER,
)


@dataclass(frozen=True)
class UserContextItem:
    user_id: str
    custom_ids: Dict[str, str] = field(default_factory=dict)


def set_user_to_otel_context(ctx: Context, user: StatsigUser) -> Context:
    item = UserContextItem(
        user_id=user.user_id or "unknown",
        custom_ids=dict(user.custom_ids or {}),
    )
    return otel_context.set_value(STATSIG_CTX_KEY_ACTIVE_USER, item, ctx)


def get_user_from_otel_context(
    ctx: Optional[Context] = None,
) -> Optional[UserContextItem]:
    user = otel_context.get_value(STATSIG_CTX_KEY_ACTIVE_USER, ctx)
    if not isinstance(user, UserContextItem):
        return None
    return user


def get_user_span_attrs_from_context(
    ctx: Optional[Context] = None,
) -> Optional[Dict[str, str]]:
    user = get_user_from_otel_context(ctx)
    if user is None:
        return None

    attrs = {STATSIG_ATTR_USER_ID: user.user_id}
    if user.custom_ids:
        attrs[STATSIG_ATTR_CUSTOM_IDS] = json.dumps(user.custom_ids, separators=(",", ":"), ensure_ascii=False)
    return attrs


@contextmanager
def statsig_user_context(user: StatsigUser) -> Generator[Context, None, None]:
    """
    Tags every span started inside the block with the user's `statsig.user_id`
    and `statsig.custom_ids`. `StatsigSpanProcessor` adds them when each span starts.
    """
    new_ctx = set_user_to_otel_context(otel_context.get_current(), user)
    token = otel_context.attach(new_ctx)
    try:
        yield new_ctx
    finally:
        otel_context.detach(token)
