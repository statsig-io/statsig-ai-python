from contextlib import contextmanager
from dataclasses import dataclass
import functools
import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    Literal,
    Mapping,
    Optional,
    ParamSpec,
    TypeVar,
    overload,
)

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Span, StatusCode
from statsig_python_core import StatsigUser

from ..otel.conventions import (
    STATSIG_ATTR_GEN_AI_SPAN_TYPE,
    STATSIG_ATTR_SPAN_LLM_ROOT,
    STATSIG_ATTR_SPAN_TYPE,
    STATSIG_CTX_KEY_LLM_ROOT,
    STATSIG_SPAN_LLM_ROOT_VALUE,
    StatsigGenAISpanType,
    StatsigSpanType,
)
from ..otel.environment_context import get_environment_span_attrs_from_context
from ..otel.prompt_context import get_prompt_span_attrs_from_context
from ..otel.singleton import OtelSingleton
from ..otel.statsig_context import (
    StatsigContext,
    get_statsig_span_attrs_from_context,
    set_statsig_context_to_otel_context,
)
from ..otel.user_context import get_user_span_attrs_from_context
from .span_telemetry import SpanTelemetry

P = ParamSpec("P")
R = TypeVar("R")

WrapType = Literal["workflow", "tool"]

_NAME_PREFIX = "gen_ai."
_OP_NAME_BY_TYPE: Dict[str, str] = {
    "workflow": "invoke_workflow",
    "tool": "execute_tool",
}
_GEN_AI_SPAN_TYPE_BY_TYPE: Dict[str, str] = {
    "workflow": StatsigGenAISpanType.WORKFLOW,
    "tool": StatsigGenAISpanType.TOOL,
}


@dataclass(frozen=True)
class _WrapOptions:
    type: WrapType
    name: str
    attributes: Optional[Mapping[str, Any]] = None
    user: Optional[StatsigUser] = None
    activity_id: Optional[str] = None
    tool_type: Optional[str] = None


class _WrappedCall:
    """The span for one call of a wrap() function, or one start_workflow() block."""

    def __init__(self, options: _WrapOptions) -> None:
        op_name = _OP_NAME_BY_TYPE[options.type]
        ctx = otel_context.get_current()
        if options.user is not None or options.activity_id:
            ctx = set_statsig_context_to_otel_context(
                ctx, StatsigContext(user=options.user, activity_id=options.activity_id or None)
            )
        # The workflow span stands for the whole turn, so the LLM calls it makes
        # aren't marked `statsig.span.llm_root`.
        if options.type == "workflow":
            ctx = otel_context.set_value(STATSIG_CTX_KEY_LLM_ROOT, op_name, ctx)

        self._include_template_variables = options.type == "workflow"

        attrs: Dict[str, Any] = dict(options.attributes or {})
        attrs[STATSIG_ATTR_SPAN_TYPE] = StatsigSpanType.GEN_AI
        attrs.update(get_user_span_attrs_from_context(ctx) or {})
        attrs.update(get_statsig_span_attrs_from_context(ctx) or {})
        attrs.update(get_environment_span_attrs_from_context(ctx) or {})
        attrs.update(self._get_prompt_attrs(ctx))
        attrs.update(_get_attributes_for_input_type(options))

        # Looked up on each call: a function is often decorated at import time,
        # before initialize_tracing() runs.
        tracer = OtelSingleton.get_tracer_provider_static().get_tracer("wrap-call")
        # As with model call spans, the span starts with the plain values and
        # SpanTelemetry writes dicts and lists as capped JSON.
        start_attrs = {
            k: v for k, v in attrs.items() if v is not None and not isinstance(v, (dict, list))
        }
        self.span: Span = tracer.start_span(op_name, context=ctx, attributes=start_attrs)
        self.ctx = trace.set_span_in_context(self.span, ctx)
        self._telemetry = SpanTelemetry(self.span, op_name)
        self._telemetry.set_attributes(attrs)

    def _get_prompt_attrs(self, ctx: Context) -> Dict[str, Any]:
        # A workflow usually renders its prompt after the span starts, so end() reads
        # the template variables again.
        return get_prompt_span_attrs_from_context(
            ctx, include_template_variables=self._include_template_variables
        )

    @contextmanager
    def activated(self) -> Iterator[None]:
        token = otel_context.attach(self.ctx)
        try:
            yield
        finally:
            otel_context.detach(token)

    def end(self, error: Optional[BaseException] = None) -> None:
        # OpenTelemetry ignores status changes once a span is OK, so unlike the Node
        # SDK the status is set when the call finishes, not before it runs.
        if error is None:
            self._telemetry.set_status({"code": StatusCode.OK})
        else:
            self._telemetry.fail(error)
        self._telemetry.set_attributes(self._get_prompt_attrs(self.ctx))
        self._telemetry.end()

    async def end_after(self, awaitable: Awaitable[R]) -> R:
        try:
            with self.activated():
                result = await awaitable
        except BaseException as e:
            self.end(e)
            raise
        self.end()
        return result


def _get_attributes_for_input_type(options: _WrapOptions) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {
        f"statsig.{_NAME_PREFIX}{options.type}.name": options.name,
    }
    if options.type == "tool" and options.tool_type:
        attrs[f"{_NAME_PREFIX}tool.type"] = options.tool_type
    attrs[STATSIG_ATTR_GEN_AI_SPAN_TYPE] = _GEN_AI_SPAN_TYPE_BY_TYPE[options.type]
    return attrs


def _wrap_function(fn: Callable[..., Any], options: _WrapOptions) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            call = _WrappedCall(options)
            try:
                with call.activated():
                    result = await fn(*args, **kwargs)
            except BaseException as e:
                call.end(e)
                raise
            call.end()
            return result

        return async_wrapper

    @functools.wraps(fn)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        call = _WrappedCall(options)
        try:
            with call.activated():
                result = fn(*args, **kwargs)
        except BaseException as e:
            call.end(e)
            raise
        # Like a promise in the Node SDK, a coroutine returned by a plain function
        # (a lambda around an async call, say) keeps the span open until it's awaited.
        if inspect.iscoroutine(result):
            return call.end_after(result)
        call.end()
        return result

    return sync_wrapper


@overload
def wrap(
    fn: Callable[P, R],
    *,
    type: WrapType,
    name: str,
    attributes: Optional[Mapping[str, Any]] = None,
    user: Optional[StatsigUser] = None,
    activity_id: Optional[str] = None,
    tool_type: Optional[str] = None,
) -> Callable[P, R]: ...


@overload
def wrap(
    fn: None = None,
    *,
    type: WrapType,
    name: str,
    attributes: Optional[Mapping[str, Any]] = None,
    user: Optional[StatsigUser] = None,
    activity_id: Optional[str] = None,
    tool_type: Optional[str] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def wrap(
    fn: Optional[Callable[..., Any]] = None,
    *,
    type: WrapType,
    name: str,
    attributes: Optional[Mapping[str, Any]] = None,
    user: Optional[StatsigUser] = None,
    activity_id: Optional[str] = None,
    tool_type: Optional[str] = None,
) -> Any:
    """
    Records each call of `fn` as an `invoke_workflow` (type="workflow") or
    `execute_tool` (type="tool") span, and logs a `statsig::gen_ai` event for it.
    Wrapped model calls and other wrap() spans made inside it are its children.

    Use it as `wrap(fn, type=..., name=...)` or as a decorator,
    `@wrap(type="tool", name="search")`. `async def` functions get an async wrapper
    that ends the span once the call is awaited.

    `user` and `activity_id` set the Statsig context for the call, as
    `statsig_context` does. `tool_type` is recorded as `gen_ai.tool.type`.
    Model calls inside a workflow aren't marked `statsig.span.llm_root`.
    """
    if type not in _OP_NAME_BY_TYPE:
        raise ValueError(f"[Statsig] wrap() type must be 'workflow' or 'tool', got {type!r}")
    options = _WrapOptions(
        type=type,
        name=name,
        attributes=attributes,
        user=user,
        activity_id=activity_id,
        tool_type=tool_type,
    )
    if fn is None:
        return lambda f: _wrap_function(f, options)
    return _wrap_function(fn, options)


@contextmanager
def start_workflow(
    name: str,
    *,
    user: Optional[StatsigUser] = None,
    attributes: Optional[Mapping[str, Any]] = None,
    activity_id: Optional[str] = None,
) -> Iterator[Span]:
    """
    Runs the block in an `invoke_workflow` span marked `statsig.span.llm_root=1`, the
    root of an LLM turn. Model calls inside it are its children and aren't marked as
    the root. Works with `with` in sync code and inside `async def`. An exception
    raised in the block is recorded on the span and propagates.
    """
    call = _WrappedCall(
        _WrapOptions(
            type="workflow",
            name=name,
            attributes={
                **(attributes or {}),
                STATSIG_ATTR_SPAN_LLM_ROOT: STATSIG_SPAN_LLM_ROOT_VALUE,
            },
            user=user,
            activity_id=activity_id,
        )
    )
    try:
        with call.activated():
            yield call.span
    except BaseException as e:
        call.end(e)
        raise
    call.end()
