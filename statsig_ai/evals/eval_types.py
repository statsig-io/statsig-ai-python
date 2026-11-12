import logging
import math
import threading
from typing import (
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Generic,
    Optional,
    Union,
    Any,
    List,
)
from dataclasses import dataclass
from typing_extensions import TypedDict

from .serializable_data_class import SerializableDataClass

Input = TypeVar("Input")
Output = TypeVar("Output")


# ================================
# Eval Data Record Types
# ================================


@dataclass
class EvalDataRecord(SerializableDataClass, Generic[Input, Output]):
    input: Input
    category: Optional[Union[Sequence[str], str]] = None
    expected: Optional[Output] = None
    # Id of the record's row in the eval's `dataset`
    row_id: Optional[str] = None


class _EvalDataRecordDictVoidExpected(TypedDict, Generic[Input], total=False):
    input: Input
    category: Optional[Union[Sequence[str], str]]
    row_id: Optional[str]


class _EvalDataRecordDict(
    _EvalDataRecordDictVoidExpected[Input], Generic[Input, Output], total=False
):
    expected: Optional[Output]


_DataclassOrDictEvalDataRecord = Union[
    EvalDataRecord[Input, Output],
    _EvalDataRecordDict[Input, Output],
    _EvalDataRecordDictVoidExpected[Input],
]

_PossibleEvalDataRecords = Union[
    Iterable[_DataclassOrDictEvalDataRecord[Input, Output]],
    Iterator[_DataclassOrDictEvalDataRecord[Input, Output]],
    Awaitable[Iterator[_DataclassOrDictEvalDataRecord[Input, Output]]],
    Callable[
        [],
        Union[
            Iterable[_DataclassOrDictEvalDataRecord[Input, Output]],
            Awaitable[Iterable[_DataclassOrDictEvalDataRecord[Input, Output]]],
        ],
    ],
]

EvalData = Union[
    _PossibleEvalDataRecords[Input, Output], Type[_PossibleEvalDataRecords[Input, Output]]
]

# ================================
# Eval Scorer Type
# ================================

OnlyScore = Union[int, float, bool]

@dataclass
class ScoreWithMetadata(SerializableDataClass):
    score: OnlyScore
    # Values that aren't strings are sent to Statsig as JSON strings.
    metadata: Optional[Dict[str, Any]] = None

Score = Union[OnlyScore, ScoreWithMetadata]


@dataclass
class EvalScorerArgs(SerializableDataClass, Generic[Input, Output]):
    input: Input
    output: Output
    expected: Optional[Output] = None
    category: Optional[Union[Sequence[str], str]] = None
    row_id: Optional[str] = None


class SyncScorer(Protocol, Generic[Input, Output]):
    def __call__(self, args: EvalScorerArgs[Input, Output]) -> Score: ...


class AsyncScorer(Protocol, Generic[Input, Output]):
    async def score(self, args: EvalScorerArgs[Input, Output]) -> Score: ...


ScorerInterface = Union[SyncScorer[Input, Output], AsyncScorer[Input, Output]]

Scorer = Union[
    ScorerInterface[Input, Output],
    Type[ScorerInterface[Input, Output]],
    Callable[[EvalScorerArgs[Input, Output]], Score],
    Callable[[EvalScorerArgs[Input, Output]], Awaitable[Score]],
]

ScorerFnMap = Dict[str, Scorer[Input, Output]]

EvalScorer = Union[Scorer[Input, Output], ScorerFnMap[Input, Output]]


# ================================
# Eval Result Types
# ================================


@dataclass
class EvalResultMetadata(SerializableDataClass):
    error: bool


@dataclass
class EvalResultRecordWithMetadata(SerializableDataClass, Generic[Input, Output]):
    input: Input
    output: Output
    scores: Dict[str, ScoreWithMetadata]
    expected: Optional[Output] = None
    error: Optional[bool] = False
    category: Optional[Union[Sequence[str], str]] = None
    row_id: Optional[str] = None
    # Trace and root span of the record's run, as lowercase hex. Set when tracing is initialized.
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    # Milliseconds the task took to produce its output. None when the task raised.
    latency_ms: Optional[int] = None
    # Totals the task reported with `EvalHook.report_tokens`. None when it reported none or raised.
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

@dataclass
class EvalResultRecord(SerializableDataClass, Generic[Input, Output]):
    input: Input
    output: Output
    scores: Dict[str, float]
    expected: Optional[Output] = None
    error: Optional[bool] = False
    category: Optional[Union[Sequence[str], str]] = None
    row_id: Optional[str] = None
    # Trace and root span of the record's run, as lowercase hex. Set when tracing is initialized.
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    # Milliseconds the task took to produce its output. None when the task raised.
    latency_ms: Optional[int] = None
    # Totals the task reported with `EvalHook.report_tokens`. None when it reported none or raised.
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class EvalResult(SerializableDataClass, Generic[Input, Output]):
    results: List[EvalResultRecord[Input, Output]]
    metadata: EvalResultMetadata
    summary_scores: Optional[Dict[str, float]] = None


# ================================
# Eval Task Type
# ================================

# Parameter values are passed to tasks as is, except a pydantic
# `Field(default=...)` or `Field(default_factory=...)`, which is resolved to its
# default before any task runs.
EvalParameters = Dict[str, Any]

# Guards every hook's token totals. A sync task runs in a worker thread, and can
# report tokens from threads of its own.
_TOKEN_TOTALS_LOCK = threading.Lock()


def _is_token_count(value: Any) -> bool:
    # A bool is an int, but not a count. Ints aren't passed to isfinite, which
    # raises for ones too large for a float.
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    return isinstance(value, float) and math.isfinite(value) and value >= 0


@dataclass
class EvalHook(SerializableDataClass):
    """The optional second argument of an eval's task. Each record gets its own."""

    parameters: Optional[EvalParameters] = None
    category: Optional[Union[Sequence[str], str]] = None

    def __post_init__(self) -> None:
        # Input and output tokens the task reported. Not a field, so it isn't
        # serialized or compared.
        self._token_totals: Optional[Tuple[int, int]] = None

    def report_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """
        Report the tokens the task used to produce its output. Each call adds to the
        record's totals, so a multi-step task can call it once per model call or once
        with its total. The totals are sent with the record's result as `inputTokens`
        and `outputTokens`. A record whose task never calls this, or raises, sends no
        tokens, and calls made after the task returns don't count.

        Counts must be ints, or finite floats, of at least 0. A call with any other
        count is ignored with a warning, since one bad count would make the API reject
        every result in the run.
        """
        if not _is_token_count(input_tokens) or not _is_token_count(output_tokens):
            logging.warning(
                "[Statsig] report_tokens() needs input_tokens and output_tokens of at least 0. "
                "Ignoring: input_tokens=%r, output_tokens=%r",
                input_tokens,
                output_tokens,
            )
            return
        with _TOKEN_TOTALS_LOCK:
            total_input, total_output = self._token_totals or (0, 0)
            self._token_totals = (total_input + input_tokens, total_output + output_tokens)


EvalTask = Union[
    Callable[[Input], Union[Output, Awaitable[Output]]],
    Callable[[Input, EvalHook], Union[Output, Awaitable[Output]]],
]

SummaryScorerFn = Callable[[List[EvalResultRecord[Input, Output]]], Dict[str, float]]
