from typing import (
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Protocol,
    Sequence,
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


class _EvalDataRecordDictVoidExpected(TypedDict, Generic[Input], total=False):
    input: Input
    category: Optional[Union[Sequence[str], str]]


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
# Eval Result Types
# ================================


@dataclass
class EvalResultMetadata(SerializableDataClass):
    error: bool


@dataclass
class EvalResultRecord(SerializableDataClass, Generic[Input, Output]):
    input: Input
    output: Output
    scores: Dict[str, float]
    expected: Optional[Output] = None
    error: Optional[bool] = False
    category: Optional[Union[Sequence[str], str]] = None


@dataclass
class EvalResult(SerializableDataClass, Generic[Input, Output]):
    results: List[EvalResultRecord[Input, Output]]
    metadata: EvalResultMetadata
    summary_scores: Optional[Dict[str, float]] = None


# ================================
# Eval Scorer Type
# ================================

Score = Union[int, float, bool]


@dataclass
class EvalScorerArgs(SerializableDataClass, Generic[Input, Output]):
    input: Input
    output: Output
    expected: Optional[Output] = None


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
# Eval Task Type
# ================================

EvalParameters = Dict[str, Any]


@dataclass
class EvalHook(SerializableDataClass):
    parameters: Optional[EvalParameters] = None
    category: Optional[Union[Sequence[str], str]] = None


EvalTask = Union[
    Callable[[Input], Union[Output, Awaitable[Output]]],
    Callable[[Input, EvalHook], Union[Output, Awaitable[Output]]],
]

SummaryScorerFn = Callable[[List[EvalResultRecord[Input, Output]]], Dict[str, float]]
