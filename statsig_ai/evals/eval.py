import asyncio
import inspect
import logging
import os
import urllib.parse
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
)

import requests

from .eval_types import (
    EvalDataRecord,
    EvalHook,
    EvalResult,
    EvalResultMetadata,
    EvalScorerArgs,
    Input,
    ScorerFnMap,
    Scorer,
    Output,
    Score,
    EvalData,
    EvalTask,
    EvalScorer,
    EvalParameters,
    _DataclassOrDictEvalDataRecord,
    EvalResultRecord,
    SummaryScorerFn,
)


T = TypeVar("T")


async def to_async_iter(iterable: Iterable[T]) -> AsyncIterator[T]:
    for item in iterable:
        yield item


def _normalize_record(
    record: _DataclassOrDictEvalDataRecord,
) -> EvalDataRecord[Input, Output]:
    """Convert a dict or dataclass record into EvalDataRecord."""
    if isinstance(record, EvalDataRecord):
        return record

    if isinstance(record, dict):
        return EvalDataRecord(
            input=cast(Input, record.get("input")),
            category=record.get("category"),
            expected=cast(Optional[Output], record.get("expected")),
        )

    raise TypeError(f"Invalid data record type: {type(record)}")


async def _normalize_into_data_iterator(
    data: EvalData,
) -> AsyncIterator[EvalDataRecord[Input, Output]]:
    data_iterator: Any = data
    if inspect.isclass(data_iterator):
        data_iterator = data_iterator()

    if inspect.isfunction(data_iterator) or inspect.isroutine(data_iterator):
        data_iterator = data_iterator()

    if inspect.iscoroutine(data_iterator) or inspect.isawaitable(data_iterator):
        data_iterator = await data_iterator

    if not inspect.isasyncgen(data_iterator):
        data_iterator = to_async_iter(data_iterator)

    async for record in data_iterator:
        yield _normalize_record(record)


def _normalize_scorers(scorers: EvalScorer) -> ScorerFnMap[Input, Output]:
    scorers_dict = scorers if isinstance(scorers, dict) else {"Grader": scorers}
    for name, scorer in scorers_dict.items():
        if inspect.isclass(scorer):
            scorers_dict[name] = scorer()
    return scorers_dict


def _normalize_score_value(score_value: Score) -> float:
    """Convert a score to a consistent float value."""
    if isinstance(score_value, bool):
        return 1.0 if score_value else 0.0
    if isinstance(score_value, (int, float)):
        return float(score_value)
    logging.warning("[Statsig] Invalid score type: %s", type(score_value))
    try:
        return float(score_value)
    except (ValueError, TypeError):
        return 0.0


def _send_eval_results(
    name: str,
    records: List[EvalResultRecord[Input, Output]],
    api_key: str,
    eval_run_name: Optional[str],
    summary_scores: Optional[Dict[str, float]],
) -> None:
    """Send eval results to Statsig API (synchronous, using requests)."""

    url = f"https://api.statsig.com/console/v1/evals/send_results/{urllib.parse.quote(name)}"
    record_as_dicts = []
    try:
        record_as_dicts = [record.to_dict() for record in records]
    except Exception as err:
        logging.warning("[Statsig] Failed to convert records to dicts: %s", err)
        record_as_dicts = []

    payload: Dict[str, Any] = {"results": record_as_dicts}
    if eval_run_name:
        payload["name"] = eval_run_name
    if summary_scores:
        payload["summaryScores"] = summary_scores
    headers = {
        "Content-Type": "application/json",
        "STATSIG-API-KEY": api_key,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.ok:
            logging.info(
                "[Statsig] Sent eval results (%d records): %s",
                len(records),
                response.reason,
            )
        else:
            logging.warning(
                "[Statsig] Failed to send eval results: %s %s - %s",
                response.status_code,
                response.reason,
                response.text,
            )
    except requests.RequestException as error:
        logging.error("[Statsig] Error sending eval results: %s", error)


def _run_summary_scorer(
    summary_scorer: Optional[SummaryScorerFn[Input, Output]],
    results: List[EvalResultRecord[Input, Output]],
) -> Optional[Dict[str, float]]:
    try:
        if summary_scorer:
            return summary_scorer(results)
        return None
    except Exception as err:
        logging.warning("[Statsig] Summary scorer failed: %s", err)
        return None


async def _run_task(
    task: EvalTask[Input, Output],
    input_value: Input,
    hook: EvalHook,
) -> Output:
    sig = inspect.signature(task)
    params = list(sig.parameters.values())
    num_args = len(params)

    if num_args == 1:
        args: Any = (input_value,)
    elif num_args >= 2:
        args = (input_value, hook)
    else:
        raise TypeError(f"Invalid task signature: expected 1 or 2 parameters, got {num_args}")

    if inspect.iscoroutinefunction(task):
        return await task(*args)

    event_loop = asyncio.get_event_loop()

    def sync_call() -> Output:
        return task(*args)  # type: ignore

    return await event_loop.run_in_executor(None, sync_call)


async def _run_scorer(
    scorer_name: str,
    scorer: Scorer[Input, Output],
    output: Output,
    expected: Optional[Output],
    input_value: Input,
) -> float:
    """Run a single scorer and return normalized score as float."""
    try:
        args = EvalScorerArgs(input=input_value, output=output, expected=expected)

        # Check for AsyncScorer protocol (has a score method)
        if hasattr(scorer, "score") and callable(getattr(scorer, "score", None)):
            score_method = getattr(scorer, "score")
            raw_score = score_method(args)
        elif callable(scorer):
            # SyncScorer or callable function
            scorer_fn = cast(
                Callable[[EvalScorerArgs[Input, Output]], Union[Score, Awaitable[Score]]], scorer
            )
            raw_score = scorer_fn(args)
        else:
            raise TypeError(f"Invalid scorer type: {type(scorer)}")

        if inspect.isawaitable(raw_score):
            raw_score = await raw_score

        return _normalize_score_value(raw_score)
    except Exception as err:
        logging.warning(
            "[Statsig] Scorer '%s' failed: %s | input=%r",
            scorer_name,
            err,
            input_value,
        )
        return 0.0


async def _run_task_and_score_one_record(
    task: EvalTask[Input, Output],
    scorers: ScorerFnMap[Input, Output],
    data_record: EvalDataRecord[Input, Output],
    parameters: Optional[EvalParameters],
) -> EvalResultRecord[Input, Optional[Output]]:
    """Run task and score for one record asynchronously.

    Runs the task first, then runs all scorers concurrently on the output.
    - If task errors: all scores are "error" and error flag is set
    - If a scorer errors: that score is "0"
    """
    eval_result: EvalResultRecord[Input, Optional[Output]] = EvalResultRecord(
        input=data_record.input,
        expected=data_record.expected,
        category=data_record.category,
        output=None,
        scores={},
        error=False,
    )

    try:
        hook = EvalHook(parameters=parameters, category=data_record.category)
        output = await _run_task(task, data_record.input, hook)
        eval_result.output = output

        scorer_tasks = [
            _run_scorer(scorer_name, scorer_fn, output, data_record.expected, data_record.input)
            for scorer_name, scorer_fn in scorers.items()
        ]

        score_results = await asyncio.gather(*scorer_tasks)

        eval_result.scores = dict(zip(scorers.keys(), score_results))

    except Exception as err:
        # Task failed - all scores should be nan, scoring errors are caught in _run_scorer
        logging.warning("[Statsig] Task failed: %s | input=%r", err, data_record.input)
        eval_result.error = True
        eval_result.scores = {name: 0.0 for name in scorers.keys()}

    return eval_result


def Eval(
    name: str,
    *,
    data: EvalData[Input, Output],
    task: Union[EvalTask[Input, Output]],
    scorer: EvalScorer[Input, Output],
    parameters: Optional[EvalParameters] = None,
    eval_run_name: Optional[str] = None,
    summary_score_fn: Optional[SummaryScorerFn[Input, Output]] = None,
) -> EvalResult[Input, Output]:
    """
    Run evaluation synchronously with support for async tasks, scorers, and data.

    Args:
        name: Name of the evaluation
        data: Evaluation data (iterable of records, can be async)
        task: Task function(s) to run on each input (can be async, can be list)
        scorer: Scorer function(s) to evaluate outputs (can be async, can be dict)
        parameters: Optional parameters to pass to tasks
        eval_run_name: Optional name for this eval run
        summary_score_fn: Optional function that takes a list of results and returns a dictionary of summary scores

    Returns:
        Dictionary with 'results' and 'metadata' keys
    """
    api_key = os.environ.get("STATSIG_API_KEY")
    if not api_key:
        raise RuntimeError("[Statsig] Missing STATSIG_API_KEY environment variable")

    data_iterator: AsyncIterator[EvalDataRecord[Input, Output]] = _normalize_into_data_iterator(
        data
    )
    normalized_scorers: ScorerFnMap[Input, Output] = _normalize_scorers(scorer)

    async def run_all():
        tasks_to_run = []
        async for record in data_iterator:
            tasks_to_run.append(
                _run_task_and_score_one_record(task, normalized_scorers, record, parameters)
            )
        return await asyncio.gather(*tasks_to_run)

    results = asyncio.run(run_all())

    any_error = any(r.error for r in results)

    summary_scores = _run_summary_scorer(summary_score_fn, results)

    _send_eval_results(name, results, api_key, eval_run_name, summary_scores)

    return EvalResult(
        results=results,
        metadata=EvalResultMetadata(error=any_error),
        summary_scores=summary_scores,
    )
