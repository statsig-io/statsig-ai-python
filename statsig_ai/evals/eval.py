import asyncio
import inspect
import logging
import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
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

import json
import requests
from tqdm.auto import tqdm

from .eval_types import (
    EvalDataRecord,
    EvalHook,
    EvalResult,
    EvalResultMetadata,
    EvalResultRecordWithMetadata,
    EvalScorerArgs,
    Input,
    ScorerFnMap,
    Scorer,
    Output,
    Score,
    ScoreWithMetadata,
    EvalData,
    EvalTask,
    EvalScorer,
    EvalParameters,
    _DataclassOrDictEvalDataRecord,
    EvalResultRecord,
    SummaryScorerFn,
)


T = TypeVar("T")

_statsig_async_context_warning_shown = False


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
    if isinstance(score_value, ScoreWithMetadata):
        # Extract the score from ScoreWithMetadata
        return _normalize_score_value(score_value.score)
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
    records: List[EvalResultRecordWithMetadata[Input, Output]],
    api_key: str,
    eval_run_name: Optional[str],
    summary_scores: Optional[Dict[str, float]],
    parameters: Optional[EvalParameters],
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
    if parameters:
        payload["parameters"] = {
            param: value if isinstance(value, str) else json.dumps(value)
            for param, value in parameters.items()
        }
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

    try:
        if inspect.iscoroutinefunction(task):
            return await task(*args)

        event_loop = asyncio.get_event_loop()

        def sync_call() -> Output:
            return task(*args)  # type: ignore

        return await event_loop.run_in_executor(None, sync_call)
    except Exception as err:
        logging.warning("[Statsig] Task failed: %s | input=%r", err, input_value)
        return cast(Output, "Error")


async def _run_scorer(
    scorer_name: str,
    scorer: Scorer[Input, Output],
    output: Output,
    expected: Optional[Output],
    input_value: Input,
) -> ScoreWithMetadata:
    """Run a single scorer and return ScoreWithMetadata."""
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

        if isinstance(raw_score, ScoreWithMetadata):
            return ScoreWithMetadata(
                score=_normalize_score_value(raw_score.score),
                metadata=raw_score.metadata
            )

        if isinstance(raw_score, dict):
            if "score" not in raw_score:
                raise ValueError(
                    f"Scorer '{scorer_name}' returned a dict without a 'score' key. "
                    f"Expected dict with {{'score': float, 'metadata': dict}} or a numeric value (int/float/bool)."
                )

            valid_keys = {"score", "metadata"}
            invalid_keys = set(raw_score.keys()) - valid_keys
            if invalid_keys:
                raise ValueError(
                    f"Scorer '{scorer_name}' returned a dict with invalid keys: {invalid_keys}. "
                    f"Only 'score' and 'metadata' keys are allowed"
                )

            return ScoreWithMetadata(
                score=_normalize_score_value(raw_score.get("score", 0.0)),
                metadata=raw_score.get("metadata")
            )

        if isinstance(raw_score, (int, float, bool)):
            return ScoreWithMetadata(score=_normalize_score_value(raw_score), metadata=None)

        raise TypeError(
            f"Scorer '{scorer_name}' returned invalid type: {type(raw_score).__name__}. "
            f"Expected one of: int, float, bool, dict with 'score' key, or ScoreWithMetadata object."
        )
    except Exception as err:
        logging.warning(
            "[Statsig] Scorer '%s' failed: %s | input=%r",
            scorer_name,
            err,
            input_value,
        )
        return ScoreWithMetadata(score=0.0, metadata=None)


async def _run_task_and_score_one_record(
    task: EvalTask[Input, Output],
    scorers: ScorerFnMap[Input, Output],
    data_record: EvalDataRecord[Input, Output],
    parameters: Optional[EvalParameters],
) -> EvalResultRecordWithMetadata[Input, Output]:
    """Run task and score for one record asynchronously.

    Runs the task first, then runs all scorers concurrently on the output.
    - If task errors: all scores are 0 and error flag is set
    - If a scorer errors: that score is "0" (handled in _run_scorer)
    """
    output: Output = await _run_task(
        task, data_record.input, EvalHook(parameters=parameters, category=data_record.category)
    )

    if output == "Error":
        return EvalResultRecordWithMetadata(
            input=data_record.input,
            expected=data_record.expected,
            category=data_record.category,
            output=output,
            scores={name: ScoreWithMetadata(score=0.0, metadata=None) for name in scorers.keys()},
            error=True,
        )

    scorer_tasks = [
        _run_scorer(scorer_name, scorer_fn, output, data_record.expected, data_record.input)
        for scorer_name, scorer_fn in scorers.items()
    ]
    score_results = await asyncio.gather(*scorer_tasks)

    return EvalResultRecordWithMetadata(
        input=data_record.input,
        expected=data_record.expected,
        category=data_record.category,
        output=output,
        scores=dict(zip(scorers.keys(), score_results)),
        error=False,
    )


def _get_running_event_loop() -> Optional[asyncio.AbstractEventLoop]:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _convert_results_to_results_without_metadata(
    results_with_metadata: List[EvalResultRecordWithMetadata[Input, Output]]
) -> List[EvalResultRecord[Input, Output]]:
    """Convert EvalResultRecordWithMetadata to EvalResultRecord by extracting float scores."""
    results_without_metadata = [EvalResultRecord(
        input=result.input,
        output=result.output,
        scores={name: _normalize_score_value(score.score) for name, score in result.scores.items()},
        expected=result.expected,
        error=result.error,
        category=result.category,
    ) for result in results_with_metadata]
    return results_without_metadata

async def _run_eval_helper(
    name: str,
    data: EvalData[Input, Output],
    task: EvalTask[Input, Output],
    scorer: EvalScorer[Input, Output],
    parameters: Optional[EvalParameters],
    eval_run_name: Optional[str],
    summary_score_fn: Optional[SummaryScorerFn[Input, Output]],
) -> EvalResult[Input, Output]:
    """
    Core eval logic that runs asynchronously.

    This helper function contains the common evaluation logic used by both
    Eval (sync) and EvalAsync (async) functions.
    """
    api_key = os.environ.get("STATSIG_API_KEY")
    if not api_key:
        raise RuntimeError("[Statsig] Missing STATSIG_API_KEY environment variable")

    logging.info("[Statsig] Running eval: %s", name)

    data_iterator: AsyncIterator[EvalDataRecord[Input, Output]] = _normalize_into_data_iterator(
        data
    )
    normalized_scorers: ScorerFnMap[Input, Output] = _normalize_scorers(scorer)

    records: List[EvalDataRecord[Input, Output]] = []
    async for record in data_iterator:
        records.append(record)

    tasks: List[asyncio.Task[EvalResultRecordWithMetadata[Input, Output]]] = []

    for record in records:
        tasks.append(
            asyncio.create_task(
                _run_task_and_score_one_record(task, normalized_scorers, record, parameters)
            )
        )

    results: List[EvalResultRecordWithMetadata[Input, Output]] = []
    for completed_task in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"Running eval: {name}",
    ):
        results.append(await completed_task)

    any_error = any(r.error for r in results)

    results_without_metadata = _convert_results_to_results_without_metadata(results)

    summary_scores = _run_summary_scorer(summary_score_fn, results_without_metadata)

    _send_eval_results(name, results, api_key, eval_run_name, summary_scores, parameters)

    return EvalResult(
        results=results_without_metadata,
        metadata=EvalResultMetadata(error=any_error),
        summary_scores=summary_scores,
    )


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
        task: Task function(s) to run on each input (can be async)
        scorer: Scorer function(s) to evaluate outputs (can be async, can be dict)
        parameters: Optional parameters to pass to tasks
        eval_run_name: Optional name for this eval run
        summary_score_fn: Optional function that takes a list of results and returns a dictionary of summary scores

    Returns:
        EvalResult containing results, metadata, and optional summary_scores

    Warning:
        If called from within an async context, this will run in a separate thread.
        Consider using EvalAsync instead for better performance and compatibility.
    """
    global _statsig_async_context_warning_shown

    running_event_loop = _get_running_event_loop()

    if running_event_loop is not None:
        # We're in an async context - need to run in a separate thread with a new event loop
        if not _statsig_async_context_warning_shown:
            logging.warning(
                "[Statsig] Eval() called from within an async context. "
                "Consider using EvalAsync() instead for better performance and to avoid event loop conflicts.",
            )
            _statsig_async_context_warning_shown = True

        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    _run_eval_helper(
                        name, data, task, scorer, parameters, eval_run_name, summary_score_fn
                    )
                )
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_new_loop)
            return future.result()

    else:
        return asyncio.run(
            _run_eval_helper(name, data, task, scorer, parameters, eval_run_name, summary_score_fn)
        )


async def EvalAsync(
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
    Run evaluation asynchronously with support for async tasks, scorers, and data.

    Args:
        name: Name of the evaluation
        data: Evaluation data (iterable of records, can be async)
        task: Task function(s) to run on each input (can be async)
        scorer: Scorer function(s) to evaluate outputs (can be async, can be dict)
        parameters: Optional parameters to pass to tasks
        eval_run_name: Optional name for this eval run
        summary_score_fn: Optional function that takes a list of results and returns a dictionary of summary scores

    Returns:
        EvalResult containing results, metadata, and optional summary_scores

    Note:
        This is the async version of Eval(). Use this when calling from within
        an async context for better performance and to avoid event loop conflicts.
    """
    return await _run_eval_helper(
        name, data, task, scorer, parameters, eval_run_name, summary_score_fn
    )
