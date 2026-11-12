import asyncio
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import Field
from pytest_httpserver import HTTPServer
from statsig_ai import (
    Eval,
    EvalScorerArgs,
    EvalDataRecord,
    EvalHook,
    StatsigDatasetRow,
    StatsigSpanProcessor,
    StatsigUser,
    statsig_dataset,
    statsig_user_context,
    wrap,
    wrap_openai,
)
from statsig_ai.evals import EvalAsync
from statsig_ai.evals import eval as eval_module
from statsig_ai.evals.eval_types import ScoreWithMetadata
from statsig_ai.otel.conventions import (
    STATSIG_ATTR_ENVIRONMENT,
    STATSIG_ATTR_EVAL_NAME,
    STATSIG_ATTR_EVAL_RECORD_INDEX,
    STATSIG_ATTR_EVAL_RUN_NAME,
    STATSIG_ATTR_EVAL_SCORER_NAME,
    STATSIG_ATTR_GEN_AI_SPAN_TYPE,
    STATSIG_ATTR_SPAN_LLM_ROOT,
    STATSIG_ATTR_SPAN_TYPE,
    STATSIG_ATTR_USER_ID,
    STATSIG_SPAN_LLM_ROOT_VALUE,
    StatsigGenAISpanType,
    StatsigSpanType,
)
from statsig_ai.otel.environment_context import get_environment_span_attrs_from_context
from statsig_ai.otel.singleton import OtelSingleton
from statsig_ai.wrappers import span_telemetry
from mock_openai import MockOpenAI
from mock_scrapi import MockScrapi


def results_by_input(results):
    result_dict = {}
    for r in results:
        input_val = r.input if hasattr(r, "input") else r["input"]
        if isinstance(input_val, dict):
            key = tuple(sorted(input_val.items()))
        else:
            key = input_val
        result_dict[key] = r
    return result_dict


@pytest.fixture
def eval_setup(httpserver: HTTPServer, monkeypatch):
    from werkzeug import Response
    import re

    mock_scrapi = MockScrapi(httpserver)

    def eval_handler(request):
        if re.match(r"/console/v1/evals/send_results/.+", request.path):
            data = request.get_data()
            import json

            req_json = json.loads(data)
            mock_scrapi.eval_payloads.append(req_json)
            return Response('{"success": true}', status=200)
        return Response("Not Found", status=404)

    httpserver.expect_request(
        re.compile(r"/console/v1/evals/send_results/.+"), method="POST"
    ).respond_with_handler(eval_handler)

    monkeypatch.setenv("STATSIG_API_KEY", "test-api-key")

    import requests

    original_post = requests.post

    def intercepted_post(url, *args, **kwargs):
        if "api.statsig.com/console/v1/evals/send_results/" in url:
            eval_name = url.split("/send_results/")[-1]
            mock_url = (
                f"{mock_scrapi.url_for_endpoint('/console/v1/evals/send_results/')}{eval_name}"
            )
            return original_post(mock_url, *args, **kwargs)
        return original_post(url, *args, **kwargs)

    monkeypatch.setattr(requests, "post", intercepted_post)

    original_get = requests.get
    mock_origin = mock_scrapi.url_for_endpoint("/").rstrip("/")

    def intercepted_get(url, *args, **kwargs):
        if url.startswith("https://api.statsig.com/console/v1/eval_datasets/"):
            return original_get(url.replace("https://api.statsig.com", mock_origin), *args, **kwargs)
        return original_get(url, *args, **kwargs)

    monkeypatch.setattr(requests, "get", intercepted_get)

    yield mock_scrapi


@pytest.fixture(params=["Eval", "EvalAsync"])
def run_eval(request):
    """Runs an eval through Eval() or EvalAsync()."""
    if request.param == "Eval":
        return Eval

    def run_eval_async(name, **kwargs):
        return asyncio.run(EvalAsync(name, **kwargs))

    return run_eval_async


def test_eval_basic_string(eval_setup):
    mock_scrapi = eval_setup

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0 if args.output == args.expected else 0.0

    result = Eval(
        "test_basic_string",
        data=[
            {"input": "world", "expected": "Hello world"},
            {"input": "test", "expected": "Hello test"},
        ],
        task=lambda input: f"Hello {input}",
        scorer=scorer,
    )

    assert len(result.results) == 2
    assert result.metadata.error == False

    results = results_by_input(result.results)
    assert results["world"].output == "Hello world"
    assert results["world"].expected == "Hello world"
    assert results["world"].scores["Grader"] == 1.0
    assert results["test"].output == "Hello test"
    assert results["test"].expected == "Hello test"
    assert results["test"].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    payload_results = results_by_input(payloads[0]["results"])
    assert len(payload_results) == 2
    assert payload_results["world"]["output"] == "Hello world"
    assert payload_results["world"]["expected"] == "Hello world"
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0
    assert payload_results["test"]["output"] == "Hello test"
    assert payload_results["test"]["expected"] == "Hello test"
    assert payload_results["test"]["scores"]["Grader"]["score"] == 1.0


def test_eval_task_with_hook(eval_setup):
    mock_scrapi = eval_setup

    def task_with_hook(input: str, hook: EvalHook) -> str:
        prefix = hook.parameters.get("prefix", "Hello") if hook.parameters else "Hello"
        return f"{prefix} {input}"

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0 if args.output == args.expected else 0.0

    result = Eval(
        "test_task_with_hook",
        data=[
            {"input": "world", "expected": "Hi world"},
        ],
        task=task_with_hook,
        scorer=scorer,
        parameters={"prefix": "Hi"},
    )

    results = results_by_input(result.results)
    assert results["world"].output == "Hi world"
    assert results["world"].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["output"] == "Hi world"
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0
    assert payloads[0]["parameters"] == {"prefix": "Hi"}


def test_eval_multiple_scorers(eval_setup):
    mock_scrapi = eval_setup

    def exact_match_scorer(args: EvalScorerArgs[str, str]) -> bool:
        return args.output == args.expected

    def length_scorer(args: EvalScorerArgs[str, str]) -> int:
        return len(args.output) > 5

    result = Eval(
        "test_multiple_scorers",
        data=[
            {"input": "world", "expected": "Hello world"},
        ],
        task=lambda input: f"Hello {input}",
        scorer={
            "exact_match": exact_match_scorer,
            "length": length_scorer,
        },
    )

    results = results_by_input(result.results)
    assert "exact_match" in results["world"].scores
    assert "length" in results["world"].scores
    assert results["world"].scores["exact_match"] == 1.0
    assert results["world"].scores["length"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["scores"]["exact_match"]["score"] == 1.0
    assert payload_results["world"]["scores"]["length"]["score"] == 1.0


def test_eval_async_task(eval_setup):
    mock_scrapi = eval_setup

    async def async_task(input: str) -> str:
        return f"Async {input}"

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0 if "Async" in args.output else 0.0

    result = Eval(
        "test_async_task",
        data=[
            {"input": "world"},
        ],
        task=async_task,
        scorer=scorer,
    )

    results = results_by_input(result.results)
    assert results["world"].output == "Async world"
    assert results["world"].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["output"] == "Async world"
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0


def test_eval_async_scorer(eval_setup):
    mock_scrapi = eval_setup

    async def async_scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0 if args.output == args.expected else 0.0

    result = Eval(
        "test_async_scorer",
        data=[
            {"input": "world", "expected": "Hello world"},
        ],
        task=lambda input: f"Hello {input}",
        scorer=async_scorer,
    )

    results = results_by_input(result.results)
    assert results["world"].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0


def test_eval_dict_input(eval_setup):
    mock_scrapi = eval_setup

    def scorer(args: EvalScorerArgs[dict, str]) -> float:
        return 1.0 if len(args.output) > 8 else 0.0

    result = Eval(
        "test_dict_input",
        data=[
            {"input": {"name": "Alice", "age": 30}, "expected": "Hello Alice"},
        ],
        task=lambda input: f"Hello {input['name']}",
        scorer=scorer,
    )

    results = results_by_input(result.results)
    dict_key = (("age", 30), ("name", "Alice"))
    assert results[dict_key].input == {"name": "Alice", "age": 30}
    assert results[dict_key].output == "Hello Alice"
    assert results[dict_key].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    payload_results = results_by_input(payloads[0]["results"])
    payload_key = '{"name":"Alice","age":30}'
    assert payload_results[payload_key]["input"] == payload_key
    assert payload_results[payload_key]["output"] == "Hello Alice"
    assert payload_results[payload_key]["scores"]["Grader"]["score"] == 1.0


def test_eval_int_input(eval_setup):
    mock_scrapi = eval_setup

    def scorer(args: EvalScorerArgs[int, int]) -> bool:
        return args.output == args.expected

    result = Eval(
        "test_int_input",
        data=[
            {"input": 5, "expected": 10},
            {"input": 3, "expected": 6},
        ],
        task=lambda input: input * 2,
        scorer=scorer,
    )

    results = results_by_input(result.results)
    assert results[5].input == 5
    assert results[5].output == 10
    assert results[5].scores["Grader"] == 1.0
    assert results[3].input == 3
    assert results[3].output == 6
    assert results[3].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["5"]["input"] == "5"
    assert payload_results["5"]["output"] == "10"
    assert payload_results["5"]["expected"] == "10"
    assert payload_results["5"]["scores"]["Grader"]["score"] == 1.0
    assert payload_results["3"]["input"] == "3"
    assert payload_results["3"]["output"] == "6"
    assert payload_results["3"]["expected"] == "6"
    assert payload_results["3"]["scores"]["Grader"]["score"] == 1.0


def test_eval_dataclass_records(eval_setup):
    mock_scrapi = eval_setup

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0 if len(args.output) > 8 else 0.0

    result = Eval(
        "test_dataclass_records",
        data=[
            EvalDataRecord(input="world", expected="Hello world"),
            EvalDataRecord(input="test", expected="Hello test"),
        ],
        task=lambda input: f"Hello {input}",
        scorer=scorer,
    )

    assert len(result.results) == 2
    results = results_by_input(result.results)
    assert results["world"].scores["Grader"] == 1.0
    assert results["world"].output == "Hello world"
    assert results["test"].scores["Grader"] == 1.0
    assert results["test"].output == "Hello test"

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert len(payloads[0]["results"]) == 2
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0
    assert payload_results["world"]["output"] == "Hello world"
    assert payload_results["test"]["scores"]["Grader"]["score"] == 1.0
    assert payload_results["test"]["output"] == "Hello test"


def test_eval_iterator_data(eval_setup):
    mock_scrapi = eval_setup

    def data_generator():
        yield {"input": "world", "expected": "Hello world"}
        yield {"input": "test", "expected": "Hello test"}

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0

    result = Eval(
        "test_iterator_data",
        data=data_generator(),
        task=lambda input: f"Hello {input}",
        scorer=scorer,
    )

    assert len(result.results) == 2
    results = results_by_input(result.results)
    assert results["world"].input == "world"
    assert results["world"].output == "Hello world"
    assert results["world"].scores["Grader"] == 1.0
    assert results["test"].input == "test"
    assert results["test"].output == "Hello test"
    assert results["test"].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert len(payloads[0]["results"]) == 2
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["input"] == "world"
    assert payload_results["world"]["output"] == "Hello world"
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0
    assert payload_results["test"]["input"] == "test"
    assert payload_results["test"]["output"] == "Hello test"
    assert payload_results["test"]["scores"]["Grader"]["score"] == 1.0


def test_eval_callable_data(eval_setup):
    mock_scrapi = eval_setup

    def data_func():
        return [
            {"input": "world", "expected": "Hello world"},
            {"input": "test", "expected": "Hello test"},
        ]

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0

    result = Eval(
        "test_callable_data",
        data=data_func,
        task=lambda input: f"Hello {input}",
        scorer=scorer,
    )

    assert len(result.results) == 2
    results = results_by_input(result.results)
    assert results["world"].input == "world"
    assert results["world"].output == "Hello world"
    assert results["world"].scores["Grader"] == 1.0
    assert results["test"].input == "test"
    assert results["test"].output == "Hello test"
    assert results["test"].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert len(payloads[0]["results"]) == 2
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["input"] == "world"
    assert payload_results["world"]["output"] == "Hello world"
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0
    assert payload_results["test"]["input"] == "test"
    assert payload_results["test"]["output"] == "Hello test"
    assert payload_results["test"]["scores"]["Grader"]["score"] == 1.0


def test_eval_async_callable_data(eval_setup):
    mock_scrapi = eval_setup

    async def async_data_func():
        return [
            {"input": "world", "expected": "Hello world"},
            {"input": "test", "expected": "Hello test"},
        ]

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0

    result = Eval(
        "test_async_callable_data",
        data=async_data_func,
        task=lambda input: f"Hello {input}",
        scorer=scorer,
    )

    assert len(result.results) == 2
    results = results_by_input(result.results)
    assert results["world"].input == "world"
    assert results["world"].output == "Hello world"
    assert results["world"].scores["Grader"] == 1.0
    assert results["test"].input == "test"
    assert results["test"].output == "Hello test"
    assert results["test"].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert len(payloads[0]["results"]) == 2
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["input"] == "world"
    assert payload_results["world"]["output"] == "Hello world"
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0
    assert payload_results["test"]["input"] == "test"
    assert payload_results["test"]["output"] == "Hello test"
    assert payload_results["test"]["scores"]["Grader"]["score"] == 1.0


def test_eval_async_iterator_data(eval_setup):
    mock_scrapi = eval_setup

    async def async_data_generator():
        yield {"input": "world", "expected": "Hello world"}
        yield {"input": "test", "expected": "Hello test"}

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0

    result = Eval(
        "test_async_iterator_data",
        data=async_data_generator(),
        task=lambda input: f"Hello {input}",
        scorer=scorer,
    )

    assert len(result.results) == 2
    results = results_by_input(result.results)
    assert results["world"].input == "world"
    assert results["world"].output == "Hello world"
    assert results["world"].scores["Grader"] == 1.0
    assert results["test"].input == "test"
    assert results["test"].output == "Hello test"
    assert results["test"].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert len(payloads[0]["results"]) == 2
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["input"] == "world"
    assert payload_results["world"]["output"] == "Hello world"
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0
    assert payload_results["test"]["input"] == "test"
    assert payload_results["test"]["output"] == "Hello test"
    assert payload_results["test"]["scores"]["Grader"]["score"] == 1.0


def test_eval_with_categories(eval_setup):
    mock_scrapi = eval_setup

    def task_with_hook(input: str, hook: EvalHook) -> str:
        return f"Hello {input}"

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0

    result = Eval(
        "test_with_categories",
        data=[
            {"input": "world", "expected": "Hello world", "category": "greeting"},
            {"input": "test", "expected": "Hello test", "category": ["greeting", "test"]},
        ],
        task=task_with_hook,
        scorer=scorer,
    )

    results = results_by_input(result.results)
    assert results["world"].category == "greeting"
    assert results["test"].category == ["greeting", "test"]

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["category"] == "greeting"
    assert payload_results["test"]["category"] == ["greeting", "test"]


def test_eval_without_expected(eval_setup):
    mock_scrapi = eval_setup

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0 if len(args.output) > 8 else 0.0

    result = Eval(
        "test_without_expected",
        data=[
            {"input": "world"},
            {"input": "hi"},
        ],
        task=lambda input: f"Hello {input}",
        scorer=scorer,
    )

    results = results_by_input(result.results)
    assert results["world"].expected is None
    assert results["world"].scores["Grader"] == 1.0
    assert results["hi"].expected is None
    assert results["hi"].scores["Grader"] == 0.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"].get("expected") is None
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0
    assert payload_results["hi"].get("expected") is None
    assert payload_results["hi"]["scores"]["Grader"]["score"] == 0.0


def test_eval_with_run_name(eval_setup):
    mock_scrapi = eval_setup

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0

    result = Eval(
        "test_with_run_name",
        data=[
            {"input": "world", "expected": "Hello world"},
        ],
        task=lambda input: f"Hello {input}",
        scorer=scorer,
        eval_run_name="test_run_123",
    )

    results = results_by_input(result.results)
    assert results["world"].input == "world"
    assert results["world"].output == "Hello world"
    assert results["world"].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert payloads[0].get("name") == "test_run_123"
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["input"] == "world"
    assert payload_results["world"]["output"] == "Hello world"
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0


def test_eval_task_error(eval_setup):
    mock_scrapi = eval_setup

    def failing_task(input: str) -> str:
        if input == "fail":
            raise ValueError("Task failed")
        return f"Hello {input}"

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0

    result = Eval(
        "test_task_error",
        data=[
            {"input": "world", "expected": "Hello world"},
            {"input": "fail", "expected": "Error"},
        ],
        task=failing_task,
        scorer=scorer,
    )

    results = results_by_input(result.results)
    assert results["world"].output == "Hello world"
    assert results["world"].error == False
    assert results["fail"].output == "[Error]"
    assert results["fail"].error == True
    assert results["fail"].scores["Grader"] == 0.0
    assert result.metadata.error == True

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1

    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["output"] == "Hello world"
    assert payload_results["world"]["error"] == False
    assert payload_results["fail"]["error"] == True
    assert payload_results["fail"]["scores"]["Grader"]["score"] == 0.0


def test_eval_scorer_error(eval_setup):
    mock_scrapi = eval_setup

    def failing_scorer(args: EvalScorerArgs[str, str]) -> float:
        if args.input == "fail":
            raise ValueError("Scorer failed")
        return 1.0

    result = Eval(
        "test_scorer_error",
        data=[
            {"input": "world", "expected": "Hello world"},
            {"input": "fail", "expected": "Hello fail"},
        ],
        task=lambda input: f"Hello {input}",
        scorer=failing_scorer,
    )

    results = results_by_input(result.results)
    print("DATA", results)
    assert results["world"].scores["Grader"] == 1.0
    assert results["fail"].scores["Grader"] == 0.0
    assert results["fail"].error == False

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0
    assert payload_results["fail"]["scores"]["Grader"]["score"] == 0.0
    assert payload_results["fail"]["error"] == False


def test_eval_boolean_scorer(eval_setup):
    mock_scrapi = eval_setup

    def bool_scorer(args: EvalScorerArgs[str, str]) -> bool:
        return args.output == args.expected

    result = Eval(
        "test_boolean_scorer",
        data=[
            {"input": "world", "expected": "Hello world"},
            {"input": "test", "expected": "Goodbye test"},
        ],
        task=lambda input: f"Hello {input}",
        scorer=bool_scorer,
    )

    results = results_by_input(result.results)
    assert results["world"].scores["Grader"] == 1.0
    assert results["test"].scores["Grader"] == 0.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0
    assert payload_results["test"]["scores"]["Grader"]["score"] == 0.0


def test_eval_scorer_with_metadata(eval_setup):
    mock_scrapi = eval_setup

    def metadata_scorer(args: EvalScorerArgs[str, str]) -> dict:
        if args.output == args.expected:
            return {
                "score": 1.0,
                "metadata": {"justification": "exact match", "confidence": "high"},
            }
        else:
            return {"score": 0.0, "metadata": {"justification": "no match"}}

    result = Eval(
        "test_scorer_metadata",
        data=[
            {"input": "world", "expected": "Hello world"},
            {"input": "test", "expected": "Goodbye test"},
        ],
        task=lambda input: f"Hello {input}",
        scorer=metadata_scorer,
    )

    results = results_by_input(result.results)

    # Check that scores are ScoreWithMetadata objects with correct values
    world_score = results["world"].scores["Grader"]
    assert world_score == 1.0

    test_score = results["test"].scores["Grader"]
    assert test_score == 0.0

    # Check that the payload includes metadata
    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    payload_results = results_by_input(payloads[0]["results"])

    world_payload_score = payload_results["world"]["scores"]["Grader"]
    assert world_payload_score["score"] == 1.0
    assert world_payload_score["metadata"] == {"justification": "exact match", "confidence": "high"}

    test_payload_score = payload_results["test"]["scores"]["Grader"]
    assert test_payload_score["score"] == 0.0
    assert test_payload_score["metadata"] == {"justification": "no match"}


def test_eval_scorer_invalid_dict_without_score_key(eval_setup):
    mock_scrapi = eval_setup

    def invalid_scorer(args: EvalScorerArgs[str, str]) -> dict:
        # Return dict without 'score' key - should fail
        return {"metadata": {"reason": "missing score key"}}

    result = Eval(
        "test_invalid_scorer_dict",
        data=[
            {"input": "world", "expected": "Hello world"},
        ],
        task=lambda input: f"Hello {input}",
        scorer=invalid_scorer,
    )

    results = results_by_input(result.results)
    # Should return 0.0 score due to error handling
    world_score = results["world"].scores["Grader"]
    assert world_score == 0.0


def test_eval_scorer_invalid_dict_with_typo(eval_setup):
    mock_scrapi = eval_setup

    def typo_scorer(args: EvalScorerArgs[str, str]) -> dict:
        # Return dict with typo in 'metadata' key - should fail
        return {"score": 0.5, "metadatwa": {"justification": "infra stuff"}}

    result = Eval(
        "test_invalid_scorer_typo",
        data=[
            {"input": "world", "expected": "Hello world"},
        ],
        task=lambda input: f"Hello {input}",
        scorer=typo_scorer,
    )

    results = results_by_input(result.results)
    # Should return 0.0 score due to error handling
    world_score = results["world"].scores["Grader"]
    assert world_score == 0.0


def test_eval_scorer_invalid_return_type(eval_setup):
    mock_scrapi = eval_setup

    def invalid_scorer(args: EvalScorerArgs[str, str]) -> str:
        # Return string - invalid type
        return "invalid"

    result = Eval(
        "test_invalid_scorer_type",
        data=[
            {"input": "world", "expected": "Hello world"},
        ],
        task=lambda input: f"Hello {input}",
        scorer=invalid_scorer,
    )

    results = results_by_input(result.results)
    # Should return 0.0 score due to error handling
    world_score = results["world"].scores["Grader"]
    assert world_score == 0.0


def test_eval_class_based_scorer(eval_setup):
    mock_scrapi = eval_setup

    class MyScorer:
        def __call__(self, args: EvalScorerArgs[str, str]) -> float:
            return 1.0 if args.output == args.expected else 0.0

    result = Eval(
        "test_class_scorer",
        data=[
            {"input": "world", "expected": "Hello world"},
        ],
        task=lambda input: f"Hello {input}",
        scorer=MyScorer,
    )

    results = results_by_input(result.results)
    world_score = results["world"].scores["Grader"]
    assert world_score == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0


def test_eval_with_parameters(eval_setup):
    mock_scrapi = eval_setup

    def task_with_params(input: str, hook: EvalHook) -> str:
        prefix = hook.parameters.get("prefix", "Hello")
        suffix = hook.parameters.get("suffix", "!")
        return f"{prefix} {input}{suffix}"

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0 if args.output == args.expected else 0.0

    result = Eval(
        "test_with_parameters",
        data=[
            {"input": "world", "expected": "Hi world!"},
        ],
        task=task_with_params,
        scorer=scorer,
        parameters={"prefix": "Hi", "suffix": "!", "number": 123, "object": {"a": "b"}},
    )

    results = results_by_input(result.results)
    assert results["world"].output == "Hi world!"
    assert results["world"].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["output"] == "Hi world!"
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0
    assert payloads[0]["parameters"] == {
        "prefix": "Hi",
        "suffix": "!",
        "number": "123",
        "object": '{"a":"b"}',
    }


def test_eval_missing_api_key(httpserver: HTTPServer, monkeypatch):
    monkeypatch.delenv("STATSIG_API_KEY", raising=False)

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0

    with pytest.raises(RuntimeError, match="Missing STATSIG_API_KEY"):
        Eval(
            "test_missing_api_key",
            data=[{"input": "world"}],
            task=lambda input: f"Hello {input}",
            scorer=scorer,
        )


def test_eval_with_summary_scorer(eval_setup):
    mock_scrapi = eval_setup

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0 if args.output == args.expected else 0.0

    def summary_scorer(results):
        total_score = 0.0
        count = 0
        for result in results:
            for score_value in result.scores.values():
                total_score += score_value
                count += 1
        avg_score = total_score / count if count > 0 else 0.0

        pass_count = sum(1 for r in results if all(s == 1.0 for s in r.scores.values()))
        pass_rate = pass_count / len(results) if results else 0.0

        return {
            "average_score": avg_score,
            "pass_rate": pass_rate,
            "total_count": len(results),
        }

    result = Eval(
        "test_with_summary_scorer",
        data=[
            {"input": "world", "expected": "Hello world"},
            {"input": "test", "expected": "Hello test"},
            {"input": "foo", "expected": "Goodbye foo"},
        ],
        task=lambda input: f"Hello {input}",
        scorer=scorer,
        summary_score_fn=summary_scorer,
    )

    assert result.summary_scores is not None
    assert result.summary_scores["average_score"] == pytest.approx(2.0 / 3.0)
    assert result.summary_scores["pass_rate"] == pytest.approx(2.0 / 3.0)
    assert result.summary_scores["total_count"] == 3

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert "summaryScores" in payloads[0]
    assert payloads[0]["summaryScores"]["average_score"] == pytest.approx(2.0 / 3.0)
    assert payloads[0]["summaryScores"]["pass_rate"] == pytest.approx(2.0 / 3.0)
    assert payloads[0]["summaryScores"]["total_count"] == 3


def test_eval_without_summary_scorer(eval_setup):
    mock_scrapi = eval_setup

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0

    result = Eval(
        "test_without_summary_scorer",
        data=[
            {"input": "world", "expected": "Hello world"},
        ],
        task=lambda input: f"Hello {input}",
        scorer=scorer,
    )

    assert result.summary_scores is None
    results = results_by_input(result.results)
    assert results["world"].input == "world"
    assert results["world"].output == "Hello world"
    assert results["world"].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert "summaryScores" not in payloads[0]
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["input"] == "world"
    assert payload_results["world"]["output"] == "Hello world"
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0


def test_eval_summary_scorer_with_multiple_scorers(eval_setup):
    mock_scrapi = eval_setup

    def exact_match_scorer(args: EvalScorerArgs[str, str]) -> bool:
        return args.output == args.expected

    def length_scorer(args: EvalScorerArgs[str, str]) -> int:
        return len(args.output) > 5

    def summary_scorer(results):
        exact_match_scores = []
        length_scores = []

        for result in results:
            exact_match_score = result.scores.get("exact_match")
            exact_match_scores.append(exact_match_score if exact_match_score else 0.0)
            length_score = result.scores.get("length")
            length_scores.append(length_score if length_score else 0.0)

        return {
            "avg_exact_match": sum(exact_match_scores) / len(exact_match_scores),
            "avg_length": sum(length_scores) / len(length_scores),
            "min_exact_match": min(exact_match_scores),
            "max_length": max(length_scores),
        }

    result = Eval(
        "test_summary_scorer_multiple_scorers",
        data=[
            {"input": "world", "expected": "Hello world"},
            {"input": "test", "expected": "Hello test"},
            {"input": "a", "expected": "Hello b"},
        ],
        task=lambda input: f"Hello {input}",
        scorer={
            "exact_match": exact_match_scorer,
            "length": length_scorer,
        },
        summary_score_fn=summary_scorer,
    )

    results = results_by_input(result.results)
    assert results["world"].input == "world"
    assert results["world"].output == "Hello world"
    assert results["world"].scores["exact_match"] == 1.0
    assert results["world"].scores["length"] == 1.0
    assert results["a"].input == "a"
    assert results["a"].output == "Hello a"
    assert results["a"].scores["exact_match"] == 0.0
    assert results["a"].scores["length"] == 1.0

    assert result.summary_scores is not None
    assert result.summary_scores["avg_exact_match"] == pytest.approx(2.0 / 3.0)
    assert result.summary_scores["avg_length"] == 1.0
    assert result.summary_scores["min_exact_match"] == 0.0
    assert result.summary_scores["max_length"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert "summaryScores" in payloads[0]
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["input"] == "world"
    assert payload_results["world"]["output"] == "Hello world"
    assert payload_results["world"]["scores"]["exact_match"]["score"] == 1.0
    assert payload_results["world"]["scores"]["length"]["score"] == 1.0
    assert payloads[0]["summaryScores"]["avg_exact_match"] == pytest.approx(2.0 / 3.0)
    assert payloads[0]["summaryScores"]["avg_length"] == 1.0
    assert payloads[0]["summaryScores"]["min_exact_match"] == 0.0
    assert payloads[0]["summaryScores"]["max_length"] == 1.0


def test_eval_summary_scorer_error_handling(eval_setup):
    mock_scrapi = eval_setup

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0

    def failing_summary_scorer(results):
        raise ValueError("Summary scorer failed")

    result = Eval(
        "test_summary_scorer_error",
        data=[
            {"input": "world", "expected": "Hello world"},
        ],
        task=lambda input: f"Hello {input}",
        scorer=scorer,
        summary_score_fn=failing_summary_scorer,
    )

    assert result.summary_scores is None
    results = results_by_input(result.results)
    assert results["world"].input == "world"
    assert results["world"].output == "Hello world"
    assert results["world"].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert "results" in payloads[0]
    assert len(payloads[0]["results"]) == 1
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["input"] == "world"
    assert payload_results["world"]["output"] == "Hello world"
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0
    assert "summaryScores" not in payloads[0]


def test_eval_summary_scorer_with_categories(eval_setup):
    mock_scrapi = eval_setup

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0 if args.output == args.expected else 0.0

    def summary_scorer(results):
        category_scores = {}
        for result in results:
            category = result.category if result.category else "uncategorized"
            if isinstance(category, list):
                category = "_".join(category)

            if category not in category_scores:
                category_scores[category] = []

            category_scores[category].append(
                sum(s for s in result.scores.values()) / len(result.scores)
                if result.scores
                else 0.0
            )

        summary = {}
        for category, scores in category_scores.items():
            summary[f"{category}_avg"] = sum(scores) / len(scores)

        return summary

    result = Eval(
        "test_summary_scorer_categories",
        data=[
            {"input": "world", "expected": "Hello world", "category": "greeting"},
            {"input": "test", "expected": "Hello test", "category": "greeting"},
            {"input": "foo", "expected": "Goodbye foo", "category": "farewell"},
        ],
        task=lambda input: f"Hello {input}",
        scorer=scorer,
        summary_score_fn=summary_scorer,
    )

    results = results_by_input(result.results)
    assert results["world"].input == "world"
    assert results["world"].output == "Hello world"
    assert results["world"].category == "greeting"
    assert results["world"].scores["Grader"] == 1.0
    assert results["foo"].input == "foo"
    assert results["foo"].output == "Hello foo"
    assert results["foo"].category == "farewell"
    assert results["foo"].scores["Grader"] == 0.0

    assert result.summary_scores is not None
    assert "greeting_avg" in result.summary_scores
    assert "farewell_avg" in result.summary_scores
    assert result.summary_scores["greeting_avg"] == 1.0
    assert result.summary_scores["farewell_avg"] == 0.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert "summaryScores" in payloads[0]
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["input"] == "world"
    assert payload_results["world"]["output"] == "Hello world"
    assert payload_results["world"]["category"] == "greeting"
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0
    assert payload_results["foo"]["input"] == "foo"
    assert payload_results["foo"]["output"] == "Hello foo"
    assert payload_results["foo"]["scores"]["Grader"]["score"] == 0.0
    assert "greeting_avg" in payloads[0]["summaryScores"]
    assert "farewell_avg" in payloads[0]["summaryScores"]
    assert payloads[0]["summaryScores"]["greeting_avg"] == 1.0
    assert payloads[0]["summaryScores"]["farewell_avg"] == 0.0


def test_eval_summary_scorer_with_empty_results(eval_setup):
    mock_scrapi = eval_setup

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0

    def summary_scorer(results):
        if not results:
            return {"count": 0, "message": "no_results"}
        return {"count": len(results)}

    result = Eval(
        "test_summary_scorer_empty",
        data=[],
        task=lambda input: f"Hello {input}",
        scorer=scorer,
        summary_score_fn=summary_scorer,
    )

    assert len(result.results) == 0
    assert result.summary_scores is not None
    assert result.summary_scores["count"] == 0
    assert result.summary_scores["message"] == "no_results"

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert len(payloads[0]["results"]) == 0
    assert "summaryScores" in payloads[0]
    assert payloads[0]["summaryScores"]["count"] == 0
    assert payloads[0]["summaryScores"]["message"] == "no_results"


def test_eval_summary_scorer_with_task_errors(eval_setup):
    mock_scrapi = eval_setup

    def failing_task(input: str) -> str:
        if input == "fail":
            raise ValueError("Task failed")
        return f"Hello {input}"

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0

    def summary_scorer(results):
        error_count = sum(1 for r in results if r.error)
        success_count = len(results) - error_count

        total_score = sum(
            sum(s for s in r.scores.values()) / len(r.scores) if r.scores else 0.0 for r in results
        )
        avg_score = total_score / len(results) if results else 0.0

        success_scores = [
            sum(s for s in r.scores.values()) / len(r.scores)
            for r in results
            if not r.error and r.scores
        ]
        avg_success_score = sum(success_scores) / len(success_scores) if success_scores else 0.0

        return {
            "error_count": error_count,
            "success_count": success_count,
            "avg_score": avg_score,
            "avg_success_score": avg_success_score,
        }

    result = Eval(
        "test_summary_scorer_task_errors",
        data=[
            {"input": "world", "expected": "Hello world"},
            {"input": "fail", "expected": "Hello fail"},
            {"input": "test", "expected": "Hello test"},
        ],
        task=failing_task,
        scorer=scorer,
        summary_score_fn=summary_scorer,
    )

    results = results_by_input(result.results)
    assert results["world"].input == "world"
    assert results["world"].output == "Hello world"
    assert results["world"].error == False
    assert results["world"].scores["Grader"] == 1.0
    assert results["fail"].input == "fail"
    assert results["fail"].error == True
    assert results["fail"].scores["Grader"] == 0.0
    assert results["test"].input == "test"
    assert results["test"].output == "Hello test"
    assert results["test"].error == False
    assert results["test"].scores["Grader"] == 1.0

    assert result.summary_scores is not None
    assert result.summary_scores["error_count"] == 1
    assert result.summary_scores["success_count"] == 2
    assert result.summary_scores["avg_score"] == pytest.approx(2.0 / 3.0)
    assert result.summary_scores["avg_success_score"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert "summaryScores" in payloads[0]
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["input"] == "world"
    assert payload_results["world"]["output"] == "Hello world"
    assert payload_results["world"]["error"] == False
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0
    assert payload_results["fail"]["input"] == "fail"
    assert payload_results["fail"]["error"] == True
    assert payload_results["fail"]["scores"]["Grader"]["score"] == 0.0
    assert payload_results["test"]["input"] == "test"
    assert payload_results["test"]["output"] == "Hello test"
    assert payload_results["test"]["scores"]["Grader"]["score"] == 1.0
    assert payloads[0]["summaryScores"]["error_count"] == 1
    assert payloads[0]["summaryScores"]["success_count"] == 2
    assert payloads[0]["summaryScores"]["avg_score"] == pytest.approx(2.0 / 3.0)
    assert payloads[0]["summaryScores"]["avg_success_score"] == 1.0


def test_eval_summary_scorer_throws_during_computation(eval_setup):
    mock_scrapi = eval_setup

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        return 1.0 if args.output == args.expected else 0.0

    def failing_summary_scorer(results):
        total = sum(r.scores.get("Grader", 0.0) for r in results)
        raise RuntimeError("Failed during summary computation")

    result = Eval(
        "test_summary_scorer_throws",
        data=[
            {"input": "world", "expected": "Hello world"},
            {"input": "test", "expected": "Hello test"},
        ],
        task=lambda input: f"Hello {input}",
        scorer=scorer,
        summary_score_fn=failing_summary_scorer,
    )

    results = results_by_input(result.results)
    assert results["world"].input == "world"
    assert results["world"].output == "Hello world"
    assert results["world"].scores["Grader"] == 1.0
    assert results["test"].input == "test"
    assert results["test"].output == "Hello test"
    assert results["test"].scores["Grader"] == 1.0
    assert result.summary_scores is None

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert len(payloads[0]["results"]) == 2
    payload_results = results_by_input(payloads[0]["results"])
    assert payload_results["world"]["input"] == "world"
    assert payload_results["world"]["output"] == "Hello world"
    assert payload_results["world"]["scores"]["Grader"]["score"] == 1.0
    assert payload_results["test"]["input"] == "test"
    assert payload_results["test"]["output"] == "Hello test"
    assert payload_results["test"]["scores"]["Grader"]["score"] == 1.0
    assert "summaryScores" not in payloads[0]


def _sync_task(fn):
    return fn


def _async_task(fn):
    async def task(input):
        return fn(input)

    return task


@pytest.mark.parametrize("wrap_task", [_sync_task, _async_task], ids=["sync", "async"])
def test_eval_task_returning_error_string_is_not_a_failure(eval_setup, run_eval, wrap_task):
    mock_scrapi = eval_setup

    result = run_eval(
        "test_task_returns_error_string",
        data=[{"input": "world", "expected": "Error"}],
        task=wrap_task(lambda input: "Error"),
        scorer=lambda args: args.output == args.expected,
    )

    record = result.results[0]
    assert record.output == "Error"
    assert record.error == False
    assert record.scores == {"Grader": 1.0}
    assert result.metadata.error == False

    payload_record = mock_scrapi.get_eval_payloads()[0]["results"][0]
    assert payload_record["output"] == "Error"
    assert payload_record["error"] == False
    assert payload_record["scores"] == {"Grader": {"score": 1.0}}


@pytest.mark.parametrize("wrap_task", [_sync_task, _async_task], ids=["sync", "async"])
def test_eval_task_that_raises_scores_zero_on_every_scorer(eval_setup, run_eval, wrap_task, caplog):
    mock_scrapi = eval_setup
    scorer_calls = []

    def failing_task(input):
        raise ValueError("Task failed")

    def scorer(args):
        scorer_calls.append(args)
        return 1.0

    result = run_eval(
        "test_task_raises",
        data=[{"input": "boom", "expected": "Anything"}],
        task=wrap_task(failing_task),
        scorer={"exact": scorer, "length": scorer},
    )

    record = result.results[0]
    assert record.input == "boom"
    assert record.expected == "Anything"
    assert record.output == "[Error]"
    assert record.error == True
    assert record.scores == {"exact": 0.0, "length": 0.0}
    assert result.metadata.error == True
    assert scorer_calls == []
    assert "Task failed" in caplog.text

    payload_record = mock_scrapi.get_eval_payloads()[0]["results"][0]
    assert payload_record == {
        "input": "boom",
        "expected": "Anything",
        "output": "[Error]",
        "error": True,
        "scores": {"exact": {"score": 0.0}, "length": {"score": 0.0}},
    }


def _async_task_with_hook(fn):
    async def task(input, hook):
        return fn(input, hook)

    return task


@pytest.mark.parametrize("wrap_task", [_sync_task, _async_task], ids=["sync", "async"])
def test_eval_sends_how_long_the_task_took_as_latency_ms(
    eval_setup, run_eval, wrap_task, monkeypatch
):
    mock_scrapi = eval_setup
    now = [1.0]
    monkeypatch.setattr(eval_module, "time", SimpleNamespace(perf_counter=lambda: now[0]))

    def task(input):
        now[0] += 0.2504
        return input

    def scorer(args):
        # Scoring isn't part of the task's latency.
        now[0] += 1.0
        return 1

    result = run_eval("latency", data=[{"input": "a"}], task=wrap_task(task), scorer=scorer)

    assert result.results[0].latency_ms == 250
    [payload_record] = mock_scrapi.get_eval_payloads()[0]["results"]
    assert payload_record["latencyMs"] == 250


@pytest.mark.parametrize("wrap_task", [_sync_task, _async_task_with_hook], ids=["sync", "async"])
def test_eval_sends_the_sum_of_the_tokens_a_task_reports(eval_setup, run_eval, wrap_task):
    mock_scrapi = eval_setup

    def task(input, hook):
        hook.report_tokens(120, 30)
        hook.report_tokens(80, 12)
        return input

    result = run_eval("tokens", data=[{"input": "a"}], task=wrap_task(task), scorer=lambda args: 1)

    record = result.results[0]
    assert (record.input_tokens, record.output_tokens) == (200, 42)
    assert isinstance(record.latency_ms, int)
    [payload_record] = mock_scrapi.get_eval_payloads()[0]["results"]
    assert payload_record["inputTokens"] == 200
    assert payload_record["outputTokens"] == 42
    assert payload_record["latencyMs"] == record.latency_ms


def test_eval_counts_each_sync_tasks_tokens_on_its_own_record(eval_setup, run_eval):
    mock_scrapi = eval_setup
    # Holds every task until all three are running, each in its own worker thread.
    all_running = threading.Barrier(3, timeout=5)

    def task(input, hook):
        all_running.wait()
        count = int(input)
        # A task can also report from threads of its own, such as parallel model calls.
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda _: hook.report_tokens(count, 10 * count), range(2)))
        return input

    result = run_eval(
        "sync tokens",
        data=[{"input": "1"}, {"input": "2"}, {"input": "3"}],
        task=task,
        scorer=lambda args: 1,
    )

    assert [(r.input_tokens, r.output_tokens) for r in result.results] == [
        (2, 20),
        (4, 40),
        (6, 60),
    ]
    payload = mock_scrapi.get_eval_payloads()[0]
    assert [(r["inputTokens"], r["outputTokens"]) for r in payload["results"]] == [
        (2, 20),
        (4, 40),
        (6, 60),
    ]


@pytest.mark.parametrize("wrap_task", [_sync_task, _async_task_with_hook], ids=["sync", "async"])
def test_eval_sends_no_tokens_for_a_record_whose_task_never_reports_them(
    eval_setup, run_eval, wrap_task
):
    mock_scrapi = eval_setup

    def task(input, hook):
        if input == "reports":
            hook.report_tokens(10, 2)
        return input

    result = run_eval(
        "no tokens",
        data=[{"input": "reports"}, {"input": "silent"}],
        task=wrap_task(task),
        scorer=lambda args: 1,
    )

    reports, silent = result.results
    assert (reports.input_tokens, reports.output_tokens) == (10, 2)
    assert (silent.input_tokens, silent.output_tokens) == (None, None)
    assert isinstance(silent.latency_ms, int)
    reports_payload, silent_payload = mock_scrapi.get_eval_payloads()[0]["results"]
    assert (reports_payload["inputTokens"], reports_payload["outputTokens"]) == (10, 2)
    assert isinstance(silent_payload["latencyMs"], int)
    assert "inputTokens" not in silent_payload
    assert "outputTokens" not in silent_payload


@pytest.mark.parametrize("wrap_task", [_sync_task, _async_task_with_hook], ids=["sync", "async"])
def test_eval_sends_no_latency_or_tokens_for_a_record_whose_task_raises(
    eval_setup, run_eval, wrap_task
):
    mock_scrapi = eval_setup

    def task(input, hook):
        hook.report_tokens(10, 2)
        raise ValueError("Task failed")

    result = run_eval("raises", data=[{"input": "boom"}], task=wrap_task(task), scorer=lambda args: 1)

    record = result.results[0]
    assert record.error == True
    assert (record.latency_ms, record.input_tokens, record.output_tokens) == (None, None, None)
    [payload_record] = mock_scrapi.get_eval_payloads()[0]["results"]
    assert payload_record["error"] == True
    assert not {"latencyMs", "inputTokens", "outputTokens"} & payload_record.keys()


def test_eval_ignores_token_counts_the_api_would_reject(eval_setup, run_eval, caplog):
    def task(input, hook):
        hook.report_tokens(float("nan"), 5)
        hook.report_tokens(5, -1)
        hook.report_tokens(True, 5)
        hook.report_tokens(None, 5)
        hook.report_tokens(7, 3)
        return input

    result = run_eval("bad tokens", data=[{"input": "a"}], task=task, scorer=lambda args: 1)

    record = result.results[0]
    assert record.error == False
    assert (record.input_tokens, record.output_tokens) == (7, 3)
    warnings = [r.getMessage() for r in caplog.records if "report_tokens()" in r.getMessage()]
    assert len(warnings) == 4
    assert "input_tokens=5, output_tokens=-1" in warnings[1]


def test_eval_ignores_tokens_reported_after_the_task_returns(eval_setup, run_eval):
    hooks = []

    def task(input, hook):
        hooks.append(hook)
        hook.report_tokens(10, 2)
        return input

    def scorer(args):
        hooks[0].report_tokens(100, 100)
        return 1

    result = run_eval("late tokens", data=[{"input": "a"}], task=task, scorer=scorer)

    assert (result.results[0].input_tokens, result.results[0].output_tokens) == (10, 2)


def test_eval_hook_built_by_hand_reports_tokens_without_serializing_them():
    hook = EvalHook(parameters={"model": "gpt-4o-mini"})
    hook.report_tokens(10, 2)

    assert hook.to_dict() == {"parameters": {"model": "gpt-4o-mini"}}
    assert hook == EvalHook(parameters={"model": "gpt-4o-mini"})


def test_eval_sends_non_string_values_as_json_strings(eval_setup, run_eval):
    mock_scrapi = eval_setup

    result = run_eval(
        "test_non_string_values",
        data=[
            {"input": {"question": "Où?"}, "expected": 3},
            {"input": [1, 2], "expected": {"answer": [1, 2]}},
            {"input": 2.5, "expected": [2.5]},
        ],
        task=lambda input: {"answer": input} if isinstance(input, list) else [input],
        scorer=lambda args: 1,
    )

    results = {str(r.input): r for r in result.results}
    assert results["{'question': 'Où?'}"].input == {"question": "Où?"}
    assert results["{'question': 'Où?'}"].output == [{"question": "Où?"}]
    assert results["{'question': 'Où?'}"].expected == 3
    assert results["[1, 2]"].output == {"answer": [1, 2]}
    assert results["2.5"].output == [2.5]

    payload_results = {r["input"]: r for r in mock_scrapi.get_eval_payloads()[0]["results"]}
    assert payload_results['{"question":"Où?"}']["output"] == '[{"question":"Où?"}]'
    assert payload_results['{"question":"Où?"}']["expected"] == "3"
    assert payload_results["[1,2]"]["output"] == '{"answer":[1,2]}'
    assert payload_results["[1,2]"]["expected"] == '{"answer":[1,2]}'
    assert payload_results["2.5"]["output"] == "[2.5]"
    assert payload_results["2.5"]["expected"] == "[2.5]"


def test_eval_sends_values_that_are_not_json_as_text(eval_setup, run_eval):
    mock_scrapi = eval_setup

    class Answer:
        def __str__(self):
            return "Answer(42)"

    run_eval(
        "test_values_not_json",
        data=[{"input": "q"}],
        task=lambda input: Answer(),
        scorer=lambda args: 1,
    )

    payload_record = mock_scrapi.get_eval_payloads()[0]["results"][0]
    assert payload_record["output"] == "Answer(42)"


def test_eval_sends_empty_output_when_task_returns_none(eval_setup, run_eval):
    mock_scrapi = eval_setup

    result = run_eval(
        "test_none_output",
        data=[{"input": "Foo"}],
        task=lambda input: None,
        scorer=lambda args: 1,
    )

    assert result.results[0].output is None
    assert result.results[0].error == False

    payload_record = mock_scrapi.get_eval_payloads()[0]["results"][0]
    assert payload_record["input"] == "Foo"
    assert payload_record["output"] == ""
    assert "expected" not in payload_record


def test_eval_sends_non_string_score_metadata_as_strings(eval_setup, run_eval):
    mock_scrapi = eval_setup

    def scorer(args):
        return {
            "score": 1,
            "metadata": {
                "reasoning": "Greets Foo by name",
                "confidence": 0.9,
                "tags": ["greeting"],
                "details": {"exact": True},
                "passed": True,
            },
        }

    def scorer_with_metadata_object(args):
        return ScoreWithMetadata(score=True, metadata={"tokens": 12})

    run_eval(
        "test_score_metadata",
        data=[{"input": "Foo"}],
        task=lambda input: "Hello " + input,
        scorer={"Grader": scorer, "Tokens": scorer_with_metadata_object},
    )

    scores = mock_scrapi.get_eval_payloads()[0]["results"][0]["scores"]
    assert scores["Grader"] == {
        "score": 1.0,
        "metadata": {
            "reasoning": "Greets Foo by name",
            "confidence": "0.9",
            "tags": '["greeting"]',
            "details": '{"exact":true}',
            "passed": "true",
        },
    }
    assert scores["Tokens"] == {"score": 1.0, "metadata": {"tokens": "12"}}


def test_eval_passes_category_to_scorers(eval_setup, run_eval):
    scorer_args = []

    def scorer(args: EvalScorerArgs[str, str]) -> bool:
        scorer_args.append(args)
        return True

    run_eval(
        "test_scorer_category",
        data=[
            {"input": "a", "expected": "A", "category": "letters"},
            {"input": "b", "category": ["letters", "lowercase"]},
            {"input": "c"},
        ],
        task=lambda input: input.upper(),
        scorer=scorer,
    )

    by_input = {args.input: args for args in scorer_args}
    assert by_input["a"] == EvalScorerArgs(
        input="a", output="A", expected="A", category="letters"
    )
    assert by_input["b"].category == ["letters", "lowercase"]
    assert by_input["b"].expected is None
    assert by_input["c"].category is None


def test_eval_resolves_pydantic_parameter_defaults(eval_setup, run_eval):
    mock_scrapi = eval_setup
    hook_parameters = []

    def task(input: str, hook: EvalHook) -> str:
        hook_parameters.append(hook.parameters)
        return f"{hook.parameters['prefix']} {input}{hook.parameters['suffix']}"

    run_eval(
        "test_pydantic_parameters",
        data=[{"input": "world", "expected": "Hi world!"}],
        task=task,
        scorer=lambda args: args.output == args.expected,
        parameters={
            "prefix": Field(default="Hi"),
            "suffix": "!",
            "count": Field(default=123),
            "config": Field(default_factory=lambda: {"nested": "value"}),
            "tags": ["a", "b"],
        },
    )

    assert hook_parameters == [
        {
            "prefix": "Hi",
            "suffix": "!",
            "count": 123,
            "config": {"nested": "value"},
            "tags": ["a", "b"],
        }
    ]

    payload = mock_scrapi.get_eval_payloads()[0]
    assert payload["results"][0]["output"] == "Hi world!"
    assert payload["parameters"] == {
        "prefix": "Hi",
        "suffix": "!",
        "count": "123",
        "config": '{"nested":"value"}',
        "tags": '["a","b"]',
    }


def test_eval_passes_plain_parameters_to_hooks(eval_setup, run_eval):
    mock_scrapi = eval_setup
    hook_parameters = []

    def task(input: str, hook: EvalHook) -> str:
        hook_parameters.append(hook.parameters)
        return input

    run_eval(
        "test_plain_parameters",
        data=[{"input": "world"}],
        task=task,
        scorer=lambda args: 1,
        parameters={"model": "gpt-4o", "temperature": 0.2, "options": {"stream": False}},
    )

    assert hook_parameters == [
        {"model": "gpt-4o", "temperature": 0.2, "options": {"stream": False}}
    ]
    assert mock_scrapi.get_eval_payloads()[0]["parameters"] == {
        "model": "gpt-4o",
        "temperature": "0.2",
        "options": '{"stream":false}',
    }


def test_eval_plain_parameters_work_without_pydantic(eval_setup, run_eval, monkeypatch):
    mock_scrapi = eval_setup
    # A None entry makes `from pydantic.fields import ...` raise ImportError.
    monkeypatch.setitem(sys.modules, "pydantic.fields", None)
    hook_parameters = []

    def task(input: str, hook: EvalHook) -> str:
        hook_parameters.append(hook.parameters)
        return input

    run_eval(
        "test_parameters_without_pydantic",
        data=[{"input": "world"}],
        task=task,
        scorer=lambda args: 1,
        parameters={"prefix": "Hi", "count": 3},
    )

    assert hook_parameters == [{"prefix": "Hi", "count": 3}]
    assert mock_scrapi.get_eval_payloads()[0]["parameters"] == {"prefix": "Hi", "count": "3"}


def test_eval_pydantic_parameter_without_default_fails_before_any_task(eval_setup, run_eval):
    mock_scrapi = eval_setup
    task_inputs = []

    with pytest.raises(ValueError, match="Parameter 'prefix' needs a default value"):
        run_eval(
            "test_pydantic_parameter_without_default",
            data=[{"input": "world"}],
            task=lambda input: task_inputs.append(input),
            scorer=lambda args: 1,
            parameters={"prefix": Field()},
        )

    assert task_inputs == []
    assert mock_scrapi.get_eval_payloads() == []


def test_eval_results_keep_the_data_order(eval_setup, run_eval):
    mock_scrapi = eval_setup

    async def task(input):
        # Later records finish first, so completion order is the reverse of the data.
        await asyncio.sleep(0.01 * (3 - int(input)))
        return input

    result = run_eval(
        "test_result_order",
        data=[{"input": "0"}, {"input": "1"}, {"input": "2"}],
        task=task,
        scorer=lambda args: 1.0,
    )

    assert [r.input for r in result.results] == ["0", "1", "2"]
    payload = mock_scrapi.get_eval_payloads()[0]
    assert [r["input"] for r in payload["results"]] == ["0", "1", "2"]


def test_eval_task_gets_empty_parameters_without_parameters(eval_setup, run_eval):
    hook_parameters = []

    def task(input, hook):
        hook_parameters.append(hook.parameters)
        return input

    run_eval("test_empty_parameters", data=[{"input": "a"}], task=task, scorer=lambda args: 1.0)

    assert hook_parameters == [{}]


def _dataset_row(index, groups=("math",)):
    row = {
        "rowID": f"row-{index}",
        "columns": [
            {"name": "question", "value": f"q{index}"},
            {"name": "answer", "value": f"a{index}"},
        ],
    }
    if groups is not None:
        row["groups"] = list(groups)
    return row


def _stub_dataset_page(mock_scrapi, path_name, page, rows, status=200):
    mock_scrapi.stub(
        f"/console/v1/eval_datasets/{path_name}/rows",
        response=json.dumps({"data": rows}),
        status=status,
        query_string={"page": str(page), "limit": "100"},
    )


def test_eval_on_statsig_dataset_loads_every_page_and_posts_row_ids(eval_setup, run_eval):
    mock_scrapi = eval_setup
    _stub_dataset_page(mock_scrapi, "golden", 1, [_dataset_row(i) for i in range(100)])
    _stub_dataset_page(mock_scrapi, "golden", 2, [_dataset_row(100)])

    result = run_eval(
        "dataset eval",
        data=statsig_dataset(
            "golden",
            lambda row: {
                "input": row.columns["question"],
                "expected": row.columns["answer"],
                "category": row.groups,
            },
        ),
        task=lambda input: input.replace("q", "a"),
        scorer=lambda args: args.output == args.expected,
    )

    assert len(result.results) == 101
    assert {r.row_id for r in result.results} == {f"row-{i}" for i in range(101)}
    assert all(r.scores == {"Grader": 1.0} for r in result.results)

    page_requests = mock_scrapi.get_requests_for_endpoint("/console/v1/eval_datasets/golden/rows")
    assert [dict(r.args) for r in page_requests] == [
        {"page": "1", "limit": "100"},
        {"page": "2", "limit": "100"},
    ]
    assert [r.headers["STATSIG-API-KEY"] for r in page_requests] == ["test-api-key"] * 2

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert payloads[0]["dataset"] == "golden"
    payload_results = {r["rowID"]: r for r in payloads[0]["results"]}
    assert len(payload_results) == 101
    assert isinstance(payload_results["row-100"].pop("latencyMs"), int)
    assert payload_results["row-100"] == {
        "input": "q100",
        "expected": "a100",
        "output": "a100",
        "category": ["math"],
        "rowID": "row-100",
        "scores": {"Grader": {"score": 1.0}},
        "error": False,
    }


def test_statsig_dataset_maps_rows_and_keeps_their_row_ids(eval_setup):
    mock_scrapi = eval_setup
    _stub_dataset_page(mock_scrapi, "golden", 1, [_dataset_row(0), _dataset_row(1, groups=None)])
    rows = []
    scorer_row_ids = []

    def to_record(row):
        rows.append(row)
        if row.row_id == "row-0":
            return {"input": row.columns["question"]}
        return EvalDataRecord(input=row.columns["question"], row_id="some-other-row")

    def scorer(args: EvalScorerArgs[str, str]) -> float:
        scorer_row_ids.append(args.row_id)
        return 1.0

    result = Eval(
        "dataset eval",
        data=statsig_dataset("golden", to_record),
        task=lambda input: input,
        scorer=scorer,
    )

    assert rows == [
        StatsigDatasetRow(
            row_id="row-0", columns={"question": "q0", "answer": "a0"}, groups=["math"]
        ),
        StatsigDatasetRow(row_id="row-1", columns={"question": "q1", "answer": "a1"}, groups=[]),
    ]
    results = results_by_input(result.results)
    assert results["q0"].row_id == "row-0"
    assert results["q1"].row_id == "row-1"
    assert sorted(scorer_row_ids) == ["row-0", "row-1"]
    # A page with fewer than 100 rows is the last one.
    assert mock_scrapi.times_called_for_endpoint("/console/v1/eval_datasets/") == 1

    payload_results = results_by_input(mock_scrapi.get_eval_payloads()[0]["results"])
    assert payload_results["q0"]["rowID"] == "row-0"
    assert payload_results["q1"]["rowID"] == "row-1"


def test_statsig_dataset_url_encodes_the_dataset_name(eval_setup):
    mock_scrapi = eval_setup
    _stub_dataset_page(mock_scrapi, "golden set/v2", 1, [_dataset_row(0)])

    Eval(
        "dataset eval",
        data=statsig_dataset("golden set/v2", lambda row: {"input": row.columns["question"]}),
        task=lambda input: input,
        scorer=lambda args: 1,
    )

    [page_request] = mock_scrapi.get_requests_for_endpoint("/console/v1/eval_datasets/")
    assert page_request.environ["RAW_URI"] == (
        "/console/v1/eval_datasets/golden%20set%2Fv2/rows?page=1&limit=100"
    )
    assert mock_scrapi.get_eval_payloads()[0]["dataset"] == "golden set/v2"


def test_eval_dataset_option_with_caller_built_records(eval_setup, run_eval):
    mock_scrapi = eval_setup

    result = run_eval(
        "dataset eval",
        data=[
            {"input": "q0", "expected": "q0", "row_id": "row-0"},
            EvalDataRecord(input="q1", row_id="row-1"),
        ],
        dataset="golden",
        task=lambda input: input,
        scorer=lambda args: 1,
    )

    results = results_by_input(result.results)
    assert results["q0"].row_id == "row-0"
    assert results["q1"].row_id == "row-1"

    payload = mock_scrapi.get_eval_payloads()[0]
    assert payload["dataset"] == "golden"
    payload_results = results_by_input(payload["results"])
    assert payload_results["q0"]["rowID"] == "row-0"
    assert payload_results["q1"]["rowID"] == "row-1"


def test_eval_dataset_option_overrides_the_statsig_dataset_name(eval_setup):
    mock_scrapi = eval_setup
    _stub_dataset_page(mock_scrapi, "golden", 1, [_dataset_row(0)])

    Eval(
        "dataset eval",
        data=statsig_dataset("golden", lambda row: {"input": row.columns["question"]}),
        dataset="golden-copy",
        task=lambda input: input,
        scorer=lambda args: 1,
    )

    assert mock_scrapi.get_eval_payloads()[0]["dataset"] == "golden-copy"


def test_eval_without_dataset_sends_no_dataset(eval_setup):
    mock_scrapi = eval_setup

    Eval(
        "no dataset eval",
        data=[{"input": "q0", "row_id": "row-0"}, {"input": "q1"}],
        task=lambda input: input,
        scorer=lambda args: 1,
    )

    payload = mock_scrapi.get_eval_payloads()[0]
    assert "dataset" not in payload
    payload_results = results_by_input(payload["results"])
    assert payload_results["q0"]["rowID"] == "row-0"
    assert "rowID" not in payload_results["q1"]


@pytest.mark.parametrize(
    "data, message",
    [
        (
            [{"input": "q0", "row_id": "row-0"}, {"input": "q1"}],
            r"data\[1\] needs a row_id when the eval names a dataset",
        ),
        (
            [EvalDataRecord(input="q0", row_id="")],
            r"data\[0\] needs a row_id when the eval names a dataset",
        ),
        (
            [{"input": "q0", "row_id": "row-0"}, EvalDataRecord(input="q1", row_id="row-0")],
            r"data\[1\] repeats row_id row-0; each record needs its own dataset row",
        ),
    ],
    ids=["missing", "empty", "duplicate"],
)
def test_eval_checks_row_ids_before_any_task_runs(eval_setup, run_eval, data, message):
    mock_scrapi = eval_setup
    task_inputs = []

    with pytest.raises(ValueError, match=message):
        run_eval(
            "dataset eval",
            data=data,
            dataset="golden",
            task=lambda input: task_inputs.append(input),
            scorer=lambda args: 1,
        )

    assert task_inputs == []
    assert mock_scrapi.get_eval_payloads() == []


def test_eval_fails_when_the_dataset_cannot_be_loaded(eval_setup, run_eval):
    mock_scrapi = eval_setup
    _stub_dataset_page(mock_scrapi, "missing", 1, [], status=404)
    task_inputs = []

    with pytest.raises(RuntimeError, match="Failed to load eval dataset 'missing': 404"):
        run_eval(
            "dataset eval",
            data=statsig_dataset("missing", lambda row: {"input": row.columns["question"]}),
            task=lambda input: task_inputs.append(input),
            scorer=lambda args: 1,
        )

    assert task_inputs == []
    assert mock_scrapi.get_eval_payloads() == []


# Record traces, ported from the Node SDK's eval tests.


class _EvalTracing:
    def __init__(self, provider: TracerProvider, exporter: InMemorySpanExporter) -> None:
        self.provider = provider
        self.exporter = exporter

    def spans(self):
        # Read without flushing, so the spans are only here if Eval() flushed them.
        return list(self.exporter.get_finished_spans())

    def span(self, span_id: str) -> ReadableSpan:
        [span] = [s for s in self.spans() if _span_id(s) == span_id]
        return span

    def children(self, span_id: str):
        return [s for s in self.spans() if s.parent is not None and _parent_id(s) == span_id]


def _span_id(span: ReadableSpan) -> str:
    return trace.format_span_id(span.context.span_id)


def _trace_id(span: ReadableSpan) -> str:
    return trace.format_trace_id(span.context.trace_id)


def _parent_id(span: ReadableSpan) -> str:
    assert span.parent is not None
    return trace.format_span_id(span.parent.span_id)


@pytest.fixture
def tracing(monkeypatch):
    exporter = InMemorySpanExporter()
    # Exports only on flush, so finished spans show that Eval() flushed.
    processor = StatsigSpanProcessor(exporter, schedule_delay_millis=60_000)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    monkeypatch.setattr(OtelSingleton, "_instance", OtelSingleton(provider))
    monkeypatch.setattr(span_telemetry, "get_statsig_instance_for_logging", lambda: None)

    yield _EvalTracing(provider, exporter)

    provider.shutdown()


class _AsyncEchoOpenAI:
    """An AsyncOpenAI stand-in that replies "Hello <first message>"."""

    def __init__(self) -> None:
        self.chat = self.ChatNamespace()

    class ChatNamespace:
        class CompletionsNamespace:
            async def create(self, *args, **kwargs):
                await asyncio.sleep(0)
                return {
                    "id": "chatcmpl_1",
                    "model": kwargs["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello " + kwargs["messages"][0]["content"],
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }

        def __init__(self) -> None:
            self.completions = self.CompletionsNamespace()


def test_eval_gives_each_record_its_own_trace_with_its_model_calls_under_it(
    eval_setup, run_eval, tracing
):
    mock_scrapi = eval_setup
    openai = wrap_openai(_AsyncEchoOpenAI())

    async def task(input):
        completion = await openai.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": input}]
        )
        return completion["choices"][0]["message"]["content"]

    with tracing.provider.get_tracer("test").start_as_current_span("outer") as outer:
        result = run_eval(
            "greeting",
            data=[{"input": "Foo"}, {"input": "Bar"}, {"input": "Baz"}],
            task=task,
            scorer=lambda args: args.output == "Hello Foo",
            eval_run_name="run-traces",
        )
    outer_trace_id = trace.format_trace_id(outer.get_span_context().trace_id)

    assert len(tracing.spans()) == 9
    trace_ids = {r.trace_id for r in result.results}
    assert len(trace_ids) == 3
    assert outer_trace_id not in trace_ids

    payload_results = mock_scrapi.get_eval_payloads()[0]["results"]
    for index, record in enumerate(result.results):
        assert payload_results[index]["input"] == record.input
        assert payload_results[index]["traceID"] == record.trace_id
        assert payload_results[index]["spanID"] == record.span_id

        record_span = tracing.span(record.span_id)
        assert _trace_id(record_span) == record.trace_id
        assert record_span.parent is None
        assert record_span.name == "invoke_workflow"
        attrs = dict(record_span.attributes or {})
        assert attrs[STATSIG_ATTR_SPAN_TYPE] == StatsigSpanType.GEN_AI
        assert attrs[STATSIG_ATTR_GEN_AI_SPAN_TYPE] == StatsigGenAISpanType.WORKFLOW
        assert attrs["statsig.gen_ai.workflow.name"] == "greeting"
        assert attrs[STATSIG_ATTR_SPAN_LLM_ROOT] == STATSIG_SPAN_LLM_ROOT_VALUE
        assert attrs[STATSIG_ATTR_ENVIRONMENT] == "eval"
        assert attrs[STATSIG_ATTR_EVAL_NAME] == "greeting"
        assert attrs[STATSIG_ATTR_EVAL_RUN_NAME] == "run-traces"
        assert attrs[STATSIG_ATTR_EVAL_RECORD_INDEX] == index
        assert record_span.status.status_code == StatusCode.OK

        children = tracing.children(record.span_id)
        assert sorted(
            (s.name, (s.attributes or {}).get(STATSIG_ATTR_EVAL_SCORER_NAME)) for s in children
        ) == [("chat gpt-test", None), ("invoke_workflow", "Grader")]
        assert all(_trace_id(s) == record.trace_id for s in children)
        [model_span] = [s for s in children if s.name == "chat gpt-test"]
        assert STATSIG_ATTR_SPAN_LLM_ROOT not in (model_span.attributes or {})
        assert (model_span.attributes or {})[STATSIG_ATTR_ENVIRONMENT] == "eval"

    assert [r.output for r in result.results] == ["Hello Foo", "Hello Bar", "Hello Baz"]
    assert [r.scores for r in result.results] == [{"Grader": 1.0}, {"Grader": 0.0}, {"Grader": 0.0}]


def test_eval_keeps_a_sync_tasks_model_calls_under_the_record_span(eval_setup, run_eval, tracing):
    openai = wrap_openai(MockOpenAI())

    def task(input):
        openai.chat.completions.create(model=f"model-{input}", messages=[])
        return input

    result = run_eval(
        "sync task", data=[{"input": "a"}, {"input": "b"}], task=task, scorer=lambda args: 1
    )

    for record in result.results:
        [model_span] = [s for s in tracing.spans() if s.name == f"chat model-{record.input}"]
        assert _parent_id(model_span) == record.span_id
        assert _trace_id(model_span) == record.trace_id
        assert (model_span.attributes or {})[STATSIG_ATTR_ENVIRONMENT] == "eval"


def test_eval_runs_each_scorer_in_a_child_span(eval_setup, run_eval, tracing):
    lookup = wrap(lambda: "found", type="tool", name="lookup")

    def exact(args):
        lookup()
        return args.output == "a"

    async def length(args):
        await asyncio.sleep(0)
        return {"score": 1, "metadata": {"length": len(args.output)}}

    result = run_eval(
        "scorer spans",
        data=[{"input": "a"}],
        task=lambda input: input,
        scorer={"exact": exact, "length": length},
    )

    [record] = result.results
    assert record.scores == {"exact": 1.0, "length": 1.0}
    scorer_spans = {
        (s.attributes or {})[STATSIG_ATTR_EVAL_SCORER_NAME]: s
        for s in tracing.children(record.span_id)
    }
    assert sorted(scorer_spans) == ["exact", "length"]
    for scorer_name, span in scorer_spans.items():
        attrs = dict(span.attributes or {})
        assert span.name == "invoke_workflow"
        assert attrs["statsig.gen_ai.workflow.name"] == scorer_name
        assert attrs[STATSIG_ATTR_GEN_AI_SPAN_TYPE] == StatsigGenAISpanType.WORKFLOW
        assert attrs[STATSIG_ATTR_ENVIRONMENT] == "eval"
        assert STATSIG_ATTR_SPAN_LLM_ROOT not in attrs

    [tool_span] = [s for s in tracing.spans() if s.name == "execute_tool"]
    assert _parent_id(tool_span) == _span_id(scorer_spans["exact"])


def test_eval_marks_the_record_span_as_failed_when_the_task_raises(
    eval_setup, run_eval, tracing
):
    mock_scrapi = eval_setup

    def task(input):
        raise ValueError("Task failed")

    result = run_eval("failing", data=[{"input": "Boom"}], task=task, scorer=lambda args: 1)

    [record] = result.results
    assert record.error == True
    assert record.output == "[Error]"
    assert record.scores == {"Grader": 0.0}
    assert record.trace_id is not None and record.span_id is not None
    record_span = tracing.span(record.span_id)
    assert record_span.status.status_code == StatusCode.ERROR
    assert record_span.status.description == "Task failed"
    assert tracing.children(record.span_id) == []

    [payload_record] = mock_scrapi.get_eval_payloads()[0]["results"]
    assert payload_record["traceID"] == record.trace_id
    assert payload_record["spanID"] == record.span_id


def test_eval_tags_spans_from_tasks_and_scorers_with_the_eval_environment(
    eval_setup, run_eval, tracing
):
    async def task(input):
        await asyncio.sleep(0)
        return input

    run_eval(
        "environment task",
        data=[{"input": "Foo"}],
        task=wrap(task, type="tool", name="task-tool"),
        scorer=wrap(lambda args: 1, type="tool", name="scorer-tool"),
    )
    wrap(lambda: "done", type="tool", name="outside-tool")()
    tracing.provider.force_flush()

    environment_by_tool = {
        (s.attributes or {})["statsig.gen_ai.tool.name"]: (s.attributes or {}).get(
            STATSIG_ATTR_ENVIRONMENT
        )
        for s in tracing.spans()
        if "statsig.gen_ai.tool.name" in (s.attributes or {})
    }
    assert environment_by_tool == {
        "task-tool": "eval",
        "scorer-tool": "eval",
        "outside-tool": None,
    }


def test_eval_flushes_the_record_spans_before_sending_results(
    eval_setup, run_eval, tracing, monkeypatch
):
    import requests

    exported_at_send = []
    post = requests.post

    def recording_post(url, *args, **kwargs):
        exported_at_send.append(len(tracing.spans()))
        return post(url, *args, **kwargs)

    monkeypatch.setattr(requests, "post", recording_post)

    run_eval("flush", data=[{"input": "a"}, {"input": "b"}], task=lambda i: i, scorer=lambda a: 1)

    # A record span and a scorer span for each record.
    assert exported_at_send == [4]


def test_eval_sends_results_when_the_flush_fails(
    eval_setup, run_eval, tracing, monkeypatch, caplog
):
    mock_scrapi = eval_setup

    def failing_flush():
        raise RuntimeError("exporter down")

    monkeypatch.setattr(OtelSingleton, "flush_instance", failing_flush)

    result = run_eval("flush fails", data=[{"input": "a"}], task=lambda i: i, scorer=lambda a: 1)

    assert result.results[0].trace_id is not None
    assert len(mock_scrapi.get_eval_payloads()) == 1
    assert "Failed to flush eval spans: exporter down" in caplog.text


def test_eval_called_from_a_running_event_loop_keeps_the_callers_context(eval_setup, tracing):
    openai = wrap_openai(MockOpenAI())
    user = StatsigUser(user_id="user-123")

    def task(input):
        openai.chat.completions.create(model="gpt-test", messages=[])
        return input

    async def main():
        tracer = tracing.provider.get_tracer("test")
        with statsig_user_context(user), tracer.start_as_current_span("outer"):
            # Runs on a new event loop in another thread.
            return Eval("from a loop", data=[{"input": "a"}], task=task, scorer=lambda args: 1)

    [record] = asyncio.run(main()).results

    record_span = tracing.span(record.span_id)
    assert record_span.parent is None
    assert (record_span.attributes or {})[STATSIG_ATTR_USER_ID] == "user-123"
    [model_span] = [s for s in tracing.spans() if s.name == "chat gpt-test"]
    assert _parent_id(model_span) == record.span_id
    assert (model_span.attributes or {})[STATSIG_ATTR_USER_ID] == "user-123"


@pytest.mark.parametrize("wrap_task", [_sync_task, _async_task], ids=["sync", "async"])
def test_eval_without_tracing_runs_in_the_eval_environment_without_a_trace(
    eval_setup, run_eval, wrap_task, monkeypatch
):
    mock_scrapi = eval_setup
    monkeypatch.setattr(OtelSingleton, "_instance", None)
    flushes = []
    monkeypatch.setattr(OtelSingleton, "flush_instance", lambda: flushes.append(True))
    seen = []

    def observe(where):
        seen.append(
            (
                where,
                get_environment_span_attrs_from_context(),
                trace.get_current_span().get_span_context().is_valid,
            )
        )

    def task(input):
        observe("task")
        return input

    def scorer(args):
        observe("scorer")
        return 1

    result = run_eval("no tracing", data=[{"input": "a"}], task=wrap_task(task), scorer=scorer)

    eval_environment = {STATSIG_ATTR_ENVIRONMENT: "eval"}
    assert seen == [("task", eval_environment, False), ("scorer", eval_environment, False)]
    assert get_environment_span_attrs_from_context() is None
    [record] = result.results
    assert record.trace_id is None and record.span_id is None
    [payload_record] = mock_scrapi.get_eval_payloads()[0]["results"]
    assert "traceID" not in payload_record and "spanID" not in payload_record
    assert flushes == []
