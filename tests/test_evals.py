import asyncio
import sys

import pytest
from pydantic import Field
from pytest_httpserver import HTTPServer
from statsig_ai import Eval, EvalScorerArgs, EvalDataRecord, EvalHook
from statsig_ai.evals import EvalAsync
from statsig_ai.evals.eval_types import ScoreWithMetadata
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
