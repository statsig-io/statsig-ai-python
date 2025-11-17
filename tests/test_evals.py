import pytest
from pytest_httpserver import HTTPServer
from statsig_ai import Eval, EvalScorerArgs, EvalDataRecord, EvalHook
from mock_scrapi import MockScrapi
import math


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
    assert result.results[0].input == "world"
    assert result.results[0].output == "Hello world"
    assert result.results[0].expected == "Hello world"
    assert result.results[0].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    payload = payloads[0]
    assert "results" in payload
    assert len(payload["results"]) == 2
    assert payload["results"][0]["input"] == "world"
    assert payload["results"][0]["output"] == "Hello world"
    assert payload["results"][0]["expected"] == "Hello world"
    assert payload["results"][0]["scores"]["Grader"] == 1.0


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

    assert result.results[0].output == "Hi world"
    assert result.results[0].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert payloads[0]["results"][0]["output"] == "Hi world"
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0


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

    assert "exact_match" in result.results[0].scores
    assert "length" in result.results[0].scores
    assert result.results[0].scores["exact_match"] == 1.0
    assert result.results[0].scores["length"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert payloads[0]["results"][0]["scores"]["exact_match"] == 1.0
    assert payloads[0]["results"][0]["scores"]["length"] == 1.0


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

    assert result.results[0].output == "Async world"
    assert result.results[0].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert payloads[0]["results"][0]["output"] == "Async world"
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0


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

    assert result.results[0].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0


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

    assert result.results[0].input == {"name": "Alice", "age": 30}
    assert result.results[0].output == "Hello Alice"
    assert result.results[0].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert payloads[0]["results"][0]["input"] == {"name": "Alice", "age": 30}
    assert payloads[0]["results"][0]["output"] == "Hello Alice"
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0


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

    assert result.results[0].input == 5
    assert result.results[0].output == 10
    assert result.results[0].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert payloads[0]["results"][0]["input"] == 5
    assert payloads[0]["results"][0]["output"] == 10
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0


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
    assert result.results[0].scores["Grader"] == 1.0
    assert result.results[0].output == "Hello world"

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert len(payloads[0]["results"]) == 2
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0
    assert payloads[0]["results"][0]["output"] == "Hello world"


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
    assert result.results[0].input == "world"
    assert result.results[0].output == "Hello world"
    assert result.results[0].scores["Grader"] == 1.0
    assert result.results[1].input == "test"
    assert result.results[1].output == "Hello test"
    assert result.results[1].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert len(payloads[0]["results"]) == 2
    assert payloads[0]["results"][0]["input"] == "world"
    assert payloads[0]["results"][0]["output"] == "Hello world"
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0


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
    assert result.results[0].input == "world"
    assert result.results[0].output == "Hello world"
    assert result.results[0].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert len(payloads[0]["results"]) == 2
    assert payloads[0]["results"][0]["input"] == "world"
    assert payloads[0]["results"][0]["output"] == "Hello world"
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0


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
    assert result.results[0].input == "world"
    assert result.results[0].output == "Hello world"
    assert result.results[0].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert len(payloads[0]["results"]) == 2
    assert payloads[0]["results"][0]["input"] == "world"
    assert payloads[0]["results"][0]["output"] == "Hello world"
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0


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
    assert result.results[0].input == "world"
    assert result.results[0].output == "Hello world"
    assert result.results[0].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert len(payloads[0]["results"]) == 2
    assert payloads[0]["results"][0]["input"] == "world"
    assert payloads[0]["results"][0]["output"] == "Hello world"
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0


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

    assert result.results[0].category == "greeting"
    assert result.results[1].category == ["greeting", "test"]

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert payloads[0]["results"][0]["category"] == "greeting"
    assert payloads[0]["results"][1]["category"] == ["greeting", "test"]


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

    assert result.results[0].expected is None
    assert result.results[0].scores["Grader"] == 1.0
    assert result.results[1].scores["Grader"] == 0.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert payloads[0]["results"][0].get("expected") is None
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0
    assert payloads[0]["results"][1]["scores"]["Grader"] == 0.0


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

    assert result.results[0].input == "world"
    assert result.results[0].output == "Hello world"
    assert result.results[0].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert payloads[0].get("name") == "test_run_123"
    assert payloads[0]["results"][0]["input"] == "world"
    assert payloads[0]["results"][0]["output"] == "Hello world"
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0


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
            {"input": "fail", "expected": "Hello fail"},
        ],
        task=failing_task,
        scorer=scorer,
    )

    assert result.results[0].output == "Hello world"
    assert result.results[0].error == False
    assert result.results[1].error == True
    assert result.results[1].scores["Grader"] == 0.0
    assert result.metadata.error == True

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert payloads[0]["results"][0]["output"] == "Hello world"
    assert payloads[0]["results"][0]["error"] == False
    assert payloads[0]["results"][1]["error"] == True
    assert result.results[1].scores["Grader"] == 0.0


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

    assert result.results[0].scores["Grader"] == 1.0
    assert result.results[1].scores["Grader"] == 0.0
    assert result.results[1].error == False

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0
    assert payloads[0]["results"][1]["scores"]["Grader"] == 0.0
    assert payloads[0]["results"][1]["error"] == False


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

    assert result.results[0].scores["Grader"] == 1.0
    assert result.results[1].scores["Grader"] == 0.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0
    assert payloads[0]["results"][1]["scores"]["Grader"] == 0.0


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

    assert result.results[0].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0


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
        parameters={"prefix": "Hi", "suffix": "!"},
    )

    assert result.results[0].output == "Hi world!"
    assert result.results[0].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert payloads[0]["results"][0]["output"] == "Hi world!"
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0


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
    assert result.results[0].input == "world"
    assert result.results[0].output == "Hello world"
    assert result.results[0].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert "summaryScores" not in payloads[0]
    assert payloads[0]["results"][0]["input"] == "world"
    assert payloads[0]["results"][0]["output"] == "Hello world"
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0


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
            exact_match_scores.append(result.scores.get("exact_match", 0.0))
            length_scores.append(result.scores.get("length", 0.0))

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

    assert result.results[0].input == "world"
    assert result.results[0].output == "Hello world"
    assert result.results[0].scores["exact_match"] == 1.0
    assert result.results[0].scores["length"] == 1.0
    assert result.results[2].input == "a"
    assert result.results[2].output == "Hello a"
    assert result.results[2].scores["exact_match"] == 0.0
    assert result.results[2].scores["length"] == 1.0

    assert result.summary_scores is not None
    assert result.summary_scores["avg_exact_match"] == pytest.approx(2.0 / 3.0)
    assert result.summary_scores["avg_length"] == 1.0
    assert result.summary_scores["min_exact_match"] == 0.0
    assert result.summary_scores["max_length"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert "summaryScores" in payloads[0]
    assert payloads[0]["results"][0]["input"] == "world"
    assert payloads[0]["results"][0]["output"] == "Hello world"
    assert payloads[0]["results"][0]["scores"]["exact_match"] == 1.0
    assert payloads[0]["results"][0]["scores"]["length"] == 1.0
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
    assert result.results[0].input == "world"
    assert result.results[0].output == "Hello world"
    assert result.results[0].scores["Grader"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert "results" in payloads[0]
    assert len(payloads[0]["results"]) == 1
    assert payloads[0]["results"][0]["input"] == "world"
    assert payloads[0]["results"][0]["output"] == "Hello world"
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0
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
                sum(result.scores.values()) / len(result.scores) if result.scores else 0.0
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

    assert result.results[0].input == "world"
    assert result.results[0].output == "Hello world"
    assert result.results[0].category == "greeting"
    assert result.results[0].scores["Grader"] == 1.0
    assert result.results[2].input == "foo"
    assert result.results[2].output == "Hello foo"
    assert result.results[2].category == "farewell"
    assert result.results[2].scores["Grader"] == 0.0

    assert result.summary_scores is not None
    assert "greeting_avg" in result.summary_scores
    assert "farewell_avg" in result.summary_scores
    assert result.summary_scores["greeting_avg"] == 1.0
    assert result.summary_scores["farewell_avg"] == 0.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert "summaryScores" in payloads[0]
    assert payloads[0]["results"][0]["input"] == "world"
    assert payloads[0]["results"][0]["output"] == "Hello world"
    assert payloads[0]["results"][0]["category"] == "greeting"
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0
    assert payloads[0]["results"][2]["input"] == "foo"
    assert payloads[0]["results"][2]["output"] == "Hello foo"
    assert payloads[0]["results"][2]["scores"]["Grader"] == 0.0
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
            sum(r.scores.values()) / len(r.scores) if r.scores else 0.0 for r in results
        )
        avg_score = total_score / len(results) if results else 0.0

        success_scores = [
            sum(r.scores.values()) / len(r.scores) for r in results if not r.error and r.scores
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

    assert result.results[0].input == "world"
    assert result.results[0].output == "Hello world"
    assert result.results[0].error == False
    assert result.results[0].scores["Grader"] == 1.0
    assert result.results[1].input == "fail"
    assert result.results[1].error == True
    assert result.results[1].scores["Grader"] == 0.0
    assert result.results[2].input == "test"
    assert result.results[2].output == "Hello test"
    assert result.results[2].error == False
    assert result.results[2].scores["Grader"] == 1.0

    assert result.summary_scores is not None
    assert result.summary_scores["error_count"] == 1
    assert result.summary_scores["success_count"] == 2
    assert result.summary_scores["avg_score"] == pytest.approx(2.0 / 3.0)
    assert result.summary_scores["avg_success_score"] == 1.0

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert "summaryScores" in payloads[0]
    assert payloads[0]["results"][0]["input"] == "world"
    assert payloads[0]["results"][0]["output"] == "Hello world"
    assert payloads[0]["results"][0]["error"] == False
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0
    assert payloads[0]["results"][1]["input"] == "fail"
    assert payloads[0]["results"][1]["error"] == True
    assert payloads[0]["results"][1]["scores"]["Grader"] == 0.0
    assert payloads[0]["results"][2]["input"] == "test"
    assert payloads[0]["results"][2]["output"] == "Hello test"
    assert payloads[0]["results"][2]["scores"]["Grader"] == 1.0
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

    assert result.results[0].input == "world"
    assert result.results[0].output == "Hello world"
    assert result.results[0].scores["Grader"] == 1.0
    assert result.results[1].input == "test"
    assert result.results[1].output == "Hello test"
    assert result.results[1].scores["Grader"] == 1.0
    assert result.summary_scores is None

    payloads = mock_scrapi.get_eval_payloads()
    assert len(payloads) == 1
    assert len(payloads[0]["results"]) == 2
    assert payloads[0]["results"][0]["input"] == "world"
    assert payloads[0]["results"][0]["output"] == "Hello world"
    assert payloads[0]["results"][0]["scores"]["Grader"] == 1.0
    assert payloads[0]["results"][1]["input"] == "test"
    assert payloads[0]["results"][1]["output"] == "Hello test"
    assert payloads[0]["results"][1]["scores"]["Grader"] == 1.0
    assert "summaryScores" not in payloads[0]
