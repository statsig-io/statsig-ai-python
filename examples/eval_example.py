"""
Example demonstrating the usage of both Eval (sync) and EvalAsync (async) functions.

This example shows:
1. Using Eval() in a synchronous context
2. Using EvalAsync() in an async context

To run this example, make sure you have STATSIG_API_KEY set in your environment:
    export STATSIG_API_KEY=your_key_here
    python examples/eval_example.py
"""

import asyncio
import random
import time
from typing import Dict, List

from statsig_ai.evals import Eval, EvalAsync, EvalDataRecord, EvalScorerArgs


# ============================================================
# Example Task Functions
# ============================================================


def simple_text_task(input_text: str) -> str:
    """A simple synchronous task that processes text."""
    time.sleep(random.uniform(0.5, 2))  # Simulate some processing
    return input_text.upper()


async def async_text_task(input_text: str) -> str:
    """An async task that processes text."""
    await asyncio.sleep(random.uniform(0.5, 2))  # Simulate async processing
    return input_text.upper()


# ============================================================
# Example Scorer Functions
# ============================================================


def exact_match_scorer(args: EvalScorerArgs[str, str]) -> bool:
    """Scorer that checks for exact match."""
    if args.expected is None:
        return False
    return args.output == args.expected


def length_scorer(args: EvalScorerArgs[str, str]) -> float:
    """Scorer that returns normalized length score."""
    expected_len = len(args.expected) if args.expected else 0
    output_len = len(args.output)
    if expected_len == 0:
        return 0.0
    return min(output_len / expected_len, 1.0)


async def async_scorer(args: EvalScorerArgs[str, str]) -> bool:
    """An async scorer."""
    await asyncio.sleep(0.01)  # Simulate async processing
    return args.output == args.expected


# ============================================================
# Example Data Generators
# ============================================================


def generate_test_data() -> List[EvalDataRecord[str, str]]:
    """Generate a simple list of test data."""
    return [
        EvalDataRecord(input="hello", expected="HELLO", category="greetings"),
        EvalDataRecord(input="world", expected="WORLD", category="greetings"),
        EvalDataRecord(input="python", expected="PYTHON", category="languages"),
    ] * 5


async def generate_async_test_data():
    """Generate test data asynchronously (as an async generator)."""
    data = [
        EvalDataRecord(input="async", expected="ASYNC", category="async_test"),
        EvalDataRecord(input="await", expected="AWAIT", category="async_test"),
        EvalDataRecord(input="coroutine", expected="COROUTINE", category="async_test"),
    ]
    for record in data * 5:
        await asyncio.sleep(0.01)  # Simulate fetching data
        yield record


# ============================================================
# Example Summary Score Function
# ============================================================


def calculate_summary_scores(results: List) -> Dict[str, float]:
    """Calculate summary statistics from results."""
    if not results:
        return {}

    # Calculate average scores
    all_scores = {}
    for result in results:
        for scorer_name, score in result.scores.items():
            if scorer_name not in all_scores:
                all_scores[scorer_name] = []
            all_scores[scorer_name].append(score)

    summary = {}
    for scorer_name, scores in all_scores.items():
        summary[f"{scorer_name}_avg"] = sum(scores) / len(scores)

    summary["total_records"] = len(results)
    summary["error_rate"] = sum(1 for r in results if r.error) / len(results)

    return summary


# ============================================================
# Example 1: Synchronous Eval
# ============================================================


def example_sync_eval():
    """Run evaluation synchronously."""
    print("\n" + "=" * 60)
    print("Example 1: Synchronous Eval")
    print("=" * 60)

    result = Eval(
        name="sync_example",
        data=generate_test_data(),
        task=simple_text_task,
        scorer={
            "exact_match": exact_match_scorer,
            "length": length_scorer,
        },
        eval_run_name="sync_run_1",
        summary_score_fn=calculate_summary_scores,
    )

    print(f"\n✓ Completed {len(result.results)} evaluations")
    print(f"  Any errors: {result.metadata.error}")
    if result.summary_scores:
        print(f"\n  Summary Scores:")
        for key, value in result.summary_scores.items():
            print(f"    {key}: {value:.3f}")

    print(f"\n  Sample Results:")
    for i, res in enumerate(result.results[:3]):
        print(f"    [{i+1}] Input: '{res.input}' -> Output: '{res.output}'")
        print(f"        Scores: {res.scores}")


# ============================================================
# Example 2: Asynchronous Eval
# ============================================================


async def example_async_eval():
    """Run evaluation asynchronously."""
    print("\n" + "=" * 60)
    print("Example 2: Asynchronous Eval")
    print("=" * 60)

    result = await EvalAsync(
        name="async_example",
        data=generate_async_test_data(),
        task=async_text_task,
        scorer={
            "exact_match": exact_match_scorer,
            "async_scorer": async_scorer,
        },
        eval_run_name="async_run_1",
        summary_score_fn=calculate_summary_scores,
    )

    print(f"\n✓ Completed {len(result.results)} evaluations")
    print(f"  Any errors: {result.metadata.error}")
    if result.summary_scores:
        print(f"\n  Summary Scores:")
        for key, value in result.summary_scores.items():
            print(f"    {key}: {value:.3f}")

    print(f"\n  Sample Results:")
    for i, res in enumerate(result.results[:3]):
        print(f"    [{i+1}] Input: '{res.input}' -> Output: '{res.output}'")
        print(f"        Scores: {res.scores}")


def main():
    import os

    if not os.environ.get("STATSIG_API_KEY"):
        print("\n⚠️  WARNING: STATSIG_API_KEY not set in environment!")
        print("   Some features may not work properly.")
        print("   Set it with: export STATSIG_API_KEY=your_key_here\n")
        os.environ["STATSIG_API_KEY"] = "demo-key-for-testing"

    print("\n" + "=" * 60)
    print("Statsig Eval Examples")
    print("=" * 60)

    example_sync_eval()

    async def run_async_examples():
        await example_async_eval()

    print("\n\nRunning async examples...")
    asyncio.run(run_async_examples())

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
