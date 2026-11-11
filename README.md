> ⚠️ **Beta Notice**
>
> This SDK is currently in **beta**. All APIs are **experimental** and subject to change.
> Please review the [release notes](https://github.com/statsig-io/statsig-ai-python/releases) for any **breaking changes** before upgrading.

## Statsig Python AI SDK

Statsig is the on-stop data platform for product development. We help you move faster with feature gates (feature flags), experiments, and the world's most powerful stats engine. If you're new to Statsig, check out our product and create an account at [statsig.com](https://www.statsig.com/?ref=gh_ai).

The AI SDK provides convenient hooks for developers working with LLMs to iterate on their promps, run evals to test their efficacy, add observability to LLM usage.

<h1 align="center">
	<a href="https://statsig.com/?ref=gh_ai">
		<img src="https://github.com/statsig-io/js-client-monorepo/assets/95646168/ae5499ed-20ff-4584-bf21-8857f800d485" />
	</a>
    <div />
	<a href="https://statsig.com/?ref=gh_ai">Statsig</a>
</h1>

<p align="center">
	<a href="https://github.com/statsig-io/statsig-ai-python/blob/main/LICENSE">
        	<img src="https://img.shields.io/badge/license-ISC-blue.svg?colorA=1b2528&colorB=ccfbc7&style=for-the-badge">
    	</a>
	<a href="https://pypi.org/project/statsig-ai/">
        	<img src="https://img.shields.io/pypi/v/statsig-ai.svg?colorA=1b2528&colorB=b2d3ff&style=for-the-badge">
    	</a>
	<a href="https://statsig.com/community?ref=gh_ai">
        	<img src="https://img.shields.io/badge/slack-statsig-brightgreen.svg?logo=slack&colorA=1b2528&colorB=FFF8BA&style=for-the-badge">
    	</a>
</p>

## Getting Started

Read through the [Documentation](https://docs.statsig.com/ai-evals/python?ref=gh_ai) to see code samples and how to get started.

## Tracing

`initialize_tracing()` sends OpenTelemetry spans to Statsig, and `wrap_openai()` records each OpenAI call as a span. The tracer provider it creates uses `StatsigSpanProcessor`, which adds the user and environment attributes below to every span when it starts.

### Tracing workflows and tools

`wrap()` records each call of a function as a span, so the model calls an agent makes are grouped under the turn or tool that made them. A `type="workflow"` function runs in an `invoke_workflow` span and a `type="tool"` function in an `execute_tool` span. Wrapped OpenAI calls and other `wrap` spans made inside it are its children. Use it as a decorator, or call `wrap(fn, type=..., name=...)`. An `async def` function gets an async wrapper, and its span ends once the call has been awaited.

```python
from openai import AsyncOpenAI
from statsig_ai import wrap, wrap_openai

openai = wrap_openai(AsyncOpenAI())

@wrap(type="tool", name="search_docs", tool_type="retrieval")
def search_docs(query: str) -> str:
    return index.search(query)

@wrap(type="workflow", name="support_turn")
async def run_turn(question: str) -> str:
    docs = search_docs(question)
    response = await openai.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": f"{question}\n\n{docs}"}],
    )
    return response.choices[0].message.content
```

Each span records `statsig.span.type=gen_ai`, `statsig.span.gen_ai.type` (`workflow` or `tool`), and the name as `statsig.gen_ai.workflow.name` or `statsig.gen_ai.tool.name`. `tool_type` is recorded as `gen_ai.tool.type`, and `attributes` adds your own. `user` and `activity_id` set the Statsig context for the call, as `statsig_context` does. If the function raises, the span records the exception and an `ERROR` status, and the exception is raised again. Like a wrapped model call, each call also logs a `statsig::gen_ai` event with the span's attributes.

To trace a block of code instead of a function, run it inside `start_workflow`. It opens an `invoke_workflow` span marked `statsig.span.llm_root=1`, and takes `user`, `activity_id` and `attributes`. It works in sync code and inside `async def`, and an exception raised in the block is recorded on the span and propagates.

```python
from statsig_ai import StatsigUser, start_workflow

async def handle_message(user: StatsigUser, question: str) -> str:
    with start_workflow("support_turn", user=user, attributes={"channel": "chat"}):
        docs = search_docs(question)
        response = await openai.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": f"{question}\n\n{docs}"}],
        )
        return response.choices[0].message.content
```

#### Marking the LLM root

`statsig.span.llm_root=1` marks the span that stands for a whole turn, so online evals don't grade each model round inside a turn as a turn of its own:

- A `start_workflow` span is the root, and the model calls inside it aren't.
- Model calls inside a `wrap(type="workflow")` span aren't the root. The workflow span isn't either, so use `start_workflow` for the outermost span of a turn.
- A wrapped model call made outside any workflow is the root, including one inside a tool span. Tool spans are never the root.

### Tagging spans with a user

Run a request inside `statsig_user_context` to tag the spans started inside it with the user's `statsig.user_id` and `statsig.custom_ids` (JSON). A user without a user ID is recorded as `unknown`.

```python
from statsig_ai import StatsigUser, statsig_user_context

user = StatsigUser(user_id="user-123", custom_ids={"org_id": "acme"})
with statsig_user_context(user):
    handle_request(request)
```

`statsig_context(StatsigContext(user=..., activity_id=...))` tags wrapped OpenAI calls and `wrap` spans with the same attributes, plus the activity ID in `statsig.custom_ids`.

### Tagging calls with a prompt version

Run the code that renders and sends a prompt inside `prompt.with_live()`, or `prompt.with_version(version)` for a Candidate. Wrapped OpenAI calls and `wrap` spans inside it get `statsig.llm.prompt_name`, `statsig.llm.prompt_version` and `statsig.llm.prompt_version_id`, which Statsig uses to attribute online eval grades to a version. Spans inside it are also tagged with the user you passed to `get_prompt`, as with `statsig_user_context`.

The variables you pass to `get_prompt_messages` are recorded as `statsig.llm.prompt_template_variables` (JSON), so the request can be re-run on another version. Model call spans only record them when input capture is on (`capture_all` or `capture_input_messages`), and only for the version the scope is for. Render the prompt inside the scope, before the call. Workflow spans record the variables rendered inside them, so start the workflow inside the version scope.

```python
from statsig_ai import GenAICaptureOptions, WrapOpenAIOptions, wrap_openai

openai = wrap_openai(OpenAI(), WrapOpenAIOptions(GenAICaptureOptions(capture_input_messages=True)))
prompt = statsig_ai.get_prompt(user, "support_bot")

with prompt.with_live() as version:
    response = openai.chat.completions.create(
        model=version.get_model(),
        messages=version.get_prompt_messages({"input": user_input}),
    )
```

To tag spans with the prompt name and user but no version, use `with prompt.with_context():`.

### Marking the environment

The `environment` option of `InitializeTracingOptions` (defaults to `PYTHON_ENV`) is recorded on the resource as `env` and `deployment.environment.name`. To tag only some spans, for example evals running in the same process as production code, run that code inside `statsig_environment`. Spans started inside it get a `statsig.environment` attribute, which overrides the resource value for those spans.

Use `production` for live traffic. `eval` and `online_eval_judge` are reserved for spans Statsig generates.

```python
from statsig_ai import InitializeTracingOptions, initialize_tracing, statsig_environment

initialize_tracing(InitializeTracingOptions(environment="production"))

with statsig_environment("staging_canary"):
    handle_request(request)
```

### Joining a Statsig eval trace

A workflow eval sends each dataset row to your endpoint as a POST request. To show your agent's model calls under the eval row in the trace drawer, run the handler inside `incoming_trace_context`. It reads the W3C `traceparent` and `tracestate` request headers, so wrapped OpenAI calls and other spans become children of the eval row's span. It accepts Flask's or FastAPI's `request.headers`, or a dict of header names (matched case-insensitively) to a value or a list of values. If there's no valid `traceparent`, the handler runs as usual.

```python
from statsig_ai import incoming_trace_context

# Flask
@app.post("/agent")
def agent():
    with incoming_trace_context(request.headers):
        return {"output": run_agent(request.get_json())}

# FastAPI
@app.post("/agent")
async def agent(request: Request):
    with incoming_trace_context(request.headers):
        return {"output": await run_agent(await request.json())}
```

If your server already uses OpenTelemetry HTTP instrumentation, its spans join the trace on their own. `incoming_trace_context` keeps the active span when it's already in the incoming trace, so using both is safe.

What Statsig sends: workflow eval requests carry `Content-Type: application/json`, the headers you configured for the workflow, and two headers of their own. `traceparent` names the eval row's root span, and `x-statsig-eval-run` holds the eval run ID. A header you configure with the same name wins, matched case-insensitively. Test Connection doesn't send either header.

Spans from your endpoint get your app's environment, so eval requests land with your production traces, where online evals can grade them. To keep them out, check for `x-statsig-eval-run` and run the handler inside `statsig_environment("eval")`. You can use the same check to skip side effects.

```python
from contextlib import nullcontext

from statsig_ai import incoming_trace_context, statsig_environment

@app.post("/agent")
def agent():
    is_eval = "x-statsig-eval-run" in request.headers
    with incoming_trace_context(request.headers):
        with statsig_environment("eval") if is_eval else nullcontext():
            return {"output": run_agent(request.get_json())}
```

## Running Examples

The `examples/` directory contains working examples demonstrating various features of the SDK.

### Prerequisites

Set up your environment variables:
```bash
export STATSIG_API_KEY="your-statsig-api-key"
export OPENAI_API_KEY="your-openai-api-key"  # For OpenAI wrapper examples
```

### Running an Example

```bash
python examples/ai_wrappers/wrap_openai.py
```

Available examples:
- `examples/ai_wrappers/wrap_openai.py` - Demonstrates wrapping OpenAI clients with Statsig telemetry (sync/async, streaming/non-streaming)

## Running evals on a Statsig dataset

Pass `statsig_dataset(name, to_record)` as `Eval()`'s `data` to run on an eval dataset from the console. `to_record` says which columns hold the input and the expected value. The run names the dataset and each result keeps its dataset row id, so the Eval page compares runs on the dataset row by row.

```python
from statsig_ai import Eval, statsig_dataset

Eval(
    "support_bot_quality",
    data=statsig_dataset(
        "support_cases",
        lambda row: {
            "input": row.columns["question"],
            "expected": row.columns["answer"],
            "category": row.groups,
        },
    ),
    task=lambda input: answer(input),
    scorer=lambda args: args.output == args.expected,
)
```

To post your own records for a dataset instead, set `dataset` and give each record the `row_id` of its dataset row.

## Community

If you need any assistance or just have a question, feel free to reach out to us on [Slack](https://statsig.com/community?ref=gh_jsm).