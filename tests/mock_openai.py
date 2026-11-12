import asyncio
import functools
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from openai import omit
from openai._models import construct_type
from openai.lib.streaming.responses import AsyncResponseStreamManager, ResponseStreamManager
from openai.types.responses import Response, ResponseStreamEvent


class DefaultMockResponses:
    chat_completion = {
        "id": "cmpl-mock",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "This is a mock chat response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
        },
    }

    chat_chunks = [
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "This is "}}],
        },
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": "gpt-4",
            "choices": [
                {"index": 0, "delta": {"content": "a mock stream"}, "finish_reason": "stop"}
            ],
        },
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": "gpt-4",
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
        },
    ]

    text_completion = {
        "id": "cmpl-mock",
        "object": "text_completion",
        "created": int(datetime.now().timestamp()),
        "model": "gpt-3.5-turbo-instruct",
        "choices": [
            {"text": "This is a mock text completion", "index": 0, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
        },
    }

    text_completion_chunks = [
        {
            "id": "cmpl-mock",
            "object": "text_completion",
            "model": "gpt-3.5-turbo-instruct",
            "choices": [{"text": "Hi ", "index": 0, "finish_reason": None}],
        },
        {
            "id": "cmpl-mock",
            "object": "text_completion",
            "model": "gpt-3.5-turbo-instruct",
            "choices": [{"text": "there", "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        },
    ]

    embedding = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.01] * 1536,
                "index": 0,
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }

    image = {
        "created": int(datetime.now().timestamp()),
        "data": [{"url": "https://example.com/mock-image.png"}],
    }

    response = {
        "id": "resp-mock",
        "object": "response",
        "created_at": int(datetime.now().timestamp()),
        "model": "gpt-4.1",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg-mock",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "This is a mock response", "annotations": []}
                ],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
    }

    moderation = {
        "id": "modr-mock",
        "model": "text-moderation-007",
        "results": [
            {
                "flagged": False,
                "categories": {
                    "sexual": False,
                    "hate": False,
                    "harassment": False,
                    "self-harm": False,
                    "sexual/minors": False,
                    "hate/threatening": False,
                    "violence/graphic": False,
                    "self-harm/intent": False,
                    "self-harm/instructions": False,
                    "harassment/threatening": False,
                    "violence": False,
                },
                "category_scores": {
                    k: 0.0
                    for k in [
                        "sexual",
                        "hate",
                        "harassment",
                        "self-harm",
                        "sexual/minors",
                        "hate/threatening",
                        "violence/graphic",
                        "self-harm/intent",
                        "self-harm/instructions",
                        "harassment/threatening",
                        "violence",
                    ]
                },
            }
        ],
    }


def responses_stream_events(response: Dict[str, Any]) -> List[Any]:
    """The Responses API stream events that end with `response`, as the OpenAI SDK parses them.

    Message text and function call arguments are streamed in two deltas each.
    """
    events: List[Dict[str, Any]] = [
        {
            "type": "response.created",
            "response": {**response, "status": "in_progress", "output": [], "usage": None},
        }
    ]
    for index, item in enumerate(response["output"]):
        if item["type"] == "message":
            events.append(
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {**item, "status": "in_progress", "content": []},
                }
            )
            for content_index, part in enumerate(item["content"]):
                events.append(
                    {
                        "type": "response.content_part.added",
                        "output_index": index,
                        "content_index": content_index,
                        "item_id": item["id"],
                        "part": {**part, "text": ""},
                    }
                )
                for delta in _halves(part["text"]):
                    events.append(
                        {
                            "type": "response.output_text.delta",
                            "output_index": index,
                            "content_index": content_index,
                            "item_id": item["id"],
                            "delta": delta,
                            "logprobs": [],
                        }
                    )
        elif item["type"] == "function_call":
            events.append(
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {**item, "status": "in_progress", "arguments": ""},
                }
            )
            for delta in _halves(item["arguments"]):
                events.append(
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": index,
                        "item_id": item["id"],
                        "delta": delta,
                    }
                )
        events.append({"type": "response.output_item.done", "output_index": index, "item": item})
    events.append({"type": "response.completed", "response": response})
    return [
        construct_type(type_=ResponseStreamEvent, value={**event, "sequence_number": number})
        for number, event in enumerate(events)
    ]


def _halves(text: str) -> List[str]:
    middle = len(text) // 2
    return [text[:middle], text[middle:]]


class MockStream:
    """Stands in for openai.Stream. Yields the chunks, then raises `error` if one is given."""

    def __init__(self, chunks: List[Any], error: Optional[BaseException] = None) -> None:
        self.chunks = chunks
        self.error = error
        self.closed = False
        # openai's ResponseStream closes the HTTP response it reads.
        self.response = self

    def __iter__(self):
        yield from self.chunks
        if self.error is not None:
            raise self.error

    def close(self) -> None:
        self.closed = True


class AsyncMockStream(MockStream):
    """Stands in for openai.AsyncStream."""

    async def __aiter__(self):
        for chunk in self.chunks:
            await asyncio.sleep(0)
            yield chunk
        if self.error is not None:
            raise self.error

    async def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


class _Requests:
    """What each mock API call does before it returns: records its keyword arguments, calls
    `on_request` (the point where the OpenAI SDK sends the HTTP request), then raises `error`
    if one is given."""

    def __init__(
        self, error: Optional[BaseException], on_request: Optional[Callable[[], None]]
    ) -> None:
        self.error = error
        self.on_request = on_request

    def send(self, calls: List[Dict[str, Any]], kwargs: Dict[str, Any]) -> None:
        calls.append(kwargs)
        if self.on_request is not None:
            self.on_request()
        if self.error is not None:
            raise self.error


class MockOpenAI:
    """Returns the default mock responses, or the ones given.

    `stream=True` calls return a MockStream of `chat_chunks`, or of `response_events` for the
    Responses API (the events of `responses_stream_events(response)` by default).
    `responses.stream()` returns the OpenAI SDK's stream manager reading those events.
    `chat.completions.calls`, `responses.calls` and `images.calls` record the keyword arguments
    of each call.
    """

    def __init__(
        self,
        chat_completion=None,
        response=None,
        *,
        chat_chunks: Optional[List[Any]] = None,
        response_events: Optional[List[Any]] = None,
        image=None,
        stream_error: Optional[BaseException] = None,
        error: Optional[BaseException] = None,
        on_request: Optional[Callable[[], None]] = None,
    ):
        requests = _Requests(error, on_request)
        self.chat = self.ChatNamespace(
            requests, chat_completion, chat_chunks or DefaultMockResponses.chat_chunks, stream_error
        )
        self.completions = self.CompletionsNamespace()
        self.embeddings = self.EmbeddingsNamespace()
        self.images = self.ImagesNamespace(requests, image)
        self.moderations = self.ModerationsNamespace()
        self.responses = self.ResponsesNamespace(requests, response, response_events, stream_error)

    class ChatNamespace:
        class CompletionsNamespace:
            def __init__(self, requests, chat_completion, chunks, stream_error):
                self.chat_completion = chat_completion or DefaultMockResponses.chat_completion
                self.chunks = chunks
                self.stream_error = stream_error
                self.requests = requests
                self.calls: List[Dict[str, Any]] = []
                self.streams: List[MockStream] = []

            def create(self, *args, **kwargs):
                self.requests.send(self.calls, kwargs)
                if kwargs.get("stream"):
                    self.streams.append(MockStream(self.chunks, self.stream_error))
                    return self.streams[-1]
                return self.chat_completion

        def __init__(self, requests=None, chat_completion=None, chunks=None, stream_error=None):
            self.completions = self.CompletionsNamespace(
                requests or _Requests(None, None), chat_completion, chunks, stream_error
            )

    class ResponsesNamespace:
        def __init__(self, requests=None, response=None, events=None, stream_error=None):
            self.response = response or DefaultMockResponses.response
            self.events = events
            self.stream_error = stream_error
            self.requests = requests or _Requests(None, None)
            self.calls: List[Dict[str, Any]] = []
            self.streams: List[MockStream] = []

        def create(self, *args, **kwargs):
            self.requests.send(self.calls, kwargs)
            if kwargs.get("stream"):
                events = self.events or responses_stream_events(_as_dict(self.response))
                self.streams.append(MockStream(events, self.stream_error))
                return self.streams[-1]
            return self.response

        def parse(self, *args, **kwargs):
            self.requests.send(self.calls, kwargs)
            return construct_type(type_=Response, value=_as_dict(self.response))

        def stream(self, *args, **kwargs):
            return ResponseStreamManager(
                functools.partial(self.create, stream=True, **kwargs),
                text_format=kwargs.get("text_format", omit),
                input_tools=kwargs.get("tools", omit),
                starting_after=None,
            )

    class CompletionsNamespace:
        def create(self, *args, **kwargs):
            if kwargs.get("stream"):
                return MockStream(DefaultMockResponses.text_completion_chunks)
            return DefaultMockResponses.text_completion

    class EmbeddingsNamespace:
        def create(self, *args, **kwargs):
            return DefaultMockResponses.embedding

    class ImagesNamespace:
        def __init__(self, requests=None, image=None):
            self.image = image or DefaultMockResponses.image
            self.requests = requests or _Requests(None, None)
            self.calls: List[Dict[str, Any]] = []

        def generate(self, *args, **kwargs):
            self.requests.send(self.calls, kwargs)
            return self.image

    class ModerationsNamespace:
        def create(self, *args, **kwargs):
            return DefaultMockResponses.moderation


class AsyncMockOpenAI:
    """The async version of MockOpenAI, shaped like openai.AsyncOpenAI.

    Its methods are coroutines that make the request when they're awaited. `stream=True` calls
    return an AsyncMockStream.
    """

    def __init__(self, *args, **kwargs):
        self.sync = MockOpenAI(*args, **kwargs)
        self.chat = self.ChatNamespace(self.sync.chat.completions)
        self.images = self.ImagesNamespace(self.sync.images)
        self.embeddings = self.EmbeddingsNamespace(self.sync.embeddings)
        self.responses = self.ResponsesNamespace(self.sync.responses)

    class ChatNamespace:
        class CompletionsNamespace:
            def __init__(self, sync):
                self.sync = sync
                self.calls = sync.calls

            async def create(self, *args, **kwargs):
                await asyncio.sleep(0)
                return _to_async(self.sync, self.sync.create(*args, **kwargs))

        def __init__(self, sync_completions):
            self.completions = self.CompletionsNamespace(sync_completions)

    class ResponsesNamespace:
        def __init__(self, sync):
            self.sync = sync
            self.calls = sync.calls

        async def create(self, *args, **kwargs):
            await asyncio.sleep(0)
            return _to_async(self.sync, self.sync.create(*args, **kwargs))

        async def parse(self, *args, **kwargs):
            await asyncio.sleep(0)
            return self.sync.parse(*args, **kwargs)

        def stream(self, *args, **kwargs):
            return AsyncResponseStreamManager(
                self.create(stream=True, **kwargs),
                text_format=kwargs.get("text_format", omit),
                input_tools=kwargs.get("tools", omit),
                starting_after=None,
            )

    class ImagesNamespace:
        def __init__(self, sync):
            self.sync = sync
            self.calls = sync.calls

        async def generate(self, *args, **kwargs):
            await asyncio.sleep(0)
            return self.sync.generate(*args, **kwargs)

    class EmbeddingsNamespace:
        def __init__(self, sync):
            self.sync = sync

        async def create(self, *args, **kwargs):
            await asyncio.sleep(0)
            return self.sync.create(*args, **kwargs)


def _to_async(sync_namespace, result):
    if isinstance(result, MockStream):
        result = AsyncMockStream(result.chunks, result.error)
        sync_namespace.streams[-1] = result
    return result


def _as_dict(response) -> Dict[str, Any]:
    return response if isinstance(response, dict) else response.to_dict()
