"""A duck-typed stand-in for the Anthropic SDK's `Anthropic` and `AsyncAnthropic` clients, so the
wrapper tests don't need `anthropic` installed.

Like the SDK, responses and stream events are objects with `to_dict()`, and `messages.stream()`
returns a manager whose stream reads its events from `_raw_stream`.
"""

import copy
import json
from typing import Any, Callable, Dict, List, Optional


class SDKObject:
    """An SDK response object: attribute access, plus to_dict() like the SDK's pydantic models."""

    def __init__(self, fields: Dict[str, Any]) -> None:
        self._fields = fields

    def __getattr__(self, name: str) -> Any:
        try:
            return self._fields[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._fields)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SDKObject) and other._fields == self._fields


class RawStream:
    """The SDK's Stream: an iterator over the SSE events, with a `response` and close()."""

    def __init__(self, events: List[Dict[str, Any]], fail_after: Optional[int] = None) -> None:
        self._events = events
        self._fail_after = fail_after
        self._iterator = self._read()
        self.response = SDKObject({"status_code": 200})
        self.closed = False

    def _read(self):
        for index, event in enumerate(self._events):
            if self._fail_after is not None and index == self._fail_after:
                raise ConnectionError("stream interrupted")
            yield SDKObject(event)

    def __iter__(self):
        return self._iterator

    def __next__(self):
        return next(self._iterator)

    def close(self) -> None:
        self.closed = True


class AsyncRawStream:
    """The SDK's AsyncStream."""

    def __init__(self, events: List[Dict[str, Any]], fail_after: Optional[int] = None) -> None:
        self._events = RawStream(events, fail_after)
        self._iterator = self._read()
        self.response = self._events.response
        self.closed = False

    async def _read(self):
        for event in self._events:
            yield event

    def __aiter__(self):
        return self._iterator

    async def __anext__(self):
        return await self._iterator.__anext__()

    async def close(self) -> None:
        self.closed = True


class MessageStream:
    """The SDK's MessageStream: every way of reading it reads `_raw_stream`."""

    def __init__(self, raw_stream: RawStream, final_message: Dict[str, Any]) -> None:
        self._raw_stream = raw_stream
        self._final_message = final_message
        self._iterator = self._read()

    def _read(self):
        for event in self._raw_stream:
            yield event

    def __iter__(self):
        return self._iterator

    def __next__(self):
        return next(self._iterator)

    @property
    def text_stream(self):
        for event in self:
            if event.type == "content_block_delta" and event.delta["type"] == "text_delta":
                yield event.delta["text"]

    def get_final_message(self) -> SDKObject:
        for _ in self:
            pass
        return SDKObject(self._final_message)

    def close(self) -> None:
        self._raw_stream.close()


class AsyncMessageStream:
    """The SDK's AsyncMessageStream."""

    def __init__(self, raw_stream: AsyncRawStream, final_message: Dict[str, Any]) -> None:
        self._raw_stream = raw_stream
        self._final_message = final_message
        self._iterator = self._read()

    async def _read(self):
        async for event in self._raw_stream:
            yield event

    def __aiter__(self):
        return self._iterator

    async def __anext__(self):
        return await self._iterator.__anext__()

    async def get_final_message(self) -> SDKObject:
        async for _ in self:
            pass
        return SDKObject(self._final_message)

    async def close(self) -> None:
        await self._raw_stream.close()


class MessageStreamManager:
    def __init__(self, open_stream: Callable[[], Any]) -> None:
        self._open_stream = open_stream
        self.stream: Any = None

    def __enter__(self) -> MessageStream:
        self.stream = self._open_stream()
        return self.stream

    def __exit__(self, *exc_info: Any) -> None:
        if self.stream is not None:
            self.stream.close()


class AsyncMessageStreamManager:
    def __init__(self, open_stream: Callable[[], Any]) -> None:
        self._open_stream = open_stream
        self.stream: Any = None

    async def __aenter__(self) -> AsyncMessageStream:
        self.stream = self._open_stream()
        return self.stream

    async def __aexit__(self, *exc_info: Any) -> None:
        if self.stream is not None:
            await self.stream.close()


class Messages:
    """`client.messages`. Records each request and answers with `response`, or the events of
    `stream_events` for a streamed call. A set `error` is raised instead."""

    def __init__(self, response: Dict[str, Any], stream_events: List[Dict[str, Any]]) -> None:
        self.response = response
        self.stream_events = stream_events
        self.error: Optional[Exception] = None
        self.fail_stream_after: Optional[int] = None
        self.on_call: Optional[Callable[[], None]] = None
        self.requests: List[Dict[str, Any]] = []
        self.raw_streams: List[Any] = []

    def count_tokens(self, **_kwargs: Any) -> Dict[str, int]:
        return {"input_tokens": 3}

    def _respond(self, kwargs: Dict[str, Any]) -> Any:
        self.requests.append(kwargs)
        if self.on_call is not None:
            self.on_call()
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            return self._raw_stream()
        return SDKObject(self.response)

    def _raw_stream(self) -> Any:
        raw_stream = RawStream(self.stream_events, self.fail_stream_after)
        self.raw_streams.append(raw_stream)
        return raw_stream

    def create(self, **kwargs: Any) -> Any:
        return self._respond(kwargs)

    def stream(self, **kwargs: Any) -> MessageStreamManager:
        return MessageStreamManager(
            lambda: MessageStream(self._respond({**kwargs, "stream": True}), self.response)
        )


class AsyncMessages(Messages):
    def _raw_stream(self) -> Any:
        raw_stream = AsyncRawStream(self.stream_events, self.fail_stream_after)
        self.raw_streams.append(raw_stream)
        return raw_stream

    async def create(self, **kwargs: Any) -> Any:
        return self._respond(kwargs)

    def stream(self, **kwargs: Any) -> AsyncMessageStreamManager:
        return AsyncMessageStreamManager(
            lambda: AsyncMessageStream(self._respond({**kwargs, "stream": True}), self.response)
        )


class MockAnthropic:
    def __init__(self, response: Dict[str, Any], stream_events: List[Dict[str, Any]]) -> None:
        self.api_key = "test-key"
        self.messages = Messages(response, stream_events)


class MockAsyncAnthropic:
    def __init__(self, response: Dict[str, Any], stream_events: List[Dict[str, Any]]) -> None:
        self.api_key = "test-key"
        self.messages = AsyncMessages(response, stream_events)


def stream_events_for(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The events the API streams for `message`: text and thinking in two deltas each, and tool
    input as partial JSON."""
    usage = message["usage"]
    events: List[Dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                **message,
                "content": [],
                "stop_reason": None,
                "usage": {**usage, "output_tokens": 1},
            },
        }
    ]
    for index, block in enumerate(message["content"]):
        events.extend(_block_events(index, block))
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
            # Like the API's newer versions, later counts can be null.
            "usage": {"output_tokens": usage["output_tokens"], "input_tokens": None},
        }
    )
    events.append({"type": "message_stop"})
    return events


def _block_events(index: int, block: Dict[str, Any]) -> List[Dict[str, Any]]:
    block_type = block["type"]
    if block_type == "text":
        start: Dict[str, Any] = {"type": "text", "text": ""}
        deltas = [{"type": "text_delta", "text": part} for part in _halves(block["text"])]
    elif block_type == "thinking":
        start = {"type": "thinking", "thinking": "", "signature": ""}
        deltas = [
            {"type": "thinking_delta", "thinking": part} for part in _halves(block["thinking"])
        ]
    else:
        start = {**block, "input": {}}
        arguments = json.dumps(block["input"])
        deltas = [{"type": "input_json_delta", "partial_json": part} for part in _halves(arguments)]
    return [
        {"type": "content_block_start", "index": index, "content_block": start},
        *({"type": "content_block_delta", "index": index, "delta": delta} for delta in deltas),
        {"type": "content_block_stop", "index": index},
    ]


def _halves(text: str) -> List[str]:
    middle = len(text) // 2
    return [text[:middle], text[middle:]]
