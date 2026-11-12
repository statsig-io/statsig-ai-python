"""Duck-typed stand-ins for a google-genai Client and its response types, so the wrapper tests
don't need google-genai installed. Field names match the SDK's (snake_case), and FinishReason
is a str enum like the SDK's.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence


class FinishReason(str, Enum):
    STOP = "STOP"
    MAX_TOKENS = "MAX_TOKENS"
    SAFETY = "SAFETY"
    RECITATION = "RECITATION"
    BLOCKLIST = "BLOCKLIST"
    PROHIBITED_CONTENT = "PROHIBITED_CONTENT"
    SPII = "SPII"
    OTHER = "OTHER"
    MALFORMED_FUNCTION_CALL = "MALFORMED_FUNCTION_CALL"


@dataclass
class FunctionCall:
    name: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


@dataclass
class FunctionResponse:
    name: Optional[str] = None
    response: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


@dataclass
class Blob:
    mime_type: Optional[str] = None
    data: Optional[bytes] = None


@dataclass
class FileData:
    file_uri: Optional[str] = None
    mime_type: Optional[str] = None


@dataclass
class Part:
    text: Optional[str] = None
    thought: Optional[bool] = None
    function_call: Optional[FunctionCall] = None
    function_response: Optional[FunctionResponse] = None
    inline_data: Optional[Blob] = None
    file_data: Optional[FileData] = None


@dataclass
class Content:
    role: Optional[str] = None
    parts: Optional[List[Any]] = None


@dataclass
class Candidate:
    content: Optional[Content] = None
    finish_reason: Optional[FinishReason] = None


@dataclass
class UsageMetadata:
    prompt_token_count: Optional[int] = None
    candidates_token_count: Optional[int] = None
    thoughts_token_count: Optional[int] = None
    cached_content_token_count: Optional[int] = None
    total_token_count: Optional[int] = None


@dataclass
class GenerateContentResponse:
    response_id: Optional[str] = None
    model_version: Optional[str] = None
    candidates: Optional[List[Candidate]] = None
    usage_metadata: Optional[UsageMetadata] = None

    def __iter__(self) -> Iterator[Any]:
        # Like a pydantic model, the SDK's response iterates over its fields. It isn't a stream.
        return iter(vars(self).items())


def text_response(text: str, finish_reason: Optional[FinishReason] = FinishReason.STOP):
    return GenerateContentResponse(
        response_id="resp_1",
        model_version="gemini-test-001",
        candidates=[
            Candidate(
                content=Content(role="model", parts=[Part(text=text)]),
                finish_reason=finish_reason,
            )
        ],
        usage_metadata=UsageMetadata(prompt_token_count=4, candidates_token_count=1),
    )


@dataclass
class Behavior:
    """What the mock models return: a response, stream chunks, or an error. `on_call` runs
    inside the call, as the SDK's HTTP request would.
    """

    response: Any = None
    chunks: Sequence[Any] = ()
    error: Optional[BaseException] = None
    stream_error: Optional[BaseException] = None
    on_call: Optional[Callable[[], None]] = None
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def call(self, kwargs: Dict[str, Any]) -> None:
        self.calls.append(kwargs)
        if self.on_call is not None:
            self.on_call()
        if self.error is not None:
            raise self.error


class MockModels:
    def __init__(self, behavior: Behavior) -> None:
        self.behavior = behavior

    def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
        self.behavior.call({"model": model, "contents": contents, "config": config})
        return self.behavior.response

    def generate_content_stream(
        self, *, model: str, contents: Any, config: Any = None
    ) -> Iterator[Any]:
        # Like the SDK's, a generator: the request is sent once the caller starts iterating.
        self.behavior.call({"model": model, "contents": contents, "config": config})
        yield from self.behavior.chunks
        if self.behavior.stream_error is not None:
            raise self.behavior.stream_error

    def get(self, *, model: str) -> str:
        return f"model:{model}"


class MockAsyncModels:
    def __init__(self, behavior: Behavior) -> None:
        self.behavior = behavior

    async def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
        self.behavior.call({"model": model, "contents": contents, "config": config})
        return self.behavior.response

    async def generate_content_stream(self, *, model: str, contents: Any, config: Any = None):
        self.behavior.call({"model": model, "contents": contents, "config": config})
        behavior = self.behavior

        async def stream():
            for chunk in behavior.chunks:
                yield chunk
            if behavior.stream_error is not None:
                raise behavior.stream_error

        return stream()


class MockAio:
    def __init__(self, behavior: Behavior) -> None:
        self.models = MockAsyncModels(behavior)
        self.files = "async-files-api"


class MockGeminiClient:
    def __init__(self, behavior: Optional[Behavior] = None) -> None:
        self.behavior = behavior or Behavior()
        self.models = MockModels(self.behavior)
        self.aio = MockAio(self.behavior)
        self.files = "files-api"
