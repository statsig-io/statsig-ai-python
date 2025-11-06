import asyncio
from datetime import datetime


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


class MockOpenAI:
    def __init__(self):
        self.chat = self.ChatNamespace()
        self.completions = self.CompletionsNamespace()
        self.embeddings = self.EmbeddingsNamespace()
        self.images = self.ImagesNamespace()
        self.moderations = self.ModerationsNamespace()

    class ChatNamespace:
        class CompletionsNamespace:
            def create(self, *args, **kwargs):
                return DefaultMockResponses.chat_completion

        def __init__(self):
            self.completions = self.CompletionsNamespace()

    class CompletionsNamespace:
        def create(self, *args, **kwargs):
            return DefaultMockResponses.text_completion

    class EmbeddingsNamespace:
        def create(self, *args, **kwargs):
            return DefaultMockResponses.embedding

    class ImagesNamespace:
        def generate(self, *args, **kwargs):
            return DefaultMockResponses.image

    class ModerationsNamespace:
        def create(self, *args, **kwargs):
            return DefaultMockResponses.moderation
