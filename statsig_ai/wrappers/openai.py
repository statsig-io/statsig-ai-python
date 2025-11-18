import logging
from typing import Any, Optional
from .openai_impl import OpenAIWrapper
from .configs import WrapOpenAIOptions


def wrap_openai(oai_client: Any, options: Optional[WrapOpenAIOptions] = None):
    if options is None:
        options = WrapOpenAIOptions()
    if hasattr(oai_client, "chat") and hasattr(oai_client.chat, "completions"):
        return OpenAIWrapper(oai_client, options)
    logging.warning("[Statsig] Object does not have OpenAI APIs, not wrapping.")
    return None
