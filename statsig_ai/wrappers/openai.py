from .openai_impl import OpenAIWrapper
from .configs import WrapOpenAIOptions
from typing import Any, Optional


def wrap_openai(oai_client: Any, options: Optional[WrapOpenAIOptions] = None):
    if options is None:
        options = WrapOpenAIOptions()
    return OpenAIWrapper(oai_client, options)
