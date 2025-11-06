from typing import Any, Optional
from .openai_impl import OpenAIWrapper
from .configs import WrapOpenAIOptions


def wrap_openai(oai_client: Any, options: Optional[WrapOpenAIOptions] = None):
    if options is None:
        options = WrapOpenAIOptions()
    return OpenAIWrapper(oai_client, options)
