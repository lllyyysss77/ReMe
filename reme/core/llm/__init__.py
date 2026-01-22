"""llm"""

from .base_llm import BaseLLM
from .lite_llm import LiteLLM
from .lite_llm_sync import LiteLLMSync
from .openai_llm import OpenAILLM
from .openai_llm_sync import OpenAILLMSync
from ..context import R

__all__ = [
    "BaseLLM",
    "LiteLLM",
    "LiteLLMSync",
    "OpenAILLM",
    "OpenAILLMSync",
]

R.llm.register("litellm")(LiteLLM)
R.llm.register("litellm_sync")(LiteLLMSync)
R.llm.register("openai")(OpenAILLM)
R.llm.register("openai_sync")(OpenAILLMSync)
