"""token counter"""

from .base_token_counter import BaseTokenCounter
from .hf_token_counter import HFTokenCounter
from .openai_token_counter import OpenAITokenCounter
from ..context import R

__all__ = [
    "BaseTokenCounter",
    "HFTokenCounter",
    "OpenAITokenCounter",
]

R.token_counter.register("base")(BaseTokenCounter)
R.token_counter.register("hf")(HFTokenCounter)
R.token_counter.register("openai")(OpenAITokenCounter)
