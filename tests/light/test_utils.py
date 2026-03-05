"""Test utilities for copaw tests."""

import os


def get_token_counter():
    """Get HF token counter instance."""
    from reme.core.utils import get_hf_token_counter

    return get_hf_token_counter()


def get_dash_chat_model(model_name: str = "qwen3.5-plus"):
    """Get DashScope chat model instance."""
    from agentscope.model import OpenAIChatModel
    from reme.core.utils import load_env

    load_env()
    return OpenAIChatModel(
        api_key=os.environ["LLM_API_KEY"],
        client_kwargs={"base_url": os.environ["LLM_BASE_URL"]},
        model_name=model_name,
    )


def get_formatter():
    """Get formatter instance."""
    from agentscope.formatter import OpenAIChatFormatter

    return OpenAIChatFormatter()
