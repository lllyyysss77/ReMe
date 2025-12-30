"""Synchronous LiteLLM-based LLM implementation for the ReMe framework.

This module provides a unified synchronous interface for 100+ LLM providers via LiteLLM,
supporting streaming completions, tool calling, and reasoning content. For
asynchronous operations, refer to the LiteLLM class in the lite_llm module.
"""

from typing import List, Generator, Optional

import litellm

from .lite_llm import LiteLLM
from ..context import C
from ..enumeration import ChunkEnum
from ..schema import Message
from ..schema import StreamChunk
from ..schema import ToolCall


@C.register_llm("litellm_sync")
class LiteLLMSync(LiteLLM):
    """
    Synchronous LiteLLM client for executing chat completions and streaming responses.

    This class extends the base LiteLLM implementation to provide synchronous
    execution of streaming methods, inheriting initialization and configuration
    logic from the parent class.

    Example:
        >>> llm = LiteLLMSync(
        ...     model_name="qwen3-max",
        ...     api_key="sk-...",
        ...     temperature=0.7
        ... )
        >>> messages = [Message(role=Role.USER, content="Hello!")]
        >>> for chunk in llm.chat(messages):
        ...     print(chunk)
    """

    def _stream_chat_sync(
        self,
        messages: List[Message],
        tools: Optional[List[ToolCall]] = None,
        stream_kwargs: Optional[dict] = None,
    ) -> Generator[StreamChunk, None, None]:
        """
        Internal synchronous generator for processing streaming chat completion chunks.

        This method orchestrates the LiteLLM completion lifecycle by categorizing
        raw API chunks into usage data, reasoning content (thinking), regular
        text responses, and aggregated tool calls.

        Args:
            messages: List of conversation messages to send to the model.
            tools: Optional list of tool definitions available for the model to call.
            stream_kwargs: Dictionary of pre-built parameters for the LiteLLM API.

        Yields:
            StreamChunk: Wrapped response fragments categorized by ChunkEnum.

        Raises:
            ValueError: If tool call arguments fail validation or serialization.
        """
        # Create streaming completion request using LiteLLM
        stream_kwargs = stream_kwargs or {}
        completion = litellm.completion(**stream_kwargs)

        # Track accumulated tool calls across chunks
        ret_tools: List[ToolCall] = []
        # Flag to track if we've started receiving answer content
        is_answering: bool = False

        for chunk in completion:
            # Handle usage information (typically the last chunk)
            if not chunk.choices:
                if hasattr(chunk, "usage") and chunk.usage:
                    yield StreamChunk(chunk_type=ChunkEnum.USAGE, chunk=chunk.usage.model_dump())

            else:
                delta = chunk.choices[0].delta

                # Check for reasoning content (models that support thinking)
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    yield StreamChunk(chunk_type=ChunkEnum.THINK, chunk=delta.reasoning_content)

                else:
                    if not is_answering:
                        is_answering = True

                    # Yield regular text content
                    if delta.content is not None:
                        yield StreamChunk(chunk_type=ChunkEnum.ANSWER, chunk=delta.content)

                    # Process tool calls - LiteLLM streams them incrementally
                    if hasattr(delta, "tool_calls") and delta.tool_calls is not None:
                        for tool_call in delta.tool_calls:
                            self._accumulate_tool_call_chunk(tool_call, ret_tools)

        # After streaming completes, validate and yield complete tool calls
        for tool_data in self._validate_and_serialize_tools(ret_tools, tools):
            yield StreamChunk(chunk_type=ChunkEnum.TOOL, chunk=tool_data)
