"""Abstract base interface for ReMe LLM implementations."""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Callable, Generator, AsyncGenerator, Any

from loguru import logger

from ..enumeration import ChunkEnum, Role
from ..schema import Message
from ..schema import StreamChunk
from ..schema import ToolCall


class BaseLLM(ABC):
    """Abstract base class defining the standard interface for LLM interactions."""

    def __init__(self, model_name: str, max_retries: int = 10, raise_exception: bool = False, request_interval: float = 0.0, **kwargs):
        """Initialize the LLM client with model configurations and retry policies.

        Args:
            model_name: The name of the model to use
            max_retries: Maximum number of retry attempts on failure
            raise_exception: Whether to raise exceptions or return default values
            request_interval: Minimum time interval (in seconds) between consecutive requests. Default is 0.0 (no interval).
            **kwargs: Additional model-specific parameters
        """
        self.model_name: str = model_name
        self.max_retries: int = max_retries
        self.raise_exception: bool = raise_exception
        self.request_interval: float = request_interval
        self.kwargs: dict = kwargs

        # Request rate control for async operations
        self._last_request_time: float = 0.0
        self._request_lock: asyncio.Lock = asyncio.Lock()

    @staticmethod
    def _accumulate_tool_call_chunk(tool_call, ret_tools: list[ToolCall]):
        """Assemble incremental tool call fragments into complete ToolCall objects."""
        index = tool_call.index

        # Ensure we have a ToolCall object at this index
        while len(ret_tools) <= index:
            ret_tools.append(ToolCall(index=index))

        # Accumulate tool call parts (id, name, arguments)
        if tool_call.id:
            ret_tools[index].id += tool_call.id

        if tool_call.function and tool_call.function.name:
            ret_tools[index].name += tool_call.function.name

        if tool_call.function and tool_call.function.arguments:
            ret_tools[index].arguments += tool_call.function.arguments

    @staticmethod
    def _validate_and_serialize_tools(ret_tool_calls: list[ToolCall], tools: list[ToolCall]) -> list[dict]:
        """Validate tool call integrity and return serialized tool dictionaries."""
        if not ret_tool_calls:
            return []

        tool_dict: dict[str, ToolCall] = {x.name: x for x in tools} if tools else {}
        validated_tools = []

        for tool in ret_tool_calls:
            if tool.name not in tool_dict:
                continue

            # First try sanitizing arguments
            if not tool.sanitize_and_check_argument():
                logger.error(f"Tool call {tool.name} has invalid JSON arguments after sanitization attempt: {tool.arguments}")
                raise ValueError(f"Tool call {tool.name} has invalid JSON arguments: {tool.arguments}")

            validated_tools.append(tool.simple_output_dump())
        return validated_tools

    @abstractmethod
    def _build_stream_kwargs(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        log_params: bool = True,
        model_name: str | None = None,
        **kwargs,
    ) -> dict:
        """Construct provider-specific parameters for streaming API requests.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool calls
            log_params: Whether to log parameters
            model_name: Optional model name to override self.model_name
            **kwargs: Additional parameters
        """

    async def _stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None,
        stream_kwargs: dict,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Internal async generator for streaming raw response chunks."""
        raise NotImplementedError

    def _stream_chat_sync(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        stream_kwargs: dict | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """Internal synchronous generator for streaming raw response chunks."""
        raise NotImplementedError

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        model_name: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Public async interface for streaming chat completions with retries.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool calls
            model_name: Optional model name to override self.model_name
            **kwargs: Additional parameters
        """
        # Apply request rate limiting if configured
        if self.request_interval > 0:
            async with self._request_lock:
                current_time = time.time()
                elapsed = current_time - self._last_request_time
                if elapsed < self.request_interval:
                    sleep_time = self.request_interval - elapsed
                    await asyncio.sleep(sleep_time)
                self._last_request_time = time.time()

        async for chunk in self._stream_chat_impl(messages, tools, model_name, **kwargs):
            yield chunk

    async def _stream_chat_impl(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        model_name: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Internal implementation of stream_chat with retry logic."""
        stream_kwargs = self._build_stream_kwargs(messages, tools, model_name=model_name, **kwargs)

        for i in range(self.max_retries):
            try:
                async for chunk in self._stream_chat(messages=messages, tools=tools, stream_kwargs=stream_kwargs):
                    yield chunk
                return

            except Exception as e:
                logger.exception(f"stream chat with model={self.model_name} encounter error with e={e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e
                    yield StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                    return

                yield StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                await asyncio.sleep(i + 1)

    def stream_chat_sync(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        model_name: str | None = None,
        **kwargs,
    ) -> Generator[StreamChunk, None, None]:
        """Public synchronous interface for streaming chat completions with retries.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool calls
            model_name: Optional model name to override self.model_name
            **kwargs: Additional parameters
        """
        stream_kwargs = self._build_stream_kwargs(messages, tools, model_name=model_name, **kwargs)

        for i in range(self.max_retries):
            try:
                yield from self._stream_chat_sync(messages=messages, tools=tools, stream_kwargs=stream_kwargs)
                return

            except Exception as e:
                logger.exception(f"stream chat sync with model={self.model_name} encounter error with e={e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e
                    yield StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                    return

                yield StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                time.sleep(i + 1)

    async def _chat(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        enable_stream_print: bool = False,
        model_name: str | None = None,
        **kwargs,
    ) -> Message:
        """Internal async method to aggregate a full response by consuming the stream.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool calls
            enable_stream_print: Whether to print stream chunks
            model_name: Optional model name to override self.model_name
            **kwargs: Additional parameters
        """
        state = {
            "enter_think": False,
            "enter_answer": False,
            "reasoning_content": "",
            "answer_content": "",
            "tool_calls": [],
        }

        stream_kwargs = self._build_stream_kwargs(messages, tools, model_name=model_name, **kwargs)
        async for stream_chunk in self._stream_chat(messages=messages, tools=tools, stream_kwargs=stream_kwargs):
            # Process stream chunk
            if stream_chunk.chunk_type is ChunkEnum.USAGE:
                if enable_stream_print:
                    print(
                        f"\n<usage>{json.dumps(stream_chunk.chunk, ensure_ascii=False, indent=2)}</usage>",
                        flush=True,
                    )

            elif stream_chunk.chunk_type is ChunkEnum.THINK:
                if enable_stream_print:
                    if not state["enter_think"]:
                        state["enter_think"] = True
                        print("<think>\n", end="", flush=True)
                    print(stream_chunk.chunk, end="", flush=True)
                state["reasoning_content"] += stream_chunk.chunk

            elif stream_chunk.chunk_type is ChunkEnum.ANSWER:
                if enable_stream_print:
                    if not state["enter_answer"]:
                        state["enter_answer"] = True
                        if state["enter_think"]:
                            print("\n</think>", flush=True)
                    print(stream_chunk.chunk, end="", flush=True)
                state["answer_content"] += stream_chunk.chunk

            elif stream_chunk.chunk_type is ChunkEnum.TOOL:
                if enable_stream_print:
                    print(f"\n<tool>{json.dumps(stream_chunk.chunk, ensure_ascii=False, indent=2)}</tool>", flush=True)
                state["tool_calls"].append(stream_chunk.chunk)

            elif stream_chunk.chunk_type is ChunkEnum.ERROR:
                if enable_stream_print:
                    print(f"\n<error>{stream_chunk.chunk}</error>", flush=True)

        return Message(
            role=Role.ASSISTANT,
            reasoning_content=state["reasoning_content"],
            content=state["answer_content"],
            tool_calls=state["tool_calls"],
        )

    def _chat_sync(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        enable_stream_print: bool = False,
        model_name: str | None = None,
        **kwargs,
    ) -> Message:
        """Internal synchronous method to aggregate a full response by consuming the stream.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool calls
            enable_stream_print: Whether to print stream chunks
            model_name: Optional model name to override self.model_name
            **kwargs: Additional parameters
        """
        state = {
            "enter_think": False,
            "enter_answer": False,
            "reasoning_content": "",
            "answer_content": "",
            "tool_calls": [],
        }

        stream_kwargs = self._build_stream_kwargs(messages, tools, model_name=model_name, **kwargs)
        for stream_chunk in self._stream_chat_sync(messages=messages, tools=tools, stream_kwargs=stream_kwargs):
            # Process stream chunk
            if stream_chunk.chunk_type is ChunkEnum.USAGE:
                if enable_stream_print:
                    print(
                        f"\n<usage>{json.dumps(stream_chunk.chunk, ensure_ascii=False, indent=2)}</usage>",
                        flush=True,
                    )

            elif stream_chunk.chunk_type is ChunkEnum.THINK:
                if enable_stream_print:
                    if not state["enter_think"]:
                        state["enter_think"] = True
                        print("<think>\n", end="", flush=True)
                    print(stream_chunk.chunk, end="", flush=True)
                state["reasoning_content"] += stream_chunk.chunk

            elif stream_chunk.chunk_type is ChunkEnum.ANSWER:
                if enable_stream_print:
                    if not state["enter_answer"]:
                        state["enter_answer"] = True
                        if state["enter_think"]:
                            print("\n</think>", flush=True)
                    print(stream_chunk.chunk, end="", flush=True)
                state["answer_content"] += stream_chunk.chunk

            elif stream_chunk.chunk_type is ChunkEnum.TOOL:
                if enable_stream_print:
                    print(f"\n<tool>{json.dumps(stream_chunk.chunk, ensure_ascii=False, indent=2)}</tool>", flush=True)
                state["tool_calls"].append(stream_chunk.chunk)

            elif stream_chunk.chunk_type is ChunkEnum.ERROR:
                if enable_stream_print:
                    print(f"\n<error>{stream_chunk.chunk}</error>", flush=True)

        return Message(
            role=Role.ASSISTANT,
            reasoning_content=state["reasoning_content"],
            content=state["answer_content"],
            tool_calls=state["tool_calls"],
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        enable_stream_print: bool = False,
        callback_fn: Callable[[Message], Any] | None = None,
        default_value: Any = None,
        model_name: str | None = None,
        **kwargs,
    ) -> Message | Any:
        """Perform an async chat completion with integrated retries and error handling.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool calls
            enable_stream_print: Whether to print stream chunks
            callback_fn: Optional callback function to process the result
            default_value: Default value to return on error
            model_name: Optional model name to override self.model_name
            **kwargs: Additional parameters
        """
        # Apply request rate limiting if configured
        if self.request_interval > 0:
            async with self._request_lock:
                current_time = time.time()
                elapsed = current_time - self._last_request_time
                if elapsed < self.request_interval:
                    sleep_time = self.request_interval - elapsed
                    await asyncio.sleep(sleep_time)
                self._last_request_time = time.time()

        return await self._chat_impl(messages, tools, enable_stream_print, callback_fn, default_value, model_name, **kwargs)

    async def _chat_impl(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        enable_stream_print: bool = False,
        callback_fn: Callable[[Message], Any] | None = None,
        default_value: Any = None,
        model_name: str | None = None,
        **kwargs,
    ) -> Message | Any:
        """Internal implementation of chat with retry and error handling logic."""
        # Use the provided model_name or fall back to self.model_name
        effective_model = model_name if model_name is not None else self.model_name

        for i in range(self.max_retries):
            try:
                result = await self._chat(
                    messages=messages,
                    tools=tools,
                    enable_stream_print=enable_stream_print,
                    model_name=model_name,
                    **kwargs,
                )
                return callback_fn(result) if callback_fn else result

            except Exception as e:
                # Check if this is an inappropriate content error
                error_message = str(e.args[0]) if e.args else str(e)
                is_inappropriate_content = "inappropriate content" in error_message.lower()
                is_rate_limit_error = (
                    "request rate increased too quickly" in error_message.lower() or
                    "exceeded your current quota" in error_message.lower() or
                    "insufficient_quota" in error_message.lower()
                )

                if is_inappropriate_content:
                    logger.error(f"chat with model={effective_model} detected inappropriate content error")
                    logger.error("=" * 80)
                    logger.error("Full message content that triggered the error:")
                    logger.error("=" * 80)
                    for idx, msg in enumerate(messages):
                        logger.error(f"Message {idx + 1} [role={msg.role}]:")
                        logger.error(f"Content: {msg.content}")
                        if msg.reasoning_content:
                            logger.error(f"Reasoning: {msg.reasoning_content}")
                        if msg.tool_calls:
                            logger.error(f"Tool calls: {msg.tool_calls}")
                        logger.error("-" * 80)
                    logger.error("=" * 80)
                    # Return empty Message immediately without retrying
                    return Message(role=Role.ASSISTANT, content="")

                if is_rate_limit_error:
                    logger.warning(f"chat with model={effective_model} hit rate limit, sleeping for 60s before retry (attempt {i + 1}/{self.max_retries})")
                    await asyncio.sleep(60)
                    continue

                logger.exception(f"chat with model={effective_model} encounter error with e={e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e
                    return default_value

                await asyncio.sleep(1 + i)
        return default_value

    def chat_sync(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        enable_stream_print: bool = False,
        callback_fn: Callable[[Message], Any] | None = None,
        default_value: Any = None,
        model_name: str | None = None,
        **kwargs,
    ) -> Message | Any:
        """Perform a synchronous chat completion with integrated retries and error handling.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool calls
            enable_stream_print: Whether to print stream chunks
            callback_fn: Optional callback function to process the result
            default_value: Default value to return on error
            model_name: Optional model name to override self.model_name
            **kwargs: Additional parameters
        """
        # Use the provided model_name or fall back to self.model_name
        effective_model = model_name if model_name is not None else self.model_name

        for i in range(self.max_retries):
            try:
                result = self._chat_sync(
                    messages=messages,
                    tools=tools,
                    enable_stream_print=enable_stream_print,
                    model_name=model_name,
                    **kwargs,
                )
                return callback_fn(result) if callback_fn else result

            except Exception as e:
                # Check if this is an inappropriate content error
                error_message = str(e.args[0]) if e.args else str(e)
                is_inappropriate_content = "inappropriate content" in error_message.lower()
                is_rate_limit_error = (
                    "request rate increased too quickly" in error_message.lower() or
                    "exceeded your current quota" in error_message.lower() or
                    "insufficient_quota" in error_message.lower()
                )

                if is_inappropriate_content:
                    logger.error(f"chat sync with model={effective_model} detected inappropriate content error")
                    logger.error("=" * 80)
                    logger.error("Full message content that triggered the error:")
                    logger.error("=" * 80)
                    for idx, msg in enumerate(messages):
                        logger.error(f"Message {idx + 1} [role={msg.role}]:")
                        logger.error(f"Content: {msg.content}")
                        if msg.reasoning_content:
                            logger.error(f"Reasoning: {msg.reasoning_content}")
                        if msg.tool_calls:
                            logger.error(f"Tool calls: {msg.tool_calls}")
                        logger.error("-" * 80)
                    logger.error("=" * 80)
                    # Return empty Message immediately without retrying
                    return Message(role=Role.ASSISTANT, content="")

                if is_rate_limit_error:
                    logger.warning(f"chat sync with model={effective_model} hit rate limit, sleeping for 60s before retry (attempt {i + 1}/{self.max_retries})")
                    time.sleep(60)
                    continue

                logger.exception(f"chat sync with model={effective_model} encounter error with e={e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e
                    return default_value

                time.sleep(1 + i)
        return default_value

    async def close(self):
        """Release any asynchronous resources or connections held by the client."""

    def close_sync(self):
        """Release any synchronous resources or connections held by the client."""
