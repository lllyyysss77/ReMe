"""
ReMe Light Application Module

This module provides the ReMeLight class, a specialized application built on top of
ReMe's core Application framework. It integrates memory management capabilities
including memory compaction, summarization, tool result management, and semantic
memory search functionality.

Key Features:
    - Memory compaction and summarization for long conversations
    - Tool result compaction with file-based storage for large outputs
    - Semantic memory search using vector and full-text search
    - Configurable embedding models and vector store backends
    - Async task management for background summarization
"""

import asyncio
import logging
from pathlib import Path

from agentscope.formatter import FormatterBase
from agentscope.message import Msg, TextBlock
from agentscope.model import ChatModelBase
from agentscope.token import HuggingFaceTokenCounter
from agentscope.tool import Toolkit, ToolResponse

from .config import ReMeConfigParser
from .core import Application
from .core.utils import get_hf_token_counter
from .memory.file_based import Compactor, Summarizer, ToolResultCompactor, ReMeInMemoryMemory, ReMeOpenAIChatFormatter, \
    FileIO
from .memory.file_based.utils import get_token_counter
from .memory.tools import MemorySearch

logger = logging.getLogger(__name__)


class ReMeLight(Application):
    """ReMe Light Application Class"""

    def __init__(
            self,
            working_dir: str = ".reme",
            llm_api_key: str | None = None,
            llm_base_url: str | None = None,
            embedding_api_key: str | None = None,
            embedding_base_url: str | None = None,
            default_as_llm_config: dict | None = None,
            default_embedding_model_config: dict | None = None,
            default_file_store_config: dict | None = None,
            vector_weight: float = 0.7,
            candidate_multiplier: float = 3.0,
            tool_result_threshold: int = 1000,
            retention_days: int = 7,
    ):
        # Initialize working directory structure
        self.working_path = Path(working_dir).absolute()
        self.working_path.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.working_path / "memory"
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.tool_result_path = self.working_path / "tool_result"
        self.tool_result_path.mkdir(parents=True, exist_ok=True)

        self.vector_weight: float = vector_weight
        self.candidate_multiplier: float = candidate_multiplier
        self.tool_result_threshold: int = tool_result_threshold
        self.retention_days: int = retention_days

        # Initialize the parent Application class with comprehensive configuration
        super().__init__(
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            working_dir=str(self.working_path),
            config_path="light",
            enable_logo=False,
            log_to_console=False,
            parser=ReMeConfigParser,
            default_as_llm_config=default_as_llm_config,
            default_embedding_model_config=default_embedding_model_config,
            default_file_store_config=default_file_store_config,
            default_file_watcher_config={
                "watch_paths": [
                    str(self.working_path / "MEMORY.md"),
                    str(self.working_path / "memory.md"),
                    str(self.memory_path),
                ],
            },
        )

        # Initialize list to track background summarization tasks
        self.summary_tasks: list[asyncio.Task] = []

    @staticmethod
    def calculate_memory_compact_threshold(max_input_length: float, compact_ratio: float) -> int:
        return int(max_input_length * compact_ratio * 0.9)

    def _cleanup_tool_results(self) -> int:
        """
        Clean up expired tool result files from the tool result directory.

        This method removes tool result files that have exceeded the retention
        period specified during initialization. It helps manage disk space by
        automatically removing old, unused tool outputs.

        Returns:
            int: The number of files that were successfully deleted
        """
        try:
            # Create a compactor instance with current configuration
            compactor = ToolResultCompactor(
                tool_result_dir=self.tool_result_path,
                tool_result_threshold=self.tool_result_threshold,
                retention_days=self.retention_days,
            )
            # Execute cleanup and return count of deleted files
            return compactor.cleanup_expired_files()
        except Exception as e:
            # Log exception details but return 0 to indicate failure gracefully
            logger.exception(f"Error cleaning up tool results: {e}")
            return 0

    async def start(self):
        """Start the application lifecycle."""
        result = await super().start()
        self._cleanup_tool_results()
        return result

    async def close(self) -> bool:
        """Close the application and perform cleanup."""
        self._cleanup_tool_results()
        return await super().close()

    async def compact_tool_result(self, messages: list[Msg]) -> list[Msg]:
        """Compact tool results by truncating large outputs and saving full content to files."""
        try:
            # Create compactor with instance configuration
            compactor = ToolResultCompactor(
                tool_result_dir=self.tool_result_path,
                tool_result_threshold=self.tool_result_threshold,
                retention_days=self.retention_days,
            )

            # Execute compaction and get processed messages
            result = await compactor.call(messages=messages, service_context=self.service_context)

            # Clean up any expired tool result files during compaction
            compactor.cleanup_expired_files()

            return result

        except Exception as e:
            # Log the error and return original messages to maintain functionality
            logger.exception(f"Error compacting tool results: {e}")
            return messages

    async def compact_memory(
            self,
            messages: list[Msg],
            as_llm: str | ChatModelBase = "default",
            as_llm_formatter: str | FormatterBase = "default",
            token_counter: HuggingFaceTokenCounter | None = None,
            language: str = "zh",
            max_input_length: float = 128 * 1024,
            compact_ratio: float = 0.7,
            previous_summary: str = "",
    ) -> str:
        """Compact a list of messages into a condensed summary."""
        try:
            if token_counter is None:
                token_counter = get_hf_token_counter()

            compactor = Compactor(
                memory_compact_threshold=self.calculate_memory_compact_threshold(max_input_length, compact_ratio),
                as_llm=as_llm,
                as_llm_formatter=as_llm_formatter,
                token_counter=token_counter,
                language=language if language == "zh" else "",
            )

            return await compactor.call(
                messages=messages,
                previous_summary=previous_summary,
                service_context=self.service_context,
            )

        except Exception as e:
            # Log error and return empty string to indicate failure
            logger.exception(f"Error compacting memory: {e}")
            return ""

    async def summary_memory(self, messages: list[Msg]) -> str:
        """Generate a comprehensive summary of the given messages."""
        try:
            # Create toolkit if not provided
            if self.toolkit is not None:
                toolkit = self.toolkit
            else:
                toolkit = Toolkit()
                file_io = FileIO(working_dir=str(self.working_path))
                toolkit.register_tool_function(file_io.read)
                toolkit.register_tool_function(file_io.write)
                toolkit.register_tool_function(file_io.edit)

            # Initialize summarizer with working directories and configuration
            summarizer = Summarizer(
                working_dir=str(self.working_path),
                memory_dir=str(self.memory_path),
                memory_compact_threshold=self.memory_compact_threshold,
                chat_model=self.chat_model,
                formatter=self.formatter,
                token_counter=self.token_counter,
                toolkit=toolkit,
                language=self.language,
            )

            # Execute summarization on the provided messages
            return await summarizer.call(messages=messages, service_context=self.service_context)

        except Exception as e:
            # Log error and return empty string to indicate failure
            logger.exception(f"Error summarizing memory: {e}")
            return ""

    async def await_summary_tasks(self) -> str:
        """
        Wait for all background summary tasks to complete and collect results.

        This method iterates through all pending summary tasks, waits for their
        completion, and collects their results or error information. It's used
        to synchronize with background summarization operations before shutdown
        or when results are needed.

        Returns:
            str: A concatenated string containing the status and results of
                all summary tasks, with each task on a new line

        Note:
            - Completed tasks are processed immediately without waiting
            - Incomplete tasks are awaited with a timeout
            - Cancelled tasks and exceptions are logged and included in results
            - The task list is cleared after processing all tasks
        """
        result = ""
        for task in self.summary_tasks:
            if task.done():
                # Task has already completed, check its status
                if task.cancelled():
                    logger.warning("Summary task was cancelled.")
                    result += "Summary task was cancelled.\n"
                else:
                    # Check if the task raised an exception
                    exc = task.exception()
                    if exc is not None:
                        logger.error(f"Summary task failed: {exc}")
                        result += f"Summary task failed: {exc}\n"
                    else:
                        # Task completed successfully, collect result
                        task_result = task.result()
                        logger.info(f"Summary task completed: {task_result}")
                        result += f"Summary task completed: {task_result}\n"

            else:
                # Task is still running, wait for it to complete
                try:
                    task_result = await task
                    logger.info(f"Summary task completed: {task_result}")
                    result += f"Summary task completed: {task_result}\n"

                except asyncio.CancelledError:
                    logger.warning("Summary task was cancelled while waiting.")
                    result += "Summary task was cancelled.\n"
                except Exception as e:
                    logger.exception(f"Summary task failed: {e}")
                    result += f"Summary task failed: {e}\n"

        # Clear the task list after processing all tasks
        self.summary_tasks.clear()
        return result

    def add_async_summary_task(self, messages: list[Msg]):
        """
        Add an asynchronous summary task for the given messages.

        This method creates a background task to summarize the provided messages
        without blocking the main execution flow. Before adding a new task, it
        cleans up any completed tasks from the task list to prevent memory leaks.

        Args:
            messages (list[Msg]): The list of messages to be summarized in the
                background task

        Note:
            - Completed tasks are removed from the tracking list before adding
            - Task status (success, failure, cancellation) is logged for monitoring
            - The new task is created using asyncio.create_task for true async execution
            - Failed or cancelled tasks are logged but do not prevent new tasks
        """
        # Clean up completed summary tasks before adding a new one
        remaining_tasks = []
        for task in self.summary_tasks:
            if task.done():
                # Process completed task status
                if task.cancelled():
                    logger.warning("Summary task was cancelled.")
                    continue
                exc = task.exception()
                if exc is not None:
                    logger.error(f"Summary task failed: {exc}")
                else:
                    # Log successful completion with result summary
                    result = task.result()
                    logger.info(f"Summary task completed: {result}")
            else:
                # Keep incomplete tasks in the tracking list
                remaining_tasks.append(task)
        self.summary_tasks = remaining_tasks

        # Create and track the new background summarization task
        task = asyncio.create_task(self.summary_memory(messages=messages))
        self.summary_tasks.append(task)

    async def memory_search(self, query: str, max_results: int = 5, min_score: float = 0.1) -> ToolResponse:
        """
        Perform semantic memory search using vector and full-text search.

        This method searches the memory store for content relevant to the given query
        using a hybrid approach combining vector similarity search and full-text search.
        Results are ranked by relevance and filtered by the minimum score threshold.

        Args:
            query (str): The search query string. Must not be empty.
            max_results (int): Maximum number of results to return (1-100, default: 5)
            min_score (float): Minimum relevance score threshold (0.001-0.999, default: 0.1)

        Returns:
            ToolResponse: A ToolResponse containing the search results as text,
                or an error message if the query is empty

        Note:
            - Vector search weight is controlled by self.vector_weight
            - Candidate retrieval uses self.candidate_multiplier for broader search
            - Parameters are validated and clamped to valid ranges
            - Requires vector search to be enabled via embedding configuration
        """
        # Validate query parameter
        if not query:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="Error: No query provided.",
                    ),
                ],
            )

        # Validate and clamp max_results to valid range [1, 100]
        if isinstance(max_results, int):
            max_results = min(max(max_results, 1), 100)
        else:
            max_results = 5

        # Validate and clamp min_score to valid range [0.001, 0.999]
        if isinstance(min_score, (int, float)):
            min_score = min(max(min_score, 0.001), 0.999)
        else:
            min_score = 0.1

        # Initialize memory search tool with configured weights
        search_tool = MemorySearch(
            vector_weight=self.vector_weight,
            candidate_multiplier=self.candidate_multiplier,
        )

        # Execute the search with validated parameters
        search_result = await search_tool.call(
            query=query,
            max_results=max_results,
            min_score=min_score,
            service_context=self.service_context,
        )

        # Return results wrapped in ToolResponse format
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=search_result,
                ),
            ],
        )

    @staticmethod
    def get_in_memory_memory(token_counter: HuggingFaceTokenCounter | None = None):
        """Create and return an in-memory memory instance."""
        if token_counter is None:
            token_counter = get_hf_token_counter()

        return ReMeInMemoryMemory(token_counter=token_counter)
