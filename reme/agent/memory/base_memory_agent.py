"""Base memory agent for handling memory operations with tool-based reasoning."""

from abc import ABCMeta
from typing import Literal

from loguru import logger

from ...core.enumeration import MemoryType
from ...core.op import BaseReact
from ...core.schema import MemoryNode


class BaseMemoryAgent(BaseReact, metaclass=ABCMeta):
    """Base class for memory agents that handle memory operations with tool-based reasoning."""

    memory_type: MemoryType | None = None

    @staticmethod
    async def read_meta_memories(meta_memories: list[dict]) -> str:
        """Read and format meta memory information from the provided metadata list."""
        from ...tool.memory import ReadMetaMemory

        meta_memory_info = ReadMetaMemory().format_memory_metadata(meta_memories)
        logger.info(f"meta_memory_info={meta_memory_info}")
        return meta_memory_info

    async def read_user_profile(self, show_id: Literal["profile", "history"] = "profile") -> str:
        """Read current user profile."""
        from ...tool.memory import ReadUserProfile

        read_tool = ReadUserProfile(show_id=show_id)
        await read_tool.call(memory_target=self.memory_target)
        return str(read_tool.response.answer)

    @staticmethod
    async def read_history_node() -> MemoryNode:
        """Read and return the current history node from the context."""
        from ...tool.memory import AddHistory

        add_history_tool = AddHistory()
        await add_history_tool.call()
        return add_history_tool.context.history_node

    @property
    def memory_target(self) -> str:
        """memory_target"""
        return self.context.get("memory_target", "")

    @property
    def query(self) -> str:
        """query"""
        return self.context.get("query", "")

    @property
    def messages(self) -> list:
        """messages"""
        return self.context.get("messages", [])

    @property
    def description(self) -> str:
        """description"""
        return self.context.get("description", "")

    @property
    def history_node(self) -> MemoryNode:
        """Returns the history node."""
        return self.context.history_node

    @property
    def author(self) -> str:
        """Returns the LLM model name as the author identifier."""
        return self.llm.model_name
