"""Base memory agent for handling memory operations with tool-based reasoning."""

from abc import ABCMeta

from ...core.enumeration import MemoryType
from ...core.op import BaseReact
from ...core.schema import MemoryNode


class BaseMemoryAgent(BaseReact, metaclass=ABCMeta):
    memory_type: MemoryType | None = None

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
