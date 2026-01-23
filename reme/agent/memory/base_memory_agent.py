"""Base memory agent for handling memory operations with tool-based reasoning."""

from abc import ABCMeta

from ...core.enumeration import MemoryType
from ...core.op import BaseReact


class BaseMemoryAgent(BaseReact, metaclass=ABCMeta):
    memory_type: MemoryType | None = None
