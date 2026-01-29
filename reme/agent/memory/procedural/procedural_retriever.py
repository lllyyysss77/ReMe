"""Procedural memory retriever agent implementation."""

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import MemoryType


class ProceduralRetriever(BaseMemoryAgent):
    """Agent responsible for retrieving procedural memories."""

    memory_type: MemoryType = MemoryType.PROCEDURAL
