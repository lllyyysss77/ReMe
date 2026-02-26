"""Procedural memory summarizer agent implementation."""

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import MemoryType


class ProceduralSummarizer(BaseMemoryAgent):
    """Agent responsible for summarizing procedural memories."""

    memory_type: MemoryType = MemoryType.PROCEDURAL
