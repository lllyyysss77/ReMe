from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import MemoryType


class ProceduralSummarizer(BaseMemoryAgent):
    memory_type: MemoryType = MemoryType.PROCEDURAL
