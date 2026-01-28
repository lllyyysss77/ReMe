from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import MemoryType


class ToolRetriever(BaseMemoryAgent):
    memory_type: MemoryType = MemoryType.TOOL
