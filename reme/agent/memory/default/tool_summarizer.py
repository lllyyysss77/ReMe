from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import MemoryType


class ToolSummarizer(BaseMemoryAgent):
    memory_type: MemoryType = MemoryType.TOOL
