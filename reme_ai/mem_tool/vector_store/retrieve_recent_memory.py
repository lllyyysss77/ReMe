"""Time-based memory retrieval to get most recent memories."""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.context import C
from ...core.schema import MemoryNode, VectorNode
from ...core.utils import deduplicate_memories


@C.register_op()
class RetrieveRecentMemory(BaseMemoryTool):
    """Retrieve most recent memories based on time_modified.

    Retrieves memories sorted by modification time in descending order.
    Uses memory_type and memory_target from context (self.memory_type, self.memory_target).
    """

    def __init__(
        self,
        top_k: int = 20,
        **kwargs,
    ):
        """Initialize RetrieveRecentMemory.

        Args:
            top_k: Max memories to retrieve.
            **kwargs: Additional args for BaseMemoryTool.
        """
        super().__init__(**kwargs)
        self.top_k: int = top_k

    async def _retrieve_recent(self) -> list[MemoryNode]:
        """Retrieve recent memories sorted by time_modified.

        Returns:
            List of recent memories sorted by modification time (newest first).
        """
        filter_dict = {
            "memory_type": [self.memory_type.value],
            "memory_target": [self.memory_target],
        }

        # Use list() with sort_key="time_modified", reverse=True (descending), and limit
        nodes: list[VectorNode] = await self.vector_store.list(
            filters=filter_dict,
            limit=self.top_k,
            sort_key="time_modified",
            reverse=True,  # Most recent first (descending order)
        )

        memory_nodes: list[MemoryNode] = [MemoryNode.from_vector_node(n) for n in nodes]

        return memory_nodes

    async def execute(self):
        """Execute recent memory retrieval.

        Uses memory_type and memory_target from context (self.memory_type, self.memory_target).
        Outputs formatted results or error message.
        """
        if not self.memory_type or not self.memory_target:
            self.output = "memory_type and memory_target are required for retrieval."
            return

        # Retrieve recent memories
        memory_nodes: list[MemoryNode] = await self._retrieve_recent()

        # Deduplicate and format output
        memory_nodes = deduplicate_memories(memory_nodes)
        self.memory_nodes = memory_nodes

        if not memory_nodes:
            self.output = "No memory_nodes found."
        else:
            self.output = "\n".join([m.format_memory() for m in memory_nodes])

        logger.info(f"Retrieved {len(memory_nodes)} recent memory_nodes")
