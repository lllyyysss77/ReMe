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

    def __init__(self, top_k: int = 20, **kwargs):
        """Initialize RetrieveRecentMemory.

        Args:
            top_k: Max memories to retrieve.
            **kwargs: Additional args for BaseMemoryTool.
        """
        kwargs["enable_multiple"] = False
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
            raise RuntimeError("memory_type and memory_target are required for retrieval.")

        # Retrieve recent memories
        memory_nodes: list[MemoryNode] = await self._retrieve_recent()

        # Deduplicate and format output
        memory_nodes = deduplicate_memories(memory_nodes)

        # Build set of historical memory_ids for fast lookup
        retrieved_memory_ids = {node.memory_id for node in self.retrieved_nodes if node.memory_id}

        # Filter out already retrieved memories by memory_id
        new_memory_nodes = [node for node in memory_nodes if node.memory_id not in retrieved_memory_ids]

        # Update retrieved_nodes in context with new memories
        self.retrieved_nodes.extend(new_memory_nodes)

        # Set output to new memories only (after deduplication)
        self.memory_nodes = new_memory_nodes

        if not new_memory_nodes:
            self.output = "No new memory_nodes found (duplicates removed)."
        else:
            self.output = "\n".join([m.format_memory() for m in new_memory_nodes])

        logger.info(f"Retrieved {len(memory_nodes)} memory_nodes, {len(new_memory_nodes)} new after deduplication")
