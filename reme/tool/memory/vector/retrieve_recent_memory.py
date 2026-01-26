"""Retrieve most recent memories from vector store"""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall, MemoryNode, VectorNode
from ....core.utils import deduplicate_memories


class RetrieveRecentMemory(BaseMemoryTool):
    """Tool to retrieve most recent memories sorted by conversation time"""

    def __init__(self, top_k: int = 20, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)
        self.top_k: int = top_k

    def _build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema"""
        return ToolCall(
            **{
                "description": "retrieve the most recent memories sorted by conversation time (newest first).",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        )

    async def _retrieve_recent(self) -> list[MemoryNode]:
        """Retrieve recent memories sorted by conversation_time descending"""
        filter_dict = {
            "memory_type": self.memory_type.value,
            "memory_target": self.memory_target,
        }

        nodes: list[VectorNode] = await self.vector_store.list(
            filters=filter_dict,
            limit=self.top_k,
            sort_key="conversation_time",
            reverse=True,
        )

        return [MemoryNode.from_vector_node(n) for n in nodes]

    async def execute(self):
        memory_nodes: list[MemoryNode] = await self._retrieve_recent()
        memory_nodes = deduplicate_memories(memory_nodes)

        retrieved_memory_ids = {node.memory_id for node in self.retrieved_nodes if node.memory_id}
        new_memory_nodes = [node for node in memory_nodes if node.memory_id not in retrieved_memory_ids]
        self.retrieved_nodes.extend(new_memory_nodes)
        self.memory_nodes.extend(new_memory_nodes)

        if not new_memory_nodes:
            output = "No new memory_nodes found (duplicates removed)."
        else:
            output = "\n".join([m.format_memory() for m in new_memory_nodes])

        logger.info(f"Retrieved {len(memory_nodes)} memory_nodes, {len(new_memory_nodes)} new after deduplication")
        return output
