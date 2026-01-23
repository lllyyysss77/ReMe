"""Retrieve memory from vector store"""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.enumeration import MemoryType
from ....core.schema import ToolCall, MemoryNode, VectorNode
from ....core.utils import deduplicate_memories


class VectorRetrieveMemory(BaseMemoryTool):
    """Tool to retrieve memories from vector store using similarity search"""

    def __init__(self, top_k: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.top_k: int = top_k

    @staticmethod
    def _build_query_parameters() -> dict:
        """Build query parameters schema for retrieval"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "query text for vector similarity search.",
                },
                "time_range": {
                    "type": "string",
                    "description": "optional time range filter. "
                    "Format: single date '20200101' or range '20200101,20200102'",
                },
            },
            "required": ["query"],
        }

    def _build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema"""
        return ToolCall(
            **{
                "description": "retrieve memories using vector similarity search.",
                "parameters": self._build_query_parameters(),
            },
        )

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the multiple tool call schema"""
        return ToolCall(
            **{
                "description": "retrieve memories using multiple queries with vector similarity search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_items": {
                            "type": "array",
                            "description": "list of query items for vector similarity search.",
                            "items": self._build_query_parameters(),
                        },
                    },
                    "required": ["query_items"],
                },
            },
        )

    async def _retrieve_by_query(
        self,
        memory_type: str,
        memory_target: str,
        query: str,
        time_range: str | None = None,
    ) -> list[MemoryNode]:
        """Retrieve memories by query with filters"""
        filter_dict: dict = {
            "memory_type": memory_type,
            "memory_target": memory_target,
        }

        if time_range:
            time_range = time_range.strip()
            if "," in time_range:
                parts = time_range.split(",")
                start_time = int(parts[0].strip())
                end_time = int(parts[1].strip())
                filter_dict["time_int"] = [start_time, end_time]
            else:
                single_time = int(time_range)
                filter_dict["time_int"] = [single_time, single_time]

        nodes: list[VectorNode] = await self.vector_store.search(query=query, limit=self.top_k, filters=filter_dict)
        return [MemoryNode.from_vector_node(n) for n in nodes]

    async def execute(self):
        memory_type: str = self.memory_type.value
        memory_target: str = self.memory_target

        if self.enable_multiple:
            query_items: list[dict] = self.context.get("query_items", [])
        else:
            query_items: list[dict] = [
                {
                    "query": self.context.get("query", ""),
                    "time_range": self.context.get("time_range", ""),
                },
            ]

        query_items = [item for item in query_items if item.get("query")]
        memory_nodes: list[MemoryNode] = []
        for item in query_items:
            retrieved = await self._retrieve_by_query(
                memory_type=memory_type,
                memory_target=memory_target,
                query=item["query"],
                time_range=item.get("time_range", ""),
            )
            memory_nodes.extend(retrieved)

        memory_nodes = deduplicate_memories(memory_nodes)
        retrieved_memory_ids = {node.memory_id for node in self.retrieved_nodes if node.memory_id}
        new_memory_nodes = [node for node in memory_nodes if node.memory_id not in retrieved_memory_ids]
        self.retrieved_nodes.extend(new_memory_nodes)
        self.memory_nodes = new_memory_nodes

        if not new_memory_nodes:
            output = "No new memory_nodes found matching the query (duplicates removed)."
        else:
            output = "\n".join([m.format_memory() for m in new_memory_nodes])

        logger.info(f"Retrieved {len(memory_nodes)} memory_nodes, {len(new_memory_nodes)} new after deduplication")
        return output
