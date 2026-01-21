import json

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core_old.schema import MemoryNode
from ...core_old.utils import deduplicate_memories


class RetrieveMemory(BaseMemoryTool):

    def __init__(self, top_k: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.top_k: int = top_k

    def _build_tool_description(self) -> str:
        return "Retrieve memories using vector similarity search."

    def _build_multiple_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query_items": {
                    "type": "array",
                    "description": "query_items",
                    "items": {
                        "type": "object",
                        "properties": {
                            "memory_type": {
                                "type": "string",
                                "description": "memory_type",
                            },
                            "memory_target": {
                                "type": "string",
                                "description": "memory_target",
                            },
                            "query": {
                                "type": "string",
                                "description": "query",
                            },
                            "time_range": {
                                "type": "string",
                                "description": "time_range(optional), e.g. [20200101, 20200101]",
                            },
                        },
                        "required": ["memory_type", "memory_target", "query"],
                    },
                },
            },
            "required": ["query_items"],
        }

    async def execute(self):
        query_items: list[dict] = self.context.get("query_items", [])
        memory_nodes: list[MemoryNode] = []
        for query_item in query_items:
            memory_type = query_item.get("memory_type")
            memory_target = query_item.get("memory_target")
            query = query_item.get("query")
            time_range = query_item.get("time_range", "")

            filter_dict = {
                "memory_type": memory_type,
                "memory_target": memory_target,
            }

            if time_range:
                time_range = json.loads(time_range)
                filter_dict["time_range"] = [int(time_range[0]), int(time_range[1])]

            nodes = await self.vector_store.search(query=query, limit=self.top_k, filters=filter_dict)
            memory_nodes.extend([MemoryNode.from_vector_node(n) for n in nodes])
        memory_nodes = deduplicate_memories(memory_nodes)

        retrieved_memory_ids = {node.memory_id for node in self.retrieved_nodes if node.memory_id}
        new_memory_nodes = [node for node in memory_nodes if node.memory_id not in retrieved_memory_ids]
        self.retrieved_nodes.extend(new_memory_nodes)
        self.memory_nodes = new_memory_nodes

        if not new_memory_nodes:
            self.output = "No new memory_nodes found matching the query (duplicates removed)."
        else:
            self.output = "\n".join([f"{m.metadata['conversation_time']} {m.content}" for m in new_memory_nodes])
        logger.info(f"Retrieved {len(memory_nodes)} memory_nodes, {len(new_memory_nodes)} new after deduplication")
