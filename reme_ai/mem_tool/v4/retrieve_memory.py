import json

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.schema import MemoryNode
from ...core.utils import deduplicate_memories


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
                            "query": {
                                "type": "string",
                                "description": "query",
                            },
                            "time_range": {
                                "type": "string",
                                "description": "time_range(optional), e.g. [20200101, 20200101]",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            "required": ["query_items"],
        }

    async def execute(self):
        query_items: list[dict] = self.context.get("query_items", [])
        memory_nodes: list[MemoryNode] = []
        for query_item in query_items:
            query = query_item.get("query")
            time_range = query_item.get("time_range", "")

            filter_dict: dict = {
                "memory_type": self.memory_type.value,
                "memory_target": self.memory_target,
            }

            if time_range:
                # Handle different time_range formats
                if isinstance(time_range, str):
                    try:
                        time_range = json.loads(time_range)
                    except json.JSONDecodeError:
                        # If it's a plain string like "20250907", treat it as a single date
                        time_range = time_range
                
                # Convert to list format [start, end]
                if isinstance(time_range, (list, tuple)):
                    if len(time_range) == 1:
                        # Single element list, use it for both start and end
                        filter_dict["time_int"] = [int(time_range[0]), int(time_range[0])]
                    else:
                        # Two element list/tuple
                        filter_dict["time_int"] = [int(time_range[0]), int(time_range[1])]
                else:
                    # Single value (int or string), use it for both start and end
                    filter_dict["time_int"] = [int(time_range), int(time_range)]
            logger.info(f"memory_type={self.memory_type} memory_target={self.memory_target} query={query} "
                        f"filter_dict={filter_dict}")

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
            output = []
            for node in new_memory_nodes:
                line = ""
                if "conversation_time" in node.metadata and node.metadata["conversation_time"]:
                    line += f"conversation_time={node.metadata['conversation_time']} "
                line += node.content.strip() + " "
                if node.ref_memory_id:
                    line += f"history_id={node.ref_memory_id} "
                output.append(line.strip())
            self.output = "### Extracted Memories\n" + "\n".join(output)

        logger.info(f"Retrieved {len(memory_nodes)} memory_nodes, {len(new_memory_nodes)} new after deduplication")
