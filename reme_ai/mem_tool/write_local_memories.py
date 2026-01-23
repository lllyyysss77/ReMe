from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.context import C
from ...core.schema.memory_node import MemoryNode


@C.register_op()
class WriteLocalMemories(BaseMemoryTool):

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = True
        super().__init__(**kwargs)

    def _build_multiple_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "memory_nodes": {
                    "type": "array",
                    "description": self.get_prompt("memory_nodes"),
                    "items": {
                        "type": "object",
                        "description": "Memory node object",
                    },
                },
            },
            "required": ["memory_nodes"],
        }

    async def execute(self):
        memory_nodes = self.context.get("memory_nodes", [])

        if not memory_nodes:
            self.output = "No memory nodes provided."
            return

        memory_nodes = [MemoryNode(**node) if isinstance(node, dict) else node for node in memory_nodes]

        grouped = {}
        for node in memory_nodes:
            key = (node.memory_type.value, node.memory_target)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(node)

        written_keys = []

        for (memory_type, memory_target), nodes in grouped.items():
            cache_key = f"{memory_type}_{memory_target}"
            nodes_data = [node.model_dump() for node in nodes]

            self.meta_memory.save(cache_key, nodes_data)
            written_keys.append(f"{memory_type}_{memory_target}")
            logger.info(f"Saved {len(nodes)} nodes to cache key: {cache_key}")

        self.output = f"Successfully written local memories: {', '.join(written_keys)}"
