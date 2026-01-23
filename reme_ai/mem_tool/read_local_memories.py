from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.context import C
from ...core.schema.memory_node import MemoryNode


@C.register_op()
class ReadLocalMemories(BaseMemoryTool):

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)

    def _build_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "memory_type": {
                    "type": "string",
                    "description": self.get_prompt("memory_type"),
                },
                "memory_target": {
                    "type": "string",
                    "description": self.get_prompt("memory_target"),
                },
            },
            "required": ["memory_type", "memory_target"],
        }

    async def execute(self):
        memory_type = self.context.get("memory_type", "")
        memory_target = self.context.get("memory_target", "")

        if not memory_type or not memory_target:
            self.output = "memory_type and memory_target are required."
            return

        cache_key = f"{memory_type}_{memory_target}"
        cached_data = self.meta_memory.load(cache_key, auto_clean=False)

        if not cached_data:
            self.output = f"Local memory not found: {memory_type}_{memory_target}"
            logger.info(self.output)
            return

        memory_nodes = [MemoryNode(**node_data) for node_data in cached_data]

        if not memory_nodes:
            self.output = f"No valid memory nodes found in {memory_type}_{memory_target}"
            return

        self.output = memory_nodes
        logger.info(f"Read {len(memory_nodes)} nodes from cache key: {cache_key}")
