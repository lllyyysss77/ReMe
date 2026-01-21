from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core_old.schema import MemoryNode


class ReadHistory(BaseMemoryTool):
    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)

    def _build_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "ref_memory_id": {
                    "type": "string",
                    "description": self.get_prompt("ref_memory_id"),
                },
            },
            "required": ["ref_memory_id"],
        }

    async def execute(self):
        ref_memory_id = self.context.get("ref_memory_id", "")

        if not ref_memory_id:
            self.output = "No valid reference memory ID provided."
            logger.warning(self.output)
            return

        nodes = await self.vector_store.get(vector_ids=[ref_memory_id])

        if not nodes:
            self.output = f"No history memory found with ID: {ref_memory_id}"
            logger.warning(self.output)
            return

        memory = MemoryNode.from_vector_node(nodes[0])
        self.output = memory.content
        logger.info(f"Successfully read history memory: {ref_memory_id}")
