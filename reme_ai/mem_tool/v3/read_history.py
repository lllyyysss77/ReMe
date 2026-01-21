from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core_old.schema import MemoryNode


class ReadHistory(BaseMemoryTool):
    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)

    def _build_tool_description(self) -> str:
        return "Read original history dialogue."

    def _build_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "history_id": {
                    "type": "string",
                    "description": "history_id",
                },
            },
            "required": ["history_id"],
        }

    async def execute(self):
        history_id = self.context.get("history_id", "")
        nodes = await self.vector_store.get(vector_ids=[history_id])

        if not nodes:
            self.output = f"No history: {history_id}"
            logger.warning(self.output)
            return

        memory = MemoryNode.from_vector_node(nodes[0])
        self.output = memory.content
        logger.info(f"Successfully read history memory: {history_id}")
