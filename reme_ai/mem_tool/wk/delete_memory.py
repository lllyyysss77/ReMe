from loguru import logger

from ..base_memory_tool import BaseMemoryTool


class DeleteMemory(BaseMemoryTool):

    def __init__(self, **kwargs):
        kwargs['enable_multiple'] = True
        super().__init__(**kwargs)

    def _build_multiple_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "memory_ids": {
                    "type": "array",
                    "description": self.get_prompt("memory_ids"),
                    "items": {"type": "string"},
                },
            },
            "required": ["memory_ids"],
        }

    async def execute(self):
        memory_ids = [m for m in self.context.get("memory_ids", []) if m]

        if not memory_ids:
            self.output = "No valid memory IDs provided for deletion."
            return

        await self.vector_store.delete(vector_ids=memory_ids)
        self.memory_nodes = memory_ids
        self.output = f"Successfully deleted {len(memory_ids)} memories from vector_store."
        logger.info(self.output)
