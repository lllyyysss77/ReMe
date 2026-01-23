"""Delete memory from vector store"""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class DeleteMemory(BaseMemoryTool):
    """Tool to delete memories from vector store"""

    def __init__(self, **kwargs):
        kwargs['enable_multiple'] = True
        super().__init__(**kwargs)

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the multiple tool call schema"""
        return ToolCall(
            **{
                "description": "delete multiple memories from vector store using their unique IDs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_ids": {
                            "type": "array",
                            "description": "list of unique identifiers (memory_ids) of memories to delete.",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["memory_ids"],
                },
            },
        )

    async def execute(self):
        memory_ids = [m for m in self.context.get("memory_ids", []) if m]

        if not memory_ids:
            self.output = "No valid memory IDs provided for deletion."
            return

        await self.vector_store.delete(vector_ids=memory_ids)
        self.memory_nodes = memory_ids
        self.output = f"Successfully deleted {len(memory_ids)} memories from vector_store."
        logger.info(self.output)
