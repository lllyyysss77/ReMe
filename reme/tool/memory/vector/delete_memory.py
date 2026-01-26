"""Delete memory from vector store"""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class DeleteMemory(BaseMemoryTool):
    """Tool to delete memories from vector store"""

    def _build_tool_call(self) -> ToolCall:
        """Build and return the single tool call schema"""
        return ToolCall(
            **{
                "description": "delete a memory from vector store using its unique ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "unique identifier (memory_id) of the memory to delete.",
                        },
                    },
                    "required": ["memory_id"],
                },
            },
        )

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
        memory_ids: list[str] = []

        # Handle multiple memories (array format)
        ids_from_array = self.context.get("memory_ids", [])
        if ids_from_array:
            memory_ids = [m for m in ids_from_array if m]
        else:
            memory_id = self.context.get("memory_id", "")
            if memory_id:
                memory_ids = [memory_id]

        if not memory_ids:
            output = "No valid memory IDs provided for deletion."
            logger.info(output)
            return output

        await self.vector_store.delete(vector_ids=memory_ids)
        self.memory_nodes = memory_ids

        output = f"Successfully deleted {len(memory_ids)} memories from vector_store."
        logger.info(output)
        return output
