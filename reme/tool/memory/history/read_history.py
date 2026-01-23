"""Read history memory tool"""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.schema import MemoryNode, ToolCall


class ReadHistory(BaseMemoryTool):
    """Read history memory tool"""

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)

    def _build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema"""
        return ToolCall(
            **{
                "description": "Read original history dialogue.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "history_id": {
                            "type": "string",
                            "description": "history_id",
                        },
                    },
                    "required": ["history_id"],
                },
            },
        )

    async def execute(self):
        history_id = self.context.history_id
        nodes = await self.vector_store.get(vector_ids=[history_id])

        if not nodes:
            output = f"No history: {history_id}"
            logger.warning(output)
            return output

        memory = MemoryNode.from_vector_node(nodes[0])
        output = f"Historical Dialogue[{history_id}]\n{memory.content}"
        logger.info(f"Successfully read history memory: {history_id}")
        return output
