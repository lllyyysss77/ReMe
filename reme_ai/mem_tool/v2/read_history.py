"""Read history memory operation."""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.context import C
from ...core.schema import MemoryNode


@C.register_op()
class ReadHistory(BaseMemoryTool):
    """Read original history dialogue by reference memory ID.

    Only supports single memory read (enable_multiple=False).
    """

    def __init__(self, **kwargs):
        """Initialize ReadHistory.

        Args:
            **kwargs: Additional args for BaseMemoryTool.
        """
        # Force disable multiple mode
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

        # Query history dialogue by ref_memory_id
        nodes = await self.vector_store.get(vector_ids=[ref_memory_id])

        if not nodes:
            self.output = f"No history memory found with ID: {ref_memory_id}"
            logger.warning(self.output)
            return

        memory = MemoryNode.from_vector_node(nodes[0])
        self.output = memory.content
        logger.info(f"Successfully read history memory: {ref_memory_id}")
