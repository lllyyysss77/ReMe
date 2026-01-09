"""Read history memory operation."""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.context import C
from ...core.schema import MemoryNode


@C.register_op()
class ReadHistoryMemory(BaseMemoryTool):
    """Read history memories by IDs."""

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

    def _build_multiple_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "ref_memory_ids": {
                    "type": "array",
                    "description": self.get_prompt("ref_memory_ids"),
                    "items": {"type": "string"},
                },
            },
            "required": ["ref_memory_ids"],
        }

    async def execute(self):
        if self.enable_multiple:
            ref_memory_ids: list[str] = self.context.get("ref_memory_ids", [])
        else:
            ref_memory_id = self.context.get("ref_memory_id", "")
            ref_memory_ids: list[str] = [ref_memory_id] if ref_memory_id else []

        ref_memory_ids = [mid for mid in ref_memory_ids if mid]

        if not ref_memory_ids:
            self.output = "No valid reference memory IDs provided for reading."
            logger.warning(self.output)
            return

        # Query original history dialogues by ref_memory_id
        nodes = await self.vector_store.get(vector_ids=ref_memory_ids)

        if not nodes:
            self.output = "No history memories found with the provided reference IDs."
            logger.warning(self.output)
            return

        memories: list[MemoryNode] = [MemoryNode.from_vector_node(n) for n in nodes]
        self.output = "---\n".join([m.content for m in memories])
        logger.info(f"Successfully read {len(memories)} history memories by reference IDs.")
