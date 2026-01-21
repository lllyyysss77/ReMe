from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core_old.schema import MemoryNode


class AddSummaryMemory(BaseMemoryTool):

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)

    def _build_tool_description(self) -> str:
        return "Add a summary memory to the vector store for future retrieval."

    @staticmethod
    def _build_item_schema() -> tuple[dict, list[str]]:
        properties = {
            "conversation_time": {"type": "string", "description": "conversation time, e.g. '2020-01-01 00:00:00'"},
            "summary_memory": {"type": "string", "description": "summary_memory"},
        }
        return properties, ["conversation_time", "summary_memory"]

    def _build_parameters(self) -> dict:
        properties, required = self._build_item_schema()
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    async def execute(self):
        summary_memory = self.context.get("summary_memory", "")
        conversation_time = self.context.get("conversation_time", "")

        if not summary_memory:
            self.output = "No summary_memory provided for addition."
            return

        metadata: dict = {"conversation_time": conversation_time}
        try:
            metadata["time_int"] = int(conversation_time.split(" ")[0].replace("-", ""))
        except Exception:
            pass

        memory_node = MemoryNode(
            memory_type=self.memory_type,
            memory_target=self.memory_target,
            when_to_use="",
            content=summary_memory,
            ref_memory_id=self.history_node.memory_id,
            author=self.author,
            metadata=metadata,
        )

        vector_node = memory_node.to_vector_node()
        vector_id = vector_node.vector_id
        await self.vector_store.delete(vector_ids=[vector_id])
        await self.vector_store.insert(nodes=[vector_node])
        self.memory_nodes.append(memory_node)

        self.output = f"Successfully added summary memory to vector_store."
        logger.info(self.output)
