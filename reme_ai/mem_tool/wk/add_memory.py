from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.schema import MemoryNode


class AddMemory(BaseMemoryTool):

    def __init__(self, **kwargs):
        kwargs['enable_multiple'] = True
        super().__init__(**kwargs)

    def _build_item_schema(self) -> tuple[dict, list[str]]:
        properties = {
            "memory_content": {
                "type": "string",
                "description": self.get_prompt("memory_content"),
            },
            "metadata": {
                "type": "object",
                "description": "metadata for the memory.",
            }
        }
        required = ["memory_content"]
        return properties, required

    def _build_multiple_parameters(self) -> dict:
        item_properties, required_fields = self._build_item_schema()
        return {
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "description": self.get_prompt("memories"),
                    "items": {
                        "type": "object",
                        "properties": item_properties,
                        "required": required_fields,
                    },
                },
            },
            "required": ["memories"],
        }

    def _extract_memory_data(self, mem_dict: dict) -> tuple[str, dict]:
        memory_content = mem_dict.get("memory_content", "")
        raw_metadata = mem_dict.get("metadata", {})
        metadata = {key: str(value).strip() for key, value in raw_metadata.items() if value}
        return memory_content, metadata

    async def execute(self):
        memory_nodes: list[MemoryNode] = []

        memories: list[dict] = self.context.get("memories", [])
        if not memories:
            self.output = "No memories provided for addition."
            return

        for mem in memories:
            memory_content, metadata = self._extract_memory_data(mem)
            if not memory_content:
                logger.warning("Skipping memory with empty content")
                continue

            memory_nodes.append(self._build_memory_node(memory_content, metadata=metadata))

        if not memory_nodes:
            self.output = "No valid memories provided for addition."
            return

        vector_nodes = [node.to_vector_node() for node in memory_nodes]
        vector_ids: list[str] = [node.vector_id for node in vector_nodes]

        await self.vector_store.delete(vector_ids=vector_ids)
        await self.vector_store.insert(nodes=vector_nodes)
        self.memory_nodes = memory_nodes

        self.output = f"Successfully added {len(memory_nodes)} memories to vector_store."
        logger.info(self.output)
