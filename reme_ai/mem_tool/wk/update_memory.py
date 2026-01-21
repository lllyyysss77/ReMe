from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core_old.schema import MemoryNode


class UpdateMemory(BaseMemoryTool):

    def __init__(self, **kwargs):
        kwargs['enable_multiple'] = True
        super().__init__(**kwargs)

    def _build_item_schema(self) -> tuple[dict, list[str]]:
        properties = {
            "memory_id": {
                "type": "string",
                "description": self.get_prompt("memory_id"),
            },
            "memory_content": {
                "type": "string",
                "description": self.get_prompt("memory_content"),
            },
            "metadata": {
                "type": "object",
                "description": "metadata for the memory.",
            }
        }
        required = ["memory_id", "memory_content", "metadata"]
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

    def _extract_memory_data(self, mem_dict: dict) -> tuple[str, str, dict]:
        memory_id = mem_dict.get("memory_id", "")
        memory_content = mem_dict.get("memory_content", "")
        raw_metadata = mem_dict.get("metadata", {})
        metadata = {key: str(value).strip() for key, value in raw_metadata.items() if value}
        return memory_id, memory_content, metadata

    async def execute(self):
        old_memory_ids: list[str] = []
        new_memory_nodes: list[MemoryNode] = []

        memories: list[dict] = self.context.get("memories", [])
        if not memories:
            self.output = "No memories provided for update."
            return

        for mem in memories:
            memory_id, memory_content, metadata = self._extract_memory_data(mem)
            if not memory_id or not memory_content:
                logger.warning(f"Skipping memory with missing id or content: {mem}")
                continue
            old_memory_ids.append(memory_id)
            new_memory_nodes.append(self._build_memory_node(memory_content, metadata=metadata))

        if not old_memory_ids or not new_memory_nodes:
            self.output = "No valid memories provided for update."
            return

        vector_nodes = [node.to_vector_node() for node in new_memory_nodes]
        new_vector_ids = [node.vector_id for node in vector_nodes]

        all_ids_to_delete = list(set(old_memory_ids + new_vector_ids))
        await self.vector_store.delete(vector_ids=all_ids_to_delete)
        await self.vector_store.insert(nodes=vector_nodes)
        self.memory_nodes = new_memory_nodes

        self.output = f"Successfully updated {len(new_memory_nodes)} memories in vector_store."
        logger.info(self.output)
