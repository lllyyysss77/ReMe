"""Update memory operation for vector store."""

from loguru import logger

from ...base_memory_tool import BaseMemoryTool
from ....core.context import C
from ....core.schema import MemoryNode


@C.register_op()
class UpdateMemory(BaseMemoryTool):
    """Update memories by deleting old ones and inserting new ones.

    Supports single/multiple update modes via `enable_multiple` parameter.
    """

    def __init__(
        self,
        add_when_to_use: bool = False,
        add_metadata: bool = True,
        **kwargs,
    ):
        """Initialize UpdateMemory.

        Args:
            add_when_to_use: Include when_to_use field for better retrieval.
            add_metadata: Include metadata field for additional info.
            **kwargs: Additional arguments for BaseMemoryTool.
        """
        super().__init__(**kwargs)
        self.add_when_to_use: bool = add_when_to_use
        self.add_metadata: bool = add_metadata

    def _build_item_schema(self) -> tuple[dict, list[str]]:
        """Build shared schema properties and required fields for memory items.

        Returns:
            Tuple of (properties dict, required fields list).
        """
        properties = {
            "memory_id": {
                "type": "string",
                "description": self.get_prompt("memory_id"),
            },
        }
        required = ["memory_id"]

        if self.add_when_to_use:
            properties["when_to_use"] = {
                "type": "string",
                "description": self.get_prompt("when_to_use"),
            }

        properties["memory_content"] = {
            "type": "string",
            "description": self.get_prompt("memory_content"),
        }
        required.append("memory_content")

        if self.add_metadata:
            properties["metadata"] = {
                "type": "object",
                "description": self.get_prompt("metadata"),
            }

        return properties, required

    def _build_parameters(self) -> dict:
        """Build input schema for single memory update."""
        properties, required = self._build_item_schema()
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _build_multiple_parameters(self) -> dict:
        """Build input schema for multiple memory update."""
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

    def _extract_memory_data(self, mem_dict: dict) -> tuple[str, str, str, dict]:
        """Extract memory update data from a dictionary with proper defaults.

        Args:
            mem_dict: Dictionary containing memory fields.

        Returns:
            Tuple of (memory_id, memory_content, when_to_use, metadata).
        """
        memory_id = mem_dict.get("memory_id", "")
        memory_content = mem_dict.get("memory_content", "")
        when_to_use = mem_dict.get("when_to_use", "") if self.add_when_to_use else ""
        metadata = mem_dict.get("metadata", {}) if self.add_metadata else {}
        return memory_id, memory_content, when_to_use, metadata

    async def execute(self):
        """Execute update: delete old memories by ID, insert new ones with updated content."""
        # Collect old IDs to delete and new nodes to insert
        old_memory_ids: list[str] = []
        new_memory_nodes: list[MemoryNode] = []

        if self.enable_multiple:
            memories: list[dict] = self.context.get("memories", [])
            if not memories:
                self.output = "No memories provided for update."
                return

            for mem in memories:
                memory_id, memory_content, when_to_use, metadata = self._extract_memory_data(mem)
                if not memory_id or not memory_content:
                    logger.warning(f"Skipping memory with missing id or content: {mem}")
                    continue
                old_memory_ids.append(memory_id)
                new_memory_nodes.append(
                    self._build_memory_node(
                        memory_content,
                        when_to_use=when_to_use,
                        metadata=metadata,
                    ),
                )

        else:
            memory_id, memory_content, when_to_use, metadata = self._extract_memory_data(self.context)
            if not memory_id or not memory_content:
                self.output = "No memory ID or content provided for update."
                return
            old_memory_ids.append(memory_id)
            new_memory_nodes.append(self._build_memory_node(memory_content, when_to_use=when_to_use, metadata=metadata))

        if not old_memory_ids or not new_memory_nodes:
            self.output = "No valid memories provided for update."
            return

        # Convert to VectorNodes and collect IDs
        vector_nodes = [node.to_vector_node() for node in new_memory_nodes]
        new_vector_ids = [node.vector_id for node in vector_nodes]

        # Delete old and duplicate new IDs (upsert behavior)
        all_ids_to_delete = list(set(old_memory_ids + new_vector_ids))
        await self.vector_store.delete(vector_ids=all_ids_to_delete)
        await self.vector_store.insert(nodes=vector_nodes)

        self.output = f"Update: deleted {len(old_memory_ids)} old memories, added {len(new_memory_nodes)} new memories."
        logger.info(self.output)
