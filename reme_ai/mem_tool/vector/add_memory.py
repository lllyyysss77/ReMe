"""Add memory operation for vector store."""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.context import C
from ...core.schema import MemoryNode


@C.register_op()
class AddMemory(BaseMemoryTool):
    """Add memories to vector store with optional when_to_use and metadata.

    Supports single/multiple addition modes via `enable_multiple` parameter.
    """

    def __init__(self, add_when_to_use: bool = False, add_metadata: bool = True, **kwargs):
        """Initialize AddMemory.

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
        properties = {}
        required = []

        if self.add_when_to_use:
            properties["when_to_use"] = {
                "type": "string",
                "description": self.get_prompt("when_to_use"),
            }
            required.append("when_to_use")

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
        """Build input schema for single memory addition."""
        properties, required = self._build_item_schema()
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _build_multiple_parameters(self) -> dict:
        """Build input schema for multiple memory addition."""
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
        """Extract memory data from a dictionary with proper defaults.

        Args:
            mem_dict: Dictionary containing memory fields.

        Returns:
            Tuple of (memory_content, when_to_use, metadata).
        """
        memory_content = mem_dict.get("memory_content", "")
        when_to_use = mem_dict.get("when_to_use", "") if self.add_when_to_use else ""
        metadata = mem_dict.get("metadata", {}) if self.add_metadata else {}
        return memory_content, when_to_use, metadata

    async def execute(self):
        """Execute addition: delete existing IDs (upsert), then insert new memories."""
        memory_nodes: list[MemoryNode] = []

        if self.enable_multiple:
            memories: list[dict] = self.context.get("memories", [])
            if not memories:
                self.output = "No memories provided for addition."
                return

            for mem in memories:
                memory_content, when_to_use, metadata = self._extract_memory_data(mem)
                if not memory_content:
                    logger.warning("Skipping memory with empty content")
                    continue

                memory_nodes.append(self._build_memory_node(memory_content, when_to_use=when_to_use, metadata=metadata))

        else:
            memory_content, when_to_use, metadata = self._extract_memory_data(self.context)
            if not memory_content:
                self.output = "No memory content provided for addition."
                return

            memory_nodes.append(self._build_memory_node(memory_content, when_to_use=when_to_use, metadata=metadata))

        if not memory_nodes:
            self.output = "No valid memories provided for addition."
            return

        # Convert to VectorNodes and collect IDs
        vector_nodes = [node.to_vector_node() for node in memory_nodes]
        vector_ids: list[str] = [node.vector_id for node in vector_nodes]

        # Delete existing IDs (upsert behavior), then insert
        await self.vector_store.delete(vector_ids=vector_ids)
        await self.vector_store.insert(nodes=vector_nodes)

        self.output = f"Successfully added {len(memory_nodes)} memories to vector_store."
        logger.info(self.output)
