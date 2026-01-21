"""Update memories operation for vector store."""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core_old.context import C
from ...core_old.schema import MemoryNode


@C.register_op()
class UpdateMemories(BaseMemoryTool):
    """Update memories by removing old ones and adding new ones in a single atomic operation.

    This tool is useful for updating memories when you need to remove outdated information
    and add updated information at the same time. Only supports batch mode (multiple operations).
    Metadata fields can be customized via `metadata_desc` parameter.
    """

    def __init__(self, add_when_to_use: bool = False, metadata_desc: dict[str, str] | None = None, **kwargs):
        """Initialize UpdateMemories.

        Args:
            add_when_to_use: Include when_to_use field for better retrieval. Defaults to True.
            metadata_desc: Dictionary defining metadata fields and their descriptions.
            **kwargs: Additional arguments for BaseMemoryTool.
        """
        kwargs["enable_multiple"] = True
        super().__init__(**kwargs)
        self.add_when_to_use: bool = add_when_to_use
        self.metadata_desc: dict[str, str] = metadata_desc or {}

    def _build_item_schema(self) -> tuple[dict, list[str]]:
        """Build shared schema properties and required fields for memory items to add.

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

        # Add metadata field if metadata_desc is provided and not empty
        if self.metadata_desc:
            metadata_properties = {
                key: {"type": "string", "description": desc} for key, desc in self.metadata_desc.items()
            }
            properties["metadata"] = {
                "type": "object",
                "description": "metadata",
                "properties": metadata_properties,
            }
            required.append("metadata")

        return properties, required

    def _build_multiple_parameters(self) -> dict:
        """Build input schema for update operation.

        Only supports batch mode with both removal and addition.
        """
        item_properties, required_fields = self._build_item_schema()
        return {
            "type": "object",
            "properties": {
                "memory_ids_to_delete": {
                    "type": "array",
                    "description": self.get_prompt("memory_ids_to_delete"),
                    "items": {"type": "string"},
                },
                "memories_to_add": {
                    "type": "array",
                    "description": self.get_prompt("memories_to_add"),
                    "items": {
                        "type": "object",
                        "properties": item_properties,
                        "required": required_fields,
                    },
                },
            },
            "required": ["memory_ids_to_delete", "memories_to_add"],
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
        metadata = mem_dict.get("metadata", {}) if self.metadata_desc else {}
        return memory_content, when_to_use, metadata

    async def execute(self):
        """Execute update operation: first remove old memories by IDs, then add new updated memories."""
        # Get removal IDs
        memory_ids_to_delete = self.context.get("memory_ids_to_delete", [])
        memory_ids_to_delete = [m for m in memory_ids_to_delete if m]
        # Deduplicate memory IDs to avoid redundant deletions
        memory_ids_to_delete = list(dict.fromkeys(memory_ids_to_delete))

        # Get memories to add
        memories_to_add = self.context.get("memories_to_add", [])

        # Validate input
        if not memory_ids_to_delete and not memories_to_add:
            self.output = "No memories to remove or add. Operation has been done."
            return

        removed_count = 0
        added_count = 0

        # Step 1: Remove old memories
        if memory_ids_to_delete:
            await self.vector_store.delete(vector_ids=memory_ids_to_delete)
            self.memory_nodes.extend(memory_ids_to_delete)
            removed_count = len(memory_ids_to_delete)
            logger.info(f"Removed {removed_count} memories from vector_store.")

        # Step 2: Add new updated memories
        if memories_to_add:
            memory_nodes: list[MemoryNode] = []
            for mem in memories_to_add:
                memory_content, when_to_use, metadata = self._extract_memory_data(mem)
                if not memory_content:
                    logger.warning("Skipping memory with empty content")
                    continue

                memory_nodes.append(self._build_memory_node(memory_content, when_to_use=when_to_use, metadata=metadata))

            if memory_nodes:
                # Convert to VectorNodes and collect IDs
                vector_nodes = [node.to_vector_node() for node in memory_nodes]
                vector_ids: list[str] = [node.vector_id for node in vector_nodes]

                # Delete existing IDs (upsert behavior), then insert
                await self.vector_store.delete(vector_ids=vector_ids)
                await self.vector_store.insert(nodes=vector_nodes)
                added_count = len(memory_nodes)
                logger.info(f"Added {added_count} new memories to vector_store.")

            self.memory_nodes.extend(memory_nodes)

        # Generate output message
        operations = []
        if removed_count > 0:
            operations.append(f"removed {removed_count} old memories")
        if added_count > 0:
            operations.append(f"added {added_count} new memories")

        if operations:
            self.output = f"Successfully {' and '.join(operations)} in vector_store."
        else:
            self.output = "Operation has been done."

        logger.info(self.output)
