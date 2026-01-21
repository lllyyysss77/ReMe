"""Add memory drafts operation for vector store."""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core_old.context import C


@C.register_op()
class AddMemoryDrafts(BaseMemoryTool):
    """Add memory drafts without persisting them to the database.

    This tool is useful for creating draft memories that can be reviewed and modified
    before final submission. Drafts are not persisted to the vector store.
    Metadata fields can be customized via `metadata_desc` parameter.
    """

    def __init__(self, add_when_to_use: bool = False, metadata_desc: dict[str, str] | None = None, **kwargs):
        """Initialize AddMemoryDrafts.

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
        """Build input schema for add drafts operation.

        Only supports batch mode for adding draft memories.
        """
        item_properties, required_fields = self._build_item_schema()
        return {
            "type": "object",
            "properties": {
                "memory_drafts": {
                    "type": "array",
                    "description": self.get_prompt("memory_drafts"),
                    "items": {
                        "type": "object",
                        "properties": item_properties,
                        "required": required_fields,
                    },
                },
            },
            "required": ["memory_drafts"],
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
        """Execute add drafts operation: create memory drafts without persisting to vector store."""
        self.output = f"Successfully created memory draft(s). These drafts are not yet persisted to the vector store."
        logger.info(self.output)
