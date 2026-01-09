"""Add summary memory operation for vector store."""

from loguru import logger

from .add_memory import AddMemory
from ...core.context import C
from ...core.enumeration import MemoryType
from ...core.schema import MemoryNode


@C.register_op()
class AddSummaryMemory(AddMemory):
    """Add LLM-summarized memories to vector store.

    Differences from AddMemory:
    - Single memory mode only (enable_multiple=False)
    - Uses 'summary_memory' parameter instead of 'memory_content'
    - No when_to_use field (add_when_to_use=False)
    - Metadata fields can be customized via `metadata_desc` parameter
    """

    def __init__(self, metadata_desc: dict[str, str] | None = None, **kwargs):
        """Initialize AddSummaryMemory.

        Args:
            metadata_desc: Dictionary defining metadata fields and their descriptions.
            **kwargs: Additional arguments for AddMemory.
        """
        # Force single mode and disable when_to_use
        kwargs["enable_multiple"] = False
        kwargs["add_when_to_use"] = False
        super().__init__(metadata_desc=metadata_desc, **kwargs)

    def _build_parameters(self) -> dict:
        """Build input schema for summary memory addition."""
        properties = {
            "summary_memory": {
                "type": "string",
                "description": self.get_prompt("summary_memory"),
            },
        }
        required = ["summary_memory"]

        # Add metadata field if metadata_desc is provided and not empty
        if self.metadata_desc:
            metadata_properties = {
                key: {"type": "string", "description": desc} for key, desc in self.metadata_desc.items()
            }
            # Generate dynamic description based on metadata_desc fields
            field_descriptions = "\n".join([f"  - {key}: {desc}" for key, desc in self.metadata_desc.items()])
            metadata_description = f"Optional metadata for the memory. Available fields:\n{field_descriptions}"

            properties["metadata"] = {
                "type": "object",
                "description": metadata_description,
                "properties": metadata_properties,
            }
            required.append("metadata")

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _build_memory_node(
        self,
        memory_content: str,
        memory_type: MemoryType | None = None,
        memory_target: str = "",
        ref_memory_id: str = "",
        when_to_use: str = "",
        author: str = "",
        metadata: dict | None = None,
    ) -> MemoryNode:
        """Build MemoryNode from content, when_to_use, and metadata."""
        node = MemoryNode(
            memory_type=MemoryType.SUMMARY,
            memory_target="",
            when_to_use="",
            content=memory_content,
            ref_memory_id=self.ref_memory_id,
            author=self.author,
            metadata=metadata or {},
        )
        logger.info(f"Adding summary memory: {node.model_dump_json(indent=2, exclude_none=True)}")

        return node

    async def execute(self):
        """Execute addition: map summary_memory to memory_content and call parent."""
        # Map summary_memory to memory_content
        summary_memory = self.context.get("summary_memory", "")
        if not summary_memory:
            self.output = "No summary memory content provided for addition."
            logger.warning(self.output)
            return

        self.context["memory_content"] = summary_memory
        await super().execute()
