"""Add summary memory operation for vector store."""

from loguru import logger

from .add_memory import AddMemory
from ....core.context import C


@C.register_op()
class AddSummaryMemory(AddMemory):
    """Add LLM-summarized memories to vector store.

    Differences from AddMemory:
    - Single memory mode only (enable_multiple=False)
    - Uses 'summary_memory' parameter instead of 'memory_content'
    - No when_to_use field (add_when_to_use=False)
    """

    def __init__(
        self,
        add_metadata: bool = True,
        **kwargs,
    ):
        """Initialize AddSummaryMemory.

        Args:
            add_metadata: Include metadata field for additional info.
            **kwargs: Additional arguments for AddMemory.
        """
        # Force single mode and disable when_to_use
        kwargs["enable_multiple"] = False
        kwargs["add_when_to_use"] = False
        super().__init__(add_metadata=add_metadata, **kwargs)

    def _build_parameters(self) -> dict:
        """Build input schema for summary memory addition."""
        properties = {
            "summary_memory": {
                "type": "string",
                "description": self.get_prompt("summary_memory"),
            },
        }
        required = ["summary_memory"]

        if self.add_metadata:
            properties["metadata"] = {
                "type": "object",
                "description": self.get_prompt("metadata"),
            }

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

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
