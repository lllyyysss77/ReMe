"""Add identity memory tool"""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class AddIdentity(BaseMemoryTool):
    """Tool to add or update agent identity memory"""

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "add or update agent identity memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "identity_memory": {
                            "type": "string",
                            "description": "Agent identity content, such as role, personality, or current state.",
                        },
                    },
                    "required": ["identity_memory"],
                },
            },
        )

    async def execute(self):
        identity_memory = self.context.get("identity_memory", "")

        if not identity_memory:
            logger.warning("No valid identity memory provided")
            return "No valid identity memory provided for update."

        self.local_memory.save("identity_memory", identity_memory)
        logger.info(f"Successfully updated identity memory: {identity_memory}")
        return "Successfully updated identity memory."
