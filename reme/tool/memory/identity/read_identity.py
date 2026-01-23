"""Read identity memory tool"""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class ReadIdentity(BaseMemoryTool):
    """Tool to read agent identity memory"""

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "read agent identity memory.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        )

    async def execute(self):
        identity_memory = self.local_memory.load("identity_memory")

        if not identity_memory:
            logger.info("No identity memory found")
            return "No identity memory found."

        logger.info(f"Read identity memory: {identity_memory}")
        return f"Identity\n{identity_memory}"
