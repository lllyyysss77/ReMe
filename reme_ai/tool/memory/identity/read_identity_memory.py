"""Read identity memory operation."""

from loguru import logger

from ...base_memory_tool import BaseMemoryTool
from ....core.context import C


@C.register_op()
class ReadIdentityMemory(BaseMemoryTool):
    """Read identity memory for agent self-cognition."""

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)

    def _build_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self):
        result = self.meta_memory.load("identity_memory")
        identity_memory = result if result is not None else ""

        if identity_memory:
            self.output = f"Identity memory:\n{identity_memory}"
            logger.info("Retrieved identity memory")
        else:
            self.output = "No identity memory found."
            logger.info(self.output)
