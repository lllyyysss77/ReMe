"""Read meta memory tool"""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.enumeration import MemoryType
from ....core.schema import ToolCall


class ReadMetaMemory(BaseMemoryTool):
    """Tool to read memory metadata from meta storage"""

    TYPE_DESC_DICT = {
        MemoryType.IDENTITY.value: "self-cognition memory storing agent's identity and state",
        MemoryType.PERSONAL.value: "person-specific memory storing preferences and context",
        MemoryType.PROCEDURAL.value: "procedural memory storing how-to knowledge and processes",
    }

    def __init__(self, enable_identity_memory: bool = False, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)
        self.enable_identity_memory = enable_identity_memory

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "read memory metadata registry to see what types of memories are being tracked.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        )

    async def execute(self):
        # Load and filter meta memories
        result = self.local_memory.load("meta_memories")
        all_memories = result if result is not None else []

        memories = [
            m for m in all_memories if m.get("memory_type") in [MemoryType.PERSONAL.value, MemoryType.PROCEDURAL.value]
        ]

        if self.enable_identity_memory:
            memories.append(
                {
                    "memory_type": MemoryType.IDENTITY.value,
                    "memory_target": "self",
                },
            )

        # Format output
        if memories:
            lines = [
                f"- {m['memory_type']}({m['memory_target']}): {self.TYPE_DESC_DICT.get(m['memory_type'], '')}"
                for m in memories
            ]

            output = "\n".join(lines)
            logger.info(f"Retrieved {len(memories)} meta memory entries")
        else:
            output = "No memory metadata found."
            logger.info(output)

        return output
