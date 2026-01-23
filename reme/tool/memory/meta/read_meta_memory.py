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

    def format_memory_metadata(self, memories: list[dict[str, str]]) -> str:
        """Format memory metadata into a readable string."""
        if not memories:
            return ""

        lines = []
        for memory in memories:
            memory_type = memory["memory_type"]
            memory_target = memory["memory_target"]
            description = self.TYPE_DESC_DICT[memory_type]
            lines.append(f"- {memory_type}({memory_target}): {description}")

        return "\n".join(lines)

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
        output = self.format_memory_metadata(memories)
        if output:
            logger.info(f"Retrieved {len(memories)} meta memory entries")
        else:
            output = "No memory metadata found."
            logger.info(output)

        return output
