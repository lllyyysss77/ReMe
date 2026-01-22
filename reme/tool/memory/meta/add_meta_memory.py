"""Add meta memory tool"""

import json

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.enumeration import MemoryType
from ....core.schema import ToolCall


class AddMetaMemory(BaseMemoryTool):
    """Tool to add memory metadata entries to meta storage"""

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = True
        super().__init__(**kwargs)

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the multiple tool call schema"""
        return ToolCall(
            **{
                "description": "add memory metadata entries to register memory types and targets. "
                "Before using, verify Main Agent's Meta Memory doesn't already contain the "
                "same memory_type(memory_target) combinations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "meta_memories": {
                            "type": "array",
                            "description": "List of memory metadata entries to add",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "memory_type": {
                                        "type": "string",
                                        "description": "Type of memory: 'personal' for person-specific preferences, "
                                        "'procedural' for how-to knowledge",
                                        "enum": [MemoryType.PERSONAL.value, MemoryType.PROCEDURAL.value],
                                    },
                                    "memory_target": {
                                        "type": "string",
                                        "description": "Target identifier, "
                                        "e.g., person's name ('John') or domain ('deployment')",
                                    },
                                },
                                "required": ["memory_type", "memory_target"],
                            },
                        },
                    },
                    "required": ["meta_memories"],
                },
            },
        )

    async def execute(self):
        existing_memories: list[dict] = self.local_memory.load("meta_memories") or []
        existing_set = {(m["memory_type"], m["memory_target"]) for m in existing_memories}

        # Filter and build new memories to add
        new_memories: list[dict] = []
        meta_memories: list[dict] = self.context.get("meta_memories", [])

        for mem in meta_memories:
            memory_type = mem.get("memory_type", "")
            memory_target = mem.get("memory_target", "")

            # Check if valid and not duplicate
            if (
                memory_type in [MemoryType.PERSONAL.value, MemoryType.PROCEDURAL.value]
                and memory_target
                and (memory_type, memory_target) not in existing_set
            ):
                new_memories.append({"memory_type": memory_type, "memory_target": memory_target})
                existing_set.add((memory_type, memory_target))

        if not new_memories:
            output = "No new meta memories to add (all entries already exist or invalid)."
            logger.info(output)
            return output

        # Merge, sort and save
        all_memories = sorted(existing_memories + new_memories, key=lambda m: (m["memory_type"], m["memory_target"]))
        self.local_memory.save("meta_memories", all_memories)

        # Format output
        output = f"Successfully update meta memory entries: {json.dumps(new_memories, ensure_ascii=False)}"
        logger.info(output)
        return output
