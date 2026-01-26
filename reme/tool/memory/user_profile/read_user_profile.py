"""Read user profile tool"""

from typing import Literal

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.enumeration import MemoryType
from ....core.schema import ToolCall, MemoryNode


class ReadUserProfile(BaseMemoryTool):
    """Tool to read user profile from local memory"""

    def __init__(self, show_id: Literal["profile", "history"] = "profile", **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)
        self.show_id = show_id

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "read user profile.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        )

    async def execute(self):
        self.context.memory_type = MemoryType.PERSONAL
        cached_data = self.local_memory.load(self.memory_cache_key, auto_clean=False)

        if not cached_data:
            logger.info(f"No cached data found for {self.memory_cache_key}")
            return ""

        nodes = [MemoryNode(**data) for data in cached_data]
        self.memory_nodes = nodes
        nodes.sort(key=lambda n: n.metadata.get("update_time", ""))

        formatted_profiles = []
        for node in nodes:
            parts = []
            if self.show_id == "profile":
                parts.append(f"profile_id={node.memory_id}")

            if update_time := node.metadata.get("update_time"):
                parts.append(f"update_time={update_time}")

            parts.append(f"{node.when_to_use}: {node.content}")

            if self.show_id == "history":
                parts.append(f"history_id={node.ref_memory_id}")

            formatted_profiles.append(" ".join(parts))

        logger.info(f"Read {len(formatted_profiles)} profiles from cache key: {self.memory_cache_key}")

        return "\n".join(formatted_profiles).strip()
