from typing import Literal
from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core_old.schema.memory_node import MemoryNode


class ReadUserProfile(BaseMemoryTool):

    def __init__(self, add_memory_type_target: bool = False, show_ids: Literal["both", "profile", "history", "none"] = "both", **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)
        self.add_memory_type_target = add_memory_type_target
        self.show_ids = show_ids

    def _build_tool_description(self) -> str:
        return "Read user profile."

    def _build_parameters(self) -> dict:
        if self.add_memory_type_target:
            return {
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "description": "memory_type",
                    },
                    "memory_target": {
                        "type": "string",
                        "description": "memory_target",
                    },
                },
                "required": ["memory_type", "memory_target"],
            }
        else:
            return {
                "type": "object",
                "properties": {},
                "required": [],
            }

    async def execute(self):
        # Determine which IDs to show
        show_profile_id = self.show_ids in ("both", "profile")
        show_history_id = self.show_ids in ("both", "history")

        cache_key = f"{self.memory_type}_{self.memory_target}".replace(" ", "_").lower()
        cached_data = self.meta_memory.load(cache_key, auto_clean=False)

        if not cached_data:
            self.output = "### User Profile\nNo user profile found."
            logger.info(f"empty cached_data={cache_key}")
            return

        memory_nodes = [MemoryNode(**node_data) for node_data in cached_data]
        memory_nodes.sort(key=lambda n: n.metadata.get("conversation_time", ""))

        memory_formated = []
        for node in memory_nodes:
            node_formated_parts = []

            # Add profile_id if enabled
            if show_profile_id:
                node_formated_parts.append(f"profile_id={node.memory_id}")

            # Always add profile_content
            node_formated_parts.append(f"profile_content={node.content}")

            # Add conversation_time if available
            if "conversation_time" in node.metadata and node.metadata["conversation_time"]:
                node_formated_parts.append(f"conversation_time={node.metadata['conversation_time']}")

            # Add history_id if enabled and available
            if show_history_id and node.ref_memory_id:
                node_formated_parts.append(f"history_id={node.ref_memory_id}")

            node_formated = " ".join(node_formated_parts)
            memory_formated.append(node_formated.strip())

        self.output = "### User Profile\n" + "\n".join(memory_formated)
        logger.info(f"Read {len(memory_formated)} nodes from cache key: {cache_key}")
