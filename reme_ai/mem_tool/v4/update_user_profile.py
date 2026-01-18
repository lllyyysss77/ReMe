from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.schema.memory_node import MemoryNode
from ...core.utils import deduplicate_memories


class UpdateUserProfile(BaseMemoryTool):

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = True
        super().__init__(**kwargs)

    def _build_tool_description(self) -> str:
        return "Update user profile."

    def _build_multiple_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "profile_ids_to_delete": {
                    "type": "array",
                    "description": "profile_ids_to_delete",
                    "items": {
                        "type": "string"
                    },
                },
                "profiles_to_add": {
                    "type": "array",
                    "description": "profiles_to_add",
                    "items": {
                        "type": "object",
                        "properties": {
                            "conversation_time": {
                                "type": "string",
                                "description": "conversation_time, e.g. '2020-01-01 00:00:00'",
                            },
                            "profile_content": {
                                "type": "string",
                                "description": "profile_content",
                            },
                        },
                        "required": ["profile_content", "conversation_time"],
                    },
                },
            },
            "required": ["profile_ids_to_delete", "profiles_to_add"],
        }

    async def execute(self):
        profile_ids_to_delete = self.context.get("profile_ids_to_delete", [])
        profile_ids_to_delete = [m for m in profile_ids_to_delete if m]
        profile_ids_to_delete = list(dict.fromkeys(profile_ids_to_delete))
        profiles_to_add = self.context.get("profiles_to_add", [])

        if not profile_ids_to_delete and not profiles_to_add:
            self.output = "No profiles to remove or add. Operation has been done."
            return

        cache_key = f"{self.memory_type}_{self.memory_target}".replace(" ", "_").lower()
        cached_data = self.meta_memory.load(cache_key, auto_clean=False)
        if cached_data:
            existing_memory_nodes = [MemoryNode(**node_data) for node_data in cached_data]
        else:
            existing_memory_nodes = []

        removed_count = 0
        if profile_ids_to_delete:
            original_count = len(existing_memory_nodes)
            existing_memory_nodes = [n for n in existing_memory_nodes if n.memory_id not in profile_ids_to_delete]
            removed_count = original_count - len(existing_memory_nodes)
            logger.info(f"Removed {removed_count} profiles.")

        added_count = 0
        new_memory_nodes = []
        if profiles_to_add:
            for mem in profiles_to_add:
                memory_node = MemoryNode(
                    memory_type=self.memory_type,
                    memory_target=self.memory_target,
                    when_to_use="",
                    content=mem.get("profile_content", ""),
                    ref_memory_id=self.history_node.memory_id,
                    author=self.author,
                    metadata={"conversation_time": mem.get("conversation_time", "")},
                )
                new_memory_nodes.append(memory_node)
            added_count = len(new_memory_nodes)
            logger.info(f"Added {added_count} new profiles.")

        updated_memory_nodes = deduplicate_memories(existing_memory_nodes + new_memory_nodes)
        nodes_data = [node.model_dump(exclude_none=True) for node in updated_memory_nodes]
        self.meta_memory.save(cache_key, nodes_data)

        operations = []
        if removed_count > 0:
            operations.append(f"removed {removed_count} old profiles")
        if added_count > 0:
            operations.append(f"added {added_count} new profiles")

        if operations:
            self.output = f"Successfully {' and '.join(operations)} in user profile."
        else:
            self.output = "Operation has been done."
        logger.info(self.output)
