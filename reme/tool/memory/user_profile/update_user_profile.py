"""Update user profile tool"""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....core.enumeration import MemoryType
from ....core.schema import ToolCall, MemoryNode
from ....core.utils import deduplicate_memories


class UpdateUserProfile(BaseMemoryTool):
    """Tool to update user profile by adding or removing profile entries"""

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = True
        super().__init__(**kwargs)

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the multiple tool call schema"""
        return ToolCall(
            **{
                "description": "update user profile by adding or removing profile entries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "profile_ids_to_delete": {
                            "type": "array",
                            "description": "List of profile IDs to delete",
                            "items": {"type": "string"},
                        },
                        "profiles_to_add": {
                            "type": "array",
                            "description": "List of profiles to add",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "update_time": {
                                        "type": "string",
                                        "description": "Update time, e.g. '2020-01-01 00:00:00'",
                                    },
                                    "profile_key": {
                                        "type": "string",
                                        "description": "Profile key or category, e.g. 'name'",
                                    },
                                    "profile_value": {
                                        "type": "string",
                                        "description": "Profile value or content, e.g. 'John Smith'",
                                    },
                                },
                                "required": ["update_time", "profile_key", "profile_value"],
                            },
                        },
                    },
                    "required": ["profile_ids_to_delete", "profiles_to_add"],
                },
            },
        )

    async def execute(self):
        # Get and deduplicate profile IDs to delete
        self.context.memory_type = MemoryType.PERSONAL

        profile_ids_to_delete = self.context.get("profile_ids_to_delete", [])
        profile_ids_to_delete = list(dict.fromkeys([pid for pid in profile_ids_to_delete if pid]))
        profiles_to_add = self.context.get("profiles_to_add", [])

        if not profile_ids_to_delete and not profiles_to_add:
            return "No profiles to remove or add. Operation completed."

        # Load existing profiles from local memory
        cached_data = self.local_memory.load(self.memory_cache_key, auto_clean=False)
        existing_nodes = [MemoryNode(**data) for data in cached_data] if cached_data else []

        # Remove profiles
        removed_count = 0
        if profile_ids_to_delete:
            original_count = len(existing_nodes)
            existing_nodes = [n for n in existing_nodes if n.memory_id not in profile_ids_to_delete]
            removed_count = original_count - len(existing_nodes)
            logger.info(f"Removed {removed_count} profiles.")

        # Add new profiles
        new_nodes = []
        if profiles_to_add:
            for profile in profiles_to_add:
                node = MemoryNode(
                    memory_type=self.memory_type,
                    memory_target=self.memory_target,
                    when_to_use=profile.get("profile_key", ""),
                    content=profile.get("profile_value", ""),
                    ref_memory_id=self.history_node.memory_id,
                    author=self.author,
                    metadata={"update_time": profile.get("update_time", "")},
                )
                new_nodes.append(node)
            logger.info(f"Added {len(new_nodes)} new profiles.")

        # Deduplicate and save updated profiles
        self.memory_nodes.extend(new_nodes)
        updated_nodes = deduplicate_memories(existing_nodes + new_nodes)
        nodes_data = [node.model_dump(exclude_none=True) for node in updated_nodes]
        self.local_memory.save(self.memory_cache_key, nodes_data)

        # Build output message
        operations = []
        if removed_count > 0:
            operations.append(f"removed {removed_count} old profiles.")
        if len(new_nodes) > 0:
            operations.append(f"added {len(new_nodes)} new profiles.")
        operations.append("Operation completed.")
        logger.info("\n".join(operations))
        return "\n".join(operations)
