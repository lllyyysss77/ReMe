"""Update user profile tool"""
from pathlib import Path

from loguru import logger

from .base_memory_tool import BaseMemoryTool
from .profile_handler import ProfileHandler
from ...core.schema import ToolCall


class UpdateProfile(BaseMemoryTool):
    """Tool to update user profile by adding or removing profile entries"""

    def __init__(self, profile_path: str, **kwargs):
        kwargs["enable_multiple"] = True
        super().__init__(**kwargs)

        self.profile_path: str = profile_path

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the multiple tool call schema"""
        return ToolCall(
            **{
                "description": "update user profile by removing and adding profile entries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "profile_ids_to_delete": {
                            "type": "array",
                            "description": "List of profile IDs to delete",
                            "items": {
                                "type": "string"
                            },
                        },
                        "profiles_to_add": {
                            "type": "array",
                            "description": "List of profiles to add",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "message_time": {
                                        "type": "string",
                                        "description": "Message time, e.g. '2020-01-01 00:00:00'",
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
                                "required": ["message_time", "profile_key", "profile_value"],
                            },
                        },
                    },
                    "required": ["profile_ids_to_delete", "profiles_to_add"],
                },
            },
        )

    async def execute(self):
        profile_handler = ProfileHandler(
            profile_path=Path(self.profile_path) / self.vector_store.collection_name,
            memory_target=self.memory_target,
        )

        # Get parameters
        profile_ids_to_delete = self.context.get("profile_ids_to_delete", [])
        profile_ids_to_delete = sorted(set([pid for pid in profile_ids_to_delete if pid]))
        profiles_to_add = self.context.get("profiles_to_add", [])

        if not profile_ids_to_delete and not profiles_to_add:
            return "No profiles to remove or add, operation completed."

        # Delete profiles using ProfileHandler (batch mode)
        removed_count = 0
        if profile_ids_to_delete:
            removed_count = profile_handler.delete(profile_ids_to_delete)

        # Add new profiles using ProfileHandler (batch mode)
        added_count = 0
        if profiles_to_add:
            new_nodes = profile_handler.add_batch(profiles=profiles_to_add, ref_memory_id=self.history_node.memory_id)
            self.memory_nodes.extend(new_nodes)
            added_count = len(new_nodes)

        # Build output message
        operations = []
        if removed_count > 0:
            operations.append(f"removed {removed_count} old profiles.")
        if added_count > 0:
            operations.append(f"added {added_count} new profiles.")
        operations.append("Operation completed.")
        logger.info("\n".join(operations))
        return "\n".join(operations)
