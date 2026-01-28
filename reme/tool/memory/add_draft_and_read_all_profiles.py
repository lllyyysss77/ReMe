"""Add draft profile and read all profiles from local storage"""
from pathlib import Path

from loguru import logger

from .base_memory_tool import BaseMemoryTool
from .profile_handler import ProfileHandler
from ...core.schema import ToolCall


class AddDraftAndReadAllProfiles(BaseMemoryTool):
    """Tool to add draft profile and read all profiles"""

    def __init__(self, profile_path: str, enable_memory_target: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.profile_path: str = profile_path
        self.enable_memory_target: bool = enable_memory_target

    def _build_query_parameters(self) -> dict:
        """Build the query parameters schema"""
        properties = {
            "profile_draft": {
                "type": "string",
                "description": "profile_draft",
            },
        }
        required = ["profile_draft"]

        if self.enable_memory_target:
            properties["memory_target"] = {
                "type": "string",
                "description": "memory_target",
            }
            required.append("memory_target")

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "Add draft profile and read all profiles from local storage.",
                "parameters": self._build_query_parameters(),
            },
        )

    def _build_multiple_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "Add draft profile and read all profiles from local storage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "draft_items": {
                            "type": "array",
                            "description": "List of draft profile items.",
                            "items": self._build_query_parameters(),
                        },
                    },
                    "required": ["draft_items"],
                },
            },
        )

    async def execute(self):
        if self.enable_multiple:
            draft_items = self.context.get("draft_items", [])
        else:
            draft_items = [self.context]

        # Collect all profiles from all targets
        all_profiles = []
        targets_processed = set()

        for item in draft_items:
            if self.enable_memory_target:
                target = item["memory_target"]
            else:
                target = self.memory_target

            # Skip if already processed this target
            if target in targets_processed:
                continue
            targets_processed.add(target)

            profile_handler = ProfileHandler(
                profile_path=Path(self.profile_path) / self.vector_store.collection_name,
                memory_target=target,
            )

            profiles_str = profile_handler.read_all(add_profile_id=True)
            if profiles_str:
                all_profiles.append(f"## Profiles for {target}:\n{profiles_str}")

        if not all_profiles:
            output = "No profiles found."
            logger.info(output)
            return output

        output = "\n\n".join(all_profiles)
        logger.info(f"Successfully read profiles for {len(targets_processed)} target(s)")
        return output
