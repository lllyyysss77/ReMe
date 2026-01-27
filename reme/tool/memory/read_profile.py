"""Read user profile tool"""
from pathlib import Path

from loguru import logger

from .base_memory_tool import BaseMemoryTool
from .profile_handler import ProfileHandler
from ...core.schema import ToolCall


class ReadProfile(BaseMemoryTool):
    """Tool to read all user profiles"""

    def __init__(self, profile_path: str, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)
        self.profile_path: str = profile_path

    def _build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema"""
        return ToolCall(
            **{
                "description": "Read all user profiles.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        )

    async def execute(self):
        profile_handler = ProfileHandler(
            profile_path=Path(self.profile_path) / self.vector_store.collection_name,
            memory_target=self.memory_target,
        )

        profiles_str = profile_handler.read_all()
        if not profiles_str:
            output = "No profiles found."
            logger.info(output)
            return output

        logger.info(f"Successfully read profiles")
        return profiles_str
