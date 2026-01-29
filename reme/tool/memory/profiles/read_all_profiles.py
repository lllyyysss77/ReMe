"""Read user profile tool"""

from loguru import logger

from .profile_handler import ProfileHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class ReadAllProfiles(BaseMemoryTool):
    """Tool to read all user profiles"""

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)

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
        profile_handler = ProfileHandler(profile_path=self.profile_path, memory_target=self.memory_target)
        profiles_str = profile_handler.read_all(add_profile_id=True)
        if not profiles_str:
            output = "No profiles found."
            logger.info(output)
            return output

        logger.info("Successfully read profiles")
        return profiles_str
