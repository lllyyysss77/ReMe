"""ReMe retriever v2 that autonomously retrieves memories from multiple angles."""

from typing import List

from ..base_memory_agent import BaseMemoryAgent
from ...core_old.enumeration import Role
from ...core_old.schema import Message
from ...core_old.utils import format_messages


class ReMeRetrieverV3(BaseMemoryAgent):

    def __init__(self, meta_memories: list[dict] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.meta_memories: list[dict] = meta_memories or []

    async def _read_meta_memories(self) -> str:
        """Fetch all meta-memory entries that define specialized memory agents."""
        from ...mem_tool import ReadMetaMemory

        op = ReadMetaMemory(enable_identity_memory=False)
        return op.format_memory_metadata(self.meta_memories)

    async def build_messages(self) -> List[Message]:
        """Build messages with system prompt and user message."""
        if self.context.get("query"):
            context = self.context.query
        elif self.context.get("messages"):
            context = format_messages(self.context.messages)
        else:
            raise ValueError("input must have either `query` or `messages`")

        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            meta_memory_info=await self._read_meta_memories(),
            context=context,
        )

        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=self.get_prompt("user_message")),
        ]

        return messages
