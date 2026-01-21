"""ReMe retriever v2 that autonomously retrieves memories from multiple angles."""

from typing import List

from ..base_memory_agent import BaseMemoryAgent
from ...core_old.context import C
from ...core_old.enumeration import Role
from ...core_old.schema import Message
from ...core_old.utils import format_messages


@C.register_op()
class ReMeRetrieverV2(BaseMemoryAgent):
    """Memory agent that autonomously retrieves memories from multiple angles.

    This retriever:
    - Directly queries memories based on user questions without time constraints
    - Tries multiple retrieval strategies: direct vector search, metadata filtering, partial filtering
    - Attempts at least 3 vector retrievals from different perspectives
    - Falls back to read_history if vector retrieval doesn't find sufficient information
    """

    def __init__(self, meta_memories: list[dict] | None = None, **kwargs):
        # Check if ReadHistory tool is available in the tools list
        tools = kwargs.get('tools', [])
        has_read_history = any(tool.__class__.__name__ == 'ReadHistory' for tool in tools)

        # Use simple prompt if ReadHistory is not available
        if not has_read_history:
            super().__init__(prompt_name="reme_retriever_v2_simple", **kwargs)
        else:
            super().__init__(**kwargs)

        self.meta_memories: list[dict] = meta_memories or []

    async def _read_meta_memories(self) -> str:
        """Fetch all meta-memory entries that define specialized memory agents."""
        from ...mem_tool import ReadMetaMemory

        op = ReadMetaMemory(enable_identity_memory=False)
        if self.meta_memories:
            return op.format_memory_metadata(self.meta_memories)
        else:
            await op.call()
            return str(op.output)

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
