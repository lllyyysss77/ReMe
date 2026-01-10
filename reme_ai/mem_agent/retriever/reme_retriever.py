"""ReMe retriever that builds messages with meta memories."""

from typing import List

from ..base_memory_agent import BaseMemoryAgent
from ...core.context import C
from ...core.enumeration import Role
from ...core.schema import Message
from ...core.utils import get_now_time, format_messages


@C.register_op()
class ReMeRetriever(BaseMemoryAgent):
    """Memory agent that retrieves and builds messages with meta memory context."""

    def __init__(self, meta_memories: list[dict] | None = None, **kwargs):
        super().__init__(prompt_name="", **kwargs)
        # super().__init__(prompt_name="reme_retriever2", **kwargs)
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
        meta_memory_info = await self._read_meta_memories()
        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            now_time=get_now_time(),
            meta_memory_info=meta_memory_info,
        )

        messages = [Message(role=Role.SYSTEM, content=system_prompt)]
        if self.context.get("query"):
            messages.append(Message(role=Role.USER, content=self.context.query))
        elif self.context.get("messages"):
            messages.extend([Message(**m) for m in self.context.messages])
        else:
            raise ValueError("input must have either `query` or `messages`")

        return messages

    async def build_messages2(self) -> List[Message]:
        """Build messages with system prompt and user message."""
        if self.context.get("query"):
            context = self.context.query
        elif self.context.get("messages"):
            context = format_messages(self.context.messages)
        else:
            raise ValueError("input must have either `query` or `messages`")

        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            now_time=get_now_time(),
            meta_memory_info=await self._read_meta_memories(),
            context=context,
        )

        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=self.get_prompt("user_message")),
        ]

        return messages
