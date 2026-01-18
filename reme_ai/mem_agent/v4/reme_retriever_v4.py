from loguru import logger

from ..base_memory_agent import BaseMemoryAgent
from ...core.enumeration import Role
from ...core.schema import Message
from ...core.utils import format_messages


class ReMeRetrieverV4(BaseMemoryAgent):

    def __init__(self, meta_memories: list[dict] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.meta_memories: list[dict] = meta_memories or []

    async def _read_meta_memories(self) -> str:
        from ...mem_tool import ReadMetaMemory
        meta_memory_info = ReadMetaMemory().format_memory_metadata(self.meta_memories)
        logger.info(f"meta_memory_info={meta_memory_info}")
        return meta_memory_info

    async def build_messages(self) -> list[Message]:
        if self.context.get("query"):
            user_query = self.context.query
        elif self.context.get("messages"):
            user_query = format_messages(self.context.messages)
        else:
            raise ValueError("Input must have either `query` or `messages`")

        messages = [
            Message(
                role=Role.SYSTEM,
                content=self.prompt_format(
                    prompt_name="system_prompt",
                    meta_memory_info=await self._read_meta_memories(),
                    user_query=user_query,
                ),
            ),
            Message(
                role=Role.USER,
                content=self.get_prompt("user_message"),
            ),
        ]

        return messages

    async def _acting_step(self, assistant_message: Message, step: int, **kwargs) -> list[Message]:
        return await super()._acting_step(
            assistant_message,
            step,
            query=self.context.get("query", ""),
            messages=self.context.get("messages", []),
            **kwargs,
        )
