from loguru import logger

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import Role, MemoryType
from ....core.schema import Message, MemoryNode
from ....core.utils import format_messages


class ReMeSummarizer(BaseMemoryAgent):

    def __init__(self, meta_memories: list[dict] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.meta_memories: list[dict] = meta_memories or []

    async def _read_meta_memories(self) -> str:
        from ....tool.memory import ReadMetaMemory
        meta_memory_info = ReadMetaMemory().format_memory_metadata(self.meta_memories)
        logger.info(f"meta_memory_info={meta_memory_info}")
        return meta_memory_info

    async def build_messages(self) -> list[Message]:
        self.context.messages = [Message(**m) if isinstance(m, dict) else m for m in self.context.messages]
        history_content = self.description + "\n" + format_messages(self.context.messages)
        self.context.history_node = history_node = MemoryNode(
            memory_type=MemoryType.HISTORY,
            memory_target="",
            when_to_use=history_content[:100],
            content=history_content,
            ref_memory_id="",
            author=self.author,
            metadata={},
        )

        logger.info(f"Adding summary node: {history_node.model_dump_json(indent=2, exclude_none=True)}")
        await self.vector_store.delete(history_node.memory_id)
        await self.vector_store.insert([history_node.to_vector_node()])

        messages = [
            Message(
                role=Role.SYSTEM,
                content=self.prompt_format(
                    prompt_name="system_prompt",
                    meta_memory_info=await self._read_meta_memories(),
                    context=history_node.content,
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
            messages=self.context.messages,
            history_node=self.context.history_node,
            author=self.author,
            **kwargs,
        )
