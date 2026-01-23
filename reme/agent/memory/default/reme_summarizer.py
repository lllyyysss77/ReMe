from loguru import logger

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import Role
from ....core.op import BaseTool
from ....core.schema import Message


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
        from ....tool.memory import AddHistory

        add_history_tool = AddHistory()
        await add_history_tool.call()
        self.context.history_node = add_history_tool.context.add_history

        messages = [
            Message(
                role=Role.SYSTEM,
                content=self.prompt_format(
                    prompt_name="system_prompt",
                    meta_memory_info=await self._read_meta_memories(),
                    context=self.context.history_node.content,
                ),
            ),
            Message(
                role=Role.USER,
                content=self.get_prompt("user_message"),
            ),
        ]

        return messages

    async def _acting_step(
        self,
        assistant_message: Message,
        step: int,
        stage: str = "",
        **kwargs,
    ) -> tuple[list[BaseTool], list[Message]]:
        return await super()._acting_step(
            assistant_message,
            step,
            description=self.description,
            messages=self.context.messages,
            history_node=self.context.history_node,
            author=self.author,
            **kwargs,
        )
