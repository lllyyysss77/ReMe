"""Personal memory retriever agent for retrieving personal memories through vector search."""

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import Role, MemoryType
from ....core.op import BaseTool
from ....core.schema import Message
from ....core.utils import format_messages


class PersonalRetriever(BaseMemoryAgent):
    """Retrieve personal memories through vector search and history reading."""

    memory_type: MemoryType = MemoryType.PERSONAL

    async def build_messages(self) -> list[Message]:
        if self.context.get("query"):
            context = self.context.query
        elif self.context.get("messages"):
            context = self.description + "\n" + format_messages(self.context.messages)
        else:
            raise ValueError("input must have either `query` or `messages`")

        return [
            Message(
                role=Role.SYSTEM,
                content=self.prompt_format(
                    prompt_name="system_prompt",
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                    user_profile=await self.read_user_profile(show_id="history"),
                    context=context.strip(),
                ),
            ),
            Message(
                role=Role.USER,
                content=self.get_prompt("user_message"),
            ),
        ]

    async def _acting_step(
        self,
        assistant_message: Message,
        tools: list[BaseTool],
        step: int,
        stage: str = "",
        **kwargs,
    ) -> tuple[list[BaseTool], list[Message]]:
        """Execute tool calls with memory context."""
        return await super()._acting_step(
            assistant_message,
            tools,
            step,
            memory_type=self.memory_type.value,
            memory_target=self.memory_target,
            **kwargs,
        )
