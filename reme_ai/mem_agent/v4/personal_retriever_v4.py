from ..base_memory_agent import BaseMemoryAgent
from ...core_old.enumeration import Role, MemoryType
from ...core_old.schema import Message
from ...core_old.utils import format_messages
from ...mem_tool.v4 import ReadUserProfile


class PersonalRetrieverV4(BaseMemoryAgent):
    memory_type: MemoryType = MemoryType.PERSONAL

    async def build_messages(self) -> list[Message]:
        context = self.context.query if self.context.get("query") else format_messages(self.context.messages) if self.context.get("messages") else None
        if not context:
            raise ValueError("input must have either `query` or `messages`")

        read_profile_tool = ReadUserProfile(show_ids="history")
        await read_profile_tool.call(memory_type=self.memory_type.value, memory_target=self.memory_target)
        self.context.user_profile = user_profile = read_profile_tool.output

        return [
            Message(
                role=Role.USER,
                content=self.prompt_format(
                    prompt_name="user_message",
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                    user_profile=user_profile,
                    context=context,
                ))
        ]

    async def _acting_step(self, assistant_message: Message, step: int, **kwargs) -> list[Message]:
        return await super()._acting_step(
            assistant_message,
            step,
            memory_type=self.memory_type.value,
            memory_target=self.memory_target,
            **kwargs,
        )

    async def execute(self):
        """Execute the retriever and determine success based on output markers."""
        await super().execute()

        # Check for memory found/not found markers in the output
        if self.output:
            if "<MEMORY_FOUND>" in self.output:
                self.success = True
            elif "<MEMORY_NOT_FOUND>" in self.output:
                self.success = False

        self.meta_info = self.context.user_profile + "\n" + self.meta_info