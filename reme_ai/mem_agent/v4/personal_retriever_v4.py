from ..base_memory_agent import BaseMemoryAgent
from ...core.enumeration import Role, MemoryType
from ...core.schema import Message
from ...core.utils import format_messages
from ...mem_tool.v4 import ReadUserProfile


class PersonalRetrieverV4(BaseMemoryAgent):
    memory_type: MemoryType = MemoryType.PERSONAL

    async def build_messages(self) -> list[Message]:
        if self.context.get("query"):
            context = self.context.query
        elif self.context.get("messages"):
            context = format_messages(self.context.messages)
        else:
            raise ValueError("input must have either `query` or `messages`")

        read_profile_tool = ReadUserProfile()
        await read_profile_tool.call(memory_type=self.memory_type.value, memory_target=self.memory_target)

        messages = [
            Message(
                role=Role.SYSTEM,
                content=self.prompt_format(
                    prompt_name="system_prompt",
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                    user_profile=read_profile_tool.output,
                    context=context,
                )),
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
            memory_type=self.memory_type.value,
            memory_target=self.memory_target,
            **kwargs,
        )
