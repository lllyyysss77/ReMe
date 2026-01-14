from ..base_memory_agent import BaseMemoryAgent
from ...core.enumeration import Role, MemoryType
from ...core.schema import Message, ToolCall
from ...core.utils import format_messages


class PersonalSummarizerWk(BaseMemoryAgent):
    memory_type: MemoryType = MemoryType.PERSONAL

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": self.get_prompt("tool"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {
                                        "type": "string",
                                        "description": "role",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "content",
                                    },
                                },
                                "required": ["role", "content"],
                            },
                        },
                    },
                    "required": ["messages"],
                },
            },
        )

    async def build_messages(self) -> list[Message]:
        """Construct messages with context, memory_target, and memory_type information."""
        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            context=self.description + "\n" + format_messages(self.get_messages()),
            memory_type=self.memory_type.value,
            memory_target=self.memory_target,
        )

        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=self.get_prompt("user_message")),
        ]
        return messages

    async def _reasoning_step(self, messages: list[Message], step: int, **kwargs) -> tuple[Message, bool]:
        return await super()._reasoning_step(messages, step, **kwargs)

    async def _acting_step(self, assistant_message: Message, step: int, **kwargs) -> list[Message]:
        """Execute tool calls with memory_target, memory_type, and author context."""
        messages: list[Message] = await super()._acting_step(
            assistant_message,
            step,
            memory_type=self.memory_type.value,
            memory_target=self.memory_target,
            ref_memory_id=self.ref_memory_id,
            author=self.author,
            **kwargs,
        )
        return messages
