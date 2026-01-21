"""Specialized agent for extracting and managing personal memories about specific individuals."""

from ..base_memory_agent import BaseMemoryAgent
from ...core.context import C
from ...core.enumeration import Role, MemoryType
from ...core.schema import Message, ToolCall
from ...core.utils import get_now_time, format_messages


@C.register_op()
class PersonalSummarizer(BaseMemoryAgent):
    """Extracts and stores personal information about individuals from conversations."""

    def __init__(self, recent_top_k: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.recent_top_k: int = recent_top_k

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

    async def _retrieve_recent_memories(self) -> str:
        """Retrieve recent memories sorted by time_modified."""
        from ...mem_tool import RetrieveRecentMemory

        op = RetrieveRecentMemory(top_k=self.recent_top_k)
        await op.call(memory_type="personal", memory_target=self.memory_target, retrieved_nodes=self.retrieved_nodes)
        return op.output

    async def build_messages(self) -> list[Message]:
        """Construct messages with context, memory_target, and memory_type information."""
        await self._retrieve_recent_memories()

        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            now_time=get_now_time(),
            recent_memories="\n".join([n.format_memory() for n in self.retrieved_nodes]),
            context=self.description + "\n" + format_messages(self.get_messages()),
            memory_type=self.memory_type.value,
            memory_target=self.memory_target,
        )

        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=self.get_prompt("user_message")),
        ]
        return messages

    async def _acting_step(self, assistant_message: Message, step: int, **kwargs) -> list[Message]:
        """Execute tool calls with memory_target, memory_type, and author context."""
        return await super()._acting_step(
            assistant_message,
            step,
            memory_type=self.memory_type.value,
            memory_target=self.memory_target,
            ref_memory_id=self.ref_memory_id,
            author=self.author,
            **kwargs,
        )
