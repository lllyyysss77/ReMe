from loguru import logger

from ..base_memory_agent import BaseMemoryAgent
from ...core_old.enumeration import Role, MemoryType
from ...core_old.schema import Message, MemoryNode, ToolCall
from ...core_old.utils import format_messages


class ReMeSummarizerWk(BaseMemoryAgent):

    def __init__(self, meta_memories: list[dict] | None = None, **kwargs):
        """Initialize with meta memories list."""
        super().__init__(**kwargs)
        self.meta_memories: list[dict] = meta_memories or []

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

    async def _read_meta_memories(self) -> str:
        from ...mem_tool import ReadMetaMemory

        return ReadMetaMemory().format_memory_metadata(self.meta_memories)

    async def build_messages(self) -> list[Message]:
        """Construct initial messages with context and meta-memory information."""
        messages = [Message(**m) if isinstance(m, dict) else m for m in self.context.messages]
        self.context["messages_formated"] = self.description + "\n" + format_messages(messages)
        self.context["ref_memory_id"] = MemoryNode(
            memory_type=MemoryType.HISTORY,
            content=self.context["messages_formated"],
        ).memory_id

        meta_memory_info = await self._read_meta_memories()
        logger.info(f"meta_memory_info={meta_memory_info}")

        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            meta_memory_info=meta_memory_info,
            context=self.context["messages_formated"],
        )

        user_message = self.get_prompt("user_message")
        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=user_message),
        ]

        return messages

    async def _acting_step(self, assistant_message: Message, step: int, **kwargs) -> list[Message]:
        """Execute tool calls with ref_memory_id and author context."""
        return await super()._acting_step(
            assistant_message,
            step,
            messages=self.context.get("messages", []),
            description=self.context.get("description"),
            ref_memory_id=self.context["ref_memory_id"],
            messages_formated=self.context["messages_formated"],
            author=self.author,
            **kwargs,
        )
