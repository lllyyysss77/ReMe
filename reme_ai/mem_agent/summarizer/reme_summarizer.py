"""Orchestrator for complete memory summarization workflow across all memory types."""

from loguru import logger

from ..base_memory_agent import BaseMemoryAgent
from ...core_old.context import C
from ...core_old.enumeration import Role, MemoryType
from ...core_old.schema import Message, MemoryNode, ToolCall
from ...core_old.utils import get_now_time, format_messages


@C.register_op()
class ReMeSummarizer(BaseMemoryAgent):
    """Coordinates memory updates by delegating to specialized memory agents."""

    def __init__(self, meta_memories: list[dict] | None = None, enable_identity_memory: bool = False, **kwargs):
        """Initialize with flags to enable/disable identity memory processing."""
        super().__init__(**kwargs)
        self.enable_identity_memory = enable_identity_memory
        self.meta_memories: list[dict] = meta_memories or []

        # Check if AddMetaMemory is in tools
        self.enable_add_meta_memory = self._check_add_meta_memory_in_tools()

    def _check_add_meta_memory_in_tools(self) -> bool:
        """Check if AddMetaMemory tool is present in the tools list."""
        from ...mem_tool import AddMetaMemory

        for tool in self.tools:
            if isinstance(tool, AddMetaMemory):
                return True
        return False

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": self.prompt_format("tool", enable_add_meta_memory=self.enable_add_meta_memory),
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

    async def _read_identity_memory(self) -> str:
        """Retrieve agent's self-perception memory."""
        if self.enable_identity_memory:
            from ...mem_tool import ReadIdentityMemory

            op = ReadIdentityMemory()
            await op.call()
            return op.output
        else:
            return ""

    async def _read_meta_memories(self) -> str:
        """Fetch all meta-memory entries that define specialized memory agents."""
        from ...mem_tool import ReadMetaMemory

        op = ReadMetaMemory(enable_identity_memory=self.enable_identity_memory)
        if self.meta_memories:
            return op.format_memory_metadata(self.meta_memories)
        else:
            await op.call()
            return str(op.output)

    async def build_messages(self) -> list[Message]:
        """Construct initial messages with context, identity, and meta-memory information."""
        messages = [Message(**m) if isinstance(m, dict) else m for m in self.context.messages]
        self.context["messages_formated"] = self.description + "\n" + format_messages(messages)
        self.context["ref_memory_id"] = MemoryNode(
            memory_type=MemoryType.HISTORY,
            content=self.context["messages_formated"],
        ).memory_id

        now_time = get_now_time()
        identity_memory = await self._read_identity_memory()
        meta_memory_info = await self._read_meta_memories()
        logger.info(f"now_time={now_time} identity_memory={identity_memory} meta_memory_info={meta_memory_info}")

        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            now_time=now_time,
            identity_memory=identity_memory,
            meta_memory_info=meta_memory_info,
            context=self.context["messages_formated"],
            enable_add_meta_memory=self.enable_add_meta_memory,
        )

        user_message = self.get_prompt("user_message")
        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=user_message),
        ]

        return messages

    async def _reasoning_step(self, messages: list[Message], step: int, **kwargs) -> tuple[Message, bool]:
        """Refresh meta-memory info in system prompt before each reasoning step."""
        system_messages = [message for message in messages if message.role is Role.SYSTEM]

        if system_messages:
            system_message = system_messages[0]
            system_message.content = self.prompt_format(
                prompt_name="system_prompt",
                now_time=get_now_time(),
                identity_memory=await self._read_identity_memory(),
                meta_memory_info=await self._read_meta_memories(),
                context=self.context["messages_formated"],
                enable_add_meta_memory=self.enable_add_meta_memory,
            )

        return await super()._reasoning_step(messages, step, **kwargs)

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
