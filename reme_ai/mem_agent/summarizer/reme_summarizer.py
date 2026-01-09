"""Orchestrator for complete memory summarization workflow across all memory types."""

import re
from typing import List

from loguru import logger

from ..base_memory_agent import BaseMemoryAgent
from ...core.context import C
from ...core.enumeration import Role
from ...core.schema import Message, MemoryNode, ToolCall
from ...core.utils import get_now_time, format_messages


@C.register_op()
class ReMeSummarizer(BaseMemoryAgent):
    """Coordinates memory updates by delegating to specialized memory agents."""

    def __init__(self, meta_memories: list[dict] | None = None, enable_identity_memory: bool = False, **kwargs):
        """Initialize with flags to enable/disable identity memory processing."""
        super().__init__(**kwargs)
        self.enable_identity_memory = enable_identity_memory
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

    async def _add_history_memory(self) -> MemoryNode:
        """Store conversation history and return the memory node."""
        from ...mem_tool import AddHistoryMemory

        op = AddHistoryMemory()
        await op.call(messages=self.get_messages())
        return op.memory_nodes[0]

    @staticmethod
    async def _read_identity_memory() -> str:
        """Retrieve agent's self-perception memory."""
        from ...mem_tool import ReadIdentityMemory

        op = ReadIdentityMemory()
        await op.call()
        return op.output

    async def _read_meta_memories(self) -> str:
        """Fetch all meta-memory entries that define specialized memory agents."""
        from ...mem_tool import ReadMetaMemory

        op = ReadMetaMemory(enable_identity_memory=self.enable_identity_memory)
        if self.meta_memories:
            return op.format_memory_metadata(self.meta_memories)
        else:
            await op.call()
            return str(op.output)

    async def build_messages(self) -> List[Message]:
        """Construct initial messages with context, identity, and meta-memory information."""
        memory_node: MemoryNode = await self._add_history_memory()
        self.context["ref_memory_id"] = memory_node.memory_id

        now_time = get_now_time()
        identity_memory = await self._read_identity_memory()
        meta_memory_info = await self._read_meta_memories()
        context = self.description + "\n" + format_messages(self.get_messages())
        logger.info(
            f"now_time={now_time} "
            f"memory_node={memory_node.content[:100]}... "
            f"identity_memory={identity_memory} "
            f"meta_memory_info={meta_memory_info} "
            f"context={context[:100]}",
        )

        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            now_time=now_time,
            identity_memory=identity_memory,
            meta_memory_info=meta_memory_info,
            context=context,
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
            now_time = get_now_time()
            identity_memory = await self._read_identity_memory()
            meta_memory_info = await self._read_meta_memories()
            context = self.description + "\n" + format_messages(self.get_messages())
            system_message.content = self.prompt_format(
                prompt_name="system_prompt",
                now_time=now_time,
                identity_memory=identity_memory,
                meta_memory_info=meta_memory_info,
                context=context,
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
            author=self.author,
            **kwargs,
        )
