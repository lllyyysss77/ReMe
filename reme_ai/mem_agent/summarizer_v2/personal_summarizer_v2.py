"""Simplified personal memory summarizer using v2 memory tools."""

from ..base_memory_agent import BaseMemoryAgent
from ...core.context import C
from ...core.enumeration import Role, MemoryType
from ...core.schema import Message, ToolCall
from ...core.utils import format_messages


@C.register_op()
class PersonalSummarizerV2(BaseMemoryAgent):
    memory_type: MemoryType = MemoryType.PERSONAL

    """Simplified personal memory summarizer that uses v2 memory tools.
    
    This summarizer follows a three-step workflow:
    1. AddMemoryDrafts: Generate initial memory drafts from context
    2. RetrieveRecentAndSimilarMemories: Retrieve similar and recent memories
    3. UpdateMemories: Delete outdated memories and add new ones
    """

    def _build_tool_call(self) -> ToolCall:
        """Build tool call schema for the agent."""
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

        # # Check if AddMemoryDrafts tool was executed
        # exist_memory_drafts = False
        # if assistant_message.tool_calls:
        #     for tool_call in assistant_message.tool_calls:
        #         if tool_call.name == "add_memory_drafts":
        #             exist_memory_drafts = True
        #             break
        #
        # # If memory drafts were added, regenerate system prompt with simplified context
        # if exist_memory_drafts:
        #     simplified_context = "The conversation context has been summarized in memory drafts."
        #     new_system_prompt = self.prompt_format(
        #         prompt_name="system_prompt",
        #         context=simplified_context,
        #         memory_type=self.memory_type.value,
        #         memory_target=self.memory_target,
        #     )
        #
        #     # Update the system message in the message history
        #     for i, msg in enumerate(self.messages):
        #         if msg.role == Role.SYSTEM:
        #             self.messages[i] = Message(role=Role.SYSTEM, content=new_system_prompt)
        #             break

        return messages
