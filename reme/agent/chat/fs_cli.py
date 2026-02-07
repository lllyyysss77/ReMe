"""FsCli system prompt"""

from datetime import datetime

from ...core.enumeration import Role, ChunkEnum
from ...core.op import BaseReactStream
from ...core.schema import Message, StreamChunk


class FsCli(BaseReactStream):
    """FsCli agent with system prompt."""

    def __init__(self, working_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.working_dir: str = working_dir
        self.messages: list[Message] = []

    def reset_history(self):
        """Reset conversation history."""
        self.messages.clear()
        return self

    async def build_messages(self) -> list[Message]:
        """Build system prompt message."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")
        system_prompt = self.prompt_format("system_prompt", workspace_dir=self.working_dir, current_time=current_time)

        return [
            Message(role=Role.SYSTEM, content=system_prompt),
            *self.messages,
            Message(role=Role.USER, content=self.context.query),
        ]

    async def execute(self):
        """Execute the agent."""
        messages = await self.build_messages()

        t_tools, messages, success = await self.react(messages, self.tools)

        # Update self.messages: react() returns [SYSTEM, ...history...],
        # so we remove the first SYSTEM message
        self.messages = messages[1:]

        # Emit final done signal
        await self.context.add_stream_chunk(
            StreamChunk(
                chunk_type=ChunkEnum.DONE,
                chunk="",
                metadata={
                    "success": success,
                    "total_steps": len(t_tools),
                },
            ),
        )

        return {
            "answer": messages[-1].content if success else "",
            "success": success,
            "messages": messages,
            "tools": t_tools,
        }
