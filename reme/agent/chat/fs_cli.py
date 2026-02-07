"""FsCli system prompt"""

from datetime import datetime

from ...core.enumeration import Role, ChunkEnum
from ...core.op import BaseReactStream
from ...core.schema import Message, StreamChunk


class FsCli(BaseReactStream):
    """FsCli agent with system prompt."""

    def __init__(
        self,
        working_dir: str,
        summary_params: dict | None = None,
        compact_params: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.working_dir: str = working_dir
        self.summary_params: dict = summary_params or {}
        self.compact_params: dict = compact_params or {}

        self.messages: list[Message] = []
        self.previous_summary: str = ""

    async def reset_history(self) -> str:
        """Reset conversation history using summary.

        Summarizes current messages to memory files and clears history.
        """
        if not self.messages:
            self.messages.clear()
            self.previous_summary = ""
            return "No history to reset."

        # Import required modules
        from ..fs import FsSummarizer

        # Summarize current conversation and save to memory files
        current_date = datetime.now().strftime("%Y-%m-%d")
        summarizer = FsSummarizer(tools=self.tools, **(self.summary_params or {}))

        result = await summarizer.call(
            messages=self.messages,
            date=current_date,
            service_context=self.service_context,
        )

        # Clear messages (no previous_summary update, as summarizer saves to files)
        self.messages.clear()
        self.previous_summary = ""

        return f"History saved to memory files and reset. Result: {result.get('answer', 'Done')}"

    async def compact_history(self) -> str:
        """Compact history then reset.

        First compacts messages if they exceed token limits (generating a summary),
        then calls reset_history to save to files and clear.
        """
        if not self.messages:
            return "No history to compact."

        # Import required modules
        from ..fs import FsCompactor

        # Step 1: Compact messages
        compactor = FsCompactor(**(self.compact_params or {}))
        compact_result = await compactor.call(
            messages=self.messages,
            previous_summary=self.previous_summary,
            service_context=self.service_context,
        )

        compacted_messages = compact_result.get("messages", self.messages)
        is_compacted = compact_result.get("compacted", False)

        if not is_compacted:
            return "History is within token limits, no compaction needed."

        # Step 2: Extract summary from compacted messages
        # The first message contains the summary wrapped in compaction_summary_format
        tokens_before = compact_result.get("tokens_before", 0)

        if compacted_messages and compacted_messages[0].role == Role.USER:
            # Extract summary content from the first message
            summary_content = compacted_messages[0].content
            self.previous_summary = summary_content

        # Step 3: Update messages and call reset_history to save and clear
        self.messages = compacted_messages
        reset_result = await self.reset_history()

        return f"History compacted from {tokens_before} tokens. {reset_result}"

    async def build_messages(self) -> list[Message]:
        """Build system prompt message."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")

        system_prompt = self.prompt_format(
            "system_prompt",
            workspace_dir=self.working_dir,
            current_time=current_time,
            has_previous_summary=bool(self.previous_summary),
            previous_summary=self.previous_summary or "",
        )

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
