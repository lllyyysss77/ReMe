"""FsCli system prompt"""

from datetime import datetime
from pathlib import Path

from loguru import logger

from ...core.enumeration import Role, ChunkEnum
from ...core.op import BaseReactStream
from ...core.schema import Message, StreamChunk
from ...tool.fs import BashTool, LsTool, ReadTool, WriteTool, EditTool


class FsCli(BaseReactStream):
    """FsCli agent with system prompt."""

    def __init__(
        self,
        working_dir: str,
        context_window_tokens: int = 128000,
        reserve_tokens: int = 36000,
        keep_recent_tokens: int = 20000,
        hybrid_enabled: bool = True,
        hybrid_vector_weight: float = 0.7,
        hybrid_text_weight: float = 0.3,
        hybrid_candidate_multiplier: float = 3.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.working_dir: str = working_dir
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        self.context_window_tokens: int = context_window_tokens
        self.reserve_tokens: int = reserve_tokens
        self.keep_recent_tokens: int = keep_recent_tokens
        self.hybrid_enabled: bool = hybrid_enabled
        self.hybrid_vector_weight: float = hybrid_vector_weight
        self.hybrid_text_weight: float = hybrid_text_weight
        self.hybrid_candidate_multiplier: float = hybrid_candidate_multiplier

        self.messages: list[Message] = []
        self.previous_summary: str = ""

    async def reset(self) -> str:
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
        summarizer = FsSummarizer(
            tools=[
                BashTool(cwd=self.working_dir),
                LsTool(cwd=self.working_dir),
                ReadTool(cwd=self.working_dir),
                WriteTool(cwd=self.working_dir),
                EditTool(cwd=self.working_dir),
            ],
            working_dir=self.working_dir,
            language=self.language,
        )

        result = await summarizer.call(
            messages=self.messages,
            date=current_date,
            service_context=self.service_context,
        )
        self.messages.clear()
        self.previous_summary = ""
        return f"History saved to memory files and reset. Result: {result.get('answer', 'Done')}"

    async def context_check(self) -> dict:
        """Check if messages exceed token limits."""
        # Import required modules
        from ..fs import FsContextChecker

        # Step 1: Check and find cut point
        checker = FsContextChecker(
            context_window_tokens=self.context_window_tokens,
            reserve_tokens=self.reserve_tokens,
            keep_recent_tokens=self.keep_recent_tokens,
        )
        return await checker.call(messages=self.messages, service_context=self.service_context)

    async def compact(self, force_compact: bool = False) -> str:
        """Compact history then reset.

        First compacts messages if they exceed token limits (generating a summary),
        then calls reset_history to save to files and clear.

        Args:
            force_compact: If True, force compaction of all messages into summary

        Returns:
            str: Summary of compaction result
        """
        if not self.messages:
            return "No history to compact."

        # Import required modules
        from ..fs import FsCompactor

        # Step 1: Check and find cut point
        cut_result = await self.context_check()
        tokens_before = cut_result.get("token_count", 0)

        if force_compact:
            # Force compact: summarize all messages, leave only summary
            messages_to_summarize = self.messages
            turn_prefix_messages = []
            left_messages = []
        elif not cut_result.get("needs_compaction", False):
            # No compaction needed
            return "History is within token limits, no compaction needed."
        else:
            # Normal compaction: use cut point result
            messages_to_summarize = cut_result.get("messages_to_summarize", [])
            turn_prefix_messages = cut_result.get("turn_prefix_messages", [])
            left_messages = cut_result.get("left_messages", [])

        # Step 2: Generate summary via Compactor
        compactor = FsCompactor(language=self.language)
        summary_content = await compactor.call(
            messages_to_summarize=messages_to_summarize,
            turn_prefix_messages=turn_prefix_messages,
            previous_summary=self.previous_summary,
            service_context=self.service_context,
        )

        # Step 3: Call reset_history to save and clear
        reset_result = await self.reset()

        # Step 4: Assemble final messages
        self.messages = left_messages
        self.previous_summary = summary_content

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
        _ = await self.compact(force_compact=False)

        messages = await self.build_messages()
        for i, message in enumerate(messages):
            role = message.name or message.role
            logger.info(f"[{self.__class__.__name__}] role={role} {message.simple_dump(as_dict=False)}")

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
