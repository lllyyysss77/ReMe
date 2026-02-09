"""Context compaction agent for long sessions."""

from loguru import logger

from ...core.enumeration import Role
from ...core.op import BaseOp
from ...core.schema import Message
from ...core.utils import format_messages


class FsCompactor(BaseOp):
    """Generate summaries for conversation history compaction."""

    @staticmethod
    def _normalize_messages(messages: list[Message | dict]) -> list[Message]:
        """Convert dict messages to Message objects."""
        return [Message(**m) if isinstance(m, dict) else m for m in messages]

    async def _generate_summary(self, prompt_messages: list[Message]) -> str:
        """Generate summary via LLM. Returns empty string if no messages."""
        assistant_message = await self.llm.chat(prompt_messages)
        return assistant_message.content

    @staticmethod
    def _serialize_conversation(messages: list[Message]) -> str:
        """Serialize conversation messages to text format."""
        return format_messages(
            messages=messages,
            add_index=False,
            add_time=False,
            use_name=True,
            add_reasoning=False,
            add_tools=True,
            strip_markdown_headers=False,
        )

    def _build_history_prompt(self, messages_to_summarize: list[Message], previous_summary: str = "") -> list[Message]:
        """Build prompt for main history summary."""
        if not messages_to_summarize:
            return []

        system_prompt = self.get_prompt("system_prompt")
        if previous_summary:
            user_prompt = self.prompt_format("update_user_message", previous_summary=previous_summary)
        else:
            user_prompt = self.get_prompt("initial_user_message")
        conversation_text = self._serialize_conversation(messages_to_summarize)

        return [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=f"<conversation>\n{conversation_text}\n</conversation>\n\n{user_prompt}"),
        ]

    def _build_turn_prefix_prompt(self, turn_prefix_messages: list[Message]) -> list[Message]:
        """Build prompt for turn prefix summary (split turn only)."""
        if not turn_prefix_messages:
            return []

        system_prompt = self.get_prompt("system_prompt")
        conversation_text = self._serialize_conversation(turn_prefix_messages)
        turn_prefix_prompt = self.prompt_format("turn_prefix_summarization", conversation_text=conversation_text)

        return [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=turn_prefix_prompt),
        ]

    async def execute(self) -> str:
        """
        Generate summary for conversation history.

        Expects context to have:
            - messages_to_summarize: list[Message] (required)
            - turn_prefix_messages: list[Message] (optional, for split turn)
            - previous_summary: str (optional, for incremental summarization)

        Returns:
            str: Generated summary text formatted with compaction_summary_format.
                 Returns empty string if no messages to summarize.
        """
        messages_to_summarize = self.context.get("messages_to_summarize", [])
        turn_prefix_messages = self.context.get("turn_prefix_messages", [])
        previous_summary = self.context.get("previous_summary", "")

        messages_to_summarize = self._normalize_messages(messages_to_summarize)
        if messages_to_summarize:
            history_prompt_messages = self._build_history_prompt(messages_to_summarize, previous_summary)
            history_summary = "**History Summary**:\n\n" + await self._generate_summary(history_prompt_messages)
        else:
            history_summary = ""

        turn_prefix_messages = self._normalize_messages(turn_prefix_messages)
        if turn_prefix_messages:
            turn_prefix_prompt_messages = self._build_turn_prefix_prompt(turn_prefix_messages)
            turn_prefix_summary = "**Turn Context**:\n\n" + await self._generate_summary(turn_prefix_prompt_messages)
        else:
            turn_prefix_summary = ""

        summary = "\n\n---".join([history_summary, turn_prefix_summary])
        logger.info(f"Generated summary: {summary}")
        return summary
