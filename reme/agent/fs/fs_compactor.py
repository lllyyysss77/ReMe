"""Context compaction agent for long sessions."""

from loguru import logger

from ...core.enumeration import Role, MemoryType
from ...core.op import BaseReact
from ...core.schema import CutPointResult, Message


class FsCompactor(BaseReact):
    """Compact long conversation history into structured summaries."""

    memory_type: MemoryType = MemoryType.PERSONAL

    def __init__(
        self,
        context_window_tokens: int = 128000,
        reserve_tokens: int = 36000,
        keep_recent_tokens: int = 20000,
        **kwargs,
    ):
        super().__init__(tools=[], **kwargs)
        self.context_window_tokens: int = context_window_tokens
        self.reserve_tokens: int = reserve_tokens
        self.keep_recent_tokens: int = keep_recent_tokens

    @staticmethod
    def _normalize_messages(messages: list[Message | dict]) -> list[Message]:
        """Convert dict messages to Message objects."""
        return [Message(**m) if isinstance(m, dict) else m for m in messages]

    @staticmethod
    def _is_user_message(message: Message) -> bool:
        """Check if message is user role."""
        return message.role is Role.USER

    def _find_turn_start_index(self, messages: list[Message], entry_index: int) -> int:
        """Find user message that starts the turn. Returns -1 if not found."""
        if not messages or entry_index < 0 or entry_index >= len(messages):
            return -1

        for i in range(entry_index, -1, -1):
            if self._is_user_message(messages[i]):
                return i
        return -1

    def _find_cut_point(self, messages: list[Message]) -> CutPointResult:
        """
        Find cut point with split turn detection.

        Split turn: User → Assistant → [CUT] → Assistant → User
        Clean cut: User → [CUT] → Assistant → User
        """
        if not messages:
            return CutPointResult()

        accumulated_tokens = 0
        cut_index = 0

        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            msg_tokens = self.token_counter.count_token([msg])
            accumulated_tokens += msg_tokens

            if accumulated_tokens >= self.keep_recent_tokens:
                cut_index = i
                logger.debug(f"Cut point at index {cut_index}, {accumulated_tokens} tokens")
                break

        if cut_index == 0:
            return CutPointResult(left_messages=messages)

        cut_message = messages[cut_index]
        is_user_cut = self._is_user_message(cut_message)

        if is_user_cut:
            return CutPointResult(
                messages_to_summarize=messages[:cut_index],
                left_messages=messages[cut_index:],
                cut_index=cut_index,
            )

        turn_start_index = self._find_turn_start_index(messages, cut_index)

        if turn_start_index == -1:
            logger.warning("Split turn detected but no turn start found, treating as clean cut")
            return CutPointResult(
                messages_to_summarize=messages[:cut_index],
                left_messages=messages[cut_index:],
                cut_index=cut_index,
            )

        return CutPointResult(
            messages_to_summarize=messages[:turn_start_index],
            turn_prefix_messages=messages[turn_start_index:cut_index],
            left_messages=messages[cut_index:],
            is_split_turn=True,
            cut_index=cut_index,
        )

    async def _generate_summary(self, prompt_messages: list[Message]) -> str:
        """Generate summary via LLM. Returns empty string if no messages."""
        if not prompt_messages:
            return ""

        try:
            assistant_message = await self.llm.chat(prompt_messages)
            return assistant_message.content if assistant_message.content else ""
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            raise RuntimeError(f"Summarization failed: {e}") from e

    @staticmethod
    def _serialize_conversation(messages: list[Message]) -> str:
        """Serialize conversation messages to text format."""
        lines = []
        for msg in messages:
            role = msg.name if msg.name else msg.role.value
            content = msg.content
            if isinstance(content, str):
                lines.append(f"[{role}]")
                lines.append(content)
                lines.append("")
            elif isinstance(content, list):
                lines.append(f"[{role}]")
                for block in content:
                    lines.append(block.model_dump_json())
                lines.append("")

        return "\n".join(lines)

    def build_messages_s1(self) -> list[Message]:
        """Build prompt for main history summary."""
        messages = self._normalize_messages(self.context.messages)
        cut_result = self._find_cut_point(messages)

        self.context.is_split_turn = cut_result.is_split_turn
        self.context.turn_prefix_messages = cut_result.turn_prefix_messages
        self.context.left_messages = cut_result.left_messages

        if not cut_result.messages_to_summarize:
            logger.info("No messages to summarize")
            return []

        system_prompt = self.get_prompt("system_prompt")
        if self.context.get("previous_summary", ""):
            user_prompt = self.prompt_format("update_user_message", previous_summary=self.context.previous_summary)
        else:
            user_prompt = self.get_prompt("initial_user_message")
        conversation_text = self._serialize_conversation(cut_result.messages_to_summarize)

        return [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=f"<conversation>\n{conversation_text}\n</conversation>\n\n{user_prompt}"),
        ]

    def build_messages_s2(self) -> list[Message]:
        """Build prompt for turn prefix summary (split turn only)."""
        if not self.context.turn_prefix_messages:
            return []

        system_prompt = self.get_prompt("system_prompt")
        conversation_text = self._serialize_conversation(self.context.turn_prefix_messages)
        turn_prefix_prompt = self.prompt_format("turn_prefix_summarization", conversation_text=conversation_text)

        return [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=turn_prefix_prompt),
        ]

    async def execute(self):
        """
        Execute compaction if needed.

        Returns: [summary_message, ...left_messages] if compacted, else original messages.
        """
        original_messages = self._normalize_messages(self.context.messages)
        token_count: int = self.token_counter.count_token(original_messages)
        threshold = self.context_window_tokens - self.reserve_tokens

        if token_count < threshold:
            logger.info(f"Token count {token_count} below threshold ({threshold}), skipping compaction")
            return {
                "compacted": False,
                "tokens_before": token_count,
                "is_split_turn": False,
                "messages": original_messages,
            }

        logger.info(f"Starting compaction, token count: {token_count}, threshold: {threshold}")

        history_prompt_messages = self.build_messages_s1()

        if not history_prompt_messages and not self.context.get("is_split_turn"):
            logger.warning("No messages to summarize and not a split turn, returning original messages")
            return {
                "compacted": False,
                "tokens_before": token_count,
                "is_split_turn": False,
                "messages": original_messages,
            }

        history_summary = await self._generate_summary(history_prompt_messages) if history_prompt_messages else ""

        if self.context.is_split_turn and self.context.turn_prefix_messages:
            logger.info("Split turn detected, generating turn prefix summary")
            turn_prefix_prompt_messages = self.build_messages_s2()
            turn_prefix_summary = await self._generate_summary(turn_prefix_prompt_messages)
            summary = f"{history_summary}\n\n---\n\n**Turn Context (split turn):**\n\n{turn_prefix_summary}"
        else:
            summary = history_summary

        logger.info(f"Compaction complete, summary length: {len(summary)}, split_turn: {self.context.is_split_turn}")

        summary_content = self.prompt_format("compaction_summary_format", summary=summary)
        summary_message = Message(role=Role.USER, content=summary_content)
        left_messages = self.context.get("left_messages", [])
        final_messages = [summary_message] + left_messages

        return {
            "compacted": True,
            "tokens_before": token_count,
            "is_split_turn": self.context.is_split_turn,
            "messages": final_messages,
        }
