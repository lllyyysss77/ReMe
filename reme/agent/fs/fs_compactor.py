"""Context compaction agent for long sessions."""

from loguru import logger

from ...core.enumeration import Role, MemoryType
from ...core.op import BaseReact
from ...core.schema import Message


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
    def _is_user_message(message: Message) -> bool:
        """Check if a message is a user-initiated message (user or tool result)."""
        return message.role is Role.USER

    def _find_turn_start_index(self, messages: list[Message], entry_index: int) -> int:
        """
        Find the user message that starts the turn containing the given entry index.
        Returns -1 if no turn start found before the index.
        """
        for i in range(entry_index, -1, -1):
            if self._is_user_message(messages[i]):
                return i
        return -1

    def _find_cut_point(self, messages: list[Message]) -> dict:
        """
        Find cut point with split turn detection.

        A "split turn" occurs when the cut point falls in the middle of a conversation turn
        rather than at a clean user message boundary. For example:
          User → Assistant → [CUT HERE] → Assistant continues → User

        In this case, we need to:
        1. Summarize complete history (before turn start)
        2. Separately summarize the turn prefix (turn start to cut point)
        3. Keep the turn suffix (cut point onwards) in full

        Returns dict with:
            - messages_to_summarize: Complete turns before the current turn
            - turn_prefix_messages: If split turn, messages from turn start to cut point
            - is_split_turn: Whether this is a split turn
            - cut_index: The actual cut point index
        """
        accumulated_tokens = 0
        cut_index = 0

        # Walk backwards from the newest messages, accumulating tokens until we hit the keep threshold
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            msg_tokens = self.token_counter.count_token([msg])
            accumulated_tokens += msg_tokens

            if accumulated_tokens >= self.keep_recent_tokens:
                cut_index = i
                break

        if cut_index == 0:
            return {
                "messages_to_summarize": [],
                "turn_prefix_messages": [],
                "is_split_turn": False,
                "cut_index": 0,
            }

        # Check if cut point is a user message (clean turn boundary) or assistant/other (mid-turn)
        cut_message = messages[cut_index]
        is_user_cut = self._is_user_message(cut_message)

        if is_user_cut:
            # Clean cut: cut point is at a turn boundary, summarize everything before
            return {
                "messages_to_summarize": messages[:cut_index],
                "turn_prefix_messages": [],
                "is_split_turn": False,
                "cut_index": cut_index,
            }

        # Split turn detected: find where the current turn started
        turn_start_index = self._find_turn_start_index(messages, cut_index)

        if turn_start_index == -1:
            # No turn start found (shouldn't happen), treat as clean cut
            return {
                "messages_to_summarize": messages[:cut_index],
                "turn_prefix_messages": [],
                "is_split_turn": False,
                "cut_index": cut_index,
            }

        # Split turn: separate complete history from turn prefix
        # History: [0, turn_start_index) - complete turns to summarize
        # Turn prefix: [turn_start_index, cut_index) - needs special context summary
        # Turn suffix: [cut_index, end) - kept in full (recent work)
        return {
            "messages_to_summarize": messages[:turn_start_index],
            "turn_prefix_messages": messages[turn_start_index:cut_index],
            "is_split_turn": True,
            "cut_index": cut_index,
        }

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
        """
        Build messages for compaction summarization.

        This creates the prompt for the main history summary. If split turn is detected,
        a separate turn prefix summary will be generated later in execute().
        """
        messages: list[Message] = [Message(**m) if isinstance(m, dict) else m for m in self.context.messages]

        cut_result = self._find_cut_point(messages)
        messages_to_summarize = cut_result["messages_to_summarize"]
        self.context.is_split_turn = cut_result["is_split_turn"]
        self.context.turn_prefix_messages = cut_result["turn_prefix_messages"]

        if not messages_to_summarize:
            logger.info("No messages to summarize")
            return []

        system_prompt = self.get_prompt("system_prompt")
        if self.context.get("previous_summary", ""):
            user_prompt = self.prompt_format("update_user_message", previous_summary=self.context.previous_summary)
        else:
            user_prompt = self.get_prompt("initial_user_message")
        conversation_text = self._serialize_conversation(messages_to_summarize)

        return [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=f"<conversation>\n{conversation_text}\n</conversation>\n\n{user_prompt}"),
        ]

    def build_messages_s2(self) -> list[Message]:
        """
        Generate summary for turn prefix when splitting a turn.

        This provides context for the retained turn suffix. The summary focuses on:
        - What the user originally asked for in this turn
        - Key decisions and early progress made in the prefix
        - Information needed to understand the kept suffix

        This is shorter and more focused than the full history summary.
        """
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

        Compaction process:
        1. Check if token count exceeds threshold
        2. Find cut point and detect if it's a split turn
        3. Generate history summary (complete turns before cut point)
        4. If split turn: generate turn prefix summary (partial turn before cut point)
        5. Merge summaries and update context

        Final context structure after compaction:
        - Summary (history + optional turn prefix context)
        - Recent messages kept in full (from cut point onwards)
        """
        messages: list[Message] = [Message(**m) if isinstance(m, dict) else m for m in self.context.messages]
        token_count: int = self.token_counter.count_token(messages)
        threshold = self.context_window_tokens - self.reserve_tokens
        if token_count < threshold:
            logger.info(f"Token count {token_count} below threshold, skipping compaction")
            return {
                "answer": "",
                "success": True,
                "messages": [],
                "tools": [],
                "skipped": True,
            }

        logger.info(f"Starting compaction, token count: {token_count}")
        messages: list[Message] = self.build_messages_s1()
        if messages:
            assistant_message = await self.llm.chat(messages)
            history_summary = assistant_message.content
        else:
            history_summary = ""

        if self.context.is_split_turn and self.context.turn_prefix_messages:
            logger.info("Split turn detected, generating turn prefix summary")
            messages: list[Message] = self.build_messages_s2()
            if messages:
                assistant_message = await self.llm.chat(messages)
                turn_prefix_summary = assistant_message.content
            else:
                turn_prefix_summary = ""
            summary = f"{history_summary}\n\n---\n\n**Turn Context (split turn):**\n\n{turn_prefix_summary}"
        else:
            summary = history_summary

        logger.info(f"Compaction complete, summary length: {len(summary)}, split_turn: {self.context.is_split_turn}")

        return {
            "compacted": True,
            "tokens_before": token_count,
            "summary": summary,
            "is_split_turn": self.context.is_split_turn,
        }
