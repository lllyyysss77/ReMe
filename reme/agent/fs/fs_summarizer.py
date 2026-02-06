"""Personal memory retriever agent for retrieving personal memories through vector search."""

from loguru import logger

from ...core.enumeration import Role, MemoryType
from ...core.op import BaseReact
from ...core.schema import Message


class FsSummarizer(BaseReact):
    """Retrieve personal memories through vector search and history reading."""

    memory_type: MemoryType = MemoryType.PERSONAL

    def __init__(
        self,
        memory_dir: str,
        version: str = "default",
        context_window_tokens: int = 128000,
        reserve_tokens: int = 32000,
        soft_threshold_tokens: int = 4000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.memory_dir: str = memory_dir
        self.version: str = version
        self.context_window_tokens: int = context_window_tokens
        self.reserve_tokens: int = reserve_tokens
        self.soft_threshold_tokens: int = soft_threshold_tokens

    async def build_messages(self) -> list[Message]:
        messages: list[Message] = [Message(**m) if isinstance(m, dict) else m for m in self.context.messages]
        if self.version == "default":
            messages.append(
                Message(
                    role=Role.USER,
                    content=self.prompt_format(
                        "user_message_v2",
                        memory_dir=self.memory_dir,
                    ),
                ),
            )
        else:
            messages.append(Message(role=Role.SYSTEM, content=self.get_prompt("system_prompt")))
            messages.append(
                Message(
                    role=Role.USER,
                    content=self.prompt_format(
                        "user_message",
                        memory_dir=self.memory_dir,
                    ),
                ),
            )
        return messages

    async def execute(self):
        context_window = max(1, int(self.context_window_tokens))
        reserve_tokens = max(0, int(self.reserve_tokens))
        soft_threshold = max(0, int(self.soft_threshold_tokens))
        threshold = max(0, context_window - reserve_tokens - soft_threshold)
        messages: list[Message] = [Message(**m) if isinstance(m, dict) else m for m in self.context.messages]
        token_count: int = self.token_counter.count_token(messages)

        if token_count >= threshold:
            logger.info(f"[{self.__class__.__name__}] Skipping summary execution based on threshold check")
            return {
                "answer": "",
                "success": True,
                "messages": [],
                "tools": [],
                "skipped": True,
            }

        # Mark that we're executing a summary in this cycle
        summary_count = self.context.get("summary_count", 0)
        self.context["last_summary_at"] = summary_count

        result = await super().execute()
        answer = result["answer"]
        logger.info(f"[{self.__class__.__name__}] answer={answer}")
        return result
