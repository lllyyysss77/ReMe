"""Personal memory retriever agent for retrieving personal memories through vector search."""

import datetime

from loguru import logger

from ...core.enumeration import Role
from ...core.op import BaseReact
from ...core.schema import Message


class FsSummarizer(BaseReact):
    """Retrieve personal memories through vector search and history reading."""

    def __init__(self, memory_dir: str = "memory", version: str = "default", **kwargs):
        super().__init__(**kwargs)
        self.memory_dir: str = memory_dir
        self.version: str = version

    async def build_messages(self) -> list[Message]:
        messages: list[Message] = [Message(**m) if isinstance(m, dict) else m for m in self.context.messages]
        date_str: str = self.context.get("date", datetime.datetime.now().strftime("%Y-%m-%d"))

        if self.version == "default":
            messages.append(
                Message(
                    role=Role.USER,
                    content=self.prompt_format(
                        "user_message_v2",
                        date=date_str,
                        memory_dir=self.memory_dir,
                    ),
                ),
            )
        else:
            messages.extend(
                [
                    Message(role=Role.SYSTEM, content=self.get_prompt("system_prompt")),
                    Message(
                        role=Role.USER,
                        content=self.prompt_format(
                            "user_message",
                            date=date_str,
                            memory_dir=self.memory_dir,
                        ),
                    ),
                ],
            )
        return messages

    async def execute(self):
        result = await super().execute()
        answer = result["answer"]
        logger.info(f"[{self.__class__.__name__}] answer={answer}")
        return result
