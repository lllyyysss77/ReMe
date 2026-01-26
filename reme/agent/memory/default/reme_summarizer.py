"""ReMe summarizer agent that orchestrates multiple memory agents to summarize information."""

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import Role
from ....core.op import BaseTool
from ....core.schema import Message


class ReMeSummarizer(BaseMemoryAgent):
    """Orchestrates multiple memory agents to summarize and store information across different memory types."""

    def __init__(self, meta_memories: list[dict] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.meta_memories: list[dict] = meta_memories or []

    async def build_messages(self) -> list[Message]:
        self.context.history_node = await self.read_history_node()

        messages = [
            Message(
                role=Role.SYSTEM,
                content=self.prompt_format(
                    prompt_name="system_prompt",
                    meta_memory_info=await self.read_meta_memories(self.meta_memories),
                    context=self.context.history_node.content,
                ),
            ),
            Message(
                role=Role.USER,
                content=self.get_prompt("user_message"),
            ),
        ]

        return messages

    async def _acting_step(
        self,
        assistant_message: Message,
        tools: list[BaseTool],
        step: int,
        stage: str = "",
        **kwargs,
    ) -> tuple[list[BaseTool], list[Message]]:
        return await super()._acting_step(
            assistant_message,
            tools,
            step,
            description=self.description,
            messages=self.messages,
            history_node=self.history_node,
            author=self.author,
            **kwargs,
        )

    async def execute(self):
        await super().execute()

        tools: list[BaseTool] = self.response.metadata["tools"]
        hands_off_tool = tools[0]
        agents: list[BaseMemoryAgent] = hands_off_tool.response.metadata["agents"]

        answer = ""
        success = True
        messages = []
        tools = []
        for agent in agents:
            answer += "\n" + agent.response.answer
            success = success and agent.response.metadata["success"]
            messages += agent.response.metadata["messages"]
            tools += agent.response.metadata["tools"]

        return {
            "answer": answer.strip(),
            "success": True,
            "messages": self.messages,
            "tools": tools,
        }
