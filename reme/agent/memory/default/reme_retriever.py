"""ReMe retriever agent that orchestrates multiple memory agents to retrieve information."""

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import Role
from ....core.op import BaseTool
from ....core.schema import Message
from ....core.utils import format_messages


class ReMeRetriever(BaseMemoryAgent):
    """Orchestrate multiple memory agents to retrieve information."""

    def __init__(self, meta_memories: list[dict] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.meta_memories: list[dict] = meta_memories or []

    async def build_messages(self) -> list[Message]:
        if self.context.get("query"):
            context = self.context.query
        elif self.context.get("messages"):
            context = self.description + "\n" + format_messages(self.context.messages)
        else:
            raise ValueError("input must have either `query` or `messages`")

        return [
            Message(
                role=Role.SYSTEM,
                content=self.prompt_format(
                    prompt_name="system_prompt",
                    meta_memory_info=await self.read_meta_memories(self.meta_memories),
                    context=context.strip(),
                ),
            ),
            Message(
                role=Role.USER,
                content=self.get_prompt("user_message"),
            ),
        ]

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
            query=self.query,
            author=self.author,
            **kwargs,
        )

    async def execute(self):
        result = await super().execute()
        tools: list[BaseTool] = result["tools"]
        hands_off_tool = tools[0]
        agents: list[BaseMemoryAgent] = hands_off_tool.response.metadata["agents"]

        answer = []
        success = True
        messages = []
        tools = []
        retrieved_nodes = []

        for agent in agents:
            answer.append(agent.response.answer)
            success = success and agent.response.success
            messages.extend(agent.response.metadata["messages"])
            tools.extend(agent.response.metadata["tools"])
            retrieved_nodes.extend(agent.response.metadata["retrieved_nodes"])

        return {
            "answer": "\n".join(answer),
            "success": True,
            "messages": self.messages,
            "tools": tools,
            "retrieved_nodes": retrieved_nodes,
        }
