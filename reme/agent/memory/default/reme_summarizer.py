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
        self.context.history_node = await self.add_history_node()

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

    async def react(self, messages: list[Message], tools: list["BaseTool"], stage: str = ""):
        """Run single ReAct step - only one tool call iteration."""
        success: bool = False
        used_tools: list[BaseTool] = []

        # Reasoning: LLM decides next action
        assistant_message, should_act = await self._reasoning_step(messages, tools, step=0, stage=stage)

        if should_act:
            # Acting: execute tools and collect results (only once)
            t_tools, tool_messages = await self._acting_step(assistant_message, tools, step=0, stage=stage)
            used_tools.extend(t_tools)
            messages.extend(tool_messages)
            success = True
        else:
            # No tools requested
            success = True

        return used_tools, messages, success

    async def execute(self):
        result = await super().execute()
        tools: list[BaseTool] = result["tools"]
        hands_off_tool = tools[0]
        agents: list[BaseMemoryAgent] = hands_off_tool.response.metadata["agents"]

        success = True
        messages = []
        tools = []
        memory_nodes = []
        for agent in agents:
            success = success and agent.response.success
            messages.extend(agent.response.metadata["messages"])
            tools.extend(agent.response.metadata["tools"])
            memory_nodes.extend(agent.response.answer)

        return {
            "answer": memory_nodes,
            "success": True,
            "messages": messages,
            "tools": tools,
        }
