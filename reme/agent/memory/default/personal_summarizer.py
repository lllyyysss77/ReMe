"""Personal memory summarizer agent for two-phase personal memory processing."""

from loguru import logger

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import Role, MemoryType
from ....core.op import BaseTool
from ....core.schema import Message, MemoryNode


class PersonalSummarizer(BaseMemoryAgent):
    """Two-phase personal memory processor: retrieve/add memories then update profile."""

    memory_type: MemoryType = MemoryType.PERSONAL

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.retrieved_nodes: list[MemoryNode] = []

    async def _build_phase1_messages(self) -> list[Message]:
        """Build messages for phase 1: retrieve and add memory."""
        return [
            Message(
                role=Role.SYSTEM,
                content=self.prompt_format(
                    prompt_name="system_prompt_phase1",
                    context=self.context.history_node.content,
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                ),
            ),
            Message(
                role=Role.USER,
                content=self.get_prompt("user_message_phase1"),
            ),
        ]

    async def _build_phase2_messages(self) -> list[Message]:
        """Build messages for phase 2: update user profile."""
        return [
            Message(
                role=Role.SYSTEM,
                content=self.prompt_format(
                    prompt_name="system_prompt_phase2",
                    context=self.context.history_node.content,
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                    user_profile=await self.read_user_profile(show_id="profile"),
                ),
            ),
            Message(
                role=Role.USER,
                content=self.get_prompt("user_message_phase2"),
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
        """Execute tool calls with memory context."""
        return await super()._acting_step(
            assistant_message,
            tools,
            step,
            stage=stage,
            memory_type=self.memory_type.value,
            memory_target=self.memory_target,
            history_node=self.history_node,
            author=self.author,
            retrieved_nodes=self.retrieved_nodes,
            **kwargs,
        )

    async def execute(self):
        """Execute two-phase memory processing: retrieve/add -> update profile."""
        tools = self.tools
        for i, tool in enumerate(tools):
            logger.info(f"[{self.__class__.__name__}] tool_call[{i}]={tool.tool_call.simple_input_dump(as_dict=False)}")

        messages_phase1 = await self._build_phase1_messages()
        for i, message in enumerate(messages_phase1):
            role = message.name or message.role
            logger.info(f"[{self.__class__.__name__} S1] role={role} {message.simple_dump(as_dict=False)}")
        tools_phase1, messages_phase1, success_phase1 = await self.react(messages_phase1, tools[:-1], stage="S1")

        messages_phase2 = await self._build_phase2_messages()
        for i, message in enumerate(messages_phase2):
            role = message.name or message.role
            logger.info(f"[{self.__class__.__name__} S2] role={role} {message.simple_dump(as_dict=False)}")
        tools_phase2, messages_phase2, success_phase2 = await self.react(messages_phase2, tools[-1:], stage="S2")

        success = success_phase1 and success_phase2
        messages = messages_phase1 + messages_phase2
        tools = tools_phase1 + tools_phase2
        memory_nodes = []
        for tool in tools:
            if tool.memory_nodes:
                memory_nodes.extend(tool.memory_nodes)

        return {
            "answer": memory_nodes,
            "success": success,
            "messages": messages,
            "tools": tools,
        }
