from loguru import logger

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import Role, MemoryType
from ....core.schema import Message


class PersonalSummarizer(BaseMemoryAgent):
    """Extract and update personal memories in two phases: add summaries then update profile."""

    memory_type: MemoryType = MemoryType.PERSONAL

    async def build_messages_phase1(self) -> list[Message]:
        """Phase 1: AddSummaryMemory"""
        history_node = self.context.history_node
        return [
            Message(
                role=Role.USER,
                content=self.prompt_format(
                    prompt_name="user_message_phase1",
                    context=history_node.content,
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                ),
            ),
        ]

    async def build_messages_phase2(self, user_profile: str) -> list[Message]:
        """Phase 2: UpdateUserProfile"""
        history_node = self.context.history_node
        return [
            Message(
                role=Role.USER,
                content=self.prompt_format(
                    prompt_name="user_message_phase2",
                    context=history_node.content,
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                    user_profile=user_profile,
                ),
            ),
        ]

    async def _acting_step(self, assistant_message: Message, step: int, stage: str = "", **kwargs) -> list[Message]:
        return await super()._acting_step(
            assistant_message,
            step,
            stage=stage,
            memory_type=self.memory_type.value,
            memory_target=self.memory_target,
            history_node=self.history_node,
            author=self.author,
            **kwargs,
        )

    async def execute(self):
        """Execute two phases: AddSummaryMemory -> UpdateUserProfile"""
        from ....tool.memory.vector import ReadUserProfile

        # Log tools
        for i, tool in enumerate(self.tools):
            logger.info(f"[{self.__class__.__name__}] step0.{i} tool_call={tool.tool_call.name}")

        # Phase 1: AddSummaryMemory
        logger.info(f"[{self.__class__.__name__}-S1] Phase 1: AddSummaryMemory")
        original_tools = self.tools.copy()
        self.tools = [t for t in self.tools if t.tool_call.name == "add_summary_memory"]

        messages_phase1 = await self.build_messages_phase1()
        for i, message in enumerate(messages_phase1):
            logger.info(
                f"[{self.__class__.__name__}-S1] phase1.step0.{i} {message.role} {message.simple_dump(enable_json_dump=True)}"
            )

        messages_phase1, success_phase1 = await self.react(messages_phase1, stage="S1")
        if not success_phase1:
            logger.warning(f"[{self.__class__.__name__}-S1] Phase 1 incomplete")

        # Phase 2: UpdateUserProfile
        logger.info(f"[{self.__class__.__name__}-S2] Phase 2: UpdateUserProfile")
        self.tools = original_tools
        read_profile_tool = next((t for t in self.tools if t.tool_call.name == "read_user_profile"), None)

        user_profile = ""
        if read_profile_tool:
            logger.info(f"[{self.__class__.__name__}-S2] Loading user profile")
            await read_profile_tool.call(
                memory_type=self.memory_type.value,
                memory_target=self.memory_target,
                show_ids="profile",
            )
            user_profile = str(read_profile_tool.output)
        else:
            logger.warning(f"[{self.__class__.__name__}-S2] ReadUserProfile tool not found")

        self.tools = [t for t in self.tools if t.tool_call.name == "update_user_profile"]

        messages_phase2 = await self.build_messages_phase2(user_profile)
        for i, message in enumerate(messages_phase2):
            logger.info(
                f"[{self.__class__.__name__}-S2] phase2.step0.{i} {message.role} {message.simple_dump(enable_json_dump=True)}"
            )

        messages_phase2, success_phase2 = await self.react(messages_phase2, stage="S2")

        # Restore tools and set output
        self.tools = original_tools
        self.messages = messages_phase1 + messages_phase2
        self.success = success_phase1 and success_phase2
        self.output = (
            messages_phase2[-1].content
            if self.success and messages_phase2
            else "Memory processing completed with issues."
        )
