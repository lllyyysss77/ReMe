from loguru import logger

from ..base_memory_agent import BaseMemoryAgent
from ...core.enumeration import Role, MemoryType
from ...core.schema import Message, MemoryNode


class PersonalSummarizerV4(BaseMemoryAgent):
    memory_type: MemoryType = MemoryType.PERSONAL

    async def build_messages_phase1(self) -> list[Message]:
        """Build messages for phase 1: AddSummaryMemory"""
        history_node: MemoryNode = self.context.history_node
        messages = [
            Message(
                role=Role.SYSTEM,
                content=self.prompt_format(
                    prompt_name="system_prompt_phase1",
                    context=history_node.content,
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                )),
            Message(
                role=Role.USER,
                content=self.get_prompt("user_message_phase1"),
            ),
        ]
        return messages

    async def build_messages_phase2(self, user_profile: str) -> list[Message]:
        """Build messages for phase 2: UpdateUserProfile"""
        history_node: MemoryNode = self.context.history_node
        messages = [
            Message(
                role=Role.SYSTEM,
                content=self.prompt_format(
                    prompt_name="system_prompt_phase2",
                    context=history_node.content,
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                    user_profile=user_profile,
                )),
            Message(
                role=Role.USER,
                content=self.get_prompt("user_message_phase2"),
            ),
        ]
        return messages

    async def _acting_step(self, assistant_message: Message, step: int, **kwargs) -> list[Message]:
        return await super()._acting_step(
            assistant_message,
            step,
            memory_type=self.memory_type.value,
            memory_target=self.memory_target,
            history_node=self.history_node,
            author=self.author,
            **kwargs,
        )

    async def execute(self):
        """Execute in two phases: 1) AddSummaryMemory, 2) UpdateUserProfile"""
        # Log available tools
        for i, tool in enumerate(self.tools):
            logger.info(
                f"[{self.__class__.__name__}] step0.{i} "
                f"tool_call={tool.tool_call.name}",
            )

        # Phase 1: AddSummaryMemory
        logger.info(f"[{self.__class__.__name__}] Starting Phase 1: AddSummaryMemory")

        # Filter tools for phase 1 (only AddSummaryMemory)
        original_tools = self.tools.copy()
        self.tools = [t for t in self.tools if t.tool_call.name == "add_summary_memory"]

        messages_phase1 = await self.build_messages_phase1()
        for i, message in enumerate(messages_phase1):
            logger.info(
                f"[{self.__class__.__name__}] phase1.step0.{i} {message.role} "
                f"{message.simple_dump(enable_json_dump=True)}",
            )

        messages_phase1, success_phase1 = await self.react(messages_phase1)
        if not success_phase1:
            logger.warning(f"[{self.__class__.__name__}] Phase 1 did not complete successfully")

        # Phase 2: Read user profile and UpdateUserProfile
        logger.info(f"[{self.__class__.__name__}] Starting Phase 2: UpdateUserProfile")

        # Restore original tools and get ReadUserProfile tool
        self.tools = original_tools
        read_profile_tool = next((t for t in self.tools if t.tool_call.name == "read_user_profile"), None)

        user_profile = ""
        if read_profile_tool:
            # Call ReadUserProfile to load current profile
            logger.info(f"[{self.__class__.__name__}] Loading user profile with ReadUserProfile")
            await read_profile_tool.call(memory_type=self.memory_type.value, memory_target=self.memory_target)
            user_profile = str(read_profile_tool.output)
            logger.info(f"[{self.__class__.__name__}] User profile loaded: {user_profile}...")
        else:
            logger.warning(f"[{self.__class__.__name__}] ReadUserProfile tool not found")

        # Filter tools for phase 2 (only UpdateUserProfile)
        self.tools = [t for t in self.tools if t.tool_call.name == "update_user_profile"]

        messages_phase2 = await self.build_messages_phase2(user_profile)
        for i, message in enumerate(messages_phase2):
            logger.info(
                f"[{self.__class__.__name__}] phase2.step0.{i} {message.role} "
                f"{message.simple_dump(enable_json_dump=True)}",
            )

        messages_phase2, success_phase2 = await self.react(messages_phase2)

        # Restore original tools
        self.tools = original_tools

        # Set final output and messages
        self.messages = messages_phase1 + messages_phase2
        self.success = success_phase1 and success_phase2

        if self.success and messages_phase2:
            self.output = messages_phase2[-1].content
        else:
            self.output = "Memory processing completed with issues."
