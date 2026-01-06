"""Base memory agent for handling memory operations with tool-based reasoning."""

import asyncio
from abc import ABCMeta

from loguru import logger

from ..core.enumeration import Role, MemoryType
from ..core.op import BaseOp
from ..core.schema import Message, ToolCall
from ..tool.memory import BaseMemoryTool, ThinkTool


class BaseMemoryAgent(BaseOp, metaclass=ABCMeta):
    """Base class for memory agents that perform reasoning and acting with memory tools."""

    memory_type: MemoryType | None = None

    def __init__(
        self,
        max_steps: int = 20,
        tool_call_interval: float = 0,
        add_think_tool: bool = False,  # only for instruct model
        tools: list[BaseMemoryTool] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_steps: int = max_steps
        self.tool_call_interval: float = tool_call_interval
        self.add_think_tool: bool = add_think_tool
        assert not self.sub_ops, "sub_ops must be empty, use `tools`~"
        if tools:
            self.sub_ops.extend([t.set_language(self.language) for t in tools])

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": self.get_prompt("tool"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "query",
                        },
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {
                                        "type": "string",
                                        "description": "role",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "content",
                                    },
                                },
                                "required": ["role", "content"],
                            },
                        },
                    },
                    "required": [],
                },
            },
        )

    @property
    def tools(self) -> list[BaseMemoryTool]:
        """Returns the list of memory tools available to the agent."""
        tools: list[BaseMemoryTool] = [o for o in self.sub_ops if isinstance(o, BaseMemoryTool)]
        if self.add_think_tool:
            tools.append(ThinkTool(language=self.language))
        return tools

    @tools.setter
    def tools(self, tools: list[BaseMemoryTool] | BaseMemoryTool):
        """Sets the memory tools for the agent."""
        self.sub_ops = tools

    def get_messages(self) -> list[Message]:
        """Extracts and returns messages from the context query or messages."""
        if self.context.get("query"):
            messages = [Message(role=Role.USER, content=self.context.query)]
        elif self.context.get("messages"):
            messages = [Message(**m) if isinstance(m, dict) else m for m in self.context.messages]
        else:
            raise ValueError("input must have either `query` or `messages`")
        return messages

    async def build_messages(self) -> list[Message]:
        """Builds and returns the initial messages for the agent."""
        return self.get_messages()

    async def _reasoning_step(self, messages: list[Message], step: int, **kwargs) -> tuple[Message, bool]:
        assistant_message: Message = await self.llm.chat(
            messages=messages,
            tools=[t.tool_call for t in self.tools],
            **kwargs,
        )
        messages.append(assistant_message)
        logger.info(f"step{step + 1}.assistant={assistant_message.model_dump_json()}")
        should_act = bool(assistant_message.tool_calls)
        return assistant_message, should_act

    async def _acting_step(self, assistant_message: Message, step: int, **kwargs) -> list[Message]:
        if not assistant_message.tool_calls:
            return []

        tool_list: list[BaseMemoryTool] = []
        tool_result_messages: list[Message] = []
        tool_dict = {t.tool_call.name: t for t in self.tools}

        for j, tool_call in enumerate(assistant_message.tool_calls):
            if tool_call.name not in tool_dict:
                logger.warning(f"unknown tool_call.name={tool_call.name}")
                continue

            logger.info(f"step{step + 1}.{j} submit tool_calls={tool_call.name} argument={tool_call.argument_dict}")
            tool_copy: BaseMemoryTool = tool_dict[tool_call.name].copy()
            tool_copy.tool_call.id = tool_call.id
            tool_list.append(tool_copy)
            kwargs.update(tool_call.argument_dict)
            self.submit_async_task(tool_copy.call, **kwargs)
            if self.tool_call_interval > 0:
                await asyncio.sleep(self.tool_call_interval)

        await self.join_async_tasks()

        for j, op in enumerate(tool_list):
            tool_result = str(op.output)
            tool_message = Message(
                role=Role.TOOL,
                content=tool_result,
                tool_call_id=op.tool_call.id,
            )
            tool_result_messages.append(tool_message)
            logger.info(f"step{step + 1}.{j} join tool_result={tool_result[:200]}...\n\n")
        return tool_result_messages

    async def react(self, messages: list[Message]):
        """Performs reasoning and acting steps until completion or max steps reached."""
        success: bool = False
        for step in range(self.max_steps):
            assistant_message, should_act = await self._reasoning_step(messages, step)

            if not should_act:
                success = True
                break

            tool_result_messages = await self._acting_step(assistant_message, step)
            messages.extend(tool_result_messages)

        return messages, success

    async def execute(self):
        messages = await self.build_messages()
        for i, message in enumerate(messages):
            logger.info(f"step0.{i} {message.role} {message.name or ''} {message.simple_dump()}")

        messages, success = await self.react(messages)
        if messages:
            if success:
                self.output = messages[-1].content
            else:
                self.output = f"react is not complete with content:\n{messages[-1].content}"
        else:
            self.output = "empty messages"

        self.context.response.metadata["messages"] = messages
        self.context.response.metadata["success"] = success

    @property
    def memory_target(self) -> str:
        """Returns the target memory identifier from context."""
        return self.context.get("memory_target", "")

    @property
    def ref_memory_id(self) -> str:
        """Returns the reference memory ID from context."""
        return self.context.get("ref_memory_id", "")

    @property
    def author(self) -> str:
        """Returns the LLM model name as the author identifier."""
        return self.llm.model_name
