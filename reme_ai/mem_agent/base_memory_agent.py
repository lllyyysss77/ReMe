"""Base memory agent for handling memory operations with tool-based reasoning."""

import asyncio
import json
from abc import ABCMeta

from loguru import logger

from ..core.enumeration import Role, MemoryType
from ..core.op import BaseOp
from ..core.schema import Message, ToolCall, MemoryNode
from ..mem_tool import BaseMemoryTool, ThinkTool


class BaseMemoryAgent(BaseOp, metaclass=ABCMeta):
    """Base class for memory agents that perform reasoning and acting with memory tools."""

    memory_type: MemoryType | None = None

    def __init__(
        self,
        tools: list[BaseMemoryTool],
        add_think_tool: bool = False,  # only for instruct model
        tool_call_interval: float = 0,
        max_steps: int = 8,
        **kwargs,
    ):
        tools = tools or []
        if add_think_tool:
            tools.append(ThinkTool())
        kwargs["sub_ops"] = tools
        super().__init__(**kwargs)
        self.sub_ops: list[BaseMemoryTool] = [t for t in self.sub_ops if isinstance(t, BaseMemoryTool)]
        self.tool_call_interval: float = tool_call_interval
        self.max_steps: int = max_steps

        self.messages: list[Message] = []
        self.tool_messages: list[Message] = []
        self.success: bool = True
        self.retrieved_nodes: list[MemoryNode] = []
        self.memory_nodes: list[MemoryNode | str] = []
        self.meta_info: str = ""

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
        """Returns the list of memory tools available to this agent."""
        return self.sub_ops

    @tools.setter
    def tools(self, tools: list[BaseMemoryTool]):
        self.sub_ops = tools

    def get_messages(self) -> list[Message] | str:
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

    async def _reasoning_step(self, messages: list[Message], step: int, stage: str = "", **kwargs) -> tuple[Message, bool]:
        assistant_message: Message = await self.llm.chat(
            messages=messages,
            tools=[t.tool_call for t in self.tools],
            **kwargs,
        )
        messages.append(assistant_message)
        stage_prefix = f"-{stage}" if stage else ""
        logger.info(
            f"[{self.__class__.__name__}{stage_prefix}] "
            f"step{step + 1}.assistant={assistant_message.simple_dump(enable_json_dump=True)}",
        )
        should_act = bool(assistant_message.tool_calls)
        return assistant_message, should_act

    async def _acting_step(self, assistant_message: Message, step: int, stage: str = "", **kwargs) -> list[Message]:
        if not assistant_message.tool_calls:
            return []

        tool_list: list[BaseMemoryTool] = []
        tool_result_messages: list[Message] = []
        tool_dict = {t.tool_call.name: t for t in self.tools}
        stage_prefix = f"-{stage}" if stage else ""

        for j, tool_call in enumerate(assistant_message.tool_calls):
            if tool_call.name not in tool_dict:
                logger.warning(f"[{self.__class__.__name__}{stage_prefix}] unknown tool_call.name={tool_call.name}")
                continue

            logger.info(
                f"[{self.__class__.__name__}{stage_prefix}] step{step + 1}.{j} "
                f"submit tool_calls={tool_call.name} argument={tool_call.arguments}",
            )
            tool_copy: BaseMemoryTool = tool_dict[tool_call.name].copy()
            tool_copy.tool_call.id = tool_call.id
            tool_list.append(tool_copy)
            kwargs.update(tool_call.argument_dict)
            self.submit_async_task(tool_copy.call, retrieved_nodes=self.retrieved_nodes, **kwargs)
            if self.tool_call_interval > 0:
                await asyncio.sleep(self.tool_call_interval)

        await self.join_async_tasks()

        for j, op in enumerate(tool_list):
            if op.memory_nodes:
                self.memory_nodes.extend(op.memory_nodes)

            if hasattr(op, "messages") and op.messages:
                self.tool_messages.extend(op.messages)

            tool_result = str(op.output)
            tool_message = Message(
                role=Role.TOOL,
                content=tool_result,
                tool_call_id=op.tool_call.id,
            )
            tool_result_messages.append(tool_message)
            
            # # Collect tool call information to meta_info
            # tool_info = f"\n## Tool Call {step + 1}.{j + 1}: {op.tool_call.name}\n"
            # tool_info += f"Arguments: {json.dumps(assistant_message.tool_calls[j].argument_dict, ensure_ascii=False)}\n"
            # tool_info += f"Result: {tool_result}\n"
            self.meta_info += tool_result + "\n"
            
            logger.info(f"[{self.__class__.__name__}{stage_prefix}] step{step + 1}.{j} join tool_result={tool_result[:2000]}...\n\n")
        return tool_result_messages

    async def react(self, messages: list[Message], stage: str = ""):
        """Performs reasoning and acting steps until completion or max steps reached."""
        success: bool = False
        for step in range(self.max_steps):
            assistant_message, should_act = await self._reasoning_step(messages, step, stage=stage)

            if not should_act:
                success = True
                break

            tool_result_messages = await self._acting_step(assistant_message, step, stage=stage)
            messages.extend(tool_result_messages)

        return messages, success

    async def execute(self):
        for i, tool in enumerate(self.tools):
            logger.info(
                f"[{self.__class__.__name__}] step0.{i} "
                f"tool_call={json.dumps(tool.tool_call.simple_input_dump(), ensure_ascii=False)}",
            )

        messages = await self.build_messages()
        for i, message in enumerate(messages):
            logger.info(
                f"[{self.__class__.__name__}] step0.{i} {message.role} {message.name or ''} "
                f"{message.simple_dump(enable_json_dump=True)}",
            )

        self.messages, self.success = await self.react(messages)
        if self.success and self.messages:
            self.output = self.messages[-1].content
        else:
            self.output = "No relevant memories found."

    @property
    def memory_target(self) -> str:
        """Returns the target memory identifier from context."""
        return self.context.get("memory_target", "")

    @property
    def description(self) -> str:
        """Returns the description of the messages."""
        return self.context.get("description", "")

    @property
    def ref_memory_id(self) -> str:
        """Returns the reference memory ID from context."""
        return self.context.get("ref_memory_id", "")

    @property
    def author(self) -> str:
        """Returns the LLM model name as the author identifier."""
        return self.llm.model_name

    @property
    def history_node(self):
        """Returns the history node."""
        return self.context.get("history_node", None)