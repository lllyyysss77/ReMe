from loguru import logger

from ..base_memory_agent import BaseMemoryAgent
from ...core_old.enumeration import Role
from ...core_old.schema import Message
from ...core_old.utils import format_messages


class ReMeRetrieverV4(BaseMemoryAgent):

    def __init__(self, meta_memories: list[dict] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.meta_memories: list[dict] = meta_memories or []
        self.meta_info_dict: dict[str, str] = {}

    async def _read_meta_memories(self) -> str:
        from ...mem_tool import ReadMetaMemory
        meta_memory_info = ReadMetaMemory().format_memory_metadata(self.meta_memories)
        logger.info(f"meta_memory_info={meta_memory_info}")
        return meta_memory_info

    async def build_messages(self) -> list[Message]:
        if self.context.get("query"):
            user_query = self.context.query
        elif self.context.get("messages"):
            user_query = format_messages(self.context.messages)
        else:
            raise ValueError("Input must have either `query` or `messages`")

        messages = [
            Message(
                role=Role.SYSTEM,
                content=self.prompt_format(
                    prompt_name="system_prompt",
                    meta_memory_info=await self._read_meta_memories(),
                    user_query=user_query,
                ),
            ),
            Message(
                role=Role.USER,
                content=self.get_prompt("user_message"),
            ),
        ]

        return messages

    async def _acting_step(self, assistant_message: Message, step: int, **kwargs) -> list[Message]:
        import asyncio
        from ...mem_tool.v4 import HandsOff

        if not assistant_message.tool_calls:
            return []

        tool_list: list = []
        tool_result_messages: list[Message] = []
        tool_dict = {t.tool_call.name: t for t in self.tools}
        stage_prefix = ""

        # Add required context parameters
        kwargs["query"] = self.context.get("query", "")
        kwargs["messages"] = self.context.get("messages", [])

        for j, tool_call in enumerate(assistant_message.tool_calls):
            if tool_call.name not in tool_dict:
                logger.warning(f"[{self.__class__.__name__}{stage_prefix}] unknown tool_call.name={tool_call.name}")
                continue

            logger.info(
                f"[{self.__class__.__name__}{stage_prefix}] step{step + 1}.{j} "
                f"submit tool_calls={tool_call.name} argument={tool_call.arguments}",
            )
            tool_copy = tool_dict[tool_call.name].copy()
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

            # Collect meta_info_dict from HandsOff tool
            if isinstance(op, HandsOff) and hasattr(op, "meta_info_dict"):
                self.meta_info_dict.update(op.meta_info_dict)
                logger.info(f"Collected meta_info_dict from HandsOff: {len(op.meta_info_dict)} entries")

            tool_result = str(op.output)
            tool_message = Message(
                role=Role.TOOL,
                content=tool_result,
                tool_call_id=op.tool_call.id,
            )
            tool_result_messages.append(tool_message)

            self.meta_info += tool_result + "\n"

            logger.info(f"[{self.__class__.__name__}{stage_prefix}] step{step + 1}.{j} join tool_result={tool_result[:2000]}...\n\n")

        return tool_result_messages

    async def execute(self):
        await super().execute()

        # Assemble meta_info_dict into output
        if self.meta_info_dict:
            output_parts = []
            for key, value in self.meta_info_dict.items():
                output_parts.append(f"## {key}\n{value}")
            self.output = "\n\n".join(output_parts)
            logger.info(f"Assembled output from meta_info_dict with {len(self.meta_info_dict)} entries")