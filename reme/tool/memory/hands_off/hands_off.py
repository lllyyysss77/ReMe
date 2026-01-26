"""Hands-off tool to delegate memory tasks to specific agents"""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ....agent.memory import BaseMemoryAgent
from ....core.enumeration import MemoryType
from ....core.schema import ToolCall


class HandsOff(BaseMemoryTool):
    """Tool to delegate memory tasks to appropriate memory agents"""

    def __init__(self, memory_agents: list[BaseMemoryAgent] = None, **kwargs):
        kwargs["enable_multiple"] = True
        kwargs["sub_ops"] = memory_agents or []
        super().__init__(**kwargs)
        self.sub_ops: list[BaseMemoryAgent] = [
            a for a in self.sub_ops if isinstance(a, BaseMemoryAgent) and a.memory_type is not None
        ]

    @property
    def memory_agent_dict(self) -> dict[MemoryType, BaseMemoryAgent]:
        """Map memory types to their corresponding agents"""
        return {a.memory_type: a for a in self.sub_ops}

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the multiple tool call schema"""
        return ToolCall(
            **{
                "description": "Delegate memory tasks to appropriate agents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_tasks": {
                            "type": "array",
                            "description": "Memory tasks to delegate to specific agents",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "memory_type": {
                                        "type": "string",
                                        "description": "Memory type to handle",
                                        "enum": [k.value for k in self.memory_agent_dict if k],
                                    },
                                    "memory_target": {
                                        "type": "string",
                                        "description": "Target or context for the memory operation",
                                    },
                                },
                                "required": ["memory_type", "memory_target"],
                            },
                        },
                    },
                    "required": ["memory_tasks"],
                },
            },
        )

    async def execute(self):
        # Deduplicate and validate tasks
        tasks = []
        seen = set()
        for task in self.context.get("memory_tasks", []):
            memory_type = MemoryType(task.get("memory_type", ""))
            memory_target = task.get("memory_target", "")

            task_key = (memory_type, memory_target)
            if task_key in seen:
                logger.info(f"Skip duplicate: {memory_type.value} - {memory_target}")
                continue
            seen.add(task_key)

            tasks.append({"memory_type": memory_type, "memory_target": memory_target})

        if not tasks:
            return "No valid memory tasks to execute."

        # Submit tasks to agents
        agent_list: list[BaseMemoryAgent] = []
        for i, task in enumerate(tasks):
            memory_type: MemoryType = task["memory_type"]
            memory_target: str = task["memory_target"]

            agent = self.memory_agent_dict[memory_type].copy()
            agent_list.append(agent)

            logger.info(f"Task {i}: {memory_type.value} agent for {memory_target}")
            task_kwargs = {"memory_type": memory_type, "memory_target": memory_target}
            for k in ["query", "messages", "description", "history_node"]:
                if k in self.context:
                    task_kwargs[k] = self.context[k]
            self.submit_async_task(agent.call, service_context=self.service_context, **task_kwargs)

        await self.join_async_tasks()

        # Collect results
        results = []
        for agent in agent_list:
            memory_type = agent.memory_type
            memory_target = agent.memory_target
            results.append(f"{memory_type.value}({memory_target}): {agent.response.answer}")

        logger.info(f"Completed {len(results)} task(s)")
        return {
            "answer": "\n".join(results),
            "agents": agent_list,
        }
