from typing import TYPE_CHECKING

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.enumeration import MemoryType
from ...core.schema import Message

if TYPE_CHECKING:
    from ...mem_agent import BaseMemoryAgent


class HandsOff(BaseMemoryTool):

    def __init__(self, memory_agents: list["BaseMemoryAgent"], **kwargs):
        kwargs["enable_multiple"] = True
        kwargs["sub_ops"] = memory_agents or []
        super().__init__(**kwargs)
        from ...mem_agent import BaseMemoryAgent

        self.sub_ops: list[BaseMemoryAgent] = [a for a in self.sub_ops if isinstance(a, BaseMemoryAgent)]
        self.messages: list[Message] = []

    @property
    def memory_agent_dict(self) -> dict[MemoryType, "BaseMemoryAgent"]:
        return {a.memory_type: a for a in self.sub_ops}

    def _build_tool_description(self) -> str:
        return "Distribute memory tasks to appropriate memory agents."

    def _build_multiple_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "memory_tasks": {
                    "type": "array",
                    "description": "List of memory tasks to distribute to specific agents",
                    "items": {
                        "type": "object",
                        "properties": {
                            "memory_type": {
                                "type": "string",
                                "description": "Type of memory to handle",
                                "enum": [k.value for k in self.memory_agent_dict],
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
        }

    async def execute(self):
        tasks = []
        seen = set()
        for task in self.context.get("memory_tasks", []):
            memory_type = MemoryType(task.get("memory_type", ""))
            memory_target = task.get("memory_target", "")
            
            # Deduplicate tasks with same memory_type and memory_target
            task_key = (memory_type, memory_target)
            if task_key in seen:
                logger.info(f"Skipping duplicate task: memory_type={memory_type.value}, memory_target={memory_target}")
                continue
            seen.add(task_key)
            
            tasks.append({
                "memory_type": memory_type,
                "memory_target": memory_target,
            })

        if not tasks:
            self.output = "No valid memory tasks to execute."
            return

        agent_list = []
        for i, task in enumerate(tasks):
            memory_type: MemoryType = task["memory_type"]
            memory_target: str = task["memory_target"]

            agent = self.memory_agent_dict[memory_type].copy()
            agent_list.append([agent, memory_type, memory_target])

            logger.info(f"Task {i}: Submitting {memory_type.value} agent for target={memory_target}")
            self.submit_async_task(
                agent.call,
                memory_type=memory_type,
                memory_target=memory_target,
                query=self.context.get("query", ""),
                messages=self.context.get("messages", []),
                description=self.context.get("description"),
                history_node=self.context.get("history_node"),
            )

        await self.join_async_tasks()

        results = []
        for i, (agent, memory_type, memory_target) in enumerate(agent_list):
            if agent.memory_nodes:
                self.memory_nodes.extend(agent.memory_nodes)
            if agent.messages:
                self.messages.extend(agent.messages)

            results.append(f"{memory_type.value} {memory_target} agent result: {agent.output}")

        self.output = "\n".join(results)
        logger.info(f"Completed {len(results)} hands-off task(s):\n{self.output}")
